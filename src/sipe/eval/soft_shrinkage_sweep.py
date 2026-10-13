"""Soft-shrinkage scanner eraser on frozen backbone features: lambda sweep on SCORPION.

Post-hoc, training-free baseline from vfm-geom-xai (vendored in
`sipe.baselines.soft_delta_legacy`): `x' = mu + P (x - mu)`, `P = A (A + lam B)^-1`,
with A the shrunk feature covariance and B the second moment of scanner deltas.

- Features: the backbone run once per split; `--poolings` picks GAP of the patch
  tokens (prefix tokens dropped, = `backbone_gap`) and/or the CLS token
  (= `backbone_cls`).
- Fit on the train split only. Deltas are vfm-geom's `scanner_slide_to_mean`: per
  slide, the mean of each scanner minus the mean over scanners. Eraser options are
  those of vfm-geom's soft configs (trace-normalized source, `match_x_trace`).
- Sweep on val: cross-scanner retrieval, scanner probes (train -> val) and the region
  probe, the metrics of `sipe.eval.scorpion_retrieval`. The selected lambda has the
  best val top-1 (ties -> smaller lambda).
- Test, once, for lambda = 0 and the selected lambda: the proxy protocol (probe
  trained on val, retrieval/probes on test), comparable to the proxy table.

    uv run python -m sipe.eval.soft_shrinkage_sweep --tag softshrink_h0mini --workers 0
"""

from __future__ import annotations

import argparse
import json
import logging
from dataclasses import asdict
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

from sipe.baselines.soft_delta_legacy import DeltaSourceSpec, PairedDeltaFitter
from sipe.bench.provenance import new_bench_run_dir, write_run_info
from sipe.data.scorpion import SCORPIONDataModule
from sipe.eval.scorpion_retrieval import TOPK, region_probe, retrieval, scanner_probes
from sipe.model.encoders import build_encoder

log = logging.getLogger(__name__)

POOLINGS = ("gap", "cls")
DEFAULT_LAMBDAS = (0.0, 0.1, 0.3, 1.0, 3.0, 10.0, 30.0, 100.0, 300.0, 1000.0)
SPLITS = ("train", "val", "test")
# vfm-geom-xai configs/fitting/chained_soft*.yaml, stage "scanner".
ERASER_CONFIG = {
    "affine": True,
    "normalize_source_weights": True,
    "delta_moment": "second_moment",
    "shrink_A": True,
    "shrink_B": False,
    "joint_normalization": "match_x_trace",
    "ridge": 1e-4,
    "svd_tol": 1e-7,
}
SOURCE = DeltaSourceSpec(
    name="scanner", moment="second_moment", shrinkage=False, normalization="trace"
)


@torch.no_grad()
def extract(
    backbone: torch.nn.Module,
    loader: DataLoader,
    device: torch.device,
    mixed_precision: bool,
) -> tuple[dict[str, np.ndarray], pd.DataFrame]:
    """One backbone pass -> {"gap", "cls"} features and the proxy's tile metadata."""
    out: dict[str, list[np.ndarray]] = {name: [] for name in POOLINGS}
    rows: list[pd.DataFrame] = []
    for batch in loader:
        images = batch["image"].to(device, non_blocking=True)
        # float16 autocast as plismbench's PrecisionModule (and the proxy).
        with torch.autocast(device.type, dtype=torch.float16, enabled=mixed_precision):
            tokens = backbone.forward_features(images)
        out["gap"].append(
            tokens[:, backbone.num_prefix_tokens :].mean(dim=1).float().cpu().numpy()
        )
        out["cls"].append(tokens[:, 0].float().cpu().numpy())
        rows.append(
            pd.DataFrame(
                {
                    "pair_id": batch["pair_id"],
                    "scanner": batch["domain_name"],
                    "domain": batch["domain"].numpy(),
                }
            )
        )
    meta = pd.concat(rows, ignore_index=True)
    meta["slide_id"] = meta["pair_id"].str.split("-").str[0]
    meta["region_id"] = meta["pair_id"].str.partition("-tile_")[0]
    meta["row"] = np.arange(len(meta))
    return {k: np.concatenate(v) for k, v in out.items()}, meta


def slide_to_mean_deltas(feats: np.ndarray, meta: pd.DataFrame) -> np.ndarray:
    """Per slide: mean(slide, scanner) - mean over scanners of those means.

    Port of vfm-geom-xai `build_group_mean_to_mean_scanner_deltas` (group = slide).
    """
    deltas = []
    for _, slide in meta.groupby("slide_id", sort=False):
        means = np.stack(
            [
                feats[rows].mean(axis=0)
                for _, rows in slide.groupby("scanner", sort=True)["row"]
            ]
        )
        if len(means) < 2:
            continue
        deltas.append(means - means.mean(axis=0))
    return np.concatenate(deltas).astype(np.float32)


def fit_family(feats: np.ndarray, meta: pd.DataFrame, device: torch.device):
    fitter = PairedDeltaFitter(x_dim=feats.shape[1], device=device, dtype=torch.float32)
    fitter.update_x(torch.from_numpy(feats).to(device))
    fitter.update_delta_source(
        SOURCE.name, torch.from_numpy(slide_to_mean_deltas(feats, meta)).to(device)
    )
    return fitter.prepare_soft_eraser_family(delta_sources=[SOURCE], **ERASER_CONFIG)


def apply(eraser, feats: np.ndarray, device: torch.device) -> np.ndarray:
    return eraser(torch.from_numpy(feats).to(device)).cpu().numpy()


def subsample(meta: pd.DataFrame, n: int | None, seed: int) -> np.ndarray:
    if n is None or n >= len(meta):
        return np.arange(len(meta))
    rng = np.random.default_rng(seed)
    return np.sort(rng.choice(len(meta), size=n, replace=False))


def evaluate(
    eval_x: np.ndarray,
    eval_meta: pd.DataFrame,
    probe_x: np.ndarray,
    probe_y: np.ndarray,
    raw_eval_x: np.ndarray,
    device: torch.device,
) -> dict[str, float]:
    pairs = retrieval(eval_x, eval_meta, device)
    diff = eval_x.astype(np.float64) - raw_eval_x
    return {
        **{
            c: float(pairs[c].mean())
            for c in ["cosine_similarity", *(f"top_{k}_accuracy" for k in TOPK)]
        },
        **scanner_probes(probe_x, probe_y, eval_x, eval_meta["domain"].to_numpy()),
        "region_probe_acc": region_probe(eval_x, eval_meta)["region_probe_acc"],
        "relative_change": float(
            np.mean(np.linalg.norm(diff, axis=1) / np.linalg.norm(raw_eval_x, axis=1))
        ),
    }


def main(argv: list[str] | None = None) -> Path:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--tag", required=True)
    parser.add_argument("--encoder", default="h0-mini")
    parser.add_argument(
        "--data-root", default="data/processed/SCORPION_tiles_224px_0p5mpp"
    )
    parser.add_argument("--split-seed", type=int, default=42)
    parser.add_argument("--val-fraction", type=float, default=0.1)
    parser.add_argument("--test-fraction", type=float, default=0.1)
    parser.add_argument(
        "--poolings", nargs="+", default=list(POOLINGS), choices=POOLINGS
    )
    parser.add_argument(
        "--lambdas", nargs="+", type=float, default=list(DEFAULT_LAMBDAS)
    )
    parser.add_argument(
        "--probe-train-max",
        type=int,
        default=20000,
        help="Train-split tiles for the sweep's scanner probes (val-sized; all: -1).",
    )
    parser.add_argument(
        "--region-guard",
        type=float,
        default=0.02,
        help="Flag the selection if its val region probe drops more than this.",
    )
    parser.add_argument(
        "--features-from",
        type=Path,
        help="Reuse features_<split>.npz from an earlier run dir of this script.",
    )
    parser.add_argument("--skip-test", action="store_true")
    parser.add_argument(
        "--limit-slides", type=int, help="Debug: slides per split (smoke test)."
    )
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--workers", type=int, default=8, help="0 in claude-box.")
    parser.add_argument("--device", type=int, default=0, help="CUDA index, -1 = CPU.")
    parser.add_argument(
        "--no-mixed-precision", dest="mixed_precision", action="store_false"
    )
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    device = torch.device("cpu" if args.device < 0 else f"cuda:{args.device}")
    if args.lambdas[0] != 0.0:
        args.lambdas = [0.0, *args.lambdas]  # lam = 0 is the raw reference

    dm_hparams = dict(
        data_root=args.data_root,
        encoder=args.encoder,
        group_column="slide_id",
        val_fraction=args.val_fraction,
        test_fraction=args.test_fraction,
        seed=args.split_seed,
        batch_size=args.batch_size,
        num_workers=args.workers,
        persistent_workers=False,
    )
    run_dir = new_bench_run_dir(f"scorpion_{args.tag}")
    write_run_info(
        run_dir,
        config={
            "benchmark": "scorpion_soft_shrinkage_sweep",
            **{k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()},
            "datamodule": dm_hparams,
            "eraser": {**ERASER_CONFIG, "source": asdict(SOURCE)},
            "deltas": "scanner_slide_to_mean (vfm-geom-xai)",
        },
        checkpoint={"encoder": args.encoder, "weights": "pretrained (timm/HF)"},
        features={
            "poolings": args.poolings,
            "transform": "Resize + ToTensor + Normalize (SCORPION eval transform)",
            "precision": "autocast float16" if args.mixed_precision else "float32",
        },
    )

    # --- features --------------------------------------------------------------
    splits = SPLITS[:2] if args.skip_test else SPLITS
    feats: dict[str, dict[str, np.ndarray]] = {}
    metas: dict[str, pd.DataFrame] = {}
    if args.features_from is not None:
        for split in splits:
            npz = np.load(args.features_from / f"features_{split}.npz")
            feats[split] = {p: npz[p] for p in POOLINGS}
            metas[split] = pd.read_csv(args.features_from / f"meta_{split}.csv")
    else:
        dm = SCORPIONDataModule(**dm_hparams)
        dm.setup()
        # Eval transform for the train split too: the eraser is fit on clean tiles.
        dm.train_dataset.transform = dm.val_dataset.transform
        datasets = {
            "train": dm.train_dataset,
            "val": dm.val_dataset,
            "test": dm.test_dataset,
        }
        if args.limit_slides is not None:
            for dataset in datasets.values():
                slides = dataset.metadata["slide_id"]
                keep = slides.isin(sorted(slides.unique())[: args.limit_slides])
                dataset.metadata = dataset.metadata.loc[keep].reset_index(drop=True)
        backbone = build_encoder(args.encoder).backbone.to(device).eval()
        for split in splits:
            log.info("Extracting %s split (%d tiles)", split, len(datasets[split]))
            loader = DataLoader(
                datasets[split],
                batch_size=args.batch_size,
                shuffle=False,
                num_workers=args.workers,
                pin_memory=device.type == "cuda",
            )
            feats[split], metas[split] = extract(
                backbone, loader, device, args.mixed_precision
            )
            np.savez(run_dir / f"features_{split}.npz", **feats[split])
            metas[split].to_csv(run_dir / f"meta_{split}.csv", index=False)
        del backbone
        torch.cuda.empty_cache()

    # --- sweep on val ----------------------------------------------------------
    train_meta, val_meta = metas["train"], metas["val"]
    probe_rows = subsample(
        train_meta, None if args.probe_train_max < 0 else args.probe_train_max, seed=0
    )
    probe_y = train_meta["domain"].to_numpy()[probe_rows]
    records, families = [], {}
    for pooling in args.poolings:
        family = fit_family(feats["train"][pooling], train_meta, device)
        families[pooling] = family
        for lam in args.lambdas:
            eraser = family.make_eraser(lam)
            metrics = evaluate(
                apply(eraser, feats["val"][pooling], device),
                val_meta,
                apply(eraser, feats["train"][pooling][probe_rows], device),
                probe_y,
                feats["val"][pooling],
                device,
            )
            records.append({"pooling": pooling, "lambda": lam, **metrics})
            log.info("val %s lam=%g: %s", pooling, lam, metrics)
            pd.DataFrame(records).to_csv(
                run_dir / "sweep_val.csv", index=False, float_format="%.4f"
            )
    sweep = pd.DataFrame(records)

    selected = {}
    for pooling, rows in sweep.groupby("pooling", sort=False):
        best = rows.sort_values(
            ["top_1_accuracy", "lambda"], ascending=[False, True]
        ).iloc[0]
        raw = rows.loc[rows["lambda"] == 0.0].iloc[0]
        region_drop = float(raw["region_probe_acc"] - best["region_probe_acc"])
        selected[pooling] = {
            "lambda": float(best["lambda"]),
            "val_top_1": float(best["top_1_accuracy"]),
            "val_top_1_raw": float(raw["top_1_accuracy"]),
            "val_region_drop": region_drop,
            "region_guard_violated": region_drop > args.region_guard,
            "at_grid_edge": float(best["lambda"]) == max(args.lambdas),
        }
        eraser = families[pooling].make_eraser(float(best["lambda"]))
        torch.save(
            {
                "P": eraser.P.cpu(),
                "bias": eraser.bias.cpu(),
                "lambda": float(best["lambda"]),
                "pooling": pooling,
                "encoder": args.encoder,
                "eraser_config": ERASER_CONFIG,
            },
            run_dir / f"eraser_{pooling}_lam{best['lambda']:g}.pt",
        )
    (run_dir / "selected.json").write_text(json.dumps(selected, indent=2) + "\n")
    log.info("Selected: %s", selected)

    # --- test, once: proxy protocol (probe trained on val) ---------------------
    if not args.skip_test:
        test_records = []
        for pooling, choice in selected.items():
            for lam in dict.fromkeys([0.0, choice["lambda"]]):
                eraser = families[pooling].make_eraser(lam)
                metrics = evaluate(
                    apply(eraser, feats["test"][pooling], device),
                    metas["test"],
                    apply(eraser, feats["val"][pooling], device),
                    val_meta["domain"].to_numpy(),
                    feats["test"][pooling],
                    device,
                )
                test_records.append({"pooling": pooling, "lambda": lam, **metrics})
                log.info("test %s lam=%g: %s", pooling, lam, metrics)
        pd.DataFrame(test_records).to_csv(
            run_dir / "summary_test.csv", index=False, float_format="%.4f"
        )
        print(pd.DataFrame(test_records).to_string(float_format="%.3f"))

    print(sweep.to_string(float_format="%.3f"))
    log.info("Soft-shrinkage sweep results: %s", run_dir)
    return run_dir


if __name__ == "__main__":
    main()
