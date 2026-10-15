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
- `--deltas` sets the pairing unit of the deltas (roadmap Experiment 2C): `slide`
  (default, the vfm-geom deltas above), `region` (ROI) or `tile` (per matched tile:
  each scanner's view minus the mean over scanners of that tile). Only B changes; A,
  the eraser options and the lambda rule are the same.
- Sweep on val: cross-scanner retrieval, scanner probes (train -> val) and the region
  probe, the metrics of `sipe.eval.scorpion_retrieval`. The selected lambda is the
  smallest one whose val top-1 gain over lambda = 0 reaches `--select-gain-frac` of
  the best gain (default 0.95). The eraser is meant to shrink scanner-induced
  distances, not to erase the scanner, so lambda is not pushed to full removal;
  scanner probes are diagnostics. `--select-gain-frac 1` = best val top-1.
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
from sipe.eval.proxy import evaluate, select_lambda, subsample  # noqa: F401 (re-export)
from sipe.features.extract import POOLINGS, extract
from sipe.model.encoders import build_encoder

log = logging.getLogger(__name__)

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
# Pairing unit of the scanner deltas -> meta column that defines the group.
DELTA_LEVELS = {"slide": "slide_id", "region": "region_id", "tile": "pair_id"}
SOURCE = DeltaSourceSpec(
    name="scanner", moment="second_moment", shrinkage=False, normalization="trace"
)


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


def group_to_mean_deltas(
    feats: np.ndarray, meta: pd.DataFrame, group: str
) -> np.ndarray:
    """Per group: mean(group, scanner) - mean over the group's scanners.

    `slide_to_mean_deltas` for any grouping column; vectorized, so `pair_id` (one
    group per matched tile) is cheap. Groups seen by fewer than 2 scanners are dropped.
    Rows are ordered by group, then scanner (B is a second moment: order-free).
    """
    g, _ = pd.factorize(meta[group], sort=True)
    s, _ = pd.factorize(meta["scanner"], sort=True)
    n_g, n_s = g.max() + 1, s.max() + 1
    rows = meta["row"].to_numpy()
    sums = np.zeros((n_g * n_s, feats.shape[1]))
    np.add.at(sums, g * n_s + s, feats[rows].astype(np.float64))
    counts = np.bincount(g * n_s + s, minlength=n_g * n_s).reshape(n_g, n_s)
    means = sums.reshape(n_g, n_s, -1) / np.maximum(counts, 1)[..., None]
    present = counts > 0
    centre = (means * present[..., None]).sum(1) / present.sum(1)[:, None]
    deltas = means - centre[:, None]
    keep = present & (present.sum(1) >= 2)[:, None]
    return deltas[keep].astype(np.float32)


def fit_family(
    feats: np.ndarray,
    meta: pd.DataFrame,
    device: torch.device,
    deltas: str = "slide",
):
    if deltas == "slide":
        delta_rows = slide_to_mean_deltas(feats, meta)
    else:
        delta_rows = group_to_mean_deltas(feats, meta, DELTA_LEVELS[deltas])
    fitter = PairedDeltaFitter(x_dim=feats.shape[1], device=device, dtype=torch.float32)
    fitter.update_x(torch.from_numpy(feats).to(device))
    fitter.update_delta_source(SOURCE.name, torch.from_numpy(delta_rows).to(device))
    return fitter.prepare_soft_eraser_family(delta_sources=[SOURCE], **ERASER_CONFIG)


def apply(eraser, feats: np.ndarray, device: torch.device) -> np.ndarray:
    return eraser(torch.from_numpy(feats).to(device)).cpu().numpy()


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
        "--deltas",
        default="slide",
        choices=list(DELTA_LEVELS),
        help="Pairing unit of the scanner deltas (B); slide = vfm-geom default.",
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
        "--select-gain-frac",
        type=float,
        default=0.95,
        help="Pick the smallest lambda reaching this share of the best val top-1 gain.",
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
            "delta_level": args.deltas,
            "deltas": "scanner_slide_to_mean (vfm-geom-xai)"
            if args.deltas == "slide"
            else f"scanner_{args.deltas}_to_mean (group: {DELTA_LEVELS[args.deltas]})",
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
        family = fit_family(
            feats["train"][pooling], train_meta, device, deltas=args.deltas
        )
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
        best = select_lambda(rows, args.select_gain_frac)
        raw = rows.loc[rows["lambda"] == 0.0].iloc[0]
        region_drop = float(raw["region_probe_acc"] - best["region_probe_acc"])
        selected[pooling] = {
            "lambda": float(best["lambda"]),
            "deltas": args.deltas,
            "rule": f"smallest lambda with val top-1 gain >= {args.select_gain_frac:g}"
            " x best gain",
            "val_top_1": float(best["top_1_accuracy"]),
            "val_top_1_best": float(rows["top_1_accuracy"].max()),
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
                "deltas": args.deltas,
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
