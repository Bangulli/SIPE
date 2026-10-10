"""In-domain proxy for PLISM: cross-scanner retrieval + scanner probes on SCORPION.

Fast screening of CATS checkpoints before the full PLISM/HEST benchmarks.

- Retrieval mirrors plismbench's `TopkAccuracy`/`CosineSimilarity`: for each held-out
  WSI and each scanner pair (a, b), the 490 tiles of a and b are matched by `pair_id`;
  each tile ranks all other tiles of the union (excluding itself) by raw cosine, and a
  hit is its matched tile from the other scanner. Both directions are averaged. Here
  the "slide pair" is one WSI seen by two scanners, so only scanner varies.
- Scanner probes (leakage the training adversary may miss): standardized logistic
  regression and a 1-hidden-layer MLP, trained on `--probe-train-split` features and
  tested on `--split` features (different WSIs). Chance = 1 / n_scanners.
- Region probe (content that survives a scanner change): within `--split`, a
  standardized logistic regression predicts the tissue region (slide + sample, 49
  tiles each) from tiles of all scanners but one and is tested on the held-out
  scanner; accuracy is averaged over held-out scanners. SCORPION has no tissue
  labels, so region identity stands in for content. Chance = 1 / n_regions.

`pair_id` is used for evaluation only.

    uv run python -m sipe.eval.scorpion_retrieval \
        --ckpt runs/<run>/checkpoints/last.ckpt --tag <tag>
"""

from __future__ import annotations

import argparse
import inspect
import itertools
import json
import logging
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader

from sipe.bench.checkpoint import load_cats_checkpoint
from sipe.bench.encoders import REPRESENTATIONS, CATSFeatures
from sipe.bench.provenance import new_bench_run_dir, write_run_info
from sipe.data.scorpion import SCORPIONDataModule

log = logging.getLogger(__name__)

TOPK = (1, 3, 5, 10)
DEFAULT_REPRESENTATIONS = ("backbone_gap", "z_gap", "z_pre_gap", "s")


@torch.no_grad()
def extract(
    features: dict[str, CATSFeatures],
    loader: DataLoader,
    device: torch.device,
    mixed_precision: bool,
) -> tuple[dict[str, np.ndarray], pd.DataFrame]:
    """One pass over the loader; every representation sees the same batches."""
    out: dict[str, list[np.ndarray]] = {name: [] for name in features}
    rows: list[pd.DataFrame] = []
    for batch in loader:
        images = batch["image"].to(device, non_blocking=True)
        # float16 autocast as plismbench's PrecisionModule.
        with torch.autocast(
            device.type, dtype=torch.float16, enabled=mixed_precision
        ):
            for name, module in features.items():
                out[name].append(module(images).float().cpu().numpy())
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
    return {k: np.concatenate(v) for k, v in out.items()}, meta


def _cosine(x: torch.Tensor) -> torch.Tensor:
    x = x / x.norm(dim=1, keepdim=True).clamp_min(1e-12)
    return x @ x.T


def plism_pair_metrics(a: np.ndarray, b: np.ndarray, device: torch.device) -> dict:
    """plismbench TopkAccuracy + CosineSimilarity; tiles row-aligned a[i] <-> b[i]."""
    n = a.shape[0]
    ab = torch.from_numpy(np.concatenate([a, b])).to(device, torch.float32)
    cos = _cosine(ab)
    cos.fill_diagonal_(-float("inf"))  # exclude self-match
    top = cos.topk(max(TOPK), dim=1).indices
    target = torch.cat([torch.arange(n, 2 * n), torch.arange(0, n)]).to(device)
    hits = top == target[:, None]
    metrics = {"cosine_similarity": float(cos[torch.arange(n), target[:n]].mean())}
    for k in TOPK:
        # mean over all 2n rows == average of the two directions (equal sizes).
        metrics[f"top_{k}_accuracy"] = float(hits[:, :k].any(dim=1).float().mean())
    return metrics


def retrieval(
    feats: np.ndarray, meta: pd.DataFrame, device: torch.device
) -> pd.DataFrame:
    records = []
    scanners = sorted(meta["scanner"].unique())
    for slide_id, slide in meta.groupby("slide_id"):
        by_scanner = {
            s: slide.loc[slide["scanner"] == s].set_index("pair_id") for s in scanners
        }
        for sa, sb in itertools.combinations(scanners, 2):
            common = by_scanner[sa].index.intersection(by_scanner[sb].index)
            if len(common) < 2:
                continue
            ia = by_scanner[sa].loc[common, "row"].to_numpy()
            ib = by_scanner[sb].loc[common, "row"].to_numpy()
            records.append(
                {
                    "slide_id": slide_id,
                    "scanner_a": sa,
                    "scanner_b": sb,
                    "n_tiles": len(common),
                    **plism_pair_metrics(feats[ia], feats[ib], device),
                }
            )
    return pd.DataFrame(records)


def scanner_probes(
    train_x: np.ndarray, train_y: np.ndarray, test_x: np.ndarray, test_y: np.ndarray
) -> dict[str, float]:
    probes = {
        "probe_logreg_acc": make_pipeline(
            StandardScaler(), LogisticRegression(max_iter=2000)
        ),
        "probe_mlp_acc": make_pipeline(
            StandardScaler(),
            MLPClassifier(
                hidden_layer_sizes=(256,),
                early_stopping=True,
                max_iter=200,
                random_state=0,
            ),
        ),
    }
    return {
        name: float(probe.fit(train_x, train_y).score(test_x, test_y))
        for name, probe in probes.items()
    }


def region_probe(feats: np.ndarray, meta: pd.DataFrame) -> dict[str, float]:
    """Leave-one-scanner-out linear probe of region identity."""
    regions = meta["region_id"].to_numpy()
    scanners = meta["scanner"].to_numpy()
    accuracies = {}
    for held_out in sorted(set(scanners)):
        test = scanners == held_out
        probe = make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000))
        probe.fit(feats[~test], regions[~test])
        accuracies[held_out] = float(probe.score(feats[test], regions[test]))
    return {
        "region_probe_acc": float(np.mean(list(accuracies.values()))),
        **{f"region_probe_acc_{k}": v for k, v in accuracies.items()},
    }


def main(argv: list[str] | None = None) -> Path:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--ckpt", type=Path, required=True)
    parser.add_argument("--tag", required=True)
    parser.add_argument(
        "--representations",
        nargs="+",
        default=list(DEFAULT_REPRESENTATIONS),
        choices=REPRESENTATIONS,
    )
    parser.add_argument("--split", default="test", choices=["val", "test"])
    parser.add_argument("--probe-train-split", default="val", choices=["val", "test"])
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--workers", type=int, default=8, help="0 in claude-box.")
    parser.add_argument("--device", type=int, default=0, help="CUDA index, -1 = CPU.")
    parser.add_argument(
        "--no-mixed-precision", dest="mixed_precision", action="store_false"
    )
    args = parser.parse_args(argv)
    if args.split == args.probe_train_split:
        parser.error("--split and --probe-train-split must differ (WSI-disjoint).")
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")

    device = torch.device("cpu" if args.device < 0 else f"cuda:{args.device}")
    ckpt = args.ckpt.resolve()
    loaded = load_cats_checkpoint(ckpt)
    network = loaded.network.to(device).eval()
    if getattr(network, "needs_domains", False) and "s" in args.representations:
        # Label-conditioned s is the scanner embedding itself: nothing to evaluate.
        log.info("Skipping representation 's' (label-conditioned network).")
        args.representations = [r for r in args.representations if r != "s"]
    features = {r: CATSFeatures(network, r) for r in args.representations}

    # Same data config (root, split seed/fractions) as the training run. Subclasses
    # such as PairedSCORPIONDataModule keep the parent's splits; their extra arguments
    # (views, ...) only change batching, so keep what the plain datamodule accepts.
    raw = torch.load(ckpt, map_location="cpu", weights_only=False, mmap=True)
    accepted = inspect.signature(SCORPIONDataModule.__init__).parameters
    dm_hparams = {
        k: v
        for k, v in raw["datamodule_hyper_parameters"].items()
        if not k.startswith("_") and k in accepted
    }
    del raw
    dm_hparams.update(
        batch_size=args.batch_size,
        num_workers=args.workers,
        persistent_workers=False,
    )
    dm = SCORPIONDataModule(**dm_hparams)
    dm.setup()
    loaders = {"val": dm.val_dataloader(), "test": dm.test_dataloader()}

    run_dir = new_bench_run_dir(f"scorpion_{args.tag}")
    write_run_info(
        run_dir,
        config={
            "benchmark": "scorpion_retrieval",
            **{k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()},
            "ckpt": str(ckpt),
            "datamodule": dm_hparams,
        },
        checkpoint=loaded.provenance,
        features={
            "representations": {r: f.output_dim for r, f in features.items()},
            "transform": "Resize + ToTensor + Normalize (SCORPION eval transform)",
            "mean": loaded.mean,
            "std": loaded.std,
            "precision": "autocast float16" if args.mixed_precision else "float32",
        },
    )

    extracted = {}
    for split in dict.fromkeys([args.split, args.probe_train_split]):
        log.info("Extracting %s split (%d tiles)", split, len(loaders[split].dataset))
        feats, meta = extract(features, loaders[split], device, args.mixed_precision)
        meta["row"] = np.arange(len(meta))
        extracted[split] = (feats, meta)

    test_feats, test_meta = extracted[args.split]
    train_feats, train_meta = extracted[args.probe_train_split]
    summary, per_pair = {}, []
    for rep in args.representations:
        pairs = retrieval(test_feats[rep], test_meta, device).assign(
            representation=rep
        )
        per_pair.append(pairs)
        metric_cols = ["cosine_similarity", *(f"top_{k}_accuracy" for k in TOPK)]
        summary[rep] = {
            **{c: float(pairs[c].mean()) for c in metric_cols},
            **scanner_probes(
                train_feats[rep],
                train_meta["domain"].to_numpy(),
                test_feats[rep],
                test_meta["domain"].to_numpy(),
            ),
            **region_probe(test_feats[rep], test_meta),
        }
        log.info("%s: %s", rep, summary[rep])

    pd.concat(per_pair).to_csv(run_dir / "retrieval_per_pair.csv", index=False)
    table = pd.DataFrame(summary).T
    table.index.name = "representation"
    table.to_csv(run_dir / "summary.csv", float_format="%.4f")
    (run_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(table.to_string(float_format="%.3f"))
    log.info("SCORPION proxy results: %s", run_dir)
    return run_dir


if __name__ == "__main__":
    main()
