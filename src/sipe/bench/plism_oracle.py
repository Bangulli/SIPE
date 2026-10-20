"""PLISM condition-mean oracle: remove each slide's (condition's) mean embedding.

Bench env only:

    bench/run python -m sipe.bench.plism_oracle \\
        --features-from runs/bench/<plism run> --tag <tag> [--center global|zero]

ORACLE, not a method: it uses the PLISM condition (slide = staining x scanner) at
inference, which the contract forbids. For each of the 91 slides, mu_c = mean over all
of its tiles of the source run's features, and every tile becomes x - mu_c + m with
m = mean of the mu_c (`global`, keeps the origin, as the affine maps whose bias is the
train mean) or m = 0 (`zero`). Pairing unit: PLISM slide (condition) means. It bounds
per-condition translations only; a linear map that also rescales can in principle
exceed it on cosine / top-k.

No extraction: the source run's `features/<tag>/<slide>/features.npy` (coords in the
first 3 columns, plismbench `save_features` layout) are rewritten into this run, then
the official `compute_metrics` runs as in `sipe.bench.plism`.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from sipe.bench.checkpoint import sha256
from sipe.bench.provenance import new_bench_run_dir, write_run_info

N_COORDS = 3
CENTERS = ("global", "zero")


def slide_mean(features: np.ndarray) -> np.ndarray:
    """Mean over tiles of the feature columns, accumulated in float64."""
    return features[:, N_COORDS:].mean(axis=0, dtype=np.float64)


def condition_mean_oracle(
    features: np.ndarray, m: np.ndarray, out: np.ndarray | None = None
) -> np.ndarray:
    """[N, 3 + D] slide matrix -> x - mean(x) + m, coords untouched, float32.

    The shift (m - mean) is computed in float64 and applied in float32, written into
    `out` if given (a memmap: large fresh allocations are slow in claude-box).
    """
    if out is None:
        out = np.empty(features.shape, dtype=np.float32)
    shift = (m - slide_mean(features)).astype(np.float32)
    out[:, :N_COORDS] = features[:, :N_COORDS]
    np.add(features[:, N_COORDS:], shift, out=out[:, N_COORDS:])
    return out


def source_features_dir(run_dir: Path) -> Path:
    dirs = [d for d in (run_dir / "features").iterdir() if d.is_dir()]
    if len(dirs) != 1:
        raise ValueError(f"{run_dir}/features: expected one extractor dir, got {dirs}.")
    return dirs[0]


def main(argv: list[str] | None = None) -> Path:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--features-from",
        type=Path,
        required=True,
        help="A sipe.bench.plism run dir (e.g. the backbone_gap reference).",
    )
    parser.add_argument("--tag", required=True)
    parser.add_argument("--center", default="global", choices=CENTERS)
    parser.add_argument("--n-tiles", type=int, default=8139)
    parser.add_argument("--metrics-device", default="gpu", choices=["gpu", "cpu"])
    parser.add_argument(
        "--smoke",
        action="store_true",
        help="Allow fewer than 91 slides (patches plismbench NUM_SLIDES).",
    )
    args = parser.parse_args(argv)

    # Bench-env imports here, so the transform is testable in the main env.
    from loguru import logger
    from plismbench.engine.evaluate import compute_metrics

    from sipe.bench.plism import _patch_num_slides

    source_run = args.features_from.resolve()
    source_dir = source_features_dir(source_run)
    slide_files = sorted(source_dir.glob("*/features.npy"))
    if args.smoke:
        _patch_num_slides(len(slide_files))

    run_dir = new_bench_run_dir(args.tag)
    features_root = run_dir / "features"
    metrics_root = run_dir / "metrics"
    export_dir = features_root / args.tag

    names = [f.parent.name for f in slide_files]
    mus = np.stack([slide_mean(np.load(f, mmap_mode="r")) for f in slide_files])
    m = mus.mean(axis=0) if args.center == "global" else np.zeros(mus.shape[1])
    np.savez(run_dir / "condition_means.npz", slides=np.array(names), mu=mus, m=m)

    source_prov_path = source_run / "provenance.json"
    source_prov = (
        json.loads(source_prov_path.read_text()) if source_prov_path.exists() else {}
    )
    config = {
        "benchmark": "plism",
        "oracle": "condition_mean",
        "features_from": str(source_run),
        "source_features_dir": str(source_dir),
        "tag": args.tag,
        "center": args.center,
        "n_tiles": args.n_tiles,
        "metrics_device": args.metrics_device,
        "smoke": args.smoke,
        "n_slides": len(slide_files),
        "evaluate": {
            "features_root_dir": str(features_root),
            "metrics_save_dir": str(metrics_root),
            "extractor": args.tag,
            "n_tiles": args.n_tiles,
            "device": args.metrics_device,
        },
    }
    write_run_info(
        run_dir,
        config=config,
        checkpoint={
            "kind": "oracle",
            "oracle": "condition_mean",
            "center": args.center,
            "uses_condition_label": True,
            "source_run_dir": str(source_run),
            "source_checkpoint": source_prov.get("checkpoint"),
            "source_provenance_sha256": (
                sha256(source_prov_path) if source_prov_path.exists() else None
            ),
        },
        features={
            **source_prov.get("features", {}),
            "oracle": f"x - mu_slide + m (m = {args.center})",
            "mu": "per-slide mean over all tiles, float64 (condition_means.npz)",
        },
    )
    logger.info(f"Oracle run dir: {run_dir}")

    for f in slide_files:
        out = export_dir / f.parent.name / "features.npy"
        out.parent.mkdir(parents=True)
        src = np.load(f, mmap_mode="r")
        dst = np.lib.format.open_memmap(
            out, mode="w+", dtype=np.float32, shape=src.shape
        )
        condition_mean_oracle(src, m, out=dst)
        dst.flush()
        del dst

    compute_metrics(
        features_root_dir=features_root,
        metrics_save_dir=metrics_root,
        extractor=args.tag,
        n_tiles=args.n_tiles,
        device=args.metrics_device,
    )
    results = metrics_root / f"{args.n_tiles}_tiles" / args.tag
    logger.success(f"PLISM results: {results}")
    return run_dir


if __name__ == "__main__":
    main()
