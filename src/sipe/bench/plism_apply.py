"""PLISM for a post-hoc map without re-extraction: apply the map to stored raw features.

Bench env only:

    bench/run python -m sipe.bench.plism_apply \\
        --features-from runs/bench/<raw PLISM run> --posthoc <map file> --tag <tag>

`sipe.bench.plism --posthoc` runs the frozen backbone on all 91 slides for every map.
A post-hoc map only sees the pooled embedding, so the same rows come from the raw
features of one earlier PLISM run (`backbone_gap`, or `posthoc_<pooling>` with the
identity map) with the map applied in float32, as `PosthocFeatures` does. Then the
official `compute_metrics` runs as in `sipe.bench.plism` (as
`sipe.bench.plism_oracle`, which rewrites a source run the same way). Checked on
2026-10-23: G0 and V1-S 7b71 applied to the features of
`runs/bench/2026-10-13_10-02-08_lxbelshark_053f_h0_mini_gap_ref` reproduce the features
of their full runs to float32 rounding (max relative difference 5e-6).

The source run must have the map's encoder and pooling (refused otherwise).
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import torch

from sipe.bench.checkpoint import sha256
from sipe.bench.plism_oracle import N_COORDS, source_features_dir
from sipe.bench.posthoc import load_posthoc_map, representation
from sipe.bench.provenance import new_bench_run_dir, write_run_info

CHUNK = 65536  # tiles per map call


def source_encoder_pooling(provenance: dict[str, Any]) -> tuple[str, str]:
    """(encoder, pooling) of a PLISM run's raw features; refuses mapped features."""
    rep = (provenance.get("features") or {}).get("representation")
    ckpt = provenance.get("checkpoint") or {}
    if rep == "backbone_gap":
        network = (ckpt.get("network") or {}).get("init_args") or {}
        return str(network.get("encoder") or "h0-mini"), "gap"
    if rep and rep.startswith("posthoc_"):
        if not (ckpt.get("file") or {}).get("identity"):
            raise ValueError(f"source features are {rep} through a non-identity map.")
        return str(ckpt.get("encoder") or "h0-mini"), rep.removeprefix("posthoc_")
    raise ValueError(f"source representation {rep!r} is not a raw backbone embedding.")


def apply_to_slide(
    fmap: torch.nn.Module, src: np.ndarray, out: np.ndarray, device: torch.device
) -> None:
    """[N, 3 + D] slide matrix -> coords + map(features), float32, into `out`."""
    out[:, :N_COORDS] = src[:, :N_COORDS]
    with torch.no_grad():
        for i in range(0, len(src), CHUNK):
            x = torch.from_numpy(np.ascontiguousarray(src[i : i + CHUNK, N_COORDS:]))
            out[i : i + CHUNK, N_COORDS:] = fmap(x.float().to(device)).cpu().numpy()


def main(argv: list[str] | None = None) -> Path:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--features-from",
        type=Path,
        required=True,
        help="sipe.bench.plism run with raw features.",
    )
    parser.add_argument("--posthoc", type=Path, required=True, help="Map file.")
    parser.add_argument("--tag", required=True)
    parser.add_argument("--n-tiles", type=int, default=8139)
    parser.add_argument("--device", type=int, default=0, help="CUDA index, -1 = CPU.")
    parser.add_argument("--metrics-device", default="gpu", choices=["gpu", "cpu"])
    parser.add_argument(
        "--smoke", action="store_true", help="Allow fewer than 91 slides."
    )
    args = parser.parse_args(argv)

    # Bench-env imports here, so the transform is testable in the main env.
    from loguru import logger
    from plismbench.engine.evaluate import compute_metrics

    from sipe.bench.plism import _patch_num_slides

    source_run = args.features_from.resolve()
    source_prov_path = source_run / "provenance.json"
    source_prov = json.loads(source_prov_path.read_text())
    encoder, pooling = source_encoder_pooling(source_prov)
    loaded = load_posthoc_map(args.posthoc)
    if (loaded.encoder, loaded.pooling) != (encoder, pooling):
        raise ValueError(
            f"map is {loaded.encoder}/{loaded.pooling}, source features are "
            f"{encoder}/{pooling}."
        )
    source_dir = source_features_dir(source_run)
    slide_files = sorted(source_dir.glob("*/features.npy"))
    if args.smoke:
        _patch_num_slides(len(slide_files))
    device = torch.device("cpu" if args.device < 0 else f"cuda:{args.device}")
    fmap = loaded.map.to(device)

    run_dir = new_bench_run_dir(args.tag)
    features_root = run_dir / "features"
    metrics_root = run_dir / "metrics"
    export_dir = features_root / args.tag
    config = {
        "benchmark": "plism",
        "tag": args.tag,
        "posthoc": str(args.posthoc.resolve()),
        "representation": representation(pooling),
        "features_from": str(source_run),
        "source_features_dir": str(source_dir),
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
            **loaded.provenance,
            "encoder": loaded.encoder,
            "pooling": loaded.pooling,
            "applied_to_stored_features": True,
            "source_run_dir": str(source_run),
            "source_provenance_sha256": sha256(source_prov_path),
        },
        features={
            **source_prov.get("features", {}),
            "representation": representation(pooling),
            "source_representation": source_prov["features"]["representation"],
            "precision": (source_prov.get("features") or {}).get("precision", "?")
            + "; map in float32 on the stored features",
        },
    )
    logger.info(f"Bench run dir: {run_dir}")

    for f in slide_files:
        out = export_dir / f.parent.name / "features.npy"
        out.parent.mkdir(parents=True)
        src = np.load(f, mmap_mode="r")
        with torch.no_grad():
            probe = fmap(torch.zeros(1, src.shape[1] - N_COORDS, device=device))
        dst = np.lib.format.open_memmap(
            out,
            mode="w+",
            dtype=np.float32,
            shape=(len(src), N_COORDS + probe.shape[1]),
        )
        apply_to_slide(fmap, src, dst, device)
        dst.flush()
        del dst

    compute_metrics(
        features_root_dir=features_root,
        metrics_save_dir=metrics_root,
        extractor=args.tag,
        n_tiles=args.n_tiles,
        device=args.metrics_device,
    )
    logger.success(
        f"PLISM results: {metrics_root / f'{args.n_tiles}_tiles' / args.tag}"
    )
    return run_dir


if __name__ == "__main__":
    main()
