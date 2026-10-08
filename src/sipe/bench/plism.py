"""PLISM robustness benchmark of a CATS checkpoint, via the official plismbench code.

Bench env only:

    bench/run python -m sipe.bench.plism \\
        --ckpt runs/<run>/checkpoints/last.ckpt --tag cats_<run>

Extraction goes through the fork hook (`init_extractor("module:factory")`), so
plismbench runs its own dataloader, `process_imgs` and `save_features`; metrics come
from the official `compute_metrics`.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import plismbench.engine.extract.extract_from_h5 as extract_from_h5
import plismbench.utils.evaluate as evaluate_utils
import torch
from loguru import logger
from plismbench.engine.evaluate import compute_metrics
from plismbench.engine.extract.core import run_extract
from plismbench.models.extractor import Extractor
from plismbench.models.utils import DEFAULT_DEVICE, prepare_module

from sipe.bench.checkpoint import load_cats_checkpoint
from sipe.bench.encoders import REPRESENTATIONS, CATSFeatures, plism_transform
from sipe.bench.provenance import SIPE_ROOT, new_bench_run_dir, write_run_info

FACTORY = "sipe.bench.plism:cats_extractor"
DEFAULT_DOWNLOAD_DIR = SIPE_ROOT.parent / "plism-benchmark" / "data" / "plism"


class CATSExtractor(Extractor):
    """A CATS checkpoint as a plismbench Extractor (mirrors plismbench's H0Mini)."""

    def __init__(
        self,
        ckpt: str | Path,
        device: int | list[int] | None = DEFAULT_DEVICE,
        representation: str = "z_gap",
        mixed_precision: bool = True,
    ) -> None:
        super().__init__()
        loaded = load_cats_checkpoint(ckpt)
        features = CATSFeatures(loaded.network, representation)
        self.output_dim = features.output_dim
        self.mixed_precision = mixed_precision
        self.transform = plism_transform(loaded.network.encoder_meta)
        self.feature_extractor, self.device = prepare_module(
            features, device, self.mixed_precision
        )
        if self.device is None:
            self.feature_extractor = self.feature_extractor.module

    def __call__(self, images: torch.Tensor) -> np.ndarray:
        return self.feature_extractor(images.to(self.device)).cpu().numpy()


def cats_extractor(
    device: int | list[int] | None,
    ckpt: str,
    representation: str = "z_gap",
    mixed_precision: bool = True,
) -> Extractor:
    """Factory for the fork hook. mixed_precision=True as FeatureExtractorsEnum.init."""
    return CATSExtractor(
        ckpt=ckpt,
        device=device,
        representation=representation,
        mixed_precision=mixed_precision,
    )


def _patch_num_slides(n_slides: int) -> None:
    """Smoke runs only: plismbench asserts all 91 slides in extraction and pairing."""
    logger.warning(f"SMOKE: plismbench NUM_SLIDES patched 91 -> {n_slides}.")
    extract_from_h5.NUM_SLIDES = n_slides
    evaluate_utils.NUM_SLIDES = n_slides


def main(argv: list[str] | None = None) -> Path:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--ckpt", type=Path, required=True)
    parser.add_argument("--tag", required=True)
    parser.add_argument("--representation", default="z_gap", choices=REPRESENTATIONS)
    parser.add_argument(
        "--n-tiles",
        type=int,
        default=8139,
        help="Tiles per slide for metrics (plismbench default 8139; 460 = debug).",
    )
    parser.add_argument("--download-dir", type=Path, default=DEFAULT_DOWNLOAD_DIR)
    parser.add_argument("--device", type=int, default=0, help="CUDA index, -1 = CPU.")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument(
        "--workers",
        type=int,
        default=8,
        help="Extraction dataloader workers (plismbench default 8; 0 in claude-box).",
    )
    parser.add_argument("--metrics-device", default="gpu", choices=["gpu", "cpu"])
    parser.add_argument(
        "--no-mixed-precision", dest="mixed_precision", action="store_false"
    )
    parser.add_argument(
        "--smoke",
        action="store_true",
        help="Allow a download dir with fewer than 91 slides (patches NUM_SLIDES).",
    )
    args = parser.parse_args(argv)

    ckpt = args.ckpt.resolve()
    download_dir = args.download_dir.resolve()
    n_slides = len(list(download_dir.glob("*.tif.h5")))
    if args.smoke:
        _patch_num_slides(n_slides)

    run_dir = new_bench_run_dir(args.tag)
    features_root = run_dir / "features"
    metrics_root = run_dir / "metrics"
    extractor_kwargs = {
        "ckpt": str(ckpt),
        "representation": args.representation,
        "mixed_precision": args.mixed_precision,
    }
    config = {
        "benchmark": "plism",
        **{k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()},
        "ckpt": str(ckpt),
        "download_dir": str(download_dir),
        "n_slides": n_slides,
        "extract": {
            "feature_extractor_name": FACTORY,
            "extractor_kwargs": extractor_kwargs,
            "export_dir": str(features_root / args.tag),
            "batch_size": args.batch_size,
            "workers": args.workers,
            "streaming": False,
        },
        "evaluate": {
            "features_root_dir": str(features_root),
            "metrics_save_dir": str(metrics_root),
            "extractor": args.tag,
            "n_tiles": args.n_tiles,
            "device": args.metrics_device,
        },
    }
    # Loads the checkpoint once more than extraction does; cheap next to 91 slides.
    loaded = load_cats_checkpoint(ckpt)
    write_run_info(
        run_dir,
        config=config,
        checkpoint=loaded.provenance,
        features={
            "representation": args.representation,
            "output_dim": CATSFeatures(loaded.network, args.representation).output_dim,
            "transform": "ToTensor + Normalize (plismbench H0Mini style)",
            "mean": loaded.mean,
            "std": loaded.std,
            "precision": "autocast (PrecisionModule)"
            if args.mixed_precision
            else "float32",
        },
    )
    del loaded
    logger.info(f"Bench run dir: {run_dir}")

    run_extract(
        feature_extractor_name=FACTORY,
        batch_size=args.batch_size,
        device=args.device,
        export_dir=features_root / args.tag,
        download_dir=download_dir,
        streaming=False,
        workers=args.workers,
        extractor_kwargs=extractor_kwargs,
    )
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
