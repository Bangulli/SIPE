"""HEST-Benchmark of a CATS checkpoint, through the official `hest.bench.benchmark`.

Bench env only:

    bench/run python -m sipe.bench.hest --ckpt runs/<run>/checkpoints/last.ckpt \\
        --tag cats_<run> --datasets '*'

All settings are passed as kwargs (no config file: `benchmark` lets a config file
override kwargs). Defaults otherwise follow HEST's `BenchmarkConfig`.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import torch
import torch.nn as nn
from hest.bench.benchmark import benchmark
from loguru import logger

from sipe.bench.checkpoint import load_cats_checkpoint
from sipe.bench.encoders import REPRESENTATIONS, CATSFeatures, hest_transform
from sipe.bench.provenance import SIPE_ROOT, new_bench_run_dir, write_run_info

DEFAULT_BENCH_DATA = SIPE_ROOT.parent / "HEST" / "data" / "bench_data"
PRECISIONS = {
    "float16": torch.float16,
    "bfloat16": torch.bfloat16,
    "float32": torch.float32,
}


class HESTEncoder(nn.Module):
    """TRIDENT-style encoder (`eval_transforms`, `precision`, `forward`) for HEST."""

    def __init__(
        self,
        features: CATSFeatures,
        eval_transforms,
        precision: torch.dtype = torch.float16,
    ) -> None:
        super().__init__()
        self.model = features
        self.eval_transforms = eval_transforms
        # float16 autocast, as TRIDENT's h0-mini encoder.
        self.precision = precision

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        return self.model(images)


def main(argv: list[str] | None = None) -> Path:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--ckpt", type=Path, required=True)
    parser.add_argument("--tag", required=True)
    parser.add_argument("--representation", default="z_gap", choices=REPRESENTATIONS)
    parser.add_argument(
        "--datasets", nargs="+", default=["IDC"], help="HEST tasks, or '*' for all."
    )
    parser.add_argument("--bench-data-root", type=Path, default=DEFAULT_BENCH_DATA)
    parser.add_argument("--precision", default="float16", choices=sorted(PRECISIONS))
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument(
        "--num-workers",
        type=int,
        default=1,
        help="Embedding dataloader workers (HEST default 1; use 0 in claude-box).",
    )
    args = parser.parse_args(argv)

    ckpt = args.ckpt.resolve()
    loaded = load_cats_checkpoint(ckpt)
    features = CATSFeatures(loaded.network, args.representation)
    encoder = HESTEncoder(
        features, hest_transform(loaded.network), PRECISIONS[args.precision]
    )

    run_dir = new_bench_run_dir(args.tag)
    bench_kwargs = {
        "bench_data_root": str(args.bench_data_root.resolve()),
        "skip_download": True,
        "embed_dataroot": str(run_dir / "embeddings"),
        "results_dir": str(run_dir / "results"),
        "custom_encoder_name": args.tag,
        "exp_code": args.tag,
        # Default ['resnet50'] would be benchmarked alongside.
        "encoders": [],
        "datasets": args.datasets,
        "batch_size": args.batch_size,
        "num_workers": args.num_workers,
    }
    config = {
        "benchmark": "hest",
        **{k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()},
        "ckpt": str(ckpt),
        "benchmark_kwargs": bench_kwargs,
    }
    write_run_info(
        run_dir,
        config=config,
        checkpoint=loaded.provenance,
        features={
            "representation": args.representation,
            "output_dim": features.output_dim,
            "transform": repr(encoder.eval_transforms),
            "mean": loaded.mean,
            "std": loaded.std,
            "precision": f"autocast {args.precision}",
        },
    )
    logger.info(f"Bench run dir: {run_dir}")

    benchmark(encoder, None, None, **bench_kwargs)
    logger.success(f"HEST results: {run_dir / 'results'}")
    return run_dir


if __name__ == "__main__":
    main()
