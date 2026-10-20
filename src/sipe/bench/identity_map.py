"""Identity map file: the raw pooled backbone embedding through the `--posthoc` path.

Raw rows of a backbone (PLISM / HEST) then go through exactly the code of the fitted
maps (same pooling, autocast, float32), without a training checkpoint. Writes
`runs/bench/<stamp>_<host>_<rand>_<tag>/map_identity.pt` (affine, P = I, bias = 0, with
`encoder` and `pooling`) plus config / git info:

    uv run python -m sipe.bench.identity_map --encoder uni2-h --pooling cls \\
        --tag identity_uni2h_cls
    bench/run python -m sipe.bench.plism \\
        --posthoc runs/bench/<that run>/map_identity.pt \\
        --tag uni2h_cls_ref
"""

from __future__ import annotations

import argparse
from pathlib import Path

import torch

from sipe.bench.posthoc import identity_map
from sipe.bench.provenance import new_bench_run_dir, write_run_info
from sipe.model.encoders import ENCODERS, POOLINGS

MAP_FILENAME = "map_identity.pt"


def main(argv: list[str] | None = None) -> Path:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--encoder", required=True, choices=sorted(ENCODERS))
    parser.add_argument("--pooling", required=True, choices=POOLINGS)
    parser.add_argument("--tag", required=True)
    args = parser.parse_args(argv)
    run_dir = new_bench_run_dir(args.tag)
    write_run_info(
        run_dir,
        config={"benchmark": "identity_map", **vars(args)},
        checkpoint={"encoder": args.encoder, "weights": "none (identity map)"},
        features={"pooling": args.pooling},
    )
    torch.save(identity_map(args.encoder, args.pooling), run_dir / MAP_FILENAME)
    print(run_dir / MAP_FILENAME)
    return run_dir


if __name__ == "__main__":
    main()
