"""SCORPION retrieval of saved map files on one cache split (no probes, no fitting).

For maps that were never part of a selection run (e.g. the intermediate
`map_cgm_step<step>.pt` of a fit): applies each map file to the cached features of
`--split` and reports the retrieval columns of `sipe.eval.proxy` (the same top-1 as
`cgm_select` / `mixture_sweep` summaries, without the scanner and region probes).
Nothing is selected here; reading test is for analysis rows only.

Outputs (run dir): `summary_<split>.csv` (one row per map, raw first).

CV mode (`--n-folds`, MIDL protocol): each `--maps` entry is `NAME=PATH_0,...,PATH_K-1`,
the map of each fold's fit (`sipe fit ... --data.init_args.n_folds=K
--data.init_args.fold=k`); map k is scored on the held-out WSIs of fold k
(`sipe.eval.proxy.load_cv_cache`, the datamodule's folds). Outputs `cv_folds.csv`
(name, fold, retrieval columns) and `cv_mean.csv` (mean and std over folds), the
columns of `sipe.eval.cv_sweep`; raw is a row of both.

    uv run python -m sipe.eval.map_retrieval --tag <tag> --split test \\
        --features-from runs/bench/<feature cache run> --maps <name>=<map file> ...
    uv run python -m sipe.eval.map_retrieval --tag <tag> --n-folds 4 \\
        --features-from runs/bench/<feature cache run> --maps <name>=<f0>,<f1>,<f2>,<f3>
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

import pandas as pd
import torch

from sipe.bench.posthoc import load_posthoc_map
from sipe.bench.provenance import new_bench_run_dir, write_run_info
from sipe.eval.cv_sweep import cv_mean
from sipe.eval.proxy import (
    apply_map,
    cache_encoder,
    check_map_matches_cache,
    load_cache,
    load_cv_cache,
    retrieval_metrics,
)


def run_cv(args, maps: dict[str, list[str]], run_dir: Path, device) -> None:
    rows = []
    for fold in range(args.n_folds):
        x, meta = load_cv_cache(
            args.features_from, args.n_folds, fold, args.pooling, args.fold_seed
        )
        x, meta = x["val"], meta["val"]
        base = {"fold": fold, "n_val_wsi": meta["slide_id"].nunique()}
        rows.append({"name": "raw", **base, **retrieval_metrics(x, meta, device),
                     "path": ""})  # fmt: skip
        for name, paths in maps.items():
            fmap = load_posthoc_map(paths[fold]).map
            rows.append({
                "name": name, **base,
                **retrieval_metrics(apply_map(fmap, x, device), meta, device),
                "path": paths[fold],
            })  # fmt: skip
            logging.info(
                "fold %d %s: top-1 %.4f", fold, name, rows[-1]["top_1_accuracy"]
            )
    folds = pd.DataFrame(rows)
    folds.to_csv(run_dir / "cv_folds.csv", index=False, float_format="%.4f")
    mean = cv_mean(folds, ["name"])
    mean.to_csv(run_dir / "cv_mean.csv", index=False, float_format="%.4f")
    print(mean.to_string(index=False, float_format="%.4f"))


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--tag", required=True)
    parser.add_argument("--features-from", type=Path, required=True)
    parser.add_argument("--maps", nargs="+", required=True, metavar="NAME=PATH")
    parser.add_argument("--split", default="test", choices=("val", "test"))
    parser.add_argument("--pooling", default="gap")
    parser.add_argument("--n-folds", type=int, help="CV mode (see the docstring).")
    parser.add_argument("--fold-seed", type=int, default=42)
    parser.add_argument("--device", type=int, default=0, help="CUDA index, -1 = CPU.")
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    device = torch.device("cpu" if args.device < 0 else f"cuda:{args.device}")
    maps = dict(m.split("=", 1) for m in args.maps)
    if args.n_folds is not None:
        maps = {name: paths.split(",") for name, paths in maps.items()}
        if bad := [n for n, p in maps.items() if len(p) != args.n_folds]:
            raise ValueError(f"{bad}: need {args.n_folds} comma-separated map files.")
    encoder = cache_encoder(args.features_from)
    for paths in maps.values():
        for path in paths if args.n_folds is not None else [paths]:
            check_map_matches_cache(
                Path(path), encoder, args.pooling, args.features_from
            )

    cv = args.n_folds is not None
    run_dir = new_bench_run_dir(f"scorpion_{args.tag}")
    write_run_info(
        run_dir,
        config={
            "benchmark": "scorpion_map_retrieval_cv"
            if cv
            else "scorpion_map_retrieval",
            **{k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()},
            "protocol": "sipe.eval.proxy.retrieval_metrics, no probes"
            + (", WSI-grouped CV folds (sipe.data.folds)" if cv else ""),
        },
        checkpoint={"features_from": str(args.features_from.resolve()), "maps": maps},
        features={"pooling": args.pooling, "source": "cached SCORPION features"},
    )
    if cv:
        run_cv(args, maps, run_dir, device)
        print(f"run dir: {run_dir}")
        return

    feats, metas = load_cache(args.features_from, (args.split,), args.pooling)
    x, meta = feats[args.split], metas[args.split]
    rows = [{"name": "raw", **retrieval_metrics(x, meta, device), "path": ""}]
    for name, path in maps.items():
        fmap = load_posthoc_map(path).map
        rows.append({
            "name": name,
            **retrieval_metrics(apply_map(fmap, x, device), meta, device),
            "path": path,
        })  # fmt: skip
        logging.info("%s: top-1 %.4f", name, rows[-1]["top_1_accuracy"])
    summary = pd.DataFrame(rows)
    summary.to_csv(
        run_dir / f"summary_{args.split}.csv", index=False, float_format="%.4f"
    )
    print(summary.drop(columns="path").to_string(index=False))
    print(f"run dir: {run_dir}")


if __name__ == "__main__":
    main()
