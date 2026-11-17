"""Grouped k-fold CV selection of the closed-form maps, then a refit on all of SCORPION.

MIDL protocol (replaces the single train/val/test split for selection): the cached
SCORPION features of every WSI (train + val + test of the cache, 48 WSIs) are dealt
into `--n-folds` WSI-grouped folds (`sipe.data.folds`, the same folds as
`SCORPIONDataModule(n_folds=, fold=)`). For each fold, every method is fit on the
other folds and each grid value is scored on the held-out fold by cross-scanner
retrieval (`sipe.eval.proxy.retrieval_metrics`, no probes). The grid value is chosen
on the mean over folds with the usual rule (`sipe.eval.proxy.select_lambda`: the
smallest value whose mean top-1 gain over the identity reaches `--select-gain-frac`
of the best gain), then refit on every WSI and saved as a `--posthoc` map file for
PLISM / HEST. There is no SCORPION test: the CV mean is the selection value, so it
is optimistic for the selected row.

Methods (fits unchanged from the single-split sweeps):
- `eraser_<slide|region|tile>`: soft-shrinkage eraser of
  `sipe.eval.soft_shrinkage_sweep.fit_family`, pairing unit of the deltas in the name;
  grid `--lambdas`;
- `g0`: G0 consensus shrinkage of `sipe.estimators.mixture` (tile pairs, all
  scanners); grid `--alphas` (the attenuation a).

Outputs (run dir): `cv_folds.csv` (method, value, fold, retrieval columns),
`cv_mean.csv` (mean and std over folds), `selected.json`, `map_<method>.pt`.

    uv run python -m sipe.eval.cv_sweep --tag cv4_baselines \\
        --features-from runs/bench/<feature cache run>   # CLAUDE.md, "Cached features"
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from sipe.bench.posthoc import AffineMap
from sipe.bench.provenance import new_bench_run_dir, write_run_info
from sipe.data.folds import REFIT
from sipe.estimators.mixture import Gate, Mixture, MixtureShrinkageMap, Moments
from sipe.eval.content_dependence import paired_views
from sipe.eval.mixture_sweep import DEFAULT_ALPHAS
from sipe.eval.proxy import (
    RETRIEVAL_COLS,
    apply_map,
    cache_encoder,
    load_cv_cache,
    retrieval_metrics,
    select_lambda,
)
from sipe.eval.soft_shrinkage_sweep import DEFAULT_LAMBDAS, ERASER_CONFIG, fit_family

log = logging.getLogger(__name__)

METHODS = ("eraser_slide", "eraser_region", "eraser_tile", "g0")
G0_SETTING = "tile_all"


class Fitted:
    """One method fit on one training set: `map(value)` -> nn.Module."""

    def __init__(self, method: str, x: np.ndarray, meta: pd.DataFrame, args, device):
        self.method = method
        if method.startswith("eraser_"):
            deltas = method.removeprefix("eraser_")
            self.family = fit_family(x, meta, device, deltas=deltas)
        elif method == "g0":
            train, _ = paired_views(x, meta, G0_SETTING)
            moments = Moments.fit(train.views, np.zeros(len(train.views), int), 1)
            self.model = Mixture.build(moments, "G0", ridge=args.ridge)
            self.gate = Gate.fit(x.astype(np.float64), args.gate_dim, args.seed)
        else:
            raise ValueError(f"unknown method {method!r}; known: {METHODS}")

    def map(self, value: float) -> torch.nn.Module:
        if self.method == "g0":
            return MixtureShrinkageMap.from_fit(self.model, self.gate, value)
        # The affine form of the map file (`--posthoc`), so CV scores what is saved.
        eraser = self.family.make_eraser(value)
        return AffineMap(eraser.P.cpu(), eraser.bias.cpu())

    def save(self, value: float, path: Path, args) -> None:
        fmap = self.map(value)
        common = {"pooling": args.pooling, "encoder": cache_encoder(args.features_from)}
        if self.method == "g0":
            payload = {
                "class_path": f"{MixtureShrinkageMap.__module__}.MixtureShrinkageMap",
                "init_args": {"k": self.model.k, "dim": self.model.mu.shape[1],
                              "gate_dim": args.gate_dim},
                "state_dict": fmap.state_dict(),
                "variant": "G0", "K": 1, "alpha": value,
                "beta": float(self.model.beta), "gamma": float(self.model.gamma),
                "pairing_unit": "tile",
            }  # fmt: skip
        else:
            payload = {
                "P": fmap.P.cpu(), "bias": fmap.bias.cpu(), "lambda": value,
                "deltas": self.method.removeprefix("eraser_"),
                "eraser_config": ERASER_CONFIG,
            }  # fmt: skip
        torch.save({**payload, **common}, path)


def grid(method: str, args) -> list[float]:
    values = args.alphas if method == "g0" else args.lambdas
    return sorted({0.0, *values})  # 0 = identity (raw), the base of the gain rule


def cv_mean(folds: pd.DataFrame, keys: list[str]) -> pd.DataFrame:
    """Mean and std (ddof 1) over folds of every retrieval column, per keys."""
    stats = folds.groupby(keys, sort=False)[list(RETRIEVAL_COLS)].agg(["mean", "std"])
    stats.columns = [f"{c}_{s}" for c, s in stats.columns]
    n = folds.groupby(keys, sort=False)["fold"].nunique().rename("n_folds")
    return stats.join(n).reset_index()


def main(argv: list[str] | None = None) -> Path:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--tag", required=True)
    parser.add_argument("--features-from", type=Path, required=True)
    parser.add_argument("--pooling", default="gap")
    parser.add_argument("--n-folds", type=int, default=4)
    parser.add_argument("--fold-seed", type=int, default=42)
    parser.add_argument("--methods", nargs="+", default=list(METHODS), choices=METHODS)
    parser.add_argument(
        "--lambdas", nargs="+", type=float, default=list(DEFAULT_LAMBDAS)
    )
    parser.add_argument("--alphas", nargs="+", type=float, default=list(DEFAULT_ALPHAS))
    parser.add_argument("--ridge", type=float, default=1e-4, help="G0 (mixture_sweep).")
    parser.add_argument("--gate-dim", type=int, default=32, help="G0 map file only.")
    parser.add_argument("--select-gain-frac", type=float, default=0.95)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", type=int, default=0, help="CUDA index, -1 = CPU.")
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    device = torch.device("cpu" if args.device < 0 else f"cuda:{args.device}")

    run_dir = new_bench_run_dir(f"scorpion_{args.tag}")
    write_run_info(
        run_dir,
        config={
            "benchmark": "scorpion_cv_sweep",
            **{k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()},
            "protocol": "WSI-grouped k-fold CV on every cached WSI (sipe.data.folds);"
            " select on the fold mean of val top-1; refit on every WSI",
            "eraser": ERASER_CONFIG,
        },
        checkpoint={"features_from": str(args.features_from.resolve())},
        features={"pooling": args.pooling, "source": "cached SCORPION features"},
    )

    rows = []
    for fold in range(args.n_folds):
        x, meta = load_cv_cache(
            args.features_from, args.n_folds, fold, args.pooling, args.fold_seed
        )
        base = {"fold": fold, "n_train_wsi": meta["train"]["slide_id"].nunique(),
                "n_val_wsi": meta["val"]["slide_id"].nunique()}  # fmt: skip
        raw = retrieval_metrics(x["val"], meta["val"], device)
        for method in args.methods:
            fitted = Fitted(method, x["train"], meta["train"], args, device)
            for value in grid(method, args):
                metrics = raw if value == 0.0 else retrieval_metrics(
                    apply_map(fitted.map(value), x["val"], device), meta["val"], device
                )  # fmt: skip
                rows.append({"method": method, "value": value, **base, **metrics})
                log.info("fold %d %s %g: top-1 %.4f", fold, method, value,
                         metrics["top_1_accuracy"])  # fmt: skip
        folds = pd.DataFrame(rows)
        folds.to_csv(run_dir / "cv_folds.csv", index=False, float_format="%.4f")

    mean = cv_mean(folds, ["method", "value"])
    mean.to_csv(run_dir / "cv_mean.csv", index=False, float_format="%.4f")

    x, meta = load_cv_cache(
        args.features_from, args.n_folds, REFIT, args.pooling, args.fold_seed
    )
    selected = {}
    for method, m in mean.groupby("method", sort=False):
        rule_rows = m.rename(
            columns={"value": "lambda", "top_1_accuracy_mean": "top_1_accuracy"}
        )
        best = select_lambda(rule_rows, args.select_gain_frac)
        value = float(best["lambda"])
        per_fold = folds.loc[(folds["method"] == method) & (folds["value"] == value)]
        path = run_dir / f"map_{method}.pt"
        Fitted(method, x["train"], meta["train"], args, device).save(value, path, args)
        selected[method] = {
            "value": value,
            "param": "alpha" if method == "g0" else "lambda",
            "rule": f"smallest value with mean CV top-1 gain >= "
            f"{args.select_gain_frac:g} x best gain",
            "cv_top_1_mean": float(best["top_1_accuracy"]),
            "cv_top_1_std": float(best["top_1_accuracy_std"]),
            "cv_top_1_folds": per_fold.sort_values("fold")["top_1_accuracy"].tolist(),
            "cv_top_1_best": float(m["top_1_accuracy_mean"].max()),
            "cv_top_1_raw": float(
                m.loc[m["value"] == 0.0, "top_1_accuracy_mean"].iloc[0]
            ),
            "at_grid_edge": value == max(grid(method, args)),
            "refit_wsi": int(meta["train"]["slide_id"].nunique()),
            "map_file": path.name,
        }  # fmt: skip
    (run_dir / "selected.json").write_text(json.dumps(selected, indent=2) + "\n")
    print(mean.to_string(index=False, float_format="%.4f"))
    print(json.dumps(selected, indent=2))
    log.info("CV sweep results: %s", run_dir)
    return run_dir


if __name__ == "__main__":
    main()
