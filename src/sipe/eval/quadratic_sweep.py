"""Experiment 4 (roadmap section 6): quadratic correction Q1 / Q2 / Q2-rand on G0.

Fits `sipe.estimators.quadratic` on the cached SCORPION train features (pairing unit:
tile-level matched pairs, all 5 scanners; target = tile consensus, base = G0 of
Experiment 3) and screens it on val with the soft-shrinkage sweep's proxy metrics.
Test is read once at the end.

Hyperparameters, all chosen on val (never test, never PLISM):
- per variant and m (directions per basis), the nonlinear ridge g by val top-1 with
  the eraser's rule applied to the strength 1/g: the largest g whose val top-1 gain
  over G0 (g = inf) reaches `--select-gain-frac` of the best gain. If no g beats G0,
  g = inf and the candidate is G0 (flagged `fallback_to_g0`).

Outputs (run dir):
- `sweep_val.csv`: per variant, m, g: train/val consensus MSE (objective, per dim)
  and val top-1/top-5/cosine (no probes); g = inf rows are G0;
- `summary_val.csv`: full proxy metrics (train-trained probes) for raw, the reference
  map (`--reference-map`, default the selected soft-shrinkage eraser), G0 (fitted in
  this run) and every selected candidate; `selected.json`;
  `map_<variant>_m<m>.pt` (posthoc module files);
- `summary_test.csv`: the same rows on test, proxy protocol (probes trained on val).

    uv run python -m sipe.eval.quadratic_sweep --tag quadratic_gap \\
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

from sipe.bench.posthoc import load_posthoc_map
from sipe.bench.provenance import new_bench_run_dir, write_run_info
from sipe.estimators.mixture import Gate, Mixture, MixtureShrinkageMap, Moments
from sipe.estimators.quadratic import (
    VARIANTS,
    QuadraticFeatures,
    QuadraticFit,
    QuadraticShrinkageMap,
)
from sipe.eval.content_dependence import paired_views
from sipe.eval.mixture_sweep import (
    DEFAULT_REFERENCE,
    SETTING,
    apply_map,
    retrieval_metrics,
)
from sipe.eval.soft_shrinkage_sweep import evaluate, select_lambda, subsample

log = logging.getLogger(__name__)

DEFAULT_GAMMAS = (0.001, 0.01, 0.1, 1.0, 10.0, 100.0)


def consensus_mse(fmap: torch.nn.Module, views: np.ndarray, device) -> float:
    """Mean ||f(x) - c||^2 / D over the paired views [n, S, D]."""
    n, s, d = views.shape
    out = apply_map(fmap, views.reshape(-1, d).astype(np.float32), device)
    target = np.repeat(views.mean(axis=1), s, axis=0)
    return float(((out - target) ** 2).mean())


def main(argv: list[str] | None = None) -> Path:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--tag", required=True)
    parser.add_argument("--features-from", type=Path, required=True)
    parser.add_argument("--pooling", default="gap", choices=["gap", "cls"])
    parser.add_argument("--variants", nargs="+", default=list(VARIANTS),
                        choices=VARIANTS)  # fmt: skip
    parser.add_argument("--ms", nargs="+", type=int, default=[8, 16])
    parser.add_argument("--gammas", nargs="+", type=float, default=list(DEFAULT_GAMMAS))
    parser.add_argument("--ridge", type=float, default=1e-4)
    parser.add_argument("--select-gain-frac", type=float, default=0.95)
    parser.add_argument("--region-guard", type=float, default=0.02)
    parser.add_argument("--probe-train-max", type=int, default=20000)
    parser.add_argument(
        "--reference-map",
        type=Path,
        default=Path(DEFAULT_REFERENCE),
        help="Affine/posthoc map evaluated alongside ('none' to skip).",
    )
    parser.add_argument("--skip-test", action="store_true")
    parser.add_argument("--seed", type=int, default=0, help="Q2-rand directions.")
    parser.add_argument("--device", type=int, default=0, help="CUDA index, -1 = CPU.")
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    device = torch.device("cpu" if args.device < 0 else f"cuda:{args.device}")
    gammas = sorted(g for g in args.gammas if np.isfinite(g))
    reference = None if str(args.reference_map) == "none" else args.reference_map

    splits = ("train", "val") if args.skip_test else ("train", "val", "test")
    feats, metas = {}, {}
    for split in splits:
        feats[split] = np.load(args.features_from / f"features_{split}.npz")[
            args.pooling
        ].astype(np.float32)
        metas[split] = pd.read_csv(args.features_from / f"meta_{split}.csv")

    run_dir = new_bench_run_dir(f"scorpion_{args.tag}")
    write_run_info(
        run_dir,
        config={
            "benchmark": "scorpion_quadratic_sweep",
            **{k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()},
            "pairing_unit": "tile-level matched pairs (pair_id), all scanners",
            "estimator": "sipe.estimators.quadratic (consensus ridge regression,"
            " linear limit = G0)",
        },
        checkpoint={
            "features_from": str(args.features_from.resolve()),
            "reference_map": str(reference.resolve()) if reference else None,
        },
        features={"pooling": args.pooling, "source": "cached SCORPION features"},
    )

    train, _ = paired_views(feats["train"], metas["train"], SETTING)
    val, _ = paired_views(feats["val"], metas["val"], SETTING)
    g0 = Mixture.build(
        Moments.fit(train.views, np.zeros(len(train.views), int), 1), "G0",
        ridge=args.ridge,
    )  # fmt: skip
    mu, C, N = g0.mu[0], g0.C[0], g0.N[0]
    # K = 1: the gate is irrelevant (responsibility 1).
    g0_map = MixtureShrinkageMap.from_fit(
        g0, Gate(np.zeros(len(mu)), np.eye(1, len(mu))), 1.0
    )

    # --- val sweep over g (retrieval + consensus MSE) --------------------------------
    g0_val = retrieval_metrics(apply_map(g0_map, feats["val"], device), metas["val"],
                               device)  # fmt: skip
    g0_mse = {"train_mse": consensus_mse(g0_map, train.views, device),
              "val_mse": consensus_mse(g0_map, val.views, device)}  # fmt: skip
    fits, sweep = {}, []
    for variant in args.variants:
        for m in args.ms:
            feat = QuadraticFeatures.build(variant, C, N, m, seed=args.seed)
            fit = QuadraticFit.fit(train.views, feat, mu, C, N)
            fits[(variant, m)] = fit
            base = {"variant": variant, "m": m, "n_features": feat.n}
            sweep.append({**base, "gamma": np.inf, "lambda": 0.0, **g0_mse, **g0_val})
            for gamma in gammas:
                fmap = QuadraticShrinkageMap.from_fit(fit, gamma)
                metrics = retrieval_metrics(
                    apply_map(fmap, feats["val"], device), metas["val"], device
                )
                sweep.append(
                    {**base, "gamma": gamma, "lambda": 1.0 / gamma,
                     "train_mse": consensus_mse(fmap, train.views, device),
                     "val_mse": consensus_mse(fmap, val.views, device), **metrics}
                )  # fmt: skip
                log.info("val %s m=%d g=%g: %s", variant, m, gamma, sweep[-1])
            pd.DataFrame(sweep).drop(columns="lambda").to_csv(
                run_dir / "sweep_val.csv", index=False, float_format="%.6g"
            )
    sweep = pd.DataFrame(sweep)

    # --- selection + full val metrics ------------------------------------------------
    probe_rows = subsample(metas["train"], args.probe_train_max, seed=0)
    probe_y = metas["train"]["domain"].to_numpy()[probe_rows]
    candidates: dict[str, torch.nn.Module | None] = {"raw": None}
    if reference is not None:
        candidates["reference"] = load_posthoc_map(reference).map
    candidates["G0"] = g0_map
    selected = {}
    for (variant, m), rows in sweep.groupby(["variant", "m"], sort=False):
        # Strength 1/g: lambda = 0 is G0, "smallest lambda" = largest g.
        best = select_lambda(rows, args.select_gain_frac)
        gamma = float(best["gamma"])
        name = f"{variant}_m{m}"
        fit = fits[(variant, m)]
        fmap = QuadraticShrinkageMap.from_fit(fit, gamma)
        candidates[name] = fmap
        selected[name] = {
            "variant": variant, "m": int(m), "n_features": fit.features.n,
            "gamma": gamma, "fallback_to_g0": bool(np.isinf(gamma)),
            "rule": f"largest gamma with val top-1 gain over G0 >= "
            f"{args.select_gain_frac:g} x best gain",
            "val_top_1": float(best["top_1_accuracy"]),
            "val_top_1_g0": float(g0_val["top_1_accuracy"]),
            "val_top_1_best": float(rows["top_1_accuracy"].max()),
            "val_mse": float(best["val_mse"]), "val_mse_g0": g0_mse["val_mse"],
            "train_mse": float(best["train_mse"]), "train_mse_g0": g0_mse["train_mse"],
            "at_grid_edge": gamma == min(gammas),
        }  # fmt: skip
        n_content = 0 if fit.features.content is None else len(fit.features.content)
        torch.save(
            {
                "class_path": f"{QuadraticShrinkageMap.__module__}"
                ".QuadraticShrinkageMap",
                "init_args": {"dim": len(mu), "n_nuisance": int(m),
                              "n_content": n_content},
                "state_dict": fmap.state_dict(),
                "variant": variant, "m": int(m), "gamma": gamma,
                "ridge": args.ridge, "pooling": args.pooling, "encoder": "h0-mini",
                "pairing_unit": "tile",
            },
            run_dir / f"map_{variant}_m{m}.pt",
        )  # fmt: skip

    def transform(fmap):
        return (lambda x: x) if fmap is None else (lambda x: apply_map(fmap, x, device))

    summary_val = []
    for name, fmap in candidates.items():
        tf = transform(fmap)
        metrics = evaluate(
            tf(feats["val"]), metas["val"], tf(feats["train"][probe_rows]), probe_y,
            feats["val"], device,
        )  # fmt: skip
        summary_val.append({"name": name, **metrics})
        log.info("val full %s: %s", name, metrics)
    summary_val = pd.DataFrame(summary_val)
    summary_val.to_csv(run_dir / "summary_val.csv", index=False, float_format="%.4f")
    raw_region = float(summary_val.iloc[0]["region_probe_acc"])
    for name, sel in selected.items():
        row = summary_val.loc[summary_val["name"] == name].iloc[0]
        sel["val_region_drop"] = raw_region - float(row["region_probe_acc"])
        sel["region_guard_violated"] = sel["val_region_drop"] > args.region_guard
    (run_dir / "selected.json").write_text(json.dumps(selected, indent=2) + "\n")

    # --- test, once: proxy protocol (probe trained on val) --------------------------
    if not args.skip_test:
        summary_test = []
        val_y = metas["val"]["domain"].to_numpy()
        for name, fmap in candidates.items():
            tf = transform(fmap)
            metrics = evaluate(
                tf(feats["test"]), metas["test"], tf(feats["val"]), val_y,
                feats["test"], device,
            )  # fmt: skip
            summary_test.append({"name": name, **metrics})
            log.info("test %s: %s", name, metrics)
        pd.DataFrame(summary_test).to_csv(
            run_dir / "summary_test.csv", index=False, float_format="%.4f"
        )
    print(summary_val.to_string(float_format="%.4f"))
    log.info("Quadratic sweep results: %s", run_dir)
    return run_dir


if __name__ == "__main__":
    main()
