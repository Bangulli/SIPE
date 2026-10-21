"""Experiment 3 (roadmap section 5): Gaussian-mixture shrinkage G0 / G1 / G2-scale / G2.

Fits the plug-in mixtures of `sipe.estimators.mixture` on the cached SCORPION train
features (pairing unit: tile-level matched pairs, all 5 scanners) and screens them on
val with the soft-shrinkage sweep's proxy metrics. Test is read once at the end.

Hyperparameters, all chosen on val (never test, never PLISM):
- per K, b (C_k shrinkage toward the pooled within-cluster covariance) by the val
  consensus log-likelihood, shared by G1 / G2-scale / G2;
- per K, g (G2's N_k shrinkage toward N_0) by the val residual log-likelihood;
- per variant and K, the attenuation a of P_k = C_k (C_k + a N_k)^-1 by val top-1
  with the eraser's rule: the smallest a whose gain over a = 0 (identity) reaches
  `--select-gain-frac` of the best gain. a = 1 (plug-in posterior mean) is always in
  the grid and reported.

Outputs (run dir):
- `likelihood.csv`: train and val log-likelihoods (nats/dim) per variant, K, b, g;
- `gating.csv`: single-view gate vs consensus cluster on val (agreement, all views of
  a tile on one cluster, mean max responsibility);
- `sweep_val.csv`: val top-1/top-5/cosine for every variant, K, a (no probes);
- `summary_val.csv`: full proxy metrics (train-trained probes) for raw, the reference
  map (`--reference-map`, default the selected soft-shrinkage eraser) and every
  selected mixture; `selected.json`; `map_<variant>_K<k>.pt` (posthoc module files);
- `summary_test.csv`: the same rows on test, proxy protocol (probes trained on val).

    uv run python -m sipe.eval.mixture_sweep --tag mixture_gap \\
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
from sipe.estimators.mixture import (
    ContentClusters,
    Gate,
    Mixture,
    MixtureShrinkageMap,
    Moments,
    heldout_likelihoods,
)
from sipe.eval.content_dependence import paired_views
from sipe.eval.proxy import (  # noqa: F401 (apply_map, retrieval_metrics re-exported)
    apply_map,
    cache_encoder,
    check_map_matches_cache,
    evaluate,
    retrieval_metrics,
    select_lambda,
    subsample,
)
from sipe.model.encoders import POOLINGS

log = logging.getLogger(__name__)

DEFAULT_ALPHAS = (0.0, 0.1, 0.25, 0.5, 1.0, 2.0, 4.0, 8.0, 16.0, 32.0)
DEFAULT_BETAS = (0.0, 0.1, 0.25, 0.5, 0.75, 1.0)
DEFAULT_GAMMAS = (0.0, 0.1, 0.25, 0.5, 0.75, 0.9)
DEFAULT_REFERENCE = (
    "runs/bench/2026-10-13_17-45-08_lxbelshark_db10_scorpion_softshrink_h0mini_gap_gain95"
    "/eraser_gap_lam0.5.pt"
)
SETTING = "tile_all"


@torch.no_grad()
def gating_stats(
    fmap: MixtureShrinkageMap, views: np.ndarray, labels: np.ndarray, device
) -> dict[str, float]:
    """Single-view gate on paired views [n, S, D] vs the consensus clusters."""
    n, s, d = views.shape
    resp = []
    flat = views.reshape(-1, d).astype(np.float32)
    fmap = fmap.to(device)
    for i in range(0, len(flat), 4096):
        x = torch.from_numpy(flat[i : i + 4096]).to(device)
        resp.append(fmap.responsibilities(x).cpu().numpy())
    resp = np.concatenate(resp).reshape(n, s, -1)
    hard = resp.argmax(axis=-1)
    return {
        "gate_agreement": float((hard == labels[:, None]).mean()),
        "gate_view_consistent": float((hard == hard[:, :1]).all(axis=1).mean()),
        "gate_mean_max_resp": float(resp.max(axis=-1).mean()),
        "gate_resp_true_cluster": float(
            np.take_along_axis(resp, labels[:, None, None].repeat(s, 1), -1).mean()
        ),
    }


def main(argv: list[str] | None = None) -> Path:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--tag", required=True)
    parser.add_argument("--features-from", type=Path, required=True)
    parser.add_argument("--pooling", default="gap", choices=POOLINGS)
    parser.add_argument("--ks", nargs="*", type=int, default=[2, 4],
                        help="Mixtures; `--ks` alone = G0 only.")  # fmt: skip
    parser.add_argument("--content-dim", type=int, default=32)
    parser.add_argument("--gate-dim", type=int, default=32)
    parser.add_argument("--alphas", nargs="+", type=float, default=list(DEFAULT_ALPHAS))
    parser.add_argument("--betas", nargs="+", type=float, default=list(DEFAULT_BETAS))
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
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", type=int, default=0, help="CUDA index, -1 = CPU.")
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    device = torch.device("cpu" if args.device < 0 else f"cuda:{args.device}")
    if args.alphas[0] != 0.0:
        args.alphas = [0.0, *args.alphas]  # a = 0 is the identity (raw)
    if 1.0 not in args.alphas:
        args.alphas = sorted([*args.alphas, 1.0])
    reference = None if str(args.reference_map) == "none" else args.reference_map
    if reference is not None:
        check_map_matches_cache(
            reference, cache_encoder(args.features_from), args.pooling,
            args.features_from,
        )  # fmt: skip

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
            "benchmark": "scorpion_mixture_sweep",
            **{k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()},
            "pairing_unit": "tile-level matched pairs (pair_id), all scanners",
            "estimator": "sipe.estimators.mixture (plug-in consensus moments)",
        },
        checkpoint={
            "features_from": str(args.features_from.resolve()),
            "reference_map": str(reference.resolve()) if reference else None,
        },
        features={"pooling": args.pooling, "source": "cached SCORPION features"},
    )

    train, _ = paired_views(feats["train"], metas["train"], SETTING)
    val, _ = paired_views(feats["val"], metas["val"], SETTING)
    gate = Gate.fit(feats["train"].astype(np.float64), args.gate_dim, args.seed)

    # --- fit: G0, then per K the clusters, b, g, G1 / G2-scale / G2 ---------------
    lik_rows, gate_rows, models = [], [], {}
    g0_moments = Moments.fit(train.views, np.zeros(len(train.views), int), 1)
    models[("G0", 1)] = Mixture.build(g0_moments, "G0", ridge=args.ridge)
    clusters_by_k = {
        1: (np.zeros(len(train.views), int), np.zeros(len(val.views), int))
    }
    for k in args.ks:
        clusters = ContentClusters.fit(train.consensus, k, args.content_dim, args.seed)
        train_lab = clusters.kmeans.labels_
        val_lab = clusters.assign(val.consensus)
        clusters_by_k[k] = (train_lab, val_lab)
        moments = Moments.fit(train.views, train_lab, k)

        def record(model: Mixture, k: int = k) -> dict[str, float]:
            row = {"variant": model.variant, "K": k, "beta": model.beta,
                   "gamma": model.gamma}  # fmt: skip
            for split, data, lab in (
                ("train", train, clusters_by_k[k][0]),
                ("val", val, clusters_by_k[k][1]),
            ):
                lik = heldout_likelihoods(model, data.views, lab, device)
                row.update({f"{split}_{m}": v for m, v in lik.items()})
            lik_rows.append(row)
            log.info("likelihood %s", row)
            return row

        # b on the val consensus likelihood (does not depend on N).
        rows = [
            record(Mixture.build(moments, "G1", beta=b, ridge=args.ridge))
            for b in args.betas
        ]
        beta = max(rows, key=lambda r: r["val_ll_consensus"])["beta"]
        models[("G1", k)] = Mixture.build(moments, "G1", beta=beta, ridge=args.ridge)
        models[("G2-scale", k)] = Mixture.build(
            moments, "G2-scale", beta=beta, ridge=args.ridge
        )
        record(models[("G2-scale", k)])
        # g on the val residual (nuisance) likelihood; g = 1 is G1.
        rows = [
            record(Mixture.build(moments, "G2", beta=beta, gamma=g, ridge=args.ridge))
            for g in args.gammas
        ]
        gamma = max(rows, key=lambda r: r["val_ll_residual"])["gamma"]
        models[("G2", k)] = Mixture.build(
            moments, "G2", beta=beta, gamma=gamma, ridge=args.ridge
        )
    record_g0 = {"variant": "G0", "K": 1, "beta": np.nan, "gamma": np.nan}
    for split, data in (("train", train), ("val", val)):
        lik = heldout_likelihoods(models[("G0", 1)], data.views, None, device)
        record_g0.update({f"{split}_{m}": v for m, v in lik.items()})
    lik_rows.insert(0, record_g0)
    pd.DataFrame(lik_rows).to_csv(run_dir / "likelihood.csv", index=False)

    for (variant, k), model in models.items():
        if k > 1:
            fmap = MixtureShrinkageMap.from_fit(model, gate, 1.0)
            gate_rows.append(
                {"variant": variant, "K": k,
                 **gating_stats(fmap, val.views, clusters_by_k[k][1], device)}
            )  # fmt: skip
    pd.DataFrame(gate_rows).to_csv(run_dir / "gating.csv", index=False)
    log.info("gating:\n%s", pd.DataFrame(gate_rows).to_string())

    # --- val sweep over a (retrieval only) -----------------------------------------
    raw_val = retrieval_metrics(feats["val"], metas["val"], device)
    sweep = []
    for (variant, k), model in models.items():
        for alpha in args.alphas:
            if alpha == 0.0:
                metrics = raw_val
            else:
                fmap = MixtureShrinkageMap.from_fit(model, gate, alpha)
                metrics = retrieval_metrics(
                    apply_map(fmap, feats["val"], device), metas["val"], device
                )
            sweep.append(
                {"variant": variant, "K": k, "beta": model.beta,
                 "gamma": model.gamma, "lambda": alpha, **metrics}
            )  # fmt: skip
            log.info("val %s K=%d a=%g: %s", variant, k, alpha, metrics)
        pd.DataFrame(sweep).rename(columns={"lambda": "alpha"}).to_csv(
            run_dir / "sweep_val.csv", index=False, float_format="%.4f"
        )
    sweep = pd.DataFrame(sweep)

    # --- selection + full val metrics ------------------------------------------------
    probe_rows = subsample(metas["train"], args.probe_train_max, seed=0)
    probe_y = metas["train"]["domain"].to_numpy()[probe_rows]
    candidates: dict[str, torch.nn.Module | None] = {"raw": None}
    if reference is not None:
        candidates["reference"] = load_posthoc_map(reference).map
    selected = {}
    for (variant, k), rows in sweep.groupby(["variant", "K"], sort=False):
        best = select_lambda(rows, args.select_gain_frac)
        alpha = float(best["lambda"])
        name = f"{variant}_K{k}"
        model = models[(variant, k)]
        fmap = MixtureShrinkageMap.from_fit(model, gate, alpha)
        candidates[name] = fmap
        selected[name] = {
            "variant": variant, "K": int(k), "alpha": alpha,
            "beta": model.beta, "gamma": model.gamma,
            "rule": f"smallest alpha with val top-1 gain >= {args.select_gain_frac:g}"
            " x best gain",
            "val_top_1": float(best["top_1_accuracy"]),
            "val_top_1_best": float(rows["top_1_accuracy"].max()),
            "val_top_1_alpha1": float(
                rows.loc[rows["lambda"] == 1.0, "top_1_accuracy"].iloc[0]
            ),
            "at_grid_edge": alpha == max(args.alphas),
        }  # fmt: skip
        torch.save(
            {
                "class_path": f"{MixtureShrinkageMap.__module__}.MixtureShrinkageMap",
                "init_args": {"k": model.k, "dim": model.mu.shape[1],
                              "gate_dim": args.gate_dim},
                "state_dict": fmap.state_dict(),
                "variant": variant, "K": int(k), "alpha": alpha,
                "beta": float(model.beta), "gamma": float(model.gamma),
                "pooling": args.pooling,
                "encoder": cache_encoder(args.features_from),
                "pairing_unit": "tile",
            },
            run_dir / f"map_{variant}_K{k}.pt",
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
    log.info("Mixture sweep results: %s", run_dir)
    return run_dir


if __name__ == "__main__":
    main()
