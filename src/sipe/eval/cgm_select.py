"""Experiment 7 selection driver: CGM fits on the shared SCORPION-proxy protocol.

The CGM training loop validates in-loop (`cross_scanner_top1`, probes on WSI halves
of val), which is not the protocol of the eraser / mixture / quadratic tables. This
driver scores every fit's evaluated map (`runs/<run>/map_cgm.pt`, the `--posthoc`
file) on the clean feature cache with `sipe.eval.proxy`, next to raw GAP, the
reference eraser and G0 fitted in the same run (K = 1, alpha = 1, as the selected
`map_G0_K1.pt` of the mixture sweep). Augmentation only affects fitting; evaluation
always uses clean features.

Selection (val only, never test or PLISM):
- fits with the same configuration (config.yaml without `seed_everything`, logger,
  callbacks and `default_root_dir`) form one candidate; its val top-1 is the mean
  over its seeds;
- candidates are ordered by simplicity: arm V0 < V0-S < V0-S+D < V1 < V1-S < V1-S+D <
  V2 < V2-S < V2-S+D (+D = ContentShiftCGM), then a
  larger residual penalty rho first (closer to the linear limit), then a smaller
  beta;
- the 95% rule of the sweeps (`proxy.select_by_gain`), with G0 as the base and the
  first row: the simplest candidate whose val top-1 gain over G0 reaches
  `--select-gain-frac` of the best gain. If no candidate beats G0, G0 is selected
  (`fallback_to_g0`).

Test is read once after the selection (probes trained on val): `--test selected`
(default) reads raw, reference, G0 and every seed of the selected configuration;
`--test all` also the other fits (ablation rows such as V0-S; the selection is
already fixed); `--test none` skips it.

Outputs (run dir): `summary_val.csv` (probes trained on train), `candidates.csv`,
`selected.json`, `summary_test.csv`.

    uv run python -m sipe.eval.cgm_select --tag cgm_select \\
        --features-from <GAP cache> --runs runs/<fit> runs/<fit> ...
"""

from __future__ import annotations

import argparse
import copy
import json
import logging
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
import yaml

from sipe.bench.posthoc import load_posthoc_map
from sipe.bench.provenance import new_bench_run_dir, write_run_info
from sipe.eval.aug_attribution import g0_map, g0_model
from sipe.eval.content_dependence import paired_views
from sipe.eval.mixture_sweep import DEFAULT_REFERENCE, SETTING
from sipe.eval.proxy import (
    cache_encoder,
    check_map_matches_cache,
    load_cache,
    select_by_gain,
    summarize,
)
from sipe.training.consensus_gen_module import MAP_FILENAME

log = logging.getLogger(__name__)

# "+D": ContentShiftCGM (content-dependent scanner shift in the likelihood), after the
# same arm without it (more fit-time parameters, the same map family).
ARM_ORDER = (
    "V0", "V0-S", "V0-S+D", "V1", "V1-S", "V1-S+D", "V2", "V2-S", "V2-S+D",
)  # fmt: skip
CONTENT_SHIFT_CLASS = "sipe.model.consensus_gen.ContentShiftCGM"
FIXED = ("raw", "reference", "G0")


def fit_info(run_dir: Path) -> dict[str, Any]:
    """Arm, seed and configuration key of a ConsensusGenModule fit."""
    config = yaml.safe_load((run_dir / "config.yaml").read_text())
    model = config["model"]["init_args"]
    net = model["network"]["init_args"]
    arm = "V2" if net.get("generator_hidden_dim", 0) else (
        "V1" if net.get("encoder_hidden_dim", 0) else "V0"
    )  # fmt: skip
    arm += "-S" if net.get("per_scanner", True) else ""
    if model["network"].get("class_path") == CONTENT_SHIFT_CLASS:
        arm += "+D"
    trainer = {
        k: v for k, v in config.get("trainer", {}).items()
        if k not in ("logger", "callbacks", "default_root_dir")
    }  # fmt: skip
    # `hold_out_scanner` (LOSO) was added later, default None: configs written before
    # it lack the key, so drop the default to keep seeds of one configuration together.
    data = copy.deepcopy(config.get("data") or {})
    if (data.get("init_args") or {}).get("hold_out_scanner", "") is None:
        del data["init_args"]["hold_out_scanner"]
    key = json.dumps(
        {"model": model, "data": data, "trainer": trainer},
        sort_keys=True,
        default=str,
    )
    return {
        "arm": arm,
        "seed": config.get("seed_everything"),
        "residual_penalty": float(model.get("residual_penalty", 0.0)),
        "kl_weight": float(model.get("kl_weight", 1.0)),
        "hold_out_scanner": (config.get("data") or {}).get("init_args", {}).get(
            "hold_out_scanner"
        ),
        "config_key": key,
    }


def candidate_table(fits: pd.DataFrame) -> pd.DataFrame:
    """One row per configuration, in simplicity order (mean val top-1 over seeds)."""
    cand = (
        fits.groupby("config_id", sort=False)
        .agg(
            arm=("arm", "first"),
            residual_penalty=("residual_penalty", "first"),
            kl_weight=("kl_weight", "first"),
            top_1_accuracy=("top_1_accuracy", "mean"),
            top_1_std=("top_1_accuracy", "std"),
            seeds=("seed", "count"),
            fits=("name", ",".join),
        )
        .reset_index()
    )
    cand["arm_rank"] = cand["arm"].map(ARM_ORDER.index)
    return cand.sort_values(
        ["arm_rank", "residual_penalty", "kl_weight"], ascending=[True, False, True]
    ).drop(columns="arm_rank")


def select(cand: pd.DataFrame, g0_top1: float, gain_frac: float) -> pd.Series:
    """95% rule over G0, with G0 itself as the first (simplest) row."""
    g0_row = pd.DataFrame([{"config_id": "G0", "top_1_accuracy": g0_top1}])
    rows = pd.concat([g0_row, cand], ignore_index=True)
    return select_by_gain(rows, gain_frac, g0_top1)


def main(argv: list[str] | None = None) -> Path:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--tag", required=True)
    parser.add_argument("--features-from", type=Path, required=True)
    parser.add_argument("--runs", nargs="+", type=Path, required=True,
                        help="ConsensusGenModule fit dirs.")  # fmt: skip
    parser.add_argument("--reference-map", type=Path, default=Path(DEFAULT_REFERENCE),
                        help="'none' to skip.")  # fmt: skip
    parser.add_argument("--ridge", type=float, default=1e-4, help="G0 ridge.")
    parser.add_argument("--select-gain-frac", type=float, default=0.95)
    parser.add_argument("--region-guard", type=float, default=0.02)
    parser.add_argument("--probe-train-max", type=int, default=20000)
    parser.add_argument("--test", default="selected",
                        choices=["selected", "all", "none"])  # fmt: skip
    parser.add_argument("--device", type=int, default=0, help="CUDA index, -1 = CPU.")
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    device = torch.device("cpu" if args.device < 0 else f"cuda:{args.device}")
    reference = None if str(args.reference_map) == "none" else args.reference_map

    fits = []
    for run in args.runs:
        if not (run / MAP_FILENAME).exists():
            raise FileNotFoundError(f"{run / MAP_FILENAME} (not a CGM fit dir?)")
        info = fit_info(run)
        if info["hold_out_scanner"] is not None:
            raise ValueError(f"{run} is a leave-one-scanner-out fit (sipe.eval.loso), "
                             "not a selection candidate.")  # fmt: skip
        fits.append({"name": f"{info['arm']}_{run.name.rsplit('_', 1)[-1]}",
                     "run_dir": str(run), **info})  # fmt: skip
    fits = pd.DataFrame(fits)
    if fits["name"].duplicated().any():
        raise ValueError(f"duplicate fit names: {fits['name'].tolist()}")
    keys = {k: f"c{i}" for i, k in enumerate(dict.fromkeys(fits["config_key"]))}
    fits["config_id"] = fits["config_key"].map(keys)

    # Pooling = the fits' (all the same); every map must match the cache's backbone.
    poolings = {
        load_posthoc_map(Path(r) / MAP_FILENAME).pooling for r in fits["run_dir"]
    }
    if len(poolings) != 1:
        raise ValueError(f"fits with different poolings: {sorted(poolings)}")
    (pooling,) = poolings
    encoder = cache_encoder(args.features_from)
    for path in [Path(r) / MAP_FILENAME for r in fits["run_dir"]] + (
        [reference] if reference is not None else []
    ):
        check_map_matches_cache(path, encoder, pooling, args.features_from)
    splits = ("train", "val") if args.test == "none" else ("train", "val", "test")
    feats, metas = load_cache(args.features_from, splits, pooling)
    run_dir = new_bench_run_dir(f"scorpion_{args.tag}")
    write_run_info(
        run_dir,
        config={
            "benchmark": "scorpion_cgm_select",
            **{k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()},
            "runs": [str(r) for r in args.runs],
            "pairing_unit": "tile-level matched pairs (pair_id), all scanners",
        },
        checkpoint={
            "features_from": str(args.features_from.resolve()),
            "fits": {
                r["name"]: str(Path(r["run_dir"]).resolve()) for _, r in fits.iterrows()
            },  # fmt: skip
            "reference_map": str(reference.resolve()) if reference else None,
        },
        features={
            "pooling": pooling,
            "encoder": encoder,
            "source": "cached SCORPION features",
        },
    )

    train, _ = paired_views(feats["train"], metas["train"], SETTING)
    maps: dict[str, torch.nn.Module | None] = {"raw": None}
    if reference is not None:
        maps["reference"] = load_posthoc_map(reference).map
    maps["G0"] = g0_map(g0_model(train.views, args.ridge), 1.0)
    for _, fit in fits.iterrows():
        maps[fit["name"]] = load_posthoc_map(Path(fit["run_dir"]) / MAP_FILENAME).map

    def logger(name: str, metrics: dict) -> None:
        log.info("val %s: %s", name, metrics)

    summary_val = summarize(maps, feats, metas, "val", device,
                            args.probe_train_max, logger)  # fmt: skip
    meta_cols = ["name", "run_dir", "arm", "seed", "config_id"]
    summary_val = summary_val.merge(fits[meta_cols], on="name", how="left")
    summary_val.to_csv(run_dir / "summary_val.csv", index=False, float_format="%.4f")

    # --- selection on val ------------------------------------------------------------
    top1 = summary_val.set_index("name")["top_1_accuracy"]
    fits["top_1_accuracy"] = fits["name"].map(top1)
    cand = candidate_table(fits)
    cand.drop(columns="fits").to_csv(
        run_dir / "candidates.csv", index=False, float_format="%.4f"
    )
    g0_top1 = float(top1["G0"])
    best = select(cand, g0_top1, args.select_gain_frac)
    chosen = fits.loc[fits["config_id"] == best["config_id"], "name"].tolist()
    region = summary_val.set_index("name")["region_probe_acc"]
    region_drop = float(region["raw"]) - float(
        np.mean([region[n] for n in chosen]) if chosen else region["G0"]
    )
    selected = {
        "config_id": best["config_id"],
        "fits": chosen,
        "arm": None if best["config_id"] == "G0" else best["arm"],
        "fallback_to_g0": best["config_id"] == "G0",
        "rule": f"simplest configuration with mean val top-1 gain over G0 >= "
        f"{args.select_gain_frac:g} x best gain (order: arm, larger rho, smaller beta)",
        "val_top_1": float(best["top_1_accuracy"]),
        "val_top_1_g0": g0_top1,
        "val_top_1_best": float(max(cand["top_1_accuracy"].max(), g0_top1)),
        "val_region_drop": region_drop,
        "region_guard_violated": region_drop > args.region_guard,
        "candidates": int(len(cand)),
    }
    (run_dir / "selected.json").write_text(json.dumps(selected, indent=2) + "\n")
    log.info("selected: %s", selected)

    # --- test, once: proxy protocol (probe trained on val) --------------------------
    if args.test != "none":
        names = [*FIXED, *(fits["name"] if args.test == "all" else chosen)]
        test_maps = {n: maps[n] for n in names if n in maps}

        def test_logger(name: str, metrics: dict) -> None:
            log.info("test %s: %s", name, metrics)

        summary_test = summarize(test_maps, feats, metas, "test", device,
                                 log=test_logger)  # fmt: skip
        summary_test = summary_test.merge(fits[meta_cols], on="name", how="left")
        summary_test["selected"] = summary_test["name"].isin(chosen)
        summary_test.to_csv(
            run_dir / "summary_test.csv", index=False, float_format="%.4f"
        )
    print(summary_val.drop(columns=["run_dir"]).to_string(float_format="%.4f"))
    log.info("CGM selection results: %s", run_dir)
    return run_dir


if __name__ == "__main__":
    main()
