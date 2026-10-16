"""Leave-one-scanner-out (LOSO) analysis of the Experiment 7 maps, SCORPION val or test.

A global map can still behave like a gate when its nonlinear encoder recognizes scanner
signatures (docs/experiment7.md, section 1). On a scanner never seen at fit time,
that turns into extrapolation. For each held-out scanner j, the maps are fitted on the
train split without j: the consensus is then taken over the 4 remaining scanners.
They are scored by cross-scanner retrieval on the `--split` pairs that include j
(`unseen`) and on the pairs that do not (`seen`). Readout: map^-j - G0^-j on `unseen`.
It is an analysis of a frozen model, not a selection step (decision 2026-10-16): nothing
here is selected, so test (`--split test`, read once for the selected arm) gives the
paper numbers; val was the first readout.

Maps per held-out scanner j:
- `G0^-j`: G0 fitted the same way (K = 1, alpha = 1, as in `cgm_select`);
- `CGM-init^-j`: the V0-S init map (closed form, all train tiles without j, as in
  `aug_attribution`);
- trained CGM fits (`--runs`) with `data.init_args.hold_out_scanner = j`
  (`scripts/loso_fits.sh`; `SCORPIONDataModule(hold_out_scanner=...)` drops j from
  train only).
Full-data references fitted in the same run: raw, G0, CGM-init, plus the `--runs` fits
that have no hold-out (e.g. V1-S 7b71, V0-S a602). Each estimator needs at most one fit
per held-out scanner (no seed averaging here).

Contrasts per j and subset: map^-j - G0^-j (`vs_G0`) and map^-j - map^full
(`vs_full`, the cost of not seeing j). Each comes with a slide-bootstrap interval:
retrieval pairs are within a slide, so a replicate draws the slides of the split with
replacement and weights each pair by its slide's count. Val and test have 5 WSIs each,
so these intervals are wide. `scanner = mean` is the mean of the per-j contrasts, with
the same slide draws for every j. Only train and `--split` are read. Pairing unit:
tile-level pairs.

Outputs (run dir, <split> = val or test): `pairs_<split>.csv` (map x slide x scanner
pair), `summary_<split>.csv` (map, held-out scanner, subset), `contrasts_<split>.csv`,
`fits.json`, `map_{G0,CGM-init}_minus<j>.pt` (posthoc module files).

    uv run python -m sipe.eval.loso --tag loso --features-from <GAP cache> \\
        [--split val|test] --runs runs/<V1-S full> runs/<V0-S full> runs/<loso fits> ...
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
from sipe.eval.aug_attribution import (
    CGM_INIT_RIDGE,
    cgm_init_map,
    g0_map,
    g0_model,
    save_map,
)
from sipe.eval.cgm_select import fit_info
from sipe.eval.content_dependence import paired_views
from sipe.eval.mixture_sweep import SETTING
from sipe.eval.proxy import RETRIEVAL_COLS, apply_map, load_cache
from sipe.eval.scorpion_retrieval import retrieval
from sipe.training.consensus_gen_module import MAP_FILENAME

log = logging.getLogger(__name__)

FULL = "none"  # held_out value of the maps fitted on every scanner
PAIR_KEY = ["slide_id", "scanner_a", "scanner_b"]


def drop_scanner(meta: pd.DataFrame, scanner: str) -> pd.DataFrame:
    """Rows of `meta` without `scanner` (the one hold-out rule of the LOSO maps)."""
    if scanner not in set(meta["scanner"]):
        raise ValueError(f"unknown scanner {scanner!r}: {sorted(set(meta['scanner']))}")
    return meta.loc[meta["scanner"] != scanner]


def label_pairs(pairs: pd.DataFrame, scanner: str) -> pd.Series:
    """`unseen` if the scanner pair includes `scanner`, else `seen`."""
    unseen = (pairs["scanner_a"] == scanner) | (pairs["scanner_b"] == scanner)
    return pd.Series(np.where(unseen, "unseen", "seen"), index=pairs.index)


def slide_counts(slides: list[str], n_boot: int, seed: int) -> pd.DataFrame:
    """[n_boot, slides] draw counts of a slide-level bootstrap (shared by all j)."""
    rng = np.random.default_rng(seed)
    n = len(slides)
    counts = np.stack(
        [np.bincount(rng.integers(0, n, n), minlength=n) for _ in range(n_boot)]
    )
    return pd.DataFrame(counts.astype(float), columns=slides)


def replicate_means(diff: pd.Series, slide: pd.Series, counts: pd.DataFrame):
    """Bootstrap replicates of the mean of per-pair values `diff` (NaN = empty draw)."""
    w = counts[slide.to_numpy()].to_numpy()  # [n_boot, pairs]
    totals = w.sum(1)
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.where(totals > 0, (w @ diff.to_numpy()) / totals, np.nan)


def _ci(reps: np.ndarray, level: float = 0.95) -> tuple[float, float]:
    tail = 100 * (1 - level) / 2
    lo, hi = np.nanpercentile(reps, [tail, 100 - tail])
    return float(lo), float(hi)


def contrasts(
    pairs: pd.DataFrame,
    entries: pd.DataFrame,
    scanners: list[str],
    counts: pd.DataFrame,
) -> pd.DataFrame:
    """map^-j - G0^-j and map^-j - map^full per j and subset, plus the mean over j."""
    by = pairs.set_index(["name", *PAIR_KEY])
    out, reps = [], {}
    for _, e in entries.loc[entries["held_out"] != FULL].iterrows():
        j = e["held_out"]
        bases = {}
        if e["estimator"] != "G0":
            bases["vs_G0"] = entries.loc[
                (entries["estimator"] == "G0") & (entries["held_out"] == j), "name"
            ]
        bases["vs_full"] = entries.loc[
            (entries["estimator"] == e["estimator"]) & (entries["held_out"] == FULL),
            "name",
        ]
        for kind, base in bases.items():
            if base.empty:
                continue
            a = by.loc[e["name"]]
            b = by.loc[base.iloc[0]]
            subset = label_pairs(a.reset_index(), j).to_numpy()
            for sub in ("unseen", "seen"):
                rows = subset == sub
                if not rows.any():
                    continue
                for col in RETRIEVAL_COLS:
                    diff = (a[col] - b[col]).loc[rows]
                    slide = pd.Series(diff.index.get_level_values("slide_id"))
                    r = replicate_means(diff.reset_index(drop=True), slide, counts)
                    reps[(e["estimator"], kind, sub, col, j)] = r
                    lo, hi = _ci(r)
                    out.append({
                        "estimator": e["estimator"], "name": e["name"],
                        "base": base.iloc[0], "contrast": kind, "scanner": j,
                        "subset": sub, "metric": col, "n_pairs": int(rows.sum()),
                        "diff": float(diff.mean()), "ci_low": lo, "ci_high": hi,
                    })  # fmt: skip
    # Mean over the held-out scanners (same slide draws for every j).
    df = pd.DataFrame(out)
    if df.empty:
        return df
    for (est, kind, sub, col), g in df.groupby(
        ["estimator", "contrast", "subset", "metric"], sort=False
    ):
        if set(g["scanner"]) != set(scanners):
            continue  # mean only over complete sets of held-out scanners
        r = np.mean([reps[(est, kind, sub, col, j)] for j in scanners], axis=0)
        lo, hi = _ci(r)
        out.append({
            "estimator": est, "name": "mean", "base": "mean", "contrast": kind,
            "scanner": "mean", "subset": sub, "metric": col,
            "n_pairs": int(g["n_pairs"].sum()), "diff": float(g["diff"].mean()),
            "ci_low": lo, "ci_high": hi,
        })  # fmt: skip
    return pd.DataFrame(out)


def main(argv: list[str] | None = None) -> Path:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--tag", required=True)
    parser.add_argument("--features-from", type=Path, required=True)
    parser.add_argument("--runs", nargs="*", type=Path, default=[],
                        help="ConsensusGenModule fit dirs (with or without "
                        "hold_out_scanner).")  # fmt: skip
    parser.add_argument("--scanners", nargs="+", default=None,
                        help="Held-out scanners (default: all).")  # fmt: skip
    parser.add_argument("--ridge", type=float, default=1e-4, help="G0 ridge.")
    parser.add_argument("--init-ridge", type=float, default=CGM_INIT_RIDGE)
    parser.add_argument("--split", choices=("val", "test"), default="val",
                        help="Split scored (fits always use train).")  # fmt: skip
    parser.add_argument("--n-boot", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=0, help="Bootstrap seed.")
    parser.add_argument("--device", type=int, default=0, help="CUDA index, -1 = CPU.")
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    device = torch.device("cpu" if args.device < 0 else f"cuda:{args.device}")

    split = args.split
    feats, metas = load_cache(args.features_from, ("train", split))
    scanners = args.scanners or sorted(metas["train"]["scanner"].unique())

    fits = []
    for run in args.runs:
        if not (run / MAP_FILENAME).exists():
            raise FileNotFoundError(f"{run / MAP_FILENAME} (not a CGM fit dir?)")
        info = fit_info(run)
        held = info["hold_out_scanner"] or FULL
        fits.append({"name": f"{info['arm']}_{run.name.rsplit('_', 1)[-1]}",
                     "estimator": info["arm"], "held_out": held,
                     "run_dir": str(run), "seed": info["seed"]})  # fmt: skip
    fits = pd.DataFrame(fits, columns=["name", "estimator", "held_out", "run_dir",
                                       "seed"])  # fmt: skip
    dup = fits.duplicated(["estimator", "held_out"], keep=False)
    if dup.any():
        raise ValueError(
            "one fit per (arm, held-out scanner): "
            f"{fits.loc[dup, ['name', 'held_out']].values.tolist()}"
        )
    if unknown := set(fits["held_out"]) - {FULL, *scanners}:
        raise ValueError(f"fits hold out scanners outside --scanners: {unknown}")

    run_dir = new_bench_run_dir(f"scorpion_{args.tag}")
    write_run_info(
        run_dir,
        config={
            "benchmark": "scorpion_loso",
            **{k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()},
            "runs": [str(r) for r in args.runs],
            "scanners": scanners,
            "split": split,
            "pairing_unit": "tile-level matched pairs (pair_id); consensus over the"
            " scanners kept at fit time",
        },
        checkpoint={
            "features_from": str(args.features_from.resolve()),
            "fits": {
                r["name"]: str(Path(r["run_dir"]).resolve()) for _, r in fits.iterrows()
            },  # fmt: skip
        },
        features={"pooling": "gap", "source": "cached SCORPION features"},
    )

    # --- closed-form maps: full data, then without each scanner -------------------
    entries = [{"name": "raw", "estimator": "raw", "held_out": FULL}]
    maps: dict[str, torch.nn.Module | None] = {"raw": None}
    for held in [FULL, *scanners]:
        meta = metas["train"] if held == FULL else drop_scanner(metas["train"], held)
        views = paired_views(feats["train"], meta, SETTING)[0].views
        suffix = "" if held == FULL else f"^-{held}"
        stem = "" if held == FULL else f"_minus{held}"
        model = g0_model(views, args.ridge)
        maps[f"G0{suffix}"] = g0_map(model, 1.0)
        maps[f"CGM-init{suffix}"] = cgm_init_map(views, device, args.init_ridge)
        if held != FULL:
            save_map(run_dir / f"map_G0{stem}.pt", maps[f"G0{suffix}"],
                     {"k": 1, "dim": model.mu.shape[1], "gate_dim": 1},
                     variant="G0", K=1, alpha=1.0, hold_out_scanner=held)  # fmt: skip
            fmap = maps[f"CGM-init{suffix}"]
            save_map(run_dir / f"map_CGM-init{stem}.pt", fmap,
                     {"dim": fmap.dim, "encoder_hidden_dim": 0,
                      "generator_hidden_dim": 0},
                     arm="V0-S init", init_ridge=args.init_ridge,
                     hold_out_scanner=held)  # fmt: skip
        for est in ("G0", "CGM-init"):
            entries.append({"name": f"{est}{suffix}", "estimator": est,
                            "held_out": held})  # fmt: skip
        log.info("fitted G0 / CGM-init%s on %d tiles x %d scanners",
                 suffix, *views.shape[:2])  # fmt: skip
    for _, fit in fits.iterrows():
        maps[fit["name"]] = load_posthoc_map(Path(fit["run_dir"]) / MAP_FILENAME).map
        entries.append(fit[["name", "estimator", "held_out"]].to_dict())
    entries = pd.DataFrame(entries)
    (run_dir / "fits.json").write_text(
        json.dumps({"entries": entries.to_dict("records"),
                    "fits": fits.to_dict("records")}, indent=2) + "\n"
    )  # fmt: skip

    # --- retrieval per pair on the split -----------------------------------------
    val_x, val_meta = feats[split], metas[split]
    pairs, summary = [], []
    for _, e in entries.iterrows():
        fmap = maps[e["name"]]
        out = val_x if fmap is None else apply_map(fmap, val_x, device)
        p = retrieval(out, val_meta, device)
        p.insert(0, "name", e["name"])
        pairs.append(p)
        change = np.linalg.norm(out.astype(np.float64) - val_x, axis=1)
        change /= np.linalg.norm(val_x, axis=1)
        for j in scanners if e["held_out"] == FULL else [e["held_out"]]:
            subset = label_pairs(p, j)
            for sub in ("unseen", "seen"):
                rows = p.loc[subset == sub]
                tiles = (val_meta["scanner"] == j) == (sub == "unseen")
                summary.append({
                    "name": e["name"], "estimator": e["estimator"],
                    "held_out": e["held_out"], "scanner": j, "subset": sub,
                    "n_pairs": len(rows),
                    **{c: float(rows[c].mean()) for c in RETRIEVAL_COLS},
                    "relative_change": float(change[tiles.to_numpy()].mean()),
                })  # fmt: skip
        log.info("%s %s: top-1 %.4f", split, e["name"], p["top_1_accuracy"].mean())
    pairs = pd.concat(pairs, ignore_index=True)
    pairs.to_csv(run_dir / f"pairs_{split}.csv", index=False, float_format="%.6g")
    summary = pd.DataFrame(summary)
    summary.to_csv(run_dir / f"summary_{split}.csv", index=False, float_format="%.4f")

    counts = slide_counts(sorted(val_meta["slide_id"].unique()), args.n_boot, args.seed)
    contrast = contrasts(pairs, entries, scanners, counts)
    contrast.to_csv(
        run_dir / f"contrasts_{split}.csv", index=False, float_format="%.5f"
    )
    print(summary.to_string(float_format="%.4f"))
    if len(contrast):
        top1 = contrast.loc[contrast["metric"] == "top_1_accuracy"]
        print(top1.drop(columns="metric").to_string(float_format="%.4f"))
    log.info("LOSO results (%d %s slides; slide-bootstrap intervals are wide): %s",
             counts.shape[1], split, run_dir)  # fmt: skip
    return run_dir


if __name__ == "__main__":
    main()
