"""Scanner-count sensitivity: maps fitted on n of the SCORPION scanners, n = 2, 3, 4.

Question: how do G0 and the trained CGM arms (V0-S, V1-S) behave when fewer scanners
are paired at fit time, and does the CGM's gain over G0 hold at small n? An analysis of
fixed configurations, not a selection: nothing is selected here (hyperparameters are
the full-data ones, the eraser keeps lambda = 0.5).

Scanner universe (`--universe`, default every train scanner): the scanners the subsets
are drawn from and the split pairs are scored on. Every subset S of the universe with
|S| in `--sizes` gets, fitted in the run on the train rows of S only:
- `G0[S]` (K = 1, alpha = 1, as in `cgm_select` / `loso`);
- `CGM-init[S]` (the V0-S init, closed form, as in `loso`);
- `eraser[S]` (soft-shrinkage eraser, slide-level deltas over the scanners of S,
  lambda `--lam`; the published baseline with fewer scanners).
`full` = the same maps on every train scanner (reference; with P1000 when the universe
leaves it out). Trained fits (`--runs`): ConsensusGenModule dirs whose
`data.init_args.scanners` (or `hold_out_scanner`) keeps a subset of the universe, or
no restriction (`full`); their arm comes from `cgm_select.fit_info`. They are fitted
with a fixed image budget per step (`scripts/scanner_count_fits.sh`: batch_size =
512 / n locations x n views); nothing about that is checked here.

Readout on `--split` (pairs of universe scanners only), per map and pair subset:
`all` (every universe pair: the same pairs for every S), `seen` (both scanners in S),
`unseen` (at least one not in S). Contrasts: map[S] - G0[S] per subset S, and the mean
over the subsets of each size n (same slide draws for every S), with slide-bootstrap
intervals (`sipe.eval.loso`). Only train and `--split` are read. Pairing unit:
tile-level pairs (G0 / CGM) or slide-level scanner means (eraser), over the scanners
of S.

Outputs: `pairs_<split>.csv`, `summary_<split>.csv` (map x pair subset),
`contrasts_<split>.csv`, `fits.json`, map files `map_<estimator>_<S>.pt`, and
`plism_jobs.tsv` (tag, map file) for `sipe.bench.plism_apply`.

    uv run python -m sipe.eval.scanner_count --tag scanner_count \\
        --features-from <GAP cache> --universe AT2 DP200 GT450 Philips \\
        [--split val|test] --runs runs/<fits> ...
"""

from __future__ import annotations

import argparse
import itertools
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
from sipe.eval.loso import _ci, replicate_means, slide_counts
from sipe.eval.mixture_sweep import SETTING
from sipe.eval.proxy import RETRIEVAL_COLS, apply_map, cache_encoder, load_cache
from sipe.eval.scorpion_retrieval import retrieval
from sipe.eval.soft_shrinkage_sweep import ERASER_CONFIG, fit_family
from sipe.training.consensus_gen_module import MAP_FILENAME

log = logging.getLogger(__name__)

FULL = "full"
CLOSED_FORM = ("G0", "CGM-init", "eraser")
PAIR_KEY = ["slide_id", "scanner_a", "scanner_b"]
SUBSETS = ("all", "seen", "unseen")


def subset_label(scanners) -> str:
    return "-".join(sorted(scanners))


def fit_scanners(info: dict, train_scanners: list[str]) -> list[str]:
    """Train scanners a CGM fit kept (`scanners`, then `hold_out_scanner`)."""
    kept = set(info.get("scanners") or train_scanners)
    if info.get("hold_out_scanner") is not None:
        kept.discard(info["hold_out_scanner"])
    return sorted(kept)


def pair_subset(pairs: pd.DataFrame, kept: list[str]) -> pd.Series:
    """`seen` if both scanners of the pair are in `kept`, else `unseen`."""
    seen = pairs["scanner_a"].isin(kept) & pairs["scanner_b"].isin(kept)
    return pd.Series(np.where(seen, "seen", "unseen"), index=pairs.index)


def keep_rows(feats: np.ndarray, meta: pd.DataFrame, scanners) -> tuple:
    """Rows of `scanners`, features re-indexed (meta `row` -> 0..n-1)."""
    meta = meta.loc[meta["scanner"].isin(list(scanners))]
    return feats[meta["row"].to_numpy()], meta.assign(row=np.arange(len(meta)))


def contrasts(
    pairs: pd.DataFrame, entries: pd.DataFrame, counts: pd.DataFrame
) -> pd.DataFrame:
    """map[S] - G0[S] per subset S and pair subset, plus the mean over S at each n."""
    by = pairs.set_index(["name", *PAIR_KEY])
    out, reps = [], {}
    for _, e in entries.loc[~entries["estimator"].isin(["raw", "G0"])].iterrows():
        base = entries.loc[
            (entries["estimator"] == "G0") & (entries["subset"] == e["subset"]), "name"
        ]
        if base.empty:
            continue
        a, b = by.loc[e["name"]], by.loc[base.iloc[0]]
        kept = e["subset"].split("-") if e["subset"] != FULL else None
        labels = (
            pair_subset(a.reset_index(), kept).to_numpy()
            if kept is not None
            else np.full(len(a), "seen")
        )
        for sub in SUBSETS:
            rows = np.ones(len(a), bool) if sub == "all" else labels == sub
            if not rows.any():
                continue
            for col in RETRIEVAL_COLS:
                diff = (a[col] - b[col]).loc[rows]
                slide = pd.Series(diff.index.get_level_values("slide_id"))
                r = replicate_means(diff.reset_index(drop=True), slide, counts)
                reps[(e["estimator"], e["n"], sub, col, e["subset"])] = r
                lo, hi = _ci(r)
                out.append(
                    {
                        "estimator": e["estimator"],
                        "name": e["name"],
                        "base": base.iloc[0],
                        "n": e["n"],
                        "subset": e["subset"],
                        "pairs": sub,
                        "metric": col,
                        "n_pairs": int(rows.sum()),
                        "diff": float(diff.mean()),
                        "ci_low": lo,
                        "ci_high": hi,
                    }
                )
    df = pd.DataFrame(out)
    if df.empty:
        return df
    for (est, n, sub, col), g in df.groupby(
        ["estimator", "n", "pairs", "metric"], sort=False
    ):
        if g["subset"].iloc[0] == FULL:
            continue
        r = np.mean([reps[(est, n, sub, col, s)] for s in g["subset"]], axis=0)
        lo, hi = _ci(r)
        out.append(
            {
                "estimator": est,
                "name": "mean",
                "base": "mean",
                "n": n,
                "subset": f"mean of {len(g)}",
                "pairs": sub,
                "metric": col,
                "n_pairs": int(g["n_pairs"].sum()),
                "diff": float(g["diff"].mean()),
                "ci_low": lo,
                "ci_high": hi,
            }
        )
    return pd.DataFrame(out)


def plism_tag(estimator: str, subset: str, name: str) -> str:
    est = {"G0": "g0", "CGM-init": "cgminit", "eraser": "eraser"}.get(estimator)
    if est is None:  # trained fit: arm + run id, e.g. v1s_ab12
        est = name.lower().replace("-", "")
    return f"sc_{est}_{subset}"


def main(argv: list[str] | None = None) -> Path:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--tag", required=True)
    parser.add_argument("--features-from", type=Path, required=True)
    parser.add_argument(
        "--universe",
        nargs="+",
        default=None,
        help="Scanners the subsets come from (default: all).",
    )
    parser.add_argument(
        "--sizes",
        nargs="+",
        type=int,
        default=None,
        help="Subset sizes (default: 2 .. |universe|).",
    )
    parser.add_argument(
        "--runs", nargs="*", type=Path, default=[], help="ConsensusGenModule fit dirs."
    )
    parser.add_argument("--lam", type=float, default=0.5, help="Eraser lambda.")
    parser.add_argument("--ridge", type=float, default=1e-4, help="G0 ridge.")
    parser.add_argument("--init-ridge", type=float, default=CGM_INIT_RIDGE)
    parser.add_argument("--split", choices=("val", "test"), default="val")
    parser.add_argument("--n-boot", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=0, help="Bootstrap seed.")
    parser.add_argument("--device", type=int, default=0, help="CUDA index, -1 = CPU.")
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    device = torch.device("cpu" if args.device < 0 else f"cuda:{args.device}")

    split = args.split
    feats, metas = load_cache(args.features_from, ("train", split))
    encoder = cache_encoder(args.features_from)
    train_scanners = sorted(metas["train"]["scanner"].unique())
    universe = sorted(args.universe or train_scanners)
    if unknown := set(universe) - set(train_scanners):
        raise ValueError(f"--universe: unknown scanners {sorted(unknown)}")
    sizes = sorted(args.sizes or range(2, len(universe) + 1))
    if not all(2 <= k <= len(universe) for k in sizes):
        raise ValueError(f"--sizes must be in 2..{len(universe)}: {sizes}")
    subsets = [c for k in sizes for c in itertools.combinations(universe, k)]

    fits = []
    for run in args.runs:
        if not (run / MAP_FILENAME).exists():
            raise FileNotFoundError(f"{run / MAP_FILENAME} (not a CGM fit dir?)")
        info = fit_info(run)
        kept = fit_scanners(info, train_scanners)
        if kept == train_scanners:
            subset, n = FULL, len(kept)
        elif set(kept) <= set(universe) and tuple(kept) in subsets:
            subset, n = subset_label(kept), len(kept)
        else:
            raise ValueError(f"{run} keeps {kept}: not a --universe subset of --sizes")
        fits.append(
            {
                "name": f"{info['arm']}_{run.name.rsplit('_', 1)[-1]}",
                "estimator": info["arm"],
                "subset": subset,
                "n": n,
                "run_dir": str(run),
                "seed": info["seed"],
            }
        )
    fits = pd.DataFrame(
        fits, columns=["name", "estimator", "subset", "n", "run_dir", "seed"]
    )
    dup = fits.duplicated(["estimator", "subset"], keep=False)
    if dup.any():
        raise ValueError(f"one fit per (arm, subset): {fits.loc[dup, 'name'].tolist()}")

    run_dir = new_bench_run_dir(f"scorpion_{args.tag}")
    write_run_info(
        run_dir,
        config={
            "benchmark": "scorpion_scanner_count",
            **{k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()},
            "runs": [str(r) for r in args.runs],
            "universe": universe,
            "sizes": sizes,
            "split": split,
            "eraser_config": ERASER_CONFIG,
            "pairing_unit": "tile-level pairs (G0, CGM) / slide-level scanner means"
            " (eraser) over the scanners of each subset",
        },
        checkpoint={
            "features_from": str(args.features_from.resolve()),
            "encoder": encoder,
            "fits": {
                r["name"]: str(Path(r["run_dir"]).resolve()) for _, r in fits.iterrows()
            },
        },
        features={"pooling": "gap", "source": "cached SCORPION features"},
    )

    # --- closed-form maps per subset ----------------------------------------------
    entries = [
        {"name": "raw", "estimator": "raw", "subset": FULL, "n": len(train_scanners)}
    ]
    maps: dict[str, torch.nn.Module | None] = {"raw": None}
    jobs = []
    for kept in [tuple(train_scanners)] + [
        s for s in subsets if list(s) != train_scanners
    ]:
        subset = FULL if list(kept) == train_scanners else subset_label(kept)
        x, meta = keep_rows(feats["train"], metas["train"], kept)
        views = paired_views(x, meta, SETTING)[0].views
        model = g0_model(views, args.ridge)
        fmaps = {
            "G0": g0_map(model, 1.0),
            "CGM-init": cgm_init_map(views, device, args.init_ridge),
        }
        eraser = fit_family(x, meta, device, deltas="slide").make_eraser(args.lam)
        for est, fmap in fmaps.items():
            path = run_dir / f"map_{est}_{subset}.pt"
            init_args = (
                {"k": 1, "dim": model.mu.shape[1], "gate_dim": 1}
                if est == "G0"
                else {
                    "dim": fmap.dim,
                    "encoder_hidden_dim": 0,
                    "generator_hidden_dim": 0,
                }
            )
            save_map(
                path,
                fmap,
                init_args,
                encoder=encoder,
                scanners=subset,
                **(
                    {"variant": "G0", "K": 1, "alpha": 1.0}
                    if est == "G0"
                    else {"arm": "V0-S init", "init_ridge": args.init_ridge}
                ),
            )
        path = run_dir / f"map_eraser_{subset}.pt"
        torch.save(
            {
                "P": eraser.P.cpu(),
                "bias": eraser.bias.cpu(),
                "lambda": args.lam,
                "pooling": "gap",
                "deltas": "slide",
                "encoder": encoder,
                "scanners": subset,
                "eraser_config": ERASER_CONFIG,
            },
            path,
        )
        fmaps["eraser"] = load_posthoc_map(path).map
        for est in CLOSED_FORM:
            name = f"{est}[{subset}]"
            maps[name] = fmaps[est]
            entries.append(
                {"name": name, "estimator": est, "subset": subset, "n": len(kept)}
            )
            if subset != FULL:
                jobs.append(
                    {
                        "tag": plism_tag(est, subset, name),
                        "name": name,
                        "map": str(run_dir / f"map_{est}_{subset}.pt"),
                    }
                )
        log.info("fitted %s on %d tiles x %d scanners", subset, *views.shape[:2])
    for _, fit in fits.iterrows():
        maps[fit["name"]] = load_posthoc_map(Path(fit["run_dir"]) / MAP_FILENAME).map
        entries.append(fit[["name", "estimator", "subset", "n"]].to_dict())
        if fit["subset"] != FULL:
            jobs.append(
                {
                    "tag": plism_tag(fit["estimator"], fit["subset"], fit["name"]),
                    "name": fit["name"],
                    "map": str(Path(fit["run_dir"]).resolve() / MAP_FILENAME),
                }
            )
    entries = pd.DataFrame(entries)
    (run_dir / "fits.json").write_text(
        json.dumps(
            {"entries": entries.to_dict("records"), "fits": fits.to_dict("records")},
            indent=2,
        )
        + "\n"
    )
    pd.DataFrame(jobs).to_csv(run_dir / "plism_jobs.tsv", sep="\t", index=False)

    # --- retrieval per pair on the split (universe scanners only) ------------------
    split_meta = metas[split].loc[metas[split]["scanner"].isin(universe)]
    pairs, summary = [], []
    for _, e in entries.iterrows():
        fmap = maps[e["name"]]
        out = feats[split] if fmap is None else apply_map(fmap, feats[split], device)
        p = retrieval(out, split_meta, device)
        p.insert(0, "name", e["name"])
        pairs.append(p)
        kept = universe if e["subset"] == FULL else e["subset"].split("-")
        labels = pair_subset(p, kept)
        for sub in SUBSETS:
            rows = p if sub == "all" else p.loc[labels == sub]
            if rows.empty:
                continue
            summary.append(
                {
                    "name": e["name"],
                    "estimator": e["estimator"],
                    "subset": e["subset"],
                    "n": e["n"],
                    "pairs": sub,
                    "n_pairs": len(rows),
                    **{c: float(rows[c].mean()) for c in RETRIEVAL_COLS},
                }
            )
        log.info("%s %s: top-1 %.4f", split, e["name"], p["top_1_accuracy"].mean())
    pairs = pd.concat(pairs, ignore_index=True)
    pairs.to_csv(run_dir / f"pairs_{split}.csv", index=False, float_format="%.6g")
    summary = pd.DataFrame(summary)
    summary.to_csv(run_dir / f"summary_{split}.csv", index=False, float_format="%.4f")

    counts = slide_counts(
        sorted(split_meta["slide_id"].unique()), args.n_boot, args.seed
    )
    contrast = contrasts(pairs, entries, counts)
    contrast.to_csv(
        run_dir / f"contrasts_{split}.csv", index=False, float_format="%.5f"
    )
    top1 = summary.loc[summary["pairs"] == "all"]
    print(
        top1.groupby(["estimator", "n"])["top_1_accuracy"].agg(
            ["mean", "min", "max", "count"]
        )
    )
    log.info(
        "scanner-count results (%d %s slides): %s", counts.shape[1], split, run_dir
    )
    return run_dir


if __name__ == "__main__":
    main()
