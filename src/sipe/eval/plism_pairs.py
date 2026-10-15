"""PLISM pair table: merge the official per-pair metrics of several PLISM runs.

Every `sipe.bench.plism` run writes plismbench's per-pair metrics to
`metrics/<n_tiles>_tiles/<tag>/metrics.csv` (4,095 slide pairs, cosine and top-k
accuracies averaged over the two retrieval directions). This module reads them (nothing
is recomputed) and writes one row per pair with every run side by side, to pick slide
pairs for `sipe.bench.plism_retrieval_viz`:

    uv run python -m sipe.eval.plism_pairs \\
        --runs runs/bench/<gap ref run> runs/bench/<method run> ... --tag <tag>

The first run is the reference. Outputs in
`runs/bench/<stamp>_<host>_<rand>_plism_pairs_<tag>/`:
- `pairs.csv`: slide/stain/scanner of a and b, `shift` (plismbench `get_results` groups:
  `inter-scanner` = same stain, `inter-staining` = same scanner, `joint` = both differ),
  `hr` (a or b has the HR stain), then `<label>__{cosine,top_1,top_10}` per run and
  `<label>__delta_top_1` (method minus reference) per method. Label = the run's tag.
- `shortlist.csv`: per shift type, the `--k` pairs with the lowest / highest reference
  top-1 (`worst_ref` / `best_ref`) and, per method, the largest gain / loss in top-1
  (`gain_<label>` / `loss_<label>`); same columns as `pairs.csv` plus `reason`.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from sipe.bench.provenance import new_bench_run_dir, write_run_info

SHIFTS = ("inter-scanner", "inter-staining", "joint")
METRICS = {
    "cosine_similarity": "cosine",
    "top_1_accuracy": "top_1",
    "top_10_accuracy": "top_10",
}
KEYS = ["slide_a", "slide_b"]


def load_run(run_dir: Path, n_tiles: int) -> tuple[str, pd.DataFrame, Path]:
    """(tag, per-pair metrics, metrics.csv path) of one sipe.bench.plism run."""
    tag = yaml.safe_load((run_dir / "config.yaml").read_text())["tag"]
    path = run_dir / "metrics" / f"{n_tiles}_tiles" / tag / "metrics.csv"
    if not path.exists():
        raise FileNotFoundError(
            f"{path} (not a finished PLISM run with {n_tiles} tiles?)"
        )
    return tag, pd.read_csv(path), path


def shift_type(df: pd.DataFrame) -> pd.Series:
    same_stain = df["staining_a"] == df["staining_b"]
    same_scanner = df["scanner_a"] == df["scanner_b"]
    if (same_stain & same_scanner).any():
        raise ValueError("pair with the same stain and scanner")
    return pd.Series(
        [
            "inter-scanner" if s else "inter-staining" if c else "joint"
            for s, c in zip(same_stain, same_scanner)
        ],
        index=df.index,
    )


def merge_runs(runs: list[tuple[str, pd.DataFrame]]) -> pd.DataFrame:
    """One row per pair, the runs' metrics side by side; first run = reference."""
    labels = [label for label, _ in runs]
    if len(set(labels)) != len(labels):
        raise ValueError(f"duplicate run tags: {labels}")
    ref_label, ref = runs[0]
    out = ref[[*KEYS, "staining_a", "scanner_a", "staining_b", "scanner_b"]].copy()
    out.insert(2, "shift", shift_type(out))
    out.insert(3, "hr", (out["staining_a"] == "HR") | (out["staining_b"] == "HR"))
    for label, df in runs:
        cols = df[[*KEYS, *METRICS]].rename(
            columns={m: f"{label}__{short}" for m, short in METRICS.items()}
        )
        merged = out.merge(cols, on=KEYS, how="left", validate="one_to_one")
        if merged[f"{label}__top_1"].isna().any() or len(df) != len(out):
            raise ValueError(
                f"run {label} does not have the same slide pairs as {ref_label}"
            )
        out = merged
    for label in labels[1:]:
        out[f"{label}__delta_top_1"] = (
            out[f"{label}__top_1"] - out[f"{ref_label}__top_1"]
        )
    return out


def slide_bootstrap_weights(
    pairs: pd.DataFrame, n_boot: int = 2000, seed: int = 0
) -> np.ndarray:
    """Pair weights [n_boot, n_pairs] of a slide-level bootstrap.

    Pairs share slides, so pairs are not independent. Each replicate draws the slides
    with replacement; a pair (a, b) then counts c_a * c_b times (c = how often each
    slide was drawn), i.e. every pair of drawn slides. A weighted mean of per-pair
    values under these weights is one bootstrap replicate of the pair mean.
    """
    slides = pd.Index(sorted(set(pairs["slide_a"]) | set(pairs["slide_b"])))
    ia = slides.get_indexer(pairs["slide_a"])
    ib = slides.get_indexer(pairs["slide_b"])
    rng = np.random.default_rng(seed)
    n = len(slides)
    counts = np.stack(
        [np.bincount(rng.integers(0, n, n), minlength=n) for _ in range(n_boot)]
    ).astype(float)
    return counts[:, ia] * counts[:, ib]


def bootstrap_mean_ci(
    values: np.ndarray, weights: np.ndarray, level: float = 0.95
) -> tuple[float, float]:
    """Percentile interval of the weighted mean of `values` over bootstrap `weights`."""
    totals = weights.sum(1)
    keep = totals > 0
    means = (weights[keep] @ values) / totals[keep]
    tail = 100 * (1 - level) / 2
    lo, hi = np.percentile(means, [tail, 100 - tail])
    return float(lo), float(hi)


def shortlist(pairs: pd.DataFrame, labels: list[str], k: int) -> pd.DataFrame:
    """Per shift: k worst/best reference pairs, k largest gains/losses per method."""
    ref = f"{labels[0]}__top_1"
    picks = []
    for shift in SHIFTS:
        sub = pairs[pairs["shift"] == shift]
        picks.append(sub.nsmallest(k, ref).assign(reason="worst_ref"))
        picks.append(sub.nlargest(k, ref).assign(reason="best_ref"))
        for label in labels[1:]:
            delta = f"{label}__delta_top_1"
            picks.append(sub.nlargest(k, delta).assign(reason=f"gain_{label}"))
            picks.append(sub.nsmallest(k, delta).assign(reason=f"loss_{label}"))
    return pd.concat(picks, ignore_index=True)


def main(argv: list[str] | None = None) -> Path:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--runs",
        type=Path,
        nargs="+",
        required=True,
        help="sipe.bench.plism run dirs; the first is the reference.",
    )
    parser.add_argument("--tag", required=True)
    parser.add_argument("--n-tiles", type=int, default=8139)
    parser.add_argument("--k", type=int, default=5, help="Pairs per shortlist reason.")
    args = parser.parse_args(argv)

    loaded = [load_run(r.resolve(), args.n_tiles) for r in args.runs]
    labels = [label for label, _, _ in loaded]
    pairs = merge_runs([(label, df) for label, df, _ in loaded])
    short = shortlist(pairs, labels, args.k)

    run_dir = new_bench_run_dir(f"plism_pairs_{args.tag}")
    write_run_info(
        run_dir,
        config={
            "benchmark": "plism_pairs",
            **{k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()},
            "runs": [str(r.resolve()) for r in args.runs],
            "labels": labels,
            "reference": labels[0],
        },
        checkpoint={"metrics_csv": {label: str(p) for label, _, p in loaded}},
        features={"source": "plismbench per-pair metrics.csv (no recomputation)"},
    )
    pairs.to_csv(run_dir / "pairs.csv", index=False)
    short.to_csv(run_dir / "shortlist.csv", index=False)

    show = [*KEYS, *(f"{label}__top_1" for label in labels)]
    with pd.option_context(
        "display.width",
        200,
        "display.max_columns",
        None,
        "display.float_format",
        "{:.3f}".format,
    ):
        for shift in SHIFTS:
            for reason, rows in short[short["shift"] == shift].groupby(
                "reason", sort=False
            ):
                print(f"\n[{shift}] {reason}")
                print(rows[show].to_string(index=False))
    print(f"\n{run_dir}")
    return run_dir


if __name__ == "__main__":
    main()
