"""PLISM retrieval viz: top-10 neighbours of hit / miss queries for chosen slide pairs.

Diagnostic for plismbench's top-k accuracy (no new metric). For a slide pair (a, b), a
query tile of a is ranked against every tile of a and b except itself, exactly as
plismbench `TopkAccuracy` (and b -> a, the other direction). A miss whose neighbours are
same-slide tiles (same acquisition, other tissue) means the acquisition distance beats
the content distance for that query.

Bench env (reads PLISM h5 tiles):

    bench/run python -m sipe.bench.plism_retrieval_viz \\
        --pairs-csv runs/bench/<plism_pairs run>/shortlist.csv --reason worst_ref \\
        --tag <tag>
    bench/run python -m sipe.bench.plism_retrieval_viz \\
        --features runs/bench/<gap ref run> runs/bench/<method run> \\
        --pair GMH_S60:GMH_P --tag <tag>

Representations: by default the runs of the `sipe.eval.plism_pairs` run that wrote the
CSV (first = reference); `--features` overrides. `--pair STAIN_SCANNER:STAIN_SCANNER`
adds pairs by hand.

Per query (`queries.csv`): `rank_true` = rank of the matching tile among the 2n - 1
candidates (1 = top-1 hit; ties counted in the query's favour), `rank_true_cross` = its
rank among the other slide's n tiles only. `rank_true > 1` with `rank_true_cross == 1`:
the match is the closest tile of the other slide, and only same-slide tiles beat it.
`n_same_top10`: same-slide tiles in the top-10; `grid_dist_top1`: Chebyshev distance in
tile units between the top-1's location and the query's (0 = the matching location).

Cosine in float32 (plismbench casts to float16, so top-k can differ at the 3rd decimal;
`summary.csv` puts plismbench's per-pair value next to ours where available).

Figures (`figures/`): per pair, direction and kind (`hits`, `misses`, and per method
`fixed_<label>` / `broken_<label>` against the reference), rows = sampled queries x
representations, columns = query | true match | top-10. Frame and tag per tile:
"T" true match, "S" same slide as the query, "O" other slide, wrong location; titles
give the cosine and the grid distance to the query's location. `rank_curve_<pair>.png`:
fraction of queries with `rank_true <= r` for every r (top-r accuracy), per
representation, both directions pooled.
"""

from __future__ import annotations

import argparse
import re
from dataclasses import dataclass
from pathlib import Path

import h5py
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import yaml
from loguru import logger
from plismbench.utils.evaluate import get_tiles_subset_idx

from sipe.bench.plism import DEFAULT_DOWNLOAD_DIR
from sipe.bench.provenance import new_bench_run_dir, write_run_info

TOP = 10
# dataviz reference palette (light): aqua, orange, blue.
COLORS = {"T": "#1baf7a", "S": "#eb6834", "O": "#2a78d6"}
MISS_BUCKETS = ((2, 10), (11, 100), (101, None))


@dataclass
class Rep:
    label: str
    features_dir: Path  # runs/bench/<run>/features/<tag>
    metrics_csv: Path | None


def rep_from_run(run_dir: Path, n_tiles: int) -> Rep:
    tag = yaml.safe_load((run_dir / "config.yaml").read_text())["tag"]
    metrics = run_dir / "metrics" / f"{n_tiles}_tiles" / tag / "metrics.csv"
    return Rep(tag, run_dir / "features" / tag, metrics if metrics.exists() else None)


def slide_name(short: str) -> str:
    """'GMH_P' -> 'GMH_P_to_GMH_S60.tif'; full names pass through."""
    if short.endswith(".tif"):
        return short
    return f"{short}_to_GMH_S60.tif"


def short_name(slide: str) -> str:
    return slide.removesuffix(".tif").removesuffix("_to_GMH_S60")


def load_slide(
    rep: Rep, slide: str, subset: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """(coords [n, 3] int, features [n, D] float32) on the metric's tile subset."""
    arr = np.load(rep.features_dir / slide / "features.npy", mmap_mode="r")[subset]
    return arr[:, :3].astype(int), np.asarray(arr[:, 3:], dtype=np.float32)


def retrieve(
    feats_a: np.ndarray, feats_b: np.ndarray, top: int = TOP, chunk: int = 2048
) -> dict[str, np.ndarray]:
    """Rank the matching tile for every query of a (rows 0..n-1) and b (rows n..2n-1).

    Candidates: all 2n tiles minus the query (plismbench TopkAccuracy). Returns per
    query: rank_true, rank_true_cross, cos_true, top [2n, top] candidate indices and
    top_cos.
    """
    n = feats_a.shape[0]
    device = "cuda" if torch.cuda.is_available() else "cpu"
    x = torch.from_numpy(np.concatenate([feats_a, feats_b])).to(device)
    x = torch.nn.functional.normalize(x, dim=1)
    true_idx = torch.cat([torch.arange(n, 2 * n), torch.arange(0, n)]).to(device)
    out: dict[str, list[torch.Tensor]] = {
        k: [] for k in ("rank_true", "rank_true_cross", "cos_true", "top", "top_cos")
    }
    for start in range(0, 2 * n, chunk):
        q = torch.arange(start, min(start + chunk, 2 * n), device=device)
        sim = x[q] @ x.T
        sim[torch.arange(len(q)), q] = -torch.inf
        cos_true = sim.gather(1, true_idx[q, None])
        above = sim > cos_true
        other = (q >= n)[:, None] ^ (torch.arange(2 * n, device=device) >= n)[None, :]
        out["rank_true"].append(1 + above.sum(1))
        out["rank_true_cross"].append(1 + (above & other).sum(1))
        out["cos_true"].append(cos_true[:, 0])
        top_cos, top_idx = sim.topk(top, dim=1)
        out["top"].append(top_idx)
        out["top_cos"].append(top_cos)
    return {k: torch.cat(v).cpu().numpy() for k, v in out.items()}


def query_table(
    res: dict[str, np.ndarray],
    coords: np.ndarray,
    label: str,
    slide_a: str,
    slide_b: str,
) -> pd.DataFrame:
    """One row per query; coords [n, 3] shared by a and b (asserted by the caller)."""
    n = coords.shape[0]
    q = np.arange(2 * n)
    same = (res["top"] >= n) == (q >= n)[:, None]
    loc_q = coords[q % n, 1:]
    loc_top1 = coords[res["top"][:, 0] % n, 1:]
    return pd.DataFrame(
        {
            "representation": label,
            "direction": np.where(q < n, "a->b", "b->a"),
            "slide_q": np.where(q < n, slide_a, slide_b),
            "slide_t": np.where(q < n, slide_b, slide_a),
            "tile": q % n,
            "x": coords[q % n, 1],
            "y": coords[q % n, 2],
            "rank_true": res["rank_true"],
            "rank_true_cross": res["rank_true_cross"],
            "cos_true": res["cos_true"],
            "cos_top1": res["top_cos"][:, 0],
            "top1_same_slide": same[:, 0],
            "n_same_top10": same.sum(1),
            "grid_dist_top1": np.abs(loc_top1 - loc_q).max(1),
        }
    )


def summarize(qt: pd.DataFrame) -> dict:
    miss = qt[qt["rank_true"] > 1]
    row = {
        "n_queries": len(qt),
        "top_1": float((qt["rank_true"] <= 1).mean()),
        "top_10": float((qt["rank_true"] <= TOP).mean()),
        "n_miss": len(miss),
    }
    if len(miss):
        row |= {
            "miss_rank_q25": float(miss["rank_true"].quantile(0.25)),
            "miss_rank_median": float(miss["rank_true"].median()),
            "miss_rank_q75": float(miss["rank_true"].quantile(0.75)),
            "miss_frac_top1_same_slide": float(miss["top1_same_slide"].mean()),
            "miss_frac_cross_rank1": float((miss["rank_true_cross"] == 1).mean()),
            "miss_frac_top1_adjacent": float((miss["grid_dist_top1"] <= 1).mean()),
            "miss_mean_n_same_top10": float(miss["n_same_top10"].mean()),
            "miss_mean_cos_gap": float((miss["cos_top1"] - miss["cos_true"]).mean()),
        }
    return row


def plismbench_value(rep: Rep, slide_a: str, slide_b: str, col: str) -> float:
    if rep.metrics_csv is None:
        return float("nan")
    df = pd.read_csv(rep.metrics_csv)
    hit = df[
        ((df.slide_a == slide_a) & (df.slide_b == slide_b))
        | ((df.slide_a == slide_b) & (df.slide_b == slide_a))
    ]
    return float(hit[col].iloc[0]) if len(hit) else float("nan")


def sample_queries(
    ref: pd.DataFrame, others: dict[str, pd.DataFrame], n: int, rng: np.random.Generator
) -> dict[str, np.ndarray]:
    """Query rows (indices into the per-direction tables) per figure kind."""

    def pick(mask: np.ndarray, k: int) -> np.ndarray:
        idx = np.flatnonzero(mask)
        return np.sort(rng.choice(idx, size=min(k, len(idx)), replace=False))

    rank = ref["rank_true"].to_numpy()
    kinds = {"hits": pick(rank == 1, n)}
    per_bucket = max(1, -(-n // len(MISS_BUCKETS)))
    misses = [
        pick((rank >= lo) & ((rank <= hi) if hi else True), per_bucket)
        for lo, hi in MISS_BUCKETS
    ]
    kinds["misses"] = np.concatenate(misses)[:n] if misses else np.array([], int)
    for label, qt in others.items():
        r = qt["rank_true"].to_numpy()
        kinds[f"fixed_{label}"] = pick((rank > 1) & (r == 1), n)
        kinds[f"broken_{label}"] = pick((rank == 1) & (r > 1), n)
    return kinds


class TileReader:
    """PLISM tiles by (slide, coords) from the plismbench h5 files."""

    def __init__(self, download_dir: Path):
        self.download_dir = download_dir
        self.files: dict[str, h5py.File] = {}

    def __call__(self, slide: str, c: np.ndarray) -> np.ndarray:
        if slide not in self.files:
            self.files[slide] = h5py.File(self.download_dir / f"{slide}.h5", "r")
        return self.files[slide][f"tile_{c[0]}_{c[1]}_{c[2]}"][:]

    def close(self) -> None:
        for f in self.files.values():
            f.close()


def _show(ax, img, tag: str | None, title: str) -> None:
    ax.imshow(img)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_title(title, fontsize=7)
    color = COLORS.get(tag, "#c3c2b7")
    for spine in ax.spines.values():
        spine.set_edgecolor(color)
        spine.set_linewidth(4 if tag else 1)
    if tag:
        ax.text(
            6,
            30,
            tag,
            color="white",
            fontsize=9,
            fontweight="bold",
            bbox={"facecolor": color, "edgecolor": "none", "pad": 1.5},
        )


def plot_queries(
    path: Path,
    title: str,
    rows: np.ndarray,
    tables: dict[str, pd.DataFrame],
    results: dict[str, dict[str, np.ndarray]],
    offset: int,
    coords: np.ndarray,
    slides: tuple[str, str],
    tiles: TileReader,
) -> None:
    """rows: query indices within one direction; offset: n for b->a, 0 for a->b."""
    n = coords.shape[0]
    labels = list(tables)
    n_rows = len(rows) * len(labels)
    fig, axes = plt.subplots(
        n_rows, TOP + 2, figsize=(1.25 * (TOP + 2), 1.45 * n_rows), squeeze=False
    )
    slide_q, slide_t = (slides[0], slides[1]) if offset == 0 else (slides[1], slides[0])
    for i, row in enumerate(rows):
        q = row + offset
        loc = coords[row, 1:]
        for j, label in enumerate(labels):
            r = i * len(labels) + j
            qt, res = tables[label].iloc[row], results[label]
            ax_row = axes[r]
            _show(
                ax_row[0],
                tiles(slide_q, coords[row]),
                None,
                f"{label}\nquery {short_name(slide_q)}",
            )
            ax_row[0].set_ylabel(f"({loc[0]},{loc[1]})", fontsize=7)
            _show(
                ax_row[1],
                tiles(slide_t, coords[row]),
                "T",
                f"true: rank {qt.rank_true}\n(cross {qt.rank_true_cross}) "
                f"cos {qt.cos_true:.3f}",
            )
            for k in range(TOP):
                c = res["top"][q, k]
                same = (c >= n) == (q >= n)
                tag = "T" if c == (q + n) % (2 * n) else ("S" if same else "O")
                slide = slide_q if same else slide_t
                dist = int(np.abs(coords[c % n, 1:] - loc).max())
                _show(
                    ax_row[k + 2],
                    tiles(slide, coords[c % n]),
                    tag,
                    f"#{k + 1} cos {res['top_cos'][q, k]:.3f}\nd={dist}",
                )
    fig.suptitle(
        f"{title}   T true match · S same slide as query · O other slide, wrong "
        "location · d = grid distance to the query location (tiles)",
        fontsize=9,
        y=1.0,
    )
    fig.tight_layout()
    fig.savefig(path, dpi=110, bbox_inches="tight")
    plt.close(fig)


def plot_rank_curve(path: Path, title: str, tables: dict[str, pd.DataFrame]) -> None:
    """Top-r accuracy for every r: fraction of queries with rank_true <= r."""
    fig, ax = plt.subplots(figsize=(7, 3.5))
    palette = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300"]
    for (label, qt), color in zip(tables.items(), palette):
        ranks = np.sort(qt["rank_true"].to_numpy())
        r, first = np.unique(ranks, return_index=True)
        frac = (first + np.diff(np.append(first, len(ranks)))) / len(ranks)
        ax.step(
            np.append(r, ranks[-1] * 1.5), np.append(frac, 1.0), where="post",
            linewidth=2, color=color,
            label=f"{label} (top-1 {frac[0] if r[0] == 1 else 0:.3f})",
        )  # fmt: skip
    ax.axvline(TOP, color="#c3c2b7", linewidth=1, linestyle="--")
    ax.set_xscale("log")
    ax.set_xlabel("r (rank of the true match among 2n - 1 candidates)")
    ax.set_ylabel("fraction of queries with rank <= r")
    ax.set_title(f"{title}, both directions", fontsize=9)
    ax.grid(alpha=0.3)
    ax.legend(fontsize=7, frameon=False, loc="lower right")
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)


def parse_pairs(args: argparse.Namespace) -> list[tuple[str, str, str]]:
    """(slide_a, slide_b, reason); deduplicated, first reason kept."""
    pairs: dict[frozenset, tuple[str, str, str]] = {}
    if args.pairs_csv is not None:
        df = pd.read_csv(args.pairs_csv)
        if args.reason:
            df = df[df["reason"].isin(args.reason)]
        if args.max_pairs is not None:
            df = df.head(args.max_pairs)
        for a, b, reason in zip(
            df.slide_a, df.slide_b, df.get("reason", pd.Series("csv", index=df.index))
        ):
            pairs.setdefault(frozenset((a, b)), (a, b, reason))
    for spec in args.pair or []:
        a, b = (slide_name(s) for s in spec.split(":"))
        pairs.setdefault(frozenset((a, b)), (a, b, "manual"))
    if not pairs:
        raise SystemExit("no pairs: pass --pairs-csv and/or --pair")
    return list(pairs.values())


def _plain(v):
    """Paths (also in lists) as strings for config.yaml."""
    if isinstance(v, list):
        return [_plain(x) for x in v]
    return str(v) if isinstance(v, Path) else v


def resolve_reps(args: argparse.Namespace) -> list[Rep]:
    if args.features:
        runs = args.features
    elif args.pairs_csv is not None:
        config = yaml.safe_load((args.pairs_csv.parent / "config.yaml").read_text())
        runs = [Path(r) for r in config["runs"]]
    else:
        raise SystemExit(
            "pass --features, or --pairs-csv from a sipe.eval.plism_pairs run"
        )
    return [rep_from_run(r.resolve(), args.n_tiles) for r in runs]


def main(argv: list[str] | None = None) -> Path:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--tag", required=True)
    parser.add_argument(
        "--pairs-csv",
        type=Path,
        help="pairs.csv / shortlist.csv of a sipe.eval.plism_pairs run.",
    )
    parser.add_argument(
        "--reason",
        nargs="+",
        help="Keep these shortlist reasons (e.g. worst_ref gain_<label>).",
    )
    parser.add_argument(
        "--max-pairs", type=int, help="First N CSV rows after --reason."
    )
    parser.add_argument(
        "--pair", nargs="+", help="Manual pairs STAIN_SCANNER:STAIN_SCANNER."
    )
    parser.add_argument(
        "--features",
        type=Path,
        nargs="+",
        help="sipe.bench.plism run dirs (first = reference).",
    )
    parser.add_argument("--n-tiles", type=int, default=8139)
    parser.add_argument("--n-examples", type=int, default=8, help="Queries per figure.")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--download-dir", type=Path, default=DEFAULT_DOWNLOAD_DIR)
    args = parser.parse_args(argv)

    reps = resolve_reps(args)
    pairs = parse_pairs(args)
    subset = get_tiles_subset_idx(args.n_tiles)
    rng = np.random.default_rng(args.seed)

    run_dir = new_bench_run_dir(f"plism_retrieval_viz_{args.tag}")
    fig_dir = run_dir / "figures"
    fig_dir.mkdir()
    write_run_info(
        run_dir,
        config={
            "benchmark": "plism_retrieval_viz",
            **{k: _plain(v) for k, v in vars(args).items()},
            "pairs": [list(p) for p in pairs],
            "representations": {r.label: str(r.features_dir) for r in reps},
            "reference": reps[0].label,
            "similarity": "cosine, float32 (plismbench: float16)",
        },
        checkpoint={"features": {r.label: str(r.features_dir) for r in reps}},
        features={"source": "sipe.bench.plism feature dirs (no re-extraction)"},
    )
    logger.info(f"Run dir: {run_dir}")

    tiles = TileReader(args.download_dir.resolve())
    all_queries, summary = [], []
    try:
        for slide_a, slide_b, reason in pairs:
            name = f"{short_name(slide_a)}--{short_name(slide_b)}"
            logger.info(f"{name} ({reason})")
            tables, results, coords = {}, {}, None
            for rep in reps:
                ca, fa = load_slide(rep, slide_a, subset)
                cb, fb = load_slide(rep, slide_b, subset)
                np.testing.assert_array_equal(ca, cb)
                if coords is not None:
                    np.testing.assert_array_equal(coords, ca)
                coords = ca
                results[rep.label] = retrieve(fa, fb)
                tables[rep.label] = query_table(
                    results[rep.label], coords, rep.label, slide_a, slide_b
                )
                qt = tables[rep.label].assign(reason=reason)
                all_queries.append(qt)
                for direction, sub in [("both", qt), *qt.groupby("direction")]:
                    summary.append(
                        {
                            "representation": rep.label,
                            "slide_a": slide_a,
                            "slide_b": slide_b,
                            "reason": reason,
                            "direction": direction,
                            **summarize(sub),
                            **(
                                {
                                    "plismbench_top_1": plismbench_value(
                                        rep, slide_a, slide_b, "top_1_accuracy"
                                    ),
                                    "plismbench_top_10": plismbench_value(
                                        rep, slide_a, slide_b, "top_10_accuracy"
                                    ),
                                }
                                if direction == "both"
                                else {}
                            ),
                        }
                    )
                    if direction != "both" and rep is not reps[0]:
                        ref = tables[reps[0].label]
                        ref_hit = ref.loc[ref.direction == direction, "rank_true"] == 1
                        hit = sub["rank_true"].to_numpy() == 1
                        summary[-1] |= {
                            "vs_ref_hit_hit": int((ref_hit.to_numpy() & hit).sum()),
                            "vs_ref_hit_miss": int((ref_hit.to_numpy() & ~hit).sum()),
                            "vs_ref_miss_hit": int((~ref_hit.to_numpy() & hit).sum()),
                            "vs_ref_miss_miss": int((~ref_hit.to_numpy() & ~hit).sum()),
                        }
            pd.DataFrame(summary).to_csv(run_dir / "summary.csv", index=False)
            plot_rank_curve(
                fig_dir / f"rank_curve_{name}.png", f"{name} ({reason})", tables
            )

            n = coords.shape[0]
            for direction, offset, short in (("a->b", 0, "a2b"), ("b->a", n, "b2a")):
                per_dir = {
                    label: t[t.direction == direction].reset_index(drop=True)
                    for label, t in tables.items()
                }
                ref_label = reps[0].label
                others = {k: v for k, v in per_dir.items() if k != ref_label}
                kinds = sample_queries(per_dir[ref_label], others, args.n_examples, rng)
                for kind, rows in kinds.items():
                    if not len(rows):
                        continue
                    stem = re.sub(r"[^\w.-]", "_", f"{name}_{short}_{kind}")
                    plot_queries(
                        fig_dir / f"{stem}.png",
                        f"{name} {direction} — {kind} (sampled on {ref_label})",
                        rows,
                        per_dir,
                        results,
                        offset,
                        coords,
                        (slide_a, slide_b),
                        tiles,
                    )
    finally:
        tiles.close()
    pd.concat(all_queries, ignore_index=True).to_csv(
        run_dir / "queries.csv.gz", index=False
    )
    logger.success(f"PLISM retrieval viz: {run_dir}")
    return run_dir


if __name__ == "__main__":
    main()
