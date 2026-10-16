"""Results tables for docs/progress.md and docs/experiment-log.md, read from run dirs
(no metric typed by hand).

Sources, all under runs/ (gitignored, so every row names the dir it came from):
  - PLISM: runs/bench/*/metrics/<n>_tiles/<tag>/results.csv (top-1, mean part); with
    drop_stain=, the same means recomputed from the sibling per-pair metrics.csv
  - PLISM per pair vs a reference: runs/bench/*/metrics/<n>_tiles/<tag>/metrics.csv
    (plismbench per-pair metrics, merged as in sipe.eval.plism_pairs)
  - HEST: runs/bench/*/results/*/dataset_results.json
  - SCORPION proxy: runs/bench/*/summary.csv (sipe.eval.scorpion_retrieval)
  - soft-shrinkage sweep: runs/bench/*/summary_test.csv + selected.json
  - content dependence (Experiment 1): runs/bench/*/summary.csv
    (sipe.eval.content_dependence)
  - mixture sweep (Experiment 3): runs/bench/*/summary_{val,test}.csv, selected.json,
    likelihood.csv, gating.csv (sipe.eval.mixture_sweep)
  - scanner offsets (exchangeability diagnostic): runs/bench/*/summary.csv
    (sipe.eval.scanner_offsets)
  - content-dependent scanner offsets: runs/bench/*/summary.csv
    (sipe.eval.content_offsets)
  - quadratic sweep (Experiment 4): runs/bench/*/summary_{val,test}.csv,
    selected.json, sweep_val.csv (sipe.eval.quadratic_sweep)
  - CGM augmentation attribution: runs/bench/*/summary_val.csv + arms.json
    (sipe.eval.aug_attribution)
  - CGM selection (Experiment 7): runs/bench/*/summary_{val,test}.csv,
    selected.json (sipe.eval.cgm_select)
  - leave-one-scanner-out (Experiment 7): runs/bench/*/summary_val.csv +
    contrasts_val.csv (sipe.eval.loso)
  - training: runs/<run>/wandb/*run-*/*.wandb (local W&B history; no API key needed)

The docs (`DOCS`: progress.md for the ISBI paper, experiment-log.md for the rest) hold
generated blocks; the same block may appear in both:

  <!-- results:plism exclude=*smoke* -->
  ...table written by this tool...
  <!-- /results:plism -->

Options on the opening marker (space-separated key=value):
  exclude=<glob>[,<glob>]   skip run dirs matching any glob (all blocks)
  extra=<dir>[,<dir>]       plism only: metrics dirs outside runs/ with results.csv
  module=<Class>[,<Class>]  training only: keep runs of these SIPEModule classes
  cols=<key>[,<key>]        training only: val/ metrics to show (last-10-val mean)
  min_steps=<n>             training only: skip runs that stopped before step n
  ref=<tag>                 plism_pairs only: reference PLISM run (by tag)
  tags=<tag>[,<tag>]        plism_pairs only: methods compared with ref
  drop_stain=<s>[,<s>]      plism / plism_pairs: drop the slide pairs with one of these
                            stains on either side (e.g. HR), then aggregate as usual

  uv run python -m sipe.eval.results_tables          # print every table
  uv run python -m sipe.eval.results_tables --write  # refresh the blocks in every doc
  uv run python -m sipe.eval.results_tables --check  # exit 1 if a doc is stale
"""

from __future__ import annotations

import argparse
import fnmatch
import glob
import json
import math
import re
import struct
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import pandas as pd
import yaml

SIPE_ROOT = Path(__file__).resolve().parents[3]
RUNS = SIPE_ROOT / "runs"
PROGRESS = SIPE_ROOT / "docs" / "progress.md"
DOCS = (PROGRESS, SIPE_ROOT / "docs" / "experiment-log.md")

PLISM_ROWS = {
    "inter-scanner": "inter-scanner",
    "inter-staining": "inter-staining",
    "inter-scanner, inter-staining": "scanner + staining",
    "all": "all",
}
HEST_TASKS = (
    "CCRCC", "COAD", "HCC", "IDC", "LUNG", "LYMPH_IDC", "PAAD", "PRAD", "READ", "SKCM",
)  # fmt: skip
PROXY_COLS = {
    "top_1_accuracy": "top-1",
    "top_5_accuracy": "top-5",
    "probe_logreg_acc": "scanner probe logreg",
    "probe_mlp_acc": "scanner probe MLP",
    "region_probe_acc": "region probe",
}
TRAIN_COLS = (
    "probe_zgap_acc",
    "probe_backbone_acc",
    "probe_s_acc",
    "retrieval_top1_zgap",
    "retrieval_top1_backbone",
    "feature_recon_loss",
    "kl",
)
LAST_N_VALS = 10

BLOCK_RE = re.compile(
    r"(<!-- results:(?P<name>\w+)(?P<opts>[^>]*?) -->\n)"
    r"(?P<body>.*?)"
    r"(<!-- /results:(?P=name) -->)",
    re.S,
)


# --- small readers -----------------------------------------------------------


def _read_yaml(path: Path) -> dict[str, Any]:
    return yaml.safe_load(path.read_text()) if path.exists() else {}


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text()) if path.exists() else {}


def _rel(path: Path | str) -> str:
    path = Path(path)
    try:
        return str(path.resolve().relative_to(SIPE_ROOT))
    except ValueError:
        return str(path)


def _fmt(x: Any, digits: int = 3) -> str:
    if x is None or (isinstance(x, float) and math.isnan(x)):
        return "–"
    if isinstance(x, float):
        return f"{x:.{digits}f}"
    return str(x)


def _commit(state: dict[str, Any] | None) -> str:
    if not state or not state.get("commit"):
        return "–"
    return state["commit"][:7] + ("*" if state.get("dirty") else "")


def _git_txt(run_dir: Path) -> dict[str, Any]:
    """runs/<run>/git.txt: 'commit: <sha>' / 'dirty: True|False'."""
    out: dict[str, Any] = {}
    path = run_dir / "git.txt"
    if path.exists():
        for line in path.read_text().splitlines():
            key, _, value = line.partition(":")
            out[key.strip()] = value.strip()
        out["dirty"] = out.get("dirty") == "True"
    return out


def _wandb_id(run_dir: Path | None) -> str:
    if run_dir is None:
        return "–"
    return str(_read_yaml(run_dir / "wandb.txt").get("id", "–"))


@dataclass
class BenchRun:
    """One runs/bench/<stamp>_<host>_<rand>_<tag>/ dir and its provenance."""

    dir: Path
    config: dict[str, Any] = field(default_factory=dict)
    provenance: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def load(cls, run_dir: Path) -> BenchRun:
        return cls(
            run_dir,
            _read_yaml(run_dir / "config.yaml"),
            _read_json(run_dir / "provenance.json"),
        )

    @property
    def benchmark(self) -> str:
        return str(self.config.get("benchmark", ""))

    @property
    def representation(self) -> str:
        rep = (self.provenance.get("features") or {}).get("representation")
        return rep or str(self.config.get("representation", "–"))

    @property
    def train_run(self) -> Path | None:
        """Training run behind the features; None for backbone-only representations."""
        ckpt = self.provenance.get("checkpoint") or {}
        if not ckpt.get("train_run_dir") or self.representation.startswith("backbone"):
            return None
        return Path(ckpt["train_run_dir"])

    def source_cells(self) -> list[str]:
        train = self.train_run
        return [
            f"`{_rel(self.dir)}`",
            _wandb_id(train) if train else "–",
            _commit((self.provenance.get("repos") or {}).get("sipe")),
        ]


SOURCE_HEADER = ["bench run dir", "train W&B", "SIPE commit"]


def _bench_runs(root: Path, benchmark: str, exclude: list[str]) -> list[BenchRun]:
    runs = []
    for run_dir in sorted((root / "bench").glob("*/")):
        if any(fnmatch.fnmatch(run_dir.name, pat) for pat in exclude):
            continue
        run = BenchRun.load(run_dir)
        if run.benchmark == benchmark:
            runs.append(run)
    return runs


def _table(header: list[str], rows: list[list[str]]) -> str:
    lines = ["| " + " | ".join(header) + " |", "|" + "---|" * len(header)]
    lines += ["| " + " | ".join(row) + " |" for row in rows]
    return "\n".join(lines)


# --- PLISM --------------------------------------------------------------------


def read_plism_results(results_csv: Path) -> dict[str, float]:
    """Top-1 mean per row label; cells read 'mean (std) ; median (IQR)'."""
    df = pd.read_csv(results_csv, index_col=0)
    return {
        PLISM_ROWS.get(label, label): float(str(cell).split("(")[0])
        for label, cell in df["top_1_accuracy"].items()
    }


def _keeps_stains(df: pd.DataFrame, drop: list[str]) -> pd.Series:
    """Pairs with none of the `drop` stains on either side."""
    return ~(df["staining_a"].isin(drop) | df["staining_b"].isin(drop))


def plism_top1_from_metrics(
    metrics_csv: Path, drop_stains: list[str]
) -> dict[str, float]:
    """Top-1 mean per row from plismbench's per-pair metrics.csv, as in plismbench
    `get_results` (same row masks), after dropping pairs with a `drop_stains` stain."""
    df = pd.read_csv(metrics_csv)
    df = df[_keeps_stains(df, drop_stains)]
    same_stain = df["staining_a"] == df["staining_b"]
    same_scanner = df["scanner_a"] == df["scanner_b"]
    masks = {
        "inter-scanner": same_stain,
        "inter-staining": same_scanner,
        "scanner + staining": ~same_stain & ~same_scanner,
        "all": pd.Series(True, index=df.index),
    }
    return {k: float(df.loc[m, "top_1_accuracy"].mean()) for k, m in masks.items()}


def plism_table(root: Path, opts: dict[str, list[str]]) -> str:
    drop = opts.get("drop_stain", [])

    def read(results_csv: Path) -> dict[str, float]:
        if not drop:
            return read_plism_results(results_csv)
        return plism_top1_from_metrics(results_csv.with_name("metrics.csv"), drop)

    header = ["tag", "repr", *PLISM_ROWS.values(), *SOURCE_HEADER]
    rows = []
    for run in _bench_runs(root, "plism", opts.get("exclude", [])):
        found = sorted(run.dir.glob("metrics/*_tiles/*/results.csv"))
        if not found:
            empty = ["no results"] + ["–"] * (len(PLISM_ROWS) - 1)
            tag = str(run.config.get("tag"))
            rows.append([tag, run.representation, *empty, *run.source_cells()])
            continue
        top1 = read(found[0])
        rows.append(
            [str(run.config.get("tag")), run.representation]
            + [_fmt(top1.get(k)) for k in PLISM_ROWS.values()]
            + run.source_cells()
        )
    for extra in opts.get("extra", []):
        top1 = read(SIPE_ROOT / extra / "results.csv")
        rows.append(
            [Path(extra).name, "–"]
            + [_fmt(top1.get(k)) for k in PLISM_ROWS.values()]
            + [f"`{extra}`", "–", "–"]
        )
    if not drop:
        caption = "Top-1 accuracy, mean over slide pairs (PLISM `results.csv`)."
    else:
        caption = (
            "Top-1 accuracy, mean over slide pairs without the stain(s)"
            f" {', '.join(drop)} on either side (recomputed from PLISM `metrics.csv`"
            " with the `results.csv` row masks)."
        )
    return _table(header, rows) + "\n\n" + caption


# --- PLISM per pair vs a reference ----------------------------------------------

PAIR_GROUPS = (
    ("inter-scanner", "inter-scanner", False),
    ("inter-staining", "inter-staining", False),
    ("scanner + staining", "joint", False),
    ("all", None, False),
    ("inter-scanner, HR", "inter-scanner", True),
    ("all, HR", None, True),
)
PAIR_BOOTSTRAPS = 2000


def _plism_run_by_tag(
    root: Path, tag: str, exclude: list[str]
) -> tuple[BenchRun, Path]:
    """Newest PLISM run with this tag and its per-pair metrics.csv."""
    found = [
        (run, path)
        for run in _bench_runs(root, "plism", exclude)
        if str(run.config.get("tag")) == tag
        for path in sorted(run.dir.glob(f"metrics/*_tiles/{tag}/metrics.csv"))
    ]
    if not found:
        raise FileNotFoundError(f"no finished PLISM run with tag {tag!r} under {root}")
    return found[-1]


def plism_pairs_table(root: Path, opts: dict[str, list[str]]) -> str:
    from sipe.eval.plism_pairs import (
        bootstrap_mean_ci,
        merge_runs,
        slide_bootstrap_weights,
    )

    if not opts.get("ref"):
        return "(needs `ref=<tag>` and `tags=<tag>,...` on the block marker)"
    exclude = opts.get("exclude", [])
    (ref,) = opts["ref"]
    tags = opts.get("tags", [])
    loaded = [_plism_run_by_tag(root, tag, exclude) for tag in (ref, *tags)]
    pairs = merge_runs(
        [(str(run.config["tag"]), pd.read_csv(path)) for run, path in loaded]
    )
    drop = opts.get("drop_stain", [])
    if drop:
        pairs = pairs[_keeps_stains(pairs, drop)].reset_index(drop=True)
    groups = [g for g in PAIR_GROUPS if not (drop and g[2])]
    weights = slide_bootstrap_weights(pairs, PAIR_BOOTSTRAPS)
    header = [
        "method", "pairs", "n", "top-1", "ref top-1", "Δ top-1", "95% CI",
        "pairs better", "bench run dir",
    ]  # fmt: skip
    rows = []
    for run, _ in loaded[1:]:
        tag = str(run.config["tag"])
        delta = (pairs[f"{tag}__top_1"] - pairs[f"{ref}__top_1"]).to_numpy()
        for label, shift, hr in groups:
            mask = pd.Series(True, index=pairs.index)
            if shift is not None:
                mask &= pairs["shift"] == shift
            if hr:
                mask &= pairs["hr"].astype(bool)
            mask = mask.to_numpy()
            if not mask.any():
                continue
            lo, hi = bootstrap_mean_ci(delta[mask], weights[:, mask])
            rows.append(
                [
                    tag,
                    label,
                    str(int(mask.sum())),
                    _fmt(float(pairs.loc[mask, f"{tag}__top_1"].mean())),
                    _fmt(float(pairs.loc[mask, f"{ref}__top_1"].mean())),
                    f"{delta[mask].mean():+.3f}",
                    f"[{lo:+.3f}, {hi:+.3f}]",
                    f"{(delta[mask] > 0).mean():.0%}",
                    f"`{_rel(run.dir)}`",
                ]
            )
    return (
        _table(header, rows)
        + f"\n\nReference: `{ref}` (`{_rel(loaded[0][0].dir)}`). Per slide pair"
        " top-1 (plismbench `metrics.csv`, mean of both directions); Δ = method"
        " minus reference, mean over the pairs of the group; 95% CI = percentile"
        f" interval of a slide-level bootstrap ({PAIR_BOOTSTRAPS} draws, seed 0;"
        " pairs share slides, so slides are resampled and a pair counts once per"
        " drawn copy of each of its slides); pairs better = share of pairs with"
        " Δ > 0. HR = either slide has the HR stain. One fit per method, so the"
        " interval covers the slide sample only, not seeds."
        + (
            f" Pairs with the stain(s) {', '.join(drop)} on either side are dropped"
            " before grouping and bootstrapping."
            if drop
            else ""
        )
    )


# --- HEST ---------------------------------------------------------------------


def read_hest_results(
    results_json: Path,
) -> tuple[dict[str, tuple[float, float]], float | None]:
    data = _read_json(results_json)
    per_task = {
        entry["dataset_name"]: (r["pearson_mean"], r["pearson_std"])
        for entry in data["results"]
        for r in entry["results"]
    }
    average = next(iter(data.get("average", {}).values()), None)
    return per_task, average


def hest_table(root: Path, opts: dict[str, list[str]]) -> str:
    header = ["tag", "repr", *HEST_TASKS, "average", *SOURCE_HEADER]
    rows = []
    for run in _bench_runs(root, "hest", opts.get("exclude", [])):
        found = sorted(run.dir.glob("results/*/dataset_results.json"))
        if not found:
            cells = ["no results"] + ["–"] * len(HEST_TASKS)
        else:
            per_task, average = read_hest_results(found[0])
            cells = [
                f"{v[0]:.3f} ({v[1]:.3f})" if (v := per_task.get(t)) else "–"
                for t in HEST_TASKS
            ]
            # The average only means something over the full task set.
            cells.append(_fmt(average) if set(HEST_TASKS) <= set(per_task) else "–")
        rows.append(
            [
                str(run.config.get("tag")),
                run.representation,
                *cells,
                *run.source_cells(),
            ]
        )
    return (
        _table(header, rows)
        + "\n\nPearson mean (std over folds), HEST `dataset_results.json`;"
        " average only when all tasks ran."
    )


# --- SCORPION proxy -----------------------------------------------------------


def scorpion_table(root: Path, opts: dict[str, list[str]]) -> str:
    header = ["tag", "repr", *PROXY_COLS.values(), *SOURCE_HEADER]
    rows = []
    for run in _bench_runs(root, "scorpion_retrieval", opts.get("exclude", [])):
        summary = run.dir / "summary.csv"
        if not summary.exists():
            empty = ["no results"] + ["–"] * len(PROXY_COLS)
            rows.append([str(run.config.get("tag")), *empty, *run.source_cells()])
            continue
        df = pd.read_csv(summary, index_col=0)
        for rep, metrics in df.iterrows():
            cells = run.source_cells()
            if str(rep).startswith("backbone"):
                cells[1] = "–"
            rows.append(
                [str(run.config.get("tag")), str(rep)]
                + [_fmt(metrics.get(c)) for c in PROXY_COLS]
                + cells
            )
    return (
        _table(header, rows)
        + "\n\nHeld-out test split; probes trained on val (`summary.csv`)."
    )


# --- soft-shrinkage sweep -----------------------------------------------------


def softshrink_table(root: Path, opts: dict[str, list[str]]) -> str:
    header = [
        "tag",
        "pooling",
        "deltas",
        "λ",
        *PROXY_COLS.values(),
        "note",
        "bench run dir",
        "SIPE commit",
    ]
    rows = []
    for run in _bench_runs(
        root, "scorpion_soft_shrinkage_sweep", opts.get("exclude", [])
    ):
        source = [f"`{_rel(run.dir)}`", run.source_cells()[2]]
        tag = str(run.config.get("tag"))
        # Pairing unit of B; runs before --deltas existed used slide-level deltas.
        level = str(run.config.get("delta_level", "slide"))
        summary = run.dir / "summary_test.csv"
        if not summary.exists():
            n_val = (
                len(pd.read_csv(run.dir / "sweep_val.csv"))
                if (run.dir / "sweep_val.csv").exists()
                else 0
            )
            note = f"no test yet ({n_val} val λ rows)"
            rows.append([tag, "–", level, "–", *["–"] * len(PROXY_COLS), note, *source])
            continue
        selected = _read_json(run.dir / "selected.json")
        for _, r in pd.read_csv(summary).iterrows():
            sel = selected.get(r["pooling"], {})
            note = "raw" if r["lambda"] == 0 else sel.get("rule", "best val top-1")
            if r["lambda"] != 0 and sel.get("region_guard_violated"):
                note += ", region guard violated"
            if r["lambda"] != 0 and sel.get("at_grid_edge"):
                note += ", at grid edge"
            rows.append(
                [tag, str(r["pooling"]), level, f"{r['lambda']:g}"]
                + [_fmt(r.get(c)) for c in PROXY_COLS]
                + [note, *source]
            )
    return (
        _table(header, rows) + "\n\nTest split, proxy protocol (`summary_test.csv`);"
        " λ selected on val, rule in the note (`selected.json`)."
    )


# --- content dependence (Experiment 1) -----------------------------------------


def content_table(root: Path, opts: dict[str, list[str]]) -> str:
    header = [
        "setting", "K", "α*", "CV Δll", "CV folds > 0", "CV shuffled", "CV scale-only",
        "val Δll [WSI 95% CI]", "val WSIs > 0", "val shuffled", "val scale-only",
        "val energy MSE ratio (shuffled)", "heterog. fit / val", "bench run dir",
        "SIPE commit",
    ]  # fmt: skip
    rows = []
    for run in _bench_runs(
        root, "scorpion_content_dependence", opts.get("exclude", [])
    ):
        source = [f"`{_rel(run.dir)}`", run.source_cells()[2]]
        summary = run.dir / "summary.csv"
        if not summary.exists():
            rows.append(["no results", *["–"] * (len(header) - 3), *source])
            continue
        for _, r in pd.read_csv(summary).iterrows():
            rows.append(
                [
                    str(r["setting"]),
                    str(r["K"]),
                    f"{r['alpha_star']:g}",
                    f"{r['cv_diff']:.3f} ± {r['cv_diff_sd']:.3f}",
                    f"{r['cv_folds_pos']}",
                    _fmt(r["cv_diff_shuffled"]),
                    _fmt(r.get("cv_diff_scale")),
                    f"{r['val_diff']:.3f} [{r['val_ci_lo']:.3f}, {r['val_ci_hi']:.3f}]",
                    f"{r['val_wsi_pos']}/{r['val_n_wsi']}",
                    _fmt(r["val_diff_shuffled"]),
                    _fmt(r.get("val_diff_scale")),
                    f"{r['val_energy_mse_ratio']:.3f}"
                    f" ({r['val_energy_mse_ratio_shuffled']:.3f})",
                    f"{r['heterogeneity_fit']:.3f} / {r['heterogeneity_val']:.3f}",
                    *source,
                ]
            )
    return (
        _table(header, rows)
        + "\n\nHeld-out mean log-likelihood difference, content-conditioned minus"
        " global residual model (nats/dim, > 0 favors content; `summary.csv`). CV ="
        " grouped folds over train WSIs (± sd over folds); val = fit on train. α* ="
        " shrinkage toward the global covariance, chosen by CV. Shuffled = cluster"
        " labels permuted in train; scale-only = cluster-specific scalar × global"
        " covariance. Energy MSE ratio < 1: cluster means predict held-out residual"
        " energy better than the global mean."
    )


# --- scanner offsets (exchangeability diagnostic) ---------------------------------


def offsets_table(root: Path, opts: dict[str, list[str]]) -> str:
    header = [
        "setting", "scanners", "energy share raw: global / slide / region / tile",
        "whitened: global / slide / region / tile", "val explained by train offsets"
        " raw / whitened", "val Δll offset", "val Δll cov", "val Δll offset+cov [WSI"
        " 95% CI]", "val Δll mixture, no label [WSI 95% CI]", "val resid. scanner acc",
        "offset span: rank / offset fraction / content share",
        "bench run dir", "SIPE commit",
    ]  # fmt: skip
    rows = []
    for run in _bench_runs(root, "scorpion_scanner_offsets", opts.get("exclude", [])):
        source = [f"`{_rel(run.dir)}`", run.source_cells()[2]]
        summary = run.dir / "summary.csv"
        if not summary.exists():
            rows.append(["no results", *["–"] * (len(header) - 3), *source])
            continue
        for _, r in pd.read_csv(summary).iterrows():

            def shares(metric: str, r=r) -> str:
                return " / ".join(
                    f"{r[f'{metric}_{lv}']:.3f}"
                    for lv in ("global", "slide", "region", "tile")
                )

            def ci(model: str, r=r) -> str:
                return (
                    f"{r[f'val_diff_{model}']:.3f} [{r[f'val_ci_lo_{model}']:.3f},"
                    f" {r[f'val_ci_hi_{model}']:.3f}]"
                )

            rows.append(
                [
                    str(r["setting"]),
                    str(r["n_scanners"]),
                    shares("raw"),
                    shares("whitened"),
                    f"{r['val_offset_explained_raw']:.3f} /"
                    f" {r['val_offset_explained_whitened']:.3f}",
                    _fmt(r["val_diff_offset"]),
                    _fmt(r["val_diff_cov"]),
                    ci("offset_cov"),
                    ci("mixture"),
                    _fmt(r["val_resid_scanner_acc"]),
                    f"{r['offset_rank']} / {r['offset_fraction']:.3f} /"
                    f" {r['offset_content_share']:.3f}"
                    if "offset_rank" in r
                    else "–",
                    *source,
                ]
            )
    return (
        _table(header, rows)
        + "\n\nTile-level residuals to the consensus over scanners (`summary.csv`)."
        " Energy shares (train) of E||r||^2 = N_0 split into nested scanner offsets:"
        " global per scanner, slide × scanner, region × scanner, tile; raw = identity"
        " metric, whitened = train N_0^-1. Val explained = residual energy removed by"
        " subtracting the train offsets per scanner. Δll = held-out val log-likelihood"
        " minus the exchangeable N(0, N_0) model (nats/dim, top residual subspace of"
        " train, > 0 favors scanner-aware): offset = per-scanner mean, cov ="
        " per-scanner covariance, offset+cov = both (these use the scanner label);"
        " mixture = equal-weight mixture of the offset+cov components, label"
        " marginalized. Resid. scanner acc = argmax component on val residuals"
        " (given the paired consensus). Offset span = span of the train global"
        " offsets: offset fraction = share of the residual energy there explained by"
        " the offsets; content share = share of the consensus variance there (what"
        " suppressing the span loses)."
    )


# --- content-dependent scanner offsets -------------------------------------------


def content_offsets_table(root: Path, opts: dict[str, list[str]]) -> str:
    header = [
        "setting", "scanners", "model", "CV gain raw", "val explained raw / whitened",
        "val gain over global raw / whitened", "val Δll vs global [WSI 95% CI]",
        "WSIs > 0", "content part in offset span", "bench run dir", "SIPE commit",
    ]  # fmt: skip
    rows = []
    for run in _bench_runs(root, "scorpion_content_offsets", opts.get("exclude", [])):
        source = [f"`{_rel(run.dir)}`", run.source_cells()[2]]
        summary = run.dir / "summary.csv"
        if not summary.exists():
            rows.append(["no results", *["–"] * (len(header) - 3), *source])
            continue
        for _, r in pd.read_csv(summary).iterrows():
            rows.append(
                [
                    str(r["setting"]),
                    str(r["n_scanners"]),
                    str(r["model"]),
                    _fmt(r["cv_gain_raw"]),
                    f"{r['val_explained_raw']:.3f} / {r['val_explained_whitened']:.3f}",
                    f"{r['val_gain_raw']:.3f} / {r['val_gain_whitened']:.3f}",
                    f"{r['val_dll']:.3f} [{r['val_dll_ci_lo']:.3f},"
                    f" {r['val_dll_ci_hi']:.3f}]",
                    f"{r['val_dll_wsi_pos']}/{r['n_wsi']}",
                    "–"
                    if pd.isna(r["offset_span_share"])
                    else _fmt(r["offset_span_share"]),
                    *source,
                ]
            )
    return (
        _table(header, rows)
        + "\n\nDoes a scanner's offset depend on the content? Per scanner pair (k, j),"
        " target x_k − x_j, content = the mean of the other views (no shared noise"
        " with the target); `summary.csv`, train and val only. global = constant"
        " δ_k − δ_j; linear_p = constant + ridge on the top-p content PCs (λ per pair"
        " by grouped CV on train); cluster_K = one mean per content cluster; shuffled"
        " = linear fitted on permuted content (control). Explained = share of the"
        " val pair-difference energy removed; gain = share of what the constant"
        " offsets leave that the content predicts (CV = out-of-fold on train WSIs);"
        " whitened = train N_0^-1 metric. Δll = val Gaussian log-likelihood minus the"
        " global model, top residual subspace (nats/dim). Offset span = share of the"
        " content-dependent part lying in the span of the global offsets."
    )


# --- mixture sweep (Experiment 3) ----------------------------------------------


def _mixture_rows(run: BenchRun, split: str) -> list[dict[str, Any]]:
    """Candidates of one mixture run with their selection, likelihood and gate."""
    summary = pd.read_csv(run.dir / f"summary_{split}.csv")
    selected = _read_json(run.dir / "selected.json")
    lik = pd.read_csv(run.dir / "likelihood.csv")
    gating = pd.read_csv(run.dir / "gating.csv")
    rows = []
    for _, r in summary.iterrows():
        row: dict[str, Any] = {**r.to_dict(), **selected.get(r["name"], {})}
        if r["name"] in selected:
            sel = selected[r["name"]]
            match = (lik["variant"] == sel["variant"]) & (lik["K"] == sel["K"])
            if sel["variant"] in ("G1", "G2"):  # rows of the b / g grids
                key = "beta" if sel["variant"] == "G1" else "gamma"
                match &= lik[key].sub(sel[key]).abs() < 1e-12
            if match.any():
                row.update(lik.loc[match].iloc[-1].to_dict())
            gate = gating.loc[
                (gating["variant"] == sel["variant"]) & (gating["K"] == sel["K"])
            ]
            if len(gate):
                row.update(gate.iloc[0].to_dict())
        rows.append(row)
    return rows


def mixture_table(root: Path, opts: dict[str, list[str]]) -> str:
    header = [
        "model", "α", "β", "γ", "val top-1 (α=1)", "val region probe",
        "val ll resid", "val ll x", "gate agree / views consistent",
        "test top-1", "test top-5", "test scanner probe logreg / MLP",
        "test region probe", "note", "bench run dir", "SIPE commit",
    ]  # fmt: skip
    rows = []
    for run in _bench_runs(root, "scorpion_mixture_sweep", opts.get("exclude", [])):
        source = [f"`{_rel(run.dir)}`", run.source_cells()[2]]
        if not (run.dir / "summary_val.csv").exists():
            rows.append(["no results", *["–"] * (len(header) - 3), *source])
            continue
        test = {}
        if (run.dir / "summary_test.csv").exists():
            test = {r["name"]: r for r in _mixture_rows(run, "test")}
        for r in _mixture_rows(run, "val"):
            t = test.get(r["name"], {})
            name = {"raw": "raw", "reference": "reference (eraser)"}.get(
                r["name"], f"{r.get('variant')} K={r.get('K')}"
            )
            note = []
            if r.get("region_guard_violated"):
                note.append("region guard violated")
            if r.get("at_grid_edge"):
                note.append("α at grid edge")
            gate = (
                f"{r['gate_agreement']:.3f} / {r['gate_view_consistent']:.3f}"
                if "gate_agreement" in r
                else "–"
            )
            rows.append(
                [
                    name,
                    _fmt(r.get("alpha"), 2) if "alpha" in r else "–",
                    _fmt(r.get("beta"), 2) if "beta" in r else "–",
                    _fmt(r.get("gamma"), 2) if "gamma" in r else "–",
                    f"{r['top_1_accuracy']:.3f}"
                    + (f" ({r['val_top_1_alpha1']:.3f})" if "val_top_1_alpha1" in r
                       else ""),
                    _fmt(r["region_probe_acc"]),
                    _fmt(r.get("val_ll_residual")),
                    _fmt(r.get("val_ll_x")),
                    gate,
                    _fmt(t.get("top_1_accuracy")),
                    _fmt(t.get("top_5_accuracy")),
                    " / ".join(
                        _fmt(t.get(c)) for c in ("probe_logreg_acc", "probe_mlp_acc")
                    ),
                    _fmt(t.get("region_probe_acc")),
                    ", ".join(note) or "–",
                    *source,
                ]
            )  # fmt: skip
    return (
        _table(header, rows)
        + "\n\nVal: selection split, scanner/region probes trained on train"
        " (`summary_val.csv`); α chosen by the 95%-of-best-val-top-1-gain rule, β by"
        " the val consensus likelihood, γ by the val residual likelihood"
        " (`selected.json`, `likelihood.csv`). ll = mean held-out log-likelihood,"
        " nats/dim: resid = residuals under N_k of their consensus cluster, x = single"
        " views under the mixture marginal. Gate = single-view argmax p(k|x) vs the"
        " consensus cluster, and share of tiles whose 5 views gate to one cluster"
        " (α=1 map, `gating.csv`). Test: proxy protocol, probes trained on val"
        " (`summary_test.csv`). Reference = soft-shrinkage eraser map evaluated in the"
        " same run."
    )


# --- quadratic correction (Experiment 4) ---------------------------------------


def quadratic_table(root: Path, opts: dict[str, list[str]]) -> str:
    header = [
        "model", "features", "γ", "val top-1", "val Δtop-1 vs G0",
        "val consensus MSE / G0 (train)", "val region probe", "test top-1",
        "test top-5", "test scanner probe logreg / MLP", "test region probe", "note",
        "bench run dir", "SIPE commit",
    ]  # fmt: skip
    rows = []
    for run in _bench_runs(root, "scorpion_quadratic_sweep", opts.get("exclude", [])):
        source = [f"`{_rel(run.dir)}`", run.source_cells()[2]]
        if not (run.dir / "summary_val.csv").exists():
            rows.append(["no results", *["–"] * (len(header) - 3), *source])
            continue
        selected = _read_json(run.dir / "selected.json")
        test = {}
        if (run.dir / "summary_test.csv").exists():
            test = {
                r["name"]: r
                for _, r in pd.read_csv(run.dir / "summary_test.csv").iterrows()
            }
        summary = pd.read_csv(run.dir / "summary_val.csv")
        g0_val = summary.loc[summary["name"] == "G0", "top_1_accuracy"]
        for _, r in summary.iterrows():
            sel = selected.get(r["name"], {})
            t = test.get(r["name"], {})
            name = {"raw": "raw", "reference": "reference (eraser)", "G0": "G0"}.get(
                r["name"], f"{sel.get('variant')} m={sel.get('m')}"
            )
            note = []
            if sel.get("fallback_to_g0"):
                note.append("no γ beats G0 (= G0)")
            if sel.get("region_guard_violated"):
                note.append("region guard violated")
            if sel.get("at_grid_edge"):
                note.append("γ at grid edge")
            mse = (
                f"{sel['val_mse'] / sel['val_mse_g0']:.3f}"
                f" ({sel['train_mse'] / sel['train_mse_g0']:.3f})"
                if sel
                else "–"
            )
            delta = (
                f"{r['top_1_accuracy'] - float(g0_val.iloc[0]):+.4f}"
                if len(g0_val) and r["name"] not in ("raw", "reference")
                else "–"
            )
            rows.append(
                [
                    name,
                    str(sel["n_features"]) if sel else "–",
                    _fmt(sel.get("gamma"), 3) if sel else "–",
                    _fmt(r["top_1_accuracy"]),
                    delta,
                    mse,
                    _fmt(r["region_probe_acc"]),
                    _fmt(t.get("top_1_accuracy")),
                    _fmt(t.get("top_5_accuracy")),
                    " / ".join(
                        _fmt(t.get(c)) for c in ("probe_logreg_acc", "probe_mlp_acc")
                    ),
                    _fmt(t.get("region_probe_acc")),
                    ", ".join(note) or "–",
                    *source,
                ]
            )  # fmt: skip
    return (
        _table(header, rows)
        + "\n\nVal: selection split, scanner/region probes trained on train"
        " (`summary_val.csv`); γ = ridge on the quadratic block, the largest γ whose"
        " val top-1 gain over G0 reaches 95% of the best gain (γ = inf is G0;"
        " `selected.json`). Features = number of quadratic terms (Q1: m squares, Q2:"
        " m × m products). Consensus MSE = mean squared error to the tile consensus"
        " per dim, ratio to G0 on val (train in brackets; `sweep_val.csv`). Test: proxy"
        " protocol, probes trained on val (`summary_test.csv`). G0 is fitted in the"
        " same run; reference = soft-shrinkage eraser map."
    )


# --- CGM augmentation attribution and CGM selection (Experiment 7) -------------


def _probes(r: Any) -> str:
    """Scanner probes of a summary row (Series or dict; missing = –)."""
    return " / ".join(_fmt(r.get(c)) for c in ("probe_logreg_acc", "probe_mlp_acc"))


def aug_attribution_table(root: Path, opts: dict[str, list[str]]) -> str:
    header = [
        "model", "data", "units", "views", "G0 α", "val top-1", "val Δtop-1 vs G0",
        "val top-5", "val scanner probe logreg / MLP", "val region probe",
        "rel. change", "bench run dir", "SIPE commit",
    ]  # fmt: skip
    rows = []
    for run in _bench_runs(root, "scorpion_aug_attribution", opts.get("exclude", [])):
        source = [f"`{_rel(run.dir)}`", run.source_cells()[2]]
        if not (run.dir / "summary_val.csv").exists():
            rows.append(["no results", *["–"] * (len(header) - 3), *source])
            continue
        arms = _read_json(run.dir / "arms.json")
        summary = pd.read_csv(run.dir / "summary_val.csv")
        g0 = summary.loc[summary["name"] == "G0|clean|all", "top_1_accuracy"]
        for _, r in summary.iterrows():
            arm = arms.get(r["name"], {})
            delta = (
                f"{r['top_1_accuracy'] - float(g0.iloc[0]):+.4f}"
                if len(g0) and r["name"] not in ("raw", "reference")
                else "–"
            )
            rows.append(
                [
                    arm.get("estimator", r["name"]),
                    arm.get("data", "–"),
                    str(arm.get("n_units", "–")),
                    str(arm.get("views", "–")),
                    _fmt(arm.get("alpha"), 2) if "alpha" in arm else "–",
                    _fmt(r["top_1_accuracy"], 4),
                    delta,
                    _fmt(r["top_5_accuracy"], 4),
                    _probes(r),
                    _fmt(r.get("region_probe_acc")),
                    _fmt(r.get("relative_change")),
                    *source,
                ]
            )
    return (
        _table(header, rows)
        + "\n\nSCORPION val, clean features, probes trained on train (– ="
        " `--retrieval-only`; `summary_val.csv`, arms in `arms.json`)."
        " G0 = consensus shrinkage (K = 1,"
        " α by the 95% rule on val); CGM-init = the V0-S init map (closed form, no"
        " training). Data: clean = the GAP cache's train split, other labels ="
        " augmented train caches (`sipe.features.extract`; a unit = one tile copy,"
        " views share the transform). Units: n = n tiles with one random copy each"
        " (same tiles in every arm), tiles = every tile once, all = every copy."
        " Δ vs G0|clean|all (= the G0 of the mixture sweep when α = 1). Explanatory"
        " ablation: nothing is selected across arms; the PLISM readout uses the"
        " saved `map_<estimator>_<data>_<units>.pt`. Other rows (raw, reference, V0-S,"
        " ...) are map files evaluated in the same run."
    )


def cgm_select_table(root: Path, opts: dict[str, list[str]]) -> str:
    header = [
        "model", "arm", "seed", "config", "val top-1", "val Δtop-1 vs G0",
        "val region probe", "test top-1", "test top-5",
        "test scanner probe logreg / MLP", "test region probe", "note",
        "bench run dir", "SIPE commit",
    ]  # fmt: skip
    rows = []
    for run in _bench_runs(root, "scorpion_cgm_select", opts.get("exclude", [])):
        source = [f"`{_rel(run.dir)}`", run.source_cells()[2]]
        if not (run.dir / "summary_val.csv").exists():
            rows.append(["no results", *["–"] * (len(header) - 3), *source])
            continue
        selected = _read_json(run.dir / "selected.json")
        test = {}
        if (run.dir / "summary_test.csv").exists():
            test = {
                r["name"]: r
                for _, r in pd.read_csv(run.dir / "summary_test.csv").iterrows()
            }
        summary = pd.read_csv(run.dir / "summary_val.csv")
        g0 = summary.loc[summary["name"] == "G0", "top_1_accuracy"]
        for _, r in summary.iterrows():
            t = test.get(r["name"], {})
            note = []
            if r["name"] in selected.get("fits", []) or (
                r["name"] == "G0" and selected.get("fallback_to_g0")
            ):
                note.append("selected")
                if selected.get("region_guard_violated"):
                    note.append("region guard violated")
            delta = (
                f"{r['top_1_accuracy'] - float(g0.iloc[0]):+.4f}"
                if len(g0) and r["name"] not in ("raw", "reference")
                else "–"
            )
            seed = r.get("seed")
            rows.append(
                [
                    r["name"],
                    _fmt(r.get("arm")) if isinstance(r.get("arm"), str) else "–",
                    "–" if pd.isna(seed) else str(int(seed)),
                    r["config_id"] if isinstance(r.get("config_id"), str) else "–",
                    _fmt(r["top_1_accuracy"], 4),
                    delta,
                    _fmt(r["region_probe_acc"]),
                    _fmt(t.get("top_1_accuracy"), 4),
                    _fmt(t.get("top_5_accuracy"), 4),
                    _probes(t),
                    _fmt(t.get("region_probe_acc")),
                    ", ".join(note) or "–",
                    *source,
                ]
            )
    return (
        _table(header, rows)
        + "\n\nCGM fits (`map_cgm.pt`) on the shared SCORPION-proxy protocol"
        " (`sipe.eval.cgm_select`), clean features. Val: probes trained on train"
        " (`summary_val.csv`); selection = the simplest configuration (arm, then larger"
        " ρ) whose mean val top-1 gain over G0 (fitted in the run, α = 1) reaches 95%"
        " of the best gain (`selected.json`, `candidates.csv`). Test: read once after"
        " the selection, probes trained on val (`summary_test.csv`). Config = fits"
        " that differ only by seed share an id."
    )


def loso_table(root: Path, opts: dict[str, list[str]]) -> str:
    header = [
        "held-out scanner", "model", "unseen top-1", "unseen Δ vs G0^-j [95% CI]",
        "unseen Δ vs full fit [95% CI]", "seen top-1", "seen Δ vs G0^-j",
        "bench run dir", "SIPE commit",
    ]  # fmt: skip
    rows = []

    def cell(c: pd.DataFrame, name: str, kind: str, sub: str, ci: bool) -> str:
        r = c.loc[(c["name"] == name) & (c["contrast"] == kind) & (c["subset"] == sub)]
        if r.empty:
            return "–"
        r = r.iloc[0]
        text = f"{r['diff']:+.4f}"
        return text + (f" [{r['ci_low']:+.4f}, {r['ci_high']:+.4f}]" if ci else "")

    for run in _bench_runs(root, "scorpion_loso", opts.get("exclude", [])):
        source = [f"`{_rel(run.dir)}`", run.source_cells()[2]]
        if not (run.dir / "contrasts_val.csv").exists():
            rows.append(["no results", *["–"] * (len(header) - 3), *source])
            continue
        summary = pd.read_csv(run.dir / "summary_val.csv")
        c = pd.read_csv(run.dir / "contrasts_val.csv")
        c = c.loc[c["metric"] == "top_1_accuracy"].copy()
        c["scanner"] = c["scanner"].astype(str)
        held = summary.loc[summary["held_out"] != "none"]
        scanners = list(dict.fromkeys(held["scanner"]))
        for j in [*scanners, "mean"]:
            if j == "mean":
                names = c.loc[c["scanner"] == "mean", "estimator"].unique()
                sel = [(f"{e} (mean over j)", e, held.loc[held["estimator"] == e])
                       for e in names]  # fmt: skip
            else:
                sel = [(n, n, g) for n, g in held.loc[held["scanner"] == j].groupby(
                    "name", sort=False)]  # fmt: skip
            for label, name, g in sel:
                key = "mean" if j == "mean" else name
                cj = c.loc[c["scanner"] == j] if j != "mean" else c.loc[
                    (c["scanner"] == "mean") & (c["estimator"] == name)
                ].assign(name="mean")  # fmt: skip
                top1 = g.groupby("subset")["top_1_accuracy"].mean()
                rows.append([
                    j, label, _fmt(top1.get("unseen"), 4),
                    cell(cj, key, "vs_G0", "unseen", True),
                    cell(cj, key, "vs_full", "unseen", True),
                    _fmt(top1.get("seen"), 4), cell(cj, key, "vs_G0", "seen", False),
                    *source,
                ])  # fmt: skip
    return (
        _table(header, rows)
        + "\n\nLeave-one-scanner-out on SCORPION val (`sipe.eval.loso`): each map^-j is"
        " fitted on train without scanner j (consensus over the 4 others); unseen ="
        " val scanner pairs that include j, seen = the other pairs (`summary_val.csv`)."
        " Δ vs G0^-j = minus G0 fitted the same way (K = 1, α = 1); Δ vs full fit ="
        " minus the same estimator fitted on all 5 scanners (the cost of not seeing j)."
        " Intervals: slide bootstrap over the few val WSIs (`contrasts_val.csv`);"
        " mean rows average the per-j contrasts. Val only; nothing is selected."
    )


# --- training runs (local W&B history) -----------------------------------------


def _wandb_records(path: Path):
    """Raw records of a W&B .wandb file (LevelDB-style log, 32 KiB blocks)."""
    data = path.read_bytes()
    pos, block, buf = 7, 32768, b""  # 7-byte file header
    while pos + 7 <= len(data):
        left = block - (pos % block)
        if left < 7:
            pos += left
            continue
        _, length, kind = struct.unpack("<IHB", data[pos : pos + 7])
        if kind == 0 and length == 0:
            pos += left
            continue
        payload = data[pos + 7 : pos + 7 + length]
        pos += 7 + length
        if len(payload) < length:
            break
        if kind == 1:  # full
            yield payload
        elif kind == 2:  # first
            buf = payload
        elif kind == 3:  # middle
            buf += payload
        elif kind == 4:  # last
            yield buf + payload
            buf = b""


def wandb_history(run_dir: Path) -> list[dict[str, Any]]:
    """Logged history rows of runs/<run>/, from the local .wandb file."""
    from wandb.proto import wandb_internal_pb2 as pb

    files = sorted(glob.glob(str(run_dir / "wandb" / "*run-*" / "*.wandb")))
    if not files:
        return []
    rows = []
    for raw in _wandb_records(Path(files[-1])):
        record = pb.Record()
        try:
            record.ParseFromString(raw)
        except Exception:  # truncated tail of a live run
            continue
        if record.WhichOneof("record_type") != "history":
            continue
        row = {}
        for item in record.history.item:
            key = item.key or "/".join(item.nested_key)
            try:
                row[key] = json.loads(item.value_json)
            except ValueError:
                pass
        rows.append(row)
    return rows


def last_val_mean(
    history: list[dict[str, Any]], key: str, n: int = LAST_N_VALS
) -> float | None:
    values = [r[key] for r in history if isinstance(r.get(key), (int, float))]
    return sum(values[-n:]) / len(values[-n:]) if values else None


def training_table(root: Path, opts: dict[str, list[str]]) -> str:
    cols = opts.get("cols") or list(TRAIN_COLS)
    modules = opts.get("module", [])
    min_steps = int(opts.get("min_steps", ["0"])[0])
    exclude = opts.get("exclude", [])
    header = ["run dir", "W&B", "module", "commit", "step", *cols]
    rows = []
    for run_dir in sorted(p for p in root.glob("*/") if (p / "wandb").is_dir()):
        if any(fnmatch.fnmatch(run_dir.name, pat) for pat in exclude):
            continue
        config = _read_yaml(run_dir / "config.yaml")
        model = config.get("model") or {}
        # Configs saved before 2026-10-10 have a flat model: section; name its network.
        network = (model.get("network") or {}).get("class_path", "?")
        module = str(model.get("class_path") or f"flat:{network}").rsplit(".", 1)[-1]
        if modules and module not in modules:
            continue
        history = wandb_history(run_dir)
        steps = [
            r["trainer/global_step"] for r in history if "trainer/global_step" in r
        ]
        last = max(steps, default=0)
        if last < min_steps:
            continue
        values = [last_val_mean(history, f"val/{c}") for c in cols]
        if all(v is None for v in values):
            continue
        max_steps = (config.get("trainer") or {}).get("max_steps")
        step = f"{last}/{max_steps}" if max_steps else str(last)
        source = [f"`{_rel(run_dir)}`", _wandb_id(run_dir)]
        commit = _commit(_git_txt(run_dir))
        rows.append([*source, module, commit, step, *[_fmt(v) for v in values]])
    return (
        _table(header, rows)
        + f"\n\nMean of the last {LAST_N_VALS} validations of `val/<col>`,"
        " read from the local `.wandb` file;"
        " step = last logged `trainer/global_step` / `max_steps`."
    )


TABLES = {
    "plism": plism_table,
    "plism_pairs": plism_pairs_table,
    "hest": hest_table,
    "scorpion": scorpion_table,
    "softshrink": softshrink_table,
    "content": content_table,
    "offsets": offsets_table,
    "content_offsets": content_offsets_table,
    "mixture": mixture_table,
    "quadratic": quadratic_table,
    "augattr": aug_attribution_table,
    "cgm": cgm_select_table,
    "loso": loso_table,
    "training": training_table,
}


# --- doc blocks ---------------------------------------------------------


def parse_opts(text: str) -> dict[str, list[str]]:
    opts = {}
    for token in text.split():
        key, _, value = token.partition("=")
        opts[key] = [v for v in value.split(",") if v]
    return opts


def render(text: str, root: Path = RUNS) -> str:
    """Return `text` with every generated block re-rendered from `root`."""

    def replace(m: re.Match) -> str:
        name = m["name"]
        if name not in TABLES:
            raise ValueError(f"unknown results block {name!r}; known: {sorted(TABLES)}")
        body = TABLES[name](root, parse_opts(m["opts"]))
        note = (
            "<!-- generated by `uv run python -m sipe.eval.results_tables --write`;"
            " do not edit -->"
        )
        return f"{m.group(1)}{note}\n\n{body}\n\n{m.group(5)}"

    return BLOCK_RE.sub(replace, text)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument(
        "--write", action="store_true", help="refresh the blocks in every --doc"
    )
    mode.add_argument("--check", action="store_true", help="exit 1 if a --doc is stale")
    parser.add_argument("--doc", type=Path, nargs="+", default=list(DOCS))
    parser.add_argument("--runs", type=Path, default=RUNS)
    args = parser.parse_args(argv)

    if not (args.write or args.check):
        for name, make in TABLES.items():
            print(f"## {name}\n\n{make(args.runs, {})}\n")
        return 0

    stale = False
    for doc in args.doc:
        text = doc.read_text()
        new = render(text, args.runs)
        if args.check:
            if new != text:
                print(f"{_rel(doc)} is stale: run with --write", file=sys.stderr)
                stale = True
            continue
        doc.write_text(new)
        print(f"refreshed {len(BLOCK_RE.findall(text))} blocks in {_rel(doc)}")
    return int(stale)


if __name__ == "__main__":
    raise SystemExit(main())
