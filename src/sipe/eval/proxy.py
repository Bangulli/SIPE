"""SCORPION-proxy evaluation shared by every post-hoc estimator driver.

One protocol for the eraser, mixture, quadratic and CGM drivers, so their rows are
comparable (the metrics themselves are in `sipe.eval.scorpion_retrieval`):

- `evaluate`: cross-scanner retrieval (cosine, top-k), linear/MLP scanner probes
  (trained on `probe_x`, scored on the evaluated split), the leave-one-scanner-out
  region probe (content guard) and the mean relative change of the embedding;
- `summarize`: `evaluate` for several maps on val (probes trained on a train
  subsample) or test (probes trained on val), the split conventions of the sweeps;
- `select_cv`: the MIDL rule on CV folds (best fold-mean top-1; ties, i.e. a per-fold
  paired difference to the best that is not resolved, go to the map that changes the
  embedding least);
- `select_by_gain`: the 95% rule (ISBI protocol). First row, in a given simplicity
  order, whose val top-1 gain over a base reaches `gain_frac` x the best gain;
  `select_lambda` is the eraser's form (order = lambda, base = lambda 0).

Moved out of `sipe.eval.soft_shrinkage_sweep` / `sipe.eval.mixture_sweep` without
behavior changes (2026-10-15); those modules re-export the old names.
"""

from __future__ import annotations

import json
from collections.abc import Callable, Mapping
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from sipe.data.folds import REFIT, check_fold, group_folds
from sipe.eval.scorpion_retrieval import TOPK, region_probe, retrieval, scanner_probes

RETRIEVAL_COLS = ("cosine_similarity", *(f"top_{k}_accuracy" for k in TOPK))


def load_cache(
    cache: Path, splits: tuple[str, ...], pooling: str = "gap"
) -> tuple[dict[str, np.ndarray], dict[str, pd.DataFrame]]:
    """features_<split>.npz[pooling] (float32) and meta_<split>.csv of a cache dir."""
    feats, metas = {}, {}
    for split in splits:
        feats[split] = np.load(cache / f"features_{split}.npz")[pooling].astype(
            np.float32
        )
        metas[split] = pd.read_csv(cache / f"meta_{split}.csv")
    return feats, metas


CACHE_SPLITS = ("train", "val", "test")


def load_cv_cache(
    cache: Path, n_folds: int, fold: int, pooling: str = "gap", seed: int = 42
) -> tuple[dict[str, np.ndarray], dict[str, pd.DataFrame]]:
    """One CV fold of a cache: its train/val/test splits pooled, then re-split by WSI.

    Same folds as `SCORPIONDataModule(n_folds=, fold=)` (`sipe.data.folds`). Returns
    {"train", "val"}, each meta with `row` renumbered into its own feature array (the
    eraser's delta builders index features by `row`). `fold = REFIT` (-1): train =
    every WSI, no val.
    """
    check_fold(fold, n_folds)
    feats, metas = load_cache(cache, CACHE_SPLITS, pooling)
    x = np.concatenate([feats[s] for s in CACHE_SPLITS])
    parts, offset = [], 0
    for split in CACHE_SPLITS:
        parts.append(metas[split].assign(row=metas[split]["row"] + offset))
        offset += len(feats[split])
    meta = pd.concat(parts, ignore_index=True)
    if len(meta) != len(x) or not meta["row"].sort_values().eq(np.arange(len(x))).all():
        raise ValueError(f"{cache}: meta rows do not index the pooled features.")
    slides = meta["slide_id"].astype(str)
    folds = slides.map(group_folds(slides, n_folds, seed))
    keep = {"train": folds != fold}
    if fold != REFIT:
        keep["val"] = folds == fold
    out_x, out_meta = {}, {}
    for name, rows in keep.items():
        sub = meta.loc[rows.to_numpy()].reset_index(drop=True)
        out_x[name] = x[sub["row"].to_numpy()]
        out_meta[name] = sub.assign(row=np.arange(len(sub)))
    return out_x, out_meta


def cache_encoder(cache: Path) -> str:
    """Encoder of a feature cache (`provenance.json` -> checkpoint.encoder).

    Map files fitted on the cache record it, so `--posthoc` runs the same backbone.
    Caches written before the field existed are H0-mini.
    """
    path = Path(cache) / "provenance.json"
    if not path.exists():
        return "h0-mini"
    checkpoint = json.loads(path.read_text()).get("checkpoint") or {}
    return str(checkpoint.get("encoder", "h0-mini"))


def check_map_matches_cache(
    path: Path, encoder: str, pooling: str, cache: Path
) -> None:
    """Refuse a map file fitted on another backbone or pooling than the cache."""
    from sipe.bench.posthoc import load_posthoc_map

    loaded = load_posthoc_map(path)
    if (loaded.encoder, loaded.pooling) != (encoder, pooling):
        raise ValueError(
            f"{path}: map for {loaded.encoder}/{loaded.pooling}, but the cache {cache} "
            f"is {encoder}/{pooling}."
        )


def subsample(meta: pd.DataFrame, n: int | None, seed: int) -> np.ndarray:
    if n is None or n >= len(meta):
        return np.arange(len(meta))
    rng = np.random.default_rng(seed)
    return np.sort(rng.choice(len(meta), size=n, replace=False))


@torch.no_grad()
def apply_map(
    fmap: torch.nn.Module, x: np.ndarray, device: torch.device, batch: int = 4096
) -> np.ndarray:
    fmap = fmap.to(device).float()
    out = [
        fmap(torch.from_numpy(x[i : i + batch]).to(device).float()).cpu().numpy()
        for i in range(0, len(x), batch)
    ]
    return np.concatenate(out)


def relative_change(x: np.ndarray, raw: np.ndarray) -> float:
    """Mean over rows of ||x - raw|| / ||raw|| (how much a map moves the embedding)."""
    diff = x.astype(np.float64) - raw
    return float(np.mean(np.linalg.norm(diff, axis=1) / np.linalg.norm(raw, axis=1)))


def retrieval_metrics(x: np.ndarray, meta: pd.DataFrame, device) -> dict[str, float]:
    pairs = retrieval(x, meta, device)
    return {c: float(pairs[c].mean()) for c in RETRIEVAL_COLS}


def evaluate(
    eval_x: np.ndarray,
    eval_meta: pd.DataFrame,
    probe_x: np.ndarray,
    probe_y: np.ndarray,
    raw_eval_x: np.ndarray,
    device: torch.device,
) -> dict[str, float]:
    pairs = retrieval(eval_x, eval_meta, device)
    return {
        **{c: float(pairs[c].mean()) for c in RETRIEVAL_COLS},
        **scanner_probes(probe_x, probe_y, eval_x, eval_meta["domain"].to_numpy()),
        "region_probe_acc": region_probe(eval_x, eval_meta)["region_probe_acc"],
        "relative_change": relative_change(eval_x, raw_eval_x),
    }


def summarize(
    maps: Mapping[str, torch.nn.Module | None],
    feats: Mapping[str, np.ndarray],
    metas: Mapping[str, pd.DataFrame],
    split: str,
    device: torch.device,
    probe_train_max: int | None = 20000,
    log: Callable[[str, dict], None] | None = None,
) -> pd.DataFrame:
    """Full proxy metrics of every map (None = raw features) on `split`.

    val: probes trained on a seeded train subsample (`probe_train_max`, the sweeps'
    convention); test: probes trained on val (the proxy protocol).
    """
    if split == "val":
        rows = subsample(metas["train"], probe_train_max, seed=0)
        probe_raw = feats["train"][rows]
        probe_y = metas["train"]["domain"].to_numpy()[rows]
    elif split == "test":
        probe_raw, probe_y = feats["val"], metas["val"]["domain"].to_numpy()
    else:
        raise ValueError(f"split must be val or test, not {split!r}.")
    out = []
    for name, fmap in maps.items():

        def tf(x, fmap=fmap):
            return x if fmap is None else apply_map(fmap, x, device)

        metrics = evaluate(
            tf(feats[split]), metas[split], tf(probe_raw), probe_y, feats[split],
            device,
        )  # fmt: skip
        out.append({"name": name, **metrics})
        if log is not None:
            log(name, metrics)
    return pd.DataFrame(out)


def select_by_gain(rows: pd.DataFrame, gain_frac: float, base_top1: float) -> pd.Series:
    """First row (rows are in simplicity order) whose val top-1 gain over
    `base_top1` is >= gain_frac x the best gain."""
    gain = rows["top_1_accuracy"] - base_top1
    # Tolerance so that gain_frac = 1 keeps the best row despite float rounding.
    return rows.loc[gain >= gain_frac * gain.max() - 1e-12].iloc[0]


def select_lambda(rows: pd.DataFrame, gain_frac: float) -> pd.Series:
    """Smallest lambda whose val top-1 gain over lambda = 0 is >= gain_frac x best."""
    rows = rows.sort_values("lambda")
    raw = float(rows.loc[rows["lambda"] == 0.0, "top_1_accuracy"].iloc[0])
    return select_by_gain(rows, gain_frac, raw)


def select_cv(
    folds: pd.DataFrame, key: str, ci: float = 0.95
) -> tuple[object, list, object]:
    """MIDL selection on CV folds: (selected key, tied keys, best key).

    `folds` has one row per (key, fold) with `top_1_accuracy` and `relative_change`.
    Best = highest fold-mean top-1. A candidate ties with the best when its per-fold
    paired difference to the best is not resolved: the Student t interval (`ci`, df =
    folds - 1) of the mean difference contains 0 (zero spread: tie iff the difference is
    0). Among the tied (the best included), the smallest fold-mean `relative_change`
    wins, i.e. the map that changes the embedding least.
    """
    from scipy.stats import t as student_t

    top1 = folds.pivot(index="fold", columns=key, values="top_1_accuracy")
    if top1.isna().any().any():
        raise ValueError("select_cv: every candidate needs a score on every fold.")
    change = folds.groupby(key, sort=False)["relative_change"].mean()
    best = top1.mean().idxmax()
    n = len(top1)
    half = student_t.ppf(0.5 + ci / 2, n - 1) if n > 1 else np.inf
    tied = []
    for cand in top1.columns:
        delta = top1[cand] - top1[best]
        sd = float(delta.std(ddof=1)) if n > 1 else np.nan
        mean = float(delta.mean())
        if mean == 0.0 or (sd > 0 and mean + half * sd / np.sqrt(n) >= 0):
            tied.append(cand)
    selected = min(tied, key=lambda c: (change[c], str(c)))
    return selected, tied, best
