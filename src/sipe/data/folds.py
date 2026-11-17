"""Grouped k-fold assignment of SCORPION WSIs (MIDL selection protocol).

One assignment shared by the datamodules (`SCORPIONDataModule(n_folds=, fold=)`) and
the feature caches (`sipe.eval.proxy.load_cv_cache`), so a fit and the retrieval
that scores it see the same folds. The groups (WSIs, `slide_id`) are sorted, shuffled
with a seeded torch generator (the same RNG as the single split), and dealt round
robin: fold sizes differ by at most one group (48 WSIs / 4 folds = 12 each).

`fold = REFIT` (-1) is the final fit on every group, after the selection.
"""

from __future__ import annotations

from collections.abc import Iterable

import torch

REFIT = -1


def group_folds(groups: Iterable[str], n_folds: int, seed: int = 42) -> dict[str, int]:
    """Fold index (0 .. n_folds - 1) of every distinct group."""
    unique = sorted({str(g) for g in groups})
    if not 2 <= n_folds <= len(unique):
        raise ValueError(f"n_folds must be in [2, {len(unique)}], not {n_folds}.")
    order = torch.randperm(len(unique), generator=torch.Generator().manual_seed(seed))
    return {unique[i]: rank % n_folds for rank, i in enumerate(order.tolist())}


def check_fold(fold: int, n_folds: int) -> None:
    if not (fold == REFIT or 0 <= fold < n_folds):
        raise ValueError(
            f"fold must be in [0, {n_folds}) or {REFIT} (refit), not {fold}."
        )
