from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
import torch

from sipe.baselines.soft_delta_legacy import PairedDeltaFitter
from sipe.eval.soft_shrinkage_sweep import (
    ERASER_CONFIG,
    SOURCE,
    fit_family,
    select_lambda,
    slide_to_mean_deltas,
)

SCANNERS = ("A", "B", "C")


def synthetic(n_slides: int = 12, n_tiles: int = 30, dim: int = 16, seed: int = 0):
    """Content varies per tile; each scanner adds a fixed shift along one axis."""
    rng = np.random.default_rng(seed)
    shifts = {s: np.eye(dim)[0] * 3.0 * i for i, s in enumerate(SCANNERS)}
    feats, rows = [], []
    for slide in range(n_slides):
        content = rng.normal(size=(n_tiles, dim))
        for scanner in SCANNERS:
            feats.append(content + shifts[scanner])
            rows += [
                {"slide_id": f"slide_{slide}", "scanner": scanner, "tile": t}
                for t in range(n_tiles)
            ]
    meta = pd.DataFrame(rows)
    meta["row"] = np.arange(len(meta))
    return np.concatenate(feats).astype(np.float32), meta


def test_deltas_sum_to_zero_per_slide():
    feats, meta = synthetic()
    deltas = slide_to_mean_deltas(feats, meta)
    assert deltas.shape == (12 * len(SCANNERS), feats.shape[1])
    per_slide = deltas.reshape(12, len(SCANNERS), -1).sum(axis=1)
    np.testing.assert_allclose(per_slide, 0.0, atol=1e-4)


def test_lambda_zero_is_identity():
    feats, meta = synthetic()
    family = fit_family(feats, meta, torch.device("cpu"))
    x = torch.from_numpy(feats[:50])
    torch.testing.assert_close(family.make_eraser(0.0)(x), x, atol=1e-3, rtol=1e-3)


@pytest.mark.parametrize("lam", [0.1, 10.0, 1000.0])
def test_family_matches_direct_solve(lam):
    feats, meta = synthetic()
    family = fit_family(feats, meta, torch.device("cpu"))
    fitter = PairedDeltaFitter(x_dim=feats.shape[1], dtype=torch.float32)
    fitter.update_x(torch.from_numpy(feats))
    fitter.update_delta_source(
        SOURCE.name, torch.from_numpy(slide_to_mean_deltas(feats, meta))
    )
    direct = fitter.make_soft_eraser(lam=lam, delta_sources=[SOURCE], **ERASER_CONFIG)
    torch.testing.assert_close(
        family.make_eraser(lam).P, direct.P, atol=1e-3, rtol=1e-3
    )


def test_large_lambda_removes_scanner_shift():
    feats, meta = synthetic()
    family = fit_family(feats, meta, torch.device("cpu"))
    x = torch.from_numpy(feats)
    scanner = meta["scanner"].to_numpy()

    def gap(erased: torch.Tensor) -> float:
        a = erased[torch.from_numpy(scanner == "A")].mean(0)
        c = erased[torch.from_numpy(scanner == "C")].mean(0)
        return float((a - c).norm())

    raw, erased = gap(x), gap(family.make_eraser(1000.0)(x))
    assert erased < 0.05 * raw
    # Content directions (orthogonal to the shift) are kept.
    kept = family.make_eraser(1000.0)(x)[:, 1:]
    assert float((kept - x[:, 1:]).norm() / x[:, 1:].norm()) < 0.1


def test_select_lambda_smallest_reaching_gain_fraction():
    rows = pd.DataFrame(
        {
            "lambda": [0.0, 0.1, 1.0, 10.0, 100.0],
            "top_1_accuracy": [0.80, 0.90, 0.96, 0.99, 1.00],
        }
    )
    # Best gain 0.20: 0.95 of it needs top-1 >= 0.99.
    assert select_lambda(rows, 0.95)["lambda"] == 10.0
    assert select_lambda(rows, 0.5)["lambda"] == 0.1
    assert select_lambda(rows, 1.0)["lambda"] == 100.0
