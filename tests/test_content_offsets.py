"""Content-dependent scanner shifts (sipe.eval.content_offsets), synthetic data."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from sipe.eval import results_tables as rt
from sipe.eval.content_offsets import cluster_predict, main, ridge_predict

SCANNERS = ("AT2", "DP200", "GT450", "P1000", "Philips")


def synthetic(content_shift: float, n_slides=12, n_tiles=40, dim=10, seed=0):
    """Content + per-scanner offset delta_k + content_shift * D_k c + scanner noise.

    The noise scale differs per scanner, the case where regressing residuals to the
    full consensus on the consensus finds a spurious content dependence.
    """
    rng = np.random.default_rng(seed)
    delta = rng.normal(size=(len(SCANNERS), dim))
    gain = content_shift * rng.normal(size=(len(SCANNERS), dim, dim)) / np.sqrt(dim)
    noise = np.array([0.3, 0.6, 1.0, 1.5, 2.0])
    feats, rows = [], []
    for sl in range(n_slides):
        for t in range(n_tiles):
            c = 2 * rng.normal(size=dim)
            for k, s in enumerate(SCANNERS):
                x = c + delta[k] + gain[k] @ c + noise[k] * rng.normal(size=dim)
                feats.append(x)
                rows.append(
                    {"pair_id": f"slide_{sl}-sample_{t // 10}-tile_{t}_0",
                     "region_id": f"slide_{sl}-sample_{t // 10}",
                     "slide_id": f"slide_{sl}", "scanner": s}
                )  # fmt: skip
    meta = pd.DataFrame(rows)
    meta["row"] = np.arange(len(meta))
    return np.asarray(feats), meta


def _cache(tmp_path, content_shift):
    feats, meta = synthetic(content_shift)
    val = meta["slide_id"].isin(["slide_10", "slide_11"])
    cache = tmp_path / "cache"
    cache.mkdir()
    for name, mask in (("train", ~val), ("val", val)):
        m = meta.loc[mask].reset_index(drop=True)
        np.savez(cache / f"features_{name}.npz", gap=feats[m["row"]].astype(np.float32))
        m["row"] = np.arange(len(m))
        m.to_csv(cache / f"meta_{name}.csv", index=False)
    return cache


def _run(tmp_path, monkeypatch, content_shift):
    cache = _cache(tmp_path, content_shift)
    bench = tmp_path / "runs" / "bench"
    monkeypatch.setattr(
        "sipe.eval.content_offsets.new_bench_run_dir",
        lambda tag: (
            (bench / f"stamp_{tag}").mkdir(parents=True) or bench / f"stamp_{tag}"
        ),
    )
    return main(
        ["--tag", "t", "--features-from", str(cache), "--dims", "4", "10",
         "--ks", "2", "--report-k", "3", "--content-dim", "5", "--residual-dim", "6",
         "--folds", "3", "--n-boot", "50"]
    )  # fmt: skip


def test_ridge_and_cluster_predict():
    rng = np.random.default_rng(0)
    x = rng.normal(size=(200, 3))
    y = 1.0 + x @ rng.normal(size=(3, 4))
    exact = ridge_predict(x, y, x[:5], [0.0, 1e6])
    np.testing.assert_allclose(exact[0], y[:5], atol=1e-8)
    np.testing.assert_allclose(exact[1], np.broadcast_to(y.mean(0), (5, 4)), atol=1e-3)
    lab = np.array([0, 0, 1, 1])
    pred = cluster_predict(lab, np.arange(4.0)[:, None], np.array([1, 0, 2]), 3)
    np.testing.assert_allclose(pred.ravel(), [2.5, 0.5, 1.5])  # 2 empty: overall mean


@pytest.mark.parametrize("content_shift", [0.0, 0.5])
def test_gain_tracks_content_dependence(tmp_path, monkeypatch, content_shift):
    run = _run(tmp_path, monkeypatch, content_shift)
    s = pd.read_csv(run / "summary.csv").set_index(["setting", "model"])
    s = s.loc["tile_all"]
    assert s.loc["global", "val_gain_raw"] == pytest.approx(0.0)
    assert s.loc["global", "val_explained_raw"] > 0.2  # constant offsets
    shuffled = s.loc["linear_p10_shuffled"]
    assert shuffled["val_gain_raw"] < 0.02
    if content_shift == 0:
        # Scanner-specific noise alone: no content dependence found (no leak).
        assert abs(s.loc["linear_p10", "val_gain_raw"]) < 0.02
        assert s.loc["linear_p10", "val_dll"] < 0.02
    else:
        assert s.loc["linear_p10", "val_gain_raw"] > 0.2
        assert s.loc["linear_p10", "val_gain_raw"] > s.loc["linear_p4", "val_gain_raw"]
        assert s.loc["linear_p10", "val_dll"] > 0.05
        assert s.loc["linear_p10", "cv_gain_raw"] > 0.2
    for name in ("per_scanner", "per_cluster", "ridge"):
        assert len(pd.read_csv(run / f"{name}.csv"))
    ridge = pd.read_csv(run / "ridge.csv")
    assert ridge.groupby(["setting", "pair", "model"])["selected"].sum().eq(1).all()
    table = rt.content_offsets_table(tmp_path / "runs", {})
    assert "linear_p10" in table and "tile_noP1000" in table
    assert "no results" not in table
