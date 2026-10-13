"""Experiment 1 diagnostic (sipe.eval.content_dependence) on synthetic paired data."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from sipe.eval.content_dependence import (
    evaluate,
    fit,
    gaussian_ll,
    main,
    paired_views,
)

SCANNERS = ("AT2", "DP200", "GT450", "P1000", "Philips")


def synthetic(dependent: bool, n_slides=10, n_regions=3, n_tiles=6, dim=12, seed=0):
    """Two content modes; residual scale differs between modes only if `dependent`."""
    rng = np.random.default_rng(seed)
    centers = np.stack([np.eye(dim)[0] * 20, -np.eye(dim)[0] * 20])
    feats, rows = [], []
    for sl in range(n_slides):
        for rg in range(n_regions):
            for t in range(n_tiles):
                mode = rng.integers(2)
                c = centers[mode] + rng.normal(size=dim)
                scale = (3.0 if mode else 0.5) if dependent else 1.0
                for s in SCANNERS:
                    feats.append(c + scale * rng.normal(size=dim))
                    rows.append(
                        {"pair_id": f"slide_{sl}-sample_{rg}-tile_{t}_0",
                         "region_id": f"slide_{sl}-sample_{rg}",
                         "slide_id": f"slide_{sl}", "scanner": s}
                    )  # fmt: skip
    meta = pd.DataFrame(rows)
    meta["row"] = np.arange(len(meta))
    return np.asarray(feats), meta


def _split(feats, meta):
    val = meta["slide_id"].isin(["slide_8", "slide_9"])
    out = {}
    for name, mask in (("train", ~val), ("val", val)):
        m = meta.loc[mask].reset_index(drop=True)
        f = feats[m["row"]]
        m["row"] = np.arange(len(m))
        out[name] = (f, m)
    return out


def test_paired_views_shapes_and_consensus():
    feats, meta = synthetic(dependent=False)
    data, _ = paired_views(feats, meta, "tile_all")
    assert data.views.shape == (10 * 3 * 6, 5, 12)
    np.testing.assert_allclose(data.residuals.sum(axis=1), 0, atol=1e-9)
    region, _ = paired_views(feats, meta, "region_noP1000")
    assert region.views.shape == (30, 4, 12) and "P1000" not in region.scanners
    # Region view = mean of its tiles on that scanner.
    m = meta[(meta.region_id == "slide_0-sample_0") & (meta.scanner == "AT2")]
    np.testing.assert_allclose(region.views[0, 0], feats[m.row].mean(axis=0))


def test_gaussian_ll_matches_scipy():
    from scipy.stats import multivariate_normal

    rng = np.random.default_rng(0)
    a = rng.normal(size=(4, 4))
    cov = a @ a.T + np.eye(4)
    z = rng.normal(size=(7, 4))
    expected = multivariate_normal(np.zeros(4), cov).logpdf(z)
    np.testing.assert_allclose(gaussian_ll(z, cov), expected)


@pytest.mark.parametrize("dependent", [True, False])
def test_content_model_wins_only_when_dependent(dependent):
    feats, meta = synthetic(dependent)
    split = _split(feats, meta)
    train, _ = paired_views(*split["train"], "tile_all")
    val, _ = paired_views(*split["val"], "tile_all")
    model = fit(train, k=2, content_dim=4, residual_dim=8, n_perm=3, seed=0)
    res, _ = evaluate(model, val, 2, [0.0, 1.0])
    row = res.set_index("alpha").loc[0.0]
    assert res.set_index("alpha").loc[1.0, "diff"] == pytest.approx(0.0, abs=1e-12)
    if dependent:
        assert row["diff"] > 0.1
        assert row["diff_shuffled"] < row["diff"] / 5
        assert row["energy_mse_ratio"] < 0.5
        assert row["heterogeneity"] > 0.5
        # Synthetic effect is pure scale: the scale-only control gets most of it.
        assert row["diff_scale"] > 0.8 * row["diff"]
    else:
        assert abs(row["diff"]) < 0.05
        assert row["heterogeneity"] < 0.2


def test_main_writes_outputs(tmp_path, monkeypatch):
    feats, meta = synthetic(dependent=True)
    for name, (f, m) in _split(feats, meta).items():
        np.savez(tmp_path / f"features_{name}.npz", gap=f.astype(np.float32))
        m.to_csv(tmp_path / f"meta_{name}.csv", index=False)
    import sipe.eval.content_dependence as cd

    runs = tmp_path / "runs"
    monkeypatch.setattr(
        cd,
        "new_bench_run_dir",
        lambda tag: (runs / tag).mkdir(parents=True) or runs / tag,
    )
    run_dir = main(
        ["--tag", "t", "--features-from", str(tmp_path), "--ks", "2",
         "--content-dim", "4", "--residual-dim", "8", "--folds", "2",
         "--n-perm", "2", "--n-boot", "50", "--settings", "tile_all", "region_all"]
    )  # fmt: skip
    summary = pd.read_csv(run_dir / "summary.csv")
    assert list(summary["setting"]) == ["tile_all", "region_all"]
    stats = pd.read_csv(run_dir / "cluster_stats.csv")
    assert {"nuisance_share", "share_P1000", "overlap_global_top5"} <= set(
        stats.columns
    )
    assert (pd.read_csv(run_dir / "likelihood.csv")["protocol"] == "cv").any()
