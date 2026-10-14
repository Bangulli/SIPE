"""Scanner-offset diagnostic (sipe.eval.scanner_offsets) on synthetic paired data."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from sipe.eval import results_tables as rt
from sipe.eval.content_dependence import second_moment
from sipe.eval.scanner_offsets import (
    ResidualModels,
    level_moments,
    main,
    offset_subspace,
    tile_views,
)
from sipe.eval.soft_shrinkage_sweep import slide_to_mean_deltas

SCANNERS = ("AT2", "DP200", "GT450", "P1000", "Philips")


def synthetic(offset: float, n_slides=10, n_regions=3, n_tiles=6, dim=12, seed=0):
    """Content + per-scanner offset (global + slide-specific) + isotropic noise."""
    rng = np.random.default_rng(seed)
    glob = offset * rng.normal(size=(len(SCANNERS), dim))
    feats, rows = [], []
    for sl in range(n_slides):
        slide_off = 0.5 * offset * rng.normal(size=(len(SCANNERS), dim))
        for rg in range(n_regions):
            for t in range(n_tiles):
                c = 5 * rng.normal(size=dim)
                for k, s in enumerate(SCANNERS):
                    feats.append(c + glob[k] + slide_off[k] + rng.normal(size=dim))
                    rows.append(
                        {"pair_id": f"slide_{sl}-sample_{rg}-tile_{t}_0",
                         "region_id": f"slide_{sl}-sample_{rg}",
                         "slide_id": f"slide_{sl}", "scanner": s}
                    )  # fmt: skip
    meta = pd.DataFrame(rows).sample(frac=1.0, random_state=seed)  # unordered rows
    meta["row"] = np.arange(len(meta))
    feats = np.asarray(feats)[meta.index.to_numpy()]
    return feats, meta.reset_index(drop=True)


def _split(feats, meta):
    val = meta["slide_id"].isin(["slide_8", "slide_9"])
    out = {}
    for name, mask in (("train", ~val), ("val", val)):
        m = meta.loc[mask].reset_index(drop=True)
        f = feats[m["row"]]
        m["row"] = np.arange(len(m))
        out[name] = (f, m)
    return out


def test_tile_views_regions_follow_units():
    feats, meta = synthetic(offset=1.0)
    data, region = tile_views(feats, meta, "tile_all")
    assert data.views.shape == (180, 5, 12)
    assert all(r.split("-")[0] == s for r, s in zip(region, data.slide, strict=True))
    with pytest.raises(ValueError):
        tile_views(feats, meta, "region_all")


def test_level_moments_sum_and_match_eraser_deltas():
    feats, meta = synthetic(offset=1.0)
    data, region = tile_views(feats, meta, "tile_all")
    r = data.residuals
    m = level_moments(r, data.slide, region)
    parts = m["global"] + m["slide"] + m["region"] + m["tile"]
    np.testing.assert_allclose(parts, second_moment(r), atol=1e-10)
    # Slide x scanner means = the eraser's slide-level deltas (equal tiles per slide).
    deltas = slide_to_mean_deltas(feats, meta).astype(np.float64)
    np.testing.assert_allclose(
        m["global"] + m["slide"], deltas.T @ deltas / len(deltas), rtol=1e-4, atol=1e-5
    )
    for level in ("global", "slide", "region", "tile"):
        assert np.linalg.eigvalsh(m[level]).min() > -1e-9


@pytest.mark.parametrize("offset", [0.0, 2.0])
def test_offset_share_and_likelihood_track_offsets(offset):
    split = _split(*synthetic(offset=offset))
    train, _ = tile_views(*split["train"], "tile_all")
    val, _ = tile_views(*split["val"], "tile_all")
    m = level_moments(
        train.residuals, train.slide, tile_views(*split["train"], "tile_all")[1]
    )
    share = np.trace(m["global"]) / np.trace(
        m["global"] + m["slide"] + m["region"] + m["tile"]
    )
    models = ResidualModels(train.residuals)
    diff = models.per_view(val.residuals)
    if offset == 0:
        assert share < 0.02
        assert abs(diff["offset"].mean()) < 0.02
    else:
        assert share > 0.5
        assert diff["offset"].mean() > 0.1
        # Label marginalized: at most log(S) / d below the labelled model.
        bound = diff["offset_cov"].mean() - np.log(5) / 12
        assert bound - 0.01 < diff["mixture"].mean() <= diff["offset_cov"].mean()
        assert models.scanner_accuracy(val.residuals) > 0.9
        space = offset_subspace(
            train.residuals.mean(axis=0),
            second_moment(train.residuals),
            train.consensus,
        )
        assert space["offset_rank"] == 4 and space["offset_fraction"] > 0.8
        assert 0 < space["offset_content_share"] < 4 / 12 + 0.2


def test_main_writes_outputs_and_table(tmp_path, monkeypatch):
    split = _split(*synthetic(offset=1.0))
    cache = tmp_path / "cache"
    cache.mkdir()
    for name, (f, m) in split.items():
        np.savez(cache / f"features_{name}.npz", gap=f.astype(np.float32))
        m.to_csv(cache / f"meta_{name}.csv", index=False)
    bench = tmp_path / "runs" / "bench"
    monkeypatch.setattr(
        "sipe.eval.scanner_offsets.new_bench_run_dir",
        lambda tag: (
            (bench / f"stamp_{tag}").mkdir(parents=True) or bench / f"stamp_{tag}"
        ),
    )
    run = main(
        ["--tag", "t", "--features-from", str(cache), "--residual-dim", "6",
         "--n-boot", "50"]
    )  # fmt: skip
    for name in ("summary", "energy", "likelihood", "scanner_stats"):
        assert (run / f"{name}.csv").exists()
    summary = pd.read_csv(run / "summary.csv")
    assert list(summary["setting"]) == ["tile_all", "tile_noP1000"]
    assert list(summary["n_scanners"]) == [5, 4]
    stats = pd.read_csv(run / "scanner_stats.csv")
    for _, g in stats.groupby("setting"):
        total = g["offset_energy_share"].sum() + g["within_energy_share"].sum()
        assert total == pytest.approx(1.0)
    table = rt.offsets_table(tmp_path / "runs", {})
    assert "tile_noP1000" in table and "no results" not in table
