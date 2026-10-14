"""Experiment 4 quadratic correction (estimator and sweep driver)."""

from __future__ import annotations

import json

import numpy as np
import pandas as pd
import pytest
import torch
from test_mixture import _write_cache, synthetic_views

from sipe.bench.posthoc import load_posthoc_map
from sipe.estimators.mixture import Mixture, Moments
from sipe.estimators.quadratic import (
    QuadraticFeatures,
    QuadraticFit,
    QuadraticShrinkageMap,
)


def _g0(views, ridge):
    model = Mixture.build(
        Moments.fit(views, np.zeros(len(views), int), 1), "G0", ridge=ridge
    )
    return model, model.mu[0], model.C[0], model.N[0]


def _fit(views, variant, m=2, ridge=1e-6):
    model, mu, C, N = _g0(views, ridge)
    feat = QuadraticFeatures.build(variant, C, N, m, seed=0)
    return model, QuadraticFit.fit(views, feat, mu, C, N, batch=97)


@pytest.mark.parametrize("variant", ["Q1", "Q2", "Q2-rand"])
def test_linear_limit_is_g0(variant):
    views, _ = synthetic_views()
    model, fit = _fit(views, variant)
    W_u, W_q = fit.solve(np.inf)
    np.testing.assert_allclose(W_u, model.operators(1.0)[0], atol=1e-10)
    assert not W_q.any()
    W_u, W_q = fit.solve(1e10)
    np.testing.assert_allclose(W_u, model.operators(1.0)[0], atol=1e-6)
    assert np.abs(W_q).max() < 1e-6


def test_small_gamma_is_least_squares_on_phi():
    """No ridge: gamma -> 0 is least squares of the consensus on [1, u, q]."""
    views, _ = synthetic_views()
    _, fit = _fit(views, "Q2", ridge=0.0)
    x = views.reshape(-1, views.shape[-1])
    target = np.repeat(views.mean(axis=1), views.shape[1], axis=0)
    design = np.hstack([np.ones((len(x), 1)), x, fit.features(x - fit.mu)])
    coef, *_ = np.linalg.lstsq(design, target, rcond=None)
    np.testing.assert_allclose(fit.predict(x, 1e-12), design @ coef, atol=1e-6)


def test_directions_unit_variance_and_nuisance_first():
    views, _ = synthetic_views()
    _, mu, C, N = _g0(views, 1e-6)
    for variant in ("Q1", "Q2", "Q2-rand"):
        feat = QuadraticFeatures.build(variant, C, N, 3, seed=0)
        dirs = [feat.nuisance] + ([] if feat.content is None else [feat.content])
        for d in dirs:
            np.testing.assert_allclose(
                np.einsum("ij,jk,ik->i", d, C + N, d), 1.0, rtol=1e-8
            )
    # Synthetic nuisance lives on axes 1 and 2, content on axis 0.
    feat = QuadraticFeatures.build("Q2", C, N, 2)
    assert set(np.abs(feat.nuisance).argmax(axis=1)) == {1, 2}
    assert np.abs(feat.content[0]).argmax() == 0


def test_q2_beats_g0_on_content_dependent_nuisance():
    views, _ = synthetic_views(n=800, dependent=True)
    held, _ = synthetic_views(n=800, seed=1, dependent=True)
    model, fit = _fit(views, "Q2", m=2)
    x = held.reshape(-1, held.shape[-1])
    target = np.repeat(held.mean(axis=1), held.shape[1], axis=0)
    # Held-out error on the two nuisance axes (where the mode decides the noise).
    mse_g0 = ((fit.predict(x, np.inf) - target)[:, 1:3] ** 2).mean()
    mse_q2 = ((fit.predict(x, 1e-3) - target)[:, 1:3] ** 2).mean()
    assert mse_q2 < 0.9 * mse_g0


@pytest.mark.parametrize("variant", ["Q1", "Q2"])
def test_module_matches_numpy(variant):
    views, _ = synthetic_views()
    _, fit = _fit(views, variant, m=3)
    x = views.reshape(-1, views.shape[-1])
    fmap = QuadraticShrinkageMap.from_fit(fit, 0.1).double()
    out = fmap(torch.from_numpy(x)).numpy()
    # Buffers are float32 (the map runs in float32 in the benchmarks).
    np.testing.assert_allclose(out, fit.predict(x, 0.1), atol=2e-4)


def test_main_writes_outputs_and_loadable_maps(tmp_path, monkeypatch):
    import sipe.eval.quadratic_sweep as qs

    _write_cache(tmp_path)
    runs = tmp_path / "runs"
    monkeypatch.setattr(
        qs,
        "new_bench_run_dir",
        lambda tag: (runs / "bench" / tag).mkdir(parents=True) or runs / "bench" / tag,
    )
    run_dir = qs.main(
        ["--tag", "t", "--features-from", str(tmp_path), "--ms", "2", "3",
         "--gammas", "0.01", "1", "--reference-map", "none", "--device", "-1"]
    )  # fmt: skip
    selected = json.loads((run_dir / "selected.json").read_text())
    names = [f"{v}_m{m}" for v in ("Q1", "Q2", "Q2-rand") for m in (2, 3)]
    assert list(selected) == names
    assert selected["Q2_m3"]["n_features"] == 9
    sweep = pd.read_csv(run_dir / "sweep_val.csv")
    assert len(sweep) == 6 * 3 and np.isinf(sweep["gamma"]).sum() == 6
    test_names = pd.read_csv(run_dir / "summary_test.csv")["name"].tolist()
    assert test_names == ["raw", "G0", *names]
    loaded = load_posthoc_map(run_dir / "map_Q2_m2.pt")
    assert loaded.kind == "module" and loaded.provenance["file"]["variant"] == "Q2"
    x = torch.from_numpy(np.load(tmp_path / "features_val.npz")["gap"][:7])
    assert loaded.map(x).shape == (7, 10)
    from sipe.eval.results_tables import quadratic_table

    table = quadratic_table(runs, {})
    assert "| Q2 m=2 |" in table and "| G0 |" in table
