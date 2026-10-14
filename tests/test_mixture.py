"""Experiment 3 mixture estimator (sipe.estimators.mixture, sipe.eval.mixture_sweep)."""

from __future__ import annotations

import json

import numpy as np
import pandas as pd
import pytest
import torch

from sipe.bench.posthoc import load_posthoc_map
from sipe.estimators.mixture import (
    ContentClusters,
    Gate,
    Mixture,
    MixtureShrinkageMap,
    Moments,
    gaussian_ll,
    heldout_likelihoods,
)

SCANNERS = ("AT2", "DP200", "GT450", "P1000", "Philips")


def synthetic_views(n=400, dim=10, seed=0, dependent=True):
    """Two content modes; the nuisance lies along a mode-specific axis if dependent."""
    rng = np.random.default_rng(seed)
    mode = rng.integers(2, size=n)
    content = np.where(mode[:, None] == 1, 8.0, -8.0) * np.eye(dim)[0] + rng.normal(
        size=(n, dim)
    )
    axis = np.where(mode == 1, 1, 2) if dependent else np.full(n, 1)
    noise = 0.3 * rng.normal(size=(n, len(SCANNERS), dim))
    noise[np.arange(n), :, axis] += 3.0 * rng.normal(size=(n, len(SCANNERS)))
    return content[:, None] + noise, mode


def test_g0_alpha1_is_affine_regression_on_consensus():
    views, _ = synthetic_views()
    moments = Moments.fit(views, np.zeros(len(views), int), 1)
    model = Mixture.build(moments, "G0", ridge=0.0)
    P = model.operators(1.0)[0]
    x = views.reshape(-1, views.shape[-1])
    target = np.repeat(views.mean(axis=1), len(SCANNERS), axis=0)
    design = np.hstack([x, np.ones((len(x), 1))])
    coef, *_ = np.linalg.lstsq(design, target, rcond=None)
    W, b = coef[:-1].T, coef[-1]
    np.testing.assert_allclose(P, W, atol=1e-8)
    np.testing.assert_allclose(model.mu[0] - P @ model.mu[0], b, atol=1e-8)


def _fit(views, labels, variant, gamma=0.5, k=2):
    moments = Moments.fit(views, labels, k)
    return Mixture.build(moments, variant, beta=0.25, gamma=gamma, ridge=1e-6)


def test_module_matches_numpy_and_alpha0_is_identity():
    views, mode = synthetic_views()
    model = _fit(views, mode, "G2")
    x = views.reshape(-1, views.shape[-1])
    gate = Gate.fit(x, dim=4, seed=0)
    fmap = MixtureShrinkageMap.from_fit(model, gate, 2.0).double()
    out = fmap(torch.from_numpy(x)).numpy()
    # numpy reference
    z = (x - gate.center) @ gate.W.T
    ll = np.stack(
        [
            np.log(p) + gaussian_ll(z, gate.W @ m - gate.W @ gate.center,
                                    gate.W @ (c + n) @ gate.W.T)
            for p, m, c, n in zip(model.pi, model.mu, model.C, model.N)
        ],
        axis=1,
    )  # fmt: skip
    resp = np.exp(ll - np.logaddexp.reduce(ll, axis=1, keepdims=True))
    P = model.operators(2.0)
    ref = sum(
        resp[:, [j]] * (model.mu[j] + (x - model.mu[j]) @ P[j].T)
        for j in range(model.k)
    )
    # Buffers are float32 (the map runs in float32 in the benchmarks).
    np.testing.assert_allclose(out, ref, atol=1e-5)
    ident = MixtureShrinkageMap.from_fit(model, gate, 0.0).double()
    np.testing.assert_allclose(ident(torch.from_numpy(x)).numpy(), x, atol=1e-5)


def test_g1_ties_nuisance_and_g2_wins_when_dependent():
    views, mode = synthetic_views(dependent=True)
    held, held_mode = synthetic_views(seed=1, dependent=True)
    g1 = _fit(views, mode, "G1")
    g2 = _fit(views, mode, "G2", gamma=0.1)
    np.testing.assert_allclose(g1.N[0], g1.N[1])
    l1 = heldout_likelihoods(g1, held, held_mode)
    l2 = heldout_likelihoods(g2, held, held_mode)
    assert l2["ll_residual"] > l1["ll_residual"] + 0.05
    assert l2["ll_x"] > l1["ll_x"]
    assert l1["ll_consensus"] == pytest.approx(l2["ll_consensus"])
    # Scale-only cannot express a change of axis: it stays below G2.
    gs = _fit(views, mode, "G2-scale")
    assert heldout_likelihoods(gs, held, held_mode)["ll_residual"] < l2["ll_residual"]


def test_g2_scale_is_ml_scale():
    """On fit data, scale-only N_k = s_k N_0 is at least as likely as shared N_0."""
    views, mode = synthetic_views(dependent=True)
    views[mode == 1] *= 1.5  # also a scale difference
    g1 = heldout_likelihoods(_fit(views, mode, "G1"), views, mode)
    gs = heldout_likelihoods(_fit(views, mode, "G2-scale"), views, mode)
    assert gs["ll_residual"] > g1["ll_residual"]


def test_clusters_recover_modes():
    views, mode = synthetic_views()
    clusters = ContentClusters.fit(views.mean(axis=1), 2, dim=3, seed=0)
    labels = clusters.kmeans.labels_
    agree = max((labels == mode).mean(), (labels != mode).mean())
    assert agree > 0.99


def _write_cache(path, dim=10, seed=0):
    rng = np.random.default_rng(seed)
    for split, n_slides in (("train", 6), ("val", 2), ("test", 2)):
        views, _ = synthetic_views(n=n_slides * 2 * 6, dim=dim, seed=rng.integers(1e6))
        feats, rows = [], []
        for u, unit in enumerate(views):
            sl, rg, t = u // 12, (u // 6) % 2, u % 6
            for s, scanner in enumerate(SCANNERS):
                feats.append(unit[s])
                rows.append(
                    {"pair_id": f"slide_{split}{sl}-sample_{rg}-tile_{t}_0",
                     "scanner": scanner, "domain": s,
                     "slide_id": f"slide_{split}{sl}",
                     "region_id": f"slide_{split}{sl}-sample_{rg}"}
                )  # fmt: skip
        meta = pd.DataFrame(rows)
        meta["row"] = np.arange(len(meta))
        np.savez(path / f"features_{split}.npz", gap=np.asarray(feats, np.float32))
        meta.to_csv(path / f"meta_{split}.csv", index=False)


def test_main_writes_outputs_and_loadable_maps(tmp_path, monkeypatch):
    import sipe.eval.mixture_sweep as ms

    _write_cache(tmp_path)
    runs = tmp_path / "runs"
    monkeypatch.setattr(
        ms,
        "new_bench_run_dir",
        lambda tag: (runs / "bench" / tag).mkdir(parents=True) or runs / "bench" / tag,
    )
    run_dir = ms.main(
        ["--tag", "t", "--features-from", str(tmp_path), "--ks", "2",
         "--content-dim", "3", "--gate-dim", "3", "--alphas", "0.5", "2",
         "--betas", "0.1", "0.5", "--gammas", "0", "0.5", "--reference-map", "none",
         "--device", "-1"]
    )  # fmt: skip
    selected = json.loads((run_dir / "selected.json").read_text())
    assert set(selected) == {"G0_K1", "G1_K2", "G2-scale_K2", "G2_K2"}
    sweep = pd.read_csv(run_dir / "sweep_val.csv")
    assert set(sweep["alpha"]) == {0.0, 0.5, 1.0, 2.0}
    lik = pd.read_csv(run_dir / "likelihood.csv")
    assert {"val_ll_residual", "val_ll_x", "train_ll_consensus"} <= set(lik.columns)
    assert len(pd.read_csv(run_dir / "gating.csv")) == 3
    names = pd.read_csv(run_dir / "summary_test.csv")["name"].tolist()
    assert names == ["raw", *selected]
    loaded = load_posthoc_map(run_dir / "map_G2_K2.pt")
    assert loaded.kind == "module" and loaded.provenance["file"]["variant"] == "G2"
    x = torch.from_numpy(np.load(tmp_path / "features_val.npz")["gap"][:7])
    assert loaded.map(x).shape == (7, 10)
    from sipe.eval.results_tables import mixture_table

    table = mixture_table(runs, {})
    assert "| G2 K=2 |" in table and "| raw |" in table
