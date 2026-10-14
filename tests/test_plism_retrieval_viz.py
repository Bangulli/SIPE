"""PLISM retrieval viz. Bench env: bench/run pytest tests/test_plism_retrieval_viz.py"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
import yaml

pytest.importorskip("plismbench")
h5py = pytest.importorskip("h5py")

from plismbench.metrics import TopkAccuracy  # noqa: E402

from sipe.bench import plism_retrieval_viz as viz  # noqa: E402

N_ROWS = 16_278  # plismbench tile subsets index the full slide


def brute_force(fa: np.ndarray, fb: np.ndarray):
    x = np.concatenate([fa, fb]).astype(np.float64)
    x /= np.linalg.norm(x, axis=1, keepdims=True)
    n = len(fa)
    rank, cross, n_same = [], [], []
    for q in range(2 * n):
        sim = x @ x[q]
        sim[q] = -np.inf
        t = (q + n) % (2 * n)
        other = (np.arange(2 * n) >= n) != (q >= n)
        rank.append(1 + int((sim > sim[t]).sum()))
        cross.append(1 + int(((sim > sim[t]) & other).sum()))
        top = np.argsort(-sim)[: viz.TOP]
        n_same.append(int((~other[top]).sum()))
    return np.array(rank), np.array(cross), np.array(n_same)


def test_ranks_match_brute_force_and_plismbench():
    rng = np.random.default_rng(0)
    n, d = 60, 16
    content = rng.normal(size=(n, d))
    fa = content + 0.6 * rng.normal(size=(n, d)) + 0.3  # slide-level shift
    fb = content + 0.6 * rng.normal(size=(n, d)) - 0.3
    res = viz.retrieve(fa.astype(np.float32), fb.astype(np.float32), chunk=17)
    coords = np.stack([np.full(n, 16), np.arange(n) // 8, np.arange(n) % 8], 1)
    qt = viz.query_table(res, coords, "r", "A", "B")
    rank, cross, n_same = brute_force(fa, fb)
    np.testing.assert_array_equal(qt["rank_true"], rank)
    np.testing.assert_array_equal(qt["rank_true_cross"], cross)
    np.testing.assert_array_equal(qt["n_same_top10"], n_same)
    assert (qt["rank_true_cross"] <= qt["rank_true"]).all()
    assert 0 < (qt["rank_true"] == 1).mean() < 1  # the shift makes some misses

    ref = TopkAccuracy(device="cpu", use_mixed_precision=False, k=[1, 10])
    expected = ref.compute_metric(fa.astype(np.float32), fb.astype(np.float32))
    ours = [(qt["rank_true"] <= k).mean() for k in (1, 10)]
    np.testing.assert_allclose(ours, expected, atol=1e-12)


def _fake_run(root, tag, slides, rng, shift):
    run = root / f"run_{tag}"
    coords = np.stack(
        [np.full(N_ROWS, 16), np.arange(N_ROWS) // 200, np.arange(N_ROWS) % 200], 1
    )
    content = rng.normal(size=(N_ROWS, 8))
    for i, slide in enumerate(slides):
        d = run / "features" / tag / slide
        d.mkdir(parents=True)
        f = content + 0.3 * rng.normal(size=content.shape) + shift * (i - 0.5)
        np.save(d / "features.npy", np.concatenate([coords, f], 1).astype(np.float32))
    (run / "config.yaml").write_text(yaml.dump({"tag": tag}))
    return run


def test_main_smoke(tmp_path, monkeypatch):
    slides = ["GMH_S60_to_GMH_S60.tif", "GMH_P_to_GMH_S60.tif"]
    rng = np.random.default_rng(1)
    runs = [_fake_run(tmp_path, "ref", slides, rng, 0.6),
            _fake_run(tmp_path, "m", slides, rng, 0.1)]  # fmt: skip
    bench = tmp_path / "runs" / "bench"
    monkeypatch.setattr(
        viz, "new_bench_run_dir",
        lambda tag: (bench / tag).mkdir(parents=True) or bench / tag,
    )  # fmt: skip
    monkeypatch.setattr(
        viz.TileReader, "__call__",
        lambda self, slide, c: np.full((224, 224, 3), c[2] % 255, np.uint8),
    )  # fmt: skip
    out = viz.main(
        ["--tag", "t", "--features", *map(str, runs), "--pair", "GMH_S60:GMH_P",
         "--n-tiles", "460", "--n-examples", "2"]
    )  # fmt: skip
    summary = pd.read_csv(out / "summary.csv")
    assert set(summary["direction"]) == {"both", "a->b", "b->a"}
    m = summary[(summary.representation == "m") & (summary.direction != "both")]
    assert (m[[c for c in m if c.startswith("vs_ref_")]].sum(1) == 460).all()
    queries = pd.read_csv(out / "queries.csv.gz")
    assert len(queries) == 2 * 2 * 460
    figs = {p.name for p in (out / "figures").glob("*.png")}
    assert "rank_curve_GMH_S60--GMH_P.png" in figs
    assert any("misses" in f for f in figs) and any("hits" in f for f in figs)


def test_tile_reader(tmp_path):
    with h5py.File(tmp_path / "S.tif.h5", "w") as f:
        f["tile_16_3_4"] = np.full((224, 224, 3), 7, np.uint8)
    reader = viz.TileReader(tmp_path)
    assert reader("S.tif", np.array([16, 3, 4]))[0, 0, 0] == 7
    reader.close()
