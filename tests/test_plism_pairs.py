"""sipe.eval.plism_pairs: merge of per-pair PLISM metrics.csv files."""

from __future__ import annotations

import itertools

import numpy as np
import pandas as pd
import pytest
import yaml

from sipe.eval.plism_pairs import (
    bootstrap_mean_ci,
    main,
    merge_runs,
    shortlist,
    slide_bootstrap_weights,
)

SLIDES = ["GMH_S60", "GMH_P", "HR_S60", "HR_AT2"]


def fake_metrics(seed: int) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    rows = []
    for a, b in itertools.combinations(SLIDES, 2):
        stain_a, scanner_a = a.split("_")
        stain_b, scanner_b = b.split("_")
        rows.append({
            "slide_a": f"{a}_to_GMH_S60.tif", "features_path_a": "x",
            "staining_a": stain_a, "scanner_a": scanner_a,
            "slide_b": f"{b}_to_GMH_S60.tif", "features_path_b": "y",
            "staining_b": stain_b, "scanner_b": scanner_b,
            "cosine_similarity": rng.random(), "top_1_accuracy": rng.random(),
            "top_3_accuracy": 0.0, "top_5_accuracy": 0.0,
            "top_10_accuracy": rng.random(),
        })  # fmt: skip
    return pd.DataFrame(rows)


def test_merge_shift_and_delta():
    ref, other = fake_metrics(0), fake_metrics(1)
    pairs = merge_runs([("ref", ref), ("m", other.iloc[::-1])])  # row order irrelevant
    assert len(pairs) == 6
    by_pair = pairs.set_index(["slide_a", "slide_b"])
    gmh = ("GMH_S60_to_GMH_S60.tif", "GMH_P_to_GMH_S60.tif")
    assert by_pair.loc[gmh, "shift"] == "inter-scanner"
    assert by_pair.loc[("GMH_S60_to_GMH_S60.tif", "HR_S60_to_GMH_S60.tif"),
                       "shift"] == "inter-staining"  # fmt: skip
    assert by_pair.loc[("GMH_P_to_GMH_S60.tif", "HR_AT2_to_GMH_S60.tif"),
                       "shift"] == "joint"  # fmt: skip
    assert not by_pair.loc[gmh, "hr"]
    np.testing.assert_allclose(
        pairs["m__delta_top_1"], pairs["m__top_1"] - pairs["ref__top_1"]
    )
    np.testing.assert_allclose(pairs["ref__top_1"], ref["top_1_accuracy"])


def test_merge_rejects_mismatched_pairs():
    with pytest.raises(ValueError, match="same slide pairs"):
        merge_runs([("ref", fake_metrics(0)), ("m", fake_metrics(1).iloc[:-1])])


def test_shortlist_order():
    pairs = merge_runs([("ref", fake_metrics(0)), ("m", fake_metrics(1))])
    short = shortlist(pairs, ["ref", "m"], k=1)
    for shift, sub in pairs.groupby("shift"):
        rows = short[short["shift"] == shift].set_index("reason")
        assert rows.loc["worst_ref", "ref__top_1"] == sub["ref__top_1"].min()
        assert rows.loc["best_ref", "ref__top_1"] == sub["ref__top_1"].max()
        assert rows.loc["gain_m", "m__delta_top_1"] == sub["m__delta_top_1"].max()
        assert rows.loc["loss_m", "m__delta_top_1"] == sub["m__delta_top_1"].min()


def test_main(tmp_path, monkeypatch):
    runs = []
    for i, tag in enumerate(["ref", "m"]):
        run = tmp_path / f"run_{tag}"
        metrics = run / "metrics" / "8139_tiles" / tag
        metrics.mkdir(parents=True)
        (run / "config.yaml").write_text(yaml.dump({"tag": tag}))
        fake_metrics(i).to_csv(metrics / "metrics.csv", index=False)
        runs.append(str(run))
    bench = tmp_path / "runs" / "bench"
    monkeypatch.setattr(
        "sipe.eval.plism_pairs.new_bench_run_dir",
        lambda tag: (
            (bench / f"stamp_{tag}").mkdir(parents=True) or bench / f"stamp_{tag}"
        ),
    )
    out = main(["--runs", *runs, "--tag", "t", "--k", "1"])
    assert len(pd.read_csv(out / "pairs.csv")) == 6
    assert "reason" in pd.read_csv(out / "shortlist.csv")
    assert yaml.safe_load((out / "config.yaml").read_text())["labels"] == ["ref", "m"]


def test_slide_bootstrap_weights_are_products_of_slide_counts():
    pairs = merge_runs([("ref", fake_metrics(0))])
    w = slide_bootstrap_weights(pairs, n_boot=50, seed=3)
    assert w.shape == (50, 6)
    # Rebuild the counts from the same draws and check w = c_a * c_b.
    slides = pd.Index(sorted(set(pairs["slide_a"]) | set(pairs["slide_b"])))
    rng = np.random.default_rng(3)
    for row in w:
        c = np.bincount(rng.integers(0, 4, 4), minlength=4)
        expected = c[slides.get_indexer(pairs["slide_a"])] * c[
            slides.get_indexer(pairs["slide_b"])
        ]
        np.testing.assert_array_equal(row, expected)


def test_bootstrap_ci_brackets_constant_and_mean():
    pairs = merge_runs([("ref", fake_metrics(0))])
    w = slide_bootstrap_weights(pairs, n_boot=200)
    assert bootstrap_mean_ci(np.full(6, 0.25), w) == pytest.approx((0.25, 0.25))
    values = pairs["ref__top_1"].to_numpy()
    lo, hi = bootstrap_mean_ci(values, w)
    assert values.min() <= lo <= hi <= values.max()
