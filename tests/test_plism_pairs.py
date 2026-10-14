"""sipe.eval.plism_pairs: merge of per-pair PLISM metrics.csv files."""

from __future__ import annotations

import itertools

import numpy as np
import pandas as pd
import pytest
import yaml

from sipe.eval.plism_pairs import main, merge_runs, shortlist

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
