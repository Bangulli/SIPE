"""Grouped k-fold CV (MIDL protocol): fold assignment, datamodule folds, cache folds,
the closed-form CV driver and map_retrieval's CV mode."""

from __future__ import annotations

import argparse
import json

import numpy as np
import pandas as pd
import pytest
import torch
from test_data_sanity import _make_synthetic_root
from test_mixture import _write_cache

from sipe.bench.posthoc import load_posthoc_map
from sipe.data.folds import REFIT, group_folds
from sipe.data.scorpion import PairedSCORPIONDataModule
from sipe.eval.proxy import load_cache, load_cv_cache, select_cv


def test_group_folds_balanced_and_deterministic():
    groups = [f"slide_{i}" for i in range(48)] * 3
    folds = group_folds(groups, 4)
    assert len(folds) == 48
    assert np.bincount(list(folds.values())).tolist() == [12, 12, 12, 12]
    assert folds == group_folds(reversed(groups), 4)  # order-free
    assert folds != group_folds(groups, 4, seed=0)
    with pytest.raises(ValueError):
        group_folds(groups, 1)
    with pytest.raises(ValueError):
        group_folds(["a", "b"], 3)


def _folds(scores, changes):
    return pd.DataFrame(
        [{"name": k, "fold": f, "top_1_accuracy": v, "relative_change": changes[k]}
         for k, per_fold in scores.items() for f, v in enumerate(per_fold)]
    )  # fmt: skip


def test_select_cv_best_then_least_change_among_ties():
    changes = {"small": 0.1, "big": 0.3, "worse": 0.05}
    # big is best; small is below it on every fold by a steady 0.002: resolved.
    resolved = {"big": [0.90, 0.92, 0.94, 0.96], "small": [0.898, 0.918, 0.938, 0.958],
                "worse": [0.80, 0.82, 0.84, 0.86]}  # fmt: skip
    assert select_cv(_folds(resolved, changes), "name") == ("big", ["big"], "big")
    # small is behind on average but not on every fold: a tie, the smaller change wins.
    noisy = {**resolved, "small": [0.91, 0.91, 0.95, 0.95]}
    selected, tied, best = select_cv(_folds(noisy, changes), "name")
    assert (selected, best) == ("small", "big") and set(tied) == {"big", "small"}
    # equal scores on every fold (zero spread) also tie.
    equal = {**resolved, "small": resolved["big"]}
    assert select_cv(_folds(equal, changes), "name")[0] == "small"


def _paired_dm(root, fold):
    dm = PairedSCORPIONDataModule(
        root, views=5, batch_size=2, num_workers=0, image_size=8,
        group_column="slide_id", persistent_workers=False, n_folds=4, fold=fold,
    )  # fmt: skip
    dm.setup()
    return dm


def test_datamodule_folds_partition_the_wsis(tmp_path):
    root = _make_synthetic_root(tmp_path, n_slides=8)
    all_slides = set(pd.read_csv(root / "metadata.csv")["slide_id"])
    vals = []
    for fold in range(4):
        dm = _paired_dm(root, fold)
        train, val = (
            set(dm.train_metadata["slide_id"]),
            set(dm.val_metadata["slide_id"]),
        )
        assert train and val and not train & val and train | val == all_slides
        assert dm.test_metadata.empty and dm.test_pairs is None
        vals.append(val)
    assert set().union(*vals) == all_slides and sum(map(len, vals)) == len(all_slides)
    refit = _paired_dm(root, REFIT)
    assert set(refit.train_metadata["slide_id"]) == all_slides
    assert set(refit.val_metadata["slide_id"]) == vals[0]  # in sample, monitoring
    with pytest.raises(ValueError):
        PairedSCORPIONDataModule(root, n_folds=4)
    with pytest.raises(ValueError):
        PairedSCORPIONDataModule(root, n_folds=4, fold=4)


def test_cache_folds_match_the_datamodule_and_keep_features(tmp_path):
    _write_cache(tmp_path)
    feats, metas = load_cache(tmp_path, ("train", "val", "test"))
    by_key = {
        (r.pair_id, r.scanner): feats[s][r.row]
        for s in metas for r in metas[s].itertuples()
    }  # fmt: skip
    slides = pd.concat(metas.values())["slide_id"]
    folds = group_folds(slides, 4)
    for fold in range(4):
        x, meta = load_cv_cache(tmp_path, 4, fold)
        assert set(meta["val"]["slide_id"]) == {
            s for s, f in folds.items() if f == fold
        }
        assert not set(meta["train"]["slide_id"]) & set(meta["val"]["slide_id"])
        for split in ("train", "val"):
            assert meta[split]["row"].tolist() == list(range(len(x[split])))
            for r in meta[split].sample(5, random_state=0).itertuples():
                np.testing.assert_array_equal(
                    x[split][r.row], by_key[(r.pair_id, r.scanner)]
                )
    x, meta = load_cv_cache(tmp_path, 4, REFIT)
    assert list(x) == ["train"] and len(x["train"]) == sum(map(len, feats.values()))


def _fake_runs(monkeypatch, module, tmp_path):
    runs = tmp_path / "runs"
    monkeypatch.setattr(
        module,
        "new_bench_run_dir",
        lambda tag: (runs / "bench" / tag).mkdir(parents=True) or runs / "bench" / tag,
    )


def test_cv_sweep_selects_on_the_fold_mean_and_refits(tmp_path, monkeypatch):
    import sipe.eval.cv_sweep as cv

    _write_cache(tmp_path)
    _fake_runs(monkeypatch, cv, tmp_path)
    run_dir = cv.main(
        ["--tag", "t", "--features-from", str(tmp_path), "--gate-dim", "3",
         "--lambdas", "1", "10", "--alphas", "1", "4", "--device", "-1"]
    )  # fmt: skip
    folds = pd.read_csv(run_dir / "cv_folds.csv")
    mean = pd.read_csv(run_dir / "cv_mean.csv")
    assert set(folds["fold"]) == {0, 1, 2, 3}
    assert len(folds) == 4 * len(mean) and set(mean["n_folds"]) == {4}
    selected = json.loads((run_dir / "selected.json").read_text())
    assert set(selected) == set(cv.METHODS)
    x = torch.from_numpy(load_cv_cache(tmp_path, 4, REFIT)[0]["train"][:7])
    for method, sel in selected.items():
        rows = mean.loc[mean["method"] == method]
        row = rows.loc[rows["value"] == sel["value"]].iloc[0]
        assert sel["cv_top_1_mean"] == pytest.approx(
            row["top_1_accuracy_mean"], abs=1e-4
        )
        assert sel["cv_top_1_mean"] == pytest.approx(
            np.mean(sel["cv_top_1_folds"]), abs=1e-3
        )
        assert (
            sel["best_value"] == rows.loc[rows["top_1_accuracy_mean"].idxmax(), "value"]
        )
        assert (
            sel["value"] in sel["tied_values"]
            and sel["best_value"] in sel["tied_values"]
        )
        tied = rows.loc[rows["value"].isin(sel["tied_values"])]
        assert row["relative_change_mean"] == tied["relative_change_mean"].min()
        fmap = load_posthoc_map(run_dir / sel["map_file"]).map
        # The saved map is the refit on every WSI at the selected value.
        refit = cv.Fitted(
            method, *(d["train"] for d in load_cv_cache(tmp_path, 4, REFIT)),
            argparse.Namespace(ridge=1e-4, gate_dim=3, seed=0),
            torch.device("cpu"),
        )  # fmt: skip
        if sel["value"] > 0:
            torch.testing.assert_close(fmap(x), refit.map(sel["value"])(x))


def test_map_retrieval_cv_mode(tmp_path, monkeypatch):
    import sipe.eval.cv_sweep as cv
    import sipe.eval.map_retrieval as mr

    _write_cache(tmp_path)
    _fake_runs(monkeypatch, cv, tmp_path / "a")
    _fake_runs(monkeypatch, mr, tmp_path / "b")
    sweep = cv.main(
        ["--tag", "t", "--features-from", str(tmp_path), "--methods", "eraser_slide",
         "--lambdas", "1", "--device", "-1"]
    )  # fmt: skip
    path = str(sweep / "map_eraser_slide.pt")
    mr.main(
        ["--tag", "m", "--features-from", str(tmp_path), "--n-folds", "4",
         "--maps", f"same={','.join([path] * 4)}", "--device", "-1"]
    )  # fmt: skip
    run_dir = tmp_path / "b" / "runs" / "bench" / "scorpion_m"
    folds = pd.read_csv(run_dir / "cv_folds.csv")
    mean = pd.read_csv(run_dir / "cv_mean.csv")
    assert set(mean["name"]) == {"raw", "same"} and set(folds["fold"]) == {0, 1, 2, 3}
    assert (folds.query("name == 'raw'")["relative_change"] == 0).all()
    assert (folds.query("name == 'same'")["relative_change"] > 0).all()
    selected = json.loads((run_dir / "selected.json").read_text())
    assert selected["selected"] in selected["tied"] and selected["best"] == "same"
    # Raw rows equal the CV sweep's identity rows (same folds, same metric).
    raw = pd.read_csv(sweep / "cv_folds.csv").query("value == 0")
    np.testing.assert_allclose(
        folds.query("name == 'raw'")["top_1_accuracy"].to_numpy(),
        raw["top_1_accuracy"].to_numpy(),
    )
    with pytest.raises(ValueError):
        mr.main(
            ["--tag", "m2", "--features-from", str(tmp_path), "--n-folds", "4",
             "--maps", f"x={path}", "--device", "-1"]
        )  # fmt: skip


def test_scorpion_cv_table_reads_both_drivers(tmp_path, monkeypatch):
    import sipe.eval.cv_sweep as cv
    import sipe.eval.map_retrieval as mr
    from sipe.eval.results_tables import scorpion_cv_table

    _write_cache(tmp_path)
    _fake_runs(monkeypatch, cv, tmp_path)
    _fake_runs(monkeypatch, mr, tmp_path)
    sweep = cv.main(
        ["--tag", "t", "--features-from", str(tmp_path), "--methods", "g0",
         "--gate-dim", "3", "--alphas", "1", "4", "--device", "-1"]
    )  # fmt: skip
    path = str(sweep / "map_g0.pt")
    mr.main(
        ["--tag", "m", "--features-from", str(tmp_path), "--n-folds", "4",
         "--maps", f"fit={','.join([path] * 4)}", "--device", "-1"]
    )  # fmt: skip
    table = scorpion_cv_table(tmp_path / "runs", {})
    lines = [line for line in table.splitlines() if line.startswith("| g0 ")]
    assert len(lines) == 3  # alpha 0, 1, 4
    assert sum("yes (`map_g0.pt`)" in line for line in lines) == 1
    assert any(line.startswith("| fit ") for line in table.splitlines())


def test_scorpion_cv_pairs_table(tmp_path, monkeypatch):
    import sipe.eval.cv_sweep as cv
    import sipe.eval.map_retrieval as mr
    from sipe.eval.results_tables import scorpion_cv_pairs_table

    _write_cache(tmp_path)
    _fake_runs(monkeypatch, cv, tmp_path)
    _fake_runs(monkeypatch, mr, tmp_path)
    sweep = cv.main(
        ["--tag", "t", "--features-from", str(tmp_path), "--methods", "g0",
         "--gate-dim", "3", "--alphas", "1", "--device", "-1"]
    )  # fmt: skip
    path = str(sweep / "map_g0.pt")
    mr.main(
        ["--tag", "m", "--features-from", str(tmp_path), "--n-folds", "4",
         "--maps", f"fit={','.join([path] * 4)}", "--device", "-1"]
    )  # fmt: skip
    table = scorpion_cv_pairs_table(
        tmp_path / "runs", {"ref": ["g0@0"], "tags": ["g0@1", "raw", "nope"]}
    )
    rows = {line.split(" | ")[0][2:]: line.split(" | ") for line in table.splitlines()
            if line.startswith("| ") and not line.startswith("| method")}  # fmt: skip
    folds = pd.read_csv(sweep / "cv_folds.csv").set_index(["value", "fold"])
    delta = folds.loc[1.0, "top_1_accuracy"] - folds.loc[0.0, "top_1_accuracy"]
    assert rows["g0@1"][5] == f"{delta.mean():+.4f}"
    assert rows["g0@1"][7] == f"{int((delta > 0).sum())}/4"
    assert rows["raw"][5] == "+0.0000"  # map_retrieval's raw = the sweep's identity
    assert "no CV run" in rows["nope"][-1]
