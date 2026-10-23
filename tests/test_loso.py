"""Leave-one-scanner-out: hold-out helpers, the datamodule option, the LOSO driver."""

from __future__ import annotations

import json

import numpy as np
import pandas as pd
import pytest
import torch
import yaml
from test_aug_attribution import _fake_runs, _write_fit
from test_data_sanity import SCANNERS, _make_synthetic_root
from test_mixture import _write_cache

from sipe.data.scorpion import PairedSCORPIONDataModule, SCORPIONDataModule
from sipe.eval.aug_attribution import cgm_init_map, g0_model
from sipe.eval.cgm_select import fit_info
from sipe.eval.content_dependence import paired_views
from sipe.eval.loso import (
    contrasts,
    drop_scanner,
    label_pairs,
    replicate_means,
    slide_counts,
)
from sipe.model.consensus_gen import ConsensusGenerativeHead


def _dm(root, cls=SCORPIONDataModule, **kw):
    dm = cls(root, batch_size=2, num_workers=0, image_size=8,
             persistent_workers=False, **kw)  # fmt: skip
    dm.setup("fit")
    return dm


def test_datamodule_hold_out_drops_train_rows_only(tmp_path):
    root = _make_synthetic_root(tmp_path)
    full = _dm(root)
    held = _dm(root, hold_out_scanner="GT450")
    assert held.num_domains == 5 and held.domain_to_index == full.domain_to_index
    assert "GT450" not in set(held.train_metadata["scanner"])
    kept = full.train_metadata.loc[full.train_metadata["scanner"] != "GT450"]
    assert held.train_metadata["filename"].tolist() == kept["filename"].tolist()
    for split in ("val_metadata", "test_metadata"):
        pd.testing.assert_frame_equal(getattr(held, split), getattr(full, split))
    with pytest.raises(ValueError, match="not a known domain"):
        _dm(root, hold_out_scanner="nope")


def test_paired_datamodule_hold_out_needs_views_4(tmp_path):
    root = _make_synthetic_root(tmp_path)
    with pytest.raises(RuntimeError, match="views=4"):
        _dm(root, PairedSCORPIONDataModule, views=5, hold_out_scanner="P1000")
    dm = _dm(root, PairedSCORPIONDataModule, views=4, hold_out_scanner="P1000")
    batch = next(iter(dm.train_dataloader()))
    assert dm.domain_to_index["P1000"] not in batch["domain"].tolist()
    assert len(dm.val_pairs[0]) == 5  # val keeps every scanner


def test_head_with_a_missing_scanner_stays_finite():
    rng = np.random.default_rng(0)
    n, s, d = 40, 5, 6
    x = torch.tensor(rng.normal(size=(n * s, d)))
    scanners = torch.arange(s).repeat(n)
    loc = torch.arange(n).repeat_interleave(s)
    keep = scanners != 2
    head = ConsensusGenerativeHead(d, num_scanners=s, encoder_hidden_dim=0).double()
    head.init_from_moments(x[keep], scanners[keep], loc[keep])
    assert torch.equal(head.delta[2], torch.zeros(d, dtype=head.delta.dtype))
    u = head.map.standardize(x[keep])
    nll = head.nll(u, head.map.generate(head.map.encode(u)), scanners[keep])
    assert torch.isfinite(nll).all()


def test_drop_scanner_and_labels():
    meta = pd.DataFrame({"scanner": ["AT2", "P1000", "AT2"]})
    assert drop_scanner(meta, "P1000")["scanner"].tolist() == ["AT2", "AT2"]
    with pytest.raises(ValueError):
        drop_scanner(meta, "GT450")
    pairs = pd.DataFrame({"scanner_a": ["AT2", "AT2", "DP200"],
                          "scanner_b": ["DP200", "P1000", "GT450"]})  # fmt: skip
    assert label_pairs(pairs, "P1000").tolist() == ["seen", "unseen", "seen"]
    assert label_pairs(pairs, "AT2").tolist() == ["unseen", "unseen", "seen"]


def test_slide_bootstrap_without_resampling_is_the_mean():
    diff = pd.Series([1.0, 2.0, 6.0])
    slide = pd.Series(["s0", "s0", "s1"])
    counts = pd.DataFrame([[1.0, 1.0], [2.0, 0.0]], columns=["s0", "s1"])
    np.testing.assert_allclose(replicate_means(diff, slide, counts), [3.0, 1.5])
    assert slide_counts(["a", "b", "c"], 50, 0).sum(axis=1).eq(3).all()


def test_contrast_of_identical_maps_is_zero():
    pairs = pd.DataFrame({
        "slide_id": ["s0", "s1"] * 2, "scanner_a": "AT2", "scanner_b": "P1000",
        "top_1_accuracy": [0.5, 0.7] * 2, "name": ["G0^-P1000"] * 2 + ["G0"] * 2,
    })  # fmt: skip
    for col in ("cosine_similarity", "top_3_accuracy", "top_5_accuracy",
                "top_10_accuracy"):  # fmt: skip
        pairs[col] = 0.0
    entries = pd.DataFrame({"name": ["G0", "G0^-P1000"], "estimator": "G0",
                            "held_out": ["none", "P1000"]})  # fmt: skip
    out = contrasts(pairs, entries, ["P1000"], slide_counts(["s0", "s1"], 20, 0))
    assert set(out["contrast"]) == {"vs_full"}  # G0 is not compared with itself
    assert (out["diff"] == 0).all() and (out["ci_low"] == 0).all()


def _write_loso_fit(path, fmap, hold_out):
    _write_fit(path, 0, 42, 1e-3, fmap)
    config = yaml.safe_load((path / "config.yaml").read_text())
    config["data"]["init_args"]["hold_out_scanner"] = hold_out
    (path / "config.yaml").write_text(yaml.dump(config))


def test_cgm_select_refuses_loso_fits(tmp_path, monkeypatch):
    import sipe.eval.cgm_select as cs
    from sipe.model.consensus_gen import CGMMap

    _write_cache(tmp_path)
    _write_loso_fit(tmp_path / "f_aaaa", CGMMap(10, 0), "AT2")
    assert fit_info(tmp_path / "f_aaaa")["hold_out_scanner"] == "AT2"
    _fake_runs(monkeypatch, cs, tmp_path)
    with pytest.raises(ValueError, match="subset of the scanners"):
        cs.main(["--tag", "t", "--features-from", str(tmp_path), "--runs",
                 str(tmp_path / "f_aaaa"), "--device", "-1"])  # fmt: skip


def test_loso_main(tmp_path, monkeypatch):
    import sipe.eval.loso as lo

    _write_cache(tmp_path)
    x = np.load(tmp_path / "features_train.npz")["gap"]
    meta = pd.read_csv(tmp_path / "meta_train.csv")
    full = paired_views(x, meta, "tile_all")[0].views
    minus = paired_views(x, drop_scanner(meta, "P1000"), "tile_all")[0]
    assert minus.scanners == [s for s in SCANNERS if s != "P1000"]
    # G0^-j = G0 on the hand-filtered views.
    hand = full[:, [i for i, s in enumerate(SCANNERS) if s != "P1000"]]
    np.testing.assert_allclose(g0_model(minus.views, 1e-4).mu,
                               g0_model(hand, 1e-4).mu)  # fmt: skip

    cpu = torch.device("cpu")
    fits = tmp_path / "fits"
    _write_fit(fits / "x_full", 0, 42, 1e-3, cgm_init_map(full, cpu))
    # A "trained" fit equal to CGM-init^-P1000: its vs_full contrast is CGM-init's.
    _write_loso_fit(fits / "x_p100", cgm_init_map(minus.views, cpu), "P1000")
    runs = _fake_runs(monkeypatch, lo, tmp_path)
    run_dir = lo.main(
        ["--tag", "loso", "--features-from", str(tmp_path), "--runs",
         str(fits / "x_full"), str(fits / "x_p100"), "--scanners", "AT2", "P1000",
         "--n-boot", "50", "--device", "-1"]
    )  # fmt: skip
    outputs = ("pairs_val.csv", "summary_val.csv", "contrasts_val.csv", "fits.json",
               "map_G0_minusP1000.pt", "map_CGM-init_minusAT2.pt")  # fmt: skip
    for name in outputs:
        assert (run_dir / name).exists(), name
    entries = json.loads((run_dir / "fits.json").read_text())["entries"]
    assert {"V0-S_full", "V0-S_p100", "G0^-AT2", "CGM-init^-P1000"} <= {
        e["name"] for e in entries
    }
    summary = pd.read_csv(run_dir / "summary_val.csv")
    assert set(summary.loc[summary["name"] == "V0-S_p100", "scanner"]) == {"P1000"}
    assert set(summary.loc[summary["name"] == "G0", "scanner"]) == {"AT2", "P1000"}
    c = pd.read_csv(run_dir / "contrasts_val.csv").query("metric == 'top_1_accuracy'")
    v0s = c.loc[(c["name"] == "V0-S_p100") & (c["contrast"] == "vs_G0")]
    init = c.loc[(c["name"] == "CGM-init^-P1000") & (c["contrast"] == "vs_G0")]
    np.testing.assert_allclose(v0s["diff"].to_numpy(), init["diff"].to_numpy(),
                               atol=1e-6)  # fmt: skip
    assert "mean" in set(c.loc[c["estimator"] == "G0", "scanner"])
    assert "mean" not in set(c.loc[c["estimator"] == "V0-S", "scanner"])

    from sipe.eval.results_tables import loso_table

    table = loso_table(runs, {})
    assert "| val | P1000 |" in table and "V0-S_p100" in table

    test_dir = lo.main(
        ["--tag", "loso_test", "--features-from", str(tmp_path), "--runs",
         str(fits / "x_p100"), "--scanners", "P1000", "--split", "test",
         "--n-boot", "50", "--device", "-1"]
    )  # fmt: skip
    assert (test_dir / "contrasts_test.csv").exists()
    assert not (test_dir / "contrasts_val.csv").exists()
    test_slides = set(pd.read_csv(test_dir / "pairs_test.csv")["slide_id"])
    assert test_slides and all(s.startswith("slide_test") for s in test_slides)
    assert yaml.safe_load((test_dir / "config.yaml").read_text())["split"] == "test"
    both = loso_table(runs, {})
    assert "| test | P1000 |" in both and "| val | P1000 |" in both
    only = loso_table(runs, {"split": ["test"]})
    assert "| test |" in only and "| val |" not in only
