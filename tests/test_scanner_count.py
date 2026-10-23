"""Scanner-count study: datamodule keep-list, driver, PLISM on stored features."""

from __future__ import annotations

import json

import numpy as np
import pandas as pd
import pytest
import torch
import yaml
from test_aug_attribution import _fake_runs, _write_fit
from test_data_sanity import _make_synthetic_root
from test_loso import _dm
from test_mixture import _write_cache

from sipe.bench.plism_apply import apply_to_slide, source_encoder_pooling
from sipe.bench.posthoc import AffineMap
from sipe.data.scorpion import PairedSCORPIONDataModule
from sipe.eval.aug_attribution import cgm_init_map
from sipe.eval.cgm_select import fit_info
from sipe.eval.content_dependence import paired_views
from sipe.eval.scanner_count import fit_scanners, keep_rows, pair_subset, plism_tag


def test_datamodule_scanners_keeps_train_rows_only(tmp_path):
    root = _make_synthetic_root(tmp_path)
    full = _dm(root)
    kept = _dm(root, scanners=["AT2", "GT450"])
    assert kept.num_domains == 5 and kept.domain_to_index == full.domain_to_index
    assert set(kept.train_metadata["scanner"]) == {"AT2", "GT450"}
    for split in ("val_metadata", "test_metadata"):
        pd.testing.assert_frame_equal(getattr(kept, split), getattr(full, split))
    both = _dm(root, scanners=["AT2", "GT450", "P1000"], hold_out_scanner="P1000")
    assert set(both.train_metadata["scanner"]) == {"AT2", "GT450"}
    with pytest.raises(ValueError, match="not known domains"):
        _dm(root, scanners=["AT2", "nope"])
    with pytest.raises(ValueError, match="at least 2"):
        _dm(root, scanners=["AT2"])


def test_paired_datamodule_scanners_batches(tmp_path):
    root = _make_synthetic_root(tmp_path)
    dm = _dm(root, PairedSCORPIONDataModule, views=2, scanners=["AT2", "Philips"])
    batch = next(iter(dm.train_dataloader()))
    ids = {dm.domain_to_index[s] for s in ("AT2", "Philips")}
    assert set(batch["domain"].tolist()) == ids
    assert len(batch["domain"]) == 2 * 2  # batch_size locations x views
    assert len(dm.val_pairs[0]) == 5  # val keeps every scanner


def test_helpers():
    pairs = pd.DataFrame(
        {"scanner_a": ["AT2", "AT2", "DP200"], "scanner_b": ["DP200", "GT450", "GT450"]}
    )
    assert pair_subset(pairs, ["AT2", "DP200"]).tolist() == ["seen", "unseen", "unseen"]
    every = ["AT2", "DP200", "GT450", "P1000", "Philips"]
    assert fit_scanners({"scanners": None, "hold_out_scanner": None}, every) == every
    assert fit_scanners({"scanners": None, "hold_out_scanner": "P1000"}, every) == [
        "AT2",
        "DP200",
        "GT450",
        "Philips",
    ]
    assert fit_scanners(
        {"scanners": ["GT450", "AT2"], "hold_out_scanner": None}, every
    ) == ["AT2", "GT450"]
    assert plism_tag("G0", "AT2-DP200", "G0[AT2-DP200]") == "sc_g0_AT2-DP200"
    assert plism_tag("V1-S", "AT2-DP200", "V1-S_ab12") == "sc_v1s_ab12_AT2-DP200"


def test_fit_info_scanners_keeps_old_config_keys(tmp_path):
    from sipe.model.consensus_gen import CGMMap

    _write_fit(tmp_path / "a_1111", 0, 42, 1e-3, CGMMap(4, 0))
    _write_fit(tmp_path / "b_2222", 0, 43, 1e-3, CGMMap(4, 0))
    config = yaml.safe_load((tmp_path / "b_2222" / "config.yaml").read_text())
    config["data"]["init_args"]["scanners"] = None  # default written by newer fits
    (tmp_path / "b_2222" / "config.yaml").write_text(yaml.dump(config))
    a, b = fit_info(tmp_path / "a_1111"), fit_info(tmp_path / "b_2222")
    assert a["config_key"] == b["config_key"] and b["scanners"] is None


def _write_subset_fit(path, fmap, scanners=None, hold_out=None):
    _write_fit(path, 0, 42, 1e-3, fmap)
    config = yaml.safe_load((path / "config.yaml").read_text())
    config["data"]["init_args"]["scanners"] = scanners
    config["data"]["init_args"]["hold_out_scanner"] = hold_out
    (path / "config.yaml").write_text(yaml.dump(config))


def test_scanner_count_main(tmp_path, monkeypatch):
    import sipe.eval.scanner_count as sc

    _write_cache(tmp_path)
    x = np.load(tmp_path / "features_train.npz")["gap"]
    meta = pd.read_csv(tmp_path / "meta_train.csv")
    cpu = torch.device("cpu")
    universe = ["AT2", "DP200", "GT450"]
    fits = tmp_path / "fits"
    two = paired_views(*keep_rows(x, meta, ["AT2", "GT450"]), "tile_all")[0].views
    # A "trained" fit equal to CGM-init[AT2-GT450]: its contrast vs G0 is CGM-init's.
    _write_subset_fit(fits / "x_aaaa", cgm_init_map(two, cpu), ["GT450", "AT2"])
    _write_subset_fit(fits / "x_bbbb", cgm_init_map(two, cpu), hold_out="P1000")
    runs = _fake_runs(monkeypatch, sc, tmp_path)
    with pytest.raises(ValueError, match="not a --universe subset"):
        sc.main(
            [
                "--tag",
                "t",
                "--features-from",
                str(tmp_path),
                "--universe",
                *universe,
                "--runs",
                str(fits / "x_bbbb"),
                "--device",
                "-1",
            ]
        )
    run_dir = sc.main(
        [
            "--tag",
            "sc",
            "--features-from",
            str(tmp_path),
            "--universe",
            *universe,
            "--runs",
            str(fits / "x_aaaa"),
            "--n-boot",
            "50",
            "--device",
            "-1",
        ]
    )
    entries = pd.DataFrame(json.loads((run_dir / "fits.json").read_text())["entries"])
    # 3 pairs + 1 triple + full, x 3 closed-form maps, + raw + the fit.
    assert len(entries) == 1 + 5 * 3 + 1
    assert set(entries.loc[entries["estimator"] == "G0", "subset"]) == {
        "full",
        "AT2-DP200",
        "AT2-GT450",
        "DP200-GT450",
        "AT2-DP200-GT450",
    }
    for est in ("G0", "CGM-init", "eraser"):
        assert (run_dir / f"map_{est}_AT2-GT450.pt").exists()
    jobs = pd.read_csv(run_dir / "plism_jobs.tsv", sep="\t")
    assert "sc_v0s_aaaa_AT2-GT450" in set(jobs["tag"]) and len(jobs) == 4 * 3 + 1
    pairs = pd.read_csv(run_dir / "pairs_val.csv")
    assert set(pairs["scanner_a"]) | set(pairs["scanner_b"]) == set(universe)
    summary = pd.read_csv(run_dir / "summary_val.csv")
    triple = summary.loc[summary["name"] == "G0[AT2-DP200-GT450]", "pairs"]
    assert set(triple) == {"all", "seen"}  # no unseen pair when S = universe
    c = pd.read_csv(run_dir / "contrasts_val.csv").query("metric == 'top_1_accuracy'")
    fit = c.loc[c["name"] == "V0-S_aaaa"].set_index("pairs")["diff"]
    init = c.loc[c["name"] == "CGM-init[AT2-GT450]"].set_index("pairs")["diff"]
    pd.testing.assert_series_equal(fit, init, atol=1e-6, check_names=False)
    means = c.loc[(c["name"] == "mean") & (c["pairs"] == "all")]
    assert set(means.loc[means["estimator"] == "CGM-init", "subset"]) == {
        "mean of 3",
        "mean of 1",
    }
    assert "G0" not in set(c["estimator"])

    from sipe.eval.results_tables import scanner_count_table

    table = scanner_count_table(runs, {})
    assert "| CGM-init | 2 |" in table and "V0-S" in table


def test_plism_apply_transform_and_source_checks():
    src = np.random.default_rng(0).normal(size=(7, 3 + 4)).astype(np.float32)
    P, b = torch.randn(4, 4), torch.randn(4)
    out = np.empty_like(src)
    apply_to_slide(AffineMap(P, b), src, out, torch.device("cpu"))
    np.testing.assert_array_equal(out[:, :3], src[:, :3])
    want = ((torch.from_numpy(src[:, 3:]) - b) @ P.T + b).numpy()
    np.testing.assert_allclose(out[:, 3:], want, rtol=1e-5, atol=1e-5)

    backbone = {
        "features": {"representation": "backbone_gap"},
        "checkpoint": {"network": {"init_args": {"encoder": "h0-mini"}}},
    }
    assert source_encoder_pooling(backbone) == ("h0-mini", "gap")
    identity = {
        "features": {"representation": "posthoc_cls"},
        "checkpoint": {"encoder": "uni2-h", "file": {"identity": True}},
    }
    assert source_encoder_pooling(identity) == ("uni2-h", "cls")
    mapped = {
        "features": {"representation": "posthoc_gap"},
        "checkpoint": {"encoder": "h0-mini", "file": {}},
    }
    with pytest.raises(ValueError, match="non-identity"):
        source_encoder_pooling(mapped)
    with pytest.raises(ValueError, match="not a raw"):
        source_encoder_pooling({"features": {"representation": "z_gap"}})


def test_scanner_count_plism_table(tmp_path):
    from test_plism_pairs import fake_metrics
    from test_results_tables import bench_dir

    import sipe.eval.results_tables as rt

    assert rt._sc_tag("sc_v1s_ab12_AT2-DP200") == ("V1-S", "AT2-DP200")
    assert rt._sc_tag("sc_cgminit_AT2-DP200") == ("CGM-init", "AT2-DP200")
    assert rt._sc_tag("cgm_v1s_7b71") is None
    runs = {
        "sc_g0_AT2-DP200": 0,
        "sc_g0_AT2-GT450": 1,
        "sc_v1s_aaaa_AT2-DP200": 2,
        "sc_v1s_bbbb_AT2-GT450": 3,
        "g0_full": 4,
        "v1s_full": 5,
    }
    for tag, seed in runs.items():
        out = bench_dir(tmp_path, tag, "plism") / "metrics" / "8139_tiles" / tag
        out.mkdir(parents=True)
        fake_metrics(seed).to_csv(out / "metrics.csv", index=False)
    table = rt.scanner_count_plism_table(
        tmp_path, {"full": ["G0:g0_full", "V1-S:v1s_full"]}
    )
    top1 = {s: fake_metrics(s)["top_1_accuracy"] for s in runs.values()}
    delta = ((top1[2] - top1[0]) + (top1[3] - top1[1])) / 2
    row = next(x for x in table.splitlines() if x.startswith("| V1-S | 2 |"))
    cells = [c.strip() for c in row.strip("|").split("|")]
    assert cells[2] == "2" and cells[7].startswith(f"{delta.mean():+.3f} [")
    full = next(x for x in table.splitlines() if x.startswith("| V1-S | 5 |"))
    assert f"{(top1[5] - top1[4]).mean():+.3f}" in full
    assert (
        next(x for x in table.splitlines() if x.startswith("| G0 | 2 |")).count("–")
        == 3
    )
