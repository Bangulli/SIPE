"""Augmentation study: feature-cache extraction helpers, the closed-form attribution
driver and the CGM selection driver."""

from __future__ import annotations

import json

import numpy as np
import pandas as pd
import pytest
import torch
import torchvision.transforms as T
import yaml
from PIL import Image
from test_mixture import _write_cache

from sipe.bench.posthoc import load_posthoc_map
from sipe.data.scorpion import RandomRot90
from sipe.eval.aug_attribution import cgm_init_map, select_units, unit_views
from sipe.eval.cgm_select import candidate_table, fit_info, select
from sipe.features.extract import AugmentedViews, Dihedral, view_seed
from sipe.model.consensus_gen import ConsensusGenerativeHead

# --- extraction ---------------------------------------------------------------


def _image(seed=0, size=8):
    rng = np.random.default_rng(seed)
    return Image.fromarray(rng.integers(0, 255, (size, size, 3), dtype=np.uint8))


def test_dihedral_is_the_d4_group():
    img = _image()
    arr = np.asarray(img)
    outs = [np.asarray(Dihedral(c)(img)) for c in range(8)]
    np.testing.assert_array_equal(outs[0], arr)
    # PIL ROTATE_90 is counter-clockwise, as np.rot90.
    np.testing.assert_array_equal(outs[1], np.rot90(arr))
    np.testing.assert_array_equal(outs[4], arr[:, ::-1])
    assert len({o.tobytes() for o in outs}) == 8
    with pytest.raises(ValueError):
        Dihedral(8)


class _FakeTiles:
    """Stand-in for SCORPIONTileDataset: two scanners per pair, same pixels."""

    def __init__(self, pair_ids):
        self.metadata = pd.DataFrame({"_pair_id": pair_ids})
        self.transform = None

    def __len__(self):
        return len(self.metadata)

    def __getitem__(self, idx):
        pair = self.metadata["_pair_id"].iloc[idx]
        image = _image(seed=hash(pair) % 1000, size=32)
        return {"image": self.transform(image), "pair_id": pair}


def test_augmented_views_share_the_transform_per_tile_copy():
    base = _FakeTiles(["a", "a", "b", "b"])
    crop = T.RandomResizedCrop(32, scale=(0.5, 1.0))
    random = T.Compose([T.RandomHorizontalFlip(), RandomRot90(), crop, T.ToTensor()])
    views = AugmentedViews(base, [random, random, random], seed=3)
    assert len(views) == 12
    items = [views[i] for i in range(12)]
    assert [int(it["aug"]) for it in items] == [0] * 4 + [1] * 4 + [2] * 4
    for copy in range(3):
        a0, a1, b0, b1 = items[4 * copy : 4 * copy + 4]
        torch.testing.assert_close(a0["image"], a1["image"])  # scanner views share
        torch.testing.assert_close(b0["image"], b1["image"])
    # Copies differ (3 random copies of the same tile are not all equal).
    assert len({items[4 * c]["image"].sum().item() for c in range(3)}) > 1
    # Deterministic: a second pass gives the same pixels.
    torch.testing.assert_close(views[5]["image"], items[5]["image"])
    assert view_seed("a", 1, 3) == view_seed("a", 1, 3) != view_seed("a", 2, 3)


# --- closed-form attribution --------------------------------------------------


def _write_aug_cache(path, clean, copies=2, scale=0.1, seed=0):
    """Train-only cache: K noisy copies of the clean train views (shared per tile)."""
    rng = np.random.default_rng(seed)
    x = np.load(clean / "features_train.npz")["gap"]
    meta = pd.read_csv(clean / "meta_train.csv")
    feats, metas = [], []
    for c in range(copies):
        shift = {p: rng.normal(size=x.shape[1]) for p in meta["pair_id"].unique()}
        feats.append(x + scale * np.stack([shift[p] for p in meta["pair_id"]]))
        metas.append(meta.assign(aug=c))
    meta = pd.concat(metas, ignore_index=True)
    meta["row"] = np.arange(len(meta))
    path.mkdir()
    np.savez(path / "features_train.npz", gap=np.concatenate(feats).astype(np.float32))
    meta.to_csv(path / "meta_train.csv", index=False)


def test_select_units_same_tiles_one_copy(tmp_path):
    _write_cache(tmp_path)
    _write_aug_cache(tmp_path / "aug", tmp_path, copies=3)
    clean = pd.read_csv(tmp_path / "meta_train.csv")
    aug = pd.read_csv(tmp_path / "aug" / "meta_train.csv")
    tiles = np.sort(clean["pair_id"].unique())[::-1]
    for meta in (clean, aug):
        rows = select_units(meta, "5", tiles, seed=0)
        assert sorted(rows["pair_id"].unique()) == sorted(tiles[:5])
        assert rows.groupby("pair_id")["aug"].nunique().eq(1).all()
        assert len(rows) == 5 * 5  # 5 tiles x 5 scanners
    assert len(select_units(aug, "tiles", tiles, 0)) == len(clean)
    assert len(select_units(aug, "all", tiles, 0)) == len(aug)
    views = unit_views(
        np.load(tmp_path / "aug" / "features_train.npz")["gap"],
        select_units(aug, "all", tiles, 0),
    )
    assert views.shape == (3 * len(tiles), 5, 10)
    with pytest.raises(ValueError):
        select_units(clean, str(len(tiles) + 1), tiles, 0)


def test_cgm_init_map_is_the_v0s_init():
    rng = np.random.default_rng(0)
    views = rng.normal(size=(300, 3, 6)) + rng.normal(size=(300, 1, 6))
    fmap = cgm_init_map(views, torch.device("cpu"))
    head = ConsensusGenerativeHead(6, 3, per_scanner=True, encoder_hidden_dim=0)
    head.init_from_moments(
        torch.from_numpy(views.reshape(-1, 6)),
        torch.arange(3).repeat(300),
        torch.arange(300).repeat_interleave(3),
        ridge=1e-3,
    )
    x = torch.randn(5, 6)
    torch.testing.assert_close(fmap(x), head.map(x))
    assert fmap.r is None and fmap.h is None


def _fake_runs(monkeypatch, module, tmp_path):
    runs = tmp_path / "runs"
    monkeypatch.setattr(
        module,
        "new_bench_run_dir",
        lambda tag: (runs / "bench" / tag).mkdir(parents=True) or runs / "bench" / tag,
    )
    return runs


def test_cgm_init_noise_weight_scales_the_noise():
    rng = np.random.default_rng(0)
    views = rng.normal(size=(300, 3, 6)) + 2 * rng.normal(size=(300, 1, 6))
    base = cgm_init_map(views, torch.device("cpu"))
    torch.testing.assert_close(
        cgm_init_map(views, torch.device("cpu"), noise_weight=1.0).M, base.M
    )
    head = ConsensusGenerativeHead(6, 3, per_scanner=True, encoder_hidden_dim=0)
    head.init_from_moments(
        torch.from_numpy(views.reshape(-1, 6)),
        torch.arange(3).repeat(300),
        torch.arange(300).repeat_interleave(3),
        ridge=1e-3,
    )
    L = head.map.L.double()
    B = head.noise_factor()[0].double()
    noise = B @ B.T
    # M = L^T marginal^-1  ->  marginal = M^-1 L^T; beta adds (beta - 1) noise.
    marginal = torch.linalg.solve(head.map.M.double(), L.T)
    beta = 3.5
    expected = torch.linalg.solve(marginal + (beta - 1) * noise, L).T
    weighted = cgm_init_map(views, torch.device("cpu"), noise_weight=beta)
    torch.testing.assert_close(weighted.M.double(), expected, rtol=1e-6, atol=1e-8)
    torch.testing.assert_close(weighted.L, base.L)
    # More noise -> stronger shrinkage of the map's linear part.
    x = torch.randn(50, 6, dtype=torch.float64)
    T1 = base.L.double() @ base.M.double()
    Tb = weighted.L.double() @ weighted.M.double()
    assert (x @ Tb.T).norm() < (x @ T1.T).norm()


def test_aug_attribution_main(tmp_path, monkeypatch):
    import sipe.eval.aug_attribution as aa

    _write_cache(tmp_path)
    _write_aug_cache(tmp_path / "geo", tmp_path)
    runs = _fake_runs(monkeypatch, aa, tmp_path)
    run_dir = aa.main(
        ["--tag", "t", "--features-from", str(tmp_path),
         "--aug-caches", f"geo={tmp_path / 'geo'}", "--n-units", "20", "tiles", "all",
         "--alphas", "0.5", "2", "--init-noise-weights", "1", "3.5",
         "--reference-map", "none", "--device", "-1"]
    )  # fmt: skip
    arms = json.loads((run_dir / "arms.json").read_text())
    expected = {f"{e}|{d}|{n}" for e in ("G0", "CGM-init", "CGM-init(b=3.5)")
                for d, n in (("clean", "20"), ("clean", "all"), ("geo", "20"),
                             ("geo", "tiles"), ("geo", "all"))}  # fmt: skip
    assert set(arms) == expected
    assert arms["G0|geo|all"]["units"] == 2 * arms["G0|clean|all"]["units"]
    assert arms["G0|geo|20"]["views"] == 20 * 5
    names = pd.read_csv(run_dir / "summary_val.csv")["name"].tolist()
    assert names == ["raw", *arms]
    for name, arm in arms.items():
        loaded = load_posthoc_map(run_dir / arm["map"])
        assert loaded.map(torch.zeros(2, 10)).shape == (2, 10), name
    from sipe.eval.results_tables import aug_attribution_table

    table = aug_attribution_table(runs, {})
    assert "| CGM-init | geo | 20 |" in table and "| raw |" in table
    assert "| CGM-init(b=3.5) | clean | all |" in table
    assert arms["CGM-init(b=3.5)|clean|all"]["map"] == "map_CGM-init-b3.5_clean_all.pt"
    assert arms["CGM-init|clean|all"]["map"] == "map_CGM-init_clean_all.pt"
    assert torch.load(run_dir / "map_CGM-init-b3.5_clean_all.pt")["noise_weight"] == 3.5


# --- CGM selection ------------------------------------------------------------


def _write_fit(path, arm_hidden, seed, rho, fmap):
    path.mkdir(parents=True)
    config = {
        "seed_everything": seed,
        "trainer": {"max_steps": 10, "logger": {"x": seed},
                    "default_root_dir": str(path)},
        "model": {"init_args": {
            "network": {"init_args": {"per_scanner": True,
                                      "encoder_hidden_dim": arm_hidden,
                                      "generator_hidden_dim": 0}},
            "residual_penalty": rho, "kl_weight": 1.0}},
        "data": {"init_args": {"augmentations": ["rot90", "zoom"]}},
    }  # fmt: skip
    (path / "config.yaml").write_text(yaml.dump(config))
    torch.save(
        {"class_path": "sipe.model.consensus_gen.CGMMap",
         "init_args": {"dim": fmap.dim, "encoder_hidden_dim": arm_hidden,
                       "generator_hidden_dim": 0},
         "state_dict": fmap.state_dict()},
        path / "map_cgm.pt",
    )  # fmt: skip


def test_fit_info_groups_seeds(tmp_path):
    from sipe.model.consensus_gen import CGMMap

    _write_fit(tmp_path / "a_1111", 8, 42, 1e-3, CGMMap(4, 8))
    _write_fit(tmp_path / "b_2222", 8, 43, 1e-3, CGMMap(4, 8))
    _write_fit(tmp_path / "c_3333", 0, 42, 1e-3, CGMMap(4, 0))
    a, b, c = (fit_info(tmp_path / n) for n in ("a_1111", "b_2222", "c_3333"))
    assert a["arm"] == "V1-S" and c["arm"] == "V0-S"
    assert a["config_key"] == b["config_key"] != c["config_key"]


def test_fit_info_ignores_default_hold_out_scanner(tmp_path):
    """Configs written after the LOSO change carry hold_out_scanner: null."""
    from sipe.model.consensus_gen import CGMMap

    _write_fit(tmp_path / "a_1111", 8, 42, 1e-3, CGMMap(4, 8))
    _write_fit(tmp_path / "b_2222", 8, 44, 1e-3, CGMMap(4, 8))
    _write_fit(tmp_path / "c_3333", 8, 44, 1e-3, CGMMap(4, 8))
    for name, value in (("b_2222", None), ("c_3333", "AT2")):
        path = tmp_path / name / "config.yaml"
        config = yaml.safe_load(path.read_text())
        config["data"]["init_args"]["hold_out_scanner"] = value
        path.write_text(yaml.dump(config))
    a, b, c = (fit_info(tmp_path / n) for n in ("a_1111", "b_2222", "c_3333"))
    assert a["config_key"] == b["config_key"] != c["config_key"]


def test_select_simplest_reaching_gain_and_fallback():
    fits = pd.DataFrame({
        "name": ["V1-S_a", "V1-S_b", "V0-S_c", "V1-S_d"],
        "config_id": ["c0", "c0", "c1", "c2"], "arm": ["V1-S", "V1-S", "V0-S", "V1-S"],
        "residual_penalty": [1e-3, 1e-3, 1e-3, 1e-1], "kl_weight": 1.0,
        "seed": [42, 43, 42, 42], "top_1_accuracy": [0.990, 0.992, 0.985, 0.9905],
    })  # fmt: skip
    cand = candidate_table(fits)
    # Simplicity order: V0-S, then V1-S with the larger rho first.
    assert cand["config_id"].tolist() == ["c1", "c2", "c0"]
    assert cand.set_index("config_id").loc["c0", "seeds"] == 2
    assert select(cand, 0.980, 0.95)["config_id"] == "c2"  # 0.0105 >= 0.95 x 0.011
    assert select(cand, 0.980, 0.4)["config_id"] == "c1"
    assert select(cand, 0.995, 0.95)["config_id"] == "G0"  # nothing beats G0


def test_cgm_select_main(tmp_path, monkeypatch):
    import sipe.eval.cgm_select as cs
    from sipe.eval.content_dependence import paired_views

    _write_cache(tmp_path)
    x = np.load(tmp_path / "features_train.npz")["gap"]
    views = paired_views(x, pd.read_csv(tmp_path / "meta_train.csv"), "tile_all")[0]
    fmap = cgm_init_map(views.views, torch.device("cpu"))
    fits = tmp_path / "fits"
    _write_fit(fits / "2026_x_aaaa", 0, 42, 1e-3, fmap)
    _write_fit(fits / "2026_x_bbbb", 0, 43, 1e-3, fmap)
    runs = _fake_runs(monkeypatch, cs, tmp_path)
    run_dir = cs.main(
        ["--tag", "t", "--features-from", str(tmp_path), "--runs",
         str(fits / "2026_x_aaaa"), str(fits / "2026_x_bbbb"),
         "--reference-map", "none", "--device", "-1"]
    )  # fmt: skip
    selected = json.loads((run_dir / "selected.json").read_text())
    assert selected["candidates"] == 1
    val = pd.read_csv(run_dir / "summary_val.csv")
    assert val["name"].tolist() == ["raw", "G0", "V0-S_aaaa", "V0-S_bbbb"]
    test = pd.read_csv(run_dir / "summary_test.csv")
    if selected["fallback_to_g0"]:
        assert test["name"].tolist() == ["raw", "G0"]
    else:
        assert test["name"].tolist() == ["raw", "G0", "V0-S_aaaa", "V0-S_bbbb"]
    from sipe.eval.results_tables import cgm_select_table

    table = cgm_select_table(runs, {})
    assert "| V0-S_aaaa |" in table and "| G0 |" in table


def test_aug_attribution_fixed_point_arms(tmp_path, monkeypatch):
    import sipe.eval.aug_attribution as aa
    from sipe.model.consensus_gen import CGMMap

    _write_cache(tmp_path)
    runs = _fake_runs(monkeypatch, aa, tmp_path)
    ref = CGMMap(10, encoder_hidden_dim=0)
    torch.save(
        {"class_path": "sipe.model.consensus_gen.CGMMap",
         "init_args": {"dim": 10, "encoder_hidden_dim": 0, "generator_hidden_dim": 0},
         "state_dict": ref.state_dict()},
        tmp_path / "v0s.pt",
    )  # fmt: skip
    run_dir = aa.main(
        ["--tag", "t", "--features-from", str(tmp_path), "--n-units", "all",
         "--estimators", "CGM-init", "--fixed-point", "shared", "per_scanner",
         "--fp-iters", "1", "2", "0", "--fp-max-iter", "20",
         "--fp-init-noise-weights", "1", "3",
         "--maps", f"V0-S={tmp_path / 'v0s.pt'}",
         "--reference-map", "none", "--retrieval-only", "--device", "-1"]
    )  # fmt: skip
    arms = json.loads((run_dir / "arms.json").read_text())
    assert set(arms) == {"CGM-init|clean|all"} | {
        f"CGM-fp({noise},k={k}{b})|clean|all"
        for noise in ("shared", "per_scanner") for k in ("1", "2", "conv")
        for b in ("", ",b=3")
    }  # fmt: skip
    b3 = arms["CGM-fp(per_scanner,k=1,b=3)|clean|all"]
    assert b3["map"] == "map_CGM-fp-per_scanner-k1-b3_clean_all.pt"
    assert b3["init_noise_weight"] == 3.0
    m1 = load_posthoc_map(run_dir / arms["CGM-fp(per_scanner,k=1)|clean|all"]["map"])
    assert not torch.equal(load_posthoc_map(run_dir / b3["map"]).map.M, m1.map.M)
    fp_arm = arms["CGM-fp(per_scanner,k=2)|clean|all"]
    assert fp_arm["map"] == "map_CGM-fp-per_scanner-k2_clean_all.pt"
    assert fp_arm["iterations"] == 2
    assert fp_arm["J_final"] <= arms["CGM-fp(per_scanner,k=1)|clean|all"]["J_final"]
    assert fp_arm["J_final"] <= fp_arm["J_init"]
    assert arms["CGM-fp(shared,k=conv)|clean|all"]["iterations"] <= 20
    one = load_posthoc_map(run_dir / arms["CGM-fp(shared,k=1)|clean|all"]["map"]).map
    two = load_posthoc_map(run_dir / arms["CGM-fp(shared,k=2)|clean|all"]["map"]).map
    assert not torch.equal(one.M, two.M)
    assert (run_dir / "fp_history_shared_clean_all.csv").exists()
    dist = pd.read_csv(run_dir / "linear_dist.csv")
    assert set(dist["name"]) == set(arms) and set(dist["to"]) == {"V0-S"}
    loaded = load_posthoc_map(run_dir / fp_arm["map"])
    assert loaded.map(torch.zeros(2, 10)).shape == (2, 10)
    from sipe.eval.results_tables import aug_attribution_table

    table = aug_attribution_table(runs, {})
    assert "| CGM-fp(shared,k=1) | clean | all |" in table
    assert "rel. distance of L M to" in table and "V0-S " in table
