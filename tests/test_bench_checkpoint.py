from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import pytest
import torch

from sipe.bench.checkpoint import load_cats_checkpoint, sha256
from sipe.bench.encoders import CATSFeatures, plism_transform
from sipe.cli import RUN_DIR_ENV
from sipe.model.arch import CATS
from sipe.model.encoders import encoder_meta

ROOT = Path(__file__).resolve().parents[1]
DATA_ROOT = ROOT / "data/processed/SCORPION_tiles_224px_0p5mpp"


def _reference_z_gap(ckpt: Path, images: torch.Tensor) -> torch.Tensor:
    """z_gap from the raw saved weights, without the loader."""
    raw = torch.load(ckpt, map_location="cpu", weights_only=True)
    network = CATS(
        **{**raw["hyper_parameters"]["network"]["init_args"], "pretrained": False}
    )
    state = {
        k.removeprefix("network."): v
        for k, v in raw["state_dict"].items()
        if k.startswith("network.")
    }
    network.load_state_dict(state)
    network.eval()
    with torch.no_grad():
        return network.encode(images)[1].mean(dim=(2, 3))


@pytest.mark.parametrize(
    ("global_step", "phase_name", "local_step"),
    [(300, "recon", 299), (600, "cycle_main", 109), (12250, "cycle_finetune", 1959)],
)
def test_loader_phase(make_cats_checkpoint, global_step, phase_name, local_step):
    loaded = load_cats_checkpoint(make_cats_checkpoint(global_step))
    assert loaded.global_step == global_step
    assert loaded.phase.name == phase_name
    assert loaded.local_step == local_step


def test_loader_matches_saved_weights(make_cats_checkpoint):
    ckpt = make_cats_checkpoint(600)
    loaded = load_cats_checkpoint(ckpt)
    assert not loaded.module.training
    meta = encoder_meta("h0-mini")
    assert (loaded.mean, loaded.std, loaded.image_size) == (meta.mean, meta.std, 224)

    images = torch.randn(2, 3, 224, 224)
    with torch.no_grad():
        z_gap = CATSFeatures(loaded.network)(images)
    assert z_gap.shape == (2, 704)
    torch.testing.assert_close(z_gap, _reference_z_gap(ckpt, images))

    prov = loaded.provenance
    assert prov["sha256"] == sha256(ckpt)
    assert prov["phase_name"] == "cycle_main"
    assert prov["network"]["init_args"]["encoder"] == "h0-mini"
    assert prov["train_git"]["commit"] == "abc123"
    assert prov["train_git"]["dirty"] is False


def test_loader_rejects_step_past_curriculum(make_cats_checkpoint):
    with pytest.raises(ValueError, match="past the curriculum end"):
        load_cats_checkpoint(make_cats_checkpoint(12251))


def test_representations():
    network = CATS(pretrained=False).eval()
    images = torch.randn(2, 3, 224, 224)
    expected = {"z_gap": 704, "s": 64, "backbone_gap": 768, "backbone_cls": 768}
    with torch.no_grad():
        for name, dim in expected.items():
            features = CATSFeatures(network, name)
            assert features.output_dim == dim
            assert features(images).shape == (2, dim)
    with pytest.raises(ValueError, match="Unknown representation"):
        CATSFeatures(network, "cls")


def test_plism_transform_uses_encoder_stats():
    import numpy as np

    meta = encoder_meta("h0-mini")
    tile = np.full((224, 224, 3), 255, dtype=np.uint8)
    out = plism_transform(meta)(tile)
    expected = (1 - torch.tensor(meta.mean)) / torch.tensor(meta.std)
    torch.testing.assert_close(out[:, 0, 0], expected)


@pytest.mark.skipif(not DATA_ROOT.is_dir(), reason="SCORPION tiles not available")
def test_cli_fit_checkpoint_roundtrip(tmp_path):
    """A real 2-step SIPECLI fit writes a checkpoint the loader rebuilds."""
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    cmd = [
        sys.executable, "-m", "sipe.cli", "fit",
        "--config", str(ROOT / "configs/cats_legacy_steps.yaml"),
        f"--data.init_args.data_root={DATA_ROOT}",
        "--data.init_args.batch_size=4",
        "--data.init_args.num_workers=0",
        "--data.init_args.persistent_workers=false",
        "--trainer.accelerator=cpu",
        "--trainer.precision=32-true",
        "--trainer.max_steps=2",
        "--trainer.limit_train_batches=2",
        "--trainer.limit_val_batches=2",  # "1" parses as 1.0 = 100%
        "--trainer.val_check_interval=2",
        "--trainer.logger=false",
        "--trainer.callbacks=[]",
    ]  # fmt: skip
    env = {**os.environ, RUN_DIR_ENV: str(run_dir)}
    subprocess.run(cmd, cwd=tmp_path, env=env, check=True)

    (ckpt,) = (run_dir / "checkpoints").glob("*.ckpt")
    raw = torch.load(ckpt, map_location="cpu", weights_only=True)
    assert raw["hyper_parameters"]["network"]["init_args"]["encoder"] == "h0-mini"
    assert raw["datamodule_hyper_parameters"]["encoder"] == "h0-mini"

    loaded = load_cats_checkpoint(ckpt)
    assert loaded.global_step == 2
    assert loaded.phase.name == "recon"
    assert loaded.mean == encoder_meta("h0-mini").mean
    images = torch.randn(2, 3, 224, 224)
    with torch.no_grad():
        z_gap = CATSFeatures(loaded.network)(images)
    torch.testing.assert_close(z_gap, _reference_z_gap(ckpt, images))
