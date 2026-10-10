from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import pytest
import torch

from sipe.bench.checkpoint import load_cats_checkpoint
from sipe.bench.encoders import CATSFeatures
from sipe.cli import RUN_DIR_ENV
from sipe.model.paired_vae import PairedVAE
from sipe.training.mathieu_gan_module import (
    MathieuGANModule,
    ProjectionDiscriminator,
    other_scanner_index,
)

ROOT = Path(__file__).resolve().parents[1]
DATA_ROOT = ROOT / "data/processed/SCORPION_tiles_224px_0p5mpp"


@pytest.fixture(scope="module")
def network() -> PairedVAE:
    torch.manual_seed(0)
    return PairedVAE(pretrained=False).eval()


def _batch() -> dict:
    # 3 locations x 2 views; scanners chosen so every scanner appears twice.
    return {
        "image": torch.randn(6, 3, 224, 224),
        "domain": torch.tensor([0, 1, 2, 0, 1, 2]),
        "location": torch.tensor([0, 0, 1, 1, 2, 2]),
        "pair_id": [f"slide_{i}-sample_1-tile_0_0" for i in (0, 0, 1, 1, 2, 2)],
    }


def test_other_scanner_index() -> None:
    domains = torch.tensor([0, 1, 2, 0, 1, 2, 3, 4])
    location = torch.tensor([0, 0, 1, 1, 2, 2, 3, 3])
    for seed in range(20):
        other = other_scanner_index(
            domains, location, torch.Generator().manual_seed(seed)
        )
        assert bool((domains[other] != domains).all())
        assert bool((location[other] != location).all())  # never the pair partner
    # Nobody valid (one scanner): falls back to i itself.
    same = torch.zeros(4, dtype=torch.long)
    assert other_scanner_index(same, torch.arange(4)).tolist() == [0, 1, 2, 3]


def test_projection_discriminator_is_class_conditional() -> None:
    torch.manual_seed(0)
    d = ProjectionDiscriminator(in_dim=32, hidden_dim=16, num_classes=5).eval()
    x = torch.randn(4, 32, 16, 16)
    a = d(x, torch.zeros(4, dtype=torch.long))
    b = d(x, torch.ones(4, dtype=torch.long))
    assert a.shape == (4,)
    assert not torch.allclose(a, b)


def test_generator_step_trains_generator_not_discriminator(
    network: PairedVAE,
) -> None:
    module = MathieuGANModule(
        network=network, kl_warmup_steps=0, adversarial_warmup_steps=0
    )
    module.train()
    assert not network.backbone.training  # frozen backbone stays in eval mode
    loss, gan = module._step(_batch(), stage="train")
    loss.backward()
    # Zero-initialized output layers (identity start): the first step reaches
    # mu.fc2 and the FiLM projection. The GAN term alone reaches s's encoder
    # through FiLM only after the first update, so it is not checked here.
    grads = {
        "mu": network.posterior.mu.fc2.weight.grad,
        "logvar": network.posterior.logvar[1].weight.grad,
        "film": network.reentangler.film.weight.grad,
        "pixel_decoder": network.decoder.decoder[0].weight.grad,
    }
    for name, grad in grads.items():
        assert grad is not None and grad.abs().sum() > 0, name
    assert all(p.grad is None for p in network.backbone.parameters())
    assert all(p.grad is None for p in module.discriminator.parameters())
    assert all(p.requires_grad for p in module.discriminator.parameters())
    assert bool((gan["fake_labels"] != gan["real_labels"]).all())
    module.zero_grad(set_to_none=True)


@pytest.mark.skipif(not DATA_ROOT.is_dir(), reason="SCORPION tiles not available")
def test_cli_fit_checkpoint_roundtrip(tmp_path: Path) -> None:
    """A real 3-step `sipe fit` (manual optimization, D steps not counted) writes a
    checkpoint the bench loader rebuilds."""
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    cmd = [
        sys.executable, "-m", "sipe.cli", "fit",
        "--config", str(ROOT / "configs" / "mathieu_gan.yaml"),
        f"--data.init_args.data_root={DATA_ROOT}",
        "--data.init_args.batch_size=2",
        "--data.init_args.val_batch_size=2",
        "--data.init_args.num_workers=0",
        "--data.init_args.persistent_workers=false",
        "--trainer.accelerator=cpu",
        "--trainer.precision=32-true",
        "--trainer.max_steps=3",
        "--trainer.limit_train_batches=3",
        "--trainer.limit_val_batches=2",  # "1" parses as 1.0 = 100%
        "--trainer.val_check_interval=3",
        "--trainer.logger=false",
        "--trainer.callbacks=[]",
    ]  # fmt: skip
    env = {**os.environ, RUN_DIR_ENV: str(run_dir)}
    subprocess.run(cmd, cwd=tmp_path, env=env, check=True)

    (ckpt,) = (run_dir / "checkpoints").glob("*.ckpt")
    raw = torch.load(ckpt, map_location="cpu", weights_only=False)
    assert len(raw["optimizer_states"]) == 2 and len(raw["lr_schedulers"]) == 2
    loaded = load_cats_checkpoint(ckpt)
    assert isinstance(loaded.module, MathieuGANModule)
    assert loaded.global_step == 3 and loaded.phase is None
    assert loaded.provenance["module"] == (
        "sipe.training.mathieu_gan_module.MathieuGANModule"
    )
    images = torch.randn(2, 3, 224, 224)
    with torch.no_grad():
        z_gap = CATSFeatures(loaded.network, "z_gap")(images)
        mu = loaded.network.encode_unspecified(images)
    torch.testing.assert_close(z_gap, mu.mean(dim=(2, 3)))
