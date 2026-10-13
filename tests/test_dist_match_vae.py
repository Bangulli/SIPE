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
from sipe.data.scorpion import UnpairedTrainSCORPIONDataModule
from sipe.model.paired_vae import PairedVAE
from sipe.training.dist_match_vae_module import (
    DistMatchVAEModule,
    scanner_mmd,
    scanner_moments,
)

ROOT = Path(__file__).resolve().parents[1]
DATA_ROOT = ROOT / "data/processed/SCORPION_tiles_224px_0p5mpp"


@pytest.fixture(scope="module")
def network() -> PairedVAE:
    torch.manual_seed(0)
    return PairedVAE(pretrained=False).eval()


def _unpaired_batch() -> dict:
    # 8 random tiles, 3 scanners, no location: what the unpaired train loader gives.
    return {
        "image": torch.randn(8, 3, 224, 224),
        "domain": torch.tensor([0, 1, 2, 0, 1, 2, 0, 1]),
    }


def test_matching_terms() -> None:
    torch.manual_seed(0)
    domains = torch.arange(3).repeat(200)
    same = torch.randn(600, 16)
    shifted = same + 2.0 * torch.nn.functional.one_hot(domains, 16).float()
    scaled = shifted.clone()
    scaled[domains == 1] *= 3.0  # covariance differs, means equal after centering
    scaled[domains == 1] -= scaled[domains == 1].mean(0) - scaled.mean(0)

    mean_same, cov_same = scanner_moments(same, domains)
    mean_shift, _ = scanner_moments(shifted, domains)
    _, cov_scaled = scanner_moments(scaled, domains)
    assert float(mean_shift) > 20 * float(mean_same)
    assert float(cov_scaled) > 20 * float(cov_same)

    mmd_same = scanner_mmd(same, domains)
    mmd_shift = scanner_mmd(shifted, domains)
    assert abs(float(mmd_same)) < 0.01  # unbiased: ~0 for one distribution
    assert float(mmd_shift) > 0.03 and float(mmd_shift) > 10 * abs(float(mmd_same))

    # Standardized: rescaling the features (e.g. shrinking z) changes nothing.
    torch.testing.assert_close(scanner_mmd(0.01 * shifted, domains), mmd_shift)
    torch.testing.assert_close(
        scanner_moments(0.01 * shifted, domains)[0], mean_shift, rtol=1e-3, atol=0
    )

    # Fewer than two scanners with >= 2 samples: nothing to match.
    lonely = torch.tensor([0, 0, 0, 1])
    assert float(scanner_mmd(torch.randn(4, 16), lonely)) == 0.0
    assert float(scanner_moments(torch.randn(4, 16), lonely)[0]) == 0.0


def test_step_backward_reaches_all_parts(network: PairedVAE) -> None:
    module = DistMatchVAEModule(
        network=network, kl_warmup_steps=0, match_warmup_steps=0
    )
    module.train()
    assert not network.backbone.training  # frozen backbone stays in eval mode
    loss = module._step(_unpaired_batch(), stage="train")  # no location needed
    loss.backward()
    # Zero-initialized output layers (identity start): the first step reaches
    # mu.fc2 and the FiLM projection, not layers behind them.
    grads = {
        "mu": network.posterior.mu.fc2.weight.grad,
        "logvar": network.posterior.logvar[1].weight.grad,
        "film": network.reentangler.film.weight.grad,
        "pixel_decoder": network.decoder.decoder[0].weight.grad,
    }
    for name, grad in grads.items():
        assert grad is not None and grad.abs().sum() > 0, name
    assert all(p.grad is None for p in network.backbone.parameters())
    module.zero_grad(set_to_none=True)


@pytest.mark.skipif(not DATA_ROOT.is_dir(), reason="SCORPION tiles not available")
def test_unpaired_train_datamodule() -> None:
    dm = UnpairedTrainSCORPIONDataModule(
        DATA_ROOT,
        batch_size=10,
        val_batch_size=2,
        num_workers=0,
        persistent_workers=False,
    )
    dm.setup()
    train = next(iter(dm.train_dataloader()))
    assert train["image"].shape[0] == 10  # images, not locations
    assert "location" not in train
    val = next(iter(dm.val_dataloader()))
    assert val["image"].shape[0] == 2 * 5  # paired: every scanner of 2 locations
    assert val["location"].tolist() == [0] * 5 + [1] * 5


@pytest.mark.skipif(not DATA_ROOT.is_dir(), reason="SCORPION tiles not available")
def test_cli_fit_checkpoint_roundtrip(tmp_path: Path) -> None:
    """A real 3-step `sipe fit` writes a checkpoint the bench loader rebuilds."""
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    cmd = [
        sys.executable, "-m", "sipe.cli", "fit",
        "--config", str(ROOT / "configs" / "dist_match_vae.yaml"),
        f"--data.init_args.data_root={DATA_ROOT}",
        "--data.init_args.batch_size=8",
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
    loaded = load_cats_checkpoint(ckpt)
    assert isinstance(loaded.module, DistMatchVAEModule)
    assert loaded.global_step == 3 and loaded.phase is None
    assert loaded.provenance["module"] == (
        "sipe.training.dist_match_vae_module.DistMatchVAEModule"
    )
    images = torch.randn(2, 3, 224, 224)
    with torch.no_grad():
        z_gap = CATSFeatures(loaded.network, "z_gap")(images)
        mu = loaded.network.encode_unspecified(images)
    torch.testing.assert_close(z_gap, mu.mean(dim=(2, 3)))
