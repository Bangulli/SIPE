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
from sipe.model.scanner_vae import ScannerVAE
from sipe.training.scanner_vae_module import (
    ScannerVAEModule,
    linear_probe_accuracy,
)

ROOT = Path(__file__).resolve().parents[1]
DATA_ROOT = ROOT / "data/processed/SCORPION_tiles_224px_0p5mpp"


@pytest.fixture(scope="module")
def network() -> ScannerVAE:
    torch.manual_seed(0)
    return ScannerVAE(pretrained=False).eval()


def _batch(n: int = 4) -> dict:
    return {
        "image": torch.randn(n, 3, 224, 224),
        "domain": torch.tensor([0, 1, 2, 1])[:n],
        "pair_id": [f"slide_{i}-sample_1-tile_0_0" for i in range(n)],
    }


def test_identity_at_init(network: ScannerVAE) -> None:
    images = torch.randn(2, 3, 224, 224)
    with torch.no_grad():
        z_gap = CATSFeatures(network, "z_gap")(images)
        backbone_gap = CATSFeatures(network, "backbone_gap")(images)
        _, logvar = network.encode_features(network.backbone_feature_map(images))
    torch.testing.assert_close(z_gap, backbone_gap)
    torch.testing.assert_close(logvar, torch.full_like(logvar, -4.0))


def test_step_backward_reaches_all_parts(network: ScannerVAE) -> None:
    module = ScannerVAEModule(
        network=network, kl_warmup_steps=0, adversary_warmup_steps=0
    )
    module.train()
    assert not network.backbone.training  # frozen backbone stays in eval mode
    loss = module._step(_batch(), stage="train")
    loss.backward()
    # Zero-initialized output layers (identity start): the first step only reaches
    # mu.fc2 and the FiLM projection, not mu.fc1 or the scanner embedding behind them.
    grads = {
        "mu": network.posterior.mu.fc2.weight.grad,
        "logvar": network.posterior.logvar[1].weight.grad,
        "film": network.reentangler.film.weight.grad,
        "adversary": module.adversary[0].weight.grad,
        "pixel_decoder": network.decoder.decoder[0].weight.grad,
    }
    for name, grad in grads.items():
        assert grad is not None and grad.abs().sum() > 0, name
    assert all(p.grad is None for p in network.backbone.parameters())
    module.zero_grad(set_to_none=True)


def test_grl_reverses_adversary_gradient(network: ScannerVAE) -> None:
    module = ScannerVAEModule(network=network)
    z_gap = torch.randn(4, network.unspecified_dim, requires_grad=True)
    domains = torch.tensor([0, 1, 2, 1])
    grads = []
    for x in (z_gap, module.grl(z_gap)):
        loss = torch.nn.functional.cross_entropy(module.adversary(x), domains)
        grads.append(torch.autograd.grad(loss, z_gap)[0])
    torch.testing.assert_close(grads[1], -grads[0])


def test_reset_adversary_clears_weights_and_state(network: ScannerVAE) -> None:
    module = ScannerVAEModule(network=network)
    optimizer = torch.optim.AdamW(module.adversary.parameters())
    module.optimizers = lambda use_pl_optimizer=False: optimizer
    module.adversary(torch.randn(2, network.unspecified_dim)).sum().backward()
    optimizer.step()
    before = module.adversary[0].weight.detach().clone()
    module.reset_adversary()
    assert not torch.equal(before, module.adversary[0].weight)
    assert all(p not in optimizer.state for p in module.adversary.parameters())


def test_linear_probe_accuracy() -> None:
    torch.manual_seed(0)
    y = torch.randint(0, 5, (1000,))
    x = torch.randn(1000, 32)
    separable = x + 4 * torch.nn.functional.one_hot(y, 32).float()
    assert (
        linear_probe_accuracy(separable[:500], y[:500], separable[500:], y[500:], 5)
        > 0.95
    )
    assert linear_probe_accuracy(x[:500], y[:500], x[500:], y[500:], 5) < 0.4


@pytest.mark.skipif(not DATA_ROOT.is_dir(), reason="SCORPION tiles not available")
def test_cli_fit_checkpoint_roundtrip(tmp_path: Path) -> None:
    """A real 2-step `sipe-vae fit` writes a checkpoint the bench loader rebuilds."""
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    cmd = [
        sys.executable, "-c", "from sipe.cli import main_vae; main_vae()", "fit",
        "--config", str(ROOT / "configs/scanner_vae.yaml"),
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
    loaded = load_cats_checkpoint(ckpt)
    assert isinstance(loaded.module, ScannerVAEModule)
    assert loaded.global_step == 2 and loaded.phase is None
    assert loaded.provenance["module"] == "scanner_vae"
    images = torch.randn(2, 3, 224, 224)
    with torch.no_grad():
        z_gap = CATSFeatures(loaded.network, "z_gap")(images)
        mu = loaded.network.encode_unspecified(images)
    torch.testing.assert_close(z_gap, mu.mean(dim=(2, 3)))
