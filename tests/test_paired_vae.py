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
from sipe.data.scorpion import PairedSCORPIONDataModule, collate_views
from sipe.model.paired_vae import PairedVAE
from sipe.training.paired_vae_module import (
    PairedVAEModule,
    cross_scanner_top1,
    pair_alignment,
    partner_index,
    same_scanner_donor,
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


def test_same_scanner_donor() -> None:
    domains = torch.tensor([0, 1, 0, 2, 1, 0, 3])
    for seed in range(20):
        donor = same_scanner_donor(domains, torch.Generator().manual_seed(seed))
        assert torch.equal(domains[donor], domains)
        alone = domains >= 2  # scanners 2 and 3 appear once
        assert torch.equal(donor[alone], torch.arange(7)[alone])
        assert bool((donor[~alone] != torch.arange(7)[~alone]).all())


def test_partner_index() -> None:
    location = torch.tensor([0, 0, 0, 1, 1, 1])
    assert partner_index(location).tolist() == [1, 2, 0, 4, 5, 3]
    with pytest.raises(ValueError):
        partner_index(torch.tensor([0, 1, 0, 1]))  # not contiguous
    with pytest.raises(ValueError):
        partner_index(torch.tensor([0, 0, 1]))  # unequal views


def test_pair_alignment_and_retrieval() -> None:
    torch.manual_seed(0)
    g = torch.randn(400, 32)
    partner = partner_index(torch.arange(200).repeat_interleave(2))
    assert pair_alignment(g, partner) == pytest.approx(2.0, rel=0.15)
    paired = g[:200].repeat_interleave(2, dim=0)
    assert float(pair_alignment(paired, partner)) == 0.0
    noisy = paired + 0.1 * torch.randn_like(paired)
    domains = torch.tensor([0, 1]).repeat(200)
    locations = torch.arange(200).repeat_interleave(2)
    slides = locations // 50  # 4 slides
    assert cross_scanner_top1(noisy, domains, locations, slides) == 1.0
    assert cross_scanner_top1(g, domains, locations, slides) < 0.1
    # A scanner offset larger than the tissue signal: same-scanner tiles win.
    shifted = noisy + 5 * torch.nn.functional.one_hot(domains, 32).float()
    assert cross_scanner_top1(shifted, domains, locations, slides) < 0.5


def test_identity_at_init(network: PairedVAE) -> None:
    images = torch.randn(2, 3, 224, 224)
    with torch.no_grad():
        z_gap = CATSFeatures(network, "z_gap")(images)
        backbone_gap = CATSFeatures(network, "backbone_gap")(images)
        s, z = network.encode(images)
        features = network.decode_features(s, z)
    torch.testing.assert_close(z_gap, backbone_gap)
    torch.testing.assert_close(features, z)
    assert s.shape == (2, network.specified_dim)


def test_step_backward_reaches_all_parts(network: PairedVAE) -> None:
    module = PairedVAEModule(network=network, kl_warmup_steps=0, align_warmup_steps=0)
    module.train()
    assert not network.backbone.training  # frozen backbone stays in eval mode
    loss = module._step(_batch(), stage="train")
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
def test_paired_datamodule() -> None:
    dm = PairedSCORPIONDataModule(
        DATA_ROOT, batch_size=3, views=2, num_workers=0, persistent_workers=False
    )
    dm.setup()
    # Same WSI-grouped splits as the unpaired datamodule; every location has 5 views.
    assert len(dm.train_pairs) * 5 == len(dm.train_metadata)
    assert len(dm.val_pairs) * 5 == len(dm.val_metadata)

    torch.manual_seed(0)
    batch = collate_views([dm.train_pairs[i] for i in range(3)])
    assert batch["image"].shape == (6, 3, 224, 224)
    assert batch["location"].tolist() == [0, 0, 1, 1, 2, 2]
    for loc in range(3):
        a, b = 2 * loc, 2 * loc + 1
        assert batch["pair_id"][a] == batch["pair_id"][b]
        assert batch["domain"][a] != batch["domain"][b]

    # Shared augmentation: the same random ops on both views. Check it with one
    # image loaded twice: the two "views" must come out identical.
    base = dm.train_pairs.base
    row = dm.train_pairs.locations[0][0]
    dm.train_pairs.locations[0] = [row, row]
    for seed in range(5):
        torch.manual_seed(seed)
        x, y = dm.train_pairs[0]
        torch.testing.assert_close(x["image"], y["image"])
    assert base[row]["pair_id"] == x["pair_id"]

    val = next(iter(dm.val_dataloader()))
    assert val["image"].shape[0] == dm._eval_batch_size() * 5
    assert torch.equal(val["domain"][:5], torch.arange(5))


@pytest.mark.skipif(not DATA_ROOT.is_dir(), reason="SCORPION tiles not available")
def test_cli_fit_checkpoint_roundtrip(tmp_path: Path) -> None:
    """A real 3-step `sipe-paired fit` writes a checkpoint the bench loader rebuilds."""
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    cmd = [
        sys.executable, "-c", "from sipe.cli import main_paired; main_paired()", "fit",
        "--config", str(ROOT / "configs" / "paired_vae.yaml"),
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
    loaded = load_cats_checkpoint(ckpt)
    assert isinstance(loaded.module, PairedVAEModule)
    assert loaded.global_step == 3 and loaded.phase is None
    assert loaded.provenance["module"] == "paired_vae"
    images = torch.randn(2, 3, 224, 224)
    with torch.no_grad():
        z_gap = CATSFeatures(loaded.network, "z_gap")(images)
        mu = loaded.network.encode_unspecified(images)
    torch.testing.assert_close(z_gap, mu.mean(dim=(2, 3)))
