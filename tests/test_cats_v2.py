from __future__ import annotations

import pytest
import torch

from sipe.bench.encoders import CATSFeatures
from sipe.model.arch import CATSv2
from sipe.training.cats_module import CATSModule

PHASE = {
    "name": "feature",
    "mode": "feature",
    "steps": 10,
    "lr": 1e-3,
    "restart_steps": 10,
    "adverse_alpha": 1.0,
}


@pytest.fixture(scope="module")
def network() -> CATSv2:
    torch.manual_seed(0)
    return CATSv2(pretrained=False).eval()


def test_identity_at_init(network: CATSv2) -> None:
    images = torch.randn(2, 3, 224, 224)
    with torch.no_grad():
        out = network(images)
        z_gap = CATSFeatures(network, "z_gap")(images)
        backbone_gap = CATSFeatures(network, "backbone_gap")(images)
    torch.testing.assert_close(out["z"], out["feature_map"])
    torch.testing.assert_close(out["reconstructed_features"], out["feature_map"])
    torch.testing.assert_close(z_gap, backbone_gap)
    assert network.unspecified_dim == network.feature_dim


def test_feature_cycle_zero_when_s_constant(network: CATSv2) -> None:
    module = CATSModule(network=network, num_domains=5, curriculum=[PHASE])
    feature_map = torch.randn(4, network.feature_dim, 16, 16)
    _, z = network.disentangle(feature_map)
    s = torch.randn(1, network.specified_dim).expand(4, -1)
    with torch.no_grad():
        cycle = module._feature_cycle_losses(
            s1=s, z1=z, domains=torch.tensor([0, 1, 2, 3])
        )
    # At init the re-entangler is identity: swapping a constant s changes nothing,
    # but s2 is recomputed from the features, so only z_l1 is exactly 0.
    assert cycle["z_l1"].item() == pytest.approx(0.0, abs=1e-6)


def test_feature_step_backward(network: CATSv2) -> None:
    module = CATSModule(
        network=network,
        num_domains=5,
        curriculum=[PHASE],
        pooled_adversary_hidden_dim=32,
    )
    network.freeze_backbone()
    network.train()
    out = module._feature_forward(torch.randn(4, 3, 224, 224))
    loss = module._feature_recon_loss(out["reconstructed_features"], out["feature_map"])
    assert loss.item() == pytest.approx(0.0, abs=1e-5)
    cycle = module._feature_cycle_losses(
        s1=out["s"], z1=out["z"], domains=torch.tensor([0, 1, 2, 3])
    )
    pooled, _ = module._pooled_adversary(
        z=out["z"], domains=torch.tensor([0, 1, 2, 3]), alpha=1.0, norm=True
    )
    (sum(cycle[k] for k in ("s_l1", "z_l1", "domain_ce")) + pooled).backward()
    film = network.reentangler.film.weight.grad
    assert film is not None and film.abs().sum() > 0
    network.zero_grad(set_to_none=True)
    module.zero_grad(set_to_none=True)


def test_feature_mode_rejects_legacy_network() -> None:
    from sipe.model.arch import CATS

    with pytest.raises(ValueError, match="CATSv2"):
        CATSModule(network=CATS(pretrained=False), num_domains=5, curriculum=[PHASE])
