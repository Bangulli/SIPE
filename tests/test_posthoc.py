"""Post-hoc map loading and the backbone-GAP + map extractor (sipe.bench.posthoc)."""

from __future__ import annotations

import torch
import torch.nn as nn

from sipe.baselines.soft_delta_legacy import SoftDeltaProjectionEraser
from sipe.bench.posthoc import (
    AffineMap,
    PosthocFeatures,
    load_posthoc,
    load_posthoc_map,
)


def _affine(dim: int = 8, seed: int = 0) -> tuple[torch.Tensor, torch.Tensor]:
    g = torch.Generator().manual_seed(seed)
    return torch.randn(dim, dim, generator=g), torch.randn(dim, generator=g)


def test_affine_map_matches_vendored_eraser():
    P, bias = _affine()
    eraser = SoftDeltaProjectionEraser(
        P=P, proj_left=None, proj_right=None, bias=bias, lam=0.5, rank=None,
        delta_moment="second_moment",
    )  # fmt: skip
    x = torch.randn(5, 8)
    torch.testing.assert_close(AffineMap(P, bias)(x), eraser(x))


def test_load_affine_file(tmp_path):
    # Layout of the sweep's eraser_<pool>_lam<λ>.pt.
    P, bias = _affine()
    path = tmp_path / "eraser_gap_lam0.5.pt"
    torch.save(
        {"P": P, "bias": bias, "lambda": 0.5, "pooling": "gap", "encoder": "h0-mini",
         "eraser_config": {"affine": True}},
        path,
    )  # fmt: skip
    (tmp_path / "git.txt").write_text("commit: abc123\ndirty: False\n")
    loaded = load_posthoc_map(path)
    assert loaded.kind == "affine" and loaded.encoder == "h0-mini"
    assert loaded.provenance["file"]["lambda"] == 0.5
    assert loaded.provenance["fit_git"]["commit"] == "abc123"
    x = torch.randn(3, 8)
    torch.testing.assert_close(loaded.map(x), (x - bias) @ P.T + bias)


def test_load_module_file(tmp_path):
    lin = nn.Linear(8, 4)
    path = tmp_path / "map.pt"
    torch.save(
        {"class_path": "torch.nn.Linear",
         "init_args": {"in_features": 8, "out_features": 4},
         "state_dict": lin.state_dict()},
        path,
    )  # fmt: skip
    loaded = load_posthoc_map(path)
    assert loaded.kind == "module" and loaded.pooling == "gap"
    x = torch.randn(3, 8)
    torch.testing.assert_close(loaded.map(x), lin(x))


class _StubBackbone(nn.Module):
    """timm-like: forward_features -> [B, prefix + N, C]."""

    num_prefix_tokens = 2

    def __init__(self, dim: int = 8) -> None:
        super().__init__()
        self.proj = nn.Linear(3, dim)

    def forward_features(self, images: torch.Tensor) -> torch.Tensor:
        tokens = self.proj(images.flatten(2).transpose(1, 2))  # [B, H*W, C]
        prefix = torch.full_like(tokens[:, :2], 100.0)
        return torch.cat([prefix, tokens], dim=1)


def test_posthoc_features_gap_then_map_in_float32():
    P, bias = _affine()
    features = PosthocFeatures(_StubBackbone(), AffineMap(P, bias), 8).eval()
    images = torch.randn(2, 3, 4, 4)
    with torch.no_grad():
        gap = features.backbone.forward_features(images)[:, 2:].mean(dim=1)
        expected = (gap - bias) @ P.T + bias
        torch.testing.assert_close(features(images), expected)
        with torch.autocast("cpu", dtype=torch.bfloat16):
            out = features(images)
    assert out.dtype == torch.float32


def test_load_posthoc_h0mini(tmp_path):
    P, bias = _affine(768)
    path = tmp_path / "eraser_gap_lam0.5.pt"
    torch.save({"P": P, "bias": bias, "pooling": "gap", "encoder": "h0-mini"}, path)
    loaded = load_posthoc(path, pretrained=False)
    assert loaded.features.output_dim == 768
    assert loaded.meta.num_prefix_tokens == loaded.features.backbone.num_prefix_tokens
    with torch.no_grad():
        out = loaded.features(torch.randn(1, 3, 224, 224))
    assert out.shape == (1, 768)


def test_posthoc_features_cls_pooling():
    P, bias = _affine()
    features = PosthocFeatures(_StubBackbone(), AffineMap(P, bias), 8, "cls").eval()
    images = torch.randn(2, 3, 4, 4)
    with torch.no_grad():
        cls = features.backbone.forward_features(images)[:, 0]
        torch.testing.assert_close(features(images), (cls - bias) @ P.T + bias)


def test_identity_map_file_gives_raw_cls(tmp_path):
    from sipe.bench.posthoc import identity_map, representation

    path = tmp_path / "map_identity.pt"
    torch.save(identity_map("h0-mini", "cls"), path)
    loaded = load_posthoc(path, pretrained=False)
    assert loaded.provenance["pooling"] == "cls"
    assert representation(loaded.provenance["pooling"]) == "posthoc_cls"
    images = torch.randn(1, 3, 224, 224)
    with torch.no_grad():
        raw = loaded.features.backbone.forward_features(images)[:, 0].float()
        torch.testing.assert_close(loaded.features(images), raw)


def test_load_posthoc_refuses_wrong_encoder_dim(tmp_path):
    import pytest

    P, bias = _affine(1536)  # a UNI2-h-sized map labelled h0-mini (768)
    path = tmp_path / "map.pt"
    torch.save({"P": P, "bias": bias, "encoder": "h0-mini"}, path)
    with pytest.raises(ValueError, match="wrong `encoder`"):
        load_posthoc(path, pretrained=False)


def test_load_posthoc_map_refuses_unknown_pooling(tmp_path):
    import pytest

    P, bias = _affine()
    path = tmp_path / "map.pt"
    torch.save({"P": P, "bias": bias, "pooling": "max"}, path)
    with pytest.raises(ValueError, match="pooling"):
        load_posthoc_map(path)
