"""CONCH v1.5 (MahmoodLab/conchv1_5, gated) as a timm ViT-L/16 matched to TRIDENT.

The authors recommend TRIDENT's `encoder_factory("conch_v15")`. TRIDENT needs
transformers<5, so it lives in the bench env only; this module rebuilds the same
model in plain timm for the main env, and `tests/test_bench_adapters.py` checks it
against `encoder_factory("conch_v15")` on real tiles (bench env).

What TRIDENT does, and how it is reproduced here:
- weights: `pytorch_model_vision.bin` (the repo has no timm config). `trunk.*` = a
  timm `vit_large_patch16_224` (1 CLS token, LayerScale init 1.0);
  `attn_pool_contrast.*` + `ln_contrast.*` = the attentional pooler (one query over all
  tokens) + LayerNorm whose 768-d output is the CONCH v1.5 embedding
  (`CONCHVisionTower.forward`). The head is reimplemented in plain torch (TRIDENT's
  uses einops): pooling `head` in `sipe.model.encoders`.
- resolution: `Resize(448)` (bilinear) of the 224 px tile, then ImageNet
  normalization. Here the tile is normalized at 224 px and upsampled inside
  `forward_features` (bilinear, on the normalized tensor; normalization is affine per
  channel, so only PIL's uint8 rounding differs), so every SIPE pipeline keeps feeding
  224 px tiles and the field of view is unchanged.
- position embedding: TRIDENT resamples the checkpoint's 28x28 grid to its model grid
  14x14 at load (bilinear, no antialias) and interpolates it back to 28x28 at every
  448 px forward (DINO-style bicubic with scale factor (28 + 0.1) / 14). Both steps
  are applied once at load (`trident_pos_embed`), into a 448 px model.
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from timm.data import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD
from timm.layers import resample_abs_pos_embed
from timm.models.vision_transformer import VisionTransformer

HF_REPO = "MahmoodLab/conchv1_5"
WEIGHTS = "pytorch_model_vision.bin"
TILE_SIZE = 224  # input of the SIPE pipelines (0.5 um/px tiles)
MODEL_SIZE = 448  # TRIDENT's img_size for conch_v15
TRIDENT_GRID = 14  # TRIDENT's model grid (its VisionTransformer default img_size 224)


class AttentionalPooler(nn.Module):
    """TRIDENT/CoCa `AttentionalPooler` with one query; [B, N, C] -> [B, d_model]."""

    def __init__(self, d_model: int = 768, context_dim: int = 1024, n_head: int = 8):
        super().__init__()
        self.query = nn.Parameter(torch.randn(1, d_model))
        self.heads = n_head
        self.scale = (d_model // n_head) ** -0.5
        self.ln_k = nn.LayerNorm(context_dim)
        self.ln_q = nn.LayerNorm(d_model)
        self.to_q = nn.Linear(d_model, d_model, bias=False)
        self.to_kv = nn.Linear(context_dim, d_model * 2, bias=False)
        self.to_out = nn.Linear(d_model, d_model, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, n, _ = x.shape
        h = self.heads
        q = self.to_q(self.ln_q(self.query))  # [1, D]
        k, v = self.to_kv(self.ln_k(x)).chunk(2, dim=-1)  # [B, N, D] each
        q = q.view(1, h, 1, -1).expand(b, -1, -1, -1)  # [B, h, 1, d]
        k = k.view(b, n, h, -1).transpose(1, 2)  # [B, h, N, d]
        v = v.view(b, n, h, -1).transpose(1, 2)
        attn = (q * self.scale @ k.transpose(-1, -2)).softmax(dim=-1)  # [B, h, 1, N]
        out = (attn @ v).transpose(1, 2).reshape(b, -1)  # [B, D]
        return self.to_out(out)


class ConchV15Head(nn.Module):
    """Official CONCH v1.5 embedding from the trunk's `forward_features` tokens."""

    def __init__(self) -> None:
        super().__init__()
        self.attn_pool_contrast = AttentionalPooler()
        self.ln_contrast = nn.LayerNorm(768)
        self.dim = 768

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        return self.ln_contrast(self.attn_pool_contrast(tokens))


class ConchV15Trunk(VisionTransformer):
    """ViT-L/16 at 448 px that takes 224 px tiles (upsampled as TRIDENT's Resize)."""

    def forward_features(self, x: torch.Tensor) -> torch.Tensor:
        if x.shape[-1] != MODEL_SIZE or x.shape[-2] != MODEL_SIZE:
            x = F.interpolate(
                x, size=(MODEL_SIZE, MODEL_SIZE), mode="bilinear", align_corners=False
            )
        return super().forward_features(x)


def trident_pos_embed(pos_embed: torch.Tensor) -> torch.Tensor:
    """Checkpoint pos embed [1, 1 + 28*28, D] -> the one TRIDENT uses at 448 px."""
    small = resample_abs_pos_embed(
        pos_embed,
        new_size=(TRIDENT_GRID, TRIDENT_GRID),
        num_prefix_tokens=1,
        interpolation="bilinear",
        antialias=False,
        verbose=False,
    )
    # TRIDENT conchv1_5.VisionTransformer.interpolate_pos_encoding (w = h = 448).
    cls, patch = small[:, :1], small[:, 1:]
    dim = patch.shape[-1]
    n = int(math.sqrt(patch.shape[1]))
    w0 = MODEL_SIZE // 16 + 0.1
    patch = F.interpolate(
        patch.reshape(1, n, n, dim).permute(0, 3, 1, 2),
        scale_factor=(w0 / n, w0 / n),
        mode="bicubic",
    )
    assert patch.shape[-2:] == (MODEL_SIZE // 16, MODEL_SIZE // 16), patch.shape
    patch = patch.permute(0, 2, 3, 1).reshape(1, -1, dim)
    return torch.cat((cls, patch), dim=1)


def build_conch_v15(pretrained: bool = True) -> nn.Module:
    """Trunk with the official head attached as `pool_head` (pooling `head`)."""
    backbone = ConchV15Trunk(
        img_size=MODEL_SIZE,
        patch_size=16,
        embed_dim=1024,
        depth=24,
        num_heads=16,
        mlp_ratio=4,
        qkv_bias=True,
        init_values=1.0,
        num_classes=0,
    )
    # Data config of the pipelines' input: 224 px tiles, ImageNet statistics.
    backbone.pretrained_cfg = {
        "hf_hub_id": HF_REPO,
        "mean": IMAGENET_DEFAULT_MEAN,
        "std": IMAGENET_DEFAULT_STD,
        "input_size": (3, TILE_SIZE, TILE_SIZE),
        "crop_pct": 1.0,
        "crop_mode": "center",
        "interpolation": "bilinear",
    }
    head = ConchV15Head()
    if pretrained:
        from huggingface_hub import hf_hub_download

        path = hf_hub_download(HF_REPO, WEIGHTS)
        state = torch.load(path, map_location="cpu", weights_only=True)
        trunk = {
            k[len("trunk.") :]: v for k, v in state.items() if k.startswith("trunk.")
        }
        trunk["pos_embed"] = trident_pos_embed(trunk["pos_embed"])
        backbone.load_state_dict(trunk, strict=True)
        head.load_state_dict(
            {k: v for k, v in state.items() if not k.startswith("trunk.")}, strict=True
        )
    backbone.pool_head = head
    return backbone
