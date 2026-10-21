"""Encoder builder: backbone, eval transform and metadata from one encoder name.

Training (normalization in the datamodule) and benchmarking both read the
preprocessing from here, so mean/std are never hard-coded elsewhere.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from functools import lru_cache
from typing import Any

import timm
import torch
import torch.nn as nn
from timm.data import create_transform, resolve_model_data_config

from sipe.model.conch import build_conch_v15


@dataclass(frozen=True)
class EncoderSpec:
    hf_id: str
    timm_kwargs: dict[str, Any] = field(default_factory=dict)
    # Custom builder (pretrained -> backbone) for models without a timm hub config.
    build: Callable[[bool], nn.Module] | None = None


ENCODERS: dict[str, EncoderSpec] = {
    "h0-mini": EncoderSpec(
        hf_id="hf-hub:bioptimus/H0-mini",
        timm_kwargs={
            "mlp_layer": timm.layers.SwiGLUPacked,
            "act_layer": nn.SiLU,
        },
    ),
    "h-optimus-1": EncoderSpec(
        hf_id="hf-hub:bioptimus/H-optimus-1",
        timm_kwargs={
            "init_values": 1e-5,
            "dynamic_img_size": False,
        },
    ),
    # Arguments of the UNI2-h model card (gated: needs an HF token with access); the
    # HF config only names vit_giant_patch14_224. 1 CLS + 8 register tokens.
    "uni2-h": EncoderSpec(
        hf_id="hf-hub:MahmoodLab/UNI2-h",
        timm_kwargs={
            "img_size": 224,
            "patch_size": 14,
            "depth": 24,
            "num_heads": 24,
            "init_values": 1e-5,
            "embed_dim": 1536,
            "mlp_ratio": 2.66667 * 2,
            "num_classes": 0,
            "no_embed_class": True,
            "mlp_layer": timm.layers.SwiGLUPacked,
            "act_layer": nn.SiLU,
            "reg_tokens": 8,
            "dynamic_img_size": True,
        },
    ),
    # Matched to TRIDENT's encoder_factory("conch_v15") (the authors' recommended
    # loader): 224 px tiles upsampled to 448 px inside the model; official embedding =
    # pooling "head" (attentional pooler, 768-d). See sipe.model.conch.
    "conch-v1.5": EncoderSpec(
        hf_id="hf-hub:MahmoodLab/conchv1_5", build=build_conch_v15
    ),
}

POOLINGS = ("gap", "cls", "head")


def pool_tokens(
    tokens: torch.Tensor, num_prefix_tokens: int, pooling: str
) -> torch.Tensor:
    """`forward_features` tokens [B, N, D] -> [B, D] float32.

    gap = mean of the patch tokens (prefix tokens, i.e. CLS and registers, dropped);
    cls = token 0 (after the final norm: timm's `global_pool="token"` output, the
    official embedding of CLS-pooled models such as UNI2-h). "head" needs the
    backbone's own pooling module: `pool_backbone`.
    """
    if pooling == "gap":
        return tokens[:, num_prefix_tokens:].mean(dim=1).float()
    if pooling == "cls":
        return tokens[:, 0].float()
    if pooling == "head":
        raise ValueError("pooling 'head' needs the backbone: use pool_backbone.")
    raise ValueError(f"Unknown pooling {pooling!r}. Known: {POOLINGS}.")


def pool_backbone(
    backbone: nn.Module, tokens: torch.Tensor, pooling: str
) -> torch.Tensor:
    """As `pool_tokens`, plus "head" = the backbone's official pooling module
    (`backbone.pool_head`, e.g. CONCH v1.5's attentional pooler), float32 output."""
    if pooling == "head":
        head = getattr(backbone, "pool_head", None)
        if head is None:
            raise ValueError("pooling 'head': this backbone has no pool_head.")
        return head(tokens).float()
    return pool_tokens(tokens, backbone.num_prefix_tokens, pooling)


@dataclass(frozen=True)
class EncoderMeta:
    name: str
    hf_id: str
    embed_dim: int
    patch_size: int
    num_prefix_tokens: int
    mean: tuple[float, ...]
    std: tuple[float, ...]
    input_size: tuple[int, int, int]
    interpolation: str
    crop_pct: float
    head_dim: int = 0  # output dim of pooling "head"; 0 = no pool_head

    def pooled_dim(self, pooling: str) -> int:
        """Dimension of the pooled embedding."""
        if pooling != "head":
            return self.embed_dim
        if not self.head_dim:
            raise ValueError(f"{self.name} has no pooling 'head'.")
        return self.head_dim


@dataclass
class EncoderBundle:
    backbone: nn.Module
    transform: Callable
    meta: EncoderMeta


def _spec(name: str) -> EncoderSpec:
    try:
        return ENCODERS[name]
    except KeyError:
        raise ValueError(
            f"Unknown encoder {name!r}. Known encoders: {sorted(ENCODERS)}."
        ) from None


def _meta(name: str, spec: EncoderSpec, backbone: nn.Module) -> EncoderMeta:
    data_config = resolve_model_data_config(backbone)
    patch_size = backbone.patch_embed.patch_size
    return EncoderMeta(
        name=name,
        hf_id=spec.hf_id,
        embed_dim=int(backbone.num_features),
        patch_size=int(patch_size[0] if isinstance(patch_size, tuple) else patch_size),
        num_prefix_tokens=int(backbone.num_prefix_tokens),
        mean=tuple(float(x) for x in data_config["mean"]),
        std=tuple(float(x) for x in data_config["std"]),
        input_size=tuple(int(x) for x in data_config["input_size"]),
        interpolation=str(data_config["interpolation"]),
        crop_pct=float(data_config["crop_pct"]),
        head_dim=int(getattr(getattr(backbone, "pool_head", None), "dim", 0)),
    )


def _create(spec: EncoderSpec, pretrained: bool) -> nn.Module:
    if spec.build is not None:
        return spec.build(pretrained)
    return timm.create_model(spec.hf_id, pretrained=pretrained, **spec.timm_kwargs)


def build_encoder(name: str, pretrained: bool = True) -> EncoderBundle:
    """Return the backbone, its timm eval transform and its metadata."""
    spec = _spec(name)
    backbone = _create(spec, pretrained)
    transform = create_transform(
        **resolve_model_data_config(backbone),
        is_training=False,
    )
    return EncoderBundle(
        backbone=backbone,
        transform=transform,
        meta=_meta(name, spec, backbone),
    )


@lru_cache
def encoder_meta(name: str) -> EncoderMeta:
    """Metadata without weights (timm reads the data config from the HF config)."""
    spec = _spec(name)
    backbone = _create(spec, pretrained=False)
    return _meta(name, spec, backbone)
