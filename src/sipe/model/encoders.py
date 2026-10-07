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
import torch.nn as nn
from timm.data import create_transform, resolve_model_data_config


@dataclass(frozen=True)
class EncoderSpec:
    hf_id: str
    timm_kwargs: dict[str, Any] = field(default_factory=dict)


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
}


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
    )


def build_encoder(name: str, pretrained: bool = True) -> EncoderBundle:
    """Return the backbone, its timm eval transform and its metadata."""
    spec = _spec(name)
    backbone = timm.create_model(
        spec.hf_id,
        pretrained=pretrained,
        **spec.timm_kwargs,
    )
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
    backbone = timm.create_model(spec.hf_id, pretrained=False, **spec.timm_kwargs)
    return _meta(name, spec, backbone)
