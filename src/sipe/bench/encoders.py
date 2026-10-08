"""Tile-level feature extractors over a trained CATS network (plain torch).

Representations:
- `z_gap` (default, the benchmarked CATS representation): GAP of the spatial z map.
- `s`: the global scanner-specific code.
- `backbone_gap`: GAP of the backbone patch tokens (prefix tokens dropped); the
  reference for `z_gap` under the same weights and normalization.
- `backbone_cls`: the backbone CLS token, what the official H0-mini entries of PLISM
  and HEST use; a leaderboard sanity row.
"""

from __future__ import annotations

from collections.abc import Callable

import torch
import torch.nn as nn
import torchvision.transforms as T

from sipe.model.arch import CATS
from sipe.model.encoders import EncoderMeta

REPRESENTATIONS = ("z_gap", "s", "backbone_gap", "backbone_cls")


class CATSFeatures(nn.Module):
    """[B, 3, H, W] normalized images -> [B, D] features."""

    def __init__(self, network: CATS, representation: str = "z_gap") -> None:
        super().__init__()
        if representation not in REPRESENTATIONS:
            raise ValueError(
                f"Unknown representation {representation!r}. Known: {REPRESENTATIONS}."
            )
        self.network = network
        self.representation = representation

    @property
    def output_dim(self) -> int:
        return {
            "z_gap": self.network.unspecified_dim,
            "s": self.network.specified_dim,
            "backbone_gap": self.network.feature_dim,
            "backbone_cls": self.network.feature_dim,
        }[self.representation]

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        if self.representation == "z_gap":
            return self.network.encode(images)[1].mean(dim=(2, 3))
        if self.representation == "s":
            return self.network.encode(images)[0]
        if self.representation == "backbone_gap":
            return self.network.backbone_feature_map(images).mean(dim=(2, 3))
        return self.network.backbone.forward_features(images)[:, 0]


def plism_transform(meta: EncoderMeta) -> Callable:
    """ToTensor + Normalize, as plismbench's own H0-mini extractor (224px tiles)."""
    return T.Compose([T.ToTensor(), T.Normalize(mean=meta.mean, std=meta.std)])


def hest_transform(network: CATS) -> Callable:
    """The timm eval transform from the encoder builder (TRIDENT uses the same)."""
    return network.eval_transform
