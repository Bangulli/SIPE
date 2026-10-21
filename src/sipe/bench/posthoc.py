"""Post-hoc estimators on frozen backbone features: a fitted map f(x) on a pooled
backbone embedding.

The benchmarked representation is `posthoc_<pooling>`: the frozen backbone's GAP over
the patch tokens (prefix tokens dropped, = `backbone_gap`; `posthoc_gap`) or its CLS
token (`posthoc_cls`), then a map fitted on SCORPION train and saved as one torch file
in a `runs/bench/...` dir. Inference
takes one embedding and the global fit parameters only (no scanner ID, no pair).

Map files (torch.save of a dict), told apart by their keys:
- affine: `P` [D, D] and `bias` [D] -> `bias + (x - bias) P^T`, the vendored
  `SoftDeltaProjectionEraser.__call__` with a dense P (the soft-shrinkage eraser files
  `eraser_<pool>_lam<λ>.pt` of `sipe.eval.soft_shrinkage_sweep`);
- module: `class_path` ("pkg.mod.Class", an nn.Module mapping [B, D] -> [B, D']),
  `init_args` and `state_dict`, for later estimators.
Both may carry `encoder` (default h0-mini) and `pooling` ("gap" default, or "cls");
they must match the feature cache the map was fitted on. An identity affine map
(`identity_map`, P = I, bias = 0) gives the raw pooled embedding through the same path.

The backbone runs under the caller's autocast (plismbench's PrecisionModule, HEST's
precision), as in the SCORPION feature cache; the map always runs in float32.
"""

from __future__ import annotations

import importlib
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn

from sipe.bench.checkpoint import _read_git_txt, sha256
from sipe.model.encoders import POOLINGS, EncoderMeta, build_encoder, pool_backbone

REPRESENTATION = "posthoc_gap"  # the default pooling; see `representation`


def representation(pooling: str) -> str:
    return f"posthoc_{pooling}"


class AffineMap(nn.Module):
    """x -> bias + (x - bias) P^T."""

    def __init__(self, P: torch.Tensor, bias: torch.Tensor) -> None:
        super().__init__()
        if P.ndim != 2 or bias.shape != (P.shape[1],):
            raise ValueError(f"Bad affine shapes: P {P.shape}, bias {bias.shape}.")
        self.register_buffer("P", P.float())
        self.register_buffer("bias", bias.float())

    @property
    def in_dim(self) -> int:
        return int(self.P.shape[1])

    @property
    def out_dim(self) -> int:
        return int(self.P.shape[0])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return (x - self.bias) @ self.P.T + self.bias


def _import(path: str) -> type:
    module_name, _, name = path.rpartition(".")
    return getattr(importlib.import_module(module_name), name)


@dataclass
class LoadedMap:
    map: nn.Module
    kind: str
    encoder: str
    pooling: str
    provenance: dict[str, Any] = field(default_factory=dict)


def load_posthoc_map(path: str | Path) -> LoadedMap:
    """Load a fitted map file (eval mode, float32, CPU) + provenance."""
    path = Path(path).resolve()
    raw = torch.load(path, map_location="cpu", weights_only=True)
    if "P" in raw and "bias" in raw:
        kind = "affine"
        fmap: nn.Module = AffineMap(raw["P"], raw["bias"])
    elif "class_path" in raw:
        kind = "module"
        cls = _import(raw["class_path"])
        if not issubclass(cls, nn.Module):
            raise TypeError(f"{raw['class_path']} is not an nn.Module.")
        fmap = cls(**raw.get("init_args", {}))
        fmap.load_state_dict(raw["state_dict"])
    else:
        raise ValueError(f"{path}: neither an affine map (P, bias) nor a class_path.")
    pooling = raw.get("pooling", "gap")
    if pooling not in POOLINGS:
        raise ValueError(f"{path}: pooling {pooling!r} not supported ({POOLINGS}).")
    fit_run_dir = path.parent
    scalars = {
        k: v
        for k, v in raw.items()
        if isinstance(v, (int, float, str, bool)) or k.endswith("_config")
    }
    provenance = {
        "kind": "posthoc",
        "path": str(path),
        "sha256": sha256(path),
        "map": kind,
        "class_path": raw.get("class_path", f"{__name__}.AffineMap"),
        "file": scalars,
        "fit_run_dir": str(fit_run_dir),
        "fit_git": _read_git_txt(fit_run_dir / "git.txt"),
    }
    return LoadedMap(
        map=fmap.eval().float(),
        kind=kind,
        encoder=str(raw.get("encoder", "h0-mini")),
        pooling=pooling,
        provenance=provenance,
    )


class PosthocFeatures(nn.Module):
    """[B, 3, H, W] normalized images -> pooled backbone embedding -> fitted map."""

    def __init__(
        self,
        backbone: nn.Module,
        fmap: nn.Module,
        output_dim: int,
        pooling: str = "gap",
    ) -> None:
        super().__init__()
        self.backbone = backbone
        self.map = fmap
        self.output_dim = output_dim
        self.pooling = pooling

    def pool(self, images: torch.Tensor) -> torch.Tensor:
        # As sipe.features.extract.extract (the SCORPION feature cache).
        tokens = self.backbone.forward_features(images)
        return pool_backbone(self.backbone, tokens, self.pooling)

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        x = self.pool(images)
        with torch.autocast(x.device.type, enabled=False):
            return self.map(x)


@dataclass
class LoadedPosthoc:
    features: PosthocFeatures
    meta: EncoderMeta
    eval_transform: Any
    provenance: dict[str, Any]


def load_posthoc(path: str | Path, pretrained: bool = True) -> LoadedPosthoc:
    """Backbone (timm/HF weights) + fitted map, in eval mode."""
    loaded = load_posthoc_map(path)
    bundle = build_encoder(loaded.encoder, pretrained=pretrained)
    with torch.no_grad():
        dim = bundle.meta.pooled_dim(loaded.pooling)
        probe = torch.zeros(1, dim)
        try:
            output_dim = int(loaded.map(probe).shape[1])
        except RuntimeError as err:
            raise ValueError(
                f"{path}: map does not take {loaded.encoder} embeddings "
                f"(dim {dim}); wrong `encoder` in the map file?"
            ) from err
    features = PosthocFeatures(
        bundle.backbone, loaded.map, output_dim, loaded.pooling
    ).eval()
    provenance = {
        **loaded.provenance,
        "encoder": loaded.encoder,
        "hf_id": bundle.meta.hf_id,
        "weights": "pretrained (timm/HF)" if pretrained else "random (test)",
        "pooling": loaded.pooling,
    }
    return LoadedPosthoc(features, bundle.meta, bundle.transform, provenance)


def identity_map(encoder: str, pooling: str) -> dict[str, Any]:
    """Map file dict of the identity (P = I, bias = 0): the raw pooled embedding."""
    dim = build_encoder(encoder, pretrained=False).meta.pooled_dim(pooling)
    return {
        "P": torch.eye(dim),
        "bias": torch.zeros(dim),
        "encoder": encoder,
        "pooling": pooling,
        "identity": True,
    }
