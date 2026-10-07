from __future__ import annotations

from pathlib import Path

import pytest
import timm
import torch
import torch.nn as nn
import torchvision.transforms as T
from timm.data import resolve_model_data_config

from sipe.data.scorpion import SCORPIONDataModule
from sipe.model.encoders import build_encoder, encoder_meta

H0_MINI_MEAN = (0.707223, 0.578729, 0.703617)
H0_MINI_STD = (0.211883, 0.230117, 0.177517)


def _normalize_op(transform: T.Compose) -> T.Normalize:
    ops = [t for t in transform.transforms if isinstance(t, T.Normalize)]
    assert len(ops) == 1
    return ops[0]


def test_h0_mini_transform_matches_timm() -> None:
    bundle = build_encoder("h0-mini", pretrained=False)
    reference = timm.create_model(
        "hf-hub:bioptimus/H0-mini",
        pretrained=False,
        mlp_layer=timm.layers.SwiGLUPacked,
        act_layer=nn.SiLU,
    )
    expected = resolve_model_data_config(reference)

    normalize = _normalize_op(bundle.transform)
    assert torch.allclose(
        torch.as_tensor(normalize.mean), torch.tensor(expected["mean"])
    )
    assert torch.allclose(
        torch.as_tensor(normalize.std), torch.tensor(expected["std"])
    )
    assert bundle.meta.input_size == tuple(expected["input_size"])
    assert bundle.meta.mean == pytest.approx(H0_MINI_MEAN, abs=1e-5)
    assert bundle.meta.std == pytest.approx(H0_MINI_STD, abs=1e-5)


def test_h0_mini_meta() -> None:
    meta = encoder_meta("h0-mini")
    bundle = build_encoder("h0-mini", pretrained=False)
    assert meta == bundle.meta
    assert meta.embed_dim == 768
    assert meta.patch_size == 14
    assert meta.input_size[1] // meta.patch_size == 16
    assert meta.num_prefix_tokens == bundle.backbone.num_prefix_tokens


def test_unknown_encoder() -> None:
    with pytest.raises(ValueError, match="Unknown encoder"):
        encoder_meta("nope")


def test_datamodule_normalization_follows_encoder(tmp_path: Path) -> None:
    dm = SCORPIONDataModule(data_root=tmp_path, encoder="h0-mini")
    meta = encoder_meta("h0-mini")
    assert dm.mean == meta.mean
    assert dm.std == meta.std
    for train in (True, False):
        normalize = dm._build_transform(train=train).transforms[-1]
        assert isinstance(normalize, T.Normalize)
        assert tuple(normalize.mean) == meta.mean
        assert tuple(normalize.std) == meta.std
