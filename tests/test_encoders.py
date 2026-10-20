from __future__ import annotations

import copy
from pathlib import Path

import numpy as np
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


def test_augmentations_default_is_legacy_and_configurable(tmp_path: Path) -> None:
    def ops(dm: SCORPIONDataModule, train: bool) -> list[type]:
        return [type(t) for t in dm._build_transform(train=train).transforms]

    legacy = [
        T.Resize, T.RandomHorizontalFlip, T.RandomVerticalFlip, T.GaussianBlur,
        T.RandomAffine, T.RandomGrayscale, T.RandomInvert, T.ToTensor,
        T.RandomErasing, T.Normalize,
    ]  # fmt: skip
    assert ops(SCORPIONDataModule(data_root=tmp_path), True) == legacy
    geometric = SCORPIONDataModule(data_root=tmp_path, augmentations=["affine"])
    assert ops(geometric, True) == [
        T.Resize, T.RandomHorizontalFlip, T.RandomVerticalFlip, T.RandomAffine,
        T.ToTensor, T.Normalize,
    ]  # fmt: skip
    plain = [T.Resize, T.ToTensor, T.Normalize]
    assert ops(geometric, False) == plain
    assert ops(SCORPIONDataModule(data_root=tmp_path, augmentations=[]), False) == plain
    with pytest.raises(ValueError, match="unknown augmentations"):
        SCORPIONDataModule(data_root=tmp_path, augmentations=["jitter"])


def _structured_root(root: Path) -> Path:
    """2 slides x 2 locations x 5 scanners, the same random image per location."""
    import pandas as pd
    from PIL import Image

    rows = []
    for slide in (1, 2):
        for tile in ("0_0", "0_1"):
            rng = np.random.default_rng(slide * 10 + int(tile[-1]))
            pixels = rng.integers(0, 256, (32, 32, 3), dtype=np.uint8)
            for scanner in ("AT2", "DP200", "GT450", "P1000", "Philips"):
                stem = f"slide_{slide}-sample_1-tile_{tile}-{scanner}"
                Image.fromarray(pixels).save(root / f"{stem}.png")
                rows.append({"slide_id": f"slide_{slide}", "sample_id": "sample_1",
                             "scanner_id": scanner, "tile_id": stem.rsplit("-", 1)[0],
                             "filename": f"{stem}.png"})  # fmt: skip
    pd.DataFrame(rows).to_csv(root / "metadata.csv", index=False)
    return root


def test_rot90_is_exact_and_zoom_keeps_size() -> None:
    from PIL import Image

    from sipe.data.scorpion import PIL_AUGMENTATIONS, RandomRot90

    img = Image.fromarray(
        torch.randint(0, 256, (16, 16, 3), dtype=torch.uint8).numpy()
    )
    pixels = torch.as_tensor(np.asarray(img))
    seen = set()
    for seed in range(20):
        torch.manual_seed(seed)
        out = torch.as_tensor(np.asarray(RandomRot90()(img)))
        seen.add(next(k for k in range(4) if torch.equal(out, torch.rot90(pixels, k))))
    assert seen == {0, 1, 2, 3}
    assert PIL_AUGMENTATIONS["zoom"](16)(img).size == (16, 16)


def test_paired_views_share_geometric_augmentation(tmp_path: Path) -> None:
    from sipe.data.scorpion import PairedSCORPIONDataModule

    dm = PairedSCORPIONDataModule(
        _structured_root(tmp_path), batch_size=2, views=5, num_workers=0,
        image_size=32, val_fraction=0.0, test_fraction=0.0,
        persistent_workers=False, augmentations=["rot90", "zoom"],
    )  # fmt: skip
    dm.setup("fit")
    plain_base = copy.copy(dm.train_dataset)
    plain_base.transform = dm._build_transform(train=False)
    changed = False
    for _ in range(5):
        for idx in range(len(dm.train_pairs)):
            views = dm.train_pairs[idx]
            for view in views[1:]:  # same pixels per scanner -> same output
                assert torch.equal(view["image"], views[0]["image"])
            original = plain_base[int(views[0]["index"])]["image"]
            changed |= not torch.equal(views[0]["image"], original)
    assert changed


def test_uni2h_meta() -> None:
    meta = encoder_meta("uni2-h")
    assert meta.embed_dim == 1536
    assert meta.patch_size == 14
    assert meta.num_prefix_tokens == 9  # CLS + 8 register tokens
    assert meta.mean == pytest.approx((0.485, 0.456, 0.406))
    assert meta.std == pytest.approx((0.229, 0.224, 0.225))


def test_pool_tokens() -> None:
    from sipe.model.encoders import pool_tokens

    tokens = torch.arange(2 * 5 * 3, dtype=torch.float16).reshape(2, 5, 3)
    gap = pool_tokens(tokens, 2, "gap")
    assert gap.dtype == torch.float32
    torch.testing.assert_close(gap, tokens[:, 2:].float().mean(dim=1))
    torch.testing.assert_close(pool_tokens(tokens, 2, "cls"), tokens[:, 0].float())
    with pytest.raises(ValueError, match="pooling"):
        pool_tokens(tokens, 2, "max")
