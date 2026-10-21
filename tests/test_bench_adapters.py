"""PLISM/HEST adapters. Bench env only: bench/run pytest tests/test_bench_adapters.py"""

from __future__ import annotations

import numpy as np
import pytest
import torch


def test_plism_factory(make_cats_checkpoint):
    pytest.importorskip("plismbench")
    from plismbench.models import init_extractor
    from plismbench.models.extractor import Extractor

    ckpt = make_cats_checkpoint(600)
    extractor = init_extractor(
        "sipe.bench.plism:cats_extractor",
        device=-1,
        extractor_kwargs={"ckpt": str(ckpt), "mixed_precision": False},
    )
    assert isinstance(extractor, Extractor)
    assert extractor.output_dim == 704

    tiles = [np.random.randint(0, 256, (224, 224, 3), dtype=np.uint8) for _ in range(3)]
    images = torch.stack([extractor.transform(t) for t in tiles])
    features = extractor(images)
    assert isinstance(features, np.ndarray)
    assert features.shape == (3, 704)
    assert features.dtype == np.float32


def test_hest_encoder_duck_type(make_cats_checkpoint):
    pytest.importorskip("hest")
    from PIL import Image

    from sipe.bench.checkpoint import load_cats_checkpoint
    from sipe.bench.encoders import CATSFeatures
    from sipe.bench.hest import HESTEncoder

    loaded = load_cats_checkpoint(make_cats_checkpoint(600))
    encoder = HESTEncoder(CATSFeatures(loaded.network), loaded.network.eval_transform)
    # The checks hest.bench.benchmark uses to accept a TRIDENT-style encoder.
    assert hasattr(encoder, "eval_transforms") and hasattr(encoder, "precision")
    assert encoder.precision == torch.float16

    tile = Image.fromarray(np.random.randint(0, 256, (224, 224, 3), dtype=np.uint8))
    x = encoder.eval_transforms(tile)
    assert x.shape == (3, 224, 224)
    with torch.inference_mode():
        out = encoder.eval()(x[None])
    assert out.shape == (1, 704)


def test_hest_resolve_datasets_skips_hidden_entries(tmp_path):
    pytest.importorskip("hest")
    from sipe.bench.hest import resolve_datasets

    for name in ("IDC", "PAAD", ".cache"):
        (tmp_path / name).mkdir()
    (tmp_path / "IDC" / "splits").mkdir()
    (tmp_path / "PAAD" / "splits").mkdir()
    (tmp_path / ".gitattributes").touch()
    assert resolve_datasets(tmp_path, ["*"]) == ["IDC", "PAAD"]
    assert resolve_datasets(tmp_path, ["LUNG"]) == ["LUNG"]


def test_plism_posthoc_factory(tmp_path):
    pytest.importorskip("plismbench")
    from plismbench.models import init_extractor

    path = tmp_path / "eraser_gap_lam0.5.pt"
    torch.save(
        {"P": torch.eye(768), "bias": torch.zeros(768), "pooling": "gap",
         "encoder": "h0-mini"},
        path,
    )  # fmt: skip
    extractor = init_extractor(
        "sipe.bench.plism:posthoc_extractor",
        device=-1,
        extractor_kwargs={
            "posthoc": str(path), "mixed_precision": False, "pretrained": False,
        },
    )  # fmt: skip
    assert extractor.output_dim == 768
    tiles = [np.random.randint(0, 256, (224, 224, 3), dtype=np.uint8) for _ in range(2)]
    features = extractor(torch.stack([extractor.transform(t) for t in tiles]))
    assert features.shape == (2, 768) and features.dtype == np.float32


def test_conch_v15_matches_trident():
    """sipe's timm CONCH v1.5 = TRIDENT encoder_factory("conch_v15") (the authors'
    recommended loader). Needs TRIDENT (bench env) and access to the gated weights."""
    pytest.importorskip("trident")
    import torch.nn.functional as F
    from PIL import Image, ImageFilter
    from trident.patch_encoder_models import encoder_factory

    from sipe.model.encoders import build_encoder, pool_backbone

    try:
        reference = encoder_factory("conch_v15").eval()
        ours = build_encoder("conch-v1.5")
    except Exception as err:  # no HF access / offline
        pytest.skip(f"CONCH v1.5 weights unavailable: {err}")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    reference.to(device)
    backbone = ours.backbone.eval().to(device)
    rng = np.random.default_rng(0)
    tiles = [Image.fromarray(rng.integers(0, 256, (224, 224, 3), dtype=np.uint8))]
    tiles.append(tiles[0].filter(ImageFilter.GaussianBlur(3)))  # smooth tile
    x_ref = torch.stack([reference.eval_transforms(t) for t in tiles]).to(device)
    x_ours = torch.stack([ours.transform(t) for t in tiles]).to(device)
    with torch.no_grad():
        expected = reference(x_ref).float()
        # Same 448 px input: position embedding, trunk and head match exactly.
        same_input = pool_backbone(backbone, backbone.forward_features(x_ref), "head")
        torch.testing.assert_close(same_input, expected, atol=1e-4, rtol=1e-4)
        # SIPE input (224 px tile, upsampled inside): only PIL's uint8 rounding differs.
        tile = pool_backbone(backbone, backbone.forward_features(x_ours), "head")
    assert F.cosine_similarity(tile, expected).min() > 0.999
