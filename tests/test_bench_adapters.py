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
