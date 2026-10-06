from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest
import torch
from PIL import Image

from sipe.data.scorpion import SCORPIONDataModule
from sipe.data.utils import infer_domain, infer_slide_id
from sipe.losses.adversarial_classif_loss import AdversarialClassifLoss

SCANNERS = ("AT2", "DP200", "GT450", "P1000", "Philips")
REAL_DATA_ROOT = (
    Path(__file__).resolve().parents[1]
    / "data/processed/SCORPION_tiles_224px_0p5mpp"
)


def _make_synthetic_root(root: Path, n_slides: int = 10) -> Path:
    """Write tiny tiles + metadata in the real SCORPION naming/column layout."""
    rows = []
    for slide in range(1, n_slides + 1):
        for sample in (1, 2):
            for tile in ("0_0", "0_1"):
                for scanner_idx, scanner in enumerate(SCANNERS):
                    stem = f"slide_{slide}-sample_{sample}-tile_{tile}-{scanner}"
                    # Encode the scanner in the pixel value so a mix-up between
                    # image and label would be detectable.
                    Image.new("RGB", (8, 8), (scanner_idx * 40, 0, 0)).save(
                        root / f"{stem}.jpg"
                    )
                    rows.append(
                        {
                            "slide_id": f"slide_{slide}",
                            "sample_id": f"sample_{sample}",
                            "scanner_id": scanner,
                            "tile_id": stem.rsplit("-", 1)[0],
                            "filename": stem,
                        }
                    )
    pd.DataFrame(rows).to_csv(root / "metadata.csv", index=False)
    return root


def _datamodule(root: Path) -> SCORPIONDataModule:
    # Mirrors configs/cats_legacy_steps.yaml data settings.
    dm = SCORPIONDataModule(
        root,
        batch_size=4,
        num_workers=0,
        image_size=8,
        domain_column="scanner",
        group_column="slide_id",
        val_fraction=0.1,
        test_fraction=0.1,
        persistent_workers=False,
    )
    dm.setup("fit")
    return dm


def _assert_no_slide_leakage(dm: SCORPIONDataModule) -> None:
    splits = {
        "train": dm.train_metadata,
        "val": dm.val_metadata,
        "test": dm.test_metadata,
    }
    slides = {
        name: set(df["filename"].map(infer_slide_id)) for name, df in splits.items()
    }
    assert slides["train"] and slides["val"] and slides["test"]
    assert not slides["train"] & slides["val"]
    assert not slides["train"] & slides["test"]
    assert not slides["val"] & slides["test"]
    total = sum(len(df) for df in splits.values())
    assert total == len(dm._prepared_metadata)


# --- Dataset leakage ---------------------------------------------------------


def test_no_wsi_leakage_synthetic(tmp_path: Path) -> None:
    _assert_no_slide_leakage(_datamodule(_make_synthetic_root(tmp_path)))


@pytest.mark.skipif(
    not (REAL_DATA_ROOT / "metadata.csv").exists(), reason="SCORPION data absent"
)
def test_no_wsi_leakage_real_data() -> None:
    _assert_no_slide_leakage(_datamodule(REAL_DATA_ROOT))


# --- Scanner labels ----------------------------------------------------------


def test_scanner_labels_match_filename(tmp_path: Path) -> None:
    dm = _datamodule(_make_synthetic_root(tmp_path))
    assert dm.domain_names == tuple(sorted(SCANNERS))

    for dataset in (dm.train_dataset, dm.val_dataset, dm.test_dataset):
        for idx in range(len(dataset)):
            item = dataset[idx]
            scanner = infer_domain(item["filename"])
            assert item["domain_name"] == scanner
            assert dm.domain_names[int(item["domain"])] == scanner
            assert scanner == dataset.metadata.iloc[idx]["scanner_id"]


@pytest.mark.skipif(
    not (REAL_DATA_ROOT / "metadata.csv").exists(), reason="SCORPION data absent"
)
def test_scanner_labels_real_metadata() -> None:
    dm = _datamodule(REAL_DATA_ROOT)
    assert len(dm.domain_names) == 5
    md = dm._prepared_metadata
    # Label source used by the dataset (filename suffix) agrees with the
    # metadata's own scanner column for every tile.
    assert (md["scanner"] == md["scanner_id"].astype(str)).all()


def test_patch_expanded_targets_follow_sample() -> None:
    """z logits flattened as in CATSModule._domain_logits must line up with
    AdversarialClassifLoss's repeat_interleave target expansion."""
    torch.manual_seed(0)
    batch, n_domains, h, w = 6, len(SCANNERS), 16, 16
    domains = torch.randint(0, n_domains, (batch,))
    domains[0], domains[1] = 0, 1  # ensure at least two distinct labels

    # Per-patch logits that confidently predict each sample's own scanner.
    z_logits_map = torch.full((batch, n_domains, h, w), -20.0)
    z_logits_map[torch.arange(batch), domains] = 20.0
    z_logits = z_logits_map.permute(0, 2, 3, 1).reshape(-1, n_domains)

    targets = torch.nn.functional.one_hot(domains, n_domains).float()
    s_logits = targets * 40.0 - 20.0
    loss_fn = AdversarialClassifLoss(norm=False)

    aligned, _ = loss_fn(s_logits, z_logits, targets, "cpu", None, False, 1.0)
    assert aligned.item() < 1e-6

    # Same logits against shifted targets must be heavily penalized.
    shifted = torch.roll(targets, 1, 0)
    misaligned, _ = loss_fn(s_logits, z_logits, shifted, "cpu", None, False, 1.0)
    assert misaligned.item() > 10.0

    # The accuracy path in CATSModule uses expand(); it must match too.
    patch_targets = domains[:, None].expand(-1, h * w).reshape(-1)
    assert torch.equal(patch_targets, domains.repeat_interleave(h * w))
    assert (z_logits.argmax(-1) == patch_targets).all()
