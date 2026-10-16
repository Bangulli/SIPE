"""SCORPION feature caches: frozen backbone GAP/CLS per tile, optionally augmented.

`extract` is the one backbone pass behind every cache (moved unchanged from
`sipe.eval.soft_shrinkage_sweep`, which still imports it): fp16 autocast as
plismbench's PrecisionModule, GAP = mean of the patch tokens without prefix tokens
(= `backbone_gap`), CLS = token 0.

The CLI writes augmented train caches for the CGM augmentation study
(docs/experiment7.md; docs/experiment-log.md, Exp 7 history). Every
augmented view goes through the whole encoder (image -> augmentation -> H0-mini ->
GAP), never through a transformed token grid of a clean pass. `--copies` K: each tile is
extracted K times (meta column `aug` = copy index). All scanner views of a tile share
the transform of a copy (seed = crc32(pair_id | copy | seed), the same idea as
`PairedSCORPIONDataset`'s seed replay), so views differ only by scanner.

Policies (built from `SCORPIONDataModule._build_transform(train=True)`, so they are
the training transforms of the configs):
- `none`: Resize + Normalize (= the eval transform; copy 0 reproduces the GAP cache);
- `geo`: flips + rot90 + zoom, the CGM / `configs/consensus_gen.yaml` policy;
- `zoom`: zoom only (square crop of 80-100% of the area, resized back);
- `dihedral`: exact, not random: copy c = rotation by (c mod 4) x 90 degrees, then a
  horizontal flip if c >= 4 (K <= 8; K = 8 is the whole group).

    uv run python -m sipe.features.extract --tag augcache_geo --policy geo --copies 4
"""

from __future__ import annotations

import argparse
import logging
import zlib
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
import torchvision.transforms as T
from PIL import Image
from torch.utils.data import DataLoader, Dataset

from sipe.bench.provenance import new_bench_run_dir, write_run_info
from sipe.data.scorpion import SCORPIONDataModule, SCORPIONTileDataset
from sipe.model.encoders import build_encoder

log = logging.getLogger(__name__)

POOLINGS = ("gap", "cls")
# Datamodule flags per random policy: (horizontal_flip, vertical_flip, augmentations).
RANDOM_POLICIES = {
    "none": (False, False, []),
    "geo": (True, True, ["rot90", "zoom"]),
    "zoom": (False, False, ["zoom"]),
}
POLICIES = (*RANDOM_POLICIES, "dihedral")


@torch.no_grad()
def extract(
    backbone: torch.nn.Module,
    loader: DataLoader,
    device: torch.device,
    mixed_precision: bool,
) -> tuple[dict[str, np.ndarray], pd.DataFrame]:
    """One backbone pass -> {"gap", "cls"} features and the proxy's tile metadata."""
    out: dict[str, list[np.ndarray]] = {name: [] for name in POOLINGS}
    rows: list[pd.DataFrame] = []
    for batch in loader:
        images = batch["image"].to(device, non_blocking=True)
        # float16 autocast as plismbench's PrecisionModule (and the proxy).
        with torch.autocast(device.type, dtype=torch.float16, enabled=mixed_precision):
            tokens = backbone.forward_features(images)
        out["gap"].append(
            tokens[:, backbone.num_prefix_tokens :].mean(dim=1).float().cpu().numpy()
        )
        out["cls"].append(tokens[:, 0].float().cpu().numpy())
        frame = pd.DataFrame(
            {
                "pair_id": batch["pair_id"],
                "scanner": batch["domain_name"],
                "domain": batch["domain"].numpy(),
            }
        )
        if "aug" in batch:  # augmented caches only
            frame["aug"] = batch["aug"].numpy()
        rows.append(frame)
    meta = pd.concat(rows, ignore_index=True)
    meta["slide_id"] = meta["pair_id"].str.split("-").str[0]
    meta["region_id"] = meta["pair_id"].str.partition("-tile_")[0]
    meta["row"] = np.arange(len(meta))
    return {k: np.concatenate(v) for k, v in out.items()}, meta


class Dihedral:
    """Element c of the dihedral group D4 on a PIL image (exact pixel permutation)."""

    _rot = (None, Image.Transpose.ROTATE_90, Image.Transpose.ROTATE_180,
            Image.Transpose.ROTATE_270)  # fmt: skip

    def __init__(self, element: int) -> None:
        if not 0 <= element < 8:
            raise ValueError(f"dihedral element must be in [0, 8), not {element}.")
        self.element = element

    def __call__(self, image: Image.Image) -> Image.Image:
        rot = self._rot[self.element % 4]
        if rot is not None:
            image = image.transpose(rot)
        if self.element >= 4:
            image = image.transpose(Image.Transpose.FLIP_LEFT_RIGHT)
        return image

    def __repr__(self) -> str:
        return f"{type(self).__name__}({self.element})"


def view_seed(pair_id: str, copy: int, seed: int) -> int:
    """Seed shared by the scanner views of one tile copy (stable across runs)."""
    return zlib.crc32(f"{pair_id}|{copy}|{seed}".encode())


class AugmentedViews(Dataset):
    """(tile row, copy) items of a SCORPIONTileDataset, copy-major order.

    `transforms[c]` is applied to copy c (one shared transform for random policies,
    one fixed transform per copy for `dihedral`) under the seed `view_seed`.
    """

    def __init__(
        self, base: SCORPIONTileDataset, transforms: list[Any], seed: int
    ) -> None:
        super().__init__()
        self.base = base
        self.transforms = transforms
        self.seed = int(seed)
        self.pair_ids = base.metadata["_pair_id"].astype(str).tolist()

    def __len__(self) -> int:
        return len(self.transforms) * len(self.base)

    def __getitem__(self, idx: int) -> dict[str, Any]:
        copy, row = divmod(idx, len(self.base))
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(view_seed(self.pair_ids[row], copy, self.seed))
            self.base.transform = self.transforms[copy]
            item = self.base[row]
        item["aug"] = torch.tensor(copy, dtype=torch.long)
        return item


def policy_transforms(
    dm_hparams: dict[str, Any], policy: str, copies: int
) -> list[Any]:
    """One transform per copy for `policy` (see the module docstring)."""
    if policy == "dihedral":
        if copies > 8:
            raise ValueError("dihedral has 8 elements: --copies <= 8.")
        dm = SCORPIONDataModule(**dm_hparams, augmentations=[])
        base = dm._build_transform(train=False).transforms  # Resize, ToTensor, Norm
        return [T.Compose([base[0], Dihedral(c), *base[1:]]) for c in range(copies)]
    hflip, vflip, augs = RANDOM_POLICIES[policy]
    dm = SCORPIONDataModule(
        **dm_hparams, horizontal_flip=hflip, vertical_flip=vflip, augmentations=augs
    )
    return [dm._build_transform(train=True)] * copies


def main(argv: list[str] | None = None) -> Path:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--tag", required=True)
    parser.add_argument("--policy", required=True, choices=POLICIES)
    parser.add_argument("--copies", type=int, default=1)
    parser.add_argument("--splits", nargs="+", default=["train"],
                        choices=["train", "val", "test"])  # fmt: skip
    parser.add_argument("--poolings", nargs="+", default=["gap"], choices=POOLINGS)
    parser.add_argument("--encoder", default="h0-mini")
    parser.add_argument(
        "--data-root", default="data/processed/SCORPION_tiles_224px_0p5mpp"
    )
    parser.add_argument("--split-seed", type=int, default=42)
    parser.add_argument("--seed", type=int, default=0, help="Augmentation seed.")
    parser.add_argument(
        "--limit-slides", type=int, help="Debug: slides per split (smoke test)."
    )
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--workers", type=int, default=8, help="0 in claude-box.")
    parser.add_argument("--device", type=int, default=0, help="CUDA index, -1 = CPU.")
    parser.add_argument(
        "--no-mixed-precision", dest="mixed_precision", action="store_false"
    )
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    device = torch.device("cpu" if args.device < 0 else f"cuda:{args.device}")
    if args.copies < 1:
        raise ValueError("--copies must be >= 1.")

    # The split of every SCORPION datamodule and of the GAP cache.
    dm_hparams = dict(
        data_root=args.data_root,
        encoder=args.encoder,
        group_column="slide_id",
        val_fraction=0.1,
        test_fraction=0.1,
        seed=args.split_seed,
        batch_size=args.batch_size,
        num_workers=args.workers,
        persistent_workers=False,
    )
    transforms = policy_transforms(dm_hparams, args.policy, args.copies)
    run_dir = new_bench_run_dir(args.tag)
    write_run_info(
        run_dir,
        config={
            "benchmark": "scorpion_feature_cache",
            **{k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()},
            "datamodule": dm_hparams,
            "transforms": [repr(t) for t in dict.fromkeys(transforms)],
            "shared_across_views": "seed = crc32(pair_id|copy|seed) per tile copy",
        },
        checkpoint={"encoder": args.encoder, "weights": "pretrained (timm/HF)"},
        features={
            "poolings": args.poolings,
            "precision": "autocast float16" if args.mixed_precision else "float32",
        },
    )

    dm = SCORPIONDataModule(**dm_hparams)
    dm.setup()
    datasets = {"train": dm.train_dataset, "val": dm.val_dataset,
                "test": dm.test_dataset}  # fmt: skip
    backbone = build_encoder(args.encoder).backbone.to(device).eval()
    for split in args.splits:
        dataset = datasets[split]
        if args.limit_slides is not None:
            slides = dataset.metadata["slide_id"]
            keep = slides.isin(sorted(slides.unique())[: args.limit_slides])
            dataset.metadata = dataset.metadata.loc[keep].reset_index(drop=True)
        views = AugmentedViews(dataset, transforms, args.seed)
        log.info("Extracting %s: %d tiles x %d copies (%s)", split, len(dataset),
                 args.copies, args.policy)  # fmt: skip
        loader = DataLoader(
            views,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=args.workers,
            pin_memory=device.type == "cuda",
        )
        feats, meta = extract(backbone, loader, device, args.mixed_precision)
        np.savez(run_dir / f"features_{split}.npz",
                 **{p: feats[p] for p in args.poolings})  # fmt: skip
        meta.to_csv(run_dir / f"meta_{split}.csv", index=False)
    log.info("Feature cache: %s", run_dir)
    return run_dir


if __name__ == "__main__":
    main()
