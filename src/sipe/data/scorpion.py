from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import Any

import lightning as L
import pandas as pd
import torch
import torchvision.transforms as T
from PIL import Image
from torch.utils.data import DataLoader, Dataset

from sipe.model.encoders import encoder_meta

from .utils import (
    filename_from_row,
    infer_domain,
    infer_pair_id,
    infer_sample_id,
    infer_slide_id,
    resolve_tile_path,
)


class SCORPIONTileDataset(Dataset):
    """Single-view SCORPION tile dataset.

    ``pair_id`` is exposed already so a future paired-view dataset can group
    acquisitions of the same tissue location across scanners without changing
    the metadata convention.
    """

    def __init__(
        self,
        *,
        data_root: Path,
        metadata: pd.DataFrame,
        domain_to_index: Mapping[str, int],
        transform: Any = None,
        path_column: str = "path",
        filename_column: str = "filename",
        domain_column: str = "scanner",
    ) -> None:
        super().__init__()
        self.data_root = Path(data_root)
        self.metadata = metadata.reset_index(drop=True).copy()
        self.domain_to_index = dict(domain_to_index)
        self.transform = transform
        self.path_column = path_column
        self.filename_column = filename_column
        self.domain_column = domain_column

    def __len__(self) -> int:
        return len(self.metadata)

    def __getitem__(self, idx: int) -> dict[str, Any]:
        row = self.metadata.iloc[idx]

        tile_path = resolve_tile_path(
            self.data_root,
            row,
            path_column=self.path_column,
            filename_column=self.filename_column,
        )

        with Image.open(tile_path) as image_file:
            image = image_file.convert("RGB")

        if self.transform is not None:
            image = self.transform(image)

        filename = filename_from_row(
            row,
            path_column=self.path_column,
            filename_column=self.filename_column,
        )

        domain_name = (
            str(row[self.domain_column])
            if self.domain_column in row.index and pd.notna(row[self.domain_column])
            else infer_domain(filename)
        )

        try:
            domain_index = self.domain_to_index[domain_name]
        except KeyError as exc:
            raise KeyError(
                f"Unknown domain {domain_name!r}. Known domains: "
                f"{tuple(self.domain_to_index)}"
            ) from exc

        pair_id = (
            str(row["_pair_id"]) if "_pair_id" in row.index else infer_pair_id(filename)
        )

        return {
            "image": image,
            "domain": torch.tensor(domain_index, dtype=torch.long),
            "index": torch.tensor(idx, dtype=torch.long),
            "pair_id": pair_id,
            "domain_name": domain_name,
            "filename": filename,
        }


class SCORPIONDataModule(L.LightningDataModule):
    """LightningDataModule for SCORPION tiles.

    Expected layout::

        data_root/
            metadata.csv
            slide_1-sample_1-tile_0_0-AT2.jpg
            slide_1-sample_1-tile_0_0-DP200.jpg
            ...

    If ``split_column`` exists in metadata it is used directly. Otherwise the
    data are split deterministically by ``group_column`` (``slide_id`` by
    default), preventing scanner views or tiles from the same WSI from leaking
    across splits.
    """

    def __init__(
        self,
        data_root: str | Path,
        *,
        metadata_filename: str = "metadata.csv",
        batch_size: int = 16,
        num_workers: int = 8,
        image_size: int = 224,
        encoder: str = "h0-mini",
        horizontal_flip: bool = True,
        vertical_flip: bool = True,
        path_column: str = "path",
        filename_column: str = "filename",
        domain_column: str = "scanner",
        split_column: str | None = "split",
        group_column: str | None = "slide_id",
        val_fraction: float = 0.1,
        test_fraction: float = 0.1,
        seed: int = 42,
        pin_memory: bool = True,
        persistent_workers: bool = True,
        drop_last: bool = True,
    ) -> None:
        super().__init__()
        self.save_hyperparameters()

        if batch_size <= 0:
            raise ValueError("batch_size must be positive.")
        if num_workers < 0:
            raise ValueError("num_workers must be non-negative.")
        if image_size <= 0:
            raise ValueError("image_size must be positive.")
        if not 0.0 <= val_fraction < 1.0:
            raise ValueError("val_fraction must be in [0, 1).")
        if not 0.0 <= test_fraction < 1.0:
            raise ValueError("test_fraction must be in [0, 1).")
        if val_fraction + test_fraction >= 1.0:
            raise ValueError("val_fraction + test_fraction must be < 1.")

        self.data_root = Path(data_root)
        self.metadata_path = self.data_root / metadata_filename

        self.batch_size = int(batch_size)
        self.num_workers = int(num_workers)
        self.image_size = int(image_size)
        # Normalization follows the backbone's timm data config (linked from
        # model.network.init_args.encoder by the CLI).
        self.encoder = encoder
        meta = encoder_meta(encoder)
        self.mean = meta.mean
        self.std = meta.std
        self.horizontal_flip = bool(horizontal_flip)
        self.vertical_flip = bool(vertical_flip)

        self.path_column = path_column
        self.filename_column = filename_column
        self.domain_column = domain_column
        self.split_column = split_column
        self.group_column = group_column

        self.val_fraction = float(val_fraction)
        self.test_fraction = float(test_fraction)
        self.seed = int(seed)

        self.pin_memory = bool(pin_memory)
        self.persistent_workers = bool(persistent_workers)
        self.drop_last = bool(drop_last)

        self.domain_names: tuple[str, ...] = ()
        self.domain_to_index: dict[str, int] = {}

        self.train_dataset: SCORPIONTileDataset | None = None
        self.val_dataset: SCORPIONTileDataset | None = None
        self.test_dataset: SCORPIONTileDataset | None = None

        self.train_metadata: pd.DataFrame | None = None
        self.val_metadata: pd.DataFrame | None = None
        self.test_metadata: pd.DataFrame | None = None

        self._prepared_metadata: pd.DataFrame | None = None

    @property
    def num_domains(self) -> int:
        if not self.domain_names:
            raise RuntimeError(
                "setup() must be called before num_domains is available."
            )
        return len(self.domain_names)

    def _build_transform(self, *, train: bool) -> T.Compose:
        transforms: list[Any] = [
            T.Resize((self.image_size, self.image_size)),
        ]

        if train and self.horizontal_flip:
            transforms.append(T.RandomHorizontalFlip())
        if train and self.vertical_flip:
            transforms.append(T.RandomVerticalFlip())

        if train:
            transforms.extend(
                [
                    T.GaussianBlur(3),
                    T.RandomAffine(3),
                    T.RandomGrayscale(p=0.1),
                    T.RandomInvert(p=0.5),
                ]
            )

        transforms.append(T.ToTensor())

        if train:
            transforms.append(T.RandomErasing(p=0.5))

        transforms.append(T.Normalize(mean=self.mean, std=self.std))

        return T.Compose(transforms)

    def _prepare_metadata(self) -> pd.DataFrame:
        if not self.metadata_path.exists():
            raise FileNotFoundError(f"Metadata file not found: {self.metadata_path}")

        metadata = pd.read_csv(self.metadata_path).copy()
        if metadata.empty:
            raise RuntimeError(f"Metadata is empty: {self.metadata_path}")

        filenames = metadata.apply(
            lambda row: filename_from_row(
                row,
                path_column=self.path_column,
                filename_column=self.filename_column,
            ),
            axis=1,
        )

        if self.domain_column not in metadata.columns:
            metadata[self.domain_column] = filenames.map(infer_domain)
        else:
            missing = metadata[self.domain_column].isna()
            if missing.any():
                metadata.loc[missing, self.domain_column] = filenames[missing].map(
                    infer_domain
                )

        metadata[self.domain_column] = metadata[self.domain_column].astype(str)

        metadata["_pair_id"] = filenames.map(infer_pair_id)
        metadata["_sample_id"] = filenames.map(infer_sample_id)
        metadata["_slide_id"] = filenames.map(infer_slide_id)

        if self.group_column is not None and self.group_column not in metadata.columns:
            inferred = {
                "slide_id": "_slide_id",
                "sample_id": "_sample_id",
                "pair_id": "_pair_id",
            }
            if self.group_column not in inferred:
                raise KeyError(
                    f"group_column={self.group_column!r} is absent from metadata. "
                    "Automatic inference is available for 'slide_id', 'sample_id', "
                    "and 'pair_id'."
                )
            metadata[self.group_column] = metadata[inferred[self.group_column]]

        return metadata

    def _split_metadata(
        self,
        metadata: pd.DataFrame,
    ) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        if self.split_column is not None and self.split_column in metadata.columns:
            split = (
                metadata[self.split_column]
                .astype(str)
                .str.lower()
                .replace({"validation": "val", "valid": "val", "dev": "val"})
            )

            train = metadata.loc[split == "train"].copy()
            val = metadata.loc[split == "val"].copy()
            test = metadata.loc[split == "test"].copy()

            if train.empty:
                raise RuntimeError(
                    f"{self.split_column!r} exists but contains no 'train' rows."
                )
            return train, val, test

        if self.group_column is None:
            groups = pd.Series(
                [f"row_{i}" for i in range(len(metadata))],
                index=metadata.index,
            )
        else:
            groups = metadata[self.group_column].astype(str)

        unique_groups = sorted(groups.unique().tolist())
        n_groups = len(unique_groups)
        if n_groups < 1:
            raise RuntimeError("No groups available for splitting.")

        generator = torch.Generator().manual_seed(self.seed)
        order = torch.randperm(n_groups, generator=generator).tolist()
        shuffled = [unique_groups[i] for i in order]

        n_test = round(n_groups * self.test_fraction)
        n_val = round(n_groups * self.val_fraction)

        if self.test_fraction > 0 and n_test == 0 and n_groups >= 3:
            n_test = 1
        if self.val_fraction > 0 and n_val == 0 and n_groups - n_test >= 2:
            n_val = 1

        while n_test + n_val >= n_groups and n_test > 0:
            n_test -= 1
        while n_test + n_val >= n_groups and n_val > 0:
            n_val -= 1

        test_groups = set(shuffled[:n_test])
        val_groups = set(shuffled[n_test : n_test + n_val])
        train_groups = set(shuffled[n_test + n_val :])

        train = metadata.loc[groups.isin(train_groups)].copy()
        val = metadata.loc[groups.isin(val_groups)].copy()
        test = metadata.loc[groups.isin(test_groups)].copy()

        if train.empty:
            raise RuntimeError("Training split is empty.")

        return train, val, test

    def setup(self, stage: str | None = None) -> None:
        del stage

        if self._prepared_metadata is not None:
            return

        metadata = self._prepare_metadata()

        self.domain_names = tuple(
            sorted(metadata[self.domain_column].astype(str).unique().tolist())
        )
        self.domain_to_index = {
            name: index for index, name in enumerate(self.domain_names)
        }

        train, val, test = self._split_metadata(metadata)

        self._prepared_metadata = metadata
        self.train_metadata = train.reset_index(drop=True)
        self.val_metadata = val.reset_index(drop=True)
        self.test_metadata = test.reset_index(drop=True)

        common = dict(
            data_root=self.data_root,
            domain_to_index=self.domain_to_index,
            path_column=self.path_column,
            filename_column=self.filename_column,
            domain_column=self.domain_column,
        )

        self.train_dataset = SCORPIONTileDataset(
            metadata=self.train_metadata,
            transform=self._build_transform(train=True),
            **common,
        )
        self.val_dataset = SCORPIONTileDataset(
            metadata=self.val_metadata,
            transform=self._build_transform(train=False),
            **common,
        )
        self.test_dataset = SCORPIONTileDataset(
            metadata=self.test_metadata,
            transform=self._build_transform(train=False),
            **common,
        )

    def _loader(
        self,
        dataset: Dataset,
        *,
        shuffle: bool,
        drop_last: bool,
    ) -> DataLoader:
        return DataLoader(
            dataset,
            batch_size=self.batch_size,
            shuffle=shuffle,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            persistent_workers=self.persistent_workers and self.num_workers > 0,
            drop_last=drop_last,
        )

    def train_dataloader(self) -> DataLoader:
        if self.train_dataset is None:
            raise RuntimeError("setup() has not been called.")
        return self._loader(
            self.train_dataset,
            shuffle=True,
            drop_last=self.drop_last,
        )

    def val_dataloader(self) -> DataLoader:
        if self.val_dataset is None:
            raise RuntimeError("setup() has not been called.")
        return self._loader(self.val_dataset, shuffle=False, drop_last=False)

    def test_dataloader(self) -> DataLoader:
        if self.test_dataset is None:
            raise RuntimeError("setup() has not been called.")
        return self._loader(self.test_dataset, shuffle=False, drop_last=False)

    def predict_dataloader(self) -> DataLoader:
        return self.test_dataloader()


class PairedSCORPIONDataset(Dataset):
    """One item = one tissue location (``pair_id``) seen by several scanners.

    Returns a list of ``views`` tile dicts (``SCORPIONTileDataset`` items) from
    distinct scanners: a random subset when ``random_views`` (training), otherwise a
    deterministic rotation (all scanners when ``views`` is None). All views of an
    item get the *same* random augmentation (shared torch RNG seed), so they differ
    only by scanner; torchvision's v1 transforms draw from the torch RNG.
    """

    def __init__(
        self,
        base: SCORPIONTileDataset,
        *,
        views: int | None = 2,
        random_views: bool = True,
    ) -> None:
        super().__init__()
        if views is not None and views < 2:
            raise ValueError("views must be >= 2 (or None for all scanners).")
        self.base = base
        self.views = views
        self.random_views = random_views
        metadata = base.metadata
        domains = metadata[base.domain_column].map(base.domain_to_index)
        self.locations: list[list[int]] = []
        for rows in metadata.groupby("_pair_id", sort=True).indices.values():
            rows = sorted(rows.tolist(), key=lambda r: domains.iloc[r])
            if len(rows) >= (views or 2):
                self.locations.append(rows)
        if not self.locations:
            raise RuntimeError("No location has enough scanner views.")

    def __len__(self) -> int:
        return len(self.locations)

    def __getitem__(self, idx: int) -> list[dict[str, Any]]:
        rows = self.locations[idx]
        n = len(rows)
        k = self.views or n
        if self.random_views:
            chosen = [rows[i] for i in torch.randperm(n)[:k].tolist()]
        else:
            chosen = [rows[(idx + i) % n] for i in range(k)]
        seed = int(torch.randint(0, 2**31 - 1, (1,)))
        items = []
        for row in chosen:
            with torch.random.fork_rng(devices=[]):
                torch.manual_seed(seed)
                items.append(self.base[row])
        return items


def collate_views(batch: list[list[dict[str, Any]]]) -> dict[str, Any]:
    """Flatten locations × views into one batch; ``location`` groups the views.

    Views of a location are contiguous: image i belongs to location
    ``location[i]``.
    """
    flat = [item for items in batch for item in items]
    return {
        "image": torch.stack([item["image"] for item in flat]),
        "domain": torch.stack([item["domain"] for item in flat]),
        "index": torch.stack([item["index"] for item in flat]),
        "location": torch.tensor(
            [loc for loc, items in enumerate(batch) for _ in items], dtype=torch.long
        ),
        "pair_id": [item["pair_id"] for item in flat],
        "domain_name": [item["domain_name"] for item in flat],
        "filename": [item["filename"] for item in flat],
    }


class PairedSCORPIONDataModule(SCORPIONDataModule):
    """SCORPION batches of locations × scanner views (same splits as the parent).

    ``batch_size`` counts locations, so a training batch holds
    ``batch_size * views`` images. Validation/test use every scanner of each location
    (``val_views``: None = all), i.e. every tile of the split exactly once, with
    ``val_batch_size`` locations per batch (default: about as many images as a
    training batch).
    """

    def __init__(
        self,
        data_root: str | Path,
        *,
        views: int = 2,
        val_views: int | None = None,
        val_batch_size: int | None = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(data_root, **kwargs)
        self.views = int(views)
        self.val_views = val_views
        self.val_batch_size = val_batch_size
        self.train_pairs: PairedSCORPIONDataset | None = None
        self.val_pairs: PairedSCORPIONDataset | None = None
        self.test_pairs: PairedSCORPIONDataset | None = None

    def setup(self, stage: str | None = None) -> None:
        super().setup(stage)
        if self.train_pairs is not None:
            return
        self.train_pairs = PairedSCORPIONDataset(
            self.train_dataset, views=self.views, random_views=True
        )
        if len(self.val_metadata):
            self.val_pairs = PairedSCORPIONDataset(
                self.val_dataset, views=self.val_views, random_views=False
            )
        if len(self.test_metadata):
            self.test_pairs = PairedSCORPIONDataset(
                self.test_dataset, views=self.val_views, random_views=False
            )

    def _paired_loader(
        self,
        dataset: PairedSCORPIONDataset | None,
        *,
        batch_size: int,
        shuffle: bool,
        drop_last: bool,
    ) -> DataLoader:
        if dataset is None:
            raise RuntimeError("setup() has not been called or the split is empty.")
        return DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            persistent_workers=self.persistent_workers and self.num_workers > 0,
            drop_last=drop_last,
            collate_fn=collate_views,
        )

    def _eval_batch_size(self) -> int:
        if self.val_batch_size is not None:
            return int(self.val_batch_size)
        views = self.val_views or self.num_domains
        return max(1, self.batch_size * self.views // views)

    def train_dataloader(self) -> DataLoader:
        return self._paired_loader(
            self.train_pairs,
            batch_size=self.batch_size,
            shuffle=True,
            drop_last=self.drop_last,
        )

    def val_dataloader(self) -> DataLoader:
        return self._paired_loader(
            self.val_pairs,
            batch_size=self._eval_batch_size(),
            shuffle=False,
            drop_last=False,
        )

    def test_dataloader(self) -> DataLoader:
        return self._paired_loader(
            self.test_pairs,
            batch_size=self._eval_batch_size(),
            shuffle=False,
            drop_last=False,
        )
