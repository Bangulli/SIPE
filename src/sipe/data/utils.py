from __future__ import annotations

from pathlib import Path

import pandas as pd


def resolve_tile_path(
    data_root: Path,
    row: pd.Series,
    *,
    path_column: str = "path",
    filename_column: str = "filename",
    default_suffix: str = ".jpg",
) -> Path:
    """Resolve a tile path from one metadata row."""
    if path_column in row.index and pd.notna(row[path_column]):
        tile_path = Path(str(row[path_column]))
    elif filename_column in row.index and pd.notna(row[filename_column]):
        tile_path = Path(str(row[filename_column]))
        if tile_path.suffix == "":
            tile_path = tile_path.with_suffix(default_suffix)
    else:
        raise KeyError(
            f"Metadata row contains neither a usable {path_column!r} nor "
            f"{filename_column!r}."
        )

    if not tile_path.is_absolute():
        tile_path = data_root / tile_path

    if not tile_path.exists():
        raise FileNotFoundError(f"Tile not found: {tile_path}")

    return tile_path


def infer_pair_id(tile_id: str) -> str:
    """Drop the final scanner/domain suffix from a tile filename."""
    stem = Path(str(tile_id)).stem
    pair_id, sep, _ = stem.rpartition("-")
    if not sep:
        raise ValueError(
            f"Cannot infer pair ID from {tile_id!r}; expected a final '-DOMAIN' suffix."
        )
    return pair_id


def infer_domain(tile_id: str) -> str:
    """Infer scanner/domain from the final filename component."""
    stem = Path(str(tile_id)).stem
    _, sep, domain = stem.rpartition("-")
    if not sep or not domain:
        raise ValueError(
            f"Cannot infer domain from {tile_id!r}; expected a final '-DOMAIN' suffix."
        )
    return domain


def infer_sample_id(tile_id: str) -> str:
    """Infer sample ID by stripping the '-tile_*' suffix."""
    pair_id = infer_pair_id(tile_id)
    sample_id, sep, _ = pair_id.partition("-tile_")
    return sample_id if sep else pair_id


def infer_slide_id(tile_id: str) -> str:
    """Infer WSI/slide ID from the filename prefix."""
    stem = Path(str(tile_id)).stem
    slide_id, sep, _ = stem.partition("-sample_")
    if sep:
        return slide_id

    pair_id = infer_pair_id(tile_id)
    return pair_id.split("-", maxsplit=1)[0]


def filename_from_row(
    row: pd.Series,
    *,
    path_column: str = "path",
    filename_column: str = "filename",
) -> str:
    """Return the metadata filename/path string used for ID inference."""
    if filename_column in row.index and pd.notna(row[filename_column]):
        return str(row[filename_column])
    if path_column in row.index and pd.notna(row[path_column]):
        return Path(str(row[path_column])).name
    raise KeyError(
        f"Metadata row contains neither {filename_column!r} nor {path_column!r}."
    )
