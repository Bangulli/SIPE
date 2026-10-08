"""Bench run dirs runs/bench/<stamp>_<host>_<rand>_<tag>/: config and provenance."""

from __future__ import annotations

import importlib.metadata
import importlib.util
import json
import tomllib
from pathlib import Path
from typing import Any

import yaml

from sipe.cli import _git, run_dir_name, write_git_info

SIPE_ROOT = Path(__file__).resolve().parents[3]
BENCH_PYPROJECT = SIPE_ROOT / "bench" / "pyproject.toml"
BENCH_RUNS = SIPE_ROOT / "runs" / "bench"

# Benchmark repos, located through their installed (editable) package.
BENCH_PACKAGES = {"hest": "hest", "plism-benchmark": "plismbench"}
VERSIONED = (
    "torch",
    "torchvision",
    "timm",
    "lightning",
    "transformers",
    "trident",
    "hest",
    "owkin-plismbench",
)


class _NoAliasDumper(yaml.SafeDumper):
    """Write repeated objects in full (no &id001 / *id001 aliases)."""

    def ignore_aliases(self, data: Any) -> bool:
        return True


def new_bench_run_dir(tag: str, root: Path = BENCH_RUNS) -> Path:
    run_dir = root / f"{run_dir_name()}_{tag}"
    run_dir.mkdir(parents=True)
    return run_dir


def git_state(repo: Path) -> dict[str, Any]:
    return {
        "path": str(repo),
        "commit": _git("rev-parse", "HEAD", cwd=repo).strip(),
        "branch": _git("rev-parse", "--abbrev-ref", "HEAD", cwd=repo).strip(),
        "dirty": bool(_git("status", "--porcelain", cwd=repo).strip()),
    }


def _package_repo(package: str) -> Path | None:
    spec = importlib.util.find_spec(package)
    if spec is None or spec.origin is None:
        return None
    top = _git("rev-parse", "--show-toplevel", cwd=Path(spec.origin).parent).strip()
    return Path(top) if top else None


def _versions() -> dict[str, str | None]:
    versions: dict[str, str | None] = {}
    for dist in VERSIONED:
        try:
            versions[dist] = importlib.metadata.version(dist)
        except importlib.metadata.PackageNotFoundError:
            versions[dist] = None
    return versions


def _bench_overrides() -> list[str]:
    with open(BENCH_PYPROJECT, "rb") as f:
        return tomllib.load(f)["tool"]["uv"]["override-dependencies"]


def write_run_info(
    run_dir: Path,
    config: dict[str, Any],
    checkpoint: dict[str, Any],
    features: dict[str, Any],
) -> dict[str, Any]:
    """Write config.yaml, git.txt (+ git_<repo>.txt per bench repo), provenance.json."""
    (run_dir / "config.yaml").write_text(
        yaml.dump(config, Dumper=_NoAliasDumper, sort_keys=False)
    )
    write_git_info(run_dir / "git.txt")
    repos = {"sipe": git_state(SIPE_ROOT)}
    for name, package in BENCH_PACKAGES.items():
        repo = _package_repo(package)
        if repo is not None:
            repos[name] = git_state(repo)
            write_git_info(run_dir / f"git_{name}.txt", repo=repo)
    provenance = {
        "run_dir": str(run_dir.resolve()),
        "checkpoint": checkpoint,
        "features": features,
        "repos": repos,
        "versions": _versions(),
        "bench_overrides": _bench_overrides(),
    }
    (run_dir / "provenance.json").write_text(json.dumps(provenance, indent=2) + "\n")
    return provenance
