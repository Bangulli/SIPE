"""Rebuild a trained CATS / ScannerVAE / PairedVAE model from a checkpoint."""

from __future__ import annotations

import hashlib
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import torch

from sipe.model.arch import CATS
from sipe.training.cats_module import CATSModule
from sipe.training.curriculum import CurriculumPhase
from sipe.training.paired_vae_module import CHECKPOINT_TAG as PAIRED_VAE_TAG
from sipe.training.paired_vae_module import PairedVAEModule
from sipe.training.scanner_vae_module import CHECKPOINT_TAG, ScannerVAEModule

log = logging.getLogger(__name__)


@dataclass
class LoadedCATS:
    module: CATSModule | ScannerVAEModule | PairedVAEModule
    network: CATS | Any
    mean: tuple[float, ...]
    std: tuple[float, ...]
    image_size: int
    global_step: int
    # Curriculum position (CATSModule only; None for the VAE modules).
    phase_index: int | None
    phase: CurriculumPhase | None
    local_step: int | None
    provenance: dict[str, Any] = field(default_factory=dict)


def sha256(path: Path, chunk_size: int = 1 << 24) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as f:
        while chunk := f.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def _read_git_txt(path: Path) -> dict[str, Any] | None:
    """commit/dirty from a training run's git.txt (written by sipe.cli)."""
    if not path.is_file():
        return None
    info: dict[str, Any] = {"path": str(path)}
    for line in path.read_text().splitlines():
        if line.startswith("commit: "):
            info["commit"] = line.removeprefix("commit: ").strip()
        elif line.startswith("dirty: "):
            info["dirty"] = line.removeprefix("dirty: ").strip() == "True"
        elif not line.strip():
            break
    return info


def load_cats_checkpoint(
    ckpt_path: str | Path,
    map_location: str | torch.device = "cpu",
) -> LoadedCATS:
    """Load a CATS / ScannerVAE / PairedVAE module checkpoint (eval mode) + provenance.

    The module is rebuilt from the hparams that LightningCLI saved (network as
    {class_path, init_args}) and loaded strictly. With `pretrained: true` timm first
    loads the backbone from the HF cache; the checkpoint weights then overwrite it.
    """
    ckpt_path = Path(ckpt_path).resolve()
    raw = torch.load(ckpt_path, map_location="cpu", weights_only=True, mmap=True)
    tag = raw.get("sipe_module")
    if tag in _VAE_MODULES:
        return _load_vae(ckpt_path, raw, map_location, tag)

    module = CATSModule.load_from_checkpoint(ckpt_path, map_location=map_location)
    module.eval()
    network = module.network
    global_step = int(raw["global_step"])
    curriculum = module.curriculum
    if global_step > curriculum.total_steps:
        raise ValueError(
            f"{ckpt_path}: global_step={global_step} is past the curriculum end "
            f"({curriculum.total_steps})."
        )
    # global_step counts completed updates: the last one belongs to step - 1
    # (same convention as CATSModule._phase_for_stage for validation).
    phase_index, phase, local_step = curriculum.at(max(global_step - 1, 0))
    if phase.mode == "recon":
        log.warning(
            "Checkpoint %s is in the reconstruction-only phase %r: its disentangler "
            "has never been trained adversarially.",
            ckpt_path,
            phase.name,
        )

    meta = network.encoder_meta
    run_dir = ckpt_path.parent.parent
    provenance = {
        "path": str(ckpt_path),
        "sha256": sha256(ckpt_path),
        "global_step": global_step,
        "curriculum_total_steps": curriculum.total_steps,
        "epoch": int(raw["epoch"]),
        "phase_index": phase_index,
        "phase_name": phase.name,
        "phase_mode": phase.mode,
        "phase_local_step": local_step + 1 if global_step > 0 else 0,
        "phase_steps": phase.steps,
        "network": raw["hyper_parameters"]["network"],
        "train_run_dir": str(run_dir),
        "train_git": _read_git_txt(run_dir / "git.txt"),
    }
    return LoadedCATS(
        module=module,
        network=network,
        mean=meta.mean,
        std=meta.std,
        image_size=meta.input_size[-1],
        global_step=global_step,
        phase_index=phase_index,
        phase=phase,
        local_step=local_step,
        provenance=provenance,
    )


_VAE_MODULES = {CHECKPOINT_TAG: ScannerVAEModule, PAIRED_VAE_TAG: PairedVAEModule}


def _load_vae(
    ckpt_path: Path, raw: dict[str, Any], map_location: str | torch.device, tag: str
) -> LoadedCATS:
    module = _VAE_MODULES[tag].load_from_checkpoint(
        ckpt_path, map_location=map_location
    )
    module.eval()
    network = module.network
    meta = network.encoder_meta
    run_dir = ckpt_path.parent.parent
    global_step = int(raw["global_step"])
    provenance = {
        "path": str(ckpt_path),
        "sha256": sha256(ckpt_path),
        "module": tag,
        "global_step": global_step,
        "epoch": int(raw["epoch"]),
        "network": raw["hyper_parameters"]["network"],
        "train_run_dir": str(run_dir),
        "train_git": _read_git_txt(run_dir / "git.txt"),
    }
    return LoadedCATS(
        module=module,
        network=network,
        mean=meta.mean,
        std=meta.std,
        image_size=meta.input_size[-1],
        global_step=global_step,
        phase_index=None,
        phase=None,
        local_step=None,
        provenance=provenance,
    )
