from __future__ import annotations

from pathlib import Path

import lightning as L
import pytest
import torch
import yaml

from sipe.model.arch import CATS
from sipe.training.cats_module import CATSModule

CONFIG = Path(__file__).resolve().parents[1] / "configs" / "cats_legacy_steps.yaml"


def _write_cats_checkpoint(run_dir: Path, global_step: int) -> Path:
    """A checkpoint shaped like a LightningCLI one, without data or HF weights.

    hparams hold `network` as {class_path, init_args} plus `_instantiator`, as
    SIPECLI writes them; the random weights stand in for trained ones.
    """
    model_cfg = yaml.safe_load(CONFIG.read_text())["model"]["init_args"]
    network_args = {**model_cfg.pop("network"), "pretrained": False}
    torch.manual_seed(0)
    module = CATSModule(network=CATS(**network_args), **model_cfg)
    # Make the disentangler distinguishable from a fresh init.
    with torch.no_grad():
        for p in module.network.disentangler.parameters():
            p.add_(0.01)
    hparams = {
        "network": {"class_path": "sipe.model.arch.CATS", "init_args": network_args},
        **model_cfg,
        "_instantiator": "lightning.pytorch.cli.instantiate_module",
    }
    ckpt_dir = run_dir / "checkpoints"
    ckpt_dir.mkdir(parents=True)
    path = ckpt_dir / "last.ckpt"
    torch.save(
        {
            "epoch": 0,
            "global_step": global_step,
            "pytorch-lightning_version": L.__version__,
            "state_dict": module.state_dict(),
            "hparams_name": "kwargs",
            "hyper_parameters": hparams,
        },
        path,
    )
    (run_dir / "git.txt").write_text("commit: abc123\ndirty: False\n")
    return path


@pytest.fixture
def make_cats_checkpoint(tmp_path):
    def make(global_step: int = 600) -> Path:
        return _write_cats_checkpoint(tmp_path / f"run_{global_step}", global_step)

    return make
