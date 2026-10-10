from __future__ import annotations

import os
import secrets
import socket
import subprocess
from datetime import datetime
from pathlib import Path
from typing import Literal

import lightning as L
import torch
from lightning.pytorch.callbacks import ModelCheckpoint
from lightning.pytorch.cli import LightningCLI, SaveConfigCallback
from lightning.pytorch.loggers import WandbLogger

from sipe.training.cats_module import CATSModule

# Set by the launching process so DDP children spawned by Lightning reuse its run dir.
RUN_DIR_ENV = "SIPE_RUN_DIR"


def _git(*args: str, cwd: Path | None = None) -> str:
    """Run git in `cwd` (default: the SIPE repo)."""
    cwd = cwd if cwd is not None else Path(__file__).resolve().parent
    return subprocess.run(
        ["git", *args], cwd=cwd, capture_output=True, text=True
    ).stdout


def write_git_info(path: Path, repo: Path | None = None) -> None:
    """Write commit, dirty flag and (if dirty) the diff of `repo` (default: SIPE)."""
    status = _git("status", "--porcelain", cwd=repo)
    lines = [
        f"commit: {_git('rev-parse', 'HEAD', cwd=repo).strip()}",
        f"dirty: {bool(status.strip())}",
    ]
    if status.strip():
        lines += [
            "",
            "# git status --porcelain",
            status,
            "# git diff HEAD",
            _git("diff", "HEAD", cwd=repo),
        ]
    path.write_text("\n".join(lines) + "\n")


def run_dir_name() -> str:
    """<YYYY-MM-DD_HH-MM-SS>_<host>_<rand>, shared by training and bench run dirs."""
    stamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    return f"{stamp}_{socket.gethostname()}_{secrets.token_hex(2)}"


class RunDirSaveConfigCallback(SaveConfigCallback):
    """Save the resolved config into the run dir, not the logger save_dir."""

    def save_config(
        self, trainer: L.Trainer, pl_module: L.LightningModule, stage: str
    ) -> None:
        self.parser.save(
            self.config,
            Path(trainer.default_root_dir) / self.config_filename,
            skip_none=False,
            overwrite=self.overwrite,
            multifile=self.multifile,
        )


class SIPECLI(LightningCLI):
    """Each `fit` gets runs/<YYYY-MM-DD_HH-MM-SS>_<host>_<rand>/.

    It holds checkpoints/, config.yaml, git.txt (commit, dirty flag, diff) and,
    with a WandbLogger, wandb.txt plus the local wandb files. The W&B run is named
    after the run dir (unless a name is configured) and gets `run_dir` in its
    config, so a run picked on the W&B website maps back to its checkpoints.
    """

    def add_arguments_to_parser(self, parser) -> None:
        # float32 matmul precision on Tensor Core GPUs. "highest" is the torch default
        # (no TF32); "high" enables TF32. Kept in the config so each run records it.
        parser.add_argument(
            "--matmul_precision",
            type=Literal["highest", "high", "medium"],
            default="highest",
        )
        # The backbone fixes the input normalization: the datamodule reads mean/std
        # from the same encoder builder, so training and benchmarks cannot diverge.
        parser.link_arguments(
            "model.network.init_args.encoder", "data.init_args.encoder"
        )

    def before_instantiate_classes(self) -> None:
        config = self.config[self.subcommand] if self.subcommand else self.config
        torch.set_float32_matmul_precision(config.matmul_precision)
        if self.subcommand != "fit":
            return
        run_dir = os.environ.get(RUN_DIR_ENV)
        if run_dir is None:
            run_dir = f"runs/{run_dir_name()}"
            Path(run_dir).mkdir(parents=True)
            write_git_info(Path(run_dir) / "git.txt")
            os.environ[RUN_DIR_ENV] = run_dir
        self.config.fit.trainer.default_root_dir = run_dir
        for logger_cfg in _wandb_logger_configs(self.config.fit.trainer.logger):
            init_args = logger_cfg.init_args
            init_args.save_dir = run_dir
            if init_args.get("name") is None:
                init_args.name = Path(run_dir).name

    def before_fit(self) -> None:
        for cb in self.trainer.checkpoint_callbacks:
            if isinstance(cb, ModelCheckpoint) and cb.dirpath is None:
                cb.dirpath = str(Path(self.trainer.default_root_dir) / "checkpoints")
        if not self.trainer.is_global_zero:
            return
        run_dir = Path(self.trainer.default_root_dir).resolve()
        for logger in self.trainer.loggers:
            if isinstance(logger, WandbLogger):
                run = logger.experiment
                run.config.update({"run_dir": str(run_dir)}, allow_val_change=True)
                (run_dir / "wandb.txt").write_text(
                    f"id: {run.id}\n"
                    f"path: {run.entity}/{run.project}/{run.id}\n"
                    f"url: {run.url}\n"
                    f"local: {run.dir}\n"
                )


def _wandb_logger_configs(logger_cfg):
    """WandbLogger entries of a trainer.logger config (bool, single or list)."""
    entries = logger_cfg if isinstance(logger_cfg, list) else [logger_cfg]
    return [
        e
        for e in entries
        if getattr(e, "class_path", None) is not None
        and e.class_path.split(".")[-1] == "WandbLogger"
    ]


def main() -> None:
    _run(CATSModule)


def main_vae() -> None:
    """`sipe-vae`: same run-dir/W&B handling, ScannerVAEModule as the model."""
    from sipe.training.scanner_vae_module import ScannerVAEModule

    _run(ScannerVAEModule)


def main_paired() -> None:
    """`sipe-paired`: same run-dir/W&B handling, PairedVAEModule as the model."""
    from sipe.training.paired_vae_module import PairedVAEModule

    _run(PairedVAEModule)


def _run(model_class: type[L.LightningModule]) -> None:
    SIPECLI(
        model_class=model_class,
        datamodule_class=L.LightningDataModule,
        subclass_mode_data=True,
        seed_everything_default=42,
        save_config_callback=RunDirSaveConfigCallback,
        save_config_kwargs={"overwrite": True, "save_to_log_dir": False},
    )


if __name__ == "__main__":
    main()
