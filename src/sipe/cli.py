from __future__ import annotations

import os
import secrets
import socket
import subprocess
from datetime import datetime
from pathlib import Path

import lightning as L
from lightning.pytorch.callbacks import ModelCheckpoint
from lightning.pytorch.cli import LightningCLI, SaveConfigCallback

from sipe.training.cats_module import CATSModule

# Set by the launching process so DDP children spawned by Lightning reuse its run dir.
RUN_DIR_ENV = "SIPE_RUN_DIR"


def _git(*args: str) -> str:
    cwd = Path(__file__).resolve().parent
    return subprocess.run(
        ["git", *args], cwd=cwd, capture_output=True, text=True
    ).stdout


def write_git_info(path: Path) -> None:
    status = _git("status", "--porcelain")
    lines = [
        f"commit: {_git('rev-parse', 'HEAD').strip()}",
        f"dirty: {bool(status.strip())}",
    ]
    if status.strip():
        lines += [
            "",
            "# git status --porcelain",
            status,
            "# git diff HEAD",
            _git("diff", "HEAD"),
        ]
    path.write_text("\n".join(lines) + "\n")


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

    It holds checkpoints/, config.yaml and git.txt (commit, dirty flag, diff).
    """

    def before_instantiate_classes(self) -> None:
        if self.subcommand != "fit":
            return
        run_dir = os.environ.get(RUN_DIR_ENV)
        if run_dir is None:
            stamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
            run_dir = f"runs/{stamp}_{socket.gethostname()}_{secrets.token_hex(2)}"
            Path(run_dir).mkdir(parents=True)
            write_git_info(Path(run_dir) / "git.txt")
            os.environ[RUN_DIR_ENV] = run_dir
        self.config.fit.trainer.default_root_dir = run_dir

    def before_fit(self) -> None:
        for cb in self.trainer.checkpoint_callbacks:
            if isinstance(cb, ModelCheckpoint) and cb.dirpath is None:
                cb.dirpath = str(Path(self.trainer.default_root_dir) / "checkpoints")


def main() -> None:
    SIPECLI(
        model_class=CATSModule,
        datamodule_class=L.LightningDataModule,
        subclass_mode_data=True,
        seed_everything_default=42,
        save_config_callback=RunDirSaveConfigCallback,
        save_config_kwargs={"overwrite": True, "save_to_log_dir": False},
    )


if __name__ == "__main__":
    main()
