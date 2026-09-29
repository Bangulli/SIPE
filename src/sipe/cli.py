from __future__ import annotations

import lightning as L
from lightning.pytorch.cli import LightningCLI

from sipe.training.cats_module import CATSModule


def main() -> None:
    LightningCLI(
        model_class=CATSModule,
        datamodule_class=L.LightningDataModule,
        subclass_mode_data=True,
        seed_everything_default=42,
        save_config_kwargs={"overwrite": True},
    )


if __name__ == "__main__":
    main()
