from __future__ import annotations

import lightning as L
import numpy as np
import torch
from lightning.pytorch.loggers import WandbLogger


def _to_image(
    x: torch.Tensor,
    mean: tuple[float, ...],
    std: tuple[float, ...],
) -> np.ndarray:
    x = x.detach().float().cpu()

    mean_t = torch.tensor(mean).view(3, 1, 1)
    std_t = torch.tensor(std).view(3, 1, 1)

    # Model space -> RGB [0, 1]
    x = x * std_t + mean_t
    x = x.clamp(0, 1)

    # CHW float -> HWC uint8
    x = x.mul(255).round().byte().permute(1, 2, 0).numpy()

    return x


class ReconstructionLogger(L.Callback):
    def __init__(
        self,
        num_images: int = 4,
        every_n_steps: int = 490,
    ) -> None:
        super().__init__()
        self.num_images = num_images
        self.every_n_steps = every_n_steps
        self._last_logged_step = -1

    @staticmethod
    def _denormalize(
        images: torch.Tensor,
        mean,
        std,
    ) -> torch.Tensor:
        mean = torch.as_tensor(
            mean,
            device=images.device,
            dtype=images.dtype,
        )[None, :, None, None]

        std = torch.as_tensor(
            std,
            device=images.device,
            dtype=images.dtype,
        )[None, :, None, None]

        return (images * std + mean).clamp(0, 1)

    def on_validation_batch_end(
        self,
        trainer: L.Trainer,
        pl_module: L.LightningModule,
        outputs,
        batch,
        batch_idx: int,
        dataloader_idx: int = 0,
    ) -> None:
        del outputs, dataloader_idx

        # Don't log Lightning's initial sanity-check validation.
        if trainer.sanity_checking:
            return

        # First validation batch only.
        if batch_idx != 0:
            return

        step = trainer.global_step

        if step == self._last_logged_step:
            return

        if step % self.every_n_steps != 0:
            return

        wandb_logger = next(
            (logger for logger in trainer.loggers if isinstance(logger, WandbLogger)),
            None,
        )

        if wandb_logger is None:
            return

        self._last_logged_step = step

        images = batch[pl_module.image_key][: self.num_images]

        with torch.no_grad():
            output = pl_module.network(images)

            s = output["s"]
            z = output["z"]
            reconstruction = output["reconstruction"]

            # Change only the specified/domain representation.
            s_swapped = torch.roll(s, shifts=1, dims=0)
            mixed = pl_module.network.decode(s_swapped, z)

            # What does the encoder recover from the synthetic image?
            s2, z2 = pl_module.network.encode(mixed)
            mixed_reconstruction = pl_module.network.decode(s2, z2)

            z_only = pl_module.network.decode(
                torch.zeros_like(s),
                z,
            )

            s_only = pl_module.network.decode(
                s,
                torch.zeros_like(z),
            )

        datamodule = trainer.datamodule

        mean = datamodule.mean
        std = datamodule.std

        panels = {
            "source": images,
            "reconstruction": reconstruction,
            "swap_s": mixed,
            "swap_s_reencoded": mixed_reconstruction,
            "z_only": z_only,
            "s_only": s_only,
        }

        wandb_images = []
        captions = []

        for sample_idx in range(images.shape[0]):
            for name, tensor in panels.items():
                wandb_images.append(
                    _to_image(
                        tensor[sample_idx],
                        mean,
                        std,
                    )
                )
                captions.append(f"sample={sample_idx} | {name}")

        wandb_logger.log_image(
            key="val/reconstructions",
            images=wandb_images,
            caption=captions,
            step=step,
        )
