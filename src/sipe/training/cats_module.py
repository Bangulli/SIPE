from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import lightning as L
import torch
import torch.nn as nn
import torch.nn.functional as F

from sipe.model.arch import CATS
from sipe.training.grl import GradientReversal


class CATSModule(L.LightningModule):
    """Lightning training wrapper for CATS.

    The architecture owns only representation/reconstruction modules.
    This wrapper owns all training-specific components:

    - specified-domain classifier on s
    - patch-wise domain classifier on z
    - gradient reversal on z
    - reconstruction/adversarial objectives
    - GRL scheduling and logging

    Batch contract
    --------------
    By default a batch must be a mapping containing:

        {
            "image": Tensor[B, C, H, W],
            "domain": LongTensor[B],
        }

    ``image_key`` and ``domain_key`` can be changed from the CLI.
    """

    def __init__(
        self,
        network: CATS,
        num_domains: int,
        reconstruction_weight: float = 20.0,
        adversarial_weight: float = 1.0,
        specified_domain_weight: float = 1.0,
        unspecified_domain_weight: float = 1.0,
        grl_alpha: float = 1.0,
        grl_warmup_steps: int = 0,
        freeze_backbone: bool = True,
        image_key: str = "image",
        domain_key: str = "domain",
    ) -> None:
        super().__init__()

        if num_domains < 2:
            raise ValueError("num_domains must be at least 2.")
        if reconstruction_weight < 0:
            raise ValueError("reconstruction_weight must be non-negative.")
        if adversarial_weight < 0:
            raise ValueError("adversarial_weight must be non-negative.")
        if specified_domain_weight < 0 or unspecified_domain_weight < 0:
            raise ValueError("Domain-loss weights must be non-negative.")
        if grl_alpha < 0:
            raise ValueError("grl_alpha must be non-negative.")
        if grl_warmup_steps < 0:
            raise ValueError("grl_warmup_steps must be non-negative.")

        self.network = network
        self.num_domains = int(num_domains)

        self.specified_classifier = nn.Linear(
            network.specified_dim,
            self.num_domains,
        )
        self.unspecified_classifier = nn.Linear(
            network.unspecified_dim,
            self.num_domains,
        )
        self.grl = GradientReversal(alpha=grl_alpha)

        self.reconstruction_weight = float(reconstruction_weight)
        self.adversarial_weight = float(adversarial_weight)
        self.specified_domain_weight = float(specified_domain_weight)
        self.unspecified_domain_weight = float(unspecified_domain_weight)
        self.grl_alpha = float(grl_alpha)
        self.grl_warmup_steps = int(grl_warmup_steps)
        self.freeze_backbone_flag = bool(freeze_backbone)

        self.image_key = image_key
        self.domain_key = domain_key

        if self.freeze_backbone_flag:
            self.network.freeze_backbone(True)

        # LightningCLI saves the full object graph/config separately. Avoid
        # serializing the nn.Module instance as a hyperparameter.
        self.save_hyperparameters(ignore=["network"])

    def forward(self, images: torch.Tensor) -> dict[str, torch.Tensor]:
        """Inference-facing forward: no adversarial heads are applied."""
        return self.network(images)

    def encode(
        self,
        images: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        return self.network.encode(images)

    def _current_grl_alpha(self) -> float:
        if self.grl_warmup_steps == 0:
            return self.grl_alpha

        progress = min(
            1.0,
            float(self.global_step) / float(self.grl_warmup_steps),
        )
        return self.grl_alpha * progress

    def _domain_logits(
        self,
        s: torch.Tensor,
        z: torch.Tensor,
        *,
        grl_alpha: float,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Return sample-wise logits from s and patch-wise logits from z."""
        logits_s = self.specified_classifier(s)

        # [B, C, H, W] -> [B, H*W, C]
        z_tokens = z.flatten(2).transpose(1, 2)
        z_reversed = self.grl(z_tokens, alpha=grl_alpha)
        logits_z = self.unspecified_classifier(z_reversed)

        return logits_s, logits_z

    def _unpack_batch(
        self,
        batch: Mapping[str, Any],
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if not isinstance(batch, Mapping):
            raise TypeError(
                f"CATSModule expects a mapping batch. Got {type(batch).__name__}."
            )

        try:
            images = batch[self.image_key]
            domains = batch[self.domain_key]
        except KeyError as exc:
            raise KeyError(
                f"Batch must contain {self.image_key!r} and {self.domain_key!r}. "
                f"Available keys: {tuple(batch.keys())}."
            ) from exc

        if not isinstance(images, torch.Tensor):
            raise TypeError(f"{self.image_key!r} must be a torch.Tensor.")
        if not isinstance(domains, torch.Tensor):
            domains = torch.as_tensor(domains, device=images.device)

        # Allow [B, 1] as a convenience, but keep the task explicitly
        # single-label categorical.
        if domains.ndim == 2 and domains.shape[1] == 1:
            domains = domains[:, 0]
        if domains.ndim != 1:
            raise ValueError(
                f"{self.domain_key!r} must have shape [B] (integer class ids), "
                f"got {tuple(domains.shape)}."
            )

        return images, domains.long()

    def _shared_step(
        self,
        batch: Mapping[str, Any],
        *,
        stage: str,
    ) -> torch.Tensor:
        images, domains = self._unpack_batch(batch)

        outputs = self.network(images)
        s = outputs["s"]
        z = outputs["z"]
        reconstruction = outputs["reconstruction"]

        alpha = self._current_grl_alpha()
        logits_s, logits_z = self._domain_logits(
            s,
            z,
            grl_alpha=alpha,
        )

        reconstruction_loss = F.mse_loss(reconstruction, images)

        specified_domain_loss = F.cross_entropy(
            logits_s,
            domains,
        )

        batch_size, num_patches, _ = logits_z.shape
        patch_domains = domains[:, None].expand(batch_size, num_patches).reshape(-1)
        unspecified_domain_loss = F.cross_entropy(
            logits_z.reshape(-1, self.num_domains),
            patch_domains,
        )

        adversarial_loss = (
            self.specified_domain_weight * specified_domain_loss
            + self.unspecified_domain_weight * unspecified_domain_loss
        )

        loss = (
            self.reconstruction_weight * reconstruction_loss
            + self.adversarial_weight * adversarial_loss
        )

        specified_accuracy = (logits_s.argmax(dim=-1) == domains).float().mean()

        unspecified_accuracy = (
            (logits_z.argmax(dim=-1).reshape(-1) == patch_domains).float().mean()
        )

        metrics = {
            f"{stage}/loss": loss,
            f"{stage}/reconstruction_loss": reconstruction_loss,
            f"{stage}/domain_s_loss": specified_domain_loss,
            f"{stage}/domain_z_loss": unspecified_domain_loss,
            f"{stage}/domain_s_acc": specified_accuracy,
            f"{stage}/domain_z_acc": unspecified_accuracy,
        }

        self.log_dict(
            metrics,
            on_step=stage == "train",
            on_epoch=True,
            prog_bar=stage != "train",
            sync_dist=True,
            batch_size=images.shape[0],
        )

        if stage == "train":
            self.log(
                "train/grl_alpha",
                alpha,
                on_step=True,
                on_epoch=False,
                prog_bar=False,
                sync_dist=False,
                batch_size=images.shape[0],
            )

        return loss

    def training_step(
        self,
        batch: Mapping[str, Any],
        batch_idx: int,
    ) -> torch.Tensor:
        del batch_idx
        return self._shared_step(batch, stage="train")

    def validation_step(
        self,
        batch: Mapping[str, Any],
        batch_idx: int,
    ) -> torch.Tensor:
        del batch_idx
        return self._shared_step(batch, stage="val")

    def test_step(
        self,
        batch: Mapping[str, Any],
        batch_idx: int,
    ) -> torch.Tensor:
        del batch_idx
        return self._shared_step(batch, stage="test")

    def predict_step(
        self,
        batch: Mapping[str, Any],
        batch_idx: int,
        dataloader_idx: int = 0,
    ) -> dict[str, torch.Tensor]:
        del batch_idx, dataloader_idx
        images, _ = self._unpack_batch(batch)
        return self.network(images)

    def on_train_epoch_start(self) -> None:
        # Trainer calls .train() recursively. Put a frozen backbone back into
        # eval mode so dropout/normalization state, if any, also stays frozen.
        if self.freeze_backbone_flag:
            self.network.backbone.eval()
