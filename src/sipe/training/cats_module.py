from __future__ import annotations

from typing import Any

import lightning as L
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts

from sipe.losses.adversarial_classif_loss import AdversarialClassifLoss
from sipe.losses.image_recon_loss import ImageReconLoss
from sipe.model.arch import CATS

from .curriculum import CurriculumPhase, StepCurriculum
from .grl import GradientReversal


class CATSModule(L.LightningModule):
    """Step-based Lightning port of the legacy CATS curriculum trainer.

    The original stain branch is adapted to one SCORPION scanner/domain branch.
    The original organ/pathology heads are intentionally not reintroduced.
    """

    def __init__(
        self,
        network: CATS,
        num_domains: int,
        curriculum: list[dict[str, Any]],
        grl_alpha: float = 1.0,
        image_key: str = "image",
        domain_key: str = "domain",
        recon_phase_reconstruction_weight: float = 1.0,
        adverse_phase_reconstruction_weight: float = 1.0,
        cycle_phase_reconstruction_weight: float = 20.0,
        cycle_s_weight: float = 1.0,
        cycle_z_weight: float = 0.5,
        freeze_backbone: bool = False,
    ) -> None:
        super().__init__()

        self.network = network
        self.num_domains = int(num_domains)
        self.image_key = image_key
        self.domain_key = domain_key

        self.curriculum = StepCurriculum(curriculum)

        self.recon_phase_reconstruction_weight = float(
            recon_phase_reconstruction_weight
        )
        self.adverse_phase_reconstruction_weight = float(
            adverse_phase_reconstruction_weight
        )
        self.cycle_phase_reconstruction_weight = float(
            cycle_phase_reconstruction_weight
        )
        self.cycle_s_weight = float(cycle_s_weight)
        self.cycle_z_weight = float(cycle_z_weight)

        self.specified_classifier = nn.Linear(
            network.specified_dim,
            self.num_domains,
        )
        self.unspecified_classifier = nn.Linear(
            network.unspecified_dim,
            self.num_domains,
        )
        self.grl = GradientReversal(alpha=grl_alpha)

        # Reuse the old SIPE losses instead of approximating them.
        self.image_recon_loss = ImageReconLoss()
        self.domain_classif_loss = AdversarialClassifLoss(logkey="S")

        self._active_phase_idx = -1
        self._phase_scheduler: CosineAnnealingWarmRestarts | None = None

        # Under LightningCLI this stores the parsed config, so `network` is saved as
        # {class_path, init_args} and CATSModule.load_from_checkpoint(ckpt) rebuilds it.
        self.save_hyperparameters()
        if freeze_backbone:
            self.network.freeze_backbone()

    @property
    def total_curriculum_steps(self) -> int:
        return self.curriculum.total_steps

    def forward(self, images: torch.Tensor) -> dict[str, torch.Tensor]:
        return self.network(images)

    def on_fit_start(self) -> None:
        if self.trainer.max_steps not in (-1, self.curriculum.total_steps):
            self.print(
                f"WARNING: trainer.max_steps={self.trainer.max_steps}, "
                f"curriculum={self.curriculum.total_steps} steps."
            )

    def on_train_batch_start(
        self,
        batch: dict[str, Any],
        batch_idx: int,
    ) -> None:
        del batch, batch_idx

        phase_idx, phase, local_step = self.curriculum.at(int(self.global_step))
        if phase_idx != self._active_phase_idx:
            self._activate_phase(
                phase_idx=phase_idx,
                phase=phase,
                local_step=local_step,
            )

    def on_train_batch_end(
        self,
        outputs: Any,
        batch: dict[str, Any],
        batch_idx: int,
    ) -> None:
        del outputs, batch, batch_idx

        if self._phase_scheduler is None or self._active_phase_idx < 0:
            return

        phase = self.curriculum.phases[self._active_phase_idx]
        phase_start = self.curriculum.phase_start(self._active_phase_idx)

        # global_step has advanced after optimizer.step().  Set the LR that will
        # be used for the next optimizer update, preserving the legacy
        # optimizer.step(); scheduler.step() ordering.
        completed_steps = int(self.global_step) - phase_start
        if completed_steps < phase.steps:
            self._phase_scheduler.step(completed_steps)

    def on_train_epoch_start(self) -> None:
        if self._active_phase_idx < 0:
            return

        phase = self.curriculum.phases[self._active_phase_idx]
        if phase.freeze_backbone:
            self.network.backbone.eval()
        if phase.freeze_tangler:
            self._set_tangler_mode(train=False)

    def training_step(
        self,
        batch: dict[str, Any],
        batch_idx: int,
    ) -> torch.Tensor:
        del batch_idx
        return self._shared_step(batch, stage="train")["loss"]

    def validation_step(
        self,
        batch: dict[str, Any],
        batch_idx: int,
    ) -> None:
        del batch_idx
        self._shared_step(batch, stage="val")

    def test_step(
        self,
        batch: dict[str, Any],
        batch_idx: int,
    ) -> None:
        del batch_idx
        self._shared_step(batch, stage="test")

    def predict_step(
        self,
        batch: dict[str, Any],
        batch_idx: int,
        dataloader_idx: int = 0,
    ) -> dict[str, torch.Tensor]:
        del batch_idx, dataloader_idx
        return self.network(batch[self.image_key])

    def _activate_phase(
        self,
        *,
        phase_idx: int,
        phase: CurriculumPhase,
        local_step: int,
    ) -> None:
        optimizer = self.optimizers(use_pl_optimizer=False)

        # Legacy trainer constructed a fresh AdamW at every phase.
        # Clearing Adam state reproduces this without replacing Lightning's
        # optimizer object. Do not clear on a mid-phase checkpoint resume.
        if phase.reset_optimizer_state and phase_idx > 0 and local_step == 0:
            optimizer.state.clear()

        self._set_backbone_frozen(phase.freeze_backbone)
        self._set_tangler_frozen(phase.freeze_tangler)

        for group in optimizer.param_groups:
            group["lr"] = phase.lr
            group["initial_lr"] = phase.lr

        if phase.restart_steps is None:
            self._phase_scheduler = None
        else:
            self._phase_scheduler = CosineAnnealingWarmRestarts(
                optimizer,
                T_0=phase.restart_steps,
            )
            if local_step > 0:
                self._phase_scheduler.step(local_step)

        self.domain_classif_loss.set_norm(phase.adverse_norm)
        self._active_phase_idx = phase_idx

        self.print(
            f"Curriculum phase {phase_idx}: {phase.name} "
            f"(mode={phase.mode}, global_step={self.global_step}, "
            f"local_step={local_step}, lr={phase.lr})"
        )

    def _shared_step(
        self,
        batch: dict[str, Any],
        stage: str,
    ) -> dict[str, torch.Tensor]:
        images, domains = self._unpack_batch(batch)
        phase_idx, phase, local_step = self._phase_for_stage(stage)
        alpha = phase.alpha_at(local_step)

        outputs = self.network(images)
        s1 = outputs["s"]
        z1 = outputs["z"]
        reconstruction = outputs["reconstruction"]

        reconstruction_loss = self.image_recon_loss(
            reconstruction,
            images,
        )

        zero = reconstruction_loss.new_zeros(())
        domain_loss = zero
        cycle_s_l1 = zero
        cycle_z_l1 = zero
        cycle_s_loss = zero
        cycle_z_loss = zero

        if phase.mode == "recon":
            # SIPE_Loss_Adversarial(recon_mode=True)
            loss = self.recon_phase_reconstruction_weight * reconstruction_loss

        elif phase.mode == "adverse":
            domain_loss = self._domain_loss(
                s=s1,
                z=z1,
                domains=domains,
                alpha=alpha,
                norm=phase.adverse_norm,
                val=stage != "train",
            )
            loss = (
                self.adverse_phase_reconstruction_weight * reconstruction_loss
                + domain_loss
            )

        elif phase.mode == "cycle":
            cycle_s_l1, cycle_z_l1 = self._cycle_losses(
                s1=s1,
                z1=z1,
            )
            cycle_s_loss = self.cycle_s_weight * cycle_s_l1
            cycle_z_loss = self.cycle_z_weight * cycle_z_l1

            domain_loss = self._domain_loss(
                s=s1,
                z=z1,
                domains=domains,
                alpha=alpha,
                norm=phase.adverse_norm,
                val=stage != "train",
            )

            # SIPE_Loss_Adversarial_Cycle:
            #   20 * recon + S_cycle + 0.5 * Z_cycle + adversarial losses
            loss = (
                self.cycle_phase_reconstruction_weight * reconstruction_loss
                + cycle_s_loss
                + cycle_z_loss
                + domain_loss
            )
        else:
            raise RuntimeError(f"Unsupported phase mode: {phase.mode}")

        domain_s_acc, domain_z_acc = self._domain_accuracies(
            s=s1,
            z=z1,
            domains=domains,
        )

        metrics = {
            f"{stage}/reconstruction_loss": reconstruction_loss,
            f"{stage}/domain_loss": domain_loss,
            f"{stage}/domain_s_acc": domain_s_acc,
            f"{stage}/domain_z_acc": domain_z_acc,
            f"{stage}/cycle_s_l1": cycle_s_l1,
            f"{stage}/cycle_z_l1": cycle_z_l1,
            f"{stage}/cycle_s_loss": cycle_s_loss,
            f"{stage}/cycle_z_loss": cycle_z_loss,
            f"{stage}/s_abs_mean": s1.abs().mean(),
            f"{stage}/z_abs_mean": z1.abs().mean(),
            f"{stage}/s_std": s1.std(),
            f"{stage}/z_std": z1.std(),
        }

        self.log_dict(
            metrics,
            on_step=stage == "train",
            on_epoch=True,
            prog_bar=False,
            batch_size=images.shape[0],
        )
        self.log(
            f"{stage}/loss",
            loss,
            on_step=stage == "train",
            on_epoch=True,
            prog_bar=True,
            batch_size=images.shape[0],
        )

        if stage == "train":
            optimizer = self.optimizers(use_pl_optimizer=False)
            self.log(
                "train/curriculum_phase",
                float(phase_idx),
                on_step=True,
                on_epoch=False,
            )
            self.log(
                "train/adverse_alpha",
                float(alpha),
                on_step=True,
                on_epoch=False,
            )
            self.log(
                "train/lr",
                float(optimizer.param_groups[0]["lr"]),
                on_step=True,
                on_epoch=False,
            )

        return {
            "loss": loss,
            "reconstruction_loss": reconstruction_loss,
            "domain_loss": domain_loss,
            "cycle_s_loss": cycle_s_loss,
            "cycle_z_loss": cycle_z_loss,
        }

    def _cycle_losses(
        self,
        *,
        s1: torch.Tensor,
        z1: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if s1.shape[0] < 2:
            raise RuntimeError("Cycle training requires a batch size of at least 2.")

        # Exact graph from legacy CurriculumTrainer:
        # roll s -> reconstruct -> detach reconstructed image -> re-encode.
        s1_prime = torch.roll(s1, shifts=1, dims=0)
        mixed_images = self.network.decode(s1_prime, z1).detach()

        s2, z2 = self.network.encode(mixed_images)
        s2_prime = torch.roll(s2, shifts=-1, dims=0)

        # s1/z1 are deliberately NOT detached here.
        return (
            F.l1_loss(s2_prime, s1),
            F.l1_loss(z2, z1),
        )

    def _domain_logits(
        self,
        *,
        s: torch.Tensor,
        z: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        specified_logits = self.specified_classifier(s)

        # Match legacy Entangler.classify_all: flatten all z patches and apply
        # gradient reversal before the unspecified classifier.
        z_tokens = z.permute(0, 2, 3, 1).reshape(
            -1,
            self.network.unspecified_dim,
        )
        unspecified_logits = self.unspecified_classifier(self.grl(z_tokens))
        return specified_logits, unspecified_logits

    def _domain_loss(
        self,
        *,
        s: torch.Tensor,
        z: torch.Tensor,
        domains: torch.Tensor,
        alpha: float,
        norm: bool,
        val: bool,
    ) -> torch.Tensor:
        specified_logits, unspecified_logits = self._domain_logits(
            s=s,
            z=z,
        )

        # Old code fed MultiLabelBinarizer vectors. Scanner IDs are mutually
        # exclusive, so one-hot vectors are the SCORPION analogue.
        targets = F.one_hot(
            domains,
            num_classes=self.num_domains,
        ).to(dtype=specified_logits.dtype)

        self.domain_classif_loss.set_norm(norm)
        legacy_logger = {
            "Adversarial CE S": [],
            "CE S": [],
        }

        domain_loss, _ = self.domain_classif_loss(
            specified_logits,
            unspecified_logits,
            targets,
            self.device,
            legacy_logger,
            val,
            alpha,
        )
        return domain_loss

    @torch.no_grad()
    def _domain_accuracies(
        self,
        *,
        s: torch.Tensor,
        z: torch.Tensor,
        domains: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        specified_logits = self.specified_classifier(s.detach())

        z_tokens = (
            z.detach()
            .permute(0, 2, 3, 1)
            .reshape(
                -1,
                self.network.unspecified_dim,
            )
        )
        unspecified_logits = self.unspecified_classifier(z_tokens)

        specified_accuracy = (specified_logits.argmax(dim=-1) == domains).float().mean()

        patch_targets = (
            domains[:, None]
            .expand(
                -1,
                z.shape[-2] * z.shape[-1],
            )
            .reshape(-1)
        )

        unspecified_accuracy = (
            (unspecified_logits.argmax(dim=-1) == patch_targets).float().mean()
        )

        return specified_accuracy, unspecified_accuracy

    def _phase_for_stage(
        self,
        stage: str,
    ) -> tuple[int, CurriculumPhase, int]:
        step = int(self.global_step)
        if stage == "train":
            return self.curriculum.at(step)

        # Validation can happen immediately after the last update of a phase.
        return self.curriculum.at(max(step - 1, 0))

    def _unpack_batch(
        self,
        batch: dict[str, Any],
    ) -> tuple[torch.Tensor, torch.Tensor]:
        images = batch[self.image_key]
        domains = batch[self.domain_key]
        if domains.ndim > 1:
            domains = domains.squeeze(-1)
        return images, domains.long()

    def _set_backbone_frozen(self, freeze: bool) -> None:
        for parameter in self.network.backbone.parameters():
            parameter.requires_grad_(not freeze)
        self.network.backbone.train(not freeze)

    def _set_tangler_frozen(self, freeze: bool) -> None:
        # Legacy entangler = disentangler + classifiers + re-entangler.
        modules: list[nn.Module] = [
            self.network.disentangler,
            self.network.reentangler,
            self.specified_classifier,
            self.unspecified_classifier,
        ]
        for module in modules:
            for parameter in module.parameters():
                parameter.requires_grad_(not freeze)
        self._set_tangler_mode(train=not freeze)

    def _set_tangler_mode(self, *, train: bool) -> None:
        modules: list[nn.Module] = [
            self.network.disentangler,
            self.network.reentangler,
            self.specified_classifier,
            self.unspecified_classifier,
        ]
        for module in modules:
            module.train(train)
