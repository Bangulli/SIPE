from __future__ import annotations

import copy
import json
from pathlib import Path
from typing import Any, Sequence

import torch
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts
from torch.utils.data import DataLoader

from sipe.utils.misc import make_name_from_list


class Curriculum(list):
    def add_step(
        self,
        step_type: str = "recon",
        epochs: int = 5,
        adverse_alpha: float | Sequence[float] = 0.1,
        lr: float = 3e-4,
        restarts: int = 5,
        norm: bool = True,
        freeze_bb: bool = True,
        freeze_tangler: bool = True,
    ) -> None:
        self.append(
            {
                "type": step_type,
                "epochs": epochs,
                "lr": lr,
                "adverse_alpha": adverse_alpha,
                "restarts": restarts,
                "adverse_norm": norm,
                "freeze_backbone": freeze_bb,
                "freeze_tangler": freeze_tangler,
            }
        )

    def save(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w") as f:
            json.dump(self, f, indent=4)

    @classmethod
    def load(cls, path: str | Path) -> "Curriculum":
        with Path(path).open() as f:
            data = json.load(f)

        curriculum = cls()
        curriculum.extend(data)
        return curriculum


class CurriculumTrainer:
    """Minimal trainer for the SIPE curriculum.

    The trainer is intentionally agnostic to the dataset implementation.
    It only receives PyTorch DataLoaders and expects batches compatible
    with the SIPE model.
    """

    def __init__(
        self,
        model: torch.nn.Module,
        loss_recon: Any,
        loss_adverse: Any,
        loss_cycle: Any | None = None,
        wdir: str | Path = "trainer",
        scheduler_cls=CosineAnnealingWarmRestarts,
        optimizer_cls=AdamW,
        device: str | torch.device = "cuda",
    ) -> None:
        self.model = model
        self.loss_r = loss_recon
        self.loss_a = loss_adverse
        self.loss_c = loss_cycle

        self.wdir = Path(wdir)
        self.wdir.mkdir(parents=True, exist_ok=True)

        self.scheduler_cls = scheduler_cls
        self.optimizer_cls = optimizer_cls
        self.device = torch.device(device)

        self.optimizer = None
        self.scheduler = None
        self.loss_history = {"training": [], "validation": []}

        self.model.to(self.device)
        self._move_losses_to_device()

    def train(
        self,
        train_loader: DataLoader,
        val_loader: DataLoader,
        curriculum: Curriculum,
        ckpt_dir: str = "checkpoints",
    ) -> dict[str, list[float]]:
        checkpoint_dir = self.wdir / ckpt_dir
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
        curriculum.save(checkpoint_dir / "curriculum.json")

        epoch = 0

        for step_idx, step in enumerate(curriculum):
            step_type = step["type"].lower()
            epochs = step["epochs"]

            if step_type not in {"recon", "adverse", "cycle"}:
                raise ValueError(
                    f"Unknown curriculum step {step_type!r}; "
                    "expected one of: recon, adverse, cycle"
                )

            if step_type == "cycle" and self.loss_c is None:
                raise ValueError("A cycle step requires loss_cycle.")

            self._configure_model(step)

            self.optimizer = self.optimizer_cls(
                (p for p in self.model.parameters() if p.requires_grad),
                lr=step["lr"],
            )
            self.scheduler = self.scheduler_cls(
                self.optimizer,
                T_0=step["restarts"],
            )

            alphas = self._expand_alpha(step["adverse_alpha"], epochs)

            for local_epoch in range(epochs):
                current_alpha = alphas[local_epoch]

                if step_type in {"adverse", "cycle"}:
                    loss_fn = self.loss_a if step_type == "adverse" else self.loss_c
                    loss_fn.set_adverse_alpha(current_alpha)
                    loss_fn.set_adverse_norm(step["adverse_norm"])

                train_loss = self._run_epoch(
                    train_loader,
                    step_type=step_type,
                    training=True,
                )
                val_loss = self._run_epoch(
                    val_loader,
                    step_type=step_type,
                    training=False,
                )

                self.loss_history["training"].append(train_loss)
                self.loss_history["validation"].append(val_loss)

                self.scheduler.step()
                epoch += 1

                print(
                    f"step={step_idx} "
                    f"type={step_type} "
                    f"epoch={epoch} "
                    f"train={train_loss:.6f} "
                    f"val={val_loss:.6f}"
                )

                self._save_checkpoint(checkpoint_dir, epoch)
                self._save_history()

        return self.loss_history

    def _run_epoch(
        self,
        loader: DataLoader,
        *,
        step_type: str,
        training: bool,
    ) -> float:
        self.model.train(training)

        total_loss = 0.0
        n_batches = 0

        context = torch.enable_grad() if training else torch.no_grad()

        with context:
            for batch in loader:
                if training:
                    self.optimizer.zero_grad(set_to_none=True)

                loss = self._compute_loss(
                    batch,
                    step_type=step_type,
                    val=not training,
                )

                if training:
                    loss.backward()
                    self.optimizer.step()

                total_loss += loss.detach().item()
                n_batches += 1

        if n_batches == 0:
            raise RuntimeError("DataLoader yielded no batches.")

        return total_loss / n_batches

    def _compute_loss(
        self,
        batch: Any,
        *,
        step_type: str,
        val: bool,
    ) -> torch.Tensor:
        logger = self._make_logger(step_type)

        if step_type == "cycle":
            loss, _ = self._compute_cycle_loss_for_batch(
                batch,
                logger=logger,
                val=val,
            )
            return loss

        if step_type == "adverse":
            loss_fn = self.loss_a
        else:
            # Preserve the old trainer behaviour: reconstruction training used
            # loss_r, while reconstruction validation used loss_a with val=True.
            loss_fn = self.loss_a if val else self.loss_r

        loss, _ = self.model.loss(
            batch,
            loss_fn,
            logger,
            val=val,
        )
        return loss

    def _compute_cycle_loss_for_batch(
        self,
        batch: dict[str, Any],
        logger: dict[str, list] | None = None,
        val: bool = False,
    ):
        batch1 = batch

        # First pass.
        s1, z1 = self.model(batch1)
        recon1 = self.model.recon_image(s1, z1)

        (
            s1_class_s,
            z1_class_s,
            s1_class_o,
            z1_class_o,
            s1_class_p,
            z1_class_p,
        ) = self.model.entangler.classify_all(s1, z1)

        # Mix specified representations across the batch.
        s1_prime = torch.roll(s1, 1, dims=0)
        metadata_prime = self._roll_list(
            copy.deepcopy(batch1["metadata"]),
            1,
        )

        batch2 = {
            "image": self.model.recon_image(s1_prime, z1).detach(),
            "metadata": metadata_prime,
        }

        # Second pass.
        s2, z2 = self.model(batch2)
        s2_prime = torch.roll(s2, -1, dims=0)

        gt_labels_s = torch.as_tensor(
            self.model.transform_labels(
                [sample["staining"] for sample in batch1["metadata"]]
            ),
            dtype=torch.float32,
            device=self.device,
        )

        gt_labels_o = torch.as_tensor(
            self.model.transform_organs(
                [make_name_from_list(sample["organ"]) for sample in batch1["metadata"]]
            ),
            dtype=torch.float32,
            device=self.device,
        )

        gt_labels_p = torch.as_tensor(
            self.model.transform_paths(
                [
                    make_name_from_list(sample["diagnosis"])
                    for sample in batch1["metadata"]
                ]
            ),
            dtype=torch.float32,
            device=self.device,
        )

        gt_images = batch1["image"].to(self.device)

        return self.loss_c(
            gt_labels_s,
            gt_labels_o,
            gt_labels_p,
            gt_images,
            recon1,
            s1,
            s2_prime,
            z1,
            z2,
            s1_class_s,
            z1_class_s,
            s1_class_o,
            z1_class_o,
            s1_class_p,
            z1_class_p,
            logger,
            val,
        )

    def _configure_model(self, step: dict[str, Any]) -> None:
        # These are SIPE model capabilities rather than concrete class checks.
        if hasattr(self.model, "freeze_backbone"):
            self.model.freeze_backbone(step["freeze_backbone"])

        if hasattr(self.model, "freeze_or_unfreeze_disentangler"):
            self.model.freeze_or_unfreeze_disentangler(step["freeze_tangler"])

    def _move_losses_to_device(self) -> None:
        for loss in (self.loss_r, self.loss_a, self.loss_c):
            if loss is None:
                continue

            if isinstance(loss, torch.nn.Module):
                loss.to(self.device)

            image_recon_loss = getattr(loss, "image_recon_loss", None)
            if isinstance(image_recon_loss, torch.nn.Module):
                image_recon_loss.to(self.device)

    @staticmethod
    def _expand_alpha(
        alpha: float | Sequence[float],
        epochs: int,
    ) -> list[float]:
        if isinstance(alpha, (int, float)):
            return [float(alpha)] * epochs

        values = list(alpha)
        if len(values) != epochs:
            raise ValueError(
                f"Expected {epochs} adverse_alpha values, got {len(values)}."
            )
        return [float(value) for value in values]

    @staticmethod
    def _roll_list(items: list[Any], shifts: int) -> list[Any]:
        if not items:
            return items

        shifts %= len(items)
        return items[-shifts:] + items[:-shifts]

    @staticmethod
    def _make_logger(step_type: str) -> dict[str, list]:
        if step_type == "cycle":
            keys = [
                "Recon Img",
                "S cycle",
                "Z cycle",
                "Adversarial CE S",
                "CE S",
                "Adversarial CE O",
                "CE O",
                "Adversarial CE P",
                "CE P",
            ]
        else:
            keys = [
                "Recon Img",
                "Stain probs2vec",
                "InfoNCE Stain",
                "InfoNCE Morph",
                "Adversarial CE S",
                "CE S",
                "Adversarial CE O",
                "CE O",
                "Adversarial CE P",
                "CE P",
                "s std",
                "s norm",
            ]

        return {key: [] for key in keys}

    def _save_checkpoint(self, checkpoint_dir: Path, epoch: int) -> None:
        checkpoint = {
            "epoch": epoch,
            "model": self.model.state_dict(),
            "optimizer": self.optimizer.state_dict() if self.optimizer else None,
            "scheduler": self.scheduler.state_dict() if self.scheduler else None,
            "loss_history": self.loss_history,
        }
        torch.save(
            checkpoint,
            checkpoint_dir / f"epoch_{epoch:04d}.pt",
        )

    def load_checkpoint(
        self,
        path: str | Path,
        *,
        load_optimizer: bool = False,
    ) -> int:
        checkpoint = torch.load(
            path,
            map_location=self.device,
            weights_only=False,
        )

        self.model.load_state_dict(checkpoint["model"])
        self.loss_history = checkpoint.get(
            "loss_history",
            {"training": [], "validation": []},
        )

        if load_optimizer:
            if self.optimizer is None or self.scheduler is None:
                raise RuntimeError(
                    "Optimizer and scheduler must be initialized before "
                    "loading their state."
                )

            if checkpoint.get("optimizer") is not None:
                self.optimizer.load_state_dict(checkpoint["optimizer"])

            if checkpoint.get("scheduler") is not None:
                self.scheduler.load_state_dict(checkpoint["scheduler"])

        return int(checkpoint["epoch"])

    def _save_history(self) -> None:
        with (self.wdir / "history.json").open("w") as f:
            json.dump(self.loss_history, f, indent=4)

    def load_best_model(
        self,
        ckpt_dir: str = "checkpoints",
    ) -> torch.nn.Module:
        values = self.loss_history["validation"]
        if not values:
            raise RuntimeError("No validation history available.")

        best_epoch = min(
            range(1, len(values) + 1),
            key=lambda epoch: values[epoch - 1],
        )

        self.load_checkpoint(self.wdir / ckpt_dir / f"epoch_{best_epoch:04d}.pt")
        return self.model
