"""Base classes for SIPE training modules.

`sipe fit` (LightningCLI, subclass mode) accepts any `SIPEModule` subclass named by
`model.class_path` in the config, so a new method needs a module class and a config,
not a new CLI entry point.
"""

from __future__ import annotations

from typing import Any

import lightning as L
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim.lr_scheduler import CosineAnnealingLR

# Checkpoint key holding the module's import path (read by sipe.bench.checkpoint).
CHECKPOINT_KEY = "sipe_module"


def class_path(cls: type) -> str:
    return f"{cls.__module__}.{cls.__qualname__}"


@torch.enable_grad()
def linear_probe_accuracy(
    train_x: torch.Tensor,
    train_y: torch.Tensor,
    test_x: torch.Tensor,
    test_y: torch.Tensor,
    num_classes: int,
    steps: int = 1000,
) -> float:
    """Fresh standardized softmax regression (full-batch Adam), test accuracy."""
    mean, std = train_x.mean(0), train_x.std(0).clamp_min(1e-6)
    train_x, test_x = (train_x - mean) / std, (test_x - mean) / std
    probe = nn.Linear(train_x.shape[1], num_classes).to(train_x.device)
    optimizer = torch.optim.Adam(probe.parameters(), lr=1e-2, weight_decay=1e-4)
    for _ in range(steps):
        optimizer.zero_grad(set_to_none=True)
        F.cross_entropy(probe(train_x), train_y).backward()
        optimizer.step()
    with torch.no_grad():
        return float((probe(test_x).argmax(-1) == test_y).float().mean())


class SIPEModule(L.LightningModule):
    """Every SIPE training module: records its class path in the checkpoint, so the
    bench loader can rebuild the right class without a registry."""

    def on_save_checkpoint(self, checkpoint: dict[str, Any]) -> None:
        checkpoint[CHECKPOINT_KEY] = class_path(type(self))


class FrozenBackboneModule(SIPEModule):
    """Shared plumbing for the feature-space modules (ScannerVAE, PairedVAE), which
    keep the backbone frozen for the whole run (no curriculum).

    Subclasses define `self.network` and an `hparams.freeze_backbone` flag.
    """

    def train(self, mode: bool = True) -> FrozenBackboneModule:
        # Lightning toggles train mode around validation; keep a frozen backbone in
        # eval mode throughout (no-op for H0-mini, which has no dropout/drop-path).
        super().train(mode)
        if self.hparams.freeze_backbone:
            self.network.backbone.eval()
        return self

    def cosine_adamw(
        self, params: Any, lr: float, weight_decay: float
    ) -> tuple[torch.optim.Optimizer, dict[str, Any]]:
        """AdamW + per-step cosine decay over trainer.max_steps."""
        if self.trainer.max_steps <= 0:
            raise ValueError(
                f"{type(self).__name__} needs trainer.max_steps (cosine LR)."
            )
        optimizer = torch.optim.AdamW(params, lr=lr, weight_decay=weight_decay)
        scheduler = CosineAnnealingLR(optimizer, T_max=self.trainer.max_steps)
        return optimizer, {"scheduler": scheduler, "interval": "step"}

    def log_wsi_probes(
        self,
        features: dict[str, torch.Tensor],
        domains: torch.Tensor,
        slides: list[str],
    ) -> None:
        """Fresh linear scanner probe per feature set, trained on half of the WSIs
        and tested on the other half; logged as `val/probe_<name>_acc`."""
        unique = sorted(set(slides))
        if len(unique) < 2:
            return  # e.g. limit_val_batches: no WSI-disjoint split possible
        test_slides = set(unique[1::2])
        test = torch.tensor([s in test_slides for s in slides], device=domains.device)
        num_classes = int(self.hparams.num_domains)
        for name, x in features.items():
            accuracy = linear_probe_accuracy(
                x[~test], domains[~test], x[test], domains[test], num_classes
            )
            self.log(f"val/probe_{name}_acc", accuracy)
