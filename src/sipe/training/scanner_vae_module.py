"""Training module for ScannerVAE (not in legacy CATS; no curriculum, no cycle).

loss = recon(Dec(e(scanner), z), x) + beta * KL(q(z|x) || N(0, I))
       + w_adv * CE(adversary(GRL(LN(GAP(mu)))), scanner)
       [+ pixel MSE of a visualization decoder on detached features]

GRL safeguards, because in CATSv2 the GRL adversary sat at chance while a fresh probe
still read the scanner from GAP(z) 92% of the time:
- the adversary input is LayerNorm-ed, so the encoder cannot win by scaling mu;
- the adversary can get a higher LR and be re-initialized every N steps;
- every validation trains a fresh linear probe on frozen GAP(mu) (WSI-disjoint halves
  of the val split) and logs `val/probe_zgap_acc` next to `val/adversary_acc`. With
  only 2-3 WSIs per half the absolute accuracy is low (backbone GAP: ~0.8 vs ~0.96 with
  more WSIs), so read it against `val/probe_backbone_acc` (same split, same probe).
"""

from __future__ import annotations

import math
from typing import Any

import lightning as L
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim.lr_scheduler import CosineAnnealingLR

from sipe.model.scanner_vae import ScannerVAE
from sipe.training.grl import GradientReversal

# Stored in checkpoints so sipe.bench.checkpoint picks the right module class.
CHECKPOINT_TAG = "scanner_vae"


def feature_recon_loss(prediction: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """Scale-free: MSE relative to target variance + per-token cosine (as CATSv2)."""
    target = target.detach().float()
    prediction = prediction.float()
    mse = F.mse_loss(prediction, target) / target.var().clamp_min(1e-6)
    cosine = F.cosine_similarity(prediction, target, dim=1).mean()
    return mse + (1 - cosine)


def gaussian_kl(mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
    """KL(N(mu, exp(logvar)) || N(0, I)), mean over all elements."""
    return 0.5 * (mu.square() + logvar.exp() - logvar - 1).mean()


def ramp(step: int, warmup_steps: int) -> float:
    return min(1.0, step / warmup_steps) if warmup_steps > 0 else 1.0


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


class ScannerVAEModule(L.LightningModule):
    def __init__(
        self,
        network: ScannerVAE,
        num_domains: int = 5,
        lr: float = 3e-4,
        weight_decay: float = 0.01,
        kl_weight: float = 0.1,
        kl_warmup_steps: int = 1000,
        adversary_weight: float = 1.0,
        adversary_warmup_steps: int = 1000,
        adversary_hidden_dim: int = 512,
        adversary_lr_mult: float = 1.0,
        adversary_reset_steps: int | None = None,
        pixel_weight: float = 1.0,
        freeze_backbone: bool = True,
        image_key: str = "image",
        domain_key: str = "domain",
    ) -> None:
        super().__init__()
        self.network = network
        self.num_domains = int(num_domains)
        self.image_key = image_key
        self.domain_key = domain_key
        self.grl = GradientReversal(alpha=1.0)
        self.adversary = nn.Sequential(
            nn.Linear(network.unspecified_dim, adversary_hidden_dim),
            nn.GELU(),
            nn.Linear(adversary_hidden_dim, self.num_domains),
        )
        self._val_outputs: list[dict[str, Any]] = []
        self.save_hyperparameters()
        if freeze_backbone:
            network.freeze_backbone()

    def train(self, mode: bool = True) -> ScannerVAEModule:
        # Lightning toggles train mode around validation; keep a frozen backbone in
        # eval mode throughout (no-op for H0-mini, which has no dropout/drop-path).
        super().train(mode)
        if self.hparams.freeze_backbone:
            self.network.backbone.eval()
        return self

    def configure_optimizers(self) -> dict[str, Any]:
        adversary = list(self.adversary.parameters())
        adversary_ids = {id(p) for p in adversary}
        main = [
            p
            for p in self.parameters()
            if p.requires_grad and id(p) not in adversary_ids
        ]
        hp = self.hparams
        optimizer = torch.optim.AdamW(
            [
                {"params": main, "lr": hp.lr},
                {"params": adversary, "lr": hp.lr * hp.adversary_lr_mult},
            ],
            weight_decay=hp.weight_decay,
        )
        if self.trainer.max_steps <= 0:
            raise ValueError("ScannerVAEModule needs trainer.max_steps (cosine LR).")
        scheduler = CosineAnnealingLR(optimizer, T_max=self.trainer.max_steps)
        return {
            "optimizer": optimizer,
            "lr_scheduler": {"scheduler": scheduler, "interval": "step"},
        }

    def on_train_batch_start(self, batch: Any, batch_idx: int) -> None:
        del batch, batch_idx
        every = self.hparams.adversary_reset_steps
        step = int(self.global_step)
        if every and step > 0 and step % every == 0:
            self.reset_adversary()

    def reset_adversary(self) -> None:
        """Fresh adversary weights and Adam moments (the encoder is untouched)."""
        for module in self.adversary.modules():
            if isinstance(module, nn.Linear):
                module.reset_parameters()
        optimizer = self.optimizers(use_pl_optimizer=False)
        for parameter in self.adversary.parameters():
            optimizer.state.pop(parameter, None)

    def on_save_checkpoint(self, checkpoint: dict[str, Any]) -> None:
        checkpoint["sipe_module"] = CHECKPOINT_TAG

    def training_step(self, batch: dict[str, Any], batch_idx: int) -> torch.Tensor:
        del batch_idx
        return self._step(batch, stage="train")

    def validation_step(self, batch: dict[str, Any], batch_idx: int) -> None:
        del batch_idx
        self._step(batch, stage="val")

    def on_validation_epoch_start(self) -> None:
        self._val_outputs = []

    def on_validation_epoch_end(self) -> None:
        outputs, self._val_outputs = self._val_outputs, []
        if not outputs:
            return
        slides = [s for o in outputs for s in o["slides"]]
        unique = sorted(set(slides))
        if len(unique) < 2:
            return  # e.g. limit_val_batches: no WSI-disjoint split possible
        test_slides = set(unique[1::2])
        test = torch.tensor([s in test_slides for s in slides], device=self.device)
        domains = torch.cat([o["domains"] for o in outputs])
        for name in ("zgap", "backbone"):
            x = torch.cat([o[name] for o in outputs])
            accuracy = linear_probe_accuracy(
                x[~test], domains[~test], x[test], domains[test], self.num_domains
            )
            self.log(f"val/probe_{name}_acc", accuracy)

    def _step(self, batch: dict[str, Any], stage: str) -> torch.Tensor:
        images = batch[self.image_key]
        domains = batch[self.domain_key].long().view(-1)
        network = self.network
        hp = self.hparams
        step = int(self.global_step)

        feature_map = network.backbone_feature_map(images)
        mu, logvar = network.encode_features(feature_map)
        logvar = logvar.float()
        if stage == "train":
            z = mu + torch.randn_like(mu) * (0.5 * logvar).exp().to(mu.dtype)
        else:
            z = mu
        features = network.decode_features(network.scanner_embedding(domains), z)

        recon = feature_recon_loss(features, feature_map)
        kl = gaussian_kl(mu.float(), logvar)
        beta = hp.kl_weight * ramp(step, hp.kl_warmup_steps)

        z_gap = mu.float().mean(dim=(2, 3))
        adversary_input = F.layer_norm(z_gap, (z_gap.shape[-1],))
        logits = self.adversary(self.grl(adversary_input))
        adversary_ce = F.cross_entropy(logits, domains) / math.log(self.num_domains)
        adversary_w = hp.adversary_weight * ramp(step, hp.adversary_warmup_steps)

        pixel = recon.new_zeros(())
        if network.decoder is not None and hp.pixel_weight > 0:
            pixel = F.mse_loss(network.decoder(features.detach()).float(), images)

        # val/loss keeps one meaning across steps: the VAE objective only.
        vae_loss = recon + beta * kl
        loss = vae_loss + adversary_w * adversary_ce + hp.pixel_weight * pixel

        tokens = mu.detach().float().permute(0, 2, 3, 1).reshape(-1, mu.shape[1])
        metrics = {
            "feature_recon_loss": recon,
            "kl": kl,
            "kl_beta": beta,
            "adversary_ce": adversary_ce,
            "adversary_acc": (logits.argmax(-1) == domains).float().mean(),
            "adversary_weight": adversary_w,
            "pixel_loss": pixel,
            "active_units": (tokens.var(dim=0) > 0.01).float().sum(),
            "sigma_mean": (0.5 * logvar.detach()).exp().mean(),
            "z_abs_mean": tokens.abs().mean(),
            "z_std": tokens.std(),
        }
        on_step = stage == "train"
        batch_size = images.shape[0]
        self.log_dict(
            {f"{stage}/{k}": v for k, v in metrics.items()},
            on_step=on_step,
            on_epoch=True,
            batch_size=batch_size,
        )
        self.log(
            f"{stage}/loss",
            loss if stage == "train" else vae_loss,
            on_step=on_step,
            on_epoch=True,
            prog_bar=True,
            batch_size=batch_size,
        )

        if stage == "val":
            self._val_outputs.append(
                {
                    "zgap": z_gap.detach(),
                    "backbone": feature_map.detach().float().mean(dim=(2, 3)),
                    "domains": domains,
                    "slides": [p.split("-")[0] for p in batch["pair_id"]],
                }
            )
        return loss
