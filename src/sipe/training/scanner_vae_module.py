"""Training module for ScannerVAE (not in legacy CATS; no curriculum, no cycle).

loss = recon(Dec(e(scanner), z), x) + beta * KL(q(z|x) || N(0, I))
       + w_adv * adversarial term on LN(GAP(mu))
       [+ pixel MSE of a visualization decoder on detached features]

Adversarial term (`adversary_mode`):
- "grl": CE(adversary(GRL(.)), scanner), one optimizer (adversary and encoder in the
  same backward, with reversed gradients for the encoder);
- "fader" (Lample et al. 2017, alternating): the encoder minimizes the adversary's CE
  to a *uniform* scanner distribution (confusion), then the adversary takes
  `adversary_steps` CE steps on the detached GAP(mu). Manual optimization; only the
  encoder step counts as a global step, so max_steps keeps its meaning.

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

import torch
import torch.nn as nn
import torch.nn.functional as F

from sipe.model.scanner_vae import ScannerVAE
from sipe.training.base import FrozenBackboneModule
from sipe.training.grl import GradientReversal


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


class ScannerVAEModule(FrozenBackboneModule):
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
        adversary_mode: str = "grl",
        adversary_steps: int = 1,
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
        if adversary_mode not in ("grl", "fader"):
            raise ValueError(f"Unknown adversary_mode {adversary_mode!r}.")
        if adversary_steps < 1:
            raise ValueError("adversary_steps must be >= 1.")
        # Fader alternates encoder and adversary updates within a batch.
        self.automatic_optimization = adversary_mode == "grl"
        self._val_outputs: list[dict[str, Any]] = []
        self.save_hyperparameters()
        if freeze_backbone:
            network.freeze_backbone()

    def configure_optimizers(self) -> Any:
        adversary = list(self.adversary.parameters())
        adversary_ids = {id(p) for p in adversary}
        main = [
            p
            for p in self.parameters()
            if p.requires_grad and id(p) not in adversary_ids
        ]
        hp = self.hparams
        if hp.adversary_mode == "fader":
            main_opt, scheduler = self.cosine_adamw(main, hp.lr, hp.weight_decay)
            # Constant LR for the adversary: it must keep up until the end.
            adversary_opt = torch.optim.AdamW(
                adversary,
                lr=hp.lr * hp.adversary_lr_mult,
                weight_decay=hp.weight_decay,
            )
            return [main_opt, adversary_opt], [scheduler]
        optimizer, scheduler = self.cosine_adamw(
            [
                {"params": main, "lr": hp.lr},
                {"params": adversary, "lr": hp.lr * hp.adversary_lr_mult},
            ],
            hp.lr,
            hp.weight_decay,
        )
        return {"optimizer": optimizer, "lr_scheduler": scheduler}

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
        for parameter in self.adversary.parameters():
            self._adversary_optimizer().state.pop(parameter, None)

    def _adversary_optimizer(self) -> torch.optim.Optimizer:
        optimizers = self.optimizers(use_pl_optimizer=False)
        if self.hparams.adversary_mode == "fader":
            return optimizers[1]
        return optimizers  # grl: one optimizer, adversary = its second group

    def training_step(self, batch: dict[str, Any], batch_idx: int) -> torch.Tensor:
        del batch_idx
        loss, z_gap, domains = self._step(batch, stage="train")
        if self.automatic_optimization:
            return loss

        main_opt, _ = self.optimizers()
        main_opt.zero_grad()
        self.manual_backward(loss)
        main_opt.step()  # the only step that advances global_step
        self.lr_schedulers().step()
        # After the encoder step: updating the adversary before the backward above
        # would modify weights its graph still needs.
        self._train_adversary(z_gap.detach(), domains)
        return loss.detach()

    def _train_adversary(self, z_gap: torch.Tensor, domains: torch.Tensor) -> None:
        """Fader adversary: `adversary_steps` CE steps on detached GAP(mu), fp32.

        Uses the raw torch optimizer (no AMP scaler needed in fp32), so these steps
        do not count as Lightning global steps.
        """
        optimizer = self._adversary_optimizer()
        x = F.layer_norm(z_gap.float(), (z_gap.shape[-1],))
        with torch.autocast(device_type=self.device.type, enabled=False):
            for _ in range(self.hparams.adversary_steps):
                optimizer.zero_grad(set_to_none=True)
                logits = self.adversary(x)
                ce = F.cross_entropy(logits, domains)
                ce.backward()
                optimizer.step()
        self.log_dict(
            {
                "train/adversary_fit_ce": ce.detach() / math.log(self.num_domains),
                "train/adversary_fit_acc": (logits.argmax(-1) == domains)
                .float()
                .mean(),
            },
            on_step=True,
            on_epoch=True,
            batch_size=domains.shape[0],
        )

    def validation_step(self, batch: dict[str, Any], batch_idx: int) -> None:
        del batch_idx
        self._step(batch, stage="val")

    def on_validation_epoch_start(self) -> None:
        self._val_outputs = []

    def on_validation_epoch_end(self) -> None:
        outputs, self._val_outputs = self._val_outputs, []
        if not outputs:
            return
        self.log_wsi_probes(
            {
                name: torch.cat([o[name] for o in outputs])
                for name in ("zgap", "backbone")
            },
            torch.cat([o["domains"] for o in outputs]),
            [s for o in outputs for s in o["slides"]],
        )

    def _step(
        self, batch: dict[str, Any], stage: str
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Loss (train: with the adversarial term), GAP(mu) and scanner labels."""
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
        log_k = math.log(self.num_domains)
        if hp.adversary_mode == "grl":
            logits = self.adversary(self.grl(adversary_input))
        else:
            logits = self.adversary(adversary_input)
        # CE on the true scanner (what the adversary minimizes), divided by log K.
        adversary_ce = F.cross_entropy(logits.float(), domains) / log_k
        # Fader encoder term: CE to the uniform distribution / log K (minimum 1).
        confusion = -F.log_softmax(logits.float(), dim=-1).mean() / log_k
        adversarial = adversary_ce if hp.adversary_mode == "grl" else confusion
        adversary_w = hp.adversary_weight * ramp(step, hp.adversary_warmup_steps)

        pixel = recon.new_zeros(())
        if network.decoder is not None and hp.pixel_weight > 0:
            pixel = F.mse_loss(network.decoder(features.detach()).float(), images)

        # val/loss keeps one meaning across steps: the VAE objective only.
        vae_loss = recon + beta * kl
        loss = vae_loss + adversary_w * adversarial + hp.pixel_weight * pixel

        tokens = mu.detach().float().permute(0, 2, 3, 1).reshape(-1, mu.shape[1])
        metrics = {
            "feature_recon_loss": recon,
            "kl": kl,
            "kl_beta": beta,
            "adversary_ce": adversary_ce,
            "adversary_confusion": confusion,
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
        return loss, z_gap, domains
