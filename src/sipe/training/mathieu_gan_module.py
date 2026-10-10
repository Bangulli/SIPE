"""Training module for stage 2 toward Mathieu et al. 2016 (not in legacy CATS).

Same network as stage 1 (PairedVAE: s = MLP(GAP(x)), per-token VAE z, FiLM decoder),
but the paired translation is replaced by Mathieu's class-conditional GAN in the
frozen backbone's token space. Training is unpaired: only scanner labels are used.

For image i (scanner y_i), `donor(i)` is another tile of the same scanner and
`other(i)` a tile of another scanner and another location (never i's pair partner).

generator loss = w_rec * recon(Dec(s[donor(i)], z_i), x_i)          (Mathieu swap)
               + w_adv * -D(Dec(s[other(i)], z_i), y_other(i))      (hinge G loss)
               + beta  * KL(q(z|x) || N(0, I))
               [+ pixel MSE of a visualization decoder on detached features]
discriminator  = hinge: relu(1 - D(x_i, y_i)) + relu(1 + D(fake_i, y_other(i)))

Why the GAN: without it the swap has a trivial solution (the decoder ignores s and z
keeps the scanner). D asks "do these features look like scanner y_other(i)?", so the
scanner signature must come from s, not from z. Unlike the adversaries on z
(ScannerVAE), nothing classifies z directly.

D is a projection discriminator (Miyato & Koyama 2018) with spectral norm on the
16x16 token grid. Manual optimization, alternating: one generator step (the only step
that counts as a global step, so max_steps keeps its meaning), then
`discriminator_steps` D steps in fp32 on detached fakes. D's LR follows the same
cosine decay as the generator's by default: in the Fader run a constant-LR adversary
dominated once the encoder's LR had decayed.

Validation (same batches and metrics as PairedVAEModule, so the two compare directly):
probes on GAP(mu)/backbone/s, cross-scanner retrieval top-1, and, as paired
*evaluation* only, the 4x4-pooled translation loss to the true partner. GAN
diagnostics: a probe trained on real backbone GAP (train-half WSIs) reads the scanner
of GAP(fake) on the test half; `val/fake_target_acc` (the requested scanner) should
rise, `val/fake_source_acc` (the scanner of z's tile) should fall.
"""

from __future__ import annotations

from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.parametrizations import spectral_norm
from torch.optim.lr_scheduler import CosineAnnealingLR

from sipe.model.paired_vae import PairedVAE
from sipe.training.base import FrozenBackboneModule, linear_probe_accuracy
from sipe.training.paired_vae_module import (
    log_paired_validation,
    partner_index,
    same_scanner_donor,
)
from sipe.training.scanner_vae_module import feature_recon_loss, gaussian_kl, ramp


def other_scanner_index(
    domains: torch.Tensor,
    location: torch.Tensor,
    generator: torch.Generator | None = None,
) -> torch.Tensor:
    """other[i]: a random index with another scanner and another location (i itself
    if there is none), so the GAN never sees i's pair partner."""
    device = domains.device
    domains, location = domains.cpu(), location.cpu()
    allowed = (domains[:, None] != domains[None, :]) & (
        location[:, None] != location[None, :]
    )
    alone = ~allowed.any(1)
    allowed[alone, alone.nonzero().flatten()] = True
    other = torch.multinomial(allowed.float(), 1, generator=generator).flatten()
    return other.to(device)


class ProjectionDiscriminator(nn.Module):
    """D(x, y) = psi(h) + <embed(y), h>, h = pooled conv features of the token grid.

    Spectral norm on every layer; 1x1 conv, then two stride-2 3x3 convs
    (16x16 -> 4x4 tokens), then mean pooling.
    """

    def __init__(self, in_dim: int, hidden_dim: int, num_classes: int) -> None:
        super().__init__()
        self.body = nn.Sequential(
            spectral_norm(nn.Conv2d(in_dim, hidden_dim, kernel_size=1)),
            nn.LeakyReLU(0.2),
            spectral_norm(nn.Conv2d(hidden_dim, hidden_dim, 3, stride=2, padding=1)),
            nn.LeakyReLU(0.2),
            spectral_norm(nn.Conv2d(hidden_dim, hidden_dim, 3, stride=2, padding=1)),
            nn.LeakyReLU(0.2),
        )
        self.psi = spectral_norm(nn.Linear(hidden_dim, 1))
        self.embed = spectral_norm(nn.Embedding(num_classes, hidden_dim))

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        h = self.body(x).mean(dim=(2, 3))
        return self.psi(h).squeeze(1) + (self.embed(y) * h).sum(1)


class MathieuGANModule(FrozenBackboneModule):
    def __init__(
        self,
        network: PairedVAE,
        num_domains: int = 5,
        lr: float = 3e-4,
        weight_decay: float = 0.01,
        kl_weight: float = 0.1,
        kl_warmup_steps: int = 1000,
        recon_weight: float = 1.0,
        adversarial_weight: float = 1.0,
        adversarial_warmup_steps: int = 1000,
        discriminator_hidden_dim: int = 256,
        discriminator_lr: float | None = None,
        discriminator_steps: int = 1,
        discriminator_lr_decay: bool = True,
        translation_pool: int = 4,
        pixel_weight: float = 1.0,
        freeze_backbone: bool = True,
        image_key: str = "image",
        domain_key: str = "domain",
        location_key: str = "location",
    ) -> None:
        super().__init__()
        if discriminator_steps < 1:
            raise ValueError("discriminator_steps must be >= 1.")
        self.network = network
        self.num_domains = int(num_domains)
        self.image_key = image_key
        self.domain_key = domain_key
        self.location_key = location_key
        self.discriminator = ProjectionDiscriminator(
            network.feature_dim, discriminator_hidden_dim, self.num_domains
        )
        # Alternating generator / discriminator updates within a batch.
        self.automatic_optimization = False
        self._val_outputs: list[dict[str, Any]] = []
        self.save_hyperparameters()
        if freeze_backbone:
            network.freeze_backbone()

    def configure_optimizers(self) -> Any:
        hp = self.hparams
        d_params = list(self.discriminator.parameters())
        d_ids = {id(p) for p in d_params}
        g_params = [
            p for p in self.parameters() if p.requires_grad and id(p) not in d_ids
        ]
        g_opt, g_scheduler = self.cosine_adamw(g_params, hp.lr, hp.weight_decay)
        # Usual GAN Adam setting for D (beta1 0.5, no weight decay; SN regularizes).
        d_opt = torch.optim.Adam(
            d_params, lr=hp.discriminator_lr or hp.lr, betas=(0.5, 0.999)
        )
        schedulers = [g_scheduler]
        if hp.discriminator_lr_decay:
            schedulers.append(
                {
                    "scheduler": CosineAnnealingLR(d_opt, T_max=self.trainer.max_steps),
                    "interval": "step",
                }
            )
        return [g_opt, d_opt], schedulers

    def training_step(self, batch: dict[str, Any], batch_idx: int) -> torch.Tensor:
        del batch_idx
        loss, gan = self._step(batch, stage="train")
        g_opt, _ = self.optimizers()
        g_opt.zero_grad()
        self.manual_backward(loss)
        g_opt.step()  # the only step that advances global_step
        # After the generator step: updating D before the backward above would
        # modify weights its graph still needs.
        self._train_discriminator(**gan)
        # Both schedulers after both optimizer steps (torch's required order).
        schedulers = self.lr_schedulers()
        for scheduler in schedulers if isinstance(schedulers, list) else [schedulers]:
            scheduler.step()
        return loss.detach()

    def _train_discriminator(
        self,
        real: torch.Tensor,
        real_labels: torch.Tensor,
        fake: torch.Tensor,
        fake_labels: torch.Tensor,
    ) -> None:
        """Hinge D steps on real tokens and detached fakes, fp32.

        Uses the raw torch optimizer (no AMP scaler needed in fp32), so these steps
        do not count as Lightning global steps.
        """
        optimizer = self.optimizers(use_pl_optimizer=False)[1]
        real, fake = real.detach().float(), fake.detach().float()
        with torch.autocast(device_type=self.device.type, enabled=False):
            for _ in range(self.hparams.discriminator_steps):
                optimizer.zero_grad(set_to_none=True)
                real_logits = self.discriminator(real, real_labels)
                fake_logits = self.discriminator(fake, fake_labels)
                d_loss = F.relu(1 - real_logits).mean() + F.relu(1 + fake_logits).mean()
                d_loss.backward()
                optimizer.step()
        self.log_dict(
            {
                "train/d_loss": d_loss.detach(),
                "train/d_real_acc": (real_logits > 0).float().mean(),
                "train/d_fake_acc": (fake_logits < 0).float().mean(),
            },
            on_step=True,
            on_epoch=True,
            batch_size=real.shape[0],
        )

    def validation_step(self, batch: dict[str, Any], batch_idx: int) -> None:
        # Fixed donors in validation, so val losses are comparable across steps.
        generator = torch.Generator().manual_seed(batch_idx)
        self._step(batch, stage="val", generator=generator)

    def on_validation_epoch_start(self) -> None:
        self._val_outputs = []

    def on_validation_epoch_end(self) -> None:
        outputs, self._val_outputs = self._val_outputs, []
        if not outputs:
            return
        log_paired_validation(self, outputs)
        self._log_fake_probe(outputs)

    def _log_fake_probe(self, outputs: list[dict[str, Any]]) -> None:
        """Scanner of GAP(fake) read by a probe trained on real backbone GAP."""
        slides = [p.split("-")[0] for o in outputs for p in o["pair_ids"]]
        unique = sorted(set(slides))
        if len(unique) < 2:
            return
        test_slides = set(unique[1::2])
        test = torch.tensor([s in test_slides for s in slides], device=self.device)
        backbone = torch.cat([o["backbone"] for o in outputs])
        domains = torch.cat([o["domains"] for o in outputs])
        fake = torch.cat([o["fake_gap"] for o in outputs])
        fake_labels = torch.cat([o["fake_labels"] for o in outputs])
        for name, target in (("target", fake_labels), ("source", domains)):
            accuracy = linear_probe_accuracy(
                backbone[~test],
                domains[~test],
                fake[test],
                target[test],
                self.num_domains,
            )
            self.log(f"val/fake_{name}_acc", accuracy)

    def _step(
        self,
        batch: dict[str, Any],
        stage: str,
        generator: torch.Generator | None = None,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        """Generator loss and the tensors D trains on."""
        images = batch[self.image_key]
        domains = batch[self.domain_key].long().view(-1)
        location = batch[self.location_key].long().view(-1)
        network = self.network
        hp = self.hparams
        step = int(self.global_step)

        feature_map = network.backbone_feature_map(images)
        s, mu, logvar = network.encode_features(feature_map)
        logvar = logvar.float()
        if stage == "train":
            z = mu + torch.randn_like(mu) * (0.5 * logvar).exp().to(mu.dtype)
        else:
            z = mu

        donor = same_scanner_donor(domains, generator)
        features = network.decode_features(s[donor], z)
        recon = feature_recon_loss(features, feature_map)

        other = other_scanner_index(domains, location, generator)
        fake_labels = domains[other]
        fake = network.decode_features(s[other], z)
        # D is frozen for the generator loss: its weights get no gradient here.
        self.discriminator.requires_grad_(False)
        fake_logits = self.discriminator(fake, fake_labels)
        self.discriminator.requires_grad_(True)
        g_adv = -fake_logits.float().mean()
        adv_w = hp.adversarial_weight * ramp(step, hp.adversarial_warmup_steps)

        kl = gaussian_kl(mu.float(), logvar)
        beta = hp.kl_weight * ramp(step, hp.kl_warmup_steps)

        pixel = recon.new_zeros(())
        if network.decoder is not None and hp.pixel_weight > 0:
            pixel = F.mse_loss(network.decoder(features.detach()).float(), images)

        # val/loss keeps one meaning across steps: swap recon + KL only.
        objective = hp.recon_weight * recon + beta * kl
        loss = objective + adv_w * g_adv + hp.pixel_weight * pixel

        tokens = mu.detach().float().permute(0, 2, 3, 1).reshape(-1, mu.shape[1])
        metrics = {
            "feature_recon_loss": recon,
            "g_adv_loss": g_adv,
            "adversarial_weight": adv_w,
            "fake_d_score": fake_logits.detach().float().mean(),
            "kl": kl,
            "kl_beta": beta,
            "pixel_loss": pixel,
            "active_units": (tokens.var(dim=0) > 0.01).float().sum(),
            "sigma_mean": (0.5 * logvar.detach()).exp().mean(),
            "z_abs_mean": tokens.abs().mean(),
            "z_std": tokens.std(),
        }
        z_gap = mu.detach().float().mean(dim=(2, 3))
        backbone_gap = feature_map.detach().float().mean(dim=(2, 3))
        if stage == "val":
            with torch.no_grad():
                real_logits = self.discriminator(feature_map.float(), domains)
            metrics["d_real_acc"] = (real_logits > 0).float().mean()
            metrics["d_fake_acc"] = (fake_logits < 0).float().mean()
            # Paired evaluation only (never trained on): z of a tile + s of another
            # tile of the partner's scanner vs the partner, 4x4-pooled as stage 1.
            partner = partner_index(location)
            translated = network.decode_features(s[donor][partner], z)
            pool = hp.translation_pool
            metrics["translation_loss"] = feature_recon_loss(
                F.avg_pool2d(translated, pool),
                F.avg_pool2d(feature_map[partner], pool),
            )
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
            loss if stage == "train" else objective,
            on_step=on_step,
            on_epoch=True,
            prog_bar=True,
            batch_size=batch_size,
        )

        if stage == "val":
            self._val_outputs.append(
                {
                    "zgap": z_gap,
                    "backbone": backbone_gap,
                    "s": s.detach().float(),
                    "domains": domains,
                    "pair_ids": list(batch["pair_id"]),
                    "fake_gap": fake.detach().float().mean(dim=(2, 3)),
                    "fake_labels": fake_labels,
                }
            )
        gan = {
            "real": feature_map,
            "real_labels": domains,
            "fake": fake,
            "fake_labels": fake_labels,
        }
        return loss, gan
