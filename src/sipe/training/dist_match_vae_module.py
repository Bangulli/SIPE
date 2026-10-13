"""Training module for DistMatchVAE: unpaired scanner removal by distribution matching
(not in legacy CATS; no curriculum, no cycle, no adversary).

Same network as PairedVAE (`PairedVAE`), but training uses scanner labels only: no
pair, no location. Training batches are plain shuffled tiles
(`UnpairedTrainSCORPIONDataModule`), so a batch holds random locations per scanner and
pairing cannot leak into the matching terms. Validation keeps paired batches (every
scanner of each val location) only to log the same metrics as PairedVAE.

With g = GAP(mu), standardized per dimension over the batch (statistics not detached,
so shrinking z or a dimension does not lower the matching terms):

loss = w_rec  * recon(Dec(s[donor(i)], z_i), x_i)        (Mathieu swap, same scanner)
     + w_mmd  * mean_{scanners a<b} MMD^2(g_a, g_b)       (multi-bandwidth RBF)
     + w_mean * mean_a ||mean(g_a)||^2 / D                (pooled mean is 0)
     + w_cov  * mean_a ||Cov(g_a) - Cov(g)||_F^2 / D^2
     + beta   * KL(q(z|x) || N(0, I))
     [+ pixel MSE of a visualization decoder on detached features]

- mean/cov: what a linear probe reads (class-conditional first and second moments).
- MMD: unbiased estimate, RBF kernels at `mmd_bandwidths` x the batch median squared
  distance (detached), for differences beyond the second moment.
- Assumption: every scanner sees the same tissue distribution (true for SCORPION by
  construction; random tiles per scanner keep it true in expectation). It breaks on a
  dataset where tissue differs by scanner.

The same terms on the detached backbone GAP are logged as `*_backbone` for scale.
In validation (paired batches, all 5 scanners per location) every scanner sees the
same tissue, so val/mmd and val/moment terms measure the scanner effect alone.
"""

from __future__ import annotations

import itertools
from typing import Any

import torch
import torch.nn.functional as F

from sipe.model.paired_vae import PairedVAE
from sipe.training.base import FrozenBackboneModule
from sipe.training.paired_vae_module import log_paired_validation, same_scanner_donor
from sipe.training.scanner_vae_module import feature_recon_loss, gaussian_kl, ramp


def standardize(g: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    """Per-dimension z-score over the batch; gradients flow through the statistics."""
    return (g - g.mean(0)) / (g.std(0) + eps)


def scanner_groups(domains: torch.Tensor, min_count: int = 2) -> list[torch.Tensor]:
    """Indices per scanner present at least `min_count` times."""
    groups = [(domains == d).nonzero().flatten() for d in domains.unique()]
    return [idx for idx in groups if idx.numel() >= min_count]


def scanner_moments(
    g: torch.Tensor, domains: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    """(mean term, covariance term) of standardized g, averaged over scanners."""
    x = standardize(g.float())
    groups = scanner_groups(domains)
    if len(groups) < 2:
        zero = x.new_zeros(())
        return zero, zero
    dim = x.shape[1]
    pooled = torch.cov(x.T)
    means, covs = [], []
    for idx in groups:
        xa = x[idx]
        means.append(xa.mean(0).square().sum() / dim)
        covs.append((torch.cov(xa.T) - pooled).square().sum() / dim**2)
    return torch.stack(means).mean(), torch.stack(covs).mean()


def scanner_mmd(
    g: torch.Tensor,
    domains: torch.Tensor,
    bandwidths: tuple[float, ...] = (0.25, 0.5, 1.0, 2.0, 4.0),
) -> torch.Tensor:
    """Unbiased MMD^2 between every pair of scanners on standardized g, mean over
    pairs. Kernel: mean of RBFs with widths `bandwidths` x median squared distance."""
    x = standardize(g.float())
    groups = scanner_groups(domains)
    if len(groups) < 2:
        return x.new_zeros(())
    d2 = torch.cdist(x, x).square()
    n = x.shape[0]
    off = ~torch.eye(n, dtype=torch.bool, device=x.device)
    scale = d2.detach()[off].median().clamp_min(1e-6)
    kernel = torch.stack([torch.exp(-d2 / (b * scale)) for b in bandwidths]).mean(0)

    def within(idx: torch.Tensor) -> torch.Tensor:
        k = kernel[idx][:, idx]
        m = idx.numel()
        return (k.sum() - k.diagonal().sum()) / (m * (m - 1))

    values = [
        within(a) + within(b) - 2 * kernel[a][:, b].mean()
        for a, b in itertools.combinations(groups, 2)
    ]
    return torch.stack(values).mean()


class DistMatchVAEModule(FrozenBackboneModule):
    def __init__(
        self,
        network: PairedVAE,
        num_domains: int = 5,
        lr: float = 3e-4,
        weight_decay: float = 0.01,
        kl_weight: float = 0.1,
        kl_warmup_steps: int = 1000,
        recon_weight: float = 1.0,
        mmd_weight: float = 1.0,
        mean_weight: float = 1.0,
        cov_weight: float = 1.0,
        match_warmup_steps: int = 1000,
        mmd_bandwidths: tuple[float, ...] = (0.25, 0.5, 1.0, 2.0, 4.0),
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
        self._val_outputs: list[dict[str, Any]] = []
        self.save_hyperparameters()
        if freeze_backbone:
            network.freeze_backbone()

    def configure_optimizers(self) -> dict[str, Any]:
        params = [p for p in self.parameters() if p.requires_grad]
        optimizer, scheduler = self.cosine_adamw(
            params, self.hparams.lr, self.hparams.weight_decay
        )
        return {"optimizer": optimizer, "lr_scheduler": scheduler}

    def training_step(self, batch: dict[str, Any], batch_idx: int) -> torch.Tensor:
        del batch_idx
        return self._step(batch, stage="train")

    def validation_step(self, batch: dict[str, Any], batch_idx: int) -> None:
        # Fixed donors in validation, so val losses are comparable across steps.
        generator = torch.Generator().manual_seed(batch_idx)
        self._step(batch, stage="val", generator=generator)

    def on_validation_epoch_start(self) -> None:
        self._val_outputs = []

    def on_validation_epoch_end(self) -> None:
        outputs, self._val_outputs = self._val_outputs, []
        if outputs:
            log_paired_validation(self, outputs)

    def _step(
        self,
        batch: dict[str, Any],
        stage: str,
        generator: torch.Generator | None = None,
    ) -> torch.Tensor:
        images = batch[self.image_key]
        domains = batch[self.domain_key].long().view(-1)
        network = self.network
        hp = self.hparams
        step = int(self.global_step)
        bandwidths = tuple(hp.mmd_bandwidths)

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

        z_gap = mu.float().mean(dim=(2, 3))
        backbone_gap = feature_map.detach().float().mean(dim=(2, 3))
        mmd = scanner_mmd(z_gap, domains, bandwidths)
        mean_term, cov_term = scanner_moments(z_gap, domains)
        match_ramp = ramp(step, hp.match_warmup_steps)

        kl = gaussian_kl(mu.float(), logvar)
        beta = hp.kl_weight * ramp(step, hp.kl_warmup_steps)

        pixel = recon.new_zeros(())
        if network.decoder is not None and hp.pixel_weight > 0:
            pixel = F.mse_loss(network.decoder(features.detach()).float(), images)

        objective = (
            hp.recon_weight * recon
            + match_ramp
            * (
                hp.mmd_weight * mmd
                + hp.mean_weight * mean_term
                + hp.cov_weight * cov_term
            )
            + beta * kl
        )
        loss = objective + hp.pixel_weight * pixel

        with torch.no_grad():
            mean_backbone, cov_backbone = scanner_moments(backbone_gap, domains)
            mmd_backbone = scanner_mmd(backbone_gap, domains, bandwidths)
        tokens = mu.detach().float().permute(0, 2, 3, 1).reshape(-1, mu.shape[1])
        metrics = {
            "feature_recon_loss": recon,
            "mmd": mmd,
            "mmd_backbone": mmd_backbone,
            "mean_loss": mean_term,
            "mean_loss_backbone": mean_backbone,
            "cov_loss": cov_term,
            "cov_loss_backbone": cov_backbone,
            "match_ramp": match_ramp,
            "kl": kl,
            "kl_beta": beta,
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
        # val/loss: the objective without the visualization decoder's pixel term.
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
                    "zgap": z_gap.detach(),
                    "backbone": backbone_gap,
                    "s": s.detach().float(),
                    "domains": domains,
                    "pair_ids": list(batch["pair_id"]),
                }
            )
        return loss
