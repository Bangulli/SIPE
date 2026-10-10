"""Training module for PairedVAE (not in legacy CATS; no curriculum, no cycle).

Batches come from PairedSCORPIONDataModule: locations × scanner views, views of a
location contiguous, all views with the same augmentation. For image i, `donor(i)` is
another image of the same scanner in the batch (another location), and `partner(i)`
the next view of the same location (another scanner).

loss = w_rec   * recon(Dec(s[donor(i)], z_i), x_i)                    (Mathieu swap)
     + w_trans * recon(P(Dec(s[donor(partner(i))], z_i)), P(x_partner(i)))
     + w_align * ||g_i - g_partner(i)||^2 / tr Var_batch(g),  g = GAP(mu)
     + beta    * KL(q(z|x) || N(0, I))
     [+ pixel MSE of a visualization decoder on detached features]

- recon: Mathieu's same-class swap, here same scanner: s comes from another tile of
  the same scanner, so s cannot carry the tile's content (no plain self-recon term).
- translation: the paired stand-in for Mathieu's GAN. z of a tile plus s of the
  partner's scanner must give the partner's features. P average-pools both to a
  coarser grid (`translation_pool`): P1000 tiles sit ~1 token (~7 µm) off the other
  scanners (checked on 64 locations, 2026-10-10), so per-token targets are shifted.
- align: pulls GAP(mu) of the same tissue on two scanners together. Dividing by the
  batch variance of g (not detached) makes collapsing z cost, not pay: random pairs
  score ~2, identical pairs 0.

No adversary. Validation trains fresh linear scanner probes on GAP(mu), backbone GAP
and s (WSI-disjoint halves of val) and measures PLISM-style cross-scanner retrieval
top-1 (as the SCORPION proxy) on GAP(mu) vs backbone GAP over the whole val split.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import torch
import torch.nn.functional as F

from sipe.model.paired_vae import PairedVAE
from sipe.training.base import FrozenBackboneModule
from sipe.training.scanner_vae_module import feature_recon_loss, gaussian_kl, ramp


def same_scanner_donor(
    domains: torch.Tensor, generator: torch.Generator | None = None
) -> torch.Tensor:
    """donor[i]: a random other index with the same scanner (i itself if alone)."""
    donor = torch.arange(domains.shape[0], device=domains.device)
    for d in domains.unique():
        idx = (domains == d).nonzero().flatten()
        if idx.numel() < 2:
            continue
        order = torch.randperm(idx.numel(), generator=generator).to(idx.device)
        shuffled = idx[order]
        donor[shuffled] = shuffled.roll(1)
    return donor


def partner_index(location: torch.Tensor) -> torch.Tensor:
    """partner[i]: the next view of i's location (cyclic); views must be contiguous."""
    n = location.shape[0]
    counts = torch.bincount(location)
    views = int(counts[0])
    idx = torch.arange(n, device=location.device)
    if views < 2 or not bool((counts == views).all()):
        raise ValueError("Every location needs the same number (>= 2) of views.")
    if not torch.equal(location, idx // views):
        raise ValueError("Views of a location must be contiguous.")
    return (idx // views) * views + (idx % views + 1) % views


def pair_alignment(g: torch.Tensor, partner: torch.Tensor) -> torch.Tensor:
    """Mean squared pair distance / total batch variance (~2 for random pairs)."""
    return (g - g[partner]).square().sum(1).mean() / g.var(0).sum().clamp_min(1e-6)


@torch.no_grad()
def cross_scanner_top1(
    x: torch.Tensor,
    domains: torch.Tensor,
    locations: torch.Tensor,
    slides: torch.Tensor,
) -> float:
    """PLISM-style top-1 as in `sipe.eval.scorpion_retrieval`: per slide and scanner
    pair (a, b), the gallery is the tiles of both scanners (self excluded) and a hit
    is the same location on the other scanner. Mean over (slide, pair) cells.

    Same-scanner tiles compete with the true match, so a scanner signature in x
    lowers the score.
    """
    x = F.normalize(x.float(), dim=1)
    scores = []
    for slide in slides.unique():
        in_slide = slides == slide
        scanners = domains[in_slide].unique().tolist()
        for i, a in enumerate(scanners):
            for b in scanners[i + 1 :]:
                ia = (in_slide & (domains == a)).nonzero().flatten()
                ib = (in_slide & (domains == b)).nonzero().flatten()
                common = torch.from_numpy(
                    np.intersect1d(locations[ia].cpu(), locations[ib].cpu())
                ).to(x.device)
                if common.numel() < 2:
                    continue
                ia = ia[torch.isin(locations[ia], common)]
                ib = ib[torch.isin(locations[ib], common)]
                ia = ia[locations[ia].argsort()]
                ib = ib[locations[ib].argsort()]
                rows = torch.cat([ia, ib])
                sim = x[rows] @ x[rows].T
                sim.fill_diagonal_(float("-inf"))
                n = ia.numel()
                target = torch.cat([torch.arange(n, 2 * n), torch.arange(n)])
                hits = sim.argmax(1).cpu() == target
                scores.append(hits.float().mean())
    return float(torch.stack(scores).mean()) if scores else float("nan")


class PairedVAEModule(FrozenBackboneModule):
    def __init__(
        self,
        network: PairedVAE,
        num_domains: int = 5,
        lr: float = 3e-4,
        weight_decay: float = 0.01,
        kl_weight: float = 0.1,
        kl_warmup_steps: int = 1000,
        recon_weight: float = 1.0,
        translation_weight: float = 1.0,
        translation_pool: int = 4,
        align_weight: float = 1.0,
        align_warmup_steps: int = 1000,
        pixel_weight: float = 1.0,
        freeze_backbone: bool = True,
        image_key: str = "image",
        domain_key: str = "domain",
        location_key: str = "location",
    ) -> None:
        super().__init__()
        self.network = network
        self.num_domains = int(num_domains)
        self.image_key = image_key
        self.domain_key = domain_key
        self.location_key = location_key
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
        if not outputs:
            return
        domains = torch.cat([o["domains"] for o in outputs])
        pair_ids = [p for o in outputs for p in o["pair_ids"]]
        location_of = {p: i for i, p in enumerate(sorted(set(pair_ids)))}
        locations = torch.tensor([location_of[p] for p in pair_ids], device=self.device)
        features = {
            name: torch.cat([o[name] for o in outputs])
            for name in ("zgap", "backbone", "s")
        }
        slides = [p.split("-")[0] for p in pair_ids]
        unique = sorted(set(slides))
        slide_index = torch.tensor(
            [unique.index(s) for s in slides], device=self.device
        )
        for name in ("zgap", "backbone"):
            top1 = cross_scanner_top1(features[name], domains, locations, slide_index)
            self.log(f"val/retrieval_top1_{name}", top1)
        self.log_wsi_probes(features, domains, slides)

    def _step(
        self,
        batch: dict[str, Any],
        stage: str,
        generator: torch.Generator | None = None,
    ) -> torch.Tensor:
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
        partner = partner_index(location)
        s_donor = s[donor]

        features = network.decode_features(s_donor, z)
        recon = feature_recon_loss(features, feature_map)

        pool = hp.translation_pool
        translated = network.decode_features(s_donor[partner], z)
        translation = feature_recon_loss(
            F.avg_pool2d(translated, pool), F.avg_pool2d(feature_map[partner], pool)
        )

        z_gap = mu.float().mean(dim=(2, 3))
        backbone_gap = feature_map.detach().float().mean(dim=(2, 3))
        align = pair_alignment(z_gap, partner)
        align_w = hp.align_weight * ramp(step, hp.align_warmup_steps)

        kl = gaussian_kl(mu.float(), logvar)
        beta = hp.kl_weight * ramp(step, hp.kl_warmup_steps)

        pixel = recon.new_zeros(())
        if network.decoder is not None and hp.pixel_weight > 0:
            pixel = F.mse_loss(network.decoder(features.detach()).float(), images)

        objective = (
            hp.recon_weight * recon
            + hp.translation_weight * translation
            + align_w * align
            + beta * kl
        )
        loss = objective + hp.pixel_weight * pixel

        tokens = mu.detach().float().permute(0, 2, 3, 1).reshape(-1, mu.shape[1])
        metrics = {
            "feature_recon_loss": recon,
            "translation_loss": translation,
            "align_loss": align,
            "align_loss_backbone": pair_alignment(backbone_gap, partner),
            "pair_cos_z": F.cosine_similarity(z_gap, z_gap[partner]).mean(),
            "pair_cos_backbone": F.cosine_similarity(
                backbone_gap, backbone_gap[partner]
            ).mean(),
            "align_weight": align_w,
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
