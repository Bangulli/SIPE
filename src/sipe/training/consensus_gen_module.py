"""Training module for the consensus generative model (Experiment 7, backbone in the
loop; docs/experiment7.md). Mirrors PairedVAEModule.

Batches come from PairedSCORPIONDataModule: locations × scanner views, views of a
location contiguous, all views with the same augmentation. Per tile and per source
view s (u = standardized backbone GAP):

loss = mean_{s, t in tile(s)} -log N(u_t; g(z_s) + delta_{k_t}, B_{k_t} B_{k_t}^T) / D
       (ContentShiftHead: mean g(z_s) + delta_{k_t} + D_{k_t} z_s)
     + beta * KL(q(z|u_s) || N(0, I)) / D
     + rho  * ||r, h output layers||^2,          z_s ~ q(z|u_s)

The first two terms are, per source view, the negative ELBO of all views of the tile
(including s) divided by D = 768; it replaces PairedVAE's swap, translation and
alignment losses. Pairing unit: tile-level pairs. Scanner labels enter only the
likelihood (delta_k, B_k), never the map.

on_fit_start sets the linear-Gaussian solution from `init_batches` train batches
(skipped on resume). Every checkpoint save also writes `map_cgm.pt` in the run dir,
a `--posthoc` module map file for sipe.bench.plism / sipe.bench.hest.

Validation: PLISM-style cross-scanner retrieval top-1 (SCORPION proxy) on f(x) and
backbone GAP over the whole val split, fresh WSI-split scanner probes on both, and
the mean val NLL / KL.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F

from sipe.model.consensus_gen import ConsensusGenModel, ContentShiftHead
from sipe.training.base import FrozenBackboneModule
from sipe.training.paired_vae_module import cross_scanner_top1, partner_index
from sipe.training.scanner_vae_module import ramp

MAP_FILENAME = "map_cgm.pt"


def tile_view_pairs(location: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """(source, target) index pairs: every ordered pair of views of the same tile,
    self included."""
    same = location[:, None] == location[None, :]
    source, target = same.nonzero(as_tuple=True)
    return source, target


class ConsensusGenModule(FrozenBackboneModule):
    def __init__(
        self,
        network: ConsensusGenModel,
        num_domains: int = 5,
        lr: float = 3e-4,
        linear_lr: float = 1e-4,
        weight_decay: float = 0.01,
        kl_weight: float = 1.0,
        kl_warmup_steps: int = 0,
        residual_penalty: float = 1e-3,
        init_batches: int = 40,
        init_ridge: float = 1e-3,
        freeze_backbone: bool = True,
        image_key: str = "image",
        domain_key: str = "domain",
        location_key: str = "location",
    ) -> None:
        super().__init__()
        if network.head.num_scanners != num_domains:
            raise ValueError(
                f"network.num_scanners={network.head.num_scanners} != "
                f"num_domains={num_domains}."
            )
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
        # Weight decay only on the MLP weights: decaying L, M or the Cholesky factors
        # would pull the linear-Gaussian solution toward zero.
        head = self.network.head
        mlp_decay, mlp_other, linear = [], [], []
        for name, p in head.named_parameters():
            if not p.requires_grad:
                continue
            if name.startswith(("map.r.", "map.h.")):
                (mlp_decay if p.ndim == 2 else mlp_other).append(p)
            else:
                linear.append(p)
        hp = self.hparams
        groups = [
            {"params": mlp_decay, "lr": hp.lr, "weight_decay": hp.weight_decay},
            {"params": mlp_other, "lr": hp.lr, "weight_decay": 0.0},
            {"params": linear, "lr": hp.linear_lr, "weight_decay": 0.0},
        ]
        groups = [g for g in groups if g["params"]]
        optimizer, scheduler = self.cosine_adamw(groups, hp.lr, hp.weight_decay)
        return {"optimizer": optimizer, "lr_scheduler": scheduler}

    def _gap(self, images: torch.Tensor) -> torch.Tensor:
        """Pooled frozen-backbone embedding (GAP or CLS, `network.pooling`)."""
        with torch.no_grad():
            return self.network.backbone_features(images)

    def on_fit_start(self) -> None:
        head = self.network.head
        if bool(head.initialized):
            return  # resumed fit: keep the trained state
        loader = self.trainer.datamodule.train_dataloader()
        xs, scanners, locations = [], [], []
        offset = 0
        with torch.no_grad(), self.trainer.precision_plugin.forward_context():
            for i, batch in enumerate(loader):
                if i >= self.hparams.init_batches:
                    break
                images = batch[self.image_key].to(self.device, non_blocking=True)
                xs.append(self._gap(images).double())
                scanners.append(batch[self.domain_key].long().view(-1))
                location = batch[self.location_key].long().view(-1)
                locations.append(location + offset)
                offset += int(location.max()) + 1
        stats = head.init_from_moments(
            torch.cat(xs),
            torch.cat(scanners).to(self.device),
            torch.cat(locations).to(self.device),
            ridge=self.hparams.init_ridge,
        )
        for logger in self.loggers:
            logger.log_metrics(stats, step=0)

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
        domains = torch.cat([o["domains"] for o in outputs])
        pair_ids = [p for o in outputs for p in o["pair_ids"]]
        location_of = {p: i for i, p in enumerate(sorted(set(pair_ids)))}
        locations = torch.tensor([location_of[p] for p in pair_ids], device=self.device)
        slides = [p.split("-")[0] for p in pair_ids]
        unique = sorted(set(slides))
        slide_index = torch.tensor(
            [unique.index(s) for s in slides], device=self.device
        )
        features = {
            name: torch.cat([o[name] for o in outputs]) for name in ("cgm", "backbone")
        }
        for name, x in features.items():
            top1 = cross_scanner_top1(x, domains, locations, slide_index)
            self.log(f"val/retrieval_top1_{name}", top1)
        self.log_wsi_probes(features, domains, slides)

    def on_save_checkpoint(self, checkpoint: dict[str, Any]) -> None:
        super().on_save_checkpoint(checkpoint)
        if self.trainer.is_global_zero:
            self.save_map(Path(self.trainer.default_root_dir) / MAP_FILENAME)

    def save_map(self, path: Path) -> None:
        network, hp = self.network, self.hparams
        state = {k: v.detach().cpu() for k, v in network.head.map.state_dict().items()}
        torch.save(
            {
                "class_path": "sipe.model.consensus_gen.CGMMap",
                "init_args": network.map_init_args,
                "state_dict": state,
                "encoder": network.encoder_name,
                "pooling": network.pooling,
                "pairing_unit": "tile",
                **network.head.fit_metadata(),
                "kl_weight": float(hp.kl_weight),
                "residual_penalty": float(hp.residual_penalty),
                "global_step": int(self.global_step),
            },
            path,
        )

    def _step(self, batch: dict[str, Any], stage: str) -> torch.Tensor:
        images = batch[self.image_key]
        domains = batch[self.domain_key].long().view(-1)
        location = batch[self.location_key].long().view(-1)
        head = self.network.head
        fmap = head.map
        hp = self.hparams
        step = int(self.global_step)
        dim = head.dim

        backbone_gap = self._gap(images)
        with torch.autocast(backbone_gap.device.type, enabled=False):
            u = fmap.standardize(backbone_gap)
            m = fmap.encode(u)
            z = head.sample(m) if stage == "train" else m
            g = fmap.generate(z)
            source, target = tile_view_pairs(location)
            # g(z_s); g(z_s) + D_k z_s for ContentShiftHead (delta_k is in nll).
            mean = head.likelihood_mean(g[source], z[source], domains[target])
            nll = head.nll(u[target], mean, domains[target]).mean() / dim
            kl = head.kl(m).mean() / dim
            beta = hp.kl_weight * ramp(step, hp.kl_warmup_steps)
            penalty = head.residual_penalty()
            loss = nll + beta * kl + hp.residual_penalty * penalty

            consensus = fmap.center + fmap.scale * fmap.generate(m.detach())

        partner = partner_index(location)
        linear = fmap.center + fmap.scale * (fmap.b + (u @ fmap.M.T) @ fmap.L.T)
        metrics = {
            "nll": nll,
            "kl": kl,
            "kl_beta": beta,
            "residual_penalty": penalty,
            # Share of f(x) that is not the linear part L M u.
            "nonlinear_rel": (consensus - linear).norm(dim=1).mean()
            / (consensus - fmap.center).norm(dim=1).mean().clamp_min(1e-12),
            "pair_cos_cgm": F.cosine_similarity(consensus, consensus[partner]).mean(),
            "pair_cos_backbone": F.cosine_similarity(
                backbone_gap, backbone_gap[partner]
            ).mean(),
            "posterior_sigma_mean": head.posterior_factor.log_diag.exp().mean(),
            "noise_sigma_mean": head.noise_factor.log_diag.exp().mean(),
            "delta_norm": head.delta.norm(dim=1).mean(),
        }
        if isinstance(head, ContentShiftHead):
            # Size of D_k z relative to the residual after the constant offsets.
            with torch.no_grad():
                shift = (mean - g[source]).norm(dim=1).mean()
                resid = u[target] - g[source] - head.delta[domains[target]]
                metrics["shift_rel"] = shift / resid.norm(dim=1).mean().clamp_min(1e-12)
        on_step = stage == "train"
        batch_size = images.shape[0]
        self.log_dict(
            {f"{stage}/{k}": v.detach() if torch.is_tensor(v) else v
             for k, v in metrics.items()},
            on_step=on_step,
            on_epoch=True,
            batch_size=batch_size,
        )  # fmt: skip
        self.log(
            f"{stage}/loss",
            loss,
            on_step=on_step,
            on_epoch=True,
            prog_bar=True,
            batch_size=batch_size,
        )

        if stage == "val":
            self._val_outputs.append(
                {
                    "cgm": consensus.detach(),
                    "backbone": backbone_gap,
                    "domains": domains,
                    "pair_ids": list(batch["pair_id"]),
                }
            )
        return loss
