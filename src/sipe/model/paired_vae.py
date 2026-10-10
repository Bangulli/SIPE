"""Mathieu-style VAE in backbone feature space, trained with SCORPION pairs.

Mathieu et al. (NeurIPS 2016) split an encoder's output into a specified part s
(here: the scanner) and an unspecified VAE latent z (here: the tissue), and decode
Dec(s, z). Unlike ScannerVAE, s is *encoded from the image* (MLP on GAP of the frozen
backbone tokens), not looked up from the scanner label, so no label is needed at
inference. The training module (PairedVAEModule) supplies Mathieu's same-class swap
and, instead of the GAN, a paired cross-scanner translation. Benchmarked
representation: GAP(mu).
"""

from __future__ import annotations

import torch
import torch.nn as nn

from sipe.model.arch import CATS, FiLMReentangler, ImageDecoder, SpecifiedMLP
from sipe.model.encoders import build_encoder
from sipe.model.scanner_vae import GaussianTokenPosterior


class PairedVAE(nn.Module):
    """s = MLP(GAP(x)) with LayerNorm; z ~ q(z|x) per token; FiLM decoder.

    At init GAP(mu) == backbone GAP and Dec(s, z) == z (zero-initialized residuals
    and FiLM). The optional pixel decoder only visualizes decoded features (the
    training module feeds it detached features).
    """

    needs_domains = False

    def __init__(
        self,
        encoder: str = "h0-mini",
        specified_dim: int = 64,
        specified_hidden_dim: int = 256,
        unspecified_hidden_dim: int = 1024,
        reentangler_hidden_dim: int = 1024,
        init_logvar: float = -4.0,
        pixel_decoder: bool = True,
        pretrained: bool = True,
        output_size: int = 224,
    ) -> None:
        super().__init__()
        bundle = build_encoder(encoder, pretrained=pretrained)
        self.backbone = bundle.backbone
        self.encoder_meta = bundle.meta
        self.eval_transform = bundle.transform

        self.feature_dim = self.encoder_meta.embed_dim
        self.specified_dim = int(specified_dim)
        self.unspecified_dim = self.feature_dim

        self.specified = SpecifiedMLP(
            self.feature_dim, specified_hidden_dim, self.specified_dim
        )
        self.posterior = GaussianTokenPosterior(
            self.feature_dim, unspecified_hidden_dim, init_logvar
        )
        self.reentangler = FiLMReentangler(
            specified_dim=self.specified_dim,
            dim=self.feature_dim,
            hidden_dim=reentangler_hidden_dim,
        )
        self.decoder = (
            ImageDecoder(in_features=self.feature_dim, output_size=output_size)
            if pixel_decoder
            else None
        )

    # Same token extraction as CATS (forward_features, prefix tokens dropped).
    backbone_feature_map = CATS.backbone_feature_map

    def freeze_backbone(self) -> None:
        self.backbone.requires_grad_(False)

    def encode_features(
        self, feature_map: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """(s [B, S], mu [B, C, H, W], logvar [B, C, H, W])."""
        mu, logvar = self.posterior(feature_map)
        return self.specified(feature_map), mu, logvar

    def encode_unspecified(self, images: torch.Tensor) -> torch.Tensor:
        """Posterior mean mu: the benchmarked z."""
        return self.posterior(self.backbone_feature_map(images))[0]

    # Bench "z_pre_gap" diagnostic: z has no output activation, so same as z.
    unspecified_preactivation = encode_unspecified

    def encode(self, images: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        s, mu, _ = self.encode_features(self.backbone_feature_map(images))
        return s, mu

    def decode_features(self, s: torch.Tensor, z: torch.Tensor) -> torch.Tensor:
        return self.reentangler(s, z)

    def decode(self, s: torch.Tensor, z: torch.Tensor) -> torch.Tensor:
        if self.decoder is None:
            raise RuntimeError("PairedVAE was built with pixel_decoder=False.")
        return self.decoder(self.decode_features(s, z))

    def forward(self, images: torch.Tensor) -> dict[str, torch.Tensor]:
        feature_map = self.backbone_feature_map(images)
        s, mu, logvar = self.encode_features(feature_map)
        features = self.decode_features(s, mu)
        out = {
            "s": s,
            "z": mu,
            "logvar": logvar,
            "feature_map": feature_map,
            "reconstructed_features": features,
        }
        if self.decoder is not None:
            out["reconstruction"] = self.decoder(features)
        return out
