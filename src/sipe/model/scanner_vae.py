"""Scanner-conditioned VAE in backbone feature space (not in legacy CATS).

Fader-network style (Lample et al., 2017) with a VAE latent (Mathieu et al., 2016):
the frozen backbone's patch tokens x are encoded per token into q(z|x); a decoder
conditioned on a learned scanner embedding reconstructs x from z. The training module
adds the KL and a GRL scanner adversary on GAP(mu). The benchmarked representation
is GAP(mu).
"""

from __future__ import annotations

import torch
import torch.nn as nn

from sipe.model.arch import (
    CATS,
    ChannelLayerNorm,
    FiLMReentangler,
    ImageDecoder,
    ResidualTokenMLP,
)
from sipe.model.encoders import build_encoder


class GaussianTokenPosterior(nn.Module):
    """Per-token q(z|x) = N(mu, diag(exp(logvar))), mu = x + g(x) (identity at init).

    logvar = h(LN(x)), with h zero-initialized to the constant `init_logvar`, so the
    posterior starts near-deterministic and GAP(mu) == backbone GAP at step 0.
    """

    def __init__(self, dim: int, hidden_dim: int, init_logvar: float) -> None:
        super().__init__()
        self.mu = ResidualTokenMLP(dim, hidden_dim)
        self.logvar = nn.Sequential(
            ChannelLayerNorm(dim), nn.Conv2d(dim, dim, kernel_size=1)
        )
        nn.init.zeros_(self.logvar[1].weight)
        nn.init.constant_(self.logvar[1].bias, init_logvar)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        return self.mu(x), self.logvar(x)


class ScannerVAE(nn.Module):
    """s = scanner embedding (needs labels); z ~ q(z|x) per token; FiLM decoder.

    The optional pixel decoder only visualizes decoded features (the training module
    feeds it detached features).
    """

    # s comes from the scanner label: callers pass `domains` wherever s is computed.
    needs_domains = True

    def __init__(
        self,
        encoder: str = "h0-mini",
        num_domains: int = 5,
        specified_dim: int = 64,
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

        self.posterior = GaussianTokenPosterior(
            self.feature_dim, unspecified_hidden_dim, init_logvar
        )
        self.scanner_embedding = nn.Embedding(num_domains, self.specified_dim)
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
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """(mu, logvar), each [B, C, H, W]."""
        return self.posterior(feature_map)

    def encode_unspecified(self, images: torch.Tensor) -> torch.Tensor:
        """Posterior mean mu (no labels needed): the benchmarked z."""
        return self.posterior(self.backbone_feature_map(images))[0]

    # Bench "z_pre_gap" diagnostic: z has no output activation, so same as z.
    unspecified_preactivation = encode_unspecified

    def encode(
        self, images: torch.Tensor, domains: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        return self.scanner_embedding(domains), self.encode_unspecified(images)

    def decode_features(self, s: torch.Tensor, z: torch.Tensor) -> torch.Tensor:
        return self.reentangler(s, z)

    def decode(self, s: torch.Tensor, z: torch.Tensor) -> torch.Tensor:
        if self.decoder is None:
            raise RuntimeError("ScannerVAE was built with pixel_decoder=False.")
        return self.decoder(self.decode_features(s, z))

    def forward(
        self, images: torch.Tensor, domains: torch.Tensor
    ) -> dict[str, torch.Tensor]:
        feature_map = self.backbone_feature_map(images)
        s = self.scanner_embedding(domains)
        mu, logvar = self.posterior(feature_map)
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
