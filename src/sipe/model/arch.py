from __future__ import annotations

import math

import torch
import torch.nn as nn

from sipe.model.encoders import build_encoder


class SpecifiedProjector(nn.Module):
    """Project a spatial feature map to the specified representation s."""

    def __init__(self, in_features: int, out_features: int) -> None:
        super().__init__()
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.proj = nn.Linear(in_features, out_features)
        self.act = nn.ReLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.pool(x).flatten(1)
        return self.act(self.proj(x))


class UnspecifiedProjector(nn.Module):
    """Project a spatial feature map to the unspecified representation z."""

    def __init__(self, in_features: int, out_features: int) -> None:
        super().__init__()
        self.proj = nn.Conv2d(in_features, out_features, kernel_size=1)
        self.act = nn.ReLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.act(self.proj(x))


class Disentangler(nn.Module):
    def __init__(
        self,
        in_features: int,
        specified_dim: int,
        unspecified_dim: int,
    ) -> None:
        super().__init__()
        self.to_specified = SpecifiedProjector(in_features, specified_dim)
        self.to_unspecified = UnspecifiedProjector(in_features, unspecified_dim)

    def forward(
        self,
        feature_map: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        s = self.to_specified(feature_map)
        z = self.to_unspecified(feature_map)
        return s, z


class Reentangler(nn.Module):
    def __init__(
        self,
        specified_dim: int,
        unspecified_dim: int,
        out_features: int,
    ) -> None:
        super().__init__()
        self.proj = nn.Conv2d(
            specified_dim + unspecified_dim,
            out_features,
            kernel_size=1,
        )

    def forward(
        self,
        s: torch.Tensor,
        z: torch.Tensor,
    ) -> torch.Tensor:
        if s.ndim != 2:
            raise ValueError(f"Expected s with shape [B, D], got {tuple(s.shape)}.")
        if z.ndim != 4:
            raise ValueError(
                f"Expected z with shape [B, C, H, W], got {tuple(z.shape)}."
            )

        _, _, height, width = z.shape
        s_map = s[:, :, None, None].expand(-1, -1, height, width)
        return self.proj(torch.cat([s_map, z], dim=1))


class ImageDecoder(nn.Module):
    """Decode a H0-mini feature map to an RGB image."""

    def __init__(
        self,
        in_features: int,
        out_channels: int = 3,
        output_size: int = 224,
    ) -> None:
        super().__init__()
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(in_features, 256, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(256),
            nn.GELU(),
            nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.GELU(),
            nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.GELU(),
            nn.ConvTranspose2d(64, out_channels, kernel_size=4, stride=2, padding=1),
            nn.Upsample(
                size=(output_size, output_size),
                mode="bilinear",
                align_corners=False,
            ),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.decoder(x)


class CATS(nn.Module):
    """Disentanglement/reconstruction architecture.

    Training-only components such as domain classifiers, gradient reversal,
    objectives, schedules, and optimization deliberately live outside this
    module.
    """

    def __init__(
        self,
        encoder: str = "h0-mini",
        specified_dim: int = 64,
        unspecified_dim: int = 704,
        pretrained: bool = True,
        output_size: int = 224,
    ) -> None:
        super().__init__()

        self.specified_dim = int(specified_dim)
        self.unspecified_dim = int(unspecified_dim)

        bundle = build_encoder(encoder, pretrained=pretrained)
        self.backbone = bundle.backbone
        self.encoder_meta = bundle.meta
        self.eval_transform = bundle.transform

        self.feature_dim = self.encoder_meta.embed_dim

        self.disentangler = Disentangler(
            in_features=self.feature_dim,
            specified_dim=self.specified_dim,
            unspecified_dim=self.unspecified_dim,
        )

        self.reentangler = Reentangler(
            specified_dim=self.specified_dim,
            unspecified_dim=self.unspecified_dim,
            out_features=self.feature_dim,
        )

        self.decoder = ImageDecoder(
            in_features=self.feature_dim,
            output_size=output_size,
        )

    def backbone_feature_map(self, images: torch.Tensor) -> torch.Tensor:
        """Return backbone patch tokens reshaped as [B, C, H, W]."""
        # forward_features, not forward: forward applies the HF config's global_pool
        # (H0-mini: none -> tokens; H-optimus-1: "token" -> pooled CLS [B, C]).
        tokens = self.backbone.forward_features(images)

        if tokens.ndim != 3:
            raise RuntimeError(
                "Expected the backbone to return token features with shape "
                f"[B, N, C], got {tuple(tokens.shape)}."
            )

        num_prefix_tokens = self.backbone.num_prefix_tokens
        patch_tokens = tokens[:, num_prefix_tokens:]

        num_patches = patch_tokens.shape[1]
        grid_size = math.isqrt(num_patches)

        if grid_size * grid_size != num_patches:
            raise RuntimeError(
                f"Expected a square patch-token grid, got {num_patches} tokens."
            )

        return patch_tokens.transpose(1, 2).reshape(
            patch_tokens.shape[0],
            patch_tokens.shape[2],
            grid_size,
            grid_size,
        )

    def encode(
        self,
        images: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        feature_map = self.backbone_feature_map(images)
        return self.disentangler(feature_map)

    def unspecified_preactivation(self, images: torch.Tensor) -> torch.Tensor:
        """z before the projector's ReLU (diagnostic representation)."""
        feature_map = self.backbone_feature_map(images)
        return self.disentangler.to_unspecified.proj(feature_map)

    def decode(
        self,
        s: torch.Tensor,
        z: torch.Tensor,
    ) -> torch.Tensor:
        feature_map = self.reentangler(s, z)
        return self.decoder(feature_map)

    def forward(self, images: torch.Tensor) -> dict[str, torch.Tensor]:
        s, z = self.encode(images)
        reconstruction = self.decode(s, z)
        return {
            "s": s,
            "z": z,
            "reconstruction": reconstruction,
        }

    def freeze_backbone(self) -> None:
        self.backbone.requires_grad_(False)

    def unfreeze_backbone(self) -> None:
        self.backbone.requires_grad_(True)


class ChannelLayerNorm(nn.Module):
    """LayerNorm over C at every position of a [B, C, H, W] map (per-token LN)."""

    def __init__(self, dim: int) -> None:
        super().__init__()
        self.norm = nn.LayerNorm(dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.norm(x.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)


class ResidualTokenMLP(nn.Module):
    """Per-token x + W2·GELU(W1·LN(x)); zero-init W2, so it starts as identity."""

    def __init__(self, dim: int, hidden_dim: int) -> None:
        super().__init__()
        self.norm = ChannelLayerNorm(dim)
        self.fc1 = nn.Conv2d(dim, hidden_dim, kernel_size=1)
        self.act = nn.GELU()
        self.fc2 = nn.Conv2d(hidden_dim, dim, kernel_size=1)
        nn.init.zeros_(self.fc2.weight)
        nn.init.zeros_(self.fc2.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.fc2(self.act(self.fc1(self.norm(x))))


class SpecifiedMLP(nn.Module):
    """GAP -> MLP -> LayerNorm. No output ReLU, so s is not forced non-negative."""

    def __init__(self, in_features: int, hidden_dim: int, out_features: int) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_features, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, out_features),
            nn.LayerNorm(out_features),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x.mean(dim=(2, 3)))


class ResidualDisentangler(nn.Module):
    """s = MLP(GAP(x)); z = x + g(x) with g zero-initialized (z == x at init)."""

    def __init__(
        self,
        in_features: int,
        specified_dim: int,
        specified_hidden_dim: int,
        unspecified_hidden_dim: int,
    ) -> None:
        super().__init__()
        self.to_specified = SpecifiedMLP(
            in_features, specified_hidden_dim, specified_dim
        )
        self.to_unspecified = ResidualTokenMLP(in_features, unspecified_hidden_dim)

    def forward(
        self,
        feature_map: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        return self.to_specified(feature_map), self.to_unspecified(feature_map)


class FiLMReentangler(nn.Module):
    """f = z * (1 + gamma(s)) + beta(s), then a residual token MLP.

    gamma/beta and the MLP output are zero-initialized, so f == z at init.
    Scanner acts as a per-channel gain and offset instead of only an offset
    (legacy concat + 1x1 conv).
    """

    def __init__(self, specified_dim: int, dim: int, hidden_dim: int) -> None:
        super().__init__()
        self.film = nn.Linear(specified_dim, 2 * dim)
        nn.init.zeros_(self.film.weight)
        nn.init.zeros_(self.film.bias)
        self.mlp = ResidualTokenMLP(dim, hidden_dim)

    def forward(self, s: torch.Tensor, z: torch.Tensor) -> torch.Tensor:
        gamma, beta = self.film(s)[:, :, None, None].chunk(2, dim=1)
        return self.mlp(z * (1 + gamma) + beta)


class CATSv2(CATS):
    """CATS variant that starts equal to the backbone and decodes to feature space.

    - z = x + g(x) (identity at init, no ReLU, dim = backbone width), so
      GAP(z) == backbone GAP before training;
    - s = MLP(GAP(x)) with LayerNorm output;
    - decode_features(s, z): FiLM(z; s) + residual token MLP, compared with the
      frozen backbone tokens by the training module;
    - decode(s, z): pixel decoder on decode_features, for qualitative images (the
      training module may train it on detached features only).

    Same public API as CATS (encode / decode / forward / backbone_feature_map).
    Subclasses CATS only so `CATSModule(network: CATS)` / LightningCLI accept it via
    class_path; it builds its own modules and does not call CATS.__init__.
    """

    def __init__(
        self,
        encoder: str = "h0-mini",
        specified_dim: int = 64,
        specified_hidden_dim: int = 256,
        unspecified_hidden_dim: int = 1024,
        reentangler_hidden_dim: int = 1024,
        pretrained: bool = True,
        output_size: int = 224,
    ) -> None:
        nn.Module.__init__(self)
        bundle = build_encoder(encoder, pretrained=pretrained)
        self.backbone = bundle.backbone
        self.encoder_meta = bundle.meta
        self.eval_transform = bundle.transform

        self.feature_dim = self.encoder_meta.embed_dim
        self.specified_dim = int(specified_dim)
        self.unspecified_dim = self.feature_dim

        self.disentangler = ResidualDisentangler(
            in_features=self.feature_dim,
            specified_dim=self.specified_dim,
            specified_hidden_dim=specified_hidden_dim,
            unspecified_hidden_dim=unspecified_hidden_dim,
        )
        self.reentangler = FiLMReentangler(
            specified_dim=self.specified_dim,
            dim=self.feature_dim,
            hidden_dim=reentangler_hidden_dim,
        )
        self.decoder = ImageDecoder(
            in_features=self.feature_dim,
            output_size=output_size,
        )

    def encode(
        self,
        images: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        return self.disentangle(self.backbone_feature_map(images))

    def disentangle(
        self,
        feature_map: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        return self.disentangler(feature_map)

    def unspecified_preactivation(self, images: torch.Tensor) -> torch.Tensor:
        # No output activation on z.
        return self.encode(images)[1]

    def decode_features(self, s: torch.Tensor, z: torch.Tensor) -> torch.Tensor:
        return self.reentangler(s, z)

    def decode(self, s: torch.Tensor, z: torch.Tensor) -> torch.Tensor:
        return self.decoder(self.decode_features(s, z))

    def forward(self, images: torch.Tensor) -> dict[str, torch.Tensor]:
        feature_map = self.backbone_feature_map(images)
        s, z = self.disentangle(feature_map)
        features = self.decode_features(s, z)
        return {
            "s": s,
            "z": z,
            "feature_map": feature_map,
            "reconstructed_features": features,
            "reconstruction": self.decoder(features),
        }
