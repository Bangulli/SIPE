from __future__ import annotations

import math

import timm
import torch
import torch.nn as nn


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
        backbone_name: str = "hf-hub:bioptimus/H0-mini",
        specified_dim: int = 64,
        unspecified_dim: int = 704,
        pretrained: bool = True,
        output_size: int = 224,
    ) -> None:
        super().__init__()

        self.specified_dim = int(specified_dim)
        self.unspecified_dim = int(unspecified_dim)

        self.backbone = timm.create_model(
            backbone_name,
            pretrained=pretrained,
            mlp_layer=timm.layers.SwiGLUPacked,
            act_layer=nn.SiLU,
        )

        self.feature_dim = int(self.backbone.num_features)

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
        tokens = self.backbone(images)

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
