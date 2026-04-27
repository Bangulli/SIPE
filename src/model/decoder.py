######## Ecosystem ########
import os, sys, pathlib as pl, pprint
sys.path.append(os.path.join(os.path.dirname(__file__), "."))
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
######## External ########
import torch, timm
import torch.nn as nn
from transformers import ViTModel, ViTMAEForPreTraining, ViTMAEConfig
from huggingface_hub import login
from timm.data import resolve_data_config
######## Internal ########
##########################

def get_decoder_simple(base_model, channels, num_features):
    return nn.Sequential(
        # 16x16 -> 28x28
        nn.ConvTranspose2d(num_features, 256, kernel_size=4, stride=2, padding=1),
        nn.BatchNorm2d(256), nn.GELU(),
        # 28x28 -> 56x56
        nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2, padding=1),
        nn.BatchNorm2d(128), nn.GELU(),
        # 56x56 -> 112x112
        nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1),
        nn.BatchNorm2d(64), nn.GELU(),
        # 112x112 -> 256x256
        nn.ConvTranspose2d(64, channels, kernel_size=4, stride=2, padding=1),
        # 256x256 -> 224x224
        nn.Upsample(size=(224, 224), mode='bilinear', align_corners=False),
        
    )

def get_proj_decoder_simple(base_model, channels, num_features):
    return nn.Sequential(
        nn.Conv2d(num_features, num_features, kernel_size=1),
        # 16x16 -> 28x28
        nn.ConvTranspose2d(num_features, 256, kernel_size=4, stride=2, padding=1),
        nn.BatchNorm2d(256), nn.GELU(),
        # 28x28 -> 56x56
        nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2, padding=1),
        nn.BatchNorm2d(128), nn.GELU(),
        # 56x56 -> 112x112
        nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1),
        nn.BatchNorm2d(64), nn.GELU(),
        # 112x112 -> 256x256
        nn.ConvTranspose2d(64, channels, kernel_size=4, stride=2, padding=1),
        # 256x256 -> 224x224
        nn.Upsample(size=(224, 224), mode='bilinear', align_corners=False),
        
    )
    
def get_proj_decoder_simple_v2(channels, num_features=768, num_stain_features=105):
    return nn.Sequential(
        ## mixing stage ## 
        nn.Conv2d(num_features+num_stain_features, num_features, kernel_size=1), ## mix the staining attachment into the data - patch wise
        nn.BatchNorm2d(num_features),
        nn.GELU(),
        nn.Conv2d(num_features, num_features, kernel_size=3, padding_mode='reflect', padding=1), ## neighbor aware mixing
        nn.BatchNorm2d(num_features),
        nn.GELU(),
        nn.Conv2d(num_features, num_features, kernel_size=1), ## patch wise cleaning
        nn.BatchNorm2d(num_features),
        nn.GELU(),
        
        ## generation stage ##
        # 16x16 -> 28x28
        nn.ConvTranspose2d(num_features, 256, kernel_size=4, stride=2, padding=1),
        nn.BatchNorm2d(256), nn.GELU(),
        # 28x28 -> 56x56
        nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2, padding=1),
        nn.BatchNorm2d(128), nn.GELU(),
        # 56x56 -> 112x112
        nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1),
        nn.BatchNorm2d(64), nn.GELU(),
        # 112x112 -> 256x256
        nn.ConvTranspose2d(64, channels, kernel_size=4, stride=2, padding=1),
        
        ## sampling stage ##
        # 256x256 -> 224x224
        nn.Upsample(size=(224, 224), mode='bilinear', align_corners=False),
    )
    
    
## below this is claude generated, maybe ill use it later
if __name__ == '__main__':
    import torch
    import torch.nn as nn
    import torch.nn.functional as F


    # ── building blocks ───────────────────────────────────────────────────────────

    class SEBlock(nn.Module):
        """Squeeze-and-Excitation channel attention."""
        def __init__(self, channels: int, reduction: int = 8):
            super().__init__()
            self.fc = nn.Sequential(
                nn.AdaptiveAvgPool2d(1),
                nn.Flatten(),
                nn.Linear(channels, channels // reduction, bias=False),
                nn.GELU(),
                nn.Linear(channels // reduction, channels, bias=False),
                nn.Sigmoid(),
            )

        def forward(self, x):
            return x * self.fc(x).view(x.size(0), x.size(1), 1, 1)


    class ResBlock(nn.Module):
        """Pre-activation residual block with optional SE attention."""
        def __init__(self, channels: int, use_se: bool = True):
            super().__init__()
            self.net = nn.Sequential(
                nn.BatchNorm2d(channels), nn.GELU(),
                nn.Conv2d(channels, channels, 3, padding=1, bias=False),
                nn.BatchNorm2d(channels), nn.GELU(),
                nn.Conv2d(channels, channels, 3, padding=1, bias=False),
            )
            self.se = SEBlock(channels) if use_se else nn.Identity()

        def forward(self, x):
            return x + self.se(self.net(x))


    class SelfAttention2d(nn.Module):
        """Lightweight spatial self-attention (single head, key-dim reduction)."""
        def __init__(self, channels: int, key_dim: int = 64):
            super().__init__()
            self.key_dim = key_dim
            self.q = nn.Conv2d(channels, key_dim, 1, bias=False)
            self.k = nn.Conv2d(channels, key_dim, 1, bias=False)
            self.v = nn.Conv2d(channels, channels, 1, bias=False)
            self.proj = nn.Conv2d(channels, channels, 1, bias=False)
            self.norm = nn.GroupNorm(1, channels)   # LN equivalent for 2-D

        def forward(self, x):
            B, C, H, W = x.shape
            N = H * W
            q = self.q(x).view(B, self.key_dim, N).permute(0, 2, 1)   # B N Dk
            k = self.k(x).view(B, self.key_dim, N)                     # B Dk N
            v = self.v(x).view(B, C, N).permute(0, 2, 1)               # B N C
            attn = torch.softmax(q @ k / self.key_dim ** 0.5, dim=-1)  # B N N
            out = (attn @ v).permute(0, 2, 1).view(B, C, H, W)
            return x + self.proj(self.norm(out))


    class UpsampleBlock(nn.Module):
        """Bilinear upsample → Conv (no checkerboard)."""
        def __init__(self, in_ch: int, out_ch: int, scale: int = 2):
            super().__init__()
            self.up = nn.Upsample(scale_factor=scale, mode='bilinear', align_corners=False)
            self.conv = nn.Sequential(
                nn.Conv2d(in_ch, out_ch, 3, padding=1, bias=False),
                nn.BatchNorm2d(out_ch),
                nn.GELU(),
            )

        def forward(self, x):
            return self.conv(self.up(x))


    # ── generator ─────────────────────────────────────────────────────────────────

    class ProjectionDecoder(nn.Module):
        """
        Upgraded decoder: 16×16 latent → 224×224 RGB image.

        Stage sizes:  16 → 32 → 64 → 128 → 256 → 224
        """
        def __init__(
            self,
            num_features: int,
            channels: int = 3,
            base_ch: int = 256,
        ):
            super().__init__()

            # ── bottleneck (stays at 16×16) ───────────────────────────────────────
            self.bottleneck = nn.Sequential(
                nn.Conv2d(num_features, base_ch, 1, bias=False),
                ResBlock(base_ch),
                SelfAttention2d(base_ch, key_dim=64),   # global context at low res
                ResBlock(base_ch),
            )

            # ── progressive upsampling with residual refinement ───────────────────
            # 16 → 32
            self.up1 = UpsampleBlock(base_ch,      base_ch // 2)     # 256 → 128
            self.r1  = ResBlock(base_ch // 2)

            # 32 → 64
            self.up2 = UpsampleBlock(base_ch // 2, base_ch // 4)     # 128 → 64
            self.r2  = ResBlock(base_ch // 4)

            # 64 → 128
            self.up3 = UpsampleBlock(base_ch // 4, base_ch // 8)     # 64 → 32
            self.r3  = ResBlock(base_ch // 8)

            # 128 → 256
            self.up4 = UpsampleBlock(base_ch // 8, base_ch // 16)    # 32 → 16
            self.r4  = ResBlock(base_ch // 16)

            # ── final head: 256 → 224, output RGB in [0, 1] ───────────────────────
            self.head = nn.Sequential(
                nn.Upsample(size=(224, 224), mode='bilinear', align_corners=False),
                nn.Conv2d(base_ch // 16, base_ch // 16, 3, padding=1, bias=False),
                nn.GELU(),
                nn.Conv2d(base_ch // 16, channels, 1),
                nn.Sigmoid(),    # swap for Tanh + rescale if your pipeline expects [-1,1]
            )

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            x = self.bottleneck(x)
            x = self.r1(self.up1(x))
            x = self.r2(self.up2(x))
            x = self.r3(self.up3(x))
            x = self.r4(self.up4(x))
            return self.head(x)


    # ── convenience constructor matching your original API ────────────────────────

    def get_proj_decoder(base_model, channels: int, num_features: int) -> nn.Module:
        return ProjectionDecoder(num_features=num_features, channels=channels)