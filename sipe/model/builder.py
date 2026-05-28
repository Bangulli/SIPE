from __future__ import annotations

from pathlib import Path
from typing import Literal

import numpy as np
import torch
from torchvision import transforms

from .arch import H0_mini_for_Adversarial

CLASSES = {
    "Hematoxylin and eosin stain method": 460935,
    "HE - Hematoxylin and eosin stain method (procedure)": 268034,
    "Chromogranin A+Hematoxylin stain (substance)": 1181,
    "hematoxylin stain+water soluble eosin stain": 52429,
    "Verhoeff-Van Gieson stain method": 87845,
    "ROS1+Hematoxylin stain (substance)": 19303,
    "Fuchsin acid stain+Orange G stain": 7057,
    "Masson trichrome stain method (procedure)": 39441,
    "hematoxylin stain+C1q": 5288,
    "Thyroid transcripton factor+Hematoxylin stain (substance)": 9170,
    "Herovici's stain method": 6613,
    "PD-L1 (SP142)+Hematoxylin stain (substance)": 16737,
    "ALK+Hematoxylin stain (substance)": 19964,
    "Methenamine silver nitrate stain": 9780,
    "TTF1(8G7G3)+Hematoxylin stain (substance)": 10525,
    "hematoxylin stain+IgM": 5118,
    "Periodic acid Schiff stain": 30978,
    "PAX-8+Hematoxylin stain (substance)": 796,
    "Cytokeratin 20+Hematoxylin stain (substance)": 2097,
    "PD-L1 (SP263)+Hematoxylin stain (substance)": 3547,
    "Estrogen receptor RTU+hematoxylin stain": 6305,
    "hematoxylin stain+IgA": 5146,
    "Periodic acid Schiff stain method": 2935,
    "Napsin+Hematoxylin stain (substance)": 692,
    "Van Gieson stain": 6866,
    "hematoxylin stain+C3": 5053,
    "Progesterone receptor RTU+hematoxylin stain": 4776,
    "hematoxylin stain+CB, TP, Cytokeratin AE1/AE3": 212,
    "Calretinin+Hematoxylin stain (substance)": 640,
    "C4D+Hematoxylin stain (substance)": 452,
    "hematoxylin stain+IgG": 5151,
    "NUT+Hematoxylin stain (substance)": 213,
    "Giemsa stain method": 138,
    "Calponin+hematoxylin stain": 3029,
    "Gomori stain method (procedure)": 167,
    "P63+hematoxylin stain": 7268,
    "pan Cytokeratin+Hematoxylin stain (substance)": 2529,
    "hematoxylin stain+kappa": 409,
    "GATA-3+Hematoxylin stain (substance)": 891,
    "hematoxylin stain+HER-2 neu": 2970,
    "Cytokeratin 7+Hematoxylin stain (substance)": 4907,
    "CD31+Hematoxylin stain (substance)": 161,
    "P40+Hematoxylin stain (substance)": 5390,
    "CDX2+Hematoxylin stain (substance)": 2576,
    "Kappa plasma cell+hematoxylin stain": 283,
    "hematoxylin stain+Cytokeratin 5": 5675,
    "Synaptophysin+Hematoxylin stain (substance)": 1704,
    "Prostate specific membrane antigen+Hematoxylin stain (substance)": 466,
    "Prussian blue stain method (Perls)": 395,
    "Cytokeratin 5-6+Hematoxylin stain (substance)": 1147,
    "hematoxylin stain+S100": 212,
    "Periodic acid Schiff stain method (procedure)": 448,
    "CD3+Hematoxylin stain (substance)": 45,
    "GCDFP15+hematoxylin stain": 370,
    "hematoxylin stain+lambda": 324,
    "CD3+hematoxylin stain": 871,
    "MIB-1 (Ki-67)+Hematoxylin stain (substance)": 839,
    "E-cadherin+hematoxylin stain": 658,
    "Thyroglobulin+Hematoxylin stain (substance)": 326,
    "P63+Hematoxylin stain (substance)": 616,
    "Cytokeratin CAM 5.2+hematoxylin stain": 2697,
    "CD68+Hematoxylin stain (substance)": 345,
    "Beta-Catenin+Hematoxylin stain (substance)": 260,
    "hematoxylin stain+cyclin D1": 284,
    "Progesterone reseptor+Hematoxylin stain (substance)": 430,
    "PD-L1+Hematoxylin stain (substance)": 1260,
    "P53+Hematoxylin stain (substance)": 244,
    "Ki-67+hematoxylin stain": 52,
    "hematoxylin stain+Androgen receptor": 179,
    "Alcian blue with Periodic acid Schiff stain method": 344,
    "SSTR2+Hematoxylin stain (substance)": 209,
    "ERG+Hematoxylin stain (substance)": 92,
    "hematoxylin stain+GATA-3": 224,
    "P40+hematoxylin stain": 451,
    "CMV-P52+Hematoxylin stain (substance)": 289,
    "hematoxylin stain+CD20": 209,
    "SATB2+Hematoxylin stain (substance)": 447,
    "CD31+CD68+Hematoxylin stain (substance)": 45,
    "Vimentin+Hematoxylin stain (substance)": 10,
    "Epithelial antigen EP4+Hematoxylin stain (substance)": 381,
    "INSM1+Hematoxylin stain (substance)": 227,
    "Lambda plasma cell+hematoxylin stain": 315,
    "hematoxylin stain+IgG4": 218,
    "Estrogen receptor alpha+Hematoxylin stain (substance)": 762,
    "Prostate specific  antigen+Hematoxylin stain (substance)": 188,
    "CD10+Hematoxylin stain (substance)": 29,
    "hematoxylin stain+C3c": 340,
    "EGFR+Hematoxylin stain (substance)": 6,
    "CD45+Hematoxylin stain (substance)": 38,
    "hematoxylin stain+CD45": 50,
    "Ziehl-Neelsen stain method (procedure)": 33,
    "Wilms tumor+Hematoxylin stain (substance)": 48,
    "AMACR+Hematoxylin stain (substance)": 22,
    "hematoxylin stain+C4d": 44,
    "Mammaglobin+Hematoxylin stain (substance)": 33,
    "Calcitonin+Hematoxylin stain (substance)": 18,
    "Congo red stain method (procedure)": 28,
    "CD20+Hematoxylin stain (substance)": 13,
    "CD21+Hematoxylin stain (substance)": 10,
    "hematoxylin stain+SV40": 25,
    "SOX10+Hematoxylin stain (substance)": 3,
    "MIB-1+CD45+Hematoxylin stain (substance)": 5,
    "S-100+Hematoxylin stain (substance)": 11,
    "Proximal nephrogen renal antigen+Hematoxylin stain (substance)": 7,
    "CD5+Hematoxylin stain (substance)": 4,
    "CD3+CD20+Hematoxylin stain (substance)": 5,
    "Cytokeratin 20+hematoxylin stain": 3,
    "EBER-CISH+Nuclear fast red stain (substance)": 6,
    "hematoxylin stain+Cytokeratin 7": 2,
    "PD1+Hematoxylin stain (substance)": 1,
}


FeatureMode = Literal["concat", "stain", "content", "featuremap"]


def build_sipe_model(
    checkpoint_dir: str | Path,
    *,
    possible_classes: dict[str, int] | None = None,
    base_model: str = "hf-hub:bioptimus/H0-mini",
    emb_stain_size: int = 64,
    device: str | torch.device = "cuda:0",
    eval_mode: bool = True,
) -> H0_mini_for_Adversarial:
    """Build and load a SIPE model from checkpoint files.

    Expected checkpoint directory structure:

        checkpoint_dir/
            backbone.pth
            image_decoder.pth
            entangler.pth
            discriminator.pth   # optional

    Parameters
    ----------
    checkpoint_dir:
        Path to the directory containing the SIPE checkpoint files.
    possible_classes:
        Staining class dictionary used to initialize the entangler classifier.
        If None, uses the module-level CLASSES dictionary.
    base_model:
        Backbone model name passed to timm.
    emb_stain_size:
        Number of stain-specific dimensions.
    device:
        Device used for inference, e.g. "cuda:0" or "cpu".
    eval_mode:
        Whether to put the model in eval mode.

    Returns
    -------
    H0_mini_for_Adversarial
        Loaded SIPE model.
    """
    checkpoint_dir = Path(checkpoint_dir)

    if possible_classes is None:
        possible_classes = CLASSES

    model = H0_mini_for_Adversarial(
        possible_classes=possible_classes,
        base_model=base_model,
        emb_stain_size=emb_stain_size,
        device=str(device),
    )

    model.load(checkpoint_dir)
    model.to(device)

    if eval_mode:
        model.eval()

    return model


class SIPEFeatureExtractor:
    """Small inference wrapper around SIPE.

    The raw SIPE model returns:

        s: stain-specific vector, shape [B, 64]
        z: content/unspecified map, shape [B, 704, 16, 16]

    For PLISM-like feature extraction, the default output is:

        concat(s, GAP(z)) -> shape [B, 768]
    """

    def __init__(
        self,
        checkpoint_dir: str | Path,
        *,
        possible_classes: dict[str, int] | None = None,
        device: str | torch.device = "cuda:0",
        mixed_precision: bool = False,
        feature_mode: FeatureMode = "concat",
    ) -> None:
        self.device = torch.device(device)
        self.mixed_precision = mixed_precision
        self.feature_mode = feature_mode

        self.model = build_sipe_model(
            checkpoint_dir=checkpoint_dir,
            possible_classes=possible_classes,
            device=self.device,
            eval_mode=True,
        )

        if feature_mode == "concat":
            self.output_dim = 768
        elif feature_mode == "stain":
            self.output_dim = 64
        elif feature_mode == "content":
            self.output_dim = 704
        elif feature_mode == "featuremap":
            self.output_dim = 704
        elif feature_mode == "featuremap_concat":
            self.output_dim = 768
        elif feature_mode == "featuremap_reen":
            self.output_dim = 768
        else:
            raise ValueError(f"Unknown feature_mode: {feature_mode}")

    @property
    def transform(self) -> transforms.Compose:
        """Transform to apply element-wise before batching.

        This matches the H0-mini normalization used in the PLISM example.
        """
        return transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=(0.707223, 0.578729, 0.703617),
                    std=(0.211883, 0.230117, 0.177517),
                ),
            ]
        )

    @torch.inference_mode()
    def __call__(self, images: torch.Tensor) -> np.ndarray:
        """Compute SIPE features.

        Parameters
        ----------
        images:
            Tensor of shape [B, 3, H, W], usually [B, 3, 224, 224].

        Returns
        -------
        np.ndarray
            Feature array of shape [B, output_dim].
        """
        images = images.to(self.device)

        autocast_enabled = self.mixed_precision and self.device.type == "cuda"

        with torch.autocast(
            device_type=self.device.type,
            enabled=autocast_enabled,
        ):
            s, z = self.model(images)

            if self.feature_mode == "stain":
                features = s

            elif self.feature_mode == "content":
                features = z.mean(dim=(2, 3))

            elif self.feature_mode == "concat":
                z_gap = z.mean(dim=(2, 3))
                features = torch.cat([s, z_gap], dim=1)
                
            elif self.feature_mode == "featuremap":
                features = z
                
            elif self.feature_mode == "featuremap_concat":
                s = s[:, :, None, None].expand(-1, -1, 16, 16)
                features = torch.cat([s, z], dim=1)

            elif self.feature_mode == "featuremap_reen":
                features = self.model.entangler.reentangle(s, z)


            else:
                raise ValueError(f"Unknown feature_mode: {self.feature_mode}")

        return features.detach().cpu().numpy()


def build_sipe_extractor(
    checkpoint_dir: str | Path,
    *,
    device: str | torch.device = "cuda:0",
    mixed_precision: bool = False,
    feature_mode: FeatureMode = "concat",
) -> SIPEFeatureExtractor:
    """Build a PLISM-like SIPE feature extractor."""
    return SIPEFeatureExtractor(
        checkpoint_dir=checkpoint_dir,
        device=device,
        mixed_precision=mixed_precision,
        feature_mode=feature_mode,
    )
