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

# def get_decoder(base_model, channels, num_features):
#     with open('token.txt', 'r') as f:
#         login(f.read())
#     model = timm.create_model(
#         base_model,
#         pretrained=True,
#         mlp_layer=timm.layers.SwiGLUPacked,
#         act_layer=torch.nn.SiLU,
#     )
#     mae_config = ViTMAEConfig(
#         hidden_size=num_features,           # must match your ViT
#         num_attention_heads=12,
#         intermediate_size=3072,
#         decoder_hidden_size=512,
#         decoder_num_attention_heads=16,
#         decoder_num_hidden_layers=8,
#         decoder_intermediate_size=2048,
#         patch_size=14,
#         image_size=224,
#         num_channels=channels,
#     )

#     # 3. Extract just the MAE decoder from a MAE model
#     mae = ViTMAEForPreTraining(mae_config)
#     decoder = mae.decoder  # ViTMAEDecoder — randomly initialized
#     return decoder

def get_decoder(base_model, channels, num_features):
    # return nn.Sequential(
    #         nn.ConvTranspose2d(num_features, 512, kernel_size=7, stride=1, padding=0),
    #         nn.BatchNorm2d(512), nn.GELU(),

    #         nn.ConvTranspose2d(512, 256, kernel_size=4, stride=2, padding=1),
    #         nn.BatchNorm2d(256), nn.GELU(),

    #         nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2, padding=1),
    #         nn.BatchNorm2d(128), nn.GELU(),

    #         nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1),
    #         nn.BatchNorm2d(64), nn.GELU(),

    #         nn.ConvTranspose2d(64, 32, kernel_size=4, stride=2, padding=1),
    #         nn.BatchNorm2d(32), nn.GELU(),

    #         nn.ConvTranspose2d(32, channels, kernel_size=4, stride=2, padding=1),
    #         nn.Sigmoid()
    #     )
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


if __name__ == '__main__':
    get_decoder("hf-hub:bioptimus/H-optimus-0", 3)