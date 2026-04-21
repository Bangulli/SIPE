from huggingface_hub import login
import torch
import timm
from timm.data import resolve_data_config
from timm.data.transforms_factory import create_transform
from torchvision import transforms



def get_encoder_and_transforms(base_model):
    with open('token.txt', 'r') as f:
        login(f.read())
    model = timm.create_model(
        base_model,
        pretrained=True,
        mlp_layer=timm.layers.SwiGLUPacked,
        act_layer=torch.nn.SiLU,
    )

    transform = create_transform(**resolve_data_config(model.pretrained_cfg, model=model))
    #print(transform)
    return model, transform

# input = torch.rand(3, 224, 224)
# input = transforms.ToPILImage()(input)

# # We recommend using mixed precision for faster inference.
# with torch.autocast(device_type="cuda", dtype=torch.float16):
#     with torch.inference_mode():
#         output = model(transform(input).unsqueeze(0).to("cuda"))  # (1, 261, 768)
#         # CLS token features (1, 768):
#         cls_features = output[:, 0]
#         # Patch token features (1, 256, 768):
#         patch_token_features = output[:, model.num_prefix_tokens :]
#         # Concatenate the CLS token features with the mean of the patch token
#         # features (1, 1536):
#         concatenated_features = torch.cat(
#             [cls_features, patch_token_features.mean(1)], dim=-1
#         )

# assert cls_features.shape == (1, 768)
# assert patch_token_features.shape == (1, 256, 768)
# assert concatenated_features.shape == (1, 1536)
