######## Ecosystem ########
import os, sys, pathlib as pl, datetime
sys.path.append(os.path.join(os.path.dirname(__file__), "."))
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
######## External ########
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.preprocessing import LabelEncoder
from torch.autograd import Function
######## Internal ########
from src.model.encoder import get_encoder_and_transforms
from torchvision.transforms import ToPILImage
from src.utils.misc import make_name_from_list
##########################

# MAIN ########################################################

class H0_mini_for_Adversarial(nn.Module):
    def __init__(self, possible_classes, base_model="hf-hub:bioptimus/H0-mini", emb_stain_size=64, device='cuda:0'):
        super().__init__()
        self.device = device
        self.to_pil = ToPILImage()
        self.emb_stain_size = emb_stain_size ## how many elements of the feature vector should contain staining information
        num_features = 768 ## embedding size, depends on base_model
        self.possible_classes = sorted(self.defrag(list(possible_classes.keys()))) ## for reproducibility
        self.enc = LabelEncoder()
        self.enc.fit(self.possible_classes)
        self.n_classes = len(self.enc.classes_)
        
        self.backbone, self.transform = get_encoder_and_transforms(base_model) ## backbone model
        print('Backbone built!')
        
        self.entangler = Entangler(768, emb_stain_size, 768-emb_stain_size, self.n_classes)
        print('Disentangler built!')
        
        self.image_decoder = get_decoder(3, num_features) ## decoder for image recon, adds a projection layer to mix stain
        print('Image Decoder built!')

        self.to(self.device)
    
    def freeze_or_unfreeze_disentangler(self, freeze=True):
        for param in self.base_projector.parameters():
            param.requires_grad = not freeze
        
    def forward(self, batch):
        if type(batch)==dict: tokens = self.backbone(batch['image'].to(self.device)) # [B, 261, 768]
        else: tokens = self.backbone(batch.to(self.device)) # [B, 261, 768]
        s, z = self.entangler.disentangle(tokens)
        return s, z
    
    def loss(self, batch, loss, logger=None, val=False):  
        s, z = self.forward(batch)
        
        ## project and split
        subsec_stain_proba, subsec_morph_proba = self.entangler.classify(s, z)
        batch['labels'] = self.transform_labels(batch)
        
        ## recon
        rec_img = self.recon_image(s, z)
        rec_morph = self.recon_image(torch.zeros_like(s), z)
        
        return loss(batch, subsec_stain_proba, subsec_morph_proba, rec_img, rec_morph, self.device, logger, val)

    def recon_image(self, s, z, transform=None):
        emb = self.entangler.reentangle(s, z)
        rec = self.image_decoder(emb)
        if transform is not None: rec = transform(rec)
        return rec
    
    def recon_image_PIL(self, emb, transform=None):
        return self.to_pil(self.recon_image(emb, transform).detach().squeeze())
    
    def save(self, pth, overwrite=False):
        pth = pl.Path(pth)
        if os.path.exists(pth) and not overwrite:
            override = datetime.datetime.now().strftime(r'H0-mini_from_%H:%M:%S-%d.%m.%y')
            print(f"INFO: {pth} already exists, using {pth.parent/override} instead")
            pth=pth.parent/override
        if not os.path.exists(pth): os.mkdir(pth)
        torch.save(self.backbone.state_dict(), pth/'backbone.pth')
        torch.save(self.image_decoder.state_dict(), pth/'image_decoder.pth')
        torch.save(self.entangler.state_dict(), pth/'entangler.pth')
        
    def load(self, pth):
        pth=pl.Path(pth)
        self.backbone.load_state_dict(torch.load(pth/'backbone.pth'))
        self.image_decoder.load_state_dict(torch.load(pth/'image_decoder.pth'))
        self.entangler.load_state_dict(torch.load(pth/'entangler.pth'))
        
    def freeze_backbone(self, freeze):
        for param in self.backbone.parameters():
            param.requires_grad = not freeze
            
    def defrag(self, raw_labels):
        new_labels = []
        for label in raw_labels:
            label = make_name_from_list(label)
            ## hematox & eosin
            if label == "HE - Hematoxylin and eosin stain method (procedure)" or label == "Hematoxylin and eosin stain method" or label == "hematoxylin stain+water soluble eosin stain":
                new_labels.append('H&E')
            ## Van Gieson
            elif label == "Van Gieson stain" or label == "Verhoeff-Van Gieson stain method":
                new_labels.append("Van Gieson stain")
            elif "Periodic acid Schiff stain" in label and "blue" not in label:
                new_labels.append("Periodic acid Schiff stain")
            elif "Herovici's stain method"==label or "Herovic's stain method"==label: new_labels.append("Herovicis stain method")
            else: new_labels.append(label)
        return new_labels
    
    def transform_labels(self, labels):
        labels = self.defrag(labels)
        return self.enc.transform(labels)

# DISENTANGLER ########################################################

class Entangler(nn.Module):
    def __init__(self, n_features, n_stain_features, n_other_features, n_classes):
        super().__init__()
        
        ### disentanglement
        self.to_specified = PooledProjector(n_features, n_stain_features)
        self.to_unspecified = ConvProjector(n_features, n_other_features)
        
        ### classification
        self.act = nn.Softmax()
        self.pred_specified = nn.Linear(n_stain_features, n_classes)
        self.pred_unspecified = nn.Linear(n_other_features, n_classes)
        
        ### reentanglement
        self.porbas_to_specified = nn.Linear(n_classes, n_stain_features)
        self.act2 = nn.ReLU()
        self.reentangler = nn.Conv2d(n_features, n_features, 1)
    
    def disentangle(self, tokens):
        patches = tokens[:,5:,:].permute(0, 2, 1) ## cut out cls and register tokens
        patches = patches.reshape(patches.shape[0], 768, 16, 16) ## reshape to feature map 
        z = self.to_unspecified(patches)
        s = self.to_specified(patches)
        return s, z # -> [B, n_stain_features], [B, n_other_features, 16, 16]
    
    def reentangle(self, s, z):
        s = self.porbas_to_specified(s)
        s = self.act2(s)
        s = s[:, :, None, None].expand(-1, -1, 16, 16)
        patches = torch.cat([s, z], dim=1)
        patches = self.reentangler(patches)
        return patches
    
    def classify(self, s, z):
        s = self.pred_specified(s)
        s_probas = self.act(s)
        z = self.pred_unspecified(self.reverse_grad(z))
        z_probas = self.act(z)
        return s_probas, z_probas
    
    def reverse_grad(self, x, alpha=1.0):
        return GradientReversal.apply(x, alpha)

class PooledProjector(nn.Module):
    def __init__(self, n_features, n_outputs):
        super().__init__()
        self.pooler = nn.AvgPool2d(16)
        self.linear = nn.Linear(n_features, n_outputs)
        self.act = nn.ReLU()
    
    def forward(self, x):
        is_batched = len(x.shape)==4
        x = self.pooler(x)
        if not is_batched: x = x.unsqueeze(0)
        x = self.linear(x)
        x = self.act(x)
        return x

class ConvProjector(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.layer = nn.Conv2d(in_channels, out_channels, 1)
        self.act = nn.ReLU()
        
    def forward(self, x):
        x = self.layer(x)
        x = self.act(x)
        return x
    
# CLASSIFIER ########################################################

class GradientReversal(Function):
    @staticmethod
    def forward(ctx, x, alpha):
        ctx.alpha = alpha
        return x.clone()

    @staticmethod
    def backward(ctx, grad_output):
        return -ctx.alpha * grad_output, None
    
# DECODER ########################################################

def get_decoder(num_features, channels):
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