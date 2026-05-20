######## Ecosystem ########
import os, sys, pathlib as pl, datetime
sys.path.append(os.path.join(os.path.dirname(__file__), "."))
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
######## External ########
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.preprocessing import LabelEncoder, MultiLabelBinarizer
from torch.autograd import Function
from huggingface_hub import login
import timm
from timm.data import resolve_data_config
from timm.data.transforms_factory import create_transform
######## Internal ########
from torchvision.transforms import ToPILImage
from src.utils.misc import make_name_from_list
##########################


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

# MAIN ########################################################

class H0_mini_for_VAE(nn.Module):
    def __init__(self, possible_classes, base_model="hf-hub:bioptimus/H0-mini", emb_stain_size=64, device='cuda:0'):
        super().__init__()
        self.device = device
        self.to_pil = ToPILImage()
        self.emb_stain_size = emb_stain_size ## how many elements of the feature vector should contain staining information
        num_features = 768 ## embedding size, depends on base_model
        self.possible_classes = sorted(self.defrag(list(possible_classes.keys()))) ## for reproducibility
        self.enc = MultiLabelBinarizer()
        self.enc.fit(self.possible_classes)
        self.n_classes = len(self.enc.classes_)
        
        self.backbone, self.transform = get_encoder_and_transforms(base_model) ## backbone model
        print('Backbone built!')
        
        self.entangler = VAEntangler(768, 64, 704, self.n_classes, 704)
        print('Disentangler built!')
        
        self.image_decoder = Decoder(num_features, 3) ## decoder for image recon, adds a projection layer to mix stain
        print('Image Decoder built!')

        self.to(self.device)
    
    def freeze_or_unfreeze_disentangler(self, freeze=True):
        for param in self.entangler.parameters():
            param.requires_grad = not freeze
        
    def forward(self, batch):
        if type(batch)==dict: tokens = self.backbone(batch['image'].to(self.device)) # [B, 261, 768]
        else: tokens = self.backbone(batch.to(self.device)) # [B, 261, 768]
        s, z, mu, logvar = self.entangler.disentangle(tokens)
        return s, z
    
    def loss(self, batch, loss, logger=None, val=False):  
        if type(batch)==dict: tokens = self.backbone(batch['image'].to(self.device)) # [B, 261, 768]
        else: tokens = self.backbone(batch.to(self.device)) # [B, 261, 768]
        
        ## get specified and unspecified maps
        s, z, mu, logvar = self.entangler.disentangle(tokens)
        
        ## get classification logits
        s_classif, z_classif = self.entangler.classify(s, z)
        
        ## project and split
        batch['labels'] = torch.tensor(self.transform_labels([s['staining'] for s in batch['metadata']]), dtype=torch.float32)

        ## recon
        rec_img = self.recon_image(s, z) ##  make image reconstruction loss only affect generator, the BCE from the VAE already takes care of the reconstruction enforcement on the vector level
        
        return loss(batch, s_classif, z_classif, rec_img, mu, logvar, self.device, logger, val)

    def recon_image(self, s, z, transform=None):
        if len(s.shape)==1: s=s.unsqueeze(0)
        emb = self.entangler.reentangle(s, z)
        rec = self.image_decoder(emb)
        if transform is not None: rec = transform(rec)
        return rec
    
    def recon_image_PIL(self, s, z, transform=None):
        return self.to_pil(self.recon_image(s, z, transform).detach().squeeze())
    
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
                new_labels.append('Hematoxylin+Eosin')
            ## Van Gieson
            elif label == "Van Gieson stain" or label == "Verhoeff-Van Gieson stain method":
                new_labels.append("Van Gieson stain")
            elif "Periodic acid Schiff stain" in label and "blue" not in label:
                new_labels.append("Periodic acid Schiff stain")
            elif "Periodic acid Schiff stain" in label and "blue" in label:
                new_labels.append("Periodic acid Schiff stain+Alcian blue")
            elif "Herovici's stain method"==label or "Herovic's stain method"==label: new_labels.append("Herovicis stain method")
            else: new_labels.append(label)
        sep_labels = []
        for lbl in new_labels:
            sep = lbl.split('+')
            if len(sep)>1:
                new_sep = []
                for l in sep:
                    l = l.strip().lower()
                    if 'hematoxylin' in l: new_sep.append('hematoxylin')
                    elif 'eosin' in l: new_sep.append('eosin')
                    elif 's-100' in l: new_sep.append('s100')
                    else: new_sep.append(l)
                sep_labels.append(new_sep)
            else: sep_labels.append(sep)
        new_labels = sep_labels
            
        return new_labels
    
    def transform_labels(self, labels):
        labels = self.defrag(labels)
        return self.enc.transform(labels)

# DISENTANGLER ########################################################

class VAEntangler(nn.Module):
    def __init__(self, n_features, n_spec_features, n_unspec_features, n_classes, n_latents):
        super().__init__()
        self.n_features = n_features
        self.n_classes = n_classes
        self.n_spec_features = n_spec_features
        self.n_unspec_features = n_unspec_features
        
        ### disentanglement
        self.to_specified = PooledProjector(n_features, n_spec_features)
        self.to_unspecified = ConVAE(n_features, n_unspec_features, n_latents)
        
        ### classification
        self.act = nn.Softmax()
        self.pred_specified = nn.Linear(n_spec_features, n_classes)
        self.pred_unspecified = nn.Linear(n_unspec_features, n_classes)
        
        ### reentanglement
        self.probas_to_specified = nn.Linear(n_classes, n_spec_features)
        self.act2 = nn.ReLU()
        self.reentangler = nn.Conv2d(n_spec_features+n_unspec_features, n_features, 1)
    
    def disentangle(self, tokens):
        patches = tokens[:,5:,:].permute(0, 2, 1) ## cut out cls and register tokens
        patches = patches.reshape(patches.shape[0], 768, 16, 16) ## reshape to feature map 
        z, mu, logvar = self.to_unspecified(patches)
        s = self.to_specified(patches)
        return s, z, mu, logvar # -> [B, n_spec_features], [B, n_unspec_features, 16, 16], [B, n_latents, 16, 16], [B, n_latents, 16, 16]
    
    def reentangle(self, s, z):
        s = s[:, :, None, None].expand(-1, -1, 16, 16)
        patches = torch.cat([s, z], dim=1)
        patches = self.reentangler(patches)
        return patches
    
    def reentangle_from_probas(self, s_probas, z):
        raise NotImplementedError
        s = self.probas_to_specified(s_probas)
        s = s[:, :, None, None].expand(-1, -1, 16, 16)
        patches = torch.cat([s, z], dim=1)
        patches = self.reentangler(patches)
        return patches
    
    def classify(self, s, z):
        s = self.pred_specified(s)
        
        ## reshape from [B, C, H, W] -> [B*W*H, C] such that each individual patch gets processed instead of a pooled version.
        z = z.permute(0, 2, 3, 1).reshape(-1, self.n_unspec_features )
        z = self.pred_unspecified(self.reverse_grad(z))
        return s, z
    
    def to_probas(self, s, z):
        s_probas = self.act(s)
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
        x = self.pooler(x).squeeze()
        x = self.linear(x)
        x = self.act(x)
        return x
    
# Convolutional VAE ########################################################

class ConvProjector(nn.Module):
    def __init__(self, n_features, n_outputs):
        super().__init__()
        self.linear = nn.Conv2d(n_features, n_outputs, 1)
        self.act = nn.ReLU()
    
    def forward(self, x):
        x = self.linear(x)
        x = self.act(x)
        return x
    
class ConVAE(nn.Module): # https://github.com/pytorch/examples/blob/main/vae/main.py
    def __init__(self, n_features, n_unspec_features, n_latents):
        super().__init__()
        self.projector = ConvProjector(n_features, n_unspec_features)
        self.to_mu = nn.Conv2d(n_unspec_features, n_latents, 1)
        self.to_logvar = nn.Conv2d(n_unspec_features, n_latents, 1)
        self.decoder = nn.Sequential(
            nn.Conv2d(n_latents, n_unspec_features, 1),
            nn.ReLU(),
            nn.Conv2d(n_unspec_features, n_unspec_features, 1),
            nn.Sigmoid()
        )
        
    def forward(self, x):
        x = self.projector(x)
        mu = self.to_mu(x)
        logvar = self.to_logvar(x)
        z = self.reparameterize(mu, logvar)
        return self.decoder(z), mu, logvar
        
    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5*logvar)
        eps = torch.randn_like(std)
        return mu + eps*std
    
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

class Decoder(nn.Module):
    def __init__(self, num_features, channels):
        super().__init__()
        self.decoder = nn.Sequential(
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
    
    def forward(self, x):
        return self.decoder(x)
    
