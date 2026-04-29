######## Ecosystem ########
import os, sys, pathlib as pl, datetime
sys.path.append(os.path.join(os.path.dirname(__file__), "."))
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
######## External ########
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.preprocessing import LabelEncoder, MultiLabelBinarizer
######## Internal ########
from src.model.encoder import get_encoder_and_transforms
from src.model.classifier import GradientReversal
from torchvision.transforms import ToPILImage
from src.utils.misc import make_name_from_list
##########################

############################################## Main ##############################################

class V2_H0_mini_for_Adversarial(nn.Module):
    def __init__(self, possible_classes, individual_compounds=True, base_model="hf-hub:bioptimus/H0-mini", device='cuda:0'):
        super().__init__()
        self.individual_compounds = individual_compounds
        self.device = device
        self.to_pil = ToPILImage()
        self.possible_classes = sorted(self.defrag(list(possible_classes.keys()))) ## for reproducibility
        self.enc = MultiLabelBinarizer() if self.individual_compounds else LabelEncoder()
        self.enc.fit(self.possible_classes)
        self.n_classes = len(self.enc.classes_)
        num_features = 768 ## embedding size, depends on base_model
        
        self.backbone, self.transform = get_encoder_and_transforms(base_model) ## patch -> features
        print('Backbone built!')
        
        self.disentangler = Disentangler(num_features, self.n_classes) ## features -> stain probabs, stain-less features
        print('Disentangler built!')
        
        self.reentangler = Reentangler(num_features, self.n_classes) ## stain probabs, stain-less features -> features
        print('Reentangler built!')
        
        self.image_decoder = Decoder(num_features, 3) ## features -> patche
        print('Image Decoder built!')
        
        self.to(self.device)
        
    def forward(self, batch):
        tokens = self.backbone(batch['image'].to(self.device)) # [B, 261, 768]
        s, z = self.disentangler(tokens) # [B, 768, 16, 16] - specified & [B, 768, 16, 16] # unspecified
        return s, z
    
    def loss(self, batch, loss, logger=None, val=False):  
        s_proba, z = self.forward(batch)
    
        z_proba = self.disentangler.classify(self.reverse_grad(z))
        gt_labels = torch.from_numpy(self.transform_labels([s['staining'] for s in batch['metadata']]))
        
        ## recon
        rec_img = self.recon_image(s_proba, z)
        rec_morph = self.recon_image(torch.zeros_like(s_proba, device=self.device), z)
        return loss(batch['image'].to(self.device), gt_labels.to(self.device), s_proba, z_proba, rec_img, rec_morph, self.device, logger, val)

    #----------------------- Recon utils
    def recon_image(self, s, z, transform=None):
        feature_map = self.reentangler(s, z)
        rec = self.image_decoder(feature_map)
        if transform is not None: rec = transform(rec)
        return rec
    
    def recon_image_PIL(self, s, z, transform=None):
        return self.to_pil(self.recon_image(s, z, transform).squeeze(0))
    
    #----------------------- I/O utils
    def save(self, pth, overwrite=False):
        pth = pl.Path(pth)
        if os.path.exists(pth) and not overwrite:
            override = datetime.datetime.now().strftime(r'H0-mini_from_%H:%M:%S-%d.%m.%y')
            print(f"INFO: {pth} already exists, using {pth.parent/override} instead")
            pth=pth.parent/override
        if not os.path.exists(pth): os.mkdir(pth)
        torch.save(self.backbone.state_dict(), pth/'backbone.pth')
        torch.save(self.image_decoder.state_dict(), pth/'image_decoder.pth')
        torch.save(self.disentangler.state_dict(), pth/'disentangler.pth')
        torch.save(self.reentangler.state_dict(), pth/'reentangler.pth')
        
    def load(self, pth):
        pth=pl.Path(pth)
        self.backbone.load_state_dict(torch.load(pth/'backbone.pth'))
        self.image_decoder.load_state_dict(torch.load(pth/'image_decoder.pth'))
        self.disentangler.load_state_dict(torch.load(pth/'disentangler.pth'))
        self.reentangler.load_state_dict(torch.load(pth/'reentangler.pth'))
    
    #----------------------- Freezing utils
    def freeze_or_unfreeze_disentangler(self, freeze=True):
        for param in self.disentangler.parameters():
            param.requires_grad = not freeze
            
    def freeze_backbone(self, freeze):
        for param in self.backbone.parameters():
            param.requires_grad = not freeze
            
    #----------------------- Label stuff        
    def defrag_and_transform_labels(self, batch):
        return self.transform_labels([s['staining'] for s in batch['metadata']])
    
    def defrag_labels(self, batch):
        return self.defrag([s['staining'] for s in batch['metadata']])
            
    def transform_labels(self, labels):
        labels = self.defrag(labels)
        return self.enc.transform(labels)
    
    def reverse_grad(self, x, alpha=1.0):
        return GradientReversal.apply(x, alpha)
    
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
        
        if self.individual_compounds:
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

############################################## Disentangler ##############################################
class Disentangler(nn.Module): ## separates the feature map into stain and morph maps
    def __init__(self, n_features, n_classes):
        super().__init__()
        self.s_projector = Classif(n_features, n_classes)
        self.z_projector = nn.Conv2d(n_features, n_features, 1)
        
    def forward(self, tokens):
        feature_map = tokens[:,5:,:].permute(0, 2, 1) ## cut out cls and register tokens
        feature_map = feature_map.reshape(feature_map.shape[0], 768, 16, 16) ## reshape to feature map 
        s = self.s_projector(feature_map)
        z = self.z_projector(feature_map)
        return s, z
    
    def classify(self, z):
        return self.s_projector(z)
class Reentangler(nn.Module):
    def __init__(self, n_features, n_classes):
        super().__init__()
        self.stain_handler = StainEncoder(n_features, n_classes)
        self.projector = nn.Conv2d(n_features*2, n_features, 1)
        
    def forward(self, s, z):
        s = self.stain_handler(s)
        x = torch.cat([s, z], dim=1)
        return self.projector(x)   
class Classif(nn.Module):
    def __init__(self, n_features, n_classes):
        super().__init__()
        self.head = nn.Conv2d(n_features, n_classes, 16)
        self.act = nn.Softmax()
        
    def forward(self, x):
        x = self.head(x).squeeze()
        x = self.act(x)
        return x
class StainEncoder(nn.Module):
    def __init__(self, n_features, n_classes):
        super().__init__()
        self.deprob = nn.Linear(n_classes, n_classes)
        self.projector = nn.ConvTranspose2d(n_classes, n_features, 16)
        
    def forward(self, x):
        x = self.deprob(x) # -> probabs to logits
        x = x.unsqueeze(-1).unsqueeze(-1) # logits to logit map
        x = self.projector(x) # -> logit map to feature map
        return x
class Decoder(nn.Module):
    def __init__(self, num_features, channels):
        super().__init__()
        self.model = nn.Sequential(
            nn.ConvTranspose2d(num_features, 256, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(256), nn.GELU(),
            nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(128), nn.GELU(),
            nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(64), nn.GELU(),
            nn.ConvTranspose2d(64, channels, kernel_size=4, stride=2, padding=1),
            nn.Upsample(size=(224, 224), mode='bilinear', align_corners=False),
        )
    
    def forward(self, x):
        return self.model(x)