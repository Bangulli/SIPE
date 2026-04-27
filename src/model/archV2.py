######## Ecosystem ########
import os, sys, pathlib as pl, datetime
sys.path.append(os.path.join(os.path.dirname(__file__), "."))
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
######## External ########
import torch
import torch.nn as nn
from sklearn.preprocessing import LabelEncoder, MultiLabelBinarizer
######## Internal ########
from src.model.encoder import get_encoder_and_transforms
from src.model.decoder import get_decoder_simple, get_proj_decoder_simple_v2
from src.model.classifier import ConvClassif, GradientReversal
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
        
        self.backbone, self.transform = get_encoder_and_transforms(base_model) ## backbone model
        print('Backbone built!')
        
        self.disentangler = DisentanglerV2(self.n_classes, num_features)
        print('Disentangler built!')
        
        self.image_decoder = get_proj_decoder_simple_v2(3, num_features, self.n_classes) ## decoder for image recon, adds a projection layer to mix stain
        print('Image Decoder built!')
        
        self.adv_classif = ConvClassif(num_features, self.n_classes)
        print(f'Classifier built! Predicts {self.n_classes} Classes')
        
        self.to(self.device)
        
    def forward(self, batch):
        tokens = self.backbone(batch['image'].to(self.device)) # [B, 261, 768]
        return self.disentangler(tokens)
    
    def loss(self, batch, loss, logger=None, val=False):  
        s, z = self.forward(batch)
        
        ## predict
        subsec_morph_proba = self.adv_classif(self.reverse_grad(z))
        gt_labels = self.transform_labels([s['staining'] for s in batch['metadata']])
        
        ## recon
        rec_img = self.image_decoder(self.disentangler.fuse(s, z))
        rec_morph = self.image_decoder(self.disentangler.fuse(torch.zeros_like(s, device=s.device), z))
        
        return loss(batch['image'].to(self.device), torch.from_numpy(gt_labels).to(self.device), s, subsec_morph_proba, rec_img, rec_morph, self.device, logger, val)

    #----------------------- Recon utils
    def recon_image(self, s, z, transform=None):
        rec = self.image_decoder(self.disentangler.fuse(s, z)).detach().squeeze()
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
        
    def load(self, pth):
        pth=pl.Path(pth)
        self.backbone.load_state_dict(torch.load(pth/'backbone.pth'))
        self.image_decoder.load_state_dict(torch.load(pth/'image_decoder.pth'))
        self.disentangler.load_state_dict(torch.load(pth/'disentangler.pth'))
    
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

class DisentanglerV2(nn.Module):
    def __init__(self, n_classes, n_features):
        super().__init__()
        self.n_c = n_classes
        self.n_f = n_features
        self.s_projector = nn.Sequential( ## present the staining in compound likelihoods, so later it can just be modeled as 50% hematox and 50% eosin for H&E
            nn.Linear(self.n_f, self.n_c),
            nn.Softmax()
        )
        self.z_projector = nn.Sequential(
            nn.Conv2d(self.n_f, self.n_f, kernel_size=1), ## mix the staining attachment into the data - patch wise
            nn.BatchNorm2d(self.n_f),
            nn.GELU(),
            nn.Conv2d(self.n_f, self.n_f, kernel_size=3, padding_mode='reflect', padding=1), ## neighbor aware mixing
            nn.BatchNorm2d(self.n_f),
            nn.GELU(),
            nn.Conv2d(self.n_f, self.n_f, kernel_size=1), ## patch wise cleaning
        )
        
    def forward(self, tokens):
        patches = tokens[:,5:,:].permute(0, 2, 1) ## cut out cls and register tokens
        patches = patches.reshape(patches.shape[0], 768, 16, 16) ## reshape to feature map 
        z = self.z_projector(patches)
        cls_toks = tokens[:,0,:]
        s = self.s_projector(cls_toks)
        return s, z
    
    def fuse(self, s, z):
        sq = False
        if z.dim() == 3: z=z.unsqueeze(0); sq=True
        s = s[:, :, None, None].expand(-1, -1, 16, 16)
        emb=torch.cat([s, z], dim=1)
        if sq: return emb.squeeze(0)
        else: return emb