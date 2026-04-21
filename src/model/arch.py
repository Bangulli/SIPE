######## Ecosystem ########
import os, sys, pathlib as pl, datetime
sys.path.append(os.path.join(os.path.dirname(__file__), "."))
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
######## External ########
import torch
import torch.nn as nn
######## Internal ########
from src.model.encoder import get_encoder_and_transforms
from src.model.decoder import get_decoder
from torchvision.transforms import ToPILImage
from src.model.projector import Projector, FullProjector, FullNormalizer, Disentangler, ConvDisentangler
from src.model.classifier import FullClassif, FullConvClassif
from src.utils.misc import make_name_from_list
##########################

#########################################################################################################################################        
class H0_mini_for_Adversarial(nn.Module):
    def __init__(self, possible_classes, base_model="hf-hub:bioptimus/H0-mini", emb_stain_size=64, device='cuda:0'):
        super().__init__()
        self.device = device
        self.to_pil = ToPILImage()
        self.emb_stain_size = emb_stain_size ## how many elements of the feature vector should contain staining information
        num_features = 768 ## embedding size, depends on base_model
        
        self.backbone, self.transform = get_encoder_and_transforms(base_model) ## backbone model
        self._freeze_model(self.backbone) ## freeze backbone
        print('Backbone built!')
        
        self.base_projector = ConvDisentangler(768, emb_stain_size, 768-emb_stain_size)
        print('Disentangler built!')
        
        self.image_decoder = get_decoder(base_model, 3, num_features) ## decoder for image recon
        print('Image Decoder built!')
        
        self.morph_decoder = get_decoder(base_model, 1, num_features-emb_stain_size) ## decoder for morpholgy recon -> canny mask
        print('Morph Decoder built!')
        
        self.classif = FullConvClassif(self.emb_stain_size, num_features-self.emb_stain_size, possible_classes)
        print(f'Classifier built! Predicts {len(self.classif.enc.classes_)} Classes')
        
        self.to(self.device)
    
    def freeze_or_unfreeze_disentangler(self, freeze=True):
        for param in self.base_projector.parameters():
            param.requires_grad = not freeze
        
    def forward(self, batch):
        tokens = self.backbone(batch['image'].to(self.device)) # [B, 261, 768]
        x = tokens[:,5:,:].permute(0, 2, 1) ## cut out cls and register tokens
        x = x.reshape(x.shape[0], 768, 16, 16) ## reshape to feature map 
        return self.base_projector(x)
    
    def loss(self, batch, loss, logger=None, val=False):  
        emb = self.forward(batch)
        
        ## project and split
        subsec_stain_proba, subsec_morph_proba = self.classif(emb)
        batch['labels'] = self.classif.transform_labels([make_name_from_list(l['staining']) for l in batch['metadata']])
        
        ## recon
        rec_img = self.image_decoder(emb)
        rec_morph = self.morph_decoder(emb[:, self.emb_stain_size:])
        
        return loss(batch, subsec_stain_proba, subsec_morph_proba, rec_img, rec_morph, self.device, logger, val)

    def recon_image(self, emb, transform=None):
        rec = self.image_decoder(emb).detach().squeeze()
        #print(rec.shape, 'recon_image')
        if transform is not None: rec = transform(rec)
        return rec
    
    def recon_image_PIL(self, emb, transform=None):
        return self.to_pil(self.recon_image(emb, transform).squeeze(0))
    
    def recon_morph(self, emb, transform=None):
        rec = self.morph_decoder(emb[:, self.emb_stain_size:]).detach().squeeze()
        if transform is not None: rec = transform(rec)
        return rec
        
    def recon_morph_PIL(self, emb, transform=None):
        return self.to_pil(self.recon_morph(emb, transform).squeeze(0))
    
    def save(self, pth, overwrite=False):
        pth = pl.Path(pth)
        if os.path.exists(pth) and not overwrite:
            override = datetime.datetime.now().strftime(r'H0-mini_from_%H:%M:%S-%d.%m.%y')
            print(f"INFO: {pth} already exists, using {pth.parent/override} instead")
            pth=pth.parent/override
        if not os.path.exists(pth): os.mkdir(pth)
        torch.save(self.backbone.state_dict(), pth/'backbone.pth')
        torch.save(self.image_decoder.state_dict(), pth/'image_decoder.pth')
        torch.save(self.morph_decoder.state_dict(), pth/'morph_decoder.pth')
        torch.save(self.classif.state_dict(), pth/'classif.pth')
        torch.save(self.base_projector.state_dict(), pth/'base_projector.pth')
        
    def load(self, pth):
        pth=pl.Path(pth)
        self.backbone.load_state_dict(torch.load(pth/'backbone.pth'))
        self.image_decoder.load_state_dict(torch.load(pth/'image_decoder.pth'))
        self.morph_decoder.load_state_dict(torch.load(pth/'morph_decoder.pth'))
        self.classif.load_state_dict(torch.load(pth/'classif.pth'))
        self.base_projector.load_state_dict(torch.load(pth/'base_projector.pth'))
        
    def _freeze_model(self, model):
        for param in model.parameters():
            param.requires_grad = False

#########################################################################################################################################        
class PixCell_uni2h(nn.Module): # pixcell diffusion with uni2h
    pass

#########################################################################################################################################
class LIC(nn.Module): # learned image compression approach
    pass