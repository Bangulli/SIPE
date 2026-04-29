######## Ecosystem ########
import os, sys, pathlib as pl
sys.path.append(os.path.join(os.path.dirname(__file__), "."))
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
######## External ########
import torch
import torch.nn as nn
######## Internal ########
from src.losses.image_recon_loss import ImageReconLoss
from src.losses.morphologic_recon_loss import MorphReconLoss_SSIM_Sobel
from src.losses.staining_cluster_loss import SimCLR_NCE_Loss
from src.losses.adversarial_classif_loss import AdversarialClassifLoss
##########################

class SIPE_Loss_Recon(nn.Module):
    def __init__(self, testmode=False):
        super().__init__()
        self.testmode=testmode
        self.image_recon_loss = ImageReconLoss(testmode=False)
        self.morph_recon_loss = MorphReconLoss_SSIM_Sobel(testmode=False)
        
        
    def forward(self, 
                gt, ## the gt, dict, 'image': [B, C, H, W]
                proj_stain, ## the embedding containing stain info [B, N]
                proj_morph, ## the embedding containing morph infor [B, N]
                rec_img, ## the full image reconstruction [B, C, H, W]
                rec_morph, ## the morophologic image reconstruction [B, C, H, W]
                device, 
                logger=None,
                val = False,
                ):
        ## Compute and fuse reconstruction losses
        image_recon_loss = self.image_recon_loss(rec_img.to(device), gt['image'].to(device))
        morph_recon_loss = self.morph_recon_loss(rec_morph.to(device), gt, device).to(device)
        recon_loss = image_recon_loss + morph_recon_loss
        ## Fuse losses recon is more important
        final_loss = recon_loss   
        
        if logger is not None:
            logger['Recon Img'].append(image_recon_loss.item())
            logger['Recon Morph'].append(morph_recon_loss.item())
        
        ## report
        if self.testmode: print('Image Recon Loss Value:',image_recon_loss.item())
        if self.testmode: print('Morph Recon Loss Value:',morph_recon_loss.item())
        if self.testmode: print('combined Recon Loss Value:',recon_loss.item())
        if self.testmode: print('Final loss:', final_loss.item())
        return final_loss, logger

class SIPE_Loss_Adversarial(nn.Module):
    """https://proceedings.neurips.cc/paper/2016/file/ef0917ea498b1665ad6c701057155abe-Paper.pdf
    """
    def __init__(self, testmode=False):
        super().__init__()
        self.testmode=testmode
        self.image_recon_loss = ImageReconLoss(testmode=False)
        self.morph_recon_loss = MorphReconLoss_SSIM_Sobel(testmode=False)
        self.stain_classif_loss = AdversarialClassifLoss(testmode=testmode) ## relies on a shuffled dataset. if not shuffled it is impossible to construct pos/neg pairs.
        self.alpha = 0.05
    
    def set_adverse_alpha(self, alpha):
        self.alpha=alpha
        
    def set_adverse_norm(self, norm):
        self.stain_classif_loss.set_norm(norm)
        
    def forward(self, 
                gt, ## the gt, dict, 'image': [B, C, H, W]
                proj_stain, ## the embedding containing stain info [B, N]
                proj_morph, ## the embedding containing morph infor [B, N]
                rec_img, ## the full image reconstruction [B, C, H, W]
                rec_morph, ## the morophologic image reconstruction [B, C, H, W]
                device, 
                logger=None,
                val=False,
                ):
        ## Compute and fuse reconstruction losses
        image_recon_loss = self.image_recon_loss(rec_img.to(device), gt['image'].to(device))
        morph_recon_loss = self.morph_recon_loss(rec_morph.to(device), gt, device).to(device)
        recon_loss = image_recon_loss + morph_recon_loss
        ## Compute staining cluster loss
        stain_loss, logger = self.stain_classif_loss(proj_stain.to(device), proj_morph.to(device), gt['labels'], device, logger, val, self.alpha)
        ## Fuse losses recon is more important
        final_loss = recon_loss + stain_loss        
        
        if logger is not None:
            logger['Recon Img'].append(image_recon_loss.item())
            logger['Recon Morph'].append(morph_recon_loss.item())
        
        ## report
        if self.testmode: print('Image Recon Loss Value:',image_recon_loss.item())
        if self.testmode: print('Morph Recon Loss Value:',morph_recon_loss.item())
        if self.testmode: print('combined Recon Loss Value:',recon_loss.item())
        if self.testmode: print('Staining cluster loss:',stain_loss.item())
        if self.testmode: print('Final loss:', final_loss.item())
        return final_loss, logger