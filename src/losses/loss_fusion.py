######## Ecosystem ########
import os, sys, pathlib as pl
sys.path.append(os.path.join(os.path.dirname(__file__), "."))
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
######## External ########
import torch
import torch.nn as nn
######## Internal ########
from src.losses.image_recon_loss import ImageReconLoss
from src.losses.morphologic_recon_loss import MorphReconLoss_MSE_Sobel
from src.losses.staining_cluster_loss import SimCLR_NCE_Loss
from src.losses.adversarial_classif_loss import AdversarialClassifLoss
##########################
class SIPE_Loss_Adversarial(nn.Module):
    """https://proceedings.neurips.cc/paper/2016/file/ef0917ea498b1665ad6c701057155abe-Paper.pdf
    """
    def __init__(self, testmode=False, recon_mode=False):
        super().__init__()
        self.testmode=testmode
        self.recon_mode = recon_mode
        self.image_recon_loss = ImageReconLoss(testmode=False)
        self.probas_to_stainvec_loss = nn.MSELoss()
        #self.morph_recon_loss = MorphReconLoss_MSE_Sobel(testmode=False)
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
                rec_stain_vectors, ## the stain vectors reconstructed from stain probability
                orig_stain_vectors, ## the original stain vectors
                device, 
                logger=None,
                val=False,
                ):
        ## Compute and fuse reconstruction losses
        image_recon_loss = self.image_recon_loss(rec_img.to(device), gt['image'].to(device))
        #morph_recon_loss = self.morph_recon_loss(rec_morph.to(device), gt, device).to(device)
        recon_loss = image_recon_loss# + morph_recon_loss
        ## Compute staining cluster loss
        if not self.recon_mode: 
            stain_loss, logger = self.stain_classif_loss(proj_stain.to(device), proj_morph.to(device), gt['labels'], device, logger, val, self.alpha)
            ## Compute probs2vec loss
            p2v_loss = self.probas_to_stainvec_loss(rec_stain_vectors, orig_stain_vectors)
            ## Fuse losses recon is more important
            final_loss = recon_loss + stain_loss + p2v_loss       
        else: final_loss = recon_loss
        
        if logger is not None:
            logger['Recon Img'].append(image_recon_loss.item())
            if not self.recon_mode: logger['Stain probs2vec'].append(p2v_loss.item())
            #logger['Recon Morph'].append(morph_recon_loss.item())
        
        ## report
        if self.testmode: print('Image Recon Loss Value:',image_recon_loss.item())
        if self.testmode: print('P2V Recon Loss Value:',p2v_loss.item())
        if self.testmode: print('combined Recon Loss Value:',recon_loss.item())
        if self.testmode: print('Staining cluster loss:',stain_loss.item())
        if self.testmode: print('Final loss:', final_loss.item())
        return final_loss, logger
    
class SIPE_Loss_Adversarial_Cycle(nn.Module):
    """https://proceedings.neurips.cc/paper/2016/file/ef0917ea498b1665ad6c701057155abe-Paper.pdf
    """
    def __init__(self, testmode=False, recon_mode=False):
        super().__init__()
        self.testmode=testmode
        self.recon_mode = recon_mode
        self.image_recon_loss = ImageReconLoss(testmode=False)
        self.cycle_consistency_loss = nn.L1Loss()
        self.stain_classif_loss = AdversarialClassifLoss(testmode=testmode) ## relies on a shuffled dataset. if not shuffled it is impossible to construct pos/neg pairs.
        self.alpha = 0.05
    
    def set_adverse_alpha(self, alpha):
        self.alpha=alpha
        
    def set_adverse_norm(self, norm):
        self.stain_classif_loss.set_norm(norm)
        
    def to(self, device):
        self.device=device
        super().to(device)
        
    def forward(self, 
                gt_labels,
                gt_images,
                recon_images,
                s_orig,
                s_cycle,
                z_orig,
                z_cycle,
                s_class,
                z_class,
                logger = None,
                val = False
                ):
        
        recon_loss = self.image_recon_loss(recon_images.to(self.device), gt_images.to(self.device))
        s_cycle_loss = self.cycle_consistency_loss(s_cycle, s_orig)
        z_cycle_loss = self.cycle_consistency_loss(z_cycle, z_orig)
        stain_loss, logger = self.stain_classif_loss(s_class, z_class, gt_labels, self.device, logger, val, self.alpha)
        logger['Recon Img'].append(recon_loss.item())
        logger['S cycle'].append(s_cycle_loss.item())
        logger['Z cycle'].append(z_cycle_loss.item())
        ## computing stain loss on cycle outputs would be redundant with the cycle loss i think...
        return recon_loss + s_cycle_loss + z_cycle_loss + stain_loss, logger