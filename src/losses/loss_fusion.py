######## Ecosystem ########
import os, sys, pathlib as pl
sys.path.append(os.path.join(os.path.dirname(__file__), "."))
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
######## External ########
import torch
import torch.nn as nn
######## Internal ########
from src.losses.image_recon_loss import ImageReconLoss, GAN_Loss
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
        self.image_recon_loss = ImageReconLoss()
        self.probas_to_stainvec_loss = nn.MSELoss()
        #self.morph_recon_loss = MorphReconLoss_MSE_Sobel(testmode=False)
        self.stain_classif_loss = AdversarialClassifLoss(testmode=testmode, logkey='S') ## relies on a shuffled dataset. if not shuffled it is impossible to construct pos/neg pairs.
        self.organ_classif_loss = AdversarialClassifLoss(testmode=testmode, logkey='O')
        self.path_classif_loss = AdversarialClassifLoss(testmode=testmode, logkey='P')
        self.alpha = 0.05
    
    def set_adverse_alpha(self, alpha):
        self.alpha=alpha
        
    def set_adverse_norm(self, norm):
        self.stain_classif_loss.set_norm(norm)
        
    def forward(self, 
                gt, ## the gt, dict, 'image': [B, C, H, W]
                s_s, ## the embedding containing stain info [B, N]
                z_s, ## the embedding containing morph infor [B, N]
                s_o, ## the embedding containing stain info [B, N]
                z_o, ## the embedding containing morph infor [B, N]
                s_p, ## the embedding containing stain info [B, N]
                z_p, ## the embedding containing morph infor [B, N]
                
                rec_img, ## the full image reconstruction [B, C, H, W]
                rec_stain_vectors, ## the stain vectors reconstructed from stain probability
                orig_stain_vectors, ## the original stain vectors
                device, 
                logger=None,
                val=False,
                disc_gt = None,
                disc_rec = None,
                ):
        ## Compute and fuse reconstruction losses
        if isinstance(self.image_recon_loss, ImageReconLoss):
            image_recon_loss = self.image_recon_loss(rec_img.to(device), gt['image'].to(device))
            recon_loss = image_recon_loss
        elif disc_gt is not None and disc_rec is not None and isinstance(self.image_recon_loss, GAN_Loss):
            perception_loss, gan_loss = self.image_recon_loss(gt['image'].to(device), rec_img.to(device), disc_gt, disc_rec)
            recon_loss = perception_loss + 0.5*gan_loss
        else: raise RuntimeError('Did not receive a valid loss/value configuration for MSE or GAN loss.')
            
        
        ## Compute staining cluster loss
        if not self.recon_mode: 
            stain_loss, logger = self.stain_classif_loss(s_s.to(device), z_s.to(device), gt['stain'], device, logger, val, self.alpha)
            organ_loss, logger = self.organ_classif_loss(s_o.to(device), z_o.to(device), gt['organ'], device, logger, val, self.alpha)
            path_loss, logger = self.path_classif_loss(s_p.to(device), z_p.to(device), gt['diagnosis'], device, logger, val, self.alpha)
            
            ## Compute probs2vec loss
            p2v_loss = self.probas_to_stainvec_loss(rec_stain_vectors, orig_stain_vectors)
            
            ## Fuse losses recon is more important
            final_loss = recon_loss + stain_loss + p2v_loss + organ_loss + path_loss   
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
        self.image_recon_loss = ImageReconLoss()
        self.cycle_consistency_loss = nn.L1Loss()
        self.stain_classif_loss = AdversarialClassifLoss(testmode=testmode, logkey='S') ## relies on a shuffled dataset. if not shuffled it is impossible to construct pos/neg pairs.
        self.organ_classif_loss = AdversarialClassifLoss(testmode=testmode, logkey='O')
        self.path_classif_loss = AdversarialClassifLoss(testmode=testmode, logkey='P')
        self.alpha = 0.05
    
    def set_adverse_alpha(self, alpha):
        self.alpha=alpha
        
    def set_adverse_norm(self, norm):
        self.stain_classif_loss.set_norm(norm)
        
    def to(self, device):
        self.device=device
        super().to(device)
        
    def forward(self, 
                gt_labels_s,
                gt_labels_o,
                gt_labels_p,
                gt_images,
                recon_images,
                s_orig,
                s_cycle,
                z_orig,
                z_cycle,
                s_class_s,
                z_class_s,
                s_class_o,
                z_class_o,
                s_class_p,
                z_class_p,
                logger = None,
                val = False,
                disc_gt = None,
                disc_rec = None
                ):
        
        if isinstance(self.image_recon_loss, ImageReconLoss):
            image_recon_loss = self.image_recon_loss(recon_images.to(self.device), gt_images.to(self.device))*20
            recon_loss = image_recon_loss
        elif disc_gt is not None and disc_rec is not None and isinstance(self.image_recon_loss, GAN_Loss):
            perception_loss, gan_loss = self.image_recon_loss(recon_images.to(self.device), gt_images.to(self.device), disc_gt, disc_rec)
            recon_loss = perception_loss + 0.5*gan_loss
        else: raise RuntimeError('Did not receive a valid loss/value configuration for MSE or GAN loss.')
            
        
        s_cycle_loss = self.cycle_consistency_loss(s_cycle, s_orig)
        z_cycle_loss = self.cycle_consistency_loss(z_cycle, z_orig)*0.5
        
        
        stain_loss, logger = self.stain_classif_loss(s_class_s, z_class_s, gt_labels_s, self.device, logger, val, self.alpha)
        organ_loss, logger = self.organ_classif_loss(s_class_o, z_class_o, gt_labels_o, self.device, logger, val, self.alpha)
        path_loss, logger = self.path_classif_loss(s_class_p, z_class_p, gt_labels_p, self.device, logger, val, self.alpha)
        
        logger['Recon Img'].append(recon_loss.item())
        logger['S cycle'].append(s_cycle_loss.item())
        logger['Z cycle'].append(z_cycle_loss.item())
        ## computing stain loss on cycle outputs would be redundant with the cycle loss i think...
        return recon_loss + s_cycle_loss + z_cycle_loss + stain_loss + organ_loss + path_loss, logger