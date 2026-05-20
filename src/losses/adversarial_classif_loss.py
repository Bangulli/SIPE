from torch import nn
import torch, math
from torchmetrics import Accuracy
   
class AdversarialClassifLoss(nn.Module): 
    def __init__(self, testmode, norm=True):
        super().__init__()
        self.testmode= testmode
        self.norm = norm
        self.compute = nn.CrossEntropyLoss()
        
    def forward(self, proj_stain_proba, proj_morph_proba, gt, device, logger, val, alpha):
        gt = gt.to(device)
        n_classes = gt.shape[1]

        if self.norm: s_loss = self.compute(proj_stain_proba, gt)/math.log(n_classes) ### gets regular logits
        else: s_loss = self.compute(proj_stain_proba, gt)
        
        if proj_morph_proba.shape[0] != proj_stain_proba.shape[0]:
            H=16;W=16
            if self.norm: z_loss = alpha*self.compute(proj_morph_proba, gt.repeat_interleave(H * W, dim=0))/math.log(n_classes) ### gets grad reverse logits
            else: z_loss = alpha*self.compute(proj_morph_proba, gt.repeat_interleave(H * W, dim=0))
        else:
            if self.norm: z_loss = self.compute(proj_morph_proba, gt)/math.log(n_classes) ### gets regular logits
            else: z_loss = self.compute(proj_morph_proba, gt)
        
        if self.testmode: print('losses:', f's={s_loss.item()}', f"z={z_loss.item()}")
        
        if logger is not None:
            logger['CE'].append(s_loss.item())
            logger['Adversarial CE'].append(z_loss.item())
            
        if not val: return (s_loss)+(z_loss), logger
        else: return s_loss, logger 
        
    def set_norm(self, norm):
        self.norm = norm
    