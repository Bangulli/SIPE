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
        gt = torch.tensor(gt, device=device)

        s_loss = self.compute(proj_stain_proba, gt) ### gets regular logits
        
        if self.norm: z_loss = alpha*self.compute(proj_morph_proba, gt)/math.log(105) ### gets grad reverse logits
        else: z_loss = alpha*self.compute(proj_morph_proba, gt)
        if self.testmode: print('losses:', f's={s_loss.item()}', f"z={z_loss.item()}")
        if logger is not None:
            logger['CE'].append(s_loss.item())
            logger['Adversarial CE'].append(z_loss.item())
        if not val: return (s_loss)+(z_loss), logger
        else: return s_loss, logger 
        
    def set_norm(self, norm):
        self.norm = norm
    