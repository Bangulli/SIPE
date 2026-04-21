from torch import nn
import torch, math
from torchmetrics import Accuracy
   
class AdversarialClassifLoss(nn.Module): 
    def __init__(self, testmode):
        super().__init__()
        self.testmode= testmode
        self.lamda_adverse = 0.1 ## avoid the signal from exploding -> maybe reduce further its at around 4.0 which is quite high compared to the rest, may outweigh the recon during backprop
        self.lamda_ce = 1.0 ## keep signal intact
        self.compute = nn.CrossEntropyLoss()
        self.acc = Accuracy("multiclass", num_classes=105)
        
    def forward(self, proj_stain_proba, proj_morph_proba, gt, device, logger, val, alpha):
        gt = torch.tensor(gt, device=device)

        s_loss = self.compute(proj_stain_proba, gt) ### gets regular logits
        
        z_loss = alpha*self.compute(proj_morph_proba, gt)/math.log(105) ### gets grad reverse logits
        if self.testmode: print('losses:', f's={s_loss.item()}', f"z={z_loss.item()}")
        if logger is not None:
            logger['CE'].append(s_loss.item())
            logger['Adversarial CE'].append(z_loss.item())
        if not val: return (s_loss)+(z_loss), logger
        else: return self.lamda_ce*s_loss, logger 
    