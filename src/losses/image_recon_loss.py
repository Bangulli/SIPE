######## Ecosystem ########
import os, sys, pathlib as pl
sys.path.append(os.path.join(os.path.dirname(__file__), "."))
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
######## External ########
import torch
import torch.nn as nn
import torchvision.transforms as T
from torchvision.models import vgg16, VGG16_Weights
######## Internal ########
##########################

class ImageReconLoss(nn.MSELoss):
    def __init__(self, size_average=None, reduce=None, reduction='mean', testmode=False):
        super().__init__(size_average, reduce, reduction)
        self.testmode=testmode
        
    def forward(self, rec, gt):
        if self.testmode: 
            print('rec', rec.shape, rec.min(), rec.max(), rec.type())
            print('gt', gt.shape, gt.min(), gt.max(), gt.type())
        return super().forward(rec, gt)

    
class GAN_Loss(nn.Module):
    def __init__(self):
        super().__init__()
        self.perceptor = vgg16(weights=VGG16_Weights)
        self.freeze_perceptor(True)
        self.mse = nn.MSELoss()
    
    def to(self, device):
        self.device=device
        super().to(device)
        
    def freeze_perceptor(self, freeze):
        for param in self.perceptor.parameters():
            param.requires_grad = not freeze
            
    def _labels(self, pred, real: bool):
        val = 1.0 if real else 0.0
        return torch.full_like(pred, val)
        
    def forward(self, gt_imgs, rec_imgs, gt_prob, rec_prob):
        gt_perc, rec_perc = self.perceptor(gt_imgs), self.perceptor(rec_imgs)
        perception_loss = self.mse(rec_perc, gt_perc)
        
        loss_real = self.mse(gt_prob, self._labels(gt_prob, True))
        loss_fake = self.mse(rec_prob, self._labels(rec_prob, False))