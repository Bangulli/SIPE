######## Ecosystem ########
import os, sys, pathlib as pl
sys.path.append(os.path.join(os.path.dirname(__file__), "."))
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
######## External ########
import torch
import torch.nn as nn
import torchvision.transforms as T
from torchmetrics.image import StructuralSimilarityIndexMeasure
import torch.nn.functional as F
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
    
class ImageReconLoss_SSIM(StructuralSimilarityIndexMeasure):
    def __init__(self, data_range=1.0, testmode=False):
        super().__init__(data_range=data_range)
        self.testmode = testmode
        
    def forward(self, rec, gt):
        self.to(rec.device)
        gt = gt['image'].to(rec.device) if type(gt)==dict else gt
        if self.testmode: 
            print('Morph recon: rec.shape, binary.shape')
            print(rec.shape, gt.shape)
        return (1-super().forward(rec, gt))+(F.mse_loss(rec, gt)) ## ssim is between 0(worst) and 1(best)