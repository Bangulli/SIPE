######## Ecosystem ########
import os, sys, pathlib as pl
sys.path.append(os.path.join(os.path.dirname(__file__), "."))
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
######## External ########
import torch
from torchmetrics.image import StructuralSimilarityIndexMeasure
import torch.nn as nn
import torchvision.transforms as T
######## Internal ########
##########################

class MorphReconLoss_MSE(nn.MSELoss):
    def __init__(self, size_average=None, reduce=None, reduction='mean', testmode=False):
        super().__init__(size_average, reduce, reduction)
        self.testmode = testmode
        
    def forward(self, rec, gt, device):
        binary = self._binarize(gt['image'].to(device), gt['metadata'])
        if self.testmode: 
            print('Morph recon: rec.shape, binary.shape')
            print(rec.shape, binary.shape)
            #T.ToPILImage()(rec[3]).save('morph_recon.png')
            #T.ToPILImage()(binary[3]).save('morph_gt.png')
        return super().forward(rec, binary)
        
    def _binarize(self, img, meta): ## to be changed to a canny filter tuned based on compound used
        return img.mean(dim=1).unsqueeze(1)
    
class MorphReconLoss_SSIM(StructuralSimilarityIndexMeasure):
    def __init__(self, data_range=1.0, testmode=False):
        super().__init__(data_range=data_range)
        self.testmode = testmode
        
    def forward(self, rec, gt, device):
        self.to(device)
        binary = self._binarize(gt['image'].to(device), gt['metadata'])
        if self.testmode: 
            print('Morph recon: rec.shape, binary.shape')
            print(rec.shape, binary.shape)
            #T.ToPILImage()(rec[3]).save('morph_recon.png')
            #T.ToPILImage()(binary[3]).save('morph_gt.png')
        return 1-super().forward(rec, binary) ## ssim is between 0(worst) and 1(best)
        
    def _binarize(self, img, meta): ## to be changed to a canny filter tuned based on compound used
        return img.mean(dim=1).unsqueeze(1)