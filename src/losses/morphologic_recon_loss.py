######## Ecosystem ########
import os, sys, pathlib as pl, json
sys.path.append(os.path.join(os.path.dirname(__file__), "."))
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
from operator import itemgetter
from src.utils.transfroms import UnNormalizeFloats
######## External ########
import torch
from torchmetrics.image import StructuralSimilarityIndexMeasure
import torch.nn as nn
import torchvision.transforms as T
import torch.nn.functional as F
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
        gt = gt['image'].to(device) if type(gt)==dict else gt
        binary = self._binarize(gt)
        if self.testmode: 
            print('Morph recon: rec.shape, binary.shape')
            print(rec.shape, binary.shape)
            #T.ToPILImage()(rec[3]).save('morph_recon.png')
            #T.ToPILImage()(binary[3]).save('morph_gt.png')
        return 1-super().forward(rec, binary) ## ssim is between 0(worst) and 1(best)
        
    def _binarize(self, img): ## to be changed to a canny filter tuned based on compound used
        return img.mean(dim=1).unsqueeze(1)
    
class MorphReconLoss_SSIM_Sobel(StructuralSimilarityIndexMeasure):
    def __init__(self, data_range=1.0, testmode=False, normalize=False):
        super().__init__(data_range=data_range)
        self.testmode = testmode
        self.normalize = normalize
        self.denormer = UnNormalizeFloats([
            0.707223,
            0.578729,
            0.703617
        ], [
            0.211883,
            0.230117,
            0.177517
        ])
        self.sobel_x = torch.tensor([
            [-1., 0., 1.],
            [-2., 0., 2.],
            [-1., 0., 1.]
        ]).view(1, 1, 3, 3)
        self.sobel_y = torch.tensor([
            [-1., -2., -1.],
            [ 0.,  0.,  0.],
            [ 1.,  2.,  1.]
        ]).view(1, 1, 3, 3)
        
    def forward(self, rec, gt, device):
        self.to(device)
        gt = gt['image'].to(device) if type(gt)==dict else gt.to(device)
        binary = self._binarize(gt)
        sobel = self._sobel(binary).to(rec.device)
        if rec.shape[1]>1: sobel=sobel.expand(-1, 3, -1, -1)
        if self.testmode: 
            print('Morph recon: rec.shape, binary.shape')
            print(rec.shape, sobel.shape)
        return 1-super().forward(rec, sobel) ## ssim is between 0(worst) and 1(best)
        
    def _binarize(self, img): ## to be changed to a canny filter tuned based on compound used
        return self.denormer(img).mean(dim=1).unsqueeze(1)
    
    def _sobel(self, img):
        gx = F.conv2d(F.pad(img, (1, 1, 1, 1), mode='reflect'), self.sobel_x.to(img.device))
        gy = F.conv2d(F.pad(img, (1, 1, 1, 1), mode='reflect'), self.sobel_y.to(img.device))
        magnitude = torch.sqrt(gx ** 2 + gy ** 2)
        magnitude = magnitude / magnitude.max().clamp(min=1e-8)
        return magnitude
