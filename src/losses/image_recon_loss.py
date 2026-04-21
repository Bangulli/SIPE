######## Ecosystem ########
import os, sys, pathlib as pl
sys.path.append(os.path.join(os.path.dirname(__file__), "."))
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
######## External ########
import torch
import torch.nn as nn
import torchvision.transforms as T
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