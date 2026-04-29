from BPTorch.datasets import BigPictureRepository, WsiDicomDataset
from torch.utils.data import DataLoader
from BPTorch.utils import bptorch_collate
from pprint import pprint
from src.model.archV2 import V2_H0_mini_for_Adversarial
from src.losses.loss_fusion import SIPE_Loss_Adversarial, SIPE_Loss_InfoNCE
from torchvision.transforms import ToPILImage
from src.utils.transfroms import UnNormalize
from src.utils.test import testV2
from src.utils.stain_comp import compareV2
from src.trainer.trainer import Trainer
import os, torch
import torch.nn.functional as F
import copy, tqdm, json, pathlib as pl
import matplotlib.pyplot as plt
# pip install "BPTorch @ git+https://github.com/Bangulli/BPTorch"
    
if __name__ == '__main__':
    
    sourcedir = pl.Path('/home/lorenz/BigPicture/SIPE/SIPE-50k-Recon')
    
    print(f'Running a quick and dirty test for trainer at {sourcedir}')
    
    with open('/home/lorenz/BigPicture/SIPE/classes.json', 'r') as f:
        classes = json.load(f)
    ## setup variables
    trainer = Trainer(V2_H0_mini_for_Adversarial(classes, device='cuda:0'), None, wdir=sourcedir)
    if (sourcedir/'history.json').exists(): model = trainer.load_best_model()
    else: model = trainer.load_model_at_epoch(1)
    
    model = trainer.load_model_at_epoch(-1)
    
    testV2(model, sourcedir, 'images')
    #compareV2(sourcedir, 'images')