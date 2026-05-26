from BPTorch.datasets import BigPictureRepository, WsiDicomDataset
from torch.utils.data import DataLoader
from BPTorch.utils import bptorch_collate
from pprint import pprint
from src.model.arch import H0_mini_for_Adversarial
from torchvision.transforms import ToPILImage
from src.utils.transfroms import UnNormalize
from src.utils.test import test
from src.utils.stain_comp import compare
from src.trainer.trainer import Trainer
import os, torch
import torch.nn.functional as F
import copy, tqdm, json, pathlib as pl
import matplotlib.pyplot as plt
# pip install "BPTorch @ git+https://github.com/Bangulli/BPTorch"
    
if __name__ == '__main__':
    
    sourcedir = pl.Path('/home/lorenz/BigPicture/SIPE/SIPE-50k-Curriculum-768')
    
    print(f'Running a quick and dirty test for trainer at {sourcedir}')
    
    with open('/home/lorenz/BigPicture/SIPE/classes.json', 'r') as f:
        classes = json.load(f)
    ## setup variables
    trainer = Trainer(H0_mini_for_Adversarial(classes, device='cuda:0'), None, wdir=sourcedir)
    
    ## vis best model in dir
    if (sourcedir/'history.json').exists(): model = trainer.load_best_model()
    else: model = trainer.load_model_at_epoch(1)
    test(model, sourcedir, 'images_best')
    compare(model, sourcedir, 'images_best')
    
    # ## vis latest
    # model = trainer.load_model_at_epoch(-1)
    # test(model, sourcedir, 'images_latest')
    # compare(model, sourcedir, 'images_latest')