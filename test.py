from BPTorch.datasets import BigPictureRepository, WsiDicomDataset
from torch.utils.data import DataLoader
from BPTorch.utils import bptorch_collate
from pprint import pprint
from src.model.arch import H0_mini_for_Adversarial
from src.losses.loss_fusion import SIPE_Loss_Adversarial, SIPE_Loss_InfoNCE
from torchvision.transforms import ToPILImage
from src.utils.transfroms import UnNormalize
from src.utils.test import test
from src.trainer.trainer import Trainer
import os, torch
import torch.nn.functional as F
import copy, tqdm, json, pathlib as pl
import matplotlib.pyplot as plt
# pip install "BPTorch @ git+https://github.com/Bangulli/BPTorch"

def make_side_by_side(images, path):
    fig, ax = plt.subplots(6, 2, figsize=(6, 18))

    col_labels = ['image1', 'image2']
    row_labels = ['source', 'recon', 'reconmorph', 'reconO', 'reconrand', 'recon0']

    for i in range(2):
        key = f"image{i+1}"                          # fix: was hardcoded "image1"
        for j, v in enumerate(row_labels):
            ax[j, i].imshow(images[f"{key}_{v}"])    # fix: use .imshow() on the axes
            ax[j, i].set_xticks([])
            ax[j, i].set_yticks([])

            if i == 0:                               # row labels on the left column
                ax[j, i].set_ylabel(v, fontsize=10, rotation=0, labelpad=60, va='center')
            if j == 0:                               # column labels on the top row
                ax[j, i].set_title(col_labels[i], fontsize=12)

    fig.tight_layout()
    fig.savefig(path)
    
if __name__ == '__main__':
    sourcedir = pl.Path('/home/lorenz/BigPicture/SIPE/SIPE-1M-Curriculum')
    
    with open('/home/lorenz/BigPicture/SIPE/classes.json', 'r') as f:
        classes = json.load(f)
    ## setup variables
    trainer = Trainer(H0_mini_for_Adversarial(classes, device='cuda:0'), None, wdir=sourcedir)
    model = trainer.load_best_model()
    test(model, sourcedir, 'images')