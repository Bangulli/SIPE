from BPTorch.datasets import BigPictureRepository, WsiDicomDataset
from torch.utils.data import DataLoader
from BPTorch.utils import bptorch_collate
from src.utils.misc import make_name_from_list
from pprint import pprint
from src.model.arch import H0_mini_for_Adversarial
from torchvision.transforms import ToPILImage, RandomAffine, RandomGrayscale, RandomInvert, RandomErasing, GaussianBlur, Compose
from src.utils.transfroms import UnNormalize
from src.trainer.trainer import Trainer
from src.trainer.curriculum_trainer import CurriculumTrainer, Curriculum
from src.losses.loss_fusion import SIPE_Loss_Adversarial, SIPE_Loss_Adversarial_Cycle
import os, torch, shutil
import torch.nn.functional as F
import copy, tqdm, random, math, json
import matplotlib.pyplot as plt
import warnings
import torchvision.transforms as T
import numpy as np
warnings.filterwarnings('ignore')
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
    
def make_name_from_list(data):
    if isinstance(data, str):
        return data
    return "+".join(data)
    
if __name__ == '__main__':
    #########################################################################################################################################
    ## setup instances of model and trainer
    # with open('/home/lorenz/BigPicture/SIPE/classes.json', 'r') as f:
    #     classes = json.load(f)
    # with open('/home/lorenz/BigPicture/SIPE/organs.json', 'r') as f:
    #     organs = json.load(f)
    # with open('/home/lorenz/BigPicture/SIPE/paths.json', 'r') as f:
    #     paths = json.load(f)
    
    kwargs = WsiDicomDataset.get_default_kwargs()
    kwargs['transforms'] = None
    ## load trainset and point to patch source
    trainset = BigPictureRepository('/mnt/nas6/data/BigPicture_CBIR/datasets/BPTorch/fold_2/BPR.json', load=True, wsidicomdataset_kwargs=kwargs, verbose=False) ## loading valset becuase the content gets overwritten by pointing to preextracted patches. this is just faster than loading the full training fold every time
    
    organs = {}
    paths = {}
    
    for i in range(len(trainset)):
        smp = trainset[i]
        
        o = make_name_from_list(smp['metadata']['organ'])
        p = make_name_from_list(smp['metadata']['diagnosis'])
        
        if o not in organs.keys():
            organs[o]=1
        else: organs[o]+=1
        if p not in paths.keys():
            paths[p]=1
        else: paths[p]+=1
        
    with open('/home/lorenz/BigPicture/SIPE/organs.json', 'w') as f:
        json.dump(organs, f, indent=2)
        
    with open('/home/lorenz/BigPicture/SIPE/paths.json', 'w') as f:
        json.dump(paths, f, indent=2)