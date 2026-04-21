from BPTorch.datasets import BigPictureRepository, WsiDicomDataset
from torch.utils.data import DataLoader
from BPTorch.utils import bptorch_collate
from pprint import pprint
from src.model.arch import H0_mini_for_Adversarial
from torchvision.transforms import ToPILImage
from src.utils.transfroms import UnNormalize
from src.trainer.trainer import Trainer
from src.utils.misc import make_name_from_list
import os, torch
import torch.nn.functional as F
import copy, tqdm, json
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
    ds = BigPictureRepository('/mnt/nas6/data/BigPicture_CBIR/datasets/BPTorch/fold_0/BPR.json', load=True, wsidicomdataset_kwargs=WsiDicomDataset.get_default_kwargs(), verbose=False)
    classes = {}
    for path in ['rnd-subset', 'rnd-subset-test', 'rnd-subset-val', 'rnd-subsubset-50k']:
        ds.source_precomputed_patches_from(path)
        for s in tqdm.tqdm(range(len(ds)), desc=path):
            name = make_name_from_list(ds[s]['metadata']['staining'])
            if name not in classes.keys(): classes[name]=1
            else: classes[name]+=1
    with open('/home/lorenz/BigPicture/SIPE/classes.json', 'w') as f:
        json.dump(classes, f, indent=2)

