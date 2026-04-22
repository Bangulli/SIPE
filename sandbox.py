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
from src.model.projector import MixedDisentangler
import numpy as np
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

def make_sobel_for_batch(batch):
    sobel_x = torch.tensor([
            [-1., 0., 1.],
            [-2., 0., 2.],
            [-1., 0., 1.]
        ]).view(1, 1, 3, 3)
    sobel_y = torch.tensor([
            [-1., -2., -1.],
            [ 0.,  0.,  0.],
            [ 1.,  2.,  1.]
        ]).view(1, 1, 3, 3)
    batch = batch.mean(dim=1).unsqueeze(1)
    gx = F.conv2d(F.pad(batch, (1, 1, 1, 1), mode='reflect'), sobel_x)
    gy = F.conv2d(F.pad(batch, (1, 1, 1, 1), mode='reflect'), sobel_y)
    magnitude = torch.sqrt(gx ** 2 + gy ** 2)
    return magnitude

def extend_label_map(label_map, labels, sobel):
    means = sobel.mean(dim=(1, 2, 3)).to('cpu')
    for k, v in zip(labels, means.tolist()):
        label_map[k].append(v)
    return label_map

    
    
if __name__ == '__main__':
    with open('/home/lorenz/BigPicture/SIPE/classes.json', 'r') as f:
        classes = json.load(f)
    model = H0_mini_for_Adversarial(classes, device='cuda:0')
    kwargs = WsiDicomDataset.get_default_kwargs()
    kwargs['transforms'] = model.transform
    trainset = BigPictureRepository('/mnt/nas6/data/BigPicture_CBIR/datasets/BPTorch/fold_1/BPR.json', load=True, wsidicomdataset_kwargs=kwargs, verbose=False) ## loading valset becuase the content gets overwritten by pointing to preextracted patches. this is just faster than loading the full training fold every time
    trainset.source_precomputed_patches_from('rnd-subset')
    dl = DataLoader(trainset, batch_size=256, collate_fn=bptorch_collate)
    
    values = {k:[] for k in classes.keys()}
    for batch in tqdm.tqdm(dl, desc='computing mean and stds'):
        sobel = make_sobel_for_batch(batch['image'])
        labels = model.parse_labels(batch)
        values = extend_label_map(values, labels, sobel)
        
    means_stds = {k:{'mean':np.mean(v), 'std':np.std(v)} for k, v in values.items()}
    with open('sobel_cfg.json', 'w') as file:
        json.dump(means_stds, file, indent=4)