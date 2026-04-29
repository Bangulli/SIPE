from BPTorch.datasets import BigPictureRepository, WsiDicomDataset
from torch.utils.data import DataLoader
from BPTorch.utils import bptorch_collate
from pprint import pprint
from src.model.archV2 import V2_H0_mini_for_Adversarial
from torchvision.transforms import ToPILImage
from src.utils.transfroms import UnNormalize
from src.trainer.trainer import Trainer
from src.trainer.curriculum_trainer import CurriculumTrainer, Curriculum
from src.losses.loss_fusion import V2_SIPE_Loss_Adversarial
import os, torch, shutil
import torch.nn.functional as F
import copy, tqdm, random, math, json
import matplotlib.pyplot as plt
import warnings
import torchvision.transforms as T
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
    with open('/home/lorenz/BigPicture/SIPE/classes.json', 'r') as f:
        classes = json.load(f)

    print('############################ Begin ############################')
    model = V2_H0_mini_for_Adversarial(classes, device='cuda:0')
    print(f'Training with the following {model.n_classes} class distribution')
    pprint(model.enc.classes_)
    
    kwargs = WsiDicomDataset.get_default_kwargs()
    kwargs['transforms'] = model.transform
    ## load trainset and point to patch source
    trainset = BigPictureRepository('/mnt/nas6/data/BigPicture_CBIR/datasets/BPTorch/fold_1/BPR.json', load=True, wsidicomdataset_kwargs=kwargs, verbose=False) ## loading valset becuase the content gets overwritten by pointing to preextracted patches. this is just faster than loading the full training fold every time
    trainset.source_precomputed_patches_from('rnd-subsubset-50k')
    
    kwargs = WsiDicomDataset.get_default_kwargs()
    kwargs['transforms'] = model.transform
    ## load valset and point to patch source
    valset = BigPictureRepository('/mnt/nas6/data/BigPicture_CBIR/datasets/BPTorch/fold_1/BPR.json', load=True, wsidicomdataset_kwargs=kwargs, verbose=False)
    valset.source_precomputed_patches_from('rnd-subset-val')
    
    ## setup and run reconstruction pretrainer
    model.freeze_backbone(True)
    pretrainer = Trainer(model, V2_SIPE_Loss_Adversarial(True), wdir='SIPE-50k-Recon', device=model.device)
    pretrainer.train(trainset, valset, 20, 3e-4, 25, batch_size=768)


    ## setup curriculum
    cr = Curriculum()
    cr.add_step('adverse', 10, 0.1, 3e-4, 10, True, True)
    cr.add_step('adverse', 10, 0.25, 3e-4, 10, True, True)
    cr.add_step('recon', 5, 0, 3e-4, 5, True, True)
    cr.add_step('adverse', 10, 0.5, 3e-4, 10, True, True)
    cr.add_step('adverse', 10, 1, 3e-4, 10, True, True)
    
    ## setup and run curriculum trainer
    cr_trainer = CurriculumTrainer(model, V2_SIPE_Loss_Adversarial(True), V2_SIPE_Loss_Adversarial(), 'SIPE-50k-Curriculum', device=model.device)
    cr_trainer.train(trainset, valset, cr, batch_size=768)