from BPTorch.datasets import BigPictureRepository, WsiDicomDataset
from torch.utils.data import DataLoader
from BPTorch.utils import bptorch_collate
from pprint import pprint
from src.model.arch import H0_mini_for_Adversarial
from torchvision.transforms import ToPILImage
from src.utils.transfroms import UnNormalize
from src.trainer.trainer import Trainer
from src.trainer.curriculum_trainer import CurriculumTrainer, Curriculum
from src.losses.loss_fusion import SIPE_Loss_Adversarial, SIPE_Loss_Recon
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
    # #########################################################################################################################################
    # ## setup instances of model and trainer
    # model = H0_mini_for_InfoNCE(device='cuda:1')
    # trainer = Trainer(model, SIPE_Loss_InfoNCE(), wdir='SIPE-50k-InfoNCE', device=model.device)
    # kwargs = WsiDicomDataset.get_default_kwargs()
    # kwargs['transforms'] = model.transform

    # ## load trainset and point to patch source
    # trainset = BigPictureRepository('/mnt/nas6/data/BigPicture_CBIR/datasets/BPTorch/fold_1/BPR.json', load=True, wsidicomdataset_kwargs=kwargs, verbose=False) ## loading valset becuase the content gets overwritten by pointing to preextracted patches. this is just faster than loading the full training fold every time
    # trainset.source_precomputed_patches_from('rnd-subsubset-50k')
    
    # ## load valset and point to patch source
    # valset = BigPictureRepository('/mnt/nas6/data/BigPicture_CBIR/datasets/BPTorch/fold_1/BPR.json', load=True, wsidicomdataset_kwargs=kwargs, verbose=False)
    # valset.source_precomputed_patches_from('rnd-subset-val')
    
    # ## yeet
    # trainer.train(trainset, valset, epochs=3, batch_size=256, lr=3e-4, restarts=25)
    
    #########################################################################################################################################
    ## setup instances of model and trainer
    with open('/home/lorenz/BigPicture/SIPE/classes.json', 'r') as f:
        classes = json.load(f)
    print(f'Training with the following {len(classes)} class distribution')
    pprint(classes)
    print('############################ Begin ############################')
    model = H0_mini_for_Adversarial(classes, device='cuda:0')
    
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
    
    cr = Curriculum()
    cr.add_step('adverse', 5, 1.0, 3e-4, 5)
    # cr.add_step('recon', 1, 0, 3e-4, 1)
    # cr.add_step('adverse', 1, 0.1, 3e-4, 1)
    # cr.add_step('adverse', 1, 0.25, 3e-4, 1)
    # cr.add_step('recon', 1, 0, 3e-4, 1)
    # cr.add_step('adverse', 1, 0.5, 3e-4, 1)
    # cr.add_step('recon', 1, 0, 3e-4, 1)
    
    pretrainer = Trainer(model, SIPE_Loss_Recon(), wdir='SIPE-50k-Curriculum', device=model.device)
    cr_trainer = CurriculumTrainer(pretrainer.load_best_model(), SIPE_Loss_Recon(), SIPE_Loss_Adversarial(), wdir='SIPE-50k-Curriculum2', device=model.device)
    cr_trainer.train(trainset, valset, cr, batch_size=768)

    # trainer = CurriculumTrainer(model, SIPE_Loss_Recon(), SIPE_Loss_Adversarial(), wdir='SIPE-50k-Curriculum', device=model.device)
    # trainer.train(trainset, valset, total_steps=4,epochs_per_step=5, lr=3e-4, restarts=5, batch_size=512, delta_adverse_alpha=0.1)
    
    # ## pretrain reconstruction
    # trainer = Trainer(model, SIPE_Loss_Recon(), wdir='SIPE-1M-Adverse', device=model.device)
    # # trainer.train(trainset, valset, epochs=20, batch_size=256, lr=3e-4)
    
    # ## full send
    # pretrainer = Trainer(model, SIPE_Loss_Recon(), wdir='SIPE-50k-Curriculum', device=model.device)
# This portion of the code is setting up a training process using a `Trainer` object with a custom
# loss function `SIPE_Loss_Adversarial()` for training a model. Here's a breakdown of what each line
# is doing:
    # loss=SIPE_Loss_Adversarial()
    # trainer = Trainer(model, loss, wdir='SIPE-50k-Curriculum', device=model.device)
    # trainer.loss.set_adverse_alpha(1.0)
    # trainer.train(trainset, valset, epochs=10, batch_size=768, lr=3e-4, restarts=5)