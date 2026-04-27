from src.model.arch import H0_mini_for_Adversarial
from src.losses.loss_fusion import SIPE_Loss_Adversarial, SIPE_Loss_Recon
from src.trainer.trainer import Trainer
from src.trainer.curriculum_trainer import CurriculumTrainer, Curriculum
from BPTorch.datasets import BigPictureRepository, WsiDicomDataset
import torch,os
import torchvision.transforms as T
import time
import warnings, json
import numpy as np
warnings.filterwarnings('ignore')


if __name__ == '__main__':
    ## setup instances of model and trainer
    with open('/home/lorenz/BigPicture/SIPE/classes.json', 'r') as f:
        classes = json.load(f)
        
    ## setup model
    model = H0_mini_for_Adversarial(classes, device='cuda:0')
    trainer = Trainer(model, SIPE_Loss_Adversarial(), wdir='SIPE-50k-ProjRecon', device=model.device)
    model = trainer.load_best_model()
    
    ## setup trainer
    cr_trainer = CurriculumTrainer(model, SIPE_Loss_Recon(), SIPE_Loss_Adversarial(), wdir='SIPE-1M-Curriculum', device=model.device)
    
    ## setup curriculum
    cr = Curriculum()
    cr.add_step(step_type='recon', epochs=5, adverse_alpha=1.0, lr=1e-5, restarts=5, norm=False, freeze_bb=True)
    
    ## prep train trans
    kwargs = WsiDicomDataset.get_default_kwargs()
    kwargs['transforms'] = model.transform#T.Compose([model.transform, T.RandomApply([T.RandomCrop(224)],p=0.5), T.RandomApply([T.ColorJitter(0.5, 0.5, 0.5, 0.1)], p=0.5), T.GaussianBlur(9)])

    ## load trainset and point to patch source
    trainset = BigPictureRepository('/mnt/nas6/data/BigPicture_CBIR/datasets/BPTorch/fold_1/BPR.json', load=True, wsidicomdataset_kwargs=kwargs, verbose=False) ## loading valset becuase the content gets overwritten by pointing to preextracted patches. this is just faster than loading the full training fold every time
    trainset.source_precomputed_patches_from('rnd-subset')
    
    ## prep val trans
    kwargs = WsiDicomDataset.get_default_kwargs()
    kwargs['transforms'] = model.transform
    
    ## load valset and point to patch source
    valset = BigPictureRepository('/mnt/nas6/data/BigPicture_CBIR/datasets/BPTorch/fold_1/BPR.json', load=True, wsidicomdataset_kwargs=kwargs, verbose=False)
    valset.source_precomputed_patches_from('rnd-subset-val')
    
    ## yeet
    cr_trainer.train(trainset, valset, cr, batch_size=768)
    
