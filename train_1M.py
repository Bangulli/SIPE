from sipe.model.arch import H0_mini_for_Adversarial
from sipe.losses.loss_fusion import SIPE_Loss_Adversarial, SIPE_Loss_Adversarial_Cycle
from sipe.trainer.trainer import Trainer
from sipe.trainer.curriculum_trainer import CurriculumTrainer, Curriculum
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
    
    ## setup trainer
    pretrainer = CurriculumTrainer(model, SIPE_Loss_Adversarial(recon_mode=True), SIPE_Loss_Adversarial(), SIPE_Loss_Adversarial_Cycle(), wdir='SIPE-1M-Curriculum', device='cuda:0')
    cr_trainer = CurriculumTrainer(pretrainer, SIPE_Loss_Adversarial(recon_mode=True), SIPE_Loss_Adversarial(), SIPE_Loss_Adversarial_Cycle(), wdir='SIPE-1M-Curriculum-BB_unfrozen', device=pretrainer.device)
    
    ## setup curriculum
    cr = Curriculum()
    cr.add_step(step_type='cycle', epochs=50, adverse_alpha=1.0, lr=1e-5, restarts=25, norm=True, freeze_bb=False, freeze_tangler=False)
    ## yeet
    cr_trainer.train(trainset, valset, cr, batch_size=128)
    
