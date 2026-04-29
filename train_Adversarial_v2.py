from src.model.archV2 import V2_H0_mini_for_Adversarial
from src.losses.loss_fusion import V2_SIPE_Loss_Adversarial
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
    model = V2_H0_mini_for_Adversarial(classes, device='cuda:0')
    
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
    
    ## setup pretrainer
    pretrainer = Trainer(model, V2_SIPE_Loss_Adversarial(True), wdir='SIPE-50k-Recon', device=model.device)
    #pretrainer.train(trainset, valset, 10, lr=1e-4, restarts=10, batch_size=768)
    
    ## setup trainer
    cr_trainer = CurriculumTrainer(pretrainer.load_best_model(), V2_SIPE_Loss_Adversarial(True), V2_SIPE_Loss_Adversarial(), wdir='SIPE-1M-Curriculum', device=model.device)
    ## setup curriculum
    cr = Curriculum()
    alpha_ramp = np.arange(0.1, 1.0, 0.1).tolist()
    alpha_ramp += (15-len(alpha_ramp))*[1.0]
    cr.add_step(step_type='adverse', epochs=15, adverse_alpha=alpha_ramp, lr=3e-4, restarts=5, norm=False, freeze_bb=True)
    cr.add_step(step_type='recon', epochs=5, adverse_alpha=0, lr=3e-4, restarts=5, norm=False, freeze_bb=True)
    ## yeet
    cr_trainer.train(trainset, valset, cr, batch_size=768)
    
