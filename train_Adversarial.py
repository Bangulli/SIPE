from src.model.arch import H0_mini_for_Adversarial
from src.losses.loss_fusion import SIPE_Loss_Adversarial, SIPE_Loss_Adversarial_Cycle
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
    model = H0_mini_for_Adversarial(classes, device='cuda:1')
        
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
    cr_trainer = CurriculumTrainer(model, SIPE_Loss_Adversarial(recon_mode=True), SIPE_Loss_Adversarial(), SIPE_Loss_Adversarial_Cycle(), wdir='SIPE-1M-Curriculum', device=model.device)
    
    ## setup curriculum
    cr = Curriculum()
    alpha_ramp = np.arange(0.05, 1.0, 0.05).tolist()
    alpha_ramp += (20-len(alpha_ramp))*[1.0]
    cr.add_step(step_type='recon', epochs=5, adverse_alpha=1.0, lr=1e-3, restarts=5, norm=True, freeze_bb=True, freeze_tangler=False)
    cr.add_step(step_type='adverse', epochs=20, adverse_alpha=alpha_ramp, lr=1e-3, restarts=10, norm=False, freeze_bb=True, freeze_tangler=False)
    cr.add_step(step_type='cycle', epochs=20, adverse_alpha=1.0, lr=1e-3, restarts=10, norm=True, freeze_bb=True, freeze_tangler=False)
    cr.add_step(step_type='recon', epochs=5, adverse_alpha=1.0, lr=3e-4, restarts=5, norm=True, freeze_bb=True, freeze_tangler=True)
    cr.add_step(step_type='cycle', epochs=20, adverse_alpha=1.0, lr=3e-4, restarts=10, norm=True, freeze_bb=True, freeze_tangler=False)
    cr.add_step(step_type='recon', epochs=5, adverse_alpha=1.0, lr=1e-4, restarts=5, norm=True, freeze_bb=True, freeze_tangler=True)
    ## yeet
    cr_trainer.train(trainset, valset, cr, batch_size=256)
    
