from src.model.arch import H0_mini_for_AutoEncoding
from src.losses.loss_fusion import SIPE_Loss_InfoNCE
from src.trainer.trainer import Trainer
from BPTorch.datasets import BigPictureRepository, WsiDicomDataset
import torch,os
import torchvision.transforms as T
import time
import warnings
warnings.filterwarnings('ignore')


if __name__ == '__main__':
    ## setup instances of model and trainer
    model = H0_mini_for_AutoEncoding(device='cuda:0')
    trainer = Trainer(model, SIPE_Loss_InfoNCE(), wdir='SIPE-1M', device=model.device)
    
    ## prep trans
    kwargs = WsiDicomDataset.get_default_kwargs()
    kwargs['transforms'] = model.transform

    ## load trainset and point to patch source
    trainset = BigPictureRepository('/mnt/nas6/data/BigPicture_CBIR/datasets/BPTorch/fold_1/BPR.json', load=True, wsidicomdataset_kwargs=kwargs, verbose=False) ## loading valset becuase the content gets overwritten by pointing to preextracted patches. this is just faster than loading the full training fold every time
    trainset.source_precomputed_patches_from('rnd-subset')
    
    ## load valset and point to patch source
    valset = BigPictureRepository('/mnt/nas6/data/BigPicture_CBIR/datasets/BPTorch/fold_1/BPR.json', load=True, wsidicomdataset_kwargs=kwargs, verbose=False)
    valset.source_precomputed_patches_from('rnd-subset-val')
    
    ## yeet
    trainer.train(trainset, valset, epochs=100, batch_size=256, lr=3e-4, restarts=25)
    
