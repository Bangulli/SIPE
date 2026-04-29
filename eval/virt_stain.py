## evaluate method for virtual staining

from pathlib import Path
import os, torch
import PIL
from tqdm import tqdm

def infer_and_eval_dir(model, output, dir='HER2'):
    output = Path(output)
    MIST = Path('/mnt/nas6/data/MIST')

    HandE = 'valA'
    other = 'valB'
    
    s_hande = torch.tensor() ## the stain vector signifying H&E
    s_her2 = torch.tensor() ## the stain vector signifying H&HER2

    images = os.listdir(MIST/dir/HandE)
    
    os.makedirs(output/dir/"he2other", exist_ok=True)
    os.makedirs(output/dir/"other2he", exist_ok=True)
    
    for img_id in tqdm(images, desc='Inference'):
        img_he = PIL.Image.open(MIST/dir/HandE/img_id)
        img_other = PIL.Image.open(MIST/dir/other/img_id)
        
        ################# inference step ##################
        ## HE 2 other
        for patch in img_he:
            pass ## infer patches
        ## fuse patches
        img_he2other = PIL.Image() ##dummy
        img_he2other.save(output/dir/"he2other"/img_id)
        
        ## other 2 HE
        for patch in img_other:
            pass ## infer patches
        ## fuse patches
        img_other2he = PIL.Image() ##dummy
        img_other2he.save(output/dir/"other2he"/img_id)

        
        ################# evaluation step ##################
        ## metrics from https://www.sciencedirect.com/science/article/pii/S2950261625000226
        
        ## image based
        PSNR = 0
        PCC = 0
        SSIM = 0
        MS_SSIM = 0
        
        ## deep feature based
        FID = 0
        DISTS = 0