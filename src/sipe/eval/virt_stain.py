## evaluate method for virtual staining
import os, torch, pathlib as pl, numpy as np, sys, json
sys.path.append(os.path.join(os.path.dirname(__file__), "."))
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
from pathlib import Path
import os, torch
import PIL
from tqdm import tqdm
from torchvision.transforms import ToTensor, Resize, Compose, Normalize, ToPILImage, CenterCrop
from src.utils.transfroms import UnNormalize
from skimage.metrics import structural_similarity as ssim
from pytorch_msssim import ms_ssim
import torch
from pprint import pprint
from src.model.arch import H0_mini_for_Adversarial
from src.trainer.trainer import Trainer
from huggingface_hub import login
from scipy.stats import pearsonr

def swap_stain(model, ten_aa, ten_bb, base_transforms, to_regsized_pil, s_a=None, s_b=None):
    out_ab = torch.zeros_like(ten_aa)
    out_ba = torch.zeros_like(ten_bb)
    
    for w in range(0, 224*5, 224):
        for h in range(0, 224*5, 224):
            patch_aa = base_transforms(ten_aa[:, h:h+224, w:w+224])
            patch_bb = base_transforms(ten_bb[:, h:h+224, w:w+224])
            
            batch = {'image': torch.stack([patch_aa, patch_bb])}
            
            s, z = model(batch)
            
            if s_a is None and s_b is None:
                out_ab[:, h:h+224, w:w+224] = model.recon_image(s[0].unsqueeze(0), z[1].unsqueeze(0))
                out_ba[:, h:h+224, w:w+224] = model.recon_image(s[1].unsqueeze(0), z[0].unsqueeze(0))
            else:
                out_ab[:, h:h+224, w:w+224] = model.recon_image(s_a.unsqueeze(0), z[1].unsqueeze(0))
                out_ba[:, h:h+224, w:w+224] = model.recon_image(s_b.unsqueeze(0), z[0].unsqueeze(0))
    
    return to_regsized_pil(out_ab), to_regsized_pil(out_ba)

def get_metrics(img1, img2):
    arr1 = np.array(img1, dtype=np.float64)
    arr2 = np.array(img2, dtype=np.float64)

    # --- PSNR ---
    mse = np.mean((arr1 - arr2) ** 2)
    if mse == 0:
        psnr = float("inf")
    else:
        max_val = 255.0 if arr1.max()>1.0 else 1.0
        psnr = 20 * np.log10(max_val / np.sqrt(mse))

    # --- PCC (Pearson Correlation Coefficient) ---
    flat1, flat2 = arr1.flatten(), arr2.flatten()
    pcc = pearsonr(flat1, flat2).statistic

    # --- SSIM ---
    ssim_val = ssim(
        arr1,
        arr2,
        data_range=arr2.max() - arr2.min(),
        channel_axis=-1
    )

    # # --- MS-SSIM (requires torch) ---
    # def to_tensor(arr):
    #     t = torch.from_numpy(arr.astype(np.float32))
    #     if t.ndim == 2:             # grayscale → (1, 1, H, W)
    #         t = t.unsqueeze(0).unsqueeze(0)
    #     else:                       # RGB → (1, C, H, W)
    #         t = t.permute(2, 0, 1).unsqueeze(0)
    #     return t

    # t1, t2 = to_tensor(arr1), to_tensor(arr2)
    # ms_ssim_val = ms_ssim(t1, t2, data_range=255.0, size_average=True).item()

    return {
        "PSNR":    float(psnr),
        "PCC":     float(pcc),
        "SSIM":    float(ssim_val),
        "MS_SSIM": 0,
    }
    
def mean(lst):
    return sum(lst)/len(lst)

def infer_and_eval_dir(model, output, dir='HER2'):
    dir = f"{dir}/TrainValAB"
    output = Path(output)
    MIST = Path('/mnt/nas6/data/MIST')
    base_transforms = model.transform

    HandE = 'valA'
    other = 'valB'

    images = os.listdir(MIST/dir/HandE)
    
    ### Get target stain vectors from training set
    img_id = sorted(os.listdir(MIST/dir/'trainA'))[0]
    to_tensor = Compose([ToTensor(), CenterCrop(224)])
    img_a = base_transforms(to_tensor(PIL.Image.open(MIST/dir/'trainA'/img_id)))
    img_b = base_transforms(to_tensor(PIL.Image.open(MIST/dir/'trainB'/img_id)))
    s_a, _ = model({'image': img_a.unsqueeze(0).to(model.device)})
    s_b, _ = model({'image': img_b.unsqueeze(0).to(model.device)})

    os.makedirs(output/dir.removesuffix('/TrainValAB')/"he2other", exist_ok=True)
    os.makedirs(output/dir.removesuffix('/TrainValAB')/"other2he", exist_ok=True)
    
    results_other2he = {
        'MS_SSIM': [],
        'PCC': [],
        'PSNR': [],
        'SSIM': []
    }
    results_he2other = {
        'MS_SSIM': [],
        'PCC': [],
        'PSNR': [],
        'SSIM': []
    }
    
    to_resized_tensor = Compose([ToTensor(), Resize(5*224)])
    to_regsized_pil = Compose([UnNormalize([0.707223,0.578729,0.703617], [0.211883,0.230117,0.177517]), Resize(1024), ToPILImage()])
    
    for img_id in tqdm(images, desc='Inference'):
        img_he = PIL.Image.open(MIST/dir/HandE/img_id)
        img_other = PIL.Image.open(MIST/dir/other/img_id)
        
        ## fuse patches
        img_other2he, img_he2other = swap_stain(model, to_resized_tensor(img_he), to_resized_tensor(img_other), base_transforms, to_regsized_pil,None,None)#, s_a, s_b)
        img_he2other.save(output/dir.removesuffix('/TrainValAB')/"he2other"/img_id)
        img_other2he.save(output/dir.removesuffix('/TrainValAB')/"other2he"/img_id)

        
        ################# evaluation step ##################
        ## metrics from https://www.sciencedirect.com/science/article/pii/S2950261625000226
        eval_other2he = get_metrics(img_he, img_other2he)
        eval_he2other = get_metrics(img_other, img_he2other)
        
        #print(eval_other2he, eval_he2other)
        
        for k in results_other2he.keys():
            results_other2he[k].append(eval_other2he[k])
            results_he2other[k].append(eval_he2other[k])
    
    for k in results_other2he.keys():
        results_other2he[k]=mean(results_other2he[k])
        results_he2other[k]=mean(results_he2other[k])
        
    return results_he2other, results_other2he

def test_virtual_staining(model_dir):
    model_dir = Path(model_dir)
    with open('/home/lorenz/BigPicture/SIPE/classes.json', 'r') as f:
        classes = json.load(f)
    trainer = Trainer(H0_mini_for_Adversarial(classes, device='cuda:0'), None, wdir=model_dir, device='cuda:0')
    model = trainer.load_best_model()
    
    for stain in ['HER2']:#, 'ER', 'Ki67', 'PR']:
        results_he2other, results_other2he = infer_and_eval_dir(model, model_dir/'virtual-staining', stain)
        
        print('######################## RESULTS HE2OTHER ########################')
        pprint(results_he2other)
        
        print('######################## RESULTS OTHER2HE ########################')
        pprint(results_other2he)
        
        with open(model_dir/'virtual-staining'/stain/'results_he2other.json', 'w') as f:
            json.dump(results_he2other, f, indent=4)
            
        with open(model_dir/'virtual-staining'/stain/'results_other2he.json', 'w') as f:
            json.dump(results_other2he, f, indent=4)
    
if __name__ == '__main__':
    test_virtual_staining('/home/lorenz/BigPicture/SIPE/SIPE-1M-Curriculum')