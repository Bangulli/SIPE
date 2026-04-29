import matplotlib.pyplot as plt, torch, torch.functional as F, json
from src.utils.stain_comp import compare
from src.model.archV2 import V2_H0_mini_for_Adversarial
from torchvision.transforms import ToPILImage, ToTensor
from src.utils.transfroms import UnNormalize, SobelTransform
from PIL import Image
import os, tqdm
from torch.utils.data import DataLoader
from BPTorch.utils import bptorch_collate
from BPTorch.datasets import BigPictureRepository, WsiDicomDataset

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
    ## setup instances of model and trainer
    with open('/home/lorenz/BigPicture/SIPE/classes.json', 'r') as f:
        classes = json.load(f)

    print('############################ Begin ############################')
    model = V2_H0_mini_for_Adversarial(classes, device='cuda:0')
    print(f'Training with the following {model.n_classes} class distribution')
 
    kwargs = WsiDicomDataset.get_default_kwargs()
    kwargs['transforms'] = model.transform
    ## load trainset and point to patch source
    trainset = BigPictureRepository('/mnt/nas6/data/BigPicture_CBIR/datasets/BPTorch/fold_1/BPR.json', load=True, wsidicomdataset_kwargs=kwargs, verbose=False) ## loading valset becuase the content gets overwritten by pointing to preextracted patches. this is just faster than loading the full training fold every time
    trainset.source_precomputed_patches_from('rnd-subsubset-50k')
    
    dl = DataLoader(trainset, 1, collate_fn=bptorch_collate)
    tr_s = SobelTransform(True)
    tr_n = SobelTransform(False)
    cnv = ToPILImage()
    
    for samp in dl:
        sobel = tr_n(samp)
        print(sobel.min(), sobel.max())
        cnv(sobel).save('sbl_n.png')
        sobel = tr_s(samp)
        print(sobel.min(), sobel.max())
        cnv(sobel).save('sbl_s.png')
        break
    
    # total_sum = 0.0
    # total_sum_sq = 0.0
    # total_pixels = 0

    # for samp in tqdm.tqdm(dl):
    #     sobel = tr_n(samp)
    #     total_sum += sobel.sum()
    #     total_sum_sq += (sobel ** 2).sum()
    #     total_pixels += 50176 # 224*224
            
    # mean = total_sum / total_pixels
    # std = (total_sum_sq / total_pixels - mean ** 2) ** 0.5
    
    # print(f"Sobel -> mean: {mean} & std: {std}") #Sobel -> mean: 0.9214555621147156 & std: 0.1259652078151703