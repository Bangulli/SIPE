import matplotlib.pyplot as plt, torch, torch.functional as F
from src.utils.stain_comp import compare
from torchvision.transforms import ToPILImage, ToTensor
from src.utils.transfroms import UnNormalize
from PIL import Image
import os
import copy
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

def _roll_list(lst, shifts):
        if shifts > 0:
            for idx in range(shifts):
                lst.insert(0, lst.pop(-1))
        else:
            for idx in range(abs(shifts)):
                lst.append(lst.pop(0))
        return lst
    
if __name__ == '__main__': 
    print('+++++++++++++++++++++ LIST ROLL +++++++++++++++++++++')
    foo = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
    print(f"Orig -> {foo}")
    for shft in range(1, 6, 1):
        f1 = _roll_list(copy.deepcopy(foo), shft)
        print(f"Shifts={shft} -> {f1}")
        f0 = _roll_list(copy.deepcopy(f1), -shft)
        print(f"Shifts={-shft} -> {f0}")
    
    print('++++++++++++++++++++ TENSOR ROLL ++++++++++++++++++++') 
    foo = torch.tensor([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
    print(f"Orig -> {foo}")
    for shft in range(1, 6, 1):
        f1 = torch.roll(copy.deepcopy(foo), shft, 0)
        print(f"Shifts={shft} -> {f1}")
        f0 = torch.roll(copy.deepcopy(f1), -shft, 0)
        print(f"Shifts={-shft} -> {f0}")