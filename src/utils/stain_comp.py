from BPTorch.datasets import BigPictureRepository, WsiDicomDataset
from torch.utils.data import DataLoader
from BPTorch.utils import bptorch_collate
from pprint import pprint
from src.model.arch import H0_mini_for_Adversarial
from torchvision.transforms import ToPILImage
from src.utils.transfroms import UnNormalize, SobelTransform
from src.trainer.trainer import Trainer
from src.utils.misc import patch_is_foreground
import os, torch
import torch.nn.functional as F
import copy, tqdm, json, pathlib as pl
import matplotlib.pyplot as plt
import PIL
import os
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
# pip install "BPTorch @ git+https://github.com/Bangulli/BPTorch"

def make_side_by_side(images, path):
   
        
    fig, ax = plt.subplots(7, 2, figsize=(6, 18))

    col_labels = ['image1', 'image2']
    row_labels = ['source', 'recon', 'sobel', 'reconmorph', 'reconO', 'reconrand', 'recon0']

    for i in range(2):
        key = f"image{i+1}"                          # fix: was hardcoded "image1"
        for j, v in enumerate(row_labels):
            if not (("morph" in v) or ("sobel" in v)): ax[j, i].imshow(images[f"{key}_{v}"])    # fix: use .imshow() on the axes
            else: ax[j, i].imshow(images[f"{key}_{v}"], cmap='Grays')
            ax[j, i].set_xticks([])
            ax[j, i].set_yticks([])

            if i == 0:                               # row labels on the left column
                ax[j, i].set_ylabel(v, fontsize=10, rotation=0, labelpad=60, va='center')
            if j == 0:                               # column labels on the top row
                ax[j, i].set_title(col_labels[i], fontsize=12)

    fig.tight_layout()
    fig.savefig(path)
    fig.clf()
    
def make_diagonal(images, originals, outdir):
    keys = list(images.keys())
    orig_keys = list(originals.keys())
    N = len(keys)

    dpi = 100
    cell_px = 224
    cell_in = cell_px / dpi          # 2.24 inches per image cell
    title_in = 0.35                   # thin header row for labels

    fig_w = (1 + N) * cell_in         # source col + N reconstruction cols
    fig_h = N * cell_in + title_in

    fig = plt.figure(figsize=(fig_w, fig_h), dpi=dpi)

    # One title row (thin) + N image rows; source col + N recon cols
    gs = gridspec.GridSpec(
        N + 1, 1 + N,
        figure=fig,
        height_ratios=[title_in] + [cell_in] * N,   # proportional — title is short
        left=0, right=1, top=1, bottom=0,
        wspace=0, hspace=0,
    )

    # ── header labels ────────────────────────────────────────────────────────
    ax_src_hdr = fig.add_subplot(gs[0, 0])
    ax_src_hdr.text(0.5, 0.5, "sources",
                    ha="center", va="center", fontsize=9, fontweight="bold")
    ax_src_hdr.axis("off")

    ax_rec_hdr = fig.add_subplot(gs[0, 1:])
    ax_rec_hdr.text(0.5, 0.5, "reconstructions",
                    ha="center", va="center", fontsize=9, fontweight="bold")
    ax_rec_hdr.axis("off")

    # ── image grid ───────────────────────────────────────────────────────────
    for i in tqdm.tqdm(range(N), desc='visualizing'):
        # left column — source image
        ax = fig.add_subplot(gs[i + 1, 0])
        ax.imshow(originals[orig_keys[i]])
        ax.axis("off")

        # NxN reconstruction block
        for j in range(N):
            ax = fig.add_subplot(gs[i + 1, j + 1])
            ax.imshow(images[keys[i]][keys[j]])
            ax.axis("off")

    # ── save ─────────────────────────────────────────────────────────────────
    os.makedirs(outdir, exist_ok=True)
    fig.savefig(
        os.path.join(outdir, "dense_side_by_side.png"),
        dpi=dpi,
        pad_inches=0,       # no bbox_inches='tight' — preserves exact cell sizes
    )
    plt.close(fig)
    
def compare(model, sourcedir, imdir_name):
    with open('classes.json', 'r') as f:
        classes = json.load(f)
        
    ## some io
    sourcedir = pl.Path(sourcedir)
    model.to(model.device)
    model.eval()
    kwargs = WsiDicomDataset.get_default_kwargs()
    kwargs['transforms'] = model.transform
    
    ## defrag classes
    classes = model.defrag(list(classes.keys()))
    
    ## dataset and loader
    ds = BigPictureRepository('/mnt/nas6/data/BigPicture_CBIR/datasets/BPTorch/fold_0/BPR.json', load=True, wsidicomdataset_kwargs=kwargs, verbose=False)
    ds.source_precomputed_patches_from('rnd-subset-test')
    print(f"Dataset contains {len(ds)} foreground patches")
    dl = DataLoader(ds, 1, True, collate_fn=bptorch_collate)
    
    ## transforms for visu
    converter = ToPILImage()
    denormer = UnNormalize([
        0.707223,
        0.578729,
        0.703617
    ], [
        0.211883,
        0.230117,
        0.177517
    ])
    
    present_classes = []
    virtual_staining_heads = {}
    morphologic_bases = {}
    source_images = {}
    for batch in tqdm.tqdm(dl, desc='Finding stain examples'):
        if not patch_is_foreground(converter(batch['image'].squeeze(0))):continue
        label = model.defrag(batch['metadata'][0]['staining'])[0][0]
        if label in present_classes: continue
        
        ## encode new image
        s_proba, z = model(batch)
        ## save stuff
        img = converter(denormer(batch['image'].squeeze(0)))
        source_images[label] = img
        virtual_staining_heads[label] = s_proba
        morphologic_bases[label] = z
        present_classes.append(label)
    
    recons = {}
    for lbl1 in tqdm.tqdm(present_classes, desc='generating'):
        current_recons = {}
        for lbl2 in present_classes:
            s = virtual_staining_heads[lbl2]
            z = morphologic_bases[lbl1]
            current_recons[lbl2] = model.recon_image_PIL(s, z, denormer)
        recons[lbl1] = current_recons
        
    make_diagonal(recons, source_images, sourcedir/imdir_name)
    