from BPTorch.datasets import BigPictureRepository, WsiDicomDataset
from torchvision.transforms import ToPILImage
from src.utils.transfroms import UnNormalize, SobelTransform
import os, torch
import copy, tqdm, json, pathlib as pl
import matplotlib.pyplot as plt
import PIL
# pip install "BPTorch @ git+https://github.com/Bangulli/BPTorch"

def make_side_by_side(images, path):
    fig, ax = plt.subplots(7, 2, figsize=(6, 18))

    col_labels = ['image1', 'image2']
    row_labels = ['source', 'recon', 'sobel', 'reconmorph', 'reconO', 'reconrand']

    for i in range(2):
        key = f"image{i+1}"                          # fix: was hardcoded "image1"
        for j, v in enumerate(row_labels):
            if not ("sobel" in v): ax[j, i].imshow(images[f"{key}_{v}"])    # fix: use .imshow() on the axes
            else: ax[j, i].imshow(images[f"{key}_{v}"], cmap='gray')
            ax[j, i].set_xticks([])
            ax[j, i].set_yticks([])

            if i == 0:                               # row labels on the left column
                ax[j, i].set_ylabel(v, fontsize=10, rotation=0, labelpad=60, va='center')
            if j == 0:                               # column labels on the top row
                ax[j, i].set_title(col_labels[i], fontsize=12)

    fig.tight_layout()
    fig.savefig(path)
    fig.clf()
    
def test(model, sourcedir, imdir_name):
    sourcedir = pl.Path(sourcedir)
    if not (sourcedir/imdir_name).exists(): os.mkdir(sourcedir/imdir_name)
    with open('/home/lorenz/BigPicture/SIPE/classes.json', 'r') as f:
        classes = json.load(f)
    ## setup variables
    model.to(model.device)
    model.eval()
    kwargs = WsiDicomDataset.get_default_kwargs()
    kwargs['transforms'] = model.transform

        
    ds = BigPictureRepository('/mnt/nas6/data/BigPicture_CBIR/datasets/BPTorch/fold_0/BPR.json', load=True, wsidicomdataset_kwargs=kwargs, verbose=False)
    print(f"Dataset contains {len(ds)} foreground patches")
    
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
    sobeler = SobelTransform()
    
    image_dict = {}
    
    ### image one
    patch = ds[512]
    img = converter(denormer(patch['image']))
    img.save(f'{sourcedir/imdir_name}/source1.png')
    image_dict['image1_source'] = copy.deepcopy(img)
    
    patch['image'] = patch['image'].unsqueeze(0)
    patch['metadata'] = [patch['metadata']]
    img = converter(sobeler(patch))
    img.save(f'{sourcedir/imdir_name}/sobel1.png')
    image_dict['image1_sobel'] = img
    
    s1, z1 = model(patch)
    
    print('Image 1')
    print(patch['metadata'])
    img = converter(denormer(model.recon_image(s1, z1)))
    img.save(f'{sourcedir/imdir_name}/recon_img1.png')
    image_dict['image1_recon'] = copy.deepcopy(img)
    
    img = model.recon_image_PIL(torch.zeros_like(s1), z1, denormer)
    img.save(f'{sourcedir/imdir_name}/recon_morph1.png')
    image_dict['image1_reconmorph'] = copy.deepcopy(img)
    
    ### image two
    patch = ds[101028]
    img = converter(denormer(patch['image']))
    img.save(f'{sourcedir/imdir_name}/source2.png')
    image_dict['image2_source'] = copy.deepcopy(img)
    
    patch['image'] = patch['image'].unsqueeze(0)
    patch['metadata'] = [patch['metadata']]
    img = converter(sobeler(patch))
    img.save(f'{sourcedir/imdir_name}/sobel2.png')
    image_dict['image2_sobel'] = img
    
    s2, z2 = model(patch)
    
    print('Image 2')
    print(patch['metadata'])
    img = model.recon_image_PIL(s2, z2, denormer)
    img.save(f'{sourcedir/imdir_name}/recon_img2.png')
    
    image_dict['image2_recon'] = copy.deepcopy(img)
    img = model.recon_image_PIL(torch.zeros_like(s2), z2, denormer)
    img.save(f'{sourcedir/imdir_name}/recon_morph2.png')
    image_dict['image2_reconmorph'] = copy.deepcopy(img)
    
    ### image two with image 1 stain
    img = model.recon_image_PIL(s1, z2, denormer)
    img.save(f'{sourcedir/imdir_name}/recon_img_2_with_1_stain.png')
    image_dict['image2_reconO'] = copy.deepcopy(img)
    
    ### image two with random stain
    img = model.recon_image_PIL(torch.rand_like(s1), z2, denormer)
    img.save(f'{sourcedir/imdir_name}/recon_img_2_with_rand_stain.png')
    image_dict['image2_reconrand'] = copy.deepcopy(img)
    
    #####################################################################
    img = model.recon_image_PIL(s2, z1, denormer)
    img.save(f'{sourcedir/imdir_name}/recon_img_1_with_2_stain.png')
    image_dict['image1_reconO'] = copy.deepcopy(img)
    
    ### image two with random stain
    img = model.recon_image_PIL(torch.rand_like(s2), z1, denormer)
    img.save(f'{sourcedir/imdir_name}/recon_img_1_with_rand_stain.png')
    image_dict['image1_reconrand'] = copy.deepcopy(img)
    
    make_side_by_side(image_dict, f'{sourcedir/imdir_name}/side-by-side.png')
