## evaluate method for metadata-embedding similarity
import os, torch, pathlib as pl, numpy as np, sys, json
sys.path.append(os.path.join(os.path.dirname(__file__), "."))
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

from BPTorch.datasets import BigPictureRepository, WsiDicomDataset
from torch.utils.data import DataLoader
from BPTorch.utils import bptorch_collate
from torch.nn.functional import avg_pool2d
from pprint import pprint
from src.model.arch import H0_mini_for_Adversarial
from torchvision.transforms import ToPILImage
from src.utils.transfroms import UnNormalize
from src.trainer.trainer import Trainer
from tqdm import tqdm
from BPTorch.utils import bptorch_collate
from sklearn.preprocessing import LabelEncoder
from sklearn.manifold import TSNE
from umap import UMAP
from sklearn.metrics import davies_bouldin_score
from src.utils.misc import make_name_from_list
from src.utils.visu import plot_dim_red_clust, make_or_load_cmap
from typing import Literal
# pip install "BPTorch @ git+https://github.com/Bangulli/BPTorch"

class InferenceWrapper():
    def __init__(self, model, mode: Literal['cls', 'patch_token', 'ours'], device='cuda:0'):
        self.model = model
        self.model.eval()
        self.model.device = device
        self.model.to(device)
        self.device = device
        if mode.lower() in ['cls', 'patch_token', 'ours']: self.mode = mode.lower()
        else: raise RuntimeError(f'Unkown mode {mode}')
        
    def __call__(self, batch):
        if self.mode == 'cls':
            x = batch['image'].to(self.device)
            if len(x.shape)==3: x = x.unsqueeze(0) ## add batch dim if neccessary
            tok = self.model.backbone(x)[:, 0, :]
            return tok.detach().cpu().squeeze()
        elif self.mode == 'patch_token':
            x = batch['image'].to(self.device)
            if len(x.shape)==3: x = x.unsqueeze(0) ## add batch dim if neccessary
            tok = self.model.backbone(x)[:, 5:, :].permute(0, 2, 1)
            tok = tok.reshape(tok.shape[0], 768, 16, 16)
            return avg_pool2d(tok, kernel_size=16).detach().cpu().squeeze()
        elif self.mode == 'ours':
            s, z = self.model(batch)
            s = s.detach().cpu().squeeze()
            z = avg_pool2d(z, kernel_size=16).detach().cpu().squeeze()
            sz = torch.cat([s, z], dim=1)
            return s, z, sz

def run_inference(datapath, outputpath, inf_name, methods, trainer_path):
    with open('/home/lorenz/BigPicture/SIPE/classes.json', 'r') as f:
        classes = json.load(f)
    os.makedirs(outputpath, exist_ok=True)
    for eval_method in methods:
        ###### model i/o
        if eval_method in ['cls', 'patch_token']: ## make default config model with default weights
            model = H0_mini_for_Adversarial(classes, device='cuda:1')
        else: 
            trainer = Trainer(H0_mini_for_Adversarial(classes, device='cuda:1'), None, wdir=trainer_path)
            model = trainer.load_best_model()

        inferer = InferenceWrapper(model, eval_method)
            
        ###### data i/o
        kwargs = WsiDicomDataset.get_default_kwargs()
        kwargs['transforms'] = model.transform
            
        ###### setup dataset
        ds = BigPictureRepository(datapath, load=True, wsidicomdataset_kwargs=kwargs, verbose=False)
        ds.source_precomputed_patches_from('rnd-subset-test')
        print(f"Dataset contains {len(ds)} foreground patches")
    
        dl = torch.utils.data.DataLoader(ds, collate_fn=bptorch_collate, batch_size=128)
        all_embeddings = []
        all_s = []
        all_z = []
        all_sz = []
        all_labels_stain = []
        all_labels_site = []
        all_labels_diag = []
        for batch in tqdm(dl, desc='Infering Batches'):
            torch.cuda.empty_cache()
            
            if eval_method  in ['cls', 'patch_token']:
                emb = inferer(batch)
                all_embeddings.append(emb)
            else:
                all_embeddings = False
                s, z, sz = inferer(batch)
                all_s.append(s)
                all_z.append(z)
                all_sz.append(sz)
                
            lbls = [make_name_from_list(samp['staining']) for samp in batch['metadata']]
            lbls_site = [make_name_from_list(samp['organ']) for samp in batch['metadata']]
            lbls_diag = [make_name_from_list(samp['diagnosis']) for samp in batch['metadata']]
            
            all_labels_stain += lbls
            all_labels_site += lbls_site
            all_labels_diag += lbls_diag
            
        if all_embeddings:
            all_embeddings = torch.concat(all_embeddings, dim=0)
            all_embeddings = all_embeddings.numpy()
            print(all_embeddings.shape)
            os.makedirs(outputpath/inf_name, exist_ok=True)
            np.save(outputpath/inf_name/f'{eval_method}_embeddings.npy', all_embeddings)
        else:
            all_s = torch.concat(all_s, dim=0)
            all_s = all_s.numpy()
            print(all_s.shape)
            os.makedirs(outputpath/inf_name, exist_ok=True)
            np.save(outputpath/inf_name/f'{eval_method}_s.npy', all_s)
            
            all_z = torch.concat(all_z, dim=0)
            all_z = all_z.numpy()
            print(all_z.shape)
            os.makedirs(outputpath/inf_name, exist_ok=True)
            np.save(outputpath/inf_name/f'{eval_method}_z.npy', all_z)
        
        
            all_sz = torch.concat(all_sz, dim=0)
            all_sz = all_sz.numpy()
            print(all_sz.shape)
            os.makedirs(outputpath/inf_name, exist_ok=True)
            np.save(outputpath/inf_name/f'{eval_method}_sz.npy', all_sz)
        
        
        enc_stain = LabelEncoder()
        all_labels_stain = enc_stain.fit_transform(all_labels_stain)
        enc_stain = enc_stain.classes_
        np.save(outputpath/inf_name/'stainings.npy', all_labels_stain)
        np.save(outputpath/inf_name/'map_stain.npy', enc_stain)
        
        enc_site = LabelEncoder()
        all_labels_site = enc_site.fit_transform(all_labels_site)
        enc_site = enc_site.classes_
        np.save(outputpath/inf_name/'organs.npy', all_labels_site)
        np.save(outputpath/inf_name/'map_site.npy', enc_site)
        
        enc_diag = LabelEncoder()
        all_labels_diag = enc_diag.fit_transform(all_labels_diag)
        enc_diag = enc_diag.classes_
        np.save(outputpath/inf_name/'diags.npy', all_labels_diag)
        np.save(outputpath/inf_name/'map_diag.npy', enc_diag)
        
def eval_clust(emb_path, stain_path, stain_map, site_path, site_map, diag_path, diag_map, outpath):
    os.makedirs(outpath, exist_ok=True)
    name = emb_path.name.split('.')[0]
    all_embeddings = np.load(emb_path)
    
    all_labels_stain = np.load(stain_path)
    enc_stain = np.load(stain_map)
    
    all_labels_site = np.load(site_path)
    enc_site = np.load(site_map)
    
    all_labels_diag = np.load(diag_path)
    enc_diag = np.load(diag_map)
    
    ############################################################################### make plots
    tsne=TSNE(n_components=2, random_state=42, perplexity=30, init="pca", learning_rate="auto", metric="cosine", early_exaggeration=12)
    projections_tsne = tsne.fit_transform(all_embeddings)
    print(f"Davies-Bouldin for clustering staining: {davies_bouldin_score(all_embeddings, all_labels_stain)}")
    print(f"Davies-Bouldin for clustering organ: {davies_bouldin_score(all_embeddings, all_labels_site)}")
    print(f"Davies-Bouldin for clustering diag: {davies_bouldin_score(all_embeddings, all_labels_diag)}")
    
    plot_dim_red_clust(projections_tsne, all_labels_stain, make_or_load_cmap(all_labels_stain, outpath/'stain_cmap.json'), outpath/f'{name}-tsne-stain-ALL.png', f'{name} t-SNE for Stain')
    plot_dim_red_clust(projections_tsne, all_labels_site, make_or_load_cmap(all_labels_site, outpath/'site_cmap.json'), outpath/f'{name}-tsne-site-ALL.png', f'{name} t-SNE for Site')
    plot_dim_red_clust(projections_tsne, all_labels_diag, make_or_load_cmap(all_labels_diag, outpath/'diag_cmap.json'), outpath/f'{name}-tsne-diag-ALL.png', f'{name} t-SNE for Diag')

if __name__ == '__main__':
    datapath = pl.Path('/mnt/nas6/data/BigPicture_CBIR/datasets/BPTorch/fold_0/BPR.json')
    outputpath = pl.Path('SIPE-1M-Curriculum/.results/clustering')
    inf_name = 'infered'

    #run_inference(datapath, outputpath, inf_name, ['cls', 'patch_token', 'ours'], '/home/lorenz/BigPicture/SIPE/SIPE-1M-Curriculum')
        
    for eval_res in ['cls_embeddings.npy', 'patch_token_embeddings.npy']:
        print(f'---------------------------------- Evaluating {eval_res} Baseline ----------------------------------')
        eval_clust(outputpath/inf_name/eval_res, outputpath/inf_name/'stainings.npy', outputpath/inf_name/'map_stain.npy', outputpath/inf_name/'organs.npy', outputpath/inf_name/'map_site.npy',  outputpath/inf_name/'diags.npy', outputpath/inf_name/'map_diag.npy', outpath=outputpath/'baseline')
        
    for eval_res in ['ours_s.npy', 'ours_z.npy', 'ours_sz.npy']:  
        print(f'---------------------------------- Evaluating {eval_res} Ours ----------------------------------')
        eval_clust(outputpath/inf_name/eval_res, outputpath/inf_name/'stainings.npy', outputpath/inf_name/'map_stain.npy', outputpath/inf_name/'organs.npy', outputpath/inf_name/'map_site.npy',  outputpath/inf_name/'diags.npy', outputpath/inf_name/'map_diag.npy', outpath=outputpath/'ours')