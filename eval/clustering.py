## evaluate method for metadata-embedding similarity

from BPTorch.datasets import BigPictureRepository, WsiDicomDataset
from torch.utils.data import DataLoader
from BPTorch.utils import bptorch_collate
from torch.nn.functional import avg_pool2d
from pprint import pprint
from src.model.arch import H0_mini_for_AutoEncoding
from src.losses.loss_fusion import SIPE_Loss
from torchvision.transforms import ToPILImage
from src.utils.transfroms import UnNormalize
from src.trainer.trainer import Trainer
import os, torch, pathlib as pl, numpy as np
from tqdm import tqdm
from BPTorch.utils import bptorch_collate
from sklearn.preprocessing import LabelEncoder
from sklearn.manifold import TSNE
from umap import UMAP
from sklearn.metrics import davies_bouldin_score
from src.utils.misc import make_name_from_list
from src.utils.visu import plot_dim_red_clust, make_or_load_cmap
# pip install "BPTorch @ git+https://github.com/Bangulli/BPTorch"

if __name__ == '__main__':
    datapath = pl.Path('/mnt/nas6/data/BigPicture_CBIR/datasets/BPTorch/fold_0/BPR.json')
    inf_name = 'infered'
    recomp = input('Recompute embeddings? (y/n) : ').lower() == 'y'

    ####### data i/o
    if (not os.path.exists(datapath.parent/inf_name)) or recomp:
        ###### model i/o
        trainer = Trainer(H0_mini_for_AutoEncoding(device='cuda:1'), None, wdir='/home/lorenz/BigPicture/SIPE/SIPE-1M')
        model = trainer.load_best_model()
        model.eval()
        model.to(model.device)
        kwargs = WsiDicomDataset.get_default_kwargs()
        kwargs['transforms'] = model.transform
            
        ###### setup dataset
        ds = BigPictureRepository(datapath, load=True, wsidicomdataset_kwargs=kwargs, verbose=False)
        ds.source_precomputed_patches_from('rnd-subset-test')
        print(f"Dataset contains {len(ds)} foreground patches")
    
        dl = torch.utils.data.DataLoader(ds, collate_fn=bptorch_collate, batch_size=128)
        all_embeddings = []
        all_labels_stain = []
        all_labels_site = []
        for batch in tqdm(dl, desc='Infering Batches'):
            torch.cuda.empty_cache()
            _, emb = model(batch)
            lbls = [make_name_from_list(samp['staining']) for samp in batch['metadata']]
            lbls_site = [make_name_from_list(samp['organ']) for samp in batch['metadata']]
            
            all_embeddings.append(avg_pool2d(emb.detach().cpu(), kernel_size=16))
            all_labels_stain += lbls
            all_labels_site += lbls_site
            
            
        all_embeddings = torch.concat(all_embeddings, dim=0)
        all_embeddings = all_embeddings.numpy()
        print(all_embeddings.shape)
        os.makedirs(datapath.parent/inf_name, exist_ok=True)
        np.save(datapath.parent/inf_name/'embeddings.npy', all_embeddings)
        
        enc_stain = LabelEncoder()
        all_labels_stain = enc_stain.fit_transform(all_labels_stain)
        enc_stain = enc_stain.classes_
        np.save(datapath.parent/inf_name/'stainings.npy', all_labels_stain)
        np.save(datapath.parent/inf_name/'map_stain.npy', enc_stain)
        
        enc_site = LabelEncoder()
        all_labels_site = enc_site.fit_transform(all_labels_site)
        enc_site = enc_site.classes_
        np.save(datapath.parent/inf_name/'organs.npy', all_labels_site)
        np.save(datapath.parent/inf_name/'map_site.npy', enc_site)
        
    else:
        all_embeddings = np.load(datapath.parent/inf_name/'embeddings.npy')
        
        all_labels_stain = np.load(datapath.parent/inf_name/'stainings.npy')
        enc_stain = np.load(datapath.parent/inf_name/'map_stain.npy')
        
        all_labels_site = np.load(datapath.parent/inf_name/'organs.npy')
        enc_site = np.load(datapath.parent/inf_name/'map_site.npy')
        
    ############################################################################### make plots
    tsne=TSNE(n_components=2, random_state=42, perplexity=30, init="pca", learning_rate="auto", metric="cosine", early_exaggeration=12)
    projections_tsne = tsne.fit_transform(all_embeddings)
    print(f"Davies-Bouldin for clustering staining in the whole vector: {davies_bouldin_score(all_embeddings, all_labels_stain)}")
    print(f"Davies-Bouldin for clustering organ in the whole vector: {davies_bouldin_score(all_embeddings, all_labels_site)}")
    
    plot_dim_red_clust(projections_tsne, all_labels_stain, make_or_load_cmap(all_labels_stain, datapath.parent/inf_name/'stain_cmap.json'), datapath.parent/inf_name/'tsne-stain-ALL.png', 't-SNE for Stain')
    plot_dim_red_clust(projections_tsne, all_labels_site, make_or_load_cmap(all_labels_site, datapath.parent/inf_name/'site_cmap.json'), datapath.parent/inf_name/'tsne-site-ALL.png', 't-SNE for Site')
    
    tsne=TSNE(n_components=2, random_state=42, perplexity=30, init="pca", learning_rate="auto", metric="cosine", early_exaggeration=12)
    projections_tsne = tsne.fit_transform(all_embeddings[:,:64])
    print(f"Davies-Bouldin for clustering staining in the speciefied part: {davies_bouldin_score(all_embeddings[:,:64], all_labels_stain)}")
    print(f"Davies-Bouldin for clustering organ in the specified part: {davies_bouldin_score(all_embeddings[:,:64], all_labels_site)}")
    
    plot_dim_red_clust(projections_tsne, all_labels_stain, make_or_load_cmap(all_labels_stain, datapath.parent/inf_name/'stain_cmap.json'), datapath.parent/inf_name/'tsne-stain-S.png', 't-SNE for Stain')
    plot_dim_red_clust(projections_tsne, all_labels_site, make_or_load_cmap(all_labels_site, datapath.parent/inf_name/'site_cmap.json'), datapath.parent/inf_name/'tsne-site-S.png', 't-SNE for Site')
    
    tsne=TSNE(n_components=2, random_state=42, perplexity=30, init="pca", learning_rate="auto", metric="cosine", early_exaggeration=12)
    projections_tsne = tsne.fit_transform(all_embeddings[:,64:])
    print(f"Davies-Bouldin for clustering staining in the unspecified part: {davies_bouldin_score(all_embeddings[:,64:], all_labels_stain)}")
    print(f"Davies-Bouldin for clustering organ in the unspecified part: {davies_bouldin_score(all_embeddings[:,64:], all_labels_site)}")
    
    plot_dim_red_clust(projections_tsne, all_labels_stain, make_or_load_cmap(all_labels_stain, datapath.parent/inf_name/'stain_cmap.json'), datapath.parent/inf_name/'tsne-stain-Z.png', 't-SNE for Stain')
    plot_dim_red_clust(projections_tsne, all_labels_site, make_or_load_cmap(all_labels_site, datapath.parent/inf_name/'site_cmap.json'), datapath.parent/inf_name/'tsne-site-Z.png', 't-SNE for Site')