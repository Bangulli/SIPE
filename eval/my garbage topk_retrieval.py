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
import faiss
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from collections import Counter
# pip install "BPTorch @ git+https://github.com/Bangulli/BPTorch"

class InferenceWrapper():
    def __init__(self, model, mode: Literal['cls', 'patch_token', 'ours'], device='cuda:0'):
        self.model = model
        if mode=="ours": self.model.load("/home/lorenz/BigPicture/SIPE/SIPE-50k-Curriculum/checkpoints/ckpt_from_epoch_65")
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
            return z
        
class Fetcher():
    def __init__(self, ds, model, mode):
        self.datasource = DataLoader(ds, batch_size=1, collate_fn=bptorch_collate)
        self.inferer = InferenceWrapper(model, mode)
        self.index = None
        self._create_index()

    def _create_index(self):
        self.queries = {}
        self.metas = {}
        for i, s in enumerate(tqdm(self.datasource, desc="Building...")):
            self.metas[i]=s["metadata"][0]
            self.metas[i]["staining"] = model.defrag([self.metas[i]["staining"]])[0]
            emb = self.inferer(s).unsqueeze(0)
            self.queries[i]=emb

            if self.index is None:
                self.index = faiss.IndexFlatL2(emb.shape[1])
                self.index = faiss.IndexIDMap(self.index)
            
            self.index.add_with_ids(emb, torch.tensor([i]))
            
    def get_keys(self):
        return list(self.queries.keys())
            
    def get(self, idx, k=6):
        dists, idxs = self.index.search(self.queries[idx], k)
        return self.metas[idx], [self.metas[i] for i in idxs[0]]
    
    
    
def add(dct, ref, res, variable):
    cref = make_name_from_list(ref[variable])
    if cref not in dct.keys():
        dct[cref] = {}
        
    for i, r in enumerate(res):
        if i == 0: continue # skip self.
        cres = make_name_from_list(res[i][variable])
        if cres not in dct[cref].keys():
            dct[cref][cres] = 1
        else:
            dct[cref][cres] += 1
        
    return dct
    
def plot(data, path, title):
    df = pd.DataFrame(data).T  # rows = outer keys, cols = inner keys
    df = (df.div(df.sum(axis=1), axis=0) * 100).round()
    sns.heatmap(df, annot=True, fmt='.0f', cmap='viridis')
    plt.title(title)
    plt.tight_layout()
    plt.savefig(path)
    plt.close()
    plt.clf()
    
def get_rareness(fet):
    rares = {}
    for field in ["staining", "organ", "diagnosis"]:
        rares[field] = dict(Counter([make_name_from_list(f[field]) for f in fet.metas.values()]))
    return rares

def israre(val, rares, thresh=5):
    return rares[make_name_from_list(val)]<=thresh

def parse_rares(dct, rare, thresh=5):
    newdct = {}
    newdct["rare"] = {k:0 for k in rare.keys() if rare[k]>thresh}
    newdct["rare"]["rare"] = 0
    for kk, vv in dct.items():
        if rare[kk]>thresh:
            cur = {k:0 for k in rare.keys() if rare[k]>thresh}
            cur["rare"]=0
            for k, v in vv.items():
                if rare[k] > thresh: cur[k] += v
                else: cur["rare"] += v
            newdct[kk]=cur
        else: 
            for k, v in vv.items():
                if v > thresh: newdct["rare"][k] += v
                else: newdct["rare"]["rare"] += v
    return newdct

def make_and_apply_lookup(data):
    lookup = {k:i for i,k in enumerate(data.keys())}
    return {lookup[k]:{lookup[kk]:vv for kk, vv in v.items()} for k, v in data.items()}, lookup

if __name__ == "__main__":
    outputpath = pl.Path('SIPE-50k-Curriculum/.results/retrieval')
    with open('/home/lorenz/BigPicture/SIPE/classes.json', 'r') as f:
        classes = json.load(f)
    model = H0_mini_for_Adversarial(classes, device='cuda:1')
    
    datapath = pl.Path('/mnt/nas6/data/BigPicture_CBIR/datasets/BPTorch/fold_0/BPR.json')
    kwargs = WsiDicomDataset.get_default_kwargs()
    kwargs['transforms'] = model.transform
    ds = BigPictureRepository(datapath, load=True, wsidicomdataset_kwargs=kwargs, verbose=False)
    ds.source_precomputed_patches_from('data/rnd-subset-test')
    print(f"Dataset contains {len(ds)} foreground patches")
    
    for mode in ["cls", "patch_token", "ours"]:
        cooc_stain = {}
        cooc_organ = {}
        cooc_diagn = {}
        index = Fetcher(ds, model, mode)
        rares = get_rareness(index)
        with open("eval/topk_occurance.json", "w") as f:
            json.dump(rares, f, indent=4)
        precs = {"staining": {"TP":0, "N":0, "FP":0, "FN":0}, "organ": {"TP":0, "N":0, "FP":0, "FN":0}, "diagnosis": {"TP":0, "N":0, "FP":0, "FN":0}}
        for k in index.get_keys():
            ref, res = index.get(k, 6)
            cooc_stain = add(cooc_stain, ref, res, "staining")
            cooc_organ = add(cooc_organ, ref, res, "organ")
            cooc_diagn = add(cooc_diagn, ref, res, "diagnosis")
            
            ## topk prec rec
            if not israre(ref["staining"], rares["staining"], 6):
                precs["staining"]["N"] += 5
                tp = len([k for k in res if make_name_from_list(k["staining"]) == make_name_from_list(ref["staining"])])
                precs["staining"]["TP"] += tp
                precs["staining"]["FP"] += 5-tp
                precs["staining"]["FN"] += rares["staining"][make_name_from_list(ref["staining"])]-tp
            if not israre(ref["organ"], rares["organ"], 6):
                precs["organ"]["N"] += 5
                tp = len([k for k in res if make_name_from_list(k["organ"]) == make_name_from_list(ref["organ"])])
                precs["organ"]["TP"] += tp
                precs["organ"]["FP"] += 5-tp
                precs["organ"]["FN"] += rares["organ"][make_name_from_list(ref["organ"])]-tp
            if not israre(ref["diagnosis"], rares["diagnosis"], 6):
                precs["diagnosis"]["N"] += 5
                tp = len([k for k in res if make_name_from_list(k["diagnosis"]) == make_name_from_list(ref["diagnosis"])])
                precs["diagnosis"]["TP"] += tp
                precs["diagnosis"]["FP"] += 5-tp
                precs["diagnosis"]["FN"] += rares["diagnosis"][make_name_from_list(ref["diagnosis"])]-tp
            
        ## clean
        cooc_stain = parse_rares(cooc_stain, rares["staining"], 6)
        cooc_organ = parse_rares(cooc_organ, rares["organ"], 6)
        cooc_diagn = parse_rares(cooc_diagn, rares["diagnosis"], 6)
        
        ## save and plot
        os.makedirs(outputpath/mode, exist_ok=True)
        with open(outputpath/mode/"cooc_stain.json", "w") as f:
            json.dump(cooc_stain, f, indent=4)
            cooc_stain, lookup_stain = make_and_apply_lookup(cooc_stain)
            with open(outputpath/mode/"lookup_stain.json", "w") as ff:
                json.dump(lookup_stain, ff, indent=4)
            plot(cooc_stain, outputpath/mode/"cooc_stain.png", "Top 5 Retrieval Co-Occurance: Staining")
            
        with open(outputpath/mode/"cooc_organ.json", "w") as f:
            json.dump(cooc_organ, f, indent=4)
            cooc_organ, lookup_organ = make_and_apply_lookup(cooc_organ)
            with open(outputpath/mode/"lookup_organ.json", "w") as ff:
                json.dump(lookup_organ, ff, indent=4)
            plot(cooc_organ, outputpath/mode/"cooc_organ.png", "Top 5 Retrieval Co-Occurance: Organ")
            
        with open(outputpath/mode/"cooc_diagn.json", "w") as f:
            json.dump(cooc_diagn, f, indent=4)
            cooc_diagn, lookup_diagn = make_and_apply_lookup(cooc_diagn)
            with open(outputpath/mode/"lookup_diagn.json", "w") as ff:
                json.dump(lookup_diagn, ff, indent=4)
            plot(cooc_diagn, outputpath/mode/"cooc_diagn.png", "Top 5 Retrieval Co-Occurance: Diagnosis")
            
        ## prec and rec
        for k, v in precs.items():
            precision = v["TP"]/(v["N"]+1e-6)
            recall = v["TP"]/(v["TP"]+v["FN"]+1e-6)
            with open(outputpath/mode/f"{k}_precrec.json", "w") as f:
                json.dump({"Precision":precision, "Recall":recall}, f, indent=4)