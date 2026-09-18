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
    def __init__(self, model, mode: Literal['cls', 'patch_token', "ours-z", "ours-s"], device='cuda:0'):
        self.model = model
        if mode == "ours":
            self.model.load("/home/lorenz/BigPicture/SIPE/SIPE-50k-Curriculum/checkpoints/ckpt_from_epoch_65")
        self.model.eval()
        self.model.device = device
        self.model.to(device)
        self.device = device
        if mode.lower() in ['cls', 'patch_token', "ours-z", "ours-s"]:
            self.mode = mode.lower()
        else:
            raise RuntimeError(f'Unknown mode {mode}')

    @torch.no_grad()
    def __call__(self, batch):
        if self.mode == 'cls':
            x = batch['image'].to(self.device)
            if len(x.shape) == 3:
                x = x.unsqueeze(0)  ## add batch dim if necessary
            tok = self.model.backbone(x)[:, 0, :]
            return tok.detach().cpu().squeeze()
        elif self.mode == 'patch_token':
            x = batch['image'].to(self.device)
            if len(x.shape) == 3:
                x = x.unsqueeze(0)  ## add batch dim if necessary
            tok = self.model.backbone(x)[:, 5:, :].permute(0, 2, 1)
            tok = tok.reshape(tok.shape[0], 768, 16, 16)
            return avg_pool2d(tok, kernel_size=16).detach().cpu().squeeze()
        elif self.mode == 'ours-z':
            s, z = self.model(batch)
            s = s.detach().cpu().squeeze()
            z = avg_pool2d(z, kernel_size=16).detach().cpu().squeeze()
            return z
        elif self.mode == "ours-s":
            s, z = self.model(batch)
            s = s.detach().cpu().squeeze()
            return s


def _to_faiss_vec(emb):
    """Ensure a 2D, contiguous, float32 numpy array for FAISS."""
    if isinstance(emb, torch.Tensor):
        emb = emb.detach().cpu().numpy()
    emb = np.ascontiguousarray(emb, dtype=np.float32)
    if emb.ndim == 1:
        emb = emb[None, :]
    return emb


class Fetcher():
    def __init__(self, ds, model, mode):
        self.model = model  # keep an explicit handle instead of relying on a global
        self.datasource = DataLoader(ds, batch_size=1, collate_fn=bptorch_collate)
        self.inferer = InferenceWrapper(model, mode)
        self.index = None
        self._create_index()

    def _create_index(self):
        self.queries = {}   # id -> np.float32 [1, D]
        self.metas = {}     # id -> metadata dict
        for i, s in enumerate(tqdm(self.datasource, desc="Building...")):
            self.metas[i] = s["metadata"][0]
            self.metas[i]["staining"] = self.model.defrag([self.metas[i]["staining"]])[0]

            emb = _to_faiss_vec(self.inferer(s))
            self.queries[i] = emb

            if self.index is None:
                self.index = faiss.IndexFlatL2(emb.shape[1])
                self.index = faiss.IndexIDMap(self.index)

            self.index.add_with_ids(emb, np.array([i], dtype=np.int64))

    def get_keys(self):
        return list(self.queries.keys())

    def get(self, idx, k=5):
        """Returns (ref_meta, [result_metas]).

        The first result is guaranteed to be the query itself: with an exact
        IndexFlatL2 the identical stored vector has distance 0, but we also
        defensively drop any entry whose id matches the query id.
        """
        dists, idxs = self.index.search(self.queries[idx], k+1)
        ids = [int(j) for i, j in enumerate(idxs[0]) if j != -1 and i != 0]
        return self.metas[idx], [self.metas[j] for j in ids], ids
    
class CrossStainFetcher():
    def __init__(self, ds, model, mode, distance_type="euclidean"):
        self.model = model  # keep an explicit handle instead of relying on a global
        self.datasource = DataLoader(ds, batch_size=1, collate_fn=bptorch_collate)
        self.inferer = InferenceWrapper(model, mode)
        self.index = None
        self.distype = distance_type
        self._create_index()

    def _create_index(self):
        self.queries = {}   # id -> np.float32 [1, D]
        self.metas = {}     # id -> metadata dict
        for i, s in enumerate(tqdm(self.datasource, desc="Building...")):
            self.metas[i] = s["metadata"][0]
            self.metas[i]["staining"] = self.model.defrag([self.metas[i]["staining"]])[0]

            emb = _to_faiss_vec(self.inferer(s))
            if self.distype == "cosine": faiss.normalize_L2(emb) 
            self.queries[i] = emb

            if self.index is None:
                if self.distype == "cosine": self.index = faiss.IndexFlatIP(emb.shape[1])
                else: self.index = faiss.IndexFlatL2(emb.shape[1])
                self.index = faiss.IndexIDMap(self.index)

            self.index.add_with_ids(emb, np.array([i], dtype=np.int64))

    def get_keys(self):
        return list(self.queries.keys())

    def get(self, idx, k=5):
        """Returns (ref_meta, [result_metas]).

        The first result is guaranteed to be the query itself: with an exact
        IndexFlatL2 the identical stored vector has distance 0, but we also
        defensively drop any entry whose id matches the query id.
        """
        params = faiss.SearchParametersIVF()
        ids_to_search = np.asarray([int(k) for k, v in self.metas.items() if v["staining"] != self.metas[idx]["staining"]], dtype='int64')
        params.sel = faiss.IDSelectorArray(ids_to_search.size, faiss.swig_ptr(ids_to_search))
        
        dists, idxs = self.index.search(self.queries[idx], k+1, params=params)
        ids = [int(j) for i, j in enumerate(idxs[0]) if j != -1 and i != 0]
        return self.metas[idx], [self.metas[j] for j in ids], ids


def add(dct, ref, res, res_ids, query_id, variable):
    """Accumulate co-occurrence counts, skipping the query itself by id."""
    cref = make_name_from_list(ref[variable])
    if cref not in dct:
        dct[cref] = {}

    for r, rid in zip(res, res_ids):
        if rid == query_id:
            continue  # skip self by id, not by position
        cres = make_name_from_list(r[variable])
        dct[cref][cres] = dct[cref].get(cres, 0) + 1
    return dct


def plot(data, path, title):
    for k, v in data.items():
        data[k] = {kk:v[kk] for kk in data.keys()}
    df = pd.DataFrame(data).T  # rows = outer keys, cols = inner keys
    df = df.fillna(0)
    row_sums = df.sum(axis=1).replace(0, np.nan)
    df = (df.div(row_sums, axis=0) * 100).round()
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
    """True when the class is RARE, i.e. appears at most `thresh` times."""
    return rares[make_name_from_list(val)] <= thresh


def parse_rares(dct, rare, thresh=5):
    newdct = {}
    newdct["rare"] = {k: 0 for k in rare.keys() if rare[k] > thresh}
    newdct["rare"]["rare"] = 0
    for kk, vv in dct.items():
        if rare[kk] > thresh:
            cur = {k: 0 for k in rare.keys() if rare[k] > thresh}
            cur["rare"] = 0
            for k, v in vv.items():
                if rare[k] > thresh:
                    cur[k] += v
                else:
                    cur["rare"] += v
            newdct[kk] = cur
        else:
            for k, v in vv.items():
                if rare[k] > thresh:
                    newdct["rare"][k] += v
                else:
                    newdct["rare"]["rare"] += v
    return newdct


def make_and_apply_lookup(data):
    lookup = {k: i for i, k in enumerate(data.keys())}
    return {lookup[k]: {lookup[kk]: vv for kk, vv in v.items()} for k, v in data.items()}, lookup


if __name__ == "__main__":
    K = 5                  # total retrieved including the query itself
    RARE_THRESH = 6        # classes with <= this many members are treated as "rare"

    outputpath = pl.Path('SIPE-50k-Curriculum/.results/cosine/retrieval')
    with open('/home/lorenz/BigPicture/SIPE/classes.json', 'r') as f:
        classes = json.load(f)
    model = H0_mini_for_Adversarial(classes, device='cuda:1')

    datapath = pl.Path('/mnt/nas6/data/BigPicture_CBIR/datasets/BPTorch/fold_0/BPR.json')
    kwargs = WsiDicomDataset.get_default_kwargs()
    kwargs['transforms'] = model.transform
    ds = BigPictureRepository(datapath, load=True, wsidicomdataset_kwargs=kwargs, verbose=False)
    ds.source_precomputed_patches_from('data/rnd-subset-test')
    print(f"Dataset contains {len(ds)} foreground patches")

    for mode in ["cls", "patch_token", "ours-z", "ours-s"]:
        cooc_stain = {}
        cooc_organ = {}
        cooc_diagn = {}
        index = Fetcher(ds, model, mode)
        rares = get_rareness(index)
        with open("eval/topk_occurance.json", "w") as f:
            json.dump(rares, f, indent=4)

        # TP: retrieved items of the same class (excluding self)
        # retrieved: number of retrieved items scored (the precision denominator)
        # relevant: max true positives that COULD be retrieved per query, capped at k
        precs = {field: {"TP": 0, "retrieved": 0, "relevant": 0}
                 for field in ["staining", "organ", "diagnosis"]}

        for q in index.get_keys():
            ref, res, res_ids = index.get(q, K)

            cooc_stain = add(cooc_stain, ref, res, res_ids, q, "staining")
            cooc_organ = add(cooc_organ, ref, res, res_ids, q, "organ")
            cooc_diagn = add(cooc_diagn, ref, res, res_ids, q, "diagnosis")

            # results with the query itself removed
            neighbours = [(r, rid) for r, rid in zip(res, res_ids) if rid != q]

            for field in ["staining", "organ", "diagnosis"]:
                # only evaluate non-rare reference classes
                if israre(ref[field], rares[field], RARE_THRESH):
                    continue

                ref_cls = make_name_from_list(ref[field])
                cls_count = rares[field][ref_cls]

                k_ret = len(neighbours)
                tp = sum(make_name_from_list(r[field]) == ref_cls for r, _ in neighbours)
                relevant = min(k_ret, cls_count - 1)  # -1 excludes the query itself

                precs[field]["TP"] += tp
                precs[field]["retrieved"] += k_ret
                precs[field]["relevant"] += relevant

        ## clean
        cooc_stain = parse_rares(cooc_stain, rares["staining"], RARE_THRESH)
        cooc_organ = parse_rares(cooc_organ, rares["organ"], RARE_THRESH)
        cooc_diagn = parse_rares(cooc_diagn, rares["diagnosis"], RARE_THRESH)

        ## save and plot
        os.makedirs(outputpath / mode, exist_ok=True)
        with open(outputpath / mode / "cooc_stain.json", "w") as f:
            json.dump(cooc_stain, f, indent=4)
        cooc_stain, lookup_stain = make_and_apply_lookup(cooc_stain)
        with open(outputpath / mode / "lookup_stain.json", "w") as ff:
            json.dump(lookup_stain, ff, indent=4)
        plot(cooc_stain, outputpath / mode / "cooc_stain.png", "Top 5 Retrieval Co-Occurance: Staining")

        with open(outputpath / mode / "cooc_organ.json", "w") as f:
            json.dump(cooc_organ, f, indent=4)
        cooc_organ, lookup_organ = make_and_apply_lookup(cooc_organ)
        with open(outputpath / mode / "lookup_organ.json", "w") as ff:
            json.dump(lookup_organ, ff, indent=4)
        plot(cooc_organ, outputpath / mode / "cooc_organ.png", "Top 5 Retrieval Co-Occurance: Organ")

        with open(outputpath / mode / "cooc_diagn.json", "w") as f:
            json.dump(cooc_diagn, f, indent=4)
        cooc_diagn, lookup_diagn = make_and_apply_lookup(cooc_diagn)
        with open(outputpath / mode / "lookup_diagn.json", "w") as ff:
            json.dump(lookup_diagn, ff, indent=4)
        plot(cooc_diagn, outputpath / mode / "cooc_diagn.png", "Top 5 Retrieval Co-Occurance: Diagnosis")

        ## prec and rec
        for k, v in precs.items():
            precision = v["TP"] / (v["retrieved"] + 1e-6)
            recall = v["TP"] / (v["relevant"] + 1e-6)
            with open(outputpath / mode / f"{k}_precrec.json", "w") as f:
                json.dump({"Precision": precision, "Recall": recall}, f, indent=4)
                
    ############################################## CROSS - STAIN - RETRIEVAL ##############################################
    outputpath = pl.Path('SIPE-50k-Curriculum/.results/cosine/retrieval-cross-stain')
    with open('/home/lorenz/BigPicture/SIPE/classes.json', 'r') as f:
        classes = json.load(f)
    model = H0_mini_for_Adversarial(classes, device='cuda:1')

    datapath = pl.Path('/mnt/nas6/data/BigPicture_CBIR/datasets/BPTorch/fold_0/BPR.json')
    kwargs = WsiDicomDataset.get_default_kwargs()
    kwargs['transforms'] = model.transform
    ds = BigPictureRepository(datapath, load=True, wsidicomdataset_kwargs=kwargs, verbose=False)
    ds.source_precomputed_patches_from('data/rnd-subset-test')
    print(f"Dataset contains {len(ds)} foreground patches")

    for mode in ["cls", "patch_token", "ours-z", "ours-s"]:
        cooc_stain = {}
        cooc_organ = {}
        cooc_diagn = {}
        index = CrossStainFetcher(ds, model, mode)
        rares = get_rareness(index)
        with open("eval/topk_occurance.json", "w") as f:
            json.dump(rares, f, indent=4)

        # TP: retrieved items of the same class (excluding self)
        # retrieved: number of retrieved items scored (the precision denominator)
        # relevant: max true positives that COULD be retrieved per query, capped at k
        precs = {field: {"TP": 0, "retrieved": 0, "relevant": 0}
                 for field in ["organ", "diagnosis"]}

        for q in index.get_keys():
            ref, res, res_ids = index.get(q, K)

            cooc_organ = add(cooc_organ, ref, res, res_ids, q, "organ")
            cooc_diagn = add(cooc_diagn, ref, res, res_ids, q, "diagnosis")

            # results with the query itself removed
            neighbours = [(r, rid) for r, rid in zip(res, res_ids) if rid != q]

            for field in ["organ", "diagnosis"]:
                # only evaluate non-rare reference classes
                if israre(ref[field], rares[field], RARE_THRESH):
                    continue

                ref_cls = make_name_from_list(ref[field])
                cls_count = rares[field][ref_cls]

                k_ret = len(neighbours)
                tp = sum(make_name_from_list(r[field]) == ref_cls for r, _ in neighbours)
                relevant = min(k_ret, cls_count - 1)  # -1 excludes the query itself

                precs[field]["TP"] += tp
                precs[field]["retrieved"] += k_ret
                precs[field]["relevant"] += relevant

        ## clean
        cooc_organ = parse_rares(cooc_organ, rares["organ"], RARE_THRESH)
        cooc_diagn = parse_rares(cooc_diagn, rares["diagnosis"], RARE_THRESH)

        ## save and plot
        os.makedirs(outputpath / mode, exist_ok=True)

        with open(outputpath / mode / "cooc_organ.json", "w") as f:
            json.dump(cooc_organ, f, indent=4)
        cooc_organ, lookup_organ = make_and_apply_lookup(cooc_organ)
        with open(outputpath / mode / "lookup_organ.json", "w") as ff:
            json.dump(lookup_organ, ff, indent=4)
        plot(cooc_organ, outputpath / mode / "cooc_organ.png", "Top 5 Retrieval Co-Occurance: Organ")

        with open(outputpath / mode / "cooc_diagn.json", "w") as f:
            json.dump(cooc_diagn, f, indent=4)
        cooc_diagn, lookup_diagn = make_and_apply_lookup(cooc_diagn)
        with open(outputpath / mode / "lookup_diagn.json", "w") as ff:
            json.dump(lookup_diagn, ff, indent=4)
        plot(cooc_diagn, outputpath / mode / "cooc_diagn.png", "Top 5 Retrieval Co-Occurance: Diagnosis")

        ## prec and rec
        for k, v in precs.items():
            precision = v["TP"] / (v["retrieved"] + 1e-6)
            recall = v["TP"] / (v["relevant"] + 1e-6)
            with open(outputpath / mode / f"{k}_precrec.json", "w") as f:
                json.dump({"Precision": precision, "Recall": recall}, f, indent=4)