import torchvision.transforms as T
import os, json, random
import numpy as np
from pathlib import Path
from matplotlib.colors import to_hex
import matplotlib.pyplot as plt

def visualize_batch(batch, dir):
    elems = batch['image'].shape[0]
    transforms = batch['contrastive'].shape[1]
    dir = Path(dir)
    for sample in range(elems):
        pth = dir/f"Sample_{sample}"
        os.makedirs(pth, exist_ok=True)
        
        ## visualize patch
        img = T.ToPILImage()(batch['image'][sample, :])
        img.save(pth/'img.jpg')
        
        ## visualize transformations
        for t in range(transforms):
            img_t = T.ToPILImage()(batch['contrastive'][sample, t, :])
            img_t.save(pth/f"Transform_{t}.jpg")
            
        ## save metadata
        with open(pth/"meta.json", "w") as f:
            json.dump(batch['metadata'][sample], f, indent=2)
            
        with open(pth/"coords.json", "w") as f:
            json.dump(batch['coordinates'][sample, :].tolist(), f , indent=2)
            
def plot_dim_red_clust(projections, labels, cmap, pth, title):
    colors = [cmap[str(l)] for l in labels]
    fig, ax = plt.subplots(figsize=(8, 8))
    ax.scatter(
        projections[:, 0],
        projections[:, 1],
        c=colors,
        s=10,
        alpha=0.8,
        edgecolors="none",
    )
    ax.set_xlabel("Dim 1")
    ax.set_ylabel("Dim 2")
    ax.set_aspect("equal")
    ax.set_title(title)
    
    unique_labels = []
    handles = []
    from matplotlib import patches as mpatches
    for v in labels:
        v_str = "None" if v is None else str(v)
        if v_str not in unique_labels:
            unique_labels.append(v_str)
            color = cmap[v_str]
            handles.append(mpatches.Patch(color=color, label=v_str))
            
    fig.tight_layout()
    fig.savefig(pth)
    plt.close(fig)
    
    # Heuristic: height scales with number of entries
    height = max(2, 0.3 * len(labels))
    fig = plt.figure(figsize=(4, height))

    # Centered legend on empty figure
    fig_legend = fig.legend(
        handles,
        labels,
        loc="center",
        frameon=False,
        ncol=1,
        title=f"Legend for {title}",
    )

    fig.tight_layout()
    fig.savefig(pth.parent/f"legend_{pth.name}", dpi=300, bbox_inches="tight")
    plt.close(fig)

def make_or_load_cmap(labels, pth):
    if os.path.exists(pth):
        with open(pth, 'r') as f:
            return json.load(f)
    else:
        cmap = {}
        for lbl in labels:
            if lbl not in cmap.keys():
                cmap[lbl]=None
        
        pltcmap = plt.cm.get_cmap('hsv', len(cmap))
        cmap = {str(k): to_hex(pltcmap(i)) for i, (k, v) in enumerate(cmap.items())}
        with open(pth, 'w') as f:
            json.dump(cmap, f, indent=4)
        return cmap
        