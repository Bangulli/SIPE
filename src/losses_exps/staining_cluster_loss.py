######## Ecosystem ########
import os, sys, pathlib as pl
sys.path.append(os.path.join(os.path.dirname(__file__), "."))
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
import math
######## External ########
import torch
import torch.nn as nn
from sklearn.preprocessing import LabelEncoder
import numpy as np
import torch.nn.functional as F
######## Internal ########
##########################

#########################################################################################################################################
# Davies bouldin based cluster optimization https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=4766909
class StainingClusterLoss(nn.Module):
    def __init__(self):
        super().__init__()
        
    def forward(self, stain, morph, meta, device, logger=None):
        
        return loss, logger
    
    def _make_name_from_list(self, data):
        if isinstance(data, str):
            return data
        return "+".join(data)
    
    def _defragment(self, raw_labels, frq):
        new_labels = []
        for label in raw_labels:
            ## hematox & antibod
            if 'hematoxylin' in label.lower() and not 'eosin' in label.lower():
                if frq[label]>1: new_labels.append(label)
                else: new_labels.append('H&A')
            ## hematox & eosin
            elif label == "HE - Hematoxylin and eosin stain method (procedure)" or label == "Hematoxylin and eosin stain method" or label == "hematoxylin stain+water soluble eosin stain":
                new_labels.append('H&E')
            ## Van Gieson
            elif label == "Van Gieson stain" or label == "Verhoeff-Van Gieson stain method":
                new_labels.append("Van Gieson stain")
            elif "Periodic acid Schiff stain" in label and "blue" not in label:
                new_labels.append("Periodic acid Schiff stain")
            else: new_labels.append(label)
        return new_labels
        ## if a hematoxylin+antibody is unique add to hematox+anti class
        ## defrag any same meaning different name classes like the procedure stuff
    
    def _remove_singles(self, stain, morph, frqs, label):
        ## delete singleton samples
        deldim = []
        for cls, i in frqs.items():
            if i < 2:
                if self.testmode: print(f'Found singleton class at {cls}')
                deldim.append(False)
            else: deldim.append(True)
        deldim = [deldim[l] for l in label]
        deldim = torch.tensor(deldim, device=stain.device)     
        
        if self.testmode:print(f"Dims BEFORE deletion:", f"Stain={stain.shape}", f"Morph={morph.shape}", f"Label={label.shape}")        
        stain = stain[deldim, :]
        morph = morph[deldim, :]
        label = label[deldim]
        if self.testmode:print(f"Dims AFTER deletion:", f"Stain={stain.shape}", f"Morph={morph.shape}", f"Label={label.shape}")
        return stain, morph, label
    
    
#########################################################################################################################################
## SimCLR based contrastive loss from https://lightning.ai/docs/pytorch/stable/notebooks/course_UvA-DL/13-contrastive-learning.html
class SimCLR_NCE_Loss(nn.Module): 
    def __init__(self, testmode=False):
        ## z is unspecified, s is specified i.e. stain component
        super().__init__()
        self.testmode = testmode
        self.temperature = 0.07
        self.defrag = {}
        self.mode='multiple'
        
    def forward(self, stain, morph, meta, device, logger=None):
        if logger is not None:
            logger['s std'].append(stain.std(dim=0).mean().item())
            logger['s norm'].append(stain.norm(dim=1).mean().item())
            
        ## prep meta
        enc = LabelEncoder()
        raw_labels = [self._make_name_from_list(s['staining']) for s in meta]
        label = torch.tensor(enc.fit_transform(raw_labels), device=device)
        
        if self.testmode:
            print('######################################## BEGINNING OF META SECTION')
            print(f'Classes: {enc.classes_} before defragmentation')
            [print(f"CLS {cls}; FRQ: {torch.sum(label==i)}") for i, cls in enumerate(enc.classes_)]
        
        ## defragment data
        frqs = {cls:torch.sum(label==i) for i, cls in enumerate(enc.classes_)}
        raw_labels = self._defragment(raw_labels, frqs)
        enc = LabelEncoder()
        label = torch.tensor(enc.fit_transform(raw_labels), device=device)
        frqs = {cls:torch.sum(label==i) for i, cls in enumerate(enc.classes_)}
        
        if self.testmode:
            print(f'Classes: {list(frqs.keys())} after defragmentation')
            [print(f"CLS {cls}; FRQ: {i}") for cls, i in frqs.items()]
            print('######################################## END OF META SECTION')
        
        ## remove singleton stains
        stain, morph, label = self._remove_singles(stain, morph, frqs, label)
            
        # Calculate cosine similarity
        cos_sim_stain = F.cosine_similarity(stain[:, None, :], stain[None, :, :], dim=-1)
        cos_sim_morph = F.cosine_similarity(morph[:, None, :], morph[None, :, :], dim=-1)
        if self.testmode:
            print('shape', 'stain', cos_sim_stain.shape, 'morph', cos_sim_morph.shape)
        
        # mask out self and apply temperature
        self_mask = torch.eye(cos_sim_stain.shape[0], dtype=torch.bool, device=cos_sim_stain.device) 
        cos_sim_stain.masked_fill_(self_mask, -9e15)
        cos_sim_morph.masked_fill_(self_mask, -9e15)
        
        ## loop meta
        pos_mask = []
        neg_mask = []
        for i in range(stain.shape[0]): ## in the loop create the mask for every iteration
            ## multiple pairs per sample
            if self.mode == 'multiple':
                p_msk = torch.tensor(label==label[i])
                p_msk[i]=False
                n_msk = torch.tensor(label==label[i])
                n_msk[i]=False
            
            ## single worst pair per sample
            if self.mode == 'unique':
                p_msk = torch.zeros_like(label, dtype=bool, device=device)
                n_msk = torch.zeros_like(label, dtype=bool, device=device)
                matches = label==label[i] ## find all pairs                    
                matches[i]=False ## mask out self
                
                ## in case there are multiple pairs
                if matches.sum()>=2:
                    indices = torch.argwhere(matches).squeeze() ## find distances at all pairs
                    id = indices[cos_sim_stain[i, indices].argmin().item()] ## find worst distance -> cosine -1=worst; 1=best
                    p_msk[id]=True ## find index in msk with worst dist and write to mask
                    id = indices[cos_sim_morph[i, indices].argmax().item()] ## find best distance -> cosine -1=worst; 1=best
                    n_msk[id]=True ## find index in msk with best dist and write to mask
                ## in case there is only one pair
                else: p_msk = matches; n_msk = matches
                
            pos_mask.append(p_msk); neg_mask.append(n_msk)
            
        pos_mask = torch.stack(pos_mask, dim=1) ## the goal is to attract the worst matches in S
        neg_mask = torch.stack(neg_mask, dim=1) ## the goal is to repel the best matches in Z
        
        # cos_sim_stain/=self.temperature
        # cos_sim_morph/=self.temperature
        
        if self.testmode:
            print('pos_mask_shape', pos_mask.shape)
            print('pos mask count', pos_mask.sum())
            print(pos_mask)
            print(cos_sim_stain)
            print(cos_sim_stain[pos_mask].shape, cos_sim_stain[pos_mask])
            
        # InfoNCE loss for multipair
        if self.mode=='multiple':
            ## repell same stain in z 
            nll_morph = cos_sim_morph[neg_mask].mean()
            
            ## attract same stain in s
            nll_stain = -((cos_sim_stain * pos_mask).sum(dim=-1) / pos_mask.sum(dim=-1).clamp(min=1)) + torch.logsumexp(cos_sim_stain, dim=-1)
            nll_stain = nll_stain.mean()
        
        # InfoNCE loss for singlepair
        if self.mode=='unique':
            ## repell same stain in z 
            nll_morph = cos_sim_morph[neg_mask].mean()
            
            ## attract same stain in s
            nll_stain = -cos_sim_stain[pos_mask] + torch.logsumexp(cos_sim_stain, dim=-1) 
            nll_stain = nll_stain.mean() - math.log(cos_sim_stain.shape[0])
            
        if logger is not None:
            logger['InfoNCE Stain'].append(nll_stain.item())
            logger['InfoNCE Morph'].append(nll_morph.item())
        
        if self.testmode:
            print(f"Stain NLL:  {nll_stain.item()}, Morph NLL:  {nll_morph.item()}")

        return nll_morph+nll_stain, logger
    
    def _make_name_from_list(self, data):
        if isinstance(data, str):
            return data
        return "+".join(data)
    
    def _defragment(self, raw_labels, frq):
        new_labels = []
        for label in raw_labels:
            ## hematox & antibod
            if 'hematoxylin' in label.lower() and not 'eosin' in label.lower():
                if frq[label]>1: new_labels.append(label)
                else: new_labels.append('H&A')
            ## hematox & eosin
            elif label == "HE - Hematoxylin and eosin stain method (procedure)" or label == "Hematoxylin and eosin stain method" or label == "hematoxylin stain+water soluble eosin stain":
                new_labels.append('H&E')
            ## Van Gieson
            elif label == "Van Gieson stain" or label == "Verhoeff-Van Gieson stain method":
                new_labels.append("Van Gieson stain")
            elif "Periodic acid Schiff stain" in label and "blue" not in label:
                new_labels.append("Periodic acid Schiff stain")
            else: new_labels.append(label)
        return new_labels
        ## if a hematoxylin+antibody is unique add to hematox+anti class
        ## defrag any same meaning different name classes like the procedure stuff
    
    def _remove_singles(self, stain, morph, frqs, label):
        ## delete singleton samples
        deldim = []
        for cls, i in frqs.items():
            if i < 2:
                if self.testmode: print(f'Found singleton class at {cls}')
                deldim.append(False)
            else: deldim.append(True)
        deldim = [deldim[l] for l in label]
        deldim = torch.tensor(deldim, device=stain.device)     
        
        if self.testmode:print(f"Dims BEFORE deletion:", f"Stain={stain.shape}", f"Morph={morph.shape}", f"Label={label.shape}")        
        stain = stain[deldim, :]
        morph = morph[deldim, :]
        label = label[deldim]
        if self.testmode:print(f"Dims AFTER deletion:", f"Stain={stain.shape}", f"Morph={morph.shape}", f"Label={label.shape}")
        return stain, morph, label
    
#########################################################################################################################################
## classification based loss

#########################################################################################################################################
## BYOL based https://towardsdatascience.com/byol-the-alternative-to-contrastive-self-supervised-learning-5d0a26983d7c-2/