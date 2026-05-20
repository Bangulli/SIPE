######## Ecosystem ########
import os, sys, pathlib as pl, datetime, json
sys.path.append(os.path.join(os.path.dirname(__file__), "."))
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
from typing import Union
######## External ########
from tqdm import tqdm as ProgBar
import torch
import matplotlib.pyplot as plt
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts
from torch.optim import AdamW
from torch.utils.data import DataLoader
import torch.nn as nn
from BPTorch.utils import bptorch_collate
from sipe.trainer.trainer import Trainer
from sipe.model.arch import H0_mini_for_AutoEncoding
######## Internal ########
##########################

class Inferer:
    def __init__(self, trainer: Union[str, pl.Path, Trainer]):
        self.trainer = self._get_trainer_object(trainer)
        
    def infer(self, dataset): pass
        
    def _get_trainer_object(self, trainer):
        if type(trainer)==Trainer: return trainer
        elif type(trainer)==str or type(trainer)==pl.Path: return Trainer(model=H0_mini_for_AutoEncoding() ,wdir=trainer)