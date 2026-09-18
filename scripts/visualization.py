import copy
import json
import os
import pathlib as pl
from pprint import pprint

import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F
import tqdm
from BPTorch.datasets import BigPictureRepository, WsiDicomDataset
from BPTorch.utils import bptorch_collate
from torch.utils.data import DataLoader
from torchvision.transforms import ToPILImage

from sipe.model.arch import H0_mini_for_Adversarial
from sipe.trainer.trainer import Trainer
from sipe.utils.stain_comp import compare
from sipe.utils.test import test
from sipe.utils.transfroms import UnNormalize

# pip install "BPTorch @ git+https://github.com/Bangulli/BPTorch"

if __name__ == "__main__":
    sourcedir = pl.Path("/home/lorenz/BigPicture/SIPE/SIPE-1M-Curriculum")

    print(f"Running a quick and dirty test for trainer at {sourcedir}")

    with open("/home/lorenz/BigPicture/SIPE/classes.json", "r") as f:
        classes = json.load(f)
    ## setup variables
    trainer = Trainer(
        H0_mini_for_Adversarial(classes, device="cuda:0"), None, wdir=sourcedir
    )

    ## vis best model in dir
    if (sourcedir / "history.json").exists():
        model = trainer.load_best_model()
    else:
        model = trainer.load_model_at_epoch(1)
    test(model, sourcedir, "images_best")
    compare(model, sourcedir, "images_best")

    # ## vis latest
    # model = trainer.load_model_at_epoch(-1)
    # test(model, sourcedir, 'images_latest')
    # compare(model, sourcedir, 'images_latest')
