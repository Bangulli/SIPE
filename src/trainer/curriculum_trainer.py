######## Ecosystem ########
import os, sys, pathlib as pl, datetime, json
from operator import itemgetter
sys.path.append(os.path.join(os.path.dirname(__file__), "."))
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
from collections.abc import Iterable
######## External ########
from tqdm import tqdm as ProgBar
import torch
import matplotlib.pyplot as plt
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts
from torch.optim import AdamW
from torch.utils.data import DataLoader
import torch.nn as nn
from BPTorch.utils import bptorch_collate
######## Internal ########
##########################

class Curriculum(list):
    def __init__(self):
        super().__init__()
        
    def add_step(self, step_type='recon', epochs=5, adverse_alpha=0.1, lr=3e-4, restarts=5, norm=True):
        self.append({
            'type': step_type,
            'epochs': epochs,
            'lr': lr,
            'adverse_alpha': adverse_alpha,
            'restarts': restarts,
            'adverse_norm': norm,
        })
        
    def save(self, path):
        with open(path, 'w') as f:
            json.dump(self, f, indent=4)
            
    @classmethod
    def load(cls, path):
        with open(path, 'r') as f:
            data = json.load(f)
        curriculum = cls()
        curriculum.extend(data)
        return curriculum
    
    def extend(self, cr):
        self += cr

class CurriculumTrainer:
    def __init__(self, model, loss_recon, loss_adverse, 
                 wdir='trainer',
                 scheduler=CosineAnnealingWarmRestarts,
                 optim=AdamW,
                 from_scratch = False,
                 device = 'cuda:1'
                 ):
        self.wdir = self._prep_wdir(wdir) if from_scratch else pl.Path(wdir)
        self.model = model
        self.loss_r = loss_recon
        self.loss_a = loss_adverse
        self.scheduler_base = scheduler
        self.optim_base = optim
        self.loss_history = {
            'training': [],
            'validation': []
        }
        self.device = device
        
    def _prep_wdir(self, pth):
        pth = pl.Path(pth)
        if os.path.exists(pth):
            override = datetime.datetime.now().strftime(r'trainer_from_%H:%M:%S-%d.%m.%y')
            print(f"INFO: {pth} already exists, using {pth.parent/override} instead")
            pth=pth.parent/override
        os.mkdir(pth)
        return pth
    
    def _save_ckpt(self, ckpt_dir, epoch):
        self.model.save(self.wdir/ckpt_dir/f"ckpt_from_epoch_{epoch}", overwrite=True)
        
    def _plt_progress(self):
        plt.plot(self.loss_history['training'], label='training loss')
        plt.plot(self.loss_history['validation'], label='validation loss')
        plt.ylabel('loss')
        plt.xlabel('epoch')
        plt.legend()
        plt.title(f'Training/Validation loss per epoch')
        plt.savefig(self.wdir/'training_losses.png')
        plt.close()
        plt.clf()
        
    def _plt_batch_progress(self, batch_losses, ckpt_dir, epoch):
        os.makedirs(self.wdir/ckpt_dir/f"ckpt_from_epoch_{epoch}", exist_ok=True)
        plt.plot(batch_losses, label='training loss')
        plt.ylabel('loss')
        plt.xlabel('batch')
        plt.legend()
        plt.title(f'Training loss per batch')
        plt.savefig(self.wdir/ckpt_dir/f"ckpt_from_epoch_{epoch}/batched_losses.png")
        plt.close()
        plt.clf()
        
    def _plt_individual_loss_progress(self, logger, ckpt_dir, epoch):
        os.makedirs(self.wdir/ckpt_dir/f"ckpt_from_epoch_{epoch}", exist_ok=True)
        [plt.plot(v, label=k) for k, v in logger.items() if any(v)]
        plt.ylabel('loss')
        plt.xlabel('batch')
        plt.legend()
        plt.title(f'Training loss per batch')
        plt.tight_layout()
        plt.savefig(self.wdir/ckpt_dir/f"ckpt_from_epoch_{epoch}/batched_individual_losses.png")
        plt.close()
        plt.clf()
        with open(self.wdir/ckpt_dir/f"ckpt_from_epoch_{epoch}/batched_individual_losses.json", "w") as f:
            json.dump(logger, f, indent=4)
        
    def save(self):
        with open(self.wdir/'history.json', 'w') as f:
            json.dump(self.loss_history, f, indent=4)
        torch.save(self.optim, self.wdir/'optim.bin')
        torch.save(self.scheduler, self.wdir/'scheduler.bin')
        #torch.save(self.loss, self.wdir/'loss.bin')
        
    def load(self, ckpt_dir, ckpt):
        if os.path.exists(self.wdir/'history.json'):
            with open(self.wdir/'history.json', 'r') as f:
                self.loss_history = json.load(f)
        self.model.load(self.wdir/ckpt_dir/f"ckpt_from_epoch_{ckpt}")
        
    def train(self, train, val, curriculum, ckpt_dir='checkpoints', batch_size=32):
        os.makedirs(self.wdir/ckpt_dir, exist_ok=True)
        
        if not os.path.exists(self.wdir/ckpt_dir/'curriculum.json'): curriculum.save(self.wdir/ckpt_dir/'curriculum.json')
        else: 
            extended_cr = Curriculum.load(self.wdir/ckpt_dir/'curriculum.json')
            extended_cr.extend(curriculum)
            extended_cr.save(self.wdir/ckpt_dir/'curriculum.json')
        
        train_loader = DataLoader(train, batch_size, collate_fn=bptorch_collate, shuffle=True) ## needs pre extracted patches to run efficiently
        val_loader = DataLoader(val, batch_size, collate_fn=bptorch_collate)
        self.model.to(self.device)
        
        strt = len([d for d in os.listdir(self.wdir/ckpt_dir) if (self.wdir/ckpt_dir/d).is_dir()])+1
        for i, step in enumerate(curriculum):
            step_type, epochs, lr, adverse_alpha, restarts, adv_norm = itemgetter('type', 'epochs', 'lr', 'adverse_alpha', 'restarts', 'adverse_norm')(step)
            self.optim = self.optim_base(self.model.parameters(), lr=lr)
            self.scheduler = self.scheduler_base(self.optim, restarts)
            if step_type.lower()=='recon': self._train_recon(strt, epochs, train_loader, val_loader, ckpt_dir, i)
            elif step_type.lower()=='adverse': self._train_adverse(strt, epochs, train_loader, val_loader, ckpt_dir, adverse_alpha, adv_norm, i)
            else: raise ValueError(f'Allowed step types are [recon, adverse], but got {step_type.lower()} instead')
            strt += epochs
        
        
    def _train_recon(self, strt, epochs, train_loader, val_loader, ckpt_dir, step):
        ######### recon step ########################################################################
        self.model.freeze_or_unfreeze_disentangler(True)
        for epoch in range(strt, epochs+strt):
            logger = {
                'Recon Img': [],
                'Recon Morph': [],
                'InfoNCE Stain': [],
                'InfoNCE Morph': [],
                'Adversarial CE': [],
                'CE':[],
                's std': [],
                's norm': [],
            }
            print(f'---------------------- Step: {step} {epoch}/{epochs+strt-1} - Recon ----------------------')
            with torch.enable_grad():
                self.model.train()
                batch_losses = []
                for i, batch in enumerate(ProgBar(train_loader, desc=f'Training Batches')):
                    self.optim.zero_grad()
                    loss, logger = self.model.loss(batch, self.loss_r, logger, val=False)
                    #print(f'Batch: {i} - Loss: {loss.item()}')
                    batch_losses.append(loss.item())
                    loss.backward()
                    self.optim.step()
                    if i%20==0:
                        self._save_ckpt(ckpt_dir, epoch)
                        self._plt_batch_progress(batch_losses, ckpt_dir, epoch)
                        self._plt_individual_loss_progress(logger, ckpt_dir, epoch)
                    
                self.scheduler.step()
                self.loss_history['training'].append(sum(batch_losses)/len(batch_losses))
            
                            ## validation step
            with torch.no_grad():
                logger = {
                    'Recon Img': [],
                    'Recon Morph': [],
                    'InfoNCE Stain': [],
                    'InfoNCE Morph': [],
                    'Adversarial CE': [],
                    'CE':[],
                    's std': [],
                    's norm': [],
                }
                self.model.eval()
                val_losses = []
                for batch in ProgBar(val_loader, desc=f'Validating Batches'):
                    loss, logger = self.model.loss(batch, self.loss_a, logger, val=True) ## add validation flag because the adversarial loss breaks the logic, so its not evaluated.
                    val_losses.append(loss.item())
                self.loss_history['validation'].append(sum(val_losses)/len(val_losses))
                
            ## save checkpoint
            self._save_ckpt(ckpt_dir, epoch)
            self.save()
            self._plt_progress()
            
    def _train_adverse(self, strt, epochs, train_loader, val_loader, ckpt_dir, alpha, norm, step):
            ######### adversarial step ########################################################################
            self.model.freeze_or_unfreeze_disentangler(False) 
            if type(alpha)!=float: assert len(alpha)==len(range(strt, epochs+strt)), 'Size of alphas list has to match amount of epochs.'
            for i, epoch in enumerate(range(strt, epochs+strt)):
                logger = {
                    'Recon Img': [],
                    'Recon Morph': [],
                    'InfoNCE Stain': [],
                    'InfoNCE Morph': [],
                    'Adversarial CE': [],
                    'CE':[],
                    's std': [],
                    's norm': [],
                }
                if type(alpha)==float: self.loss_a.set_adverse_alpha(alpha)
                else: self.loss_a.set_adverse_alpha(alpha[i])
                cur_alpha = alpha if type(alpha)==float else alpha[i]
                self.loss_a.set_adverse_norm(norm)
                print(f'---------------------- Step: {step} {epoch}/{epochs+strt-1} - Adverse - Alpha: {cur_alpha:.2f} ----------------------')
                with torch.enable_grad():
                    self.model.train()
                    batch_losses = []
                    for i, batch in enumerate(ProgBar(train_loader, desc=f'Training Batches')):
                        self.optim.zero_grad()
                        loss, logger = self.model.loss(batch, self.loss_a, logger, val=False)
                        #print(f'Batch: {i} - Loss: {loss.item()}')
                        batch_losses.append(loss.item())
                        loss.backward()
                        self.optim.step()
                        if i%20==0:
                            self._save_ckpt(ckpt_dir, epoch)
                            self._plt_batch_progress(batch_losses, ckpt_dir, epoch)
                            self._plt_individual_loss_progress(logger, ckpt_dir, epoch)
                        
                    self.scheduler.step()
                    self.loss_history['training'].append(sum(batch_losses)/len(batch_losses))
            
                ## validation step
                with torch.no_grad():
                    logger = {
                        'Recon Img': [],
                        'Recon Morph': [],
                        'InfoNCE Stain': [],
                        'InfoNCE Morph': [],
                        'Adversarial CE': [],
                        'CE':[],
                        's std': [],
                        's norm': [],
                    }
                    self.model.eval()
                    val_losses = []
                    for batch in ProgBar(val_loader, desc=f'Validating Batches'):
                        loss, logger = self.model.loss(batch, self.loss_a, logger, val=True) ## add validation flag because the adversarial loss breaks the logic, so its not evaluated.
                        val_losses.append(loss.item())
                    self.loss_history['validation'].append(sum(val_losses)/len(val_losses))
                    
                ## save checkpoint
                self._save_ckpt(ckpt_dir, epoch)
                self.save()
                self._plt_progress()
            
    def load_best_model(self, ckpt_dir='checkpoints'):
        if os.path.exists(self.wdir/'history.json'):
            with open(self.wdir/'history.json', 'r') as f:
                self.loss_history = json.load(f)
            values = self.loss_history['validation']
            best_epoch = min(range(len(values)), key=values.__getitem__)+1 ## epochs are 1 based indexed.
            print(f'Best loss value of {values[best_epoch-1]} was achieved at epoch {best_epoch}')
            self.load(ckpt_dir, best_epoch)
            return self.model
        elif os.path.exists(self.wdir/ckpt_dir/'ckpt_from_epoch_1'):
            print(f'Cant find loss history but falling back to available checkpoint')
            self.load(ckpt_dir, 1)
            return self.model
        else: raise RuntimeError('No loss history found to infer the best model from')
        
    def load_model_at_epoch(self, epoch, ckpt_dir='checkpoints'):
        if epoch>-1:
            self.load(ckpt_dir, epoch)
            return self.model
        else:
            epoch = len(os.listdir(self.wdir/ckpt_dir))
            print('Loading latest model from epoch', epoch)
            self.load(ckpt_dir, epoch)
            return self.model