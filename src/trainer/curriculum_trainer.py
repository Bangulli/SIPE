######## Ecosystem ########
import os, sys, pathlib as pl, datetime, json
from operator import itemgetter
sys.path.append(os.path.join(os.path.dirname(__file__), "."))
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
from collections.abc import Iterable
from pprint import pprint
######## External ########
from tqdm import tqdm as ProgBar
import torch
import matplotlib.pyplot as plt
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts
from torch.optim import AdamW
from torch.utils.data import DataLoader
import torch.nn as nn
import torch.nn.functional as F
from BPTorch.utils import bptorch_collate
import copy
######## Internal ########
from src.losses.image_recon_loss import GAN_Loss
from src.model.arch import H0_mini_for_Adversarial
from src.model.arch_cls import H0_mini_for_Adversarial_on_CLS
from src.model.vae import H0_mini_for_VAE
##########################

class Curriculum(list):
    def __init__(self):
        super().__init__()
        
    def add_step(self, step_type='recon', epochs=5, adverse_alpha=0.1, lr=3e-4, restarts=5, norm=True, freeze_bb=True, freeze_tangler=True):
        self.append({
            'type': step_type,
            'epochs': epochs,
            'lr': lr,
            'adverse_alpha': adverse_alpha,
            'restarts': restarts,
            'adverse_norm': norm,
            'freeze_backbone': freeze_bb,
            'freeze_tangler': freeze_tangler
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
    def __init__(self, model, loss_recon, loss_adverse, loss_cycle=None,
                 wdir='trainer',
                 scheduler=CosineAnnealingWarmRestarts,
                 optim=AdamW,
                 device = 'cuda:1'
                 ):
        self.wdir = pl.Path(wdir)
        os.makedirs(self.wdir, exist_ok=True)
        if isinstance(model, CurriculumTrainer): 
            print(f'Forking model from {model.wdir}')
            self.model, epoch = model._load_pretrained()
            with open(self.wdir/'NOTE.txt', 'w') as f:
                f.write(f'Forked from {model.wdir}, epoch {epoch}')
        else: self.model = model
        self.loss_r = loss_recon
        self.loss_a = loss_adverse
        self.loss_c = loss_cycle
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
        
    def _save_history(self):
        with open(self.wdir/'history.json', 'w') as f:
            json.dump(self.loss_history, f, indent=4)
        # torch.save(self.optim, self.wdir/'optim.bin')
        # torch.save(self.scheduler, self.wdir/'scheduler.bin')
        #torch.save(self.loss, self.wdir/'loss.bin')
        
    def load(self, ckpt_dir, ckpt):
        if os.path.exists(self.wdir/'history.json'):
            with open(self.wdir/'history.json', 'r') as f:
                self.loss_history = json.load(f)
        self.model.load(self.wdir/ckpt_dir/f"ckpt_from_epoch_{ckpt}")
        
    def train(self, train, val, curriculum, ckpt_dir='checkpoints', batch_size=32):
        os.makedirs(self.wdir/ckpt_dir, exist_ok=True)
        
        ## send perceptive loss to device
        if isinstance(self.loss_a.image_recon_loss, GAN_Loss): self.loss_a.image_recon_loss.to(self.device)
        if isinstance(self.loss_c.image_recon_loss, GAN_Loss): self.loss_c.image_recon_loss.to(self.device)
        if isinstance(self.loss_r.image_recon_loss, GAN_Loss): self.loss_r.image_recon_loss.to(self.device)
        
        if not os.path.exists(self.wdir/ckpt_dir/'curriculum.json'): curriculum.save(self.wdir/ckpt_dir/'curriculum.json')
        else: 
            extended_cr = Curriculum.load(self.wdir/ckpt_dir/'curriculum.json')
            extended_cr.extend(curriculum)
            extended_cr.save(self.wdir/ckpt_dir/'curriculum.json')
            
        if os.path.exists(self.wdir/'history.json'):
            with open(self.wdir/'history.json', 'r') as f:
                self.loss_history = json.load(f)
            self.model = self.load_model_at_epoch(-1)
        
        train_loader = DataLoader(train, batch_size, collate_fn=bptorch_collate, shuffle=True) ## needs pre extracted patches to run efficiently
        val_loader = DataLoader(val, batch_size, collate_fn=bptorch_collate)
        self.model.to(self.device)
        
        strt = len([d for d in os.listdir(self.wdir/ckpt_dir) if (self.wdir/ckpt_dir/d).is_dir()])+1
        for i, step in enumerate(curriculum):
            step_type, epochs, lr, adverse_alpha, restarts, adv_norm, freeze_bb, freeze_tangler = itemgetter('type', 'epochs', 'lr', 'adverse_alpha', 'restarts', 'adverse_norm', 'freeze_backbone', 'freeze_tangler')(step)
            self.optim = self.optim_base(self.model.parameters(), lr=lr)
            self.scheduler = self.scheduler_base(self.optim, restarts)
            self.model.freeze_backbone(freeze_bb)
            self.model.freeze_or_unfreeze_disentangler(freeze_tangler) 
            if step_type.lower()=='recon': self._train_recon(strt, epochs, train_loader, val_loader, ckpt_dir, i)
            elif step_type.lower()=='adverse': self._train_adverse(strt, epochs, train_loader, val_loader, ckpt_dir, adverse_alpha, adv_norm, i)
            elif step_type.lower()=='cycle': 
                assert self.loss_c is not None, "Can't train a cycle paradigm when no cycle loss function has been passed to the trainer."
                self._train_cycle(strt, epochs, train_loader, val_loader, ckpt_dir, adverse_alpha, adv_norm, i)
            elif step_type.lower()=='vae': self._train_VAE(strt, epochs, train_loader, val_loader, ckpt_dir, adverse_alpha, adv_norm, i)
            else: raise ValueError(f'Allowed step types are [recon, adverse], but got {step_type.lower()} instead')
            strt += epochs
        
        
    def _train_recon(self, strt, epochs, train_loader, val_loader, ckpt_dir, step):
        ######### recon step ########################################################################
        for epoch in range(strt, epochs+strt):
            logger = {
                'Recon Img': [],
                'Stain probs2vec': [],
                'InfoNCE Stain': [],
                'InfoNCE Morph': [],
                'Adversarial CE': [],
                'CE':[],
                's std': [],
                's norm': [],
                'KDE':[]
            }
            print(f'---------------------- Step: {step} - {epoch}/{epochs+strt-1} - Recon ----------------------')
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
                    'Stain probs2vec': [],
                    'InfoNCE Stain': [],
                    'InfoNCE Morph': [],
                    'Adversarial CE': [],
                    'CE':[],
                    's std': [],
                    's norm': [],
                    'KDE':[]
                }
                self.model.eval()
                val_losses = []
                for batch in ProgBar(val_loader, desc=f'Validating Batches'):
                    loss, logger = self.model.loss(batch, self.loss_a, logger, val=True) ## add validation flag because the adversarial loss breaks the logic, so its not evaluated.
                    val_losses.append(loss.item())
                self.loss_history['validation'].append(sum(val_losses)/len(val_losses))
                
            ## save checkpoint
            self._save_ckpt(ckpt_dir, epoch)
            self._save_history()
            self._plt_progress()
            
    def _train_adverse(self, strt, epochs, train_loader, val_loader, ckpt_dir, alpha, norm, step):
            ######### adversarial step ########################################################################
            if type(alpha)!=float: assert len(alpha)==len(range(strt, epochs+strt)), 'Size of alphas list has to match amount of epochs.'
            for i, epoch in enumerate(range(strt, epochs+strt)):
                logger = {
                    'Recon Img': [],
                    'Stain probs2vec': [],
                    'InfoNCE Stain': [],
                    'InfoNCE Morph': [],
                    'Adversarial CE': [],
                    'CE':[],
                    's std': [],
                    's norm': [],
                    'KDE':[]
                }
                if type(alpha)==float: self.loss_a.set_adverse_alpha(alpha)
                else: self.loss_a.set_adverse_alpha(alpha[i])
                cur_alpha = alpha if type(alpha)==float else alpha[i]
                self.loss_a.set_adverse_norm(norm)
                print(f'---------------------- Step: {step} - {epoch}/{epochs+strt-1} - Adverse - Alpha: {cur_alpha:.2f} ----------------------')
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
                        'Stain probs2vec': [],
                        'InfoNCE Stain': [],
                        'InfoNCE Morph': [],
                        'Adversarial CE': [],
                        'CE':[],
                        's std': [],
                        's norm': [],
                        'KDE':[]
                    }
                    self.model.eval()
                    val_losses = []
                    for batch in ProgBar(val_loader, desc=f'Validating Batches'):
                        loss, logger = self.model.loss(batch, self.loss_a, logger, val=True) ## add validation flag because the adversarial loss breaks the logic, so its not evaluated.
                        val_losses.append(loss.item())
                    self.loss_history['validation'].append(sum(val_losses)/len(val_losses))
                    
                ## save checkpoint
                self._save_ckpt(ckpt_dir, epoch)
                self._save_history()
                self._plt_progress()
                
    def _train_VAE(self, strt, epochs, train_loader, val_loader, ckpt_dir, alpha, norm, step):
        if not isinstance(self.model, H0_mini_for_VAE): raise ValueError('Cant run VAE training on non VAE model')
        ######### adversarial step ########################################################################
        if type(alpha)!=float: assert len(alpha)==len(range(strt, epochs+strt)), 'Size of alphas list has to match amount of epochs.'
        for i, epoch in enumerate(range(strt, epochs+strt)):
            logger = {
                'Recon Img': [],
                'Stain probs2vec': [],
                'InfoNCE Stain': [],
                'InfoNCE Morph': [],
                'Adversarial CE': [],
                'CE':[],
                's std': [],
                's norm': [],
                'KDE':[],
                'MSE':[]
            }
            if type(alpha)==float: self.loss_a.set_adverse_alpha(alpha)
            else: self.loss_a.set_adverse_alpha(alpha[i])
            cur_alpha = alpha if type(alpha)==float else alpha[i]
            self.loss_a.set_adverse_norm(norm)
            print(f'---------------------- Step: {step} - {epoch}/{epochs+strt-1} - VAE - Alpha: {cur_alpha:.2f} ----------------------')
            with torch.enable_grad():
                self.model.train()
                batch_losses = []
                for i, batch in enumerate(ProgBar(train_loader, desc=f'Training Batches')):
                    self.optim.zero_grad()
                    emb = self.model.backbone(batch['image'].to(self.model.device))
                    s, z, mu, logvar = self.model.entangler.disentangle(emb)
                    emb_orig = emb[:,5:,:].permute(0, 2, 1) ## cut out cls and register tokens
                    emb_orig = emb_orig.reshape(emb_orig.shape[0], 768, 16, 16) ## reshape to feature map 
                    emb_recon = self.model.entangler.reentangle(s, z)
                    
                    mse = F.mse_loss(emb_recon, emb_orig, reduction='sum')
                    logger['MSE'].append(mse.item())
                    kde = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
                    logger['KDE'].append(kde.item())
                    loss = mse+kde
                    
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
                    'Stain probs2vec': [],
                    'InfoNCE Stain': [],
                    'InfoNCE Morph': [],
                    'Adversarial CE': [],
                    'CE':[],
                    's std': [],
                    's norm': [],
                    'KDE':[],
                    'MSE':[]
                }
                self.model.eval()
                val_losses = []
                for batch in ProgBar(val_loader, desc=f'Validating Batches'):
                    emb = self.model.backbone(batch['image'].to(self.model.device))
                    s, z, mu, logvar = self.model.entangler.disentangle(emb)
                    emb_orig = emb[:,5:,:].permute(0, 2, 1) ## cut out cls and register tokens
                    emb_orig = emb_orig.reshape(emb_orig.shape[0], 768, 16, 16) ## reshape to feature map 
                    emb_recon = self.model.entangler.reentangle(s, z)
                    
                    mse = F.mse_loss(emb_recon, emb_orig, reduction='sum')
                    logger['MSE'].append(mse.item())
                    kde = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
                    logger['KDE'].append(kde.item())
                    loss = mse+kde
                    
                    val_losses.append(loss.item())
                self.loss_history['validation'].append(sum(val_losses)/len(val_losses))
                
            ## save checkpoint
            self._save_ckpt(ckpt_dir, epoch)
            self._save_history()
            self._plt_progress()
                
    def _train_cycle(self, strt, epochs, train_loader, val_loader, ckpt_dir, alpha, norm, step):
        self.loss_c.to(self.device)
        ######### recon step ########################################################################
        for i, epoch in enumerate(range(strt, epochs+strt)):
            logger = {
                    'Recon Img': [],
                    'S cycle': [],
                    'Z cycle': [],
                    'Adversarial CE': [],
                    'CE':[],
                    'KDE':[]
                }
            if type(alpha)==float: self.loss_c.set_adverse_alpha(alpha)
            else: self.loss_c.set_adverse_alpha(alpha[i])
            cur_alpha = alpha if type(alpha)==float else alpha[i]
            self.loss_c.set_adverse_norm(norm)
            print(f'---------------------- Step: {step} - {epoch}/{epochs+strt-1} - alpha: {cur_alpha:.2f} - Cycle ----------------------')
            with torch.enable_grad():
                self.model.train()
                batch_losses = []
                for i, batch in enumerate(ProgBar(train_loader, desc=f'Training Batches')):
                    self.optim.zero_grad()
                    
                    loss, logger = self._compute_cycle_loss_for_batch(batch, logger)
                    
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
                    'S cycle': [],
                    'Z cycle': [],
                    'Adversarial CE': [],
                    'CE':[],
                    'KDE':[]
                }
                self.model.eval()
                val_losses = []
                for batch in ProgBar(val_loader, desc=f'Validating Batches'):
                    loss, logger = self._compute_cycle_loss_for_batch(batch, logger, True)
                    val_losses.append(loss.item())
                self.loss_history['validation'].append(sum(val_losses)/len(val_losses))
                
            ## save checkpoint
            self._save_ckpt(ckpt_dir, epoch)
            self._save_history()
            self._plt_progress()
            
    def _compute_cycle_loss_for_batch(self, batch, logger=None, val=False):
        ## comp adverse loss
        adv_loss, logger = self.model.loss(batch, self.loss_a, logger, val=False)
        
        ## encode initial batch
        batch1 = batch
        s1, z1 = self.model(batch1)
        
        ## shift along the batch dimension to mix and specified/unspecified pairs and create second batch
        s1_prime = torch.roll(s1, 1, 0) 
        meta_prime = self._roll_list(copy.deepcopy(batch1['metadata']), 1)
        batch2= {'image':self.model.recon_image(s1_prime, z1).detach(), 'metadata':meta_prime} ## detach to avoid gradient flow through first pass
        
        ## encode second batch
        s2, z2 = self.model(batch2)
        
        ## unshift to re-establish correspondence
        s2_prime = torch.roll(s2, -1, 0)
        
        ## comp cycle loss
        cycle_loss, logger = self.loss_c(s1, s2_prime, z1, z2, logger)
        
        ## compute loss
        return adv_loss+cycle_loss, logger
        
    def _roll_list(self, lst, shifts): ## Equivalent to torch.roll(tensor, shifts, dim=0)
        if shifts > 0:
            for idx in range(shifts):
                lst.insert(0, lst.pop(-1))
        else:
            for idx in range(abs(shifts)):
                lst.append(lst.pop(0))
        return lst
    
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
            with open(self.wdir/'history.json', 'r') as f:
                self.loss_history = json.load(f)
            epoch = len(self.loss_history['validation'])
            print('Loading latest model from epoch', epoch)
            self.load(ckpt_dir, epoch)
            return self.model

    def _load_pretrained(self, ckpt_dir='checkpoints'):
        if os.path.exists(self.wdir/'history.json'):
            with open(self.wdir/'history.json', 'r') as f:
                self.loss_history = json.load(f)
            values = self.loss_history['validation']
            best_epoch = min(range(len(values)), key=values.__getitem__)+1 ## epochs are 1 based indexed.
            print(f'Best loss value of {values[best_epoch-1]} was achieved at epoch {best_epoch}')
            self.load(ckpt_dir, best_epoch)
            return self.model, best_epoch
        elif os.path.exists(self.wdir/ckpt_dir/'ckpt_from_epoch_1'):
            print(f'Cant find loss history but falling back to available checkpoint')
            self.load(ckpt_dir, 1)
            return self.model, 1
        else: raise RuntimeError('No loss history found to infer the best model from')
        