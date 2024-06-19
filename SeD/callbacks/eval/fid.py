"""
StarGAN v2
Copyright (c) 2020-present NAVER Corp.
This work is licensed under the Creative Commons Attribution-NonCommercial
4.0 International License. To view a copy of this license, visit
http://creativecommons.org/licenses/by-nc/4.0/ or send a letter to
Creative Commons, PO Box 1866, Mountain View, CA 94042, USA.
"""

import pytorch_lightning as pl
from pytorch_lightning.utilities import rank_zero_only
from scipy import linalg
import torch
import numpy as np
from .models.inception_v3 import InceptionV3
from tqdm import tqdm
import torch.distributed as dist

class FIDCallback(pl.callbacks.Callback):
    
    def __init__(self, dataloader, eval_every=10000, dataset_type="validation"):
        self.eval_every = eval_every
        self.dataloader = dataloader
        self.dataset_type = dataset_type
        assert dataset_type in ["validation", "test"]
        self.metric_log_name = "fid"
        self.metric_log_name += "_val" if self.dataset_type == "validation" else "_test"
        self.last_fid = 0.0 # Arbitrary value to log on devices other than rank 0, which broadcasts the FID value
        # TODO: With if rank 0 condition in broadcast phase, use of self.last_fid can be removed for a better code

    @rank_zero_only
    def on_train_start(self, trainer, pl_module):
        self.inception = InceptionV3().to('cpu')
        self.inception.eval()
        self.device = pl_module.device
        self.real_stats = self.get_dataset_stats()
        
    @rank_zero_only
    def get_sr_image_stats(self, pl_module):
        pl_module.eval()
        self.inception.to(self.device)
        actvs = []
        with torch.no_grad():
            for batch in tqdm(self.dataloader, desc=f"Calculating {self.dataset_type} FID on sr images", total=len(self.dataloader)):
                sr_images = pl_module.make_high_resolution(batch)["image_lr"]
                actv = self.inception(sr_images.to(self.device))
                actvs.append(actv.cpu())
        actvs = torch.cat(actvs, dim=0).numpy()
        mean = np.mean(actvs, axis=0)
        cov = np.cov(actvs, rowvar=False)
        self.inception.to('cpu')
        pl_module.train()
        return {'mean': mean, 'cov':cov}

    @rank_zero_only
    def get_dataset_stats(self):
        self.inception.to(self.device)
        actvs = []
        with torch.no_grad():
            for image_batch in tqdm(self.dataloader, desc=f"Calculating {self.dataset_type} FID on dataset images", total=len(self.dataloader)):
                actv = self.inception(image_batch["image_hr"].to(self.device))
                actvs.append(actv.cpu())
        actvs = torch.cat(actvs, dim=0).numpy()
        mean = np.mean(actvs, axis=0)
        cov = np.cov(actvs, rowvar=False)
        self.inception.to('cpu')
        return {'mean': mean, 'cov':cov}
        
    @rank_zero_only
    def frechet_distance(self, real_stats, fake_stats):
        mu, cov = real_stats['mean'], real_stats['cov']
        mu2, cov2 = fake_stats['mean'], fake_stats['cov']
        cc, _ = linalg.sqrtm(np.dot(cov, cov2), disp=False)
        dist = np.sum((mu - mu2)**2) + np.trace(cov + cov2 - 2*cc)
        return np.real(dist)

    
    @rank_zero_only
    def calculate_and_update_fid(self, pl_module):
        fake_stats = self.get_sr_image_stats(pl_module)
        fid = self.frechet_distance(self.real_stats, fake_stats)
        self.last_fid = fid

    def on_train_batch_start(self, trainer, pl_module, *args, **kwargs):
        current_step = trainer.global_step // 2 # x2 step on each iter due to GAN loss
        if (current_step + 1) % self.eval_every != 0 and current_step != 0:
            return
        self.calculate_and_update_fid(pl_module) # Using only rank 0 for the calculation

        if torch.distributed.is_initialized():  # If distributed training is running
            dist.barrier() # Wait until rank 0 completes the calculation
            local_fid = torch.tensor(self.last_fid, dtype=torch.float32).cuda() # Convert FID to tensor for broadcasting
            dist.broadcast(local_fid, src=0) # Broadcast the FID value from rank 0 to other processes (Sync)
            self.last_fid = local_fid.item()
        
        pl_module.eval_metric_dict[self.metric_log_name] = self.last_fid