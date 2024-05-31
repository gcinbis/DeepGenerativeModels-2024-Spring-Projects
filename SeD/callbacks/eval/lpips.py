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
import torch
import numpy as np
from tqdm import tqdm
import torch.distributed as dist
from losses.lpips.lpips import LPIPS

class LPIPSCallback(pl.callbacks.Callback):
    
    def __init__(self, dataloader, eval_every=10000, dataset_type="validation"):
        self.eval_every = eval_every
        self.dataloader = dataloader
        self.dataset_type = dataset_type
        assert dataset_type in ["validation", "test"]
        self.metric_log_name = "lpips"
        self.metric_log_name += "_val" if self.dataset_type == "validation" else "_test"
        self.last_lpips = 0.0 # Arbitrary value to log on devices other than rank 0, which broadcasts the LPIPS value
        # TODO: With if rank 0 condition in broadcast phase, use of self.last_lpips can be removed for a better code

    @rank_zero_only
    def on_train_start(self, trainer, pl_module):
        self.device = pl_module.device
        self.lpips_model = LPIPS(net_type='alex', device=self.device).to('cpu')
        self.lpips_model.eval()
        
    @rank_zero_only
    def get_lpips_mean(self, pl_module):
        pl_module.eval()
        self.lpips_model.to(self.device)
        lpips_list = []
        with torch.no_grad():
            for batch in tqdm(self.dataloader, desc=f"Calculating {self.dataset_type} LPIPS on sr images", total=len(self.dataloader)):
                sr_imges = pl_module.make_high_resolution(batch)["generated_super_resolution_image"].to(self.device) * 0.5 + 0.5
                gt_image = batch["image_hr"].to(self.device) * 0.5 + 0.5
                lpips = self.lpips_model(sr_imges, gt_image, return_similarity=True)
                lpips_list.append(lpips.cpu())
        lpips_list = torch.cat(lpips_list).numpy()
        lpips_mean = np.nanmean(lpips_list)
        self.lpips_model.to('cpu')
        pl_module.train()
        return lpips_mean
    
    @rank_zero_only
    def calculate_and_update_lpips(self, pl_module):
        lpips_mean = self.get_lpips_mean(pl_module)
        self.last_lpips = lpips_mean

    def on_train_batch_start(self, trainer, pl_module, *args, **kwargs):
        current_step = trainer.global_step // 2 # x2 step on each iter due to GAN loss
        if (current_step + 1) % self.eval_every != 0 and current_step != 0:
            return
        self.calculate_and_update_lpips(pl_module) # Using only rank 0 for the calculation

        if torch.distributed.is_initialized():  # If distributed training is running
            dist.barrier() # Wait until rank 0 completes the calculation
            local_lpips = torch.tensor(self.last_lpips, dtype=torch.float32).cuda() # Convert LPIPS to tensor for broadcasting
            dist.broadcast(local_lpips, src=0) # Broadcast the LPIPS value from rank 0 to other processes (Sync)
            self.last_lpips = local_lpips.item()

        pl_module.eval_metric_dict[self.metric_log_name] = self.last_lpips