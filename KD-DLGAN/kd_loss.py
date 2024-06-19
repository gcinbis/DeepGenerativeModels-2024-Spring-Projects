import numpy as np
import torch
from . import CLIP
from .loss import Loss
from .text_labels import TextLabels
import torch.nn as nn
import torchvision.transforms as T
from torch_utils import training_stats
from torch_utils import misc
from torch_utils.ops import conv2d_gradfix

from DiffAugment_pytorch import DiffAugment

#----------------------------------------------------------------------------

class KDLoss(Loss):
    def __init__(self, device, G_mapping, G_synthesis, D, diffaugment='', augment_pipe=None, style_mixing_prob=0.9, r1_gamma=10, pl_batch_shrink=2, pl_decay=0.01, pl_weight=2, AGKD = False, CGKD = False):
        super().__init__()
        self.device = device
        self.G_mapping = G_mapping
        self.G_synthesis = G_synthesis
        self.D = D
        self.diffaugment = diffaugment
        self.augment_pipe = augment_pipe
        self.style_mixing_prob = style_mixing_prob
        self.r1_gamma = r1_gamma
        self.pl_batch_shrink = pl_batch_shrink
        self.pl_decay = pl_decay
        self.pl_weight = pl_weight
        self.pl_mean = torch.zeros([], device=device)
        self.AGKD = AGKD
        self.CGKD = CGKD

    def run_G(self, z, c, sync):
        with misc.ddp_sync(self.G_mapping, sync):
            ws = self.G_mapping(z, c)
            if self.style_mixing_prob > 0:
                with torch.autograd.profiler.record_function('style_mixing'):
                    cutoff = torch.empty([], dtype=torch.int64, device=ws.device).random_(1, ws.shape[1])
                    cutoff = torch.where(torch.rand([], device=ws.device) < self.style_mixing_prob, cutoff, torch.full_like(cutoff, ws.shape[1]))
                    ws[:, cutoff:] = self.G_mapping(torch.randn_like(z), c, skip_w_avg_update=True)[:, cutoff:]
        with misc.ddp_sync(self.G_synthesis, sync):
            img = self.G_synthesis(ws) 
            # img -> torch.Size([8, 3, 256, 256])
            # ws -> torch.Size([8, 14, 512])
            # ws.shape -> [batch_size, num_ws, w_dim]
            # num_ws is the number of intermediate latent codes.
            # w_dim is the dimension of the intermediate latent codes.
        return img, ws

    def run_D(self, img, c, sync, return_features=False):
        if self.diffaugment:
            img = DiffAugment(img, policy=self.diffaugment)
        if self.augment_pipe is not None:
            img = self.augment_pipe(img)
        with misc.ddp_sync(self.D, sync):
            logits, features = self.D(img, c)
        if return_features:
            return logits, features    
        return logits

    def accumulate_gradients(self, phase, real_img, real_c, gen_z, gen_c, sync, gain):
        assert phase in ['Gmain', 'Greg', 'Gboth', 'Dmain', 'Dreg', 'Dboth']
        do_Gmain = (phase in ['Gmain', 'Gboth'])
        do_Dmain = (phase in ['Dmain', 'Dboth'])
        do_Gpl   = (phase in ['Greg', 'Gboth']) and (self.pl_weight != 0)
        do_Dr1   = (phase in ['Dreg', 'Dboth']) and (self.r1_gamma != 0)
        gen_img = None
        gen_features = None
        # Gmain: Maximize logits for generated images.
        if do_Gmain:
            with torch.autograd.profiler.record_function('Gmain_forward'):
                gen_img, _gen_ws = self.run_G(gen_z, gen_c, sync=(sync and not do_Gpl)) # May get synced by Gpl.
                gen_logits = self.run_D(gen_img, gen_c, sync=False, return_features=False)
                training_stats.report('Loss/scores/fake', gen_logits)
                training_stats.report('Loss/signs/fake', gen_logits.sign())
                loss_Gmain = torch.nn.functional.softplus(-gen_logits) # -log(sigmoid(gen_logits))
                training_stats.report('Loss/G/loss', loss_Gmain)
            with torch.autograd.profiler.record_function('Gmain_backward'):
                loss_Gmain.mean().mul(gain).backward()

        # Gpl: Apply path length regularization.
        if do_Gpl:
            with torch.autograd.profiler.record_function('Gpl_forward'):
                batch_size = gen_z.shape[0] // self.pl_batch_shrink
                gen_img, gen_ws = self.run_G(gen_z[:batch_size], gen_c[:batch_size], sync=sync)
                pl_noise = torch.randn_like(gen_img) / np.sqrt(gen_img.shape[2] * gen_img.shape[3])
                with torch.autograd.profiler.record_function('pl_grads'), conv2d_gradfix.no_weight_gradients():
                    pl_grads = torch.autograd.grad(outputs=[(gen_img * pl_noise).sum()], inputs=[gen_ws], create_graph=True, only_inputs=True)[0]
                pl_lengths = pl_grads.square().sum(2).mean(1).sqrt()
                pl_mean = self.pl_mean.lerp(pl_lengths.mean(), self.pl_decay)
                self.pl_mean.copy_(pl_mean.detach())
                pl_penalty = (pl_lengths - pl_mean).square()
                training_stats.report('Loss/pl_penalty', pl_penalty)
                loss_Gpl = pl_penalty * self.pl_weight
                training_stats.report('Loss/G/reg', loss_Gpl)
            with torch.autograd.profiler.record_function('Gpl_backward'):
                (gen_img[:, 0, 0, 0] * 0 + loss_Gpl).mean().mul(gain).backward()

        # Dmain: Minimize logits for generated images.
        loss_Dgen = 0
        if do_Dmain:
            with torch.autograd.profiler.record_function('Dgen_forward'):
                gen_img, _gen_ws = self.run_G(gen_z, gen_c, sync=False)
                gen_logits, gen_features = self.run_D(gen_img, gen_c, sync=False, return_features=True) # Gets synced by loss_Dreal.
                training_stats.report('Loss/scores/fake', gen_logits)
                training_stats.report('Loss/signs/fake', gen_logits.sign())
                
                loss_Dgen = torch.nn.functional.softplus(gen_logits) # -log(1 - sigmoid(gen_logits))

        # Dmain: Maximize logits for real images.
        # Dr1: Apply R1 regularization.
        if do_Dmain or do_Dr1:
            name = 'Dreal_Dr1' if do_Dmain and do_Dr1 else 'Dreal' if do_Dmain else 'Dr1'
            with torch.autograd.profiler.record_function(name + '_forward'):
                real_img_tmp = real_img.detach().requires_grad_(do_Dr1)
                real_logits, real_features = self.run_D(real_img_tmp, real_c, sync=sync, return_features=True)
                training_stats.report('Loss/scores/real', real_logits)
                training_stats.report('Loss/signs/real', real_logits.sign())

                loss_Dreal = 0
                loss_kd = 0
                if do_Dmain:
                    loss_Dreal = torch.nn.functional.softplus(-real_logits) # -log(sigmoid(real_logits))
                    training_stats.report('Loss/D/loss', loss_Dgen + loss_Dreal)
                    with torch.autograd.profiler.record_function('KD_Discriminator_forward'):
                        batch_size = gen_z.shape[0] // self.pl_batch_shrink
                        KD_d_loss = KDDiscriminatorLoss(AGKD = self.AGKD, CGKD = self.CGKD, nz = 128, batch_size = batch_size)
                        loss_kd = KD_d_loss.forward(real_features = real_features, gen_features = gen_features, real_data = real_img, gen_data = gen_img)
                        training_stats.report('Loss/KD/loss', loss_kd)

                loss_Dr1 = 0
                if do_Dr1:
                    with torch.autograd.profiler.record_function('r1_grads'), conv2d_gradfix.no_weight_gradients():
                        r1_grads = torch.autograd.grad(outputs=[real_logits.sum()], inputs=[real_img_tmp], create_graph=True, only_inputs=True)[0]
                    r1_penalty = r1_grads.square().sum([1,2,3])
                    loss_Dr1 = r1_penalty * (self.r1_gamma / 2)
                    training_stats.report('Loss/r1_penalty', r1_penalty)
                    training_stats.report('Loss/D/reg', loss_Dr1)

            with torch.autograd.profiler.record_function(name + '_backward'):
                (loss_kd + loss_Dgen + loss_Dreal + loss_Dr1).mean().mul(gain).backward()
                

#----------------------------------------------------------------------------

# custom Loss function for the discriminator
class KDDiscriminatorLoss(nn.Module):
    # initialize the loss module
    def __init__(self,
                 AGKD : bool = False,
                 CGKD : bool = False,
                 nz : int = 128,
                 p : float = 0.7,
                 batch_size : int = 128
                 ):
        super(KDDiscriminatorLoss, self).__init__()

        # set the device
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        # size of z latent vector
        self.nz = nz

        # batch size
        self.batch_size = batch_size


        # AGKD and CGKD enables
        self.AGKD = AGKD
        self.CGKD = CGKD

        # init clip feature extractor
        self.clip = CLIP.CLIP()

        # predetermine text labels for knowledge distillation
        text_label_retriever = TextLabels()
        self.text_labels = text_label_retriever.get_labels()

        self.text_label_step = 0

        self.K = self.batch_size

        if self.CGKD:
            # define cosine similarity for further usage
            self.cos_sim = torch.nn.CosineSimilarity(dim=0, eps=1e-6)

        if self.AGKD:
            # set probabilty of AGKD loss
            self.p = p

    # define loss funciton
    def forward(self, real_features, real_data, gen_features, gen_data) -> torch.Tensor:
        L_D = torch.zeros(8, 1).to(self.device)
        # add aggregated distilation loss
        if self.AGKD:
            agkd_loss = self.aggregatedDistillation(real_data, real_features, gen_data, gen_features)
            L_D += agkd_loss

        # add correlated distilation loss
        if self.CGKD:
            cgkd_loss = self.correlatedDistillation(gen_data, gen_features)
            L_D += cgkd_loss

        # return total loss
        return L_D.to(self.device)

    
    def aggregatedDistillation(self, real_data, real_embeddings, fake_data, fake_embeddings):
        # get real and fake image features from clip
        real_features = self.clip.image_features(real_data)
        fake_features = self.clip.image_features(fake_data)

        # sample random number between 0, 1
        q = torch.rand(1).item()

        # compose L_AGG
        if q <= self.p:
            L_AGG = torch.abs(real_features - fake_embeddings)
            L_AGG += torch.abs(fake_features - real_embeddings)
        else:
            L_AGG = 0

        # compose L_KD
        L_KD = torch.abs(real_features - real_embeddings)
        L_KD += torch.abs(fake_features - fake_embeddings)

        # compose L_AGKD
        L_AGKD = L_AGG + L_KD
        
        #agkd_loss: torch.Size([8, 512])
        return torch.mean(L_AGKD, dim=1, keepdim=True).to(self.device) #torch.Size([8, 1])
    
    
    def correlatedDistillation(self, fake_data, fake_embeddings):
        # get image and text features from clip
        image_features = self.clip.image_features(fake_data)
        text_features = self.clip.text_features(self.text_labels[self.text_label_step:self.text_label_step + self.batch_size])

        self.text_label_step += self.batch_size
        self.text_label_step = self.text_label_step % len(self.text_labels)

        # compose C_T
        C_T = (image_features @ text_features.T) / torch.norm(image_features @ text_features.T, p=2)

        # compose pairwise diversity loss
        L_PD = 0

        for i in range(self.K):
            for j in range(self.K):
                if i == j:
                    continue

                L_PD += self.cos_sim(C_T[i, :], C_T[j, :])

        # compose C_S
        C_S = (fake_embeddings @ text_features.T) / (torch.norm(fake_embeddings @ text_features.T, p=2))

        L_KD = torch.abs(C_T - C_S)

        # compose overall correlated distillation loss
        L_CGKD = L_PD + L_KD
        """
        L_CGKD: torch.Size([8, 8])
        L_PD: torch.Size([])
        L_KD: torch.Size([8, 8])
        C_T: torch.Size([8, 8])
        C_S: torch.Size([8, 8])
        fake_embeddings: torch.Size([8, 512])
        text_features: torch.Size([8, 512]) 
        """
        
        return torch.mean(L_CGKD, dim=1, keepdim=True).to(self.device) # torch.Size([8, 1])


