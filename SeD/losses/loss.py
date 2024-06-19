from .vgg_model import PerceptualModel
import torch

import torch.nn.functional as F

class MSE:
    def __init__(self, weight, **others):
        self.weight = weight

    def __call__(self, image, fake_image_final, **others):
        # Calculate mean squared error loss
        fakes = fake_image_final  
        targets = image  
        mse_loss = F.mse_loss(fakes, targets, reduction='mean')
        return mse_loss
    
    
class VGG: 
    def __init__(self, weight, model_config, device, **others):
        self.weight = weight
        self.device = device
        self.vgg = PerceptualModel(**model_config).to(device)

    def __call__(self, image, fake_image_final, **others):
        # Calculate VGG loss
        fakes = fake_image_final 
        targets = image 
        vgg_loss = F.mse_loss(self.vgg(fakes), self.vgg(targets), reduction='mean')
        return vgg_loss
        
        
class Adversarial_G: 
    def __init__(self, weight, discriminator, **others):
        self.weight = weight
        self.discriminator = discriminator

    def __call__(self, fake_image_final, semantic_feature_maps=None, **others):
        # Calculate adversarial generator loss
        fakes = fake_image_final 
        if semantic_feature_maps is not None:
            fake_scores = self.discriminator(semantic_feature_maps=semantic_feature_maps, fs=fakes)
        else:
            fake_scores = self.discriminator(fs=fakes)
        
        fake_target = torch.ones_like(fake_scores)
        g_loss = F.binary_cross_entropy_with_logits(fake_scores, fake_target)
        return g_loss
    
class Adversarial_D: 
    def __init__(self, discriminator, r1_gamma, r2_gamma, **others):
        self.discriminator = discriminator
        self.r1_gamma = r1_gamma
        self.r2_gamma = r2_gamma

    @staticmethod
    def compute_grad_penalty(images, scores):
        """Computes gradient penalty."""
        image_grad = torch.autograd.grad(
            outputs=scores.sum(),
            inputs=images,
            create_graph=True,
            retain_graph=True)[0].view(images.shape[0], -1)
        penalty = image_grad.pow(2).sum(dim=1).mean()
        return penalty

    def __call__(self, image, fake_image_final, semantic_feature_maps=None, **others):
        # Calculate adversarial discriminator loss
        reals = image.clone()
        fake_image_final_clone = fake_image_final.detach().clone()
        fakes =  fake_image_final_clone 

        reals.requires_grad = True # To calculate gradient penalty
        if semantic_feature_maps is not None:
            real_scores = self.discriminator(semantic_feature_maps=semantic_feature_maps, fs=reals)
            fake_scores = self.discriminator(semantic_feature_maps=semantic_feature_maps, fs=fakes)
        else:
            real_scores = self.discriminator(fs=reals)
            fake_scores = self.discriminator(fs=fakes)
            
        real_target = torch.ones_like(real_scores)
        fake_target = torch.zeros_like(fake_scores)
        loss_real = F.binary_cross_entropy_with_logits(real_scores, real_target)
        loss_fake = F.binary_cross_entropy_with_logits(fake_scores, fake_target)
        d_loss = loss_fake + loss_real
        real_grad_penalty = torch.zeros_like(d_loss)
        fake_grad_penalty = torch.zeros_like(d_loss)
        if self.r1_gamma != 0:
            real_grad_penalty = self.compute_grad_penalty(reals, real_scores)
        if self.r2_gamma != 0:
            fake_grad_penalty = self.compute_grad_penalty(fakes, fake_scores)
        reals.requires_grad = False
        return d_loss + real_grad_penalty * (self.r1_gamma * 0.5) + fake_grad_penalty * (self.r2_gamma * 0.5)
