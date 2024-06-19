import pytorch_lightning as pl
import torch
from models.rrdb import RRDBNet
from models.patchgan_discriminator import PatchDiscriminatorWithSeD, PatchDiscriminator
from models.pixelwise_discriminator import UNetPixelDiscriminator, UNetPixelDiscriminatorwithSed
import importlib
from torch.optim.lr_scheduler import MultiStepLR
from feature_extractor_model import CLIPRN50

class SuperResolutionModule(pl.LightningModule):
    def __init__(self, 
                 loss_dict,
                 generator_learning_rate=1e-4,
                 discriminator_learning_rate=1e-4,
                 discriminator_decay_steps=[],
                 discriminator_decay_gamma=0.5,
                 generator_decay_gamma=0.5,
                 generator_decay_steps=[],
                 clip_generator_outputs=False,
                 use_sed_discriminator=False,
                 is_pixelwise_disc=False):
        super().__init__()

        torch.set_float32_matmul_precision('high')

        self.save_hyperparameters()
        self.automatic_optimization = False # To handle the GAN loss, optimization steps are implemented manually
        self.eval_metric_dict = {} # Updated by eval callback funcitons

    
        self.generator = RRDBNet(3, 64, 23, clip_output=clip_generator_outputs)
        self.generator.train()

        if is_pixelwise_disc:
            if use_sed_discriminator:
                self.discriminator = UNetPixelDiscriminatorwithSed()
            else:
                self.discriminator = UNetPixelDiscriminator()
        else:
            if use_sed_discriminator:
                self.discriminator = PatchDiscriminatorWithSeD(3, 64)
            else:
                self.discriminator = PatchDiscriminator(3, 64)
        self.discriminator.train()

        self.use_sed_discriminator = use_sed_discriminator
        self.loss_dict = loss_dict
        self.discriminator_learning_rate = discriminator_learning_rate
        self.discriminator_decay_steps = discriminator_decay_steps
        self.discriminator_decay_gamma = discriminator_decay_gamma
        self.clip_generator_outputs = clip_generator_outputs
        self.generator_learning_rate = generator_learning_rate
        self.generator_decay_steps = generator_decay_steps
        self.generator_decay_gamma = generator_decay_gamma
        self.clip = CLIPRN50([3, 4, 6, 3], 1024, 32)
        self.clip.freeze()


    def on_train_start(self):
        self.generator.device = self.device
        self.discriminator.device = self.device

        losses = {}
        for loss_class, loss_config in self.loss_dict.items():
            module = importlib.import_module("losses.loss")
            class_ = getattr(module, loss_class)
            loss_config["device"] = self.device
            loss_config["discriminator"] = self.discriminator
            loss = class_(**loss_config)
            losses[loss_class] = loss
        self.losses = losses
        
    def configure_optimizers(self):
        generator_optimizer_parameters = []
        generator_optimizer_parameters += list(self.generator.parameters())

        optimizer_generator = torch.optim.Adam(generator_optimizer_parameters, lr=self.generator_learning_rate, betas=(0.9,0.99))
        optimizer_discriminator = torch.optim.Adam(list(self.discriminator.parameters()), lr=self.discriminator_learning_rate, betas=(0.9,0.99))
 
        scheduler_generator = {
            'scheduler': MultiStepLR(optimizer_generator, milestones=self.generator_decay_steps, gamma=self.generator_decay_gamma),
            'name': 'LR_Generator' # For logging LR via the callback properly
            }
        scheduler_discriminator = {
            'scheduler': MultiStepLR(optimizer_discriminator, milestones=self.discriminator_decay_steps, gamma=self.discriminator_decay_gamma),
            'name': 'LR_Discriminator' # For logging LR via the callback properly
            }

        return [optimizer_generator, optimizer_discriminator], [scheduler_generator, scheduler_discriminator]
    

    #define the inference method

    @torch.no_grad()
    def make_high_resolution(self, input_batch):
        self.eval()
        self.generator.eval()
        image_lr = input_batch['image_lr'].to(self.device)
        generated_super_resolution_image = self.generator(image_lr).to(self.device)

        #return the generated super resolution image and the low resolution image, and original high resolution image
        results = {}
        results['image_lr'] = image_lr.to('cpu')
        results['image_hr'] = input_batch['image_hr'].to('cpu')
        results['generated_super_resolution_image'] = generated_super_resolution_image.to('cpu')

        return results

    def get_loss_forward_dict(self, batch):
        image_lr = batch['image_lr'].to(self.device)
        #call the generator
        generated_super_resolution_image = self.generator(image_lr).to(self.device)

        image_hr = batch['image_hr'].to(self.device)
        
        if self.use_sed_discriminator:
            semantic_feature_maps = self.clip(image_hr).to(self.device)

        loss_forward_dict = {
            "image": image_hr,
            "fake_image_final": generated_super_resolution_image,
            "semantic_feature_maps": semantic_feature_maps
            } if self.use_sed_discriminator else {
            "image": image_hr,
            "fake_image_final": generated_super_resolution_image
            }
        return loss_forward_dict
        
    def training_step(self, batch, batch_idx):

        loss_value_dict = {}
        for loss_name, loss in self.losses.items():
            loss_value_dict[loss_name] = 0.0

        optimizer_generator, optimizer_discriminator = self.optimizers()
        scheduler_generator, scheduler_discriminator = self.lr_schedulers()

        batch_size = batch['image_hr'].shape[0]

        self.train()
        self.generator.train()
        self.discriminator.train()

        loss_forward_dict = self.get_loss_forward_dict(batch)

        # Train Discriminator
        if "Adversarial_D" in self.losses:
            self.toggle_optimizer(optimizer_discriminator)
            adversarial_d_loss = self.losses["Adversarial_D"]
            d_loss = adversarial_d_loss(**loss_forward_dict)
            loss_value_dict["Adversarial_D"] = d_loss.item()
            optimizer_discriminator.zero_grad()
            self.manual_backward(d_loss)
            optimizer_discriminator.step()
            scheduler_discriminator.step()
            self.untoggle_optimizer(optimizer_discriminator)

        self.toggle_optimizer(optimizer_generator)
        loss_total = 0.0
        for loss_name, loss in self.losses.items():
            if loss_name == "Adversarial_D":
                continue
            loss_value = loss.weight * loss(**loss_forward_dict)
            loss_total += loss_value
            loss_value_dict[loss_name] = loss_value.item()
        optimizer_generator.zero_grad()
        self.manual_backward(loss_total)
        optimizer_generator.step()
        scheduler_generator.step()
        self.untoggle_optimizer(optimizer_generator)

        self.log_dict(self.eval_metric_dict, prog_bar=True, logger=True, on_step=True, on_epoch=False, batch_size=batch_size, sync_dist=True)
        self.log_dict(loss_value_dict, prog_bar=True, logger=True, on_step=True, on_epoch=True, batch_size=batch_size, sync_dist=True)


    
    