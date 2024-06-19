import pytorch_lightning as pl
from pytorch_lightning.utilities import rank_zero_only
import torch
from torch.utils.data import DataLoader
import torchvision
from PIL import Image, ImageOps
import numpy as np

import pytorch_lightning.loggers as pl_loggers

def normalize_image(image):
    # Normalize the image to have values between 0 and 1
    return (image - image.min()) / (image.max() - image.min())

def get_labeled_image(image, border_color):
    # Convert the image to a labeled image with a border color
    image = image.detach().cpu().numpy()
    image = (image + 1) / 2  # Range: [-1,1] -> [0,1]
    image = (image * 255).astype(np.uint8)
    if image.shape[0] == 1:
        image = np.stack((image[0],) * 3, axis=0)
    image = image.transpose(1, 2, 0)
    image = Image.fromarray(image, "RGB")
    border_size = int(min(image.size) * 0.05)
    image = ImageOps.expand(image, border=border_size, fill=border_color)
    image = torchvision.transforms.ToTensor()(image).unsqueeze_(0)
    return image

def get_labeled_images(image_list, border_color):
    # Convert a list of images to labeled images with a border color
    labeled_image_list = []
    for image in image_list:
        labeled_image_list.append(get_labeled_image(image, border_color))
    return labeled_image_list

class ImageLoggerCallback(pl.callbacks.Callback):

    def __init__(self, dataset, dataset_name, num_samples=20, eval_every=10000):
        # Initialize the ImageLoggerCallback
        self.num_samples = num_samples
        self.eval_every = eval_every
        dataloader = DataLoader(dataset, batch_size=num_samples, shuffle=False)
        self.input_batch = next(iter(dataloader))
        self.dataset_name = dataset_name

    @rank_zero_only
    def on_train_start(self, trainer, pl_module):
        # Get the TensorBoardLogger and WandbLogger instances
        self.tb_logger = None
        self.wandb_logger = None
        for logger in trainer.loggers:
            if isinstance(logger, pl_loggers.TensorBoardLogger):
                self.tb_logger = logger.experiment
            if isinstance(logger, pl_loggers.WandbLogger):
                self.wandb_logger = logger
        if self.tb_logger is None:
            raise ValueError('TensorBoard Logger is not found')

    def pad_image(self, image):
        # Pad the image with zeros to add 96 pixels on all sides
        # The image is of shape 3x64x64
        # Return the padded image of shape 3x256x256
        return torch.nn.functional.pad(image, (96, 96, 96, 96), mode='constant', value=0)

    @rank_zero_only
    def on_train_batch_start(self, trainer, pl_module, *args, **kwargs):
        current_step = trainer.global_step // 2  # x2 step on each iter due to GAN loss

        if current_step % self.eval_every != 0:
            return

        results = pl_module.make_high_resolution(self.input_batch)

        grid_list = []
        image_lr = results["image_lr"]
        padded_image_lr = self.pad_image(image_lr)
        grid_list.append(torch.cat(get_labeled_images(padded_image_lr, 'darkgreen'), dim=2)[0])
        grid_list.append(torch.cat(get_labeled_images(results["image_hr"], 'black'), dim=2)[0])
        grid_list.append(torch.cat(get_labeled_images(results["generated_super_resolution_image"], 'darkred'), dim=2)[0])

        grid = torchvision.utils.make_grid(grid_list, padding=2, normalize=False, nrow=len(grid_list))
        self.tb_logger.add_image(f"{self.dataset_name}/high_resolution", grid, current_step)
        if self.wandb_logger is not None:
            self.wandb_logger.log_image(key=f"{self.dataset_name}/high_resolution", images=[grid],
                                        caption=[f"Step: {current_step}"], step=current_step)
