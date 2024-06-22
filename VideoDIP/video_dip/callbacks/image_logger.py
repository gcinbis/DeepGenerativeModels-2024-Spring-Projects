import pytorch_lightning as pl
import torch
from pytorch_lightning.loggers import TensorBoardLogger, WandbLogger
import torchvision

import numpy as np

class ImageLogger(pl.Callback):
    def __init__(self, num_images=4):
        super().__init__()
        self.num_images = num_images

    def on_validation_batch_end(self, trainer: pl.Trainer, pl_module: pl.LightningModule, outputs: torch.Tensor | torch.Dict[str, torch.Any] | None, batch: torch.Any, batch_idx: int, dataloader_idx: int = 0) -> None:
        num_total_batches = len(trainer.val_dataloaders)
        log_points = np.linspace(0, num_total_batches, self.num_images + 1, endpoint=False).astype(int)[1:] if self.num_images > 1 else [num_total_batches - 2]
        if batch_idx in log_points:
            flow_image = torchvision.utils.flow_to_image(batch['flow']) / 255.0
            if isinstance(trainer.logger, TensorBoardLogger):
                self.log_images_tensorboard(
                    logger=trainer.logger, 
                    inputs=outputs['input'], 
                    labels=batch['target'], 
                    preds_rgb=outputs['rgb_output'], 
                    preds_rgb2=outputs["rgb_output2"] if "rgb_output2" in outputs else None,
                    preds_alpha=outputs['alpha_output'],
                    preds_reconstructed=outputs['reconstructed'],
                    flow=flow_image,
                    stage='val', 
                    global_step=trainer.global_step
                )
            elif isinstance(trainer.logger, WandbLogger):
                self.log_images_wandb(
                    logger=trainer.logger, 
                    inputs=outputs['input'], 
                    labels=batch['target'] if "target" in batch else None, 
                    preds_rgb=outputs['rgb_output'],
                    preds_rgb2=outputs["rgb_output2"] if "rgb_output2" in outputs else None,
                    preds_alpha=outputs['alpha_output'],
                    preds_reconstructed=outputs['reconstructed'],
                    segmentation=outputs["segmentation"] if "segmentation" in outputs else None,
                    segmentation_gt = outputs["segmentation_gt"] if "segmentation_gt" in outputs else None,
                    flow=flow_image,
                    stage='val', 
                    global_step=trainer.global_step
                )

    def log_images_tensorboard(self, logger, inputs, labels, preds_rgb, preds_rgb2, preds_alpha, preds_reconstructed, stage, global_step, segmentation = None, segmentation_gt = None, flow = None) :
        import torchvision.utils as vutils

        # Create a grid of input images
        grid_inputs = vutils.make_grid(inputs)
        logger.experiment.add_image(f'{stage}/inputs', grid_inputs, global_step)

        # Create a grid of label images (assuming labels are single-channel)
        grid_labels = vutils.make_grid(labels)
        logger.experiment.add_image(f'{stage}/labels', grid_labels, global_step)

        # Create a grid of prediction images (assuming preds are single-channel)
        grid_preds = vutils.make_grid(preds_rgb)
        logger.experiment.add_image(f'{stage}/predictions_rgb', grid_preds, global_step)

        # Create a grid of alpha images (assuming preds are single-channel)
        grid_alpha = vutils.make_grid(preds_alpha)
        logger.experiment.add_image(f'{stage}/predictions_alpha', grid_alpha, global_step)

        grid_reconstructed = vutils.make_grid(preds_reconstructed)
        logger.experiment.add_image(f'{stage}/reconstructed', grid_reconstructed, global_step)

        if preds_rgb2 is not None:
            grid_preds2 = vutils.make_grid(preds_rgb2)
            logger.experiment.add_image(f'{stage}/predictions_rgb2', grid_preds2, global_step)

        if segmentation is not None:
            # Create a grid of prediction images (assuming preds are single-channel)
            grid_preds = vutils.make_grid(segmentation)
            logger.experiment.add_image(f'{stage}/segmentation', grid_preds, global_step)

        if segmentation_gt is not None:
            # Create a grid of prediction images (assuming preds are single-channel)
            grid_preds = vutils.make_grid(segmentation_gt)
            logger.experiment.add_image(f'{stage}/segmentation_gt', grid_preds, global_step)

        grid_flow = vutils.make_grid(flow)
        logger.experiment.add_image(f'{stage}/flow', grid_flow, global_step)        

    def log_images_wandb(self, logger, inputs, labels, preds_rgb, preds_rgb2, preds_alpha, preds_reconstructed, stage, global_step, segmentation = None, segmentation_gt = None, flow = None) :
        import wandb
        import torchvision.utils as vutils

        logging_dict = {}
        # Create a grid of input images
        grid_inputs = vutils.make_grid(inputs)
        grid_inputs = grid_inputs.permute(1, 2, 0).cpu().float().numpy()  # Convert to HWC format
        wandb_inputs = wandb.Image(grid_inputs, caption=f'{stage}/inputs')
        logging_dict[f'{stage}/inputs'] = wandb_inputs
        
        if preds_rgb2 is not None:
            grid_preds = vutils.make_grid(preds_rgb2)
            grid_preds = grid_preds.permute(1, 2, 0).cpu().float().numpy()  # Convert to HWC format
            wandb_preds2 = wandb.Image(grid_preds, caption=f'{stage}/preds_rgb2')
            logging_dict[f'{stage}/preds_rgb2'] = wandb_preds2

        if segmentation is not None:
            # Create a grid of prediction images (assuming preds are single-channel)
            grid_preds = vutils.make_grid(segmentation)
            grid_preds = grid_preds.permute(1, 2, 0).cpu().float().numpy()  # Convert to HWC format
            wandb_segmentation = wandb.Image(grid_preds, caption=f'{stage}/segmentation')
            logging_dict[f'{stage}/segmentation'] = wandb_segmentation

        if segmentation_gt is not None:
            # Create a grid of prediction images (assuming preds are single-channel)
            grid_preds = vutils.make_grid(segmentation_gt)
            grid_preds = grid_preds.permute(1, 2, 0).cpu().float().numpy()  # Convert to HWC format
            wandb_segmentation_gt = wandb.Image(grid_preds, caption=f'{stage}/segmentation_gt')
            logging_dict[f'{stage}/segmentation_gt'] = wandb_segmentation_gt
        
        grid_labels = vutils.make_grid(labels)
        grid_labels = grid_labels.permute(1, 2, 0).cpu().float().numpy()  # Convert to HWC format
        wandb_labels = wandb.Image(grid_labels, caption=f'{stage}/labels')
        logging_dict[f'{stage}/labels'] = wandb_labels

        # Create a grid of prediction images (assuming preds are single-channel)
        grid_preds = vutils.make_grid(preds_rgb)
        grid_preds = grid_preds.permute(1, 2, 0).cpu().float().numpy()  # Convert to HWC format
        wandb_preds = wandb.Image(grid_preds, caption=f'{stage}/predictions')
        logging_dict[f'{stage}/predictions'] = wandb_preds

        # Create a grid of alpha images (assuming preds are single-channel)
        grid_alpha = vutils.make_grid(preds_alpha)
        grid_alpha = grid_alpha.permute(1, 2, 0).cpu().float().numpy()
        wandb_alpha = wandb.Image(grid_alpha, caption=f'{stage}/alpha')
        logging_dict[f'{stage}/alpha'] = wandb_alpha

        grid_reconstructed = vutils.make_grid(preds_reconstructed)
        grid_reconstructed = grid_reconstructed.permute(1, 2, 0).cpu().float().numpy()
        wandb_reconstructed = wandb.Image(grid_reconstructed, caption=f'{stage}/reconstructed')
        logging_dict[f'{stage}/reconstructed'] = wandb_reconstructed

        grid_flow = vutils.make_grid(flow)
        grid_flow = grid_flow.permute(1, 2, 0).cpu().float().numpy()
        wandb_flow = wandb.Image(grid_flow, caption=f'{stage}/flow')
        logging_dict[f'{stage}/flow'] = wandb_flow
        logging_dict['global_step'] = global_step

        logger.experiment.log(logging_dict)