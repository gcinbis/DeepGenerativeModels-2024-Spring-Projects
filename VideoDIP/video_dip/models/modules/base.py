import pytorch_lightning as pl
import torch
import torchvision
from torch.optim.lr_scheduler import CosineAnnealingLR

from video_dip.models.unet import UNet
from video_dip.losses import ReconstructionLoss, OpticalFlowWarpLoss

class VDPModule(pl.LightningModule):
    """
    VDPModule is a PyTorch Lightning module for video deep image prior (VDIP) models.

    Args:
        learning_rate (float): The learning rate for the optimizer. Default is 1e-3.
        loss_weights (list): The weights for the reconstruction and warp losses. Default is [1, .02].

    Attributes:
        rgb_net (UNet): The UNet model for RGB inputs.
        alpha_net (UNet): The UNet model for optical flow inputs.
        learning_rate (float): The learning rate for the optimizer.
        reconstruction_loss (ReconstructionLoss): The reconstruction loss function.
        warp_loss (OpticalFlowWarpLoss): The optical flow warp loss function.
        loss_weights (list): The weights for the reconstruction and warp losses.

    Methods:
        forward(img=None, flow=None): Performs forward pass through the network.
        reconstruction_fn(rgb_output, alpha_output): Computes the reconstructed frame.
        inference(batch, batch_idx): Performs inference on a batch of data.
        training_step(batch, batch_idx): Defines the training step.
        validation_step(batch, batch_idx): Defines the validation step.
        configure_optimizers(): Configures the optimizer.

    """

    def __init__(self, learning_rate=1e-3, loss_weights=[1, .02], multi_step_scheduling_kwargs=None, warmup=False):
        super().__init__()
        self.rgb_net = UNet(out_channels=3)  # RGB-Net with 3 input and 3 output channels
        self.alpha_net = UNet(out_channels=1)  # Alpha-Net with 3 input and 1 output channels (for optical flow
        self.learning_rate = learning_rate

        self.reconstruction_loss = ReconstructionLoss()
        self.warp_loss = OpticalFlowWarpLoss()

        self.loss_weights = loss_weights

        self.warmup = warmup

        self.multi_step_scheduling_kwargs = multi_step_scheduling_kwargs

    def forward(self, img=None, flow=None):
        """
        Performs forward pass through the network.

        Args:
            img (torch.Tensor): The input image tensor. Default is None.
            flow (torch.Tensor): The input optical flow tensor. Default is None.

        Returns:
            dict: A dictionary containing the output tensors.

        """
        ret = {}
        if img is not None:
            ret['rgb'] = self.rgb_net(img)
        if flow is not None:
            ret['alpha'] = self.alpha_net(flow)
        return ret
        
    def reconstruction_fn(self, rgb_output, alpha_output, **kwargs):
        """
        Computes the reconstructed frame.

        Args:
            rgb_output (torch.Tensor): The RGB output tensor.
            alpha_output (torch.Tensor): The alpha output tensor.

        Raises:
            NotImplementedError: If the reconstruction function is not implemented.

        """
        raise NotImplementedError("The reconstruction function is not implemented.")

    def inference(self, batch, batch_idx, **kwargs):
        """
        Performs inference on a batch of data.

        Args:
            batch (dict): A dictionary containing the input and flow tensors.
            batch_idx (int): The index of the current batch.

        Returns:
            dict: A dictionary containing the input, flow, reconstructed, rgb_output, and alpha_output tensors.

        """
        input_frames = batch['input']
        flows = batch['flow']

        flow_frames = torchvision.utils.flow_to_image(flows) / 255.0

        output = self(img=input_frames, flow=flow_frames)
        rgb_output = output['rgb']
        alpha_output = output['alpha']

        reconstructed_frame = self.reconstruction_fn(rgb_output, alpha_output, **kwargs)

        return {
            "input": input_frames,
            "flow": flows,
            "reconstructed": reconstructed_frame,
            "rgb_output": rgb_output,
            "alpha_output": alpha_output
        }

    def training_step(self, batch, batch_idx):
        """
        Defines the training step.

        Args:
            batch (dict): A dictionary containing the input and flow tensors.
            batch_idx (int): The index of the current batch.

        Returns:
            torch.Tensor: The computed loss value.

        """
        outputs = self.inference(batch, batch_idx)
        # prev_alpha_output = self(flow=torchvision.utils.flow_to_image(batch['prev_input']) / 255.0)['alpha'].detach()
        prev_output = self(img=batch['prev_input'])['rgb'].detach()

        rec_loss = self.reconstruction_loss(outputs['input'], outputs['reconstructed'])
        warp_loss = self.warp_loss(outputs['flow'], prev_output, outputs['alpha_output'])

        loss = self.loss_weights[0] * rec_loss + self.loss_weights[1] * warp_loss

        self.log("train_loss", loss, prog_bar=True, on_step=False, on_epoch=True)
        self.log("rec_loss", rec_loss, prog_bar=False, on_step=False, on_epoch=True)
        self.log("warp_loss", warp_loss, prog_bar=False, on_step=False, on_epoch=True)

        return loss
    
    def validation_step(self, batch, batch_idx):
        """
        Defines the validation step.

        Args:
            batch (dict): A dictionary containing the input and flow tensors.
            batch_idx (int): The index of the current batch.

        """
        outputs = self.inference(batch, batch_idx)

        return outputs
        
    def configure_optimizers(self):
        """
        Configures the optimizer and scheduler.

        Returns:
            dict: A dictionary containing the optimizer and the LR scheduler.

        """
        optimizer = torch.optim.Adam(self.parameters(), lr=self.learning_rate)
        # MultiStep Scheduler
        ret = {'optimizer': optimizer}
        
        schedulers = []
        if self.warmup:
            # Warmup for 5 epochs from 2e-5 to self.learning_rate
            def lr_lambda(epoch):
                if epoch < 5:
                    warmup_lr = 2e-5 + epoch * (self.learning_rate - 2e-5) / 5
                    return warmup_lr / self.learning_rate
                return 1.0
            
            schedulers.append(torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_lambda))
            ret['lr_scheduler'] = {
                'scheduler': schedulers[0],
                'interval': 'epoch',
                'frequency': 1,
            }

        if self.multi_step_scheduling_kwargs is not None:
            schedulers.append(torch.optim.lr_scheduler.MultiStepLR(optimizer, **self.multi_step_scheduling_kwargs))
            ret['lr_scheduler'] = {
                'scheduler': schedulers[0],
                'interval': 'epoch',
                'frequency': 1,
            }

        if len(schedulers) > 1:
            combined_scheduler = torch.optim.lr_scheduler.SequentialLR(optimizer, schedulers, milestones=[5])
            ret['lr_scheduler'] = {
                'scheduler': combined_scheduler,
                'interval': 'epoch',
                'frequency': 1,
            }
        
        return ret
            
    def test_step(self, batch, batch_idx):        
        return self.validation_step(batch, batch_idx)