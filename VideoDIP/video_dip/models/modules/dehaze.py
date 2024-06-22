import torch
from . import VDPModule
from torchmetrics import PeakSignalNoiseRatio, StructuralSimilarityIndexMeasure

class DehazeVDPModule(VDPModule):
    """
    A module for dehazing in VideoDIP.

    Args:
        learning_rate (float): The learning rate for optimization (default: 1e-3).
        loss_weights (list): The weights for different losses (default: [1, .02]).
    """

    def __init__(self, learning_rate=1e-3, loss_weights=[1, .02], **kwargs):
        super().__init__(learning_rate, loss_weights, **kwargs)

        self.ssim = StructuralSimilarityIndexMeasure(data_range=1.0)
        self.psnr = PeakSignalNoiseRatio(data_range=1.0)

        self.save_hyperparameters()

    def reconstruction_fn(self, rgb_output, alpha_output, airlight):
        """
        Reconstructs the output by performing element-wise multiplication of RGB layers with alpha layers.

        Args:
            rgb_output (torch.Tensor): The RGB output tensor.
            alpha_output (torch.Tensor): The alpha output tensor.

        Returns:
            torch.Tensor: The reconstructed output tensor.
        """

        return (alpha_output * rgb_output) + ((1 - alpha_output) * airlight.view(airlight.size(0), airlight.size(1), 1, 1))
    
    def training_step(self, batch, batch_idx):
        outputs = self.inference(batch, batch_idx, airlight=batch['airlight'])

        prev_output = self(img=batch['prev_input'])['rgb'].detach()

        rec_loss = self.reconstruction_loss(outputs['input'], outputs['reconstructed'])
        warp_loss = self.warp_loss(outputs['flow'], prev_output, outputs['alpha_output'])

        loss = self.loss_weights[0] * rec_loss + self.loss_weights[1] * warp_loss

        self.log('train_loss', loss, on_step=True, on_epoch=True, prog_bar=True, batch_size=4)
        self.log('rec_loss', rec_loss, on_step=False, on_epoch=True, prog_bar=False, batch_size=4)
        self.log('warp_loss', warp_loss, on_step=False, on_epoch=True, prog_bar=False, batch_size=4)

        return loss
    
    def validation_step(self, batch, batch_idx):
        outputs = self.inference(batch, batch_idx, airlight=batch['airlight'])
        
        rgb_output = outputs['rgb_output']
        gt = batch['target']

        # Compute PSNR and SSIM
        self.psnr(rgb_output, gt)
        self.ssim(rgb_output, gt)

        self.log('psnr', self.psnr, on_step=False, on_epoch=True, prog_bar=True, batch_size=4)
        self.log('ssim', self.ssim, on_step=False, on_epoch=True, prog_bar=True, batch_size=4)

        return outputs
    