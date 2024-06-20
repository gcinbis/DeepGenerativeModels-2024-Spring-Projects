import torch
from torch import nn

from video_dip.losses.perceptual_loss import PerceptualLoss


class ReconstructionLoss(nn.Module):
    """
    Reconstruction loss module combining L1 loss and perceptual loss.
    
    Attributes:
        perceptual_loss (PerceptualLoss): Perceptual loss module.
    """
    def __init__(self):
        super(ReconstructionLoss, self).__init__()
        self.perceptual_loss = PerceptualLoss()
        self.l1_loss = nn.L1Loss()

    def forward(self, x, x_hat):
        """
        Calculate the reconstruction loss.

        Args:
            x (torch.Tensor): Ground truth tensor.
            x_hat (torch.Tensor): Predicted tensor.

        Returns:
            torch.Tensor: Combined reconstruction loss.
        """
        # l1_loss = torch.mean(torch.abs(x - x_hat))  # L1 loss
        l1_loss = self.l1_loss(x, x_hat)
        perceptual_loss = self.perceptual_loss(x, x_hat)  # Perceptual loss
        return l1_loss + perceptual_loss
