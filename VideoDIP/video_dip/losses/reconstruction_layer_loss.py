import torch
from torch import nn

from video_dip.losses.perceptual_loss import PerceptualLoss


class ReconstructionLayerLoss(nn.Module):
    """
    Reconstruction loss module combining L1 loss and perceptual loss.
    
    Attributes:
        l1_loss (L1Loss): L1 loss module.
    """
    def __init__(self):
        super(ReconstructionLayerLoss, self).__init__()
        self.l1_loss = nn.L1Loss()

    def forward(self, m, x, x_hat):
        """
        Calculate the reconstruction loss.

        Args:
            m (torch.Tensor) : alpha net output
            x (torch.Tensor): Ground truth tensor.
            x_hat (torch.Tensor): Predicted tensor.

        Returns:
            torch.Tensor: Combined reconstruction loss.
        """

        (b,i,h,w) = m.size()
        (b,c,h,w) = x.size()
        
        
        second_dim = i*c

        m_repeated = m.repeat(1,c,1,1)
        x_repeated = x.repeat(1,i,1,1)
        x_hat_repeated = x_hat.repeat(1,i,1,1)

        # their product will be b,i,c,h,w we need to get rid of the i in order to feed to the VGG
        # therefore put that dimension into the batch as well
        production = x_repeated * m_repeated
        pred_production = x_hat_repeated * m_repeated

        layer_loss = self.l1_loss(production, pred_production)
        
        return layer_loss
