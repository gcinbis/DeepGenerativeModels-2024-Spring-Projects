import torch
from torch import nn
# from torchvision.models import vgg16
from video_dip.models import VGG

class PerceptualLoss(nn.Module):
    """
    Perceptual loss module using a pretrained VGG16 network.
    
    Attributes:
        vgg (nn.Module): Pretrained VGG16 network truncated at the 16th layer.
    """
    def __init__(self):
        super(PerceptualLoss, self).__init__()
        self.l1_loss = nn.L1Loss()

    def forward(self, x, y):
        """
        Calculate the perceptual loss between two tensors.

        Args:
            x (torch.Tensor): Predicted tensor.
            y (torch.Tensor): Ground truth tensor.

        Returns:
            torch.Tensor: Perceptual loss.
        """
        try:
            x_features = VGG(x)
        except RuntimeError:
            VGG.to(x.device)
            x_features = VGG(x)
        y_features = VGG(y)
        return self.l1_loss(x_features, y_features)
