import torch
from torch import nn
from torchvision.models import vgg16
from video_dip.models.vgg import VGG
class MaskLoss(nn.Module):
    """
    
    
    Attributes:
        
    """
    def __init__(self):
        super(MaskLoss, self).__init__()
        

    def forward(self, m):
        """
        Calculate the flow similarity loss.

        Args:
            M (torch.Tensor): b,i(ith layer),h,w Motion tensor of all video layers at time t(mask)


        Returns:
            torch.Tensor: Mask Loss
        """
        (b,i,h,w) = m.size()
        return torch.sum(torch.abs(m - 0.5)) ** -1

