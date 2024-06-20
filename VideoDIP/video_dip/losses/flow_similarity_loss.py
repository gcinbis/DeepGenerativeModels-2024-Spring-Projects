import torch
from torch import nn
from torchvision.models import vgg16
from video_dip.models.vgg import VGG
class FlowSimilarityLoss(nn.Module):
    """
    
    
    Attributes:
        
    """
    def __init__(self):
        super(FlowSimilarityLoss, self).__init__()
        

    def forward(self, m, frgb):
        """
        Calculate the flow similarity loss.

        Args:
            M (torch.Tensor): b,i(ith layer),h,w Motion tensor of all video layers at time t(mask)
            F^RGB (torch.Tensor): b,c,h,w RGB image flow from t-1 to t

        Returns:
            torch.Tensor: Flow Similarity Loss
        """
        (b,i,h,w) = m.size()
        (b,c,h,w) = frgb.size()

        

        m = m.repeat(1,c,1,1)
        frgb = frgb.repeat(1,i,1,1)
        
        # their product will be b,i,c,h,w we need to get rid of the i in order to feed to the VGG
        # therefore put that dimension into the batch as well
        

        production = frgb * m
        neg_production = frgb * (1-m)
        VGG.to(m.device)
        product = VGG(production)
        neg_product = VGG(neg_production)
        lfsim = (product * neg_product) / (torch.norm(product) * torch.norm(neg_product))
        return lfsim
