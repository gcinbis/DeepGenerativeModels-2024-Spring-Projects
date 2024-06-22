import torch
from torch import nn
import torch.nn.functional as F

class OpticalFlowWarpLoss(nn.Module):
    """
    Optical flow warp loss module to ensure temporal coherence.
    """
    def __init__(self):
        super(OpticalFlowWarpLoss, self).__init__()
        self.l2_loss = nn.MSELoss()

    def forward(self, flow, prev_out, alpha_out):
        """
        Calculate the optical flow warp loss.

        Args:
            flow (torch.Tensor): Optical flow tensor.
            x (torch.Tensor): Ground truth tensor at time t-1.
            x_hat (torch.Tensor): Predicted tensor at time t.

        Returns:
            torch.Tensor: Optical flow warp loss.
        """
        warped = self.warp(prev_out, flow)
        return self.l2_loss(warped, alpha_out.repeat(1, 3, 1, 1))
    
    @staticmethod
    def warp(x, flow):
        """
        Warp an image/tensor (x) according to the given flow.

        Args:
            x (torch.Tensor): Image tensor of shape (N, C, H, W).
            flow (torch.Tensor): Optical flow tensor of shape (N, 2, H, W).

        Returns:
            torch.Tensor: Warped image tensor of shape (N, C, H, W).
        """
        N, C, H, W = x.size()
        # Create mesh grid
        grid_y, grid_x = torch.meshgrid(torch.arange(0, H), torch.arange(0, W))
        grid = torch.stack((grid_x, grid_y), 2).float()  # (H, W, 2)
        grid = grid.unsqueeze(0).repeat(N, 1, 1, 1).to(x.device)  # (N, H, W, 2)
        
        # Normalize the grid to [-1, 1]
        grid[:, :, :, 0] = 2.0 * grid[:, :, :, 0] / (W - 1) - 1.0
        grid[:, :, :, 1] = 2.0 * grid[:, :, :, 1] / (H - 1) - 1.0
        
        # Add optical flow to the grid
        flow = flow.permute(0, 2, 3, 1)  # (N, H, W, 2)
        vgrid = grid + flow
        
        # Normalize vgrid to [-1, 1]
        vgrid[:, :, :, 0] = 2.0 * vgrid[:, :, :, 0] / (W - 1) - 1.0
        vgrid[:, :, :, 1] = 2.0 * vgrid[:, :, :, 1] / (H - 1) - 1.0
        
        # Perform grid sampling
        output = F.grid_sample(x, vgrid, mode='bilinear', padding_mode='border', align_corners=True)
        return output