import torch
import torchvision
from . import VDPModule
from torchmetrics import PeakSignalNoiseRatio, StructuralSimilarityIndexMeasure
from video_dip.losses import FlowSimilarityLoss, MaskLoss, ReconstructionLayerLoss
from video_dip.models.unet import UNet
import torch.nn as nn
from torchmetrics.classification import BinaryJaccardIndex
from torchvision.transforms.functional import rgb_to_grayscale
from video_dip.models.optical_flow.raft import RAFT, RAFTModelSize

class SegmentationVDPModule(VDPModule):
    """
    A module for segmentation in VideoDIP.

    Args:
        learning_rate (float): The learning rate for optimization (default: 1e-3).
        loss_weights (list): The weights for different losses (default: [1, .02]).
            flow similarity weight
            reconstruction loss weight
            layer loss weight
            warp loss weight
            mask mask loss weight
    """
        #λrec = 1, λFsim = 0.001, λlayer = 1,
        #λwarp = 0.01 and λMask = 0.01.


    def __init__(self, learning_rate=1e-3, loss_weights=[.001, 1 , 1, .01, .01], **kwargs):
        super().__init__(learning_rate, loss_weights, **kwargs)


        # one additional rgb net
        self.rgb_net2 = UNet(out_channels=3)  # RGB-Net with 3 input and 3 output channels

        self.flow_similarity_loss = FlowSimilarityLoss()
        self.rec_layer_loss = ReconstructionLayerLoss()
        self.mask_loss = MaskLoss()


        self.iou_metric = BinaryJaccardIndex(threshold=0.5)

        self.save_hyperparameters()

    def reconstruction_fn(self, rgb_output1, rgb_output2, alpha_output):
        """
        Reconstructs the output by performing element-wise multiplication of RGB layers with alpha layers.

        Args:
            rgb_output1 (torch.Tensor): The RGB output tensor.
            rgb_output2 (torch.Tensor): The RGB output tensor.
            alpha_output (torch.Tensor) b,i,h,w : The alpha output tensor.

        Returns:
            torch.Tensor: The reconstructed output tensor.
        """
        # bloew part is only for L>2
        # rescale Mi's so that after addition of all masks they sum up to one (falpha)
        # summed_masks = torch.sum(alpha_output, dim = 1)
        # summed_masks = torch.unsqueeze(summed_masks, 1)

        # alpha_output = alpha_output / summed_masks

        # remap alpha output values to 1 and 0's

        # alpha_copy = torch.ones(alpha_output.shape, device = alpha_output.device)        
        # alpha_copy[alpha_output <= 0.5] = 0


        return alpha_output * rgb_output1 + (1 - alpha_output) * rgb_output2
    
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
            ret["rgb2"] = self.rgb_net2(img)
        if flow is not None:
            ret['alpha'] = self.alpha_net(flow)
        return ret

    def inference(self, batch, batch_idx):
        input_frames = batch['input']
        flows = batch['flow']
        flow_frames = torchvision.utils.flow_to_image(flows) / 255.0

        output = self(img=input_frames, flow=flow_frames)
        rgb_output = output['rgb']
        rgb_output2 = output["rgb2"]
        alpha_output = output['alpha']

        reconstructed_frame = self.reconstruction_fn(rgb_output, rgb_output2 ,alpha_output)

        return {
            "input": input_frames,
            "flow": flows,
            "flow_rgb": flow_frames,
            "reconstructed": reconstructed_frame,
            "rgb_output": rgb_output,
            "rgb_output2": rgb_output2,
            "alpha_output": alpha_output
        }
    
    

    def training_step(self, batch, batch_idx):
        outputs = self.inference(batch, batch_idx)
        
        flow_estimate = torchvision.utils.flow_to_image(batch['prev_flow']) / 255.0
        prev_output = self(img=batch['prev_input'])['rgb'].detach()

        x = batch["input"]
        x_hat = outputs["reconstructed"]



        flow_sim_loss = self.flow_similarity_loss(m = outputs['alpha_output'], frgb = flow_estimate)
        rec_loss = self.reconstruction_loss(x = x, x_hat = x_hat)
        rec_layer_loss = self.rec_layer_loss(m = flow_estimate, x = x, x_hat = x_hat)
        warp_loss = self.warp_loss(outputs['flow'], prev_output, outputs['alpha_output'])
        mask_loss = self.mask_loss(outputs['alpha_output'])


        #     flow similarity weight
        #     reconstruction loss weight
        #     layer loss weight
        #     warp loss weight
        #     mask mask loss weight



        flow_sim_loss =self.loss_weights[0] * torch.sum( flow_sim_loss)
        rec_loss = self.loss_weights[1] * torch.sum( rec_loss)
        rec_layer_loss = self.loss_weights[2] * torch.sum( rec_layer_loss)
        warp_loss = self.loss_weights[3] * torch.sum( warp_loss)
        mask_loss = self.loss_weights[4] * torch.sum( mask_loss)


        loss = flow_sim_loss + rec_loss + rec_layer_loss + warp_loss + mask_loss

        



        self.log("train_loss", loss, prog_bar=True, on_step=False, on_epoch=True)
        self.log("rec_loss", rec_loss, prog_bar=False, on_step=False, on_epoch=True)
        self.log("warp_loss", warp_loss, prog_bar=False, on_step=False, on_epoch=True)
        self.log("flow_sim_loss", flow_sim_loss, prog_bar=False, on_step=False, on_epoch=True)
        self.log("rec_layer_loss", rec_layer_loss, prog_bar=False, on_step=False, on_epoch=True)
        self.log("mask_loss", mask_loss, prog_bar=False, on_step=False, on_epoch=True)
        

        return loss
    

    
    def validation_step(self, batch, batch_idx):
        outputs = self.inference(batch, batch_idx)

        # binarize segmentation groundtruth
        gray_gt = rgb_to_grayscale(batch["target"])
        gray_gt[gray_gt <= 0.5] = 0
        gray_gt[gray_gt > 0.5] = 1
        iou_score = self.iou_metric(outputs["alpha_output"], gray_gt)
        self.log('iou_score', iou_score, on_step=False, on_epoch=True, prog_bar=True)
        
        
        treshold = 0.5
        outputs["segmentation"] = torch.ones(outputs["alpha_output"].shape)
        outputs["segmentation"][outputs["alpha_output"] <= treshold] = 0
        
        outputs["segmentation_gt"] = gray_gt
        
        return outputs
    
