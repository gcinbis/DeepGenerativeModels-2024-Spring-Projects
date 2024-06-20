import cv2
import numpy as np
import torch
from torch import nn

class Farneback:
    """
    Optical flow estimation using Lucas-Kanade method from OpenCV.
    """
    def __init__(self):
        pass

    def _farneback_one_image(self, image1, image2):
        """
        Estimate the optical flow between two images.

        Args:
            image1 (np.ndarray): First image (grayscale).
            image2 (np.ndarray): Second image (grayscale).

        Returns:
            np.ndarray: Estimated optical flow.
        """
        # Permute dimensions if necessary
        if (len(image1.shape) == 3) and (image1.shape[0] == 3):
            image1 = np.transpose(image1, (1, 2, 0))
            image2 = np.transpose(image2, (1, 2, 0))

        if (len(image1.shape) == 3) and (image1.shape[2] == 3):
            image1 = cv2.cvtColor(image1, cv2.COLOR_RGB2GRAY)
            image2 = cv2.cvtColor(image2, cv2.COLOR_RGB2GRAY)

        # Calculate optical flow
        flow = cv2.calcOpticalFlowFarneback(image1, image2, None, 0.5, 3, 15, 3, 5, 1.2, 0)
        # # Compute magnite and angle of 2D vector
        # mag, ang = cv2.cartToPolar(flow[..., 0], flow[..., 1])

        # hsv_mask = np.zeros((image1.shape[0], image1.shape[1], 3), dtype=np.uint8)
        # hsv_mask[..., 1] = 255

        # # Set image hue value according to the angle of optical flow
        # hsv_mask[..., 0] = ang * 180 / np.pi / 2
        # # Set value as per the normalized magnitude of optical flow
        # hsv_mask[..., 2] = cv2.normalize(mag, None, 0, 255, cv2.NORM_MINMAX)
        # # Convert to rgb
        # rgb_representation = cv2.cvtColor(hsv_mask, cv2.COLOR_HSV2BGR)
        # Convert flow to tensor
        flow_rgb = torch.tensor(flow).permute(2, 0, 1).unsqueeze(0).float()
        return flow_rgb
    
    def forward(self, image1, image2):
        """
        Estimate the optical flow between two images.

        Args:
            image1 (torch.Tensor): First image (grayscale).
            image2 (torch.Tensor): Second image (grayscale).

        Returns:
            torch.Tensor: Estimated optical flow.
        """
        image1 = image1.squeeze().cpu().numpy()
        image2 = image2.squeeze().cpu().numpy()

        # Convert images to uint8
        image1 = np.uint8(image1 * 255)
        image2 = np.uint8(image2 * 255)

        ret = None
        # Convert images to grayscale if they are RGB
        if len(image1.shape) == 4:
            flows = []
            for i in range(len(image1)):
                flows.append(self._farneback_one_image(image1[i], image2[i]))
            ret = torch.cat(flows)
        else:
            ret = self._farneback_one_image(image1, image2)

        return self.postprocess(ret)
        
    def postprocess(self, flow):
        """
        Postprocess the estimated optical flow.

        Args:
            flow (torch.Tensor): Estimated optical flow.

        Returns:
            np.ndarray: Postprocessed optical flow.
        """
        return flow.squeeze().numpy()
        
    def forward_clip(self, clip):
        """
        Estimate the optical flow between frames in a video clip.

        Args:
            clip (torch.Tensor): Video clip with frames (grayscale).

        Returns:
            torch.Tensor: Estimated optical flow.
        """
        clip = clip.squeeze().cpu().numpy()
        clip = np.uint8(clip * 255)

        flows = []
        for i in range(len(clip) - 1):
            flows.append(self._farneback_one_image(clip[i], clip[i + 1]))
        return torch.cat(flows)
        
    def __call__(self, inp, inp2 = None):
        if inp2 is not None:
            return self.forward(inp, inp2)
        else:
            return self.forward_clip(inp)

if __name__ == '__main__':
    import sys
    import os
    # Add parent of parent directory to path
    sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir, os.pardir)))

    from video_dip.data import VideoDIPDataset
    from torch.utils.data import DataLoader

    dataset = VideoDIPDataset("video_dip/data/sora.mp4")
    data_loader = DataLoader(dataset, batch_size=2, num_workers=8)

    lucas_kanade = Farneback()
    batch = next(iter(data_loader))
    flow = lucas_kanade(batch['input'])

    print(flow.shape)

    import PIL
    PIL.Image.fromarray((flow.squeeze().permute(1, 2, 0).numpy()).astype('uint8')).show()
