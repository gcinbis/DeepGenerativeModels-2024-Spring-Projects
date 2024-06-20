from math import log10, sqrt 
import cv2 
import numpy as np 
import torch
import tqdm
from torch import Tensor
from torchmetrics import Metric
import torchvision
import torchvision.io as io

class PNSR(Metric):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.add_state("num_calculations", default=torch.tensor(0), dist_reduce_fx="sum")
        self.add_state("cumulative_pnsr", default=torch.tensor(0.), dist_reduce_fx="sum")

    def update(self, original: Tensor, calculated: Tensor) -> None:

        """
            
        C,H,W Tensor

        """
        assert original.shape == calculated.shape, "Two inputs should be same size"

        mse = torch.mean((original.float() - calculated.float()) ** 2) 
        if(mse == 0): 
            pnsr = 100
        else:
            max_pixel = torch.as_tensor(255.0)
            pnsr = 20 * torch.log10(max_pixel / sqrt(mse)) 

        self.num_calculations += 1
        self.cumulative_pnsr += pnsr

    def compute(self) -> Tensor:
        return self.cumulative_pnsr.float() / self.num_calculations

def pnsr_video(video_path1, video_path2):
    """
    Calculates PNSR score over two videos
    """

    video1, audio1, info1 = io.read_video(video_path1, pts_unit='sec', output_format = "TCHW")
    video2, audio2, info2 = io.read_video(video_path2, pts_unit='sec', output_format = "TCHW")

    assert len(video1) == len(video2), "Given videos do not have equal number of frames"
    print(video1.shape)
    pnsr_metric = PNSR()
    for i in tqdm.tqdm(range(video1.shape[0])):
         pnsr_metric.update(video1[i,...], video2[i,...]*0.8)
    return pnsr_metric.compute()

def main(): 
    # original = cv2.imread(r"C:\Users\bahcekap\VideoDIP\video_dip\metrics\test\compressed_image.png", cv2.IMREAD_COLOR) 
    # compressed = cv2.imread(r"C:\Users\bahcekap\VideoDIP\video_dip\metrics\test\original_image.png", cv2.IMREAD_COLOR) 
    # original = torch.Tensor(original)
    # compressed = torch.Tensor(compressed)
    # pnsr_metric = PNSR() 
    # pnsr_metric.update(original, compressed)
    # value = pnsr_metric.compute()
     
    # print(f"PSNR value is {value} dB") 

    video_p = r"C:\Users\bahcekap\VideoDIP\video_dip\data\sora.mp4"

    value = pnsr_video( video_p, video_p)
    print(value)
	
if __name__ == "__main__": 
	main() 
