from skimage.metrics import structural_similarity as ssim
from torchmetrics import Metric

import cv2
import torch
import tqdm
from torch import Tensor
import torchvision
import torchvision.io as io

class SSIM(Metric):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.add_state("num_calculations", default=torch.tensor(0), dist_reduce_fx="sum")
        self.add_state("cumulative_ssim", default=torch.tensor(0.), dist_reduce_fx="sum")

    def update(self, original: Tensor, calculated: Tensor) -> None:

        """
            
        C,H,W Tensor

        """
        assert original.shape == calculated.shape, "Two inputs should be same size"
        
        # cast to np
        origin_np = original.numpy()
        calculated_np = calculated.numpy()
        cumulative_ssim = 0
        # iterate over channels
        for c in range(origin_np.shape[0]): 
            temp_s = ssim(origin_np[c], calculated_np[c], data_range = 255.0)
            cumulative_ssim += temp_s

        self.num_calculations += 1
        self.cumulative_ssim += (cumulative_ssim / origin_np.shape[0])

    def compute(self) -> Tensor:
        return self.cumulative_ssim.float() / self.num_calculations



def ssim_video(video_path1, video_path2):
    """
    Calculates SSIM score over two videos
    """

    video1, audio1, info1 = io.read_video(video_path1, pts_unit='sec', output_format = "TCHW")
    video2, audio2, info2 = io.read_video(video_path2, pts_unit='sec', output_format = "TCHW")

    assert len(video1) == len(video2), "Given videos do not have equal number of frames"
    print(video1.shape)
    ssim_metric = SSIM()
    for i in tqdm.tqdm(range(len(video1))):
         ssim_metric.update(video1[i], video2[i])
    return ssim_metric.compute()
        
		
    


def main(): 
     # original = cv2.imread(r"C:\Users\bahcekap\VideoDIP\video_dip\metrics\test\compressed_image.png", cv2.IMREAD_COLOR) 
     # compressed = cv2.imread(r"C:\Users\bahcekap\VideoDIP\video_dip\metrics\test\original_image.png", cv2.IMREAD_COLOR) 
     # original = torch.as_tensor(original).permute((2,0,1))
     # compressed = torch.as_tensor(compressed).permute((2,0,1))
     # ssim_metric = SSIM()
     # ssim_metric.update(original, compressed)
     # ssim_metric.update(original, compressed*0.8)
     # value = ssim_metric.compute()
     # print(f"SSIM value is {value} ") 
    
    video_p = r"C:\Users\bahcekap\VideoDIP\video_dip\data\sora.mp4"
    ssim_score = ssim_video(video_p, video_p)
    print(ssim_score)
	
if __name__ == "__main__": 
	main() 


