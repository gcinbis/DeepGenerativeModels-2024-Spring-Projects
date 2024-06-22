import pytorch_lightning as pl

try:
    from video_dip.data.dataset import VideoDIPDataset
    from video_dip.models.optical_flow import Farneback
except ImportError:
    import sys
    import os
    # Add the parent of the parent directory to the path
    sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir, os.pardir)))
    
    from video_dip.data.dataset import VideoDIPDataset
    from video_dip.models.optical_flow import Farneback

from torch.utils.data import DataLoader
import os

class VideoDIPDataModule(pl.LightningDataModule):
    def __init__(self, input_path, batch_size, num_workers, flow_path=None, target_path=None, airlight_est_path=None):
        """
        Initializes the VideoDIPDataModule.

        Args:
            input_path (str): Path to the input data.
            batch_size (int): Batch size for data loading.
            num_workers (int): Number of workers for data loading.
            target_path (str, optional): Path to the target data. Defaults to None.
            flow_path (str, optional): Path to the optical flow data. Defaults to None.
            airlight_est_path (str, optional): Path to the airlight estimation data. Defaults to None.
        """
        super().__init__()
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.input_path = input_path
        self.target_path = target_path
        self.flow_path = flow_path
        self.airlight_est_path = airlight_est_path


    def dump_optical_flow(self, flow_model):
        """
        Dump the optical flow to a folder.

        Args:
            path (str, optional): Path to the folder to save the optical flow. Defaults to None.

        Returns:
            str: Path to the saved optical flow folder.
        """
        from tqdm.auto import tqdm
        import numpy as np

        flow_folder = self.flow_path
        if os.path.exists(flow_folder):
            import shutil
            shutil.rmtree(flow_folder)
        os.makedirs(flow_folder)
        
        dataset = VideoDIPDataset(
            self.input_path,
            transforms=VideoDIPDataset.default_flow_transforms()
        )
        
        for i in tqdm(range(1, len(dataset))):
            img1 = dataset[i - 1]['input']
            img2 = dataset[i]['input']
            base_name = dataset[i]['filename']
            flow = flow_model(img1, img2)
            # Save the flow as npy
            np.save(os.path.join(flow_folder, base_name), flow.transpose((1, 2, 0)))

            from torchvision.utils import flow_to_image, save_image
            import torch
            # Save the flow as image
            flow_image = flow_to_image(torch.tensor(flow)) / 255.0
            save_image(flow_image, os.path.join(flow_folder, base_name.replace('.npy', '.png')))
            
        return flow_folder

    def setup(self, stage=None):
        """
        Set up the data module.

        Args:
            stage (str, optional): Stage of the training. Defaults to None.
        """
        assert os.path.exists(self.flow_path), "The optical flow path does not exist. Did you forget to dump the optical flow before the training loop?"
        self.dataset = VideoDIPDataset(input_path=self.input_path, target_path=self.target_path, flow_path=self.flow_path, airlight_est_path=self.airlight_est_path)

    def train_dataloader(self):
        """
        Returns the data loader for training.

        Returns:
            torch.utils.data.DataLoader: Data loader for training.
        """
        return DataLoader(self.dataset, batch_size=self.batch_size, num_workers=self.num_workers)
    
    def val_dataloader(self):
        """
        Returns the data loader for validation.

        Returns:
            torch.utils.data.DataLoader: Data loader for validation.
        """
        return DataLoader(self.dataset, batch_size=self.batch_size, num_workers=self.num_workers)
    
    def test_dataloader(self):
        """
        Returns the data loader for testing.

        Returns:
            torch.utils.data.DataLoader: Data loader for testing.
        """
        return DataLoader(self.dataset, batch_size=self.batch_size, num_workers=self.num_workers)
    
if __name__ == '__main__':
    from video_dip.models.optical_flow.raft import RAFT, RAFTModelSize

    module = VideoDIPDataModule("datasets/input/pair1", batch_size=2, num_workers=8, flow_model=Farneback())#RAFT(RAFTModelSize.LARGE))
    import os
    if os.path.exists("flow_outputs_2"):
        import shutil
        shutil.rmtree("flow_outputs_2")
    module.dump_optical_flow('flow_outputs_2')
    