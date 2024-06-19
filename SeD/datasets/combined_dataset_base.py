import os
from torch.utils.data import Dataset

class CombinedDatasetBaseClass(Dataset):
    """
    Base class for the combined dataset
    Will be subclassed by the main dataset classes
    """
    def __init__(self, image_size=256, image_dir_hr="dataset_cropped_hr", image_dir_lr="dataset_cropped_lr", downsample_factor=4):
        # Initialize the dataset with the given parameters
        self.image_size = image_size
        self.image_dir_hr = image_dir_hr
        self.image_dir_lr = image_dir_lr
        self.image_names_hr = sorted(os.listdir(self.image_dir_hr))
        self.image_names_lr = sorted(os.listdir(self.image_dir_lr))
        self.num_samples = len(self.image_names_hr)

        self.down_sample_factor = downsample_factor
        self.down_sampled_image_size = self.image_size // self.down_sample_factor
        
        # Ensure that the number of images in HR and LR directories are the same
        assert len(self.image_names_hr) == len(self.image_names_lr), "Number of images in HR and LR directories should be same"

    def __getitem__(self, index):
        # Implement the logic to retrieve an item from the dataset at the given index
        pass

    def __len__(self):
        # Return the total number of samples in the dataset
        return self.num_samples
