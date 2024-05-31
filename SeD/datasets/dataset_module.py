from .combined_dataset_train import CombinedDatasetTrain
from .combined_dataset_test import CombinedDatasetTest
from torch.utils.data import DataLoader
import pytorch_lightning as pl

class DatasetModule(pl.LightningDataModule):
    def __init__(self,
                 num_workers,  # Number of workers for data loading
                 train_batch_size,  # Batch size for training dataset
                 val_batch_size,  # Batch size for validation dataset
                 test_batch_size,  # Batch size for test dataset
                 train_dataset_config,  # Configuration for training dataset
                 val_dataset_config,  # Configuration for validation dataset
                 test_dataset_config  # Configuration for test dataset
                 ):
        super().__init__()
        self.num_workers = num_workers
        self.train_batch_size = train_batch_size
        self.val_batch_size = val_batch_size
        self.test_batch_size = test_batch_size
        self.train_dataset_config = train_dataset_config
        self.val_dataset_config = val_dataset_config
        self.test_dataset_config = test_dataset_config
        
    def setup(self, stage: str):
        if stage == 'training':
            # Create training dataset using the provided configuration
            self.train_dataset = CombinedDatasetTrain(**self.train_dataset_config)
        elif stage == 'test':
            # Create test dataset using the provided configuration
            self.test_dataset = CombinedDatasetTest(**self.test_dataset_config)

    def train_dataloader(self):
        # Return a DataLoader for the training dataset
        return DataLoader(self.train_dataset, batch_size=self.train_batch_size,
                            shuffle=True, num_workers=self.num_workers)
    
    def test_dataloader(self):
        # Return a DataLoader for the test dataset
        return DataLoader(self.test_dataset, batch_size=self.test_batch_size,
                          shuffle=False, num_workers=self.num_workers)
