# -*- coding: utf-8 -*-
"""ceng796project.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1M25NzqPGKpVK47wtCj7tsWu6csxF6v8b
"""


from IPython.display import Image as displayImage, display
import os
import matplotlib.pyplot as plt
import numpy as np
import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import os
from PIL import Image
import torch
from torch.utils.data import Dataset

"""# Check the matching of the images"""

def check_images(DATASET_TRAIN_HIGH_PATH,DATASET_TRAIN_LOW_PATH):
  files_train_high = os.listdir(DATASET_TRAIN_HIGH_PATH)
  files_train_low = os.listdir(DATASET_TRAIN_LOW_PATH)
  images_high = sorted([filename for filename in files_train_high if filename.endswith((".png"))])
  images_low = sorted([filename for filename in files_train_low if filename.endswith((".png"))])
  if not images_high:
      print("No images found in the directory.")
  else:
      # Get the path of the first image
      first_image_path_high = os.path.join(DATASET_TRAIN_HIGH_PATH, images_high[0])
      # Display the first image
      display(displayImage(filename=first_image_path_high))

  if not images_low:
      print("No images found in the directory.")
  else:
      # Get the path of the first image
      first_image_path_low = os.path.join(DATASET_TRAIN_LOW_PATH, images_low[0])
      # Display the first image
      display(displayImage(filename=first_image_path_low))

"""# Dataset Loader"""

class LOL_Dataset(Dataset):
    def __init__(self, root_dir):
        self.root_dir = root_dir
        self.transform =  transforms.Compose([
                          #transforms.Resize((224, 224)),  # Resize images to 224x224
                          transforms.ToTensor(),           # Convert images to tensors
                          #transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # Normalize images
                          ])

        self.image_files = self._list_image_files()

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        img_name = self.image_files[idx]
        img_path = os.path.join(self.root_dir, img_name)
        image = Image.open(img_path).convert('RGB')

        if self.transform:
            image = self.transform(image)

        return image

    def _list_image_files(self):
        image_files = []
        for root, _, files in os.walk(self.root_dir):
            for file in files:
                if file.endswith(('jpg', 'jpeg', 'png', 'bmp')):
                    image_files.append(os.path.relpath(os.path.join(root, file), self.root_dir))
        return image_files


# Define your dataset class
class LOL_Dataset_Diffusion(Dataset):
    def __init__(self, high_reflectance_dir, low_reflectance_dir, transform=None):
        self.high_reflectance_dir = high_reflectance_dir
        self.low_reflectance_dir = low_reflectance_dir
        self.transform = transform

        # Get the list of filenames in both directories
        self.high_filenames = sorted(os.listdir(high_reflectance_dir))
        self.low_filenames = sorted(os.listdir(low_reflectance_dir))

    def __len__(self):
        return len(self.high_filenames)

    def __getitem__(self, idx):
        high_reflectance_path = os.path.join(self.high_reflectance_dir, self.high_filenames[idx])
        low_reflectance_path = os.path.join(self.low_reflectance_dir, self.low_filenames[idx])

        high_reflectance_img = Image.open(high_reflectance_path)
        low_reflectance_img = Image.open(low_reflectance_path)

        if self.transform:
            high_reflectance_img = self.transform(high_reflectance_img)
            low_reflectance_img = self.transform(low_reflectance_img)

        return high_reflectance_img, low_reflectance_img
    
from torchvision.transforms import InterpolationMode
from torchvision.transforms import transforms
import os, cv2
from torch.utils.data import Dataset, DataLoader
class SRDataset(Dataset):
    def __init__(self, dataset_path_low, dataset_path_high, limit=-1) -> None:
        super().__init__()
        self.limit = limit
        self.valid_extensions = ["jpg", "jpeg", "png", "JPEG", "JPG"]

        self.images_path_low = dataset_path_low
        self.images_low = os.listdir(self.images_path_low)
        self.images_low = sorted([os.path.join(self.images_path_low, image) for image in self.images_low if
                                  image.split(".")[-1] in self.valid_extensions])
        # print(self.images_low)
        self.images_path_high = dataset_path_high
        self.images_high = os.listdir(self.images_path_high)
        self.images_high = sorted([os.path.join(self.images_path_high, image) for image in self.images_high if
                                   image.split(".")[-1] in self.valid_extensions])
        # print(self.images_high)
        self.transforms = transforms.Compose([
            transforms.ToTensor(),
        ])

    def __len__(self):
        return len(self.images_low)

    def __getitem__(self, index):
        image_path_low = self.images_low[index]
        image_path_high = self.images_high[index]

        # print(f"image_path: {image_path_low}, image_path: {image_path_high}")

        image_high = cv2.imread(image_path_high)
        image_low = cv2.imread(image_path_low)

        image_high = self.transforms(image_high)
        image_low = self.transforms(image_low)

        return image_low, image_high

