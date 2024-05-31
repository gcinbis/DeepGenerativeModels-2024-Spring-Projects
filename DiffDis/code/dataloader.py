from config import *
import torch
from torch.utils.data import DataLoader, Dataset
import torch.nn.functional as F
from torchvision import transforms
from random import randint
import os
from PIL import Image
from transformers import CLIPTokenizer

# apply the following transformations to the images
# Preprocessing the datasets.
# train_transforms = transforms.Compose(
#     [
#         transforms.Resize(args.resolution, interpolation=transforms.InterpolationMode.BILINEAR),
#         transforms.CenterCrop(args.resolution) if args.center_crop else transforms.RandomCrop(args.resolution),
#         transforms.RandomHorizontalFlip() if args.random_flip else transforms.Lambda(lambda x: x),
#         transforms.ToTensor(),
#         transforms.Normalize([0.5], [0.5]),
#     ]
# )

# def preprocess_train(examples):
#         images = [image.convert("RGB") for image in examples[image_column]]
#         examples["pixel_values"] = [train_transforms(image) for image in images]
#         examples["input_ids"] = tokenize_captions(examples)
#         return examples


# Parameters for the dummy dataset
# num_samples = 100
# batch_size = BATCH_SIZE
# image_dim = (WIDTH, HEIGHT)
# num_tokens = 77

# class DummyDataset(Dataset):
#     def __init__(self, num_samples=100, image_dim=(512, 512), num_tokens=10):
#         self.num_samples = num_samples
#         self.image_dim = image_dim
#         self.num_tokens = num_tokens
    
#     def __len__(self):
#         return self.num_samples
    
#     def __getitem__(self, idx):
#         # Generate a dummy image
#         image = torch.rand(3, *self.image_dim)  # Random image
#         # Generate dummy tokenized text (integer IDs)
#         text = torch.randint(1, 1000, (self.num_tokens,))  # Random token IDs
#         return {"pixel_values": image, "input_ids": text}


# # Create a dummy dataset and dataloader
# dummy_dataset = DummyDataset(num_samples=num_samples, image_dim=image_dim, num_tokens=num_tokens)
# train_dataloader = DataLoader(dummy_dataset, batch_size=batch_size, shuffle=True)


# Initialize the tokenizer
tokenizer = CLIPTokenizer("./data/vocab.json", merges_file="./data/merges.txt")

class CC3MDataset(Dataset):
    def __init__(self, root_dir, tokenizer, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.tokenizer = tokenizer
        self.samples = self._load_samples()

    def _load_samples(self):
        samples = []

        # Extract unique base names (without extensions)
        base_names = set()
        for file in os.listdir(self.root_dir):
            base_name = file.split('.')[0]
            base_names.add(base_name)

        # Create samples
        counter = 0
        for base_name in base_names:
            counter += 1
            print(f"\rLoading samples: {counter}/{len(base_names)}", end="")
            image_path = os.path.join(self.root_dir, f"{base_name}.jpg")
            text_path = os.path.join(self.root_dir, f"{base_name}.txt")
            json_path = os.path.join(self.root_dir, f"{base_name}.json")
            if os.path.exists(image_path) and os.path.exists(text_path) and os.path.exists(json_path):
                samples.append((image_path, text_path, json_path))
                
        print(f"\nLoaded {len(samples)} samples.\n")

        return samples

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        image_path, text_path, json_path = self.samples[idx]
        
        # Load image
        image = Image.open(image_path).convert("RGB")
        if self.transform:
            image = self.transform(image)
        
        # Load text
        with open(text_path, 'r') as file:
            text = file.read()
        
        # Tokenize text
        tokens = self.tokenizer.batch_encode_plus(
            [text], padding="max_length", max_length=77
        ).input_ids
        tokens = torch.tensor(tokens, dtype=torch.long)

        return {
            'pixel_values': image,
            'input_ids': tokens.squeeze()
        }

transform = transforms.Compose([
    transforms.Resize((WIDTH, HEIGHT), interpolation=transforms.InterpolationMode.BILINEAR),
    transforms.ToTensor(),
    transforms.Normalize([0.5], [0.5])
])

dataset = CC3MDataset(root_dir, tokenizer, transform=transform)
train_dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

