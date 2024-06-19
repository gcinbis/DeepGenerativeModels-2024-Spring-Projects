from datasets import load_dataset
from pathlib import Path
import os
import sys
import argparse

parser = argparse.ArgumentParser(description='Download dataset and save images.')

parser.add_argument('-p', '--path', type=str, help='Path to save the images.')

args = parser.parse_args()

if args.path:
    dataset_path = Path(args.path)
    dataset = load_dataset("huggan/few-shot-obama")
    train_dataset = dataset['train']
    if not os.path.exists(dataset_path):
        os.mkdir(dataset_path)
    for i, image in enumerate(train_dataset):
        image['image'].save(dataset_path / f'{i}.jpg')
else:
    print("No path provided. Use -p or --path to provide a path to save the images.")