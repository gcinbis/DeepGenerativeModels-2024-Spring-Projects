import os
import argparse
from PIL import Image
from tqdm import tqdm

def downscale_images(folder_path, save_path):
    # Iterate over each file in the folder
    for filename in tqdm(os.listdir(folder_path), desc="Downscaling images"):
        # Check if the file is an image
        if filename.endswith(".jpg") or filename.endswith(".png"):
            # Open the image
            image_path = os.path.join(folder_path, filename)
            image = Image.open(image_path)
            
            # Downscale the image by 4
            width, height = image.size
            new_width = width // 4
            new_height = height // 4
            downscaled_image = image.resize((new_width, new_height))
            
            # Save the downscaled image to the specified save path
            save_filename = os.path.join(save_path, filename)
            downscaled_image.save(save_filename)
            
            print(f"Downscaled {filename} and saved to {save_filename}")

if __name__ == "__main__":
    # Parse the command-line arguments
    parser = argparse.ArgumentParser(description="Downscale images in a folder by 4")
    parser.add_argument("--folder", type=str, help="Path to the folder containing images")
    parser.add_argument("--save_path", type=str, help="Path to save the downscaled images")
    args = parser.parse_args()
    
    # Get the folder path and save path from the command-line arguments
    folder_name = args.folder
    save_path = args.save_path
    
    # Call the function to downscale the images
    downscale_images(folder_name, save_path)