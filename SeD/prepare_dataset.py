import os
from PIL import Image
from tqdm import tqdm
import argparse

def remove_postfix(image_path):
    """Remove scaling factor (e.g., x2, x3, x4, x8) from image filename."""
    base_name = image_path.split('/')[-1]
    clean_name = base_name.replace('x2', '').replace('x3', '').replace('x4', '').replace('x8', '')
    return clean_name

def prepare_dataset(input_image_path, image_name, save_folder, crop_size, file_extension, tqdm_desc=None):
    """Crop patches from input image and save them to the specified folder."""
    # Open the input image using PIL (Pillow)
    input_image = Image.open(input_image_path)
    image_width, image_height = input_image.size

    # Calculate the number of patches in each dimension
    num_patches_width = (image_width - crop_size) // crop_size + 1
    num_patches_height = (image_height - crop_size) // crop_size + 1

    # Initialize index for cropped image naming
    cnt = 0

    # Loop through all possible cropping positions
    for i in tqdm(range(num_patches_height), desc=tqdm_desc):
        for j in range(num_patches_width):
            cnt += 1
            # Calculate the starting position for the current patch
            start_width = j * crop_size
            start_height = i * crop_size

            # Crop the image using PIL
            cropped_image = input_image.crop((start_width, start_height, 
                                            start_width + crop_size, start_height + crop_size))

            # Convert the cropped image to RGB mode if it's not already
            if cropped_image.mode != 'RGB':
                cropped_image = cropped_image.convert('RGB')

            # Construct output file path
            output_file_path = os.path.join(save_folder, f'{image_name}_crop{cnt:03d}{file_extension}')

            # Save the cropped image
            cropped_image.save(output_file_path)

if __name__ == '__main__':
    # Define command line arguments
    parser = argparse.ArgumentParser(description='Prepare dataset by cropping images.')
    parser.add_argument('--crop_size_hr', type=int, default=400, help='Crop size for high resolution (HR) images')
    parser.add_argument('--crop_size_lr', type=int, default=100, help='Crop size for low resolution (LR) images')
    parser.add_argument('--hr_folder', type=str, default='data/hr', help='Directory containing the input images for HR')
    parser.add_argument('--lr_folder', type=str, default='data/lr', help='Directory containing the input images for LR')
    parser.add_argument('--output_folder', type=str, default='data/dataset_cropped', help='Output folder for cropped images')
    args = parser.parse_args()

    # Directory containing the input images for HR and LR
    input_folders = {
        'hr': args.hr_folder,
        'lr': args.lr_folder
    }
    os.makedirs(args.output_folder, exist_ok=True)
    # Loop through HR and LR image folders
    for image_type, input_folder in input_folders.items():
        # Determine save folder based on image type (hr or lr)
        save_folder = os.path.join(args.output_folder, image_type)
        os.makedirs(save_folder, exist_ok=True)

        # Find all PNG files in the input folder
        matching_files = [file for file in os.listdir(input_folder) if file.endswith('.png')]
        matching_files = sorted([os.path.join(input_folder, file) for file in matching_files])

        # Process each image file
        print(f"Processing {image_type.upper()} images:")
        for img_path in tqdm(matching_files):
            img_name, file_extension = os.path.splitext(os.path.basename(img_path))
            image_name = remove_postfix(img_name)
            
            # Choose appropriate crop size based on image type (hr or lr)
            crop_size = args.crop_size_hr if image_type == 'hr' else args.crop_size_lr
            
            # Prepare dataset by cropping images
            prepare_dataset(img_path, image_name, save_folder, crop_size, file_extension,
                            tqdm_desc=f"Processing {image_type.upper()} images")
