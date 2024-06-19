import torchvision
from PIL import Image, ImageOps
import numpy as np

def normalize_image(image):
    return (image-image.min()) / (image.max()-image.min())

def get_labeled_image(image, border_color):
        image = image.detach().cpu().numpy()
        image = (image + 1) / 2 # Range: [-1,1] -> [0,1]
        image = (image * 255).astype(np.uint8)
        if image.shape[0] == 1:
            image = np.stack((image[0],)*3, axis=0)
        image = image.transpose(1, 2, 0)
        image = Image.fromarray(image, "RGB")
        border_size = int(min(image.size) * 0.05)
        image = ImageOps.expand(image, border=border_size, fill=border_color)
        image = torchvision.transforms.ToTensor()(image).unsqueeze_(0)
        return image

def get_labeled_images(image_list, border_color):
    labeled_image_list = []
    for image in image_list:
        labeled_image_list.append(get_labeled_image(image, border_color))
    return labeled_image_list