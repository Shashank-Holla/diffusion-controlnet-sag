import cv2
from PIL import Image
import numpy as np


def canny_edge_detection(imagename, low_threshold=100, high_threshold=200):
    image = Image.open(imagename)
    image = np.array(image)
    image = cv2.Canny(image, low_threshold, high_threshold)
    image = image[:, :, None]
    image = np.concatenate([image, image, image], axis=2)
    canny_image = Image.fromarray(image)
    return canny_image


def openpose_detection(imagename, openpose_model=openpose):
    image = Image.open(imagename)
    image = np.array(image)
    image = openpose_model(image)
    return image

def resize(image):
    height = image.height
    width = image.width
    width, height = (x - x % 8 for x in (width, height))  # resize to integer multiple of vae_scale_factor
    image = image.resize((width, height), resample=Image.BILINEAR)
    return image

def convert_to_rgb(image):
    image = image.convert("RGB")
    return image




