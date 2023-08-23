import cv2
from PIL import Image
import numpy as np
import torch
import torch.nn.functional as F


def canny_edge_detection(imagename, low_threshold=100, high_threshold=200):
    image = Image.open(imagename)
    image = np.array(image)
    image = cv2.Canny(image, low_threshold, high_threshold)
    image = image[:, :, None]
    image = np.concatenate([image, image, image], axis=2)
    canny_image = Image.fromarray(image)
    return canny_image


def openpose_detection(imagename, openpose_model):
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

def prepare_control_image(image, device, dtype):
    # preprocess step
    images = [image]
    # control image needs to be turned RGB
    image = [convert_to_rgb(i) for i in images]
    # image resize
    images = [resize(image) for image in images]
    # PIL to numpy
    images = [np.array(image).astype(np.float32) / 255.0 for image in images]
    images = np.stack(images, axis=0)
    # numpy to tensor
    if images.ndim == 3:
        images = images[..., None]
    images = torch.from_numpy(images.transpose(0, 3, 1, 2)).to(dtype=torch.float32)
    # preprocess step end

    images = images.repeat_interleave(1, dim=0)
    images = images.to(device=device, dtype=dtype)
    # 3rd thing to check
    images = torch.cat([images] * 2)
    return images


# SAG method 1: Gaussian blur
def gaussian_blur_2d(img, kernel_size, sigma):
    ksize_half = (kernel_size - 1) * 0.5

    x = torch.linspace(-ksize_half, ksize_half, steps=kernel_size)

    pdf = torch.exp(-0.5 * (x / sigma).pow(2))

    x_kernel = pdf / pdf.sum()
    x_kernel = x_kernel.to(device=img.device, dtype=img.dtype)

    kernel2d = torch.mm(x_kernel[:, None], x_kernel[None, :])
    kernel2d = kernel2d.expand(img.shape[-3], 1, kernel2d.shape[0], kernel2d.shape[1])

    padding = [kernel_size // 2, kernel_size // 2, kernel_size // 2, kernel_size // 2]

    img = F.pad(img, padding, mode="reflect")
    img = F.conv2d(img, kernel2d, groups=img.shape[-3])
    return img




