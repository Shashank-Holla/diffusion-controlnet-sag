import cv2
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import torch
import argparse

from controlnet_aux import OpenposeDetector
from diffusers import ControlNetModel
from diffusers.utils import load_image

from utils.image_utils import canny_edge_detection, openpose_detection
from generate import StableDiffusionWithSAGAndControlNet

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--prompt', type=str, default="./", help='prompt to guide image generation')
    parser.add_argument('--seed', type=int, default=3, help='Seed to make generation deterministic')
    parser.add_argument('--batch_size', type=int, default=1, help='size of each batch to run')
    parser.add_argument('--controlNet_image', type=str, default="", help='path to controlNet image to guide generation')
    parser.add_argument('--controlNet_type', type=str, default=132720, help='ControlNet model type to guide')
    parser.add_argument('--style_flag', type=int, default=0, help='flag to use style')
    parser.add_argument('--sag_scale', type=float, default=7.5, help='SAG scal')
    opt = parser.parse_args()

    ##### Setting parameters #####
    prompt = opt.prompt
    seed = opt.seed
    batch_size = opt.batch_size
    controlNet_image = opt.controlNet_image
    controlNet_type = opt.controlNet_type
    style_flag = opt.style_flag
    sag_scale = opt.sag_scale

    # prompt settings
    height = 512                        # default height of Stable Diffusion
    width = 512                         # default width of Stable Diffusion
    num_inference_steps = 1            # Number of denoising steps, using less for controlnet
    guidance_scale = 7.5                # Scale for classifier-free guidance
    generator = torch.manual_seed(seed)   # Seed generator to create the inital latent noise
    batch_size = 1

    # load controlnet models
    openpose_controlnet = OpenposeDetector.from_pretrained("lllyasviel/ControlNet", torch_dtype=torch.float16)
    canny_controlnet = ControlNetModel.from_pretrained("lllyasviel/sd-controlnet-canny", torch_dtype=torch.float16)

    if controlNet_type == "canny":
        control_img = canny_edge_detection(controlNet_image)
        controlnet = canny_controlnet
    elif controlNet_type == "openpose":
        control_img = openpose_detection(controlNet_image, openpose_controlnet)
    else:
        print("Invalid ControlNet model requested.")

    # setup torch device. Need gpu for cpu_offload in sequence.
    torch_device = "cuda" if torch.cuda.is_available() else "cpu"

    # instantiate and generate image
    stablediffusion_model = StableDiffusionWithSAGAndControlNet(controlnet, torch_device)

    output_image = stablediffusion_model.generateDiffusion(prompt, 
                                                           control_img, 
                                                           generator, 
                                                           batch_size, 
                                                           num_inference_steps=20)
    
    output_image.save("./diffused_image.png")
    print("Diffused image saved!")

