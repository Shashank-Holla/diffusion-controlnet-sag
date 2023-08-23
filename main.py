import cv2
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import torch
import argparse
import os
import time

from controlnet_aux import OpenposeDetector
from diffusers import ControlNetModel
from diffusers.utils import load_image

from utils.image_utils import canny_edge_detection, openpose_detection
from generate import StableDiffusionWithSAGAndControlNet

def generate():
    parser = argparse.ArgumentParser()
    parser.add_argument('--prompt', type=ascii, required=True, default="", help='prompt to guide image generation')
    parser.add_argument('--seed', type=int, default=3, help='Seed to make generation deterministic')
    parser.add_argument('--batch_size', type=int, default=1, help='Size of each batch to run')
    parser.add_argument('--controlNet_image', type=str, default="", help='Path to controlNet image to guide generation')
    parser.add_argument('--controlNet_type', type=str, choices=['canny', 'openpose'], help='ControlNet model type to guide')
    parser.add_argument('--style_flag', type=str, choices=['F', 'T'], default="T", help='Flag to use style')
    parser.add_argument('--sag_scale', type=float, default=7.5, help='SAG scal')
    parser.add_argument('--controlnet_cond_scale', type=float, default=1.0, help='Controlnet conditioning scale')
    opt = parser.parse_args()

    ##### Setting parameters #####
    prompt = opt.prompt
    seed = opt.seed
    batch_size = opt.batch_size
    controlNet_image = opt.controlNet_image
    controlNet_type = opt.controlNet_type
    style_flag = opt.style_flag
    sag_scale = opt.sag_scale
    controlnet_cond_scale = opt.controlnet_cond_scale

    # prompt settings
    num_inference_steps = 20            # Number of denoising steps, using less for controlnet
    guidance_scale = 7.5                # Scale for classifier-free guidance
    generator = torch.manual_seed(seed)   # Seed generator to create the inital latent noise
    batch_size = 1
    

    # load controlnet models
    if controlNet_type == "canny":
        control_img = canny_edge_detection(controlNet_image)
        controlnet = ControlNetModel.from_pretrained("lllyasviel/sd-controlnet-canny", torch_dtype=torch.float16)
    elif controlNet_type == "openpose":
        openpose = OpenposeDetector.from_pretrained("lllyasviel/ControlNet")
        control_img = openpose_detection(controlNet_image, openpose)
        controlnet = ControlNetModel.from_pretrained("fusing/stable-diffusion-v1-5-controlnet-openpose", torch_dtype=torch.float16)
    
    if style_flag == "T":
        style_flag = True
    else:
        style_flag = False


    # setup torch device. Need gpu for cpu_offload in sequence.
    torch_device = "cuda" if torch.cuda.is_available() else "cpu"

    # instantiate and generate image
    stablediffusion_model = StableDiffusionWithSAGAndControlNet(torch_device, controlnet, style_flag)

    output_image = stablediffusion_model.generateDiffusion(prompt, 
                                                           control_img, 
                                                           generator, 
                                                           batch_size, 
                                                           num_inference_steps=num_inference_steps,
                                                           controlnet_conditioning_scale=controlnet_cond_scale,
                                                           guidance_scale=guidance_scale,
                                                           sag_scale=sag_scale    
                                                           )
    

    if not os.path.exists("results"):
        os.mkdir("results")
    result_filename = time.strftime("%Y%m%d-%H%M%S") + ".png"

    output_image.save(os.path.join("results", result_filename))
    print(f"Diffused image saved at {os.path.join('results', result_filename)}")

if __name__ == '__main__':
    generate()
    

