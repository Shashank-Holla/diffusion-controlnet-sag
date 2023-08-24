import cv2
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import torch
import argparse
import os
import time
import random

from controlnet_aux import OpenposeDetector
from diffusers import ControlNetModel

# from utils import canny_edge_detection, openpose_detection
from diffusion.utils import image_utils
from diffusion.generate import StableDiffusionWithSAGAndControlNet

def generate(prompt, seed, batch_size, controlNet_image, controlNet_type, style_flag, sag_scale, controlnet_cond_scale, style_file_path):
    # prompt settings
    num_inference_steps = 20            # Number of denoising steps, using less for controlnet
    guidance_scale = 7.5                # Scale for classifier-free guidance
    generator = torch.manual_seed(seed)   # Seed generator to create the inital latent noise
    batch_size = 1
    

    # load controlnet models
    if controlNet_type == "canny":
        control_img = image_utils.canny_edge_detection(controlNet_image)
        controlnet = ControlNetModel.from_pretrained("lllyasviel/sd-controlnet-canny", torch_dtype=torch.float16)
    elif controlNet_type == "openpose":
        openpose = OpenposeDetector.from_pretrained("lllyasviel/ControlNet")
        control_img = image_utils.openpose_detection(controlNet_image, openpose)
        controlnet = ControlNetModel.from_pretrained("fusing/stable-diffusion-v1-5-controlnet-openpose", torch_dtype=torch.float16)
    else:
        control_img = None
        controlnet = None


    # setup torch device. Need gpu for cpu_offload in sequence.
    torch_device = "cuda" if torch.cuda.is_available() else "cpu"

    # instantiate and generate image
    print("\n\n")
    print("Loading models")
    print("\n\n")
    stablediffusion_model = StableDiffusionWithSAGAndControlNet(torch_device, controlnet, style_file_path)

    print("\n\n")
    print("Inferencing...")
    print("\n\n")
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
    print("\n\n")
    print(f"Diffused image saved at {os.path.join('results', result_filename)}")


def take_input():
    valid_image_type = ["jpeg", "jpg", "png"]
    print("\n\n")

    # receive user input and validate
    prompt                  = input("Prompt (str)\t: ")
    batch_size              = input("Batch Size (int)\t: ")
    batch_size = 1

    try:
        seed = int(input("Seed (int)\t: "))
    except ValueError:
        seed = random.randint(0,100)

    controlNet_type = input("ControlNet type (str:Canny/Openpose/empty)\t: ")
    if controlNet_type.lower() not in ["canny", "openpose"]:
        controlNet_type = None

    controlNet_image = None
    if controlNet_type is not None:
        while True:
            controlNet_image = input("Control Image filepath (str)\t: ")
            if os.path.exists(controlNet_image) and controlNet_image.lower().split(".")[-1] in valid_image_type:
                break
    
    style_flag = input("Style Flag (str: T/F)\t: ")
    if style_flag == "T":
        prompt += " in <pop-art> style"
    
    try:
        sag_scale = float(input("SAG scale (float)\t: "))
    except ValueError:
        sag_scale = 0.75

    try:
        controlnet_cond_scale = float(input("ControlNet Condition scale (float)\t: "))
    except ValueError:
        controlnet_cond_scale = 1.0
    
    while True:
        style_file_path = input("Style model filepath (str)\t: ")
        if os.path.exists(style_file_path) and style_file_path.lower().split(".")[-1] == "bin":
            break

    return [prompt, seed, batch_size, controlNet_image, controlNet_type, style_flag, sag_scale, controlnet_cond_scale, style_file_path]

    
def main():
    user_inp = take_input()
    if user_inp[0] is None:
        return
    else:
        prompt, seed, batch_size, controlNet_image, controlNet_type, style_flag, sag_scale, controlnet_cond_scale, style_file_path = user_inp

    generate(prompt, seed, batch_size, controlNet_image, controlNet_type, style_flag, sag_scale, controlnet_cond_scale, style_file_path)



if __name__ == '__main__':
    main()
    

