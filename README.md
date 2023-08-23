# Stable Diffusion with Self-Guided Attention and ControlNet

## Overview

This is Stable Diffusion built on pre-trained Stable Diffusion v1.5 weights and uses Self-Attention Guidelines (SAG) to enhance generated image's stability. It also uses ControlNet, a neural network model, to support additional input to control the image generation.

This model is built on Hugging Face modules.

[Deploy and Run](https://github.com/Shashank-Holla/diffusion-controlnet-sag/edit/main/README.md#Deploy and Run)
## Deploy and Run

### Run model

```
!python main.py --prompt "Margot Robbie as wonderwoman in style" --seed 3 --batch_size 1 --controlNet_image ./control_images/controlimage_1.jpg --controlNet_type canny --style_flag T --sag_scale 0.75 --controlnet_cond_scale 1.0
```

## Results

| Prompt   | Generated Image |
|----------|-----------------|
| Prompt:  |                 |
| Prompt:  |                 |
