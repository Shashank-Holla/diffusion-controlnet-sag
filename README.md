# Stable Diffusion with Self-Guided Attention and ControlNet

## 1. Overview

This is Stable Diffusion built on pre-trained Stable Diffusion v1.5 weights and uses Self-Attention Guidelines (SAG) to enhance generated image's stability. It also uses ControlNet, a neural network model, to support additional input to control the image generation. Additionally, the model can add artistic features to the generated image by utilizing trained style weights.

This model is built on Hugging Face modules. It utilizes Tokenizer, Text Encoder, Variational Auto Encoder and Unet model from it.

[Deploy and Run](https://github.com/Shashank-Holla/diffusion-controlnet-sag/edit/main/README.md#Deploy and Run)

## 2. Features
### 2.1 Self Attention Guidelines
Self attention guidelines helps stable diffusion to improve generated image. It uses the intermediate self-attention maps to adversially blur and guides the model. Parameter ```sag_scale```  controls the SAG influence on the model.

### 2.2 ControlNet support

ControlNet conditions the diffusion model to learn specific user input conditions (like edges, depth). This helps it generate images which are related to the desired spatial context. ```canny``` and ```openpose``` controlnets are supported in this application. Conditional input image such as edge map, keypoints are also provided along with the controlnet model for inference.
```controlnet_cond_scale``` parameter controls the scale to which the generated image are faithful to the conditional image.

### 2.3 Style

The application is trained on a novel art via Textual Inversion. In our case, images stylistically related to pop-art are trained in order to associate it with ```<pop-art>``` word within the text encoder embedding. Training images and the weights for style training are available here [<pop-art>](https://huggingface.co/sd-concepts-library/pop-art)

To use the style, add <pop-art> in the prompt. While running the model, enable ```style_flag``` to use the style.



## Deploy and Run

Stable Diffusion can be run in the following two ways-

### Clone Repository and execute

Clone repository and change directory-
```
git clone https://github.com/Shashank-Holla/diffusion-controlnet-sag.git

cd diffusion-controlnet-sag/
```

Install dependencies-

```
pip install -r requirements.txt
```

Run model 
```
!python main.py --prompt "Margot Robbie as wonderwoman in style" --seed 3 --batch_size 1 --controlNet_image ./control_images/controlimage_1.jpg --controlNet_type canny --style_flag T --sag_scale 0.75 --controlnet_cond_scale 1.0
```

### Install CLI application and run

This repository is also available as CLI application. Build files are available in ```dist``` folder in this repository.

Install distribution-

```
!pip install dist/diffusion-0.0.7-py3-none-any.whl
```

Run application ```generate```. Provide input as prompted-

```
/usr/local/bin/generate
```

## Results

Here are few run results-
```
Prompt: "Margot Robbie as wonderwoman in polychrome, good anatomy, best and quality, extremely detailed"
SAG_scale: 0.25 MR_p6
```

```
Prompt: "Margot Robbie as wonderwoman in polychrome, good anatomy, best and quality, extremely detailed"
SAG_scale: 1.0 MR_p7
```

```
Prompt: "Margot Robbie as wonderwoman in <pop-art> style"
SAG_scale: 0.5 MR_p8
```



| Prompt                                                                                                                               | Generated Image             |
|--------------------------------------------------------------------------------------------------------------------------------------|-----------------------------|
| ``` Prompt: "Margot Robbie as wonderwoman in  polychrome, good anatomy,  best and quality, extremely detailed"  SAG_scale: 0.25  ``` | ![img_6](results/MR_p6.png) |
| ``` Prompt: "Margot Robbie as wonderwoman in  polychrome, good anatomy,  best and quality, extremely detailed" SAG_scale: 1.0 ```    | ![img_7](results/MR_p7.png) |
