# Stable Diffusion with Self-Guided Attention and ControlNet

### Run model

```
!python main.py --prompt "Margot Robbie as wonderwoman in style" --seed 3 --batch_size 1 --controlNet_image ./control_images/controlimage_1.jpg --controlNet_type canny --style_flag T --sag_scale 7.5 --controlnet_cond_scale 1.0
```
