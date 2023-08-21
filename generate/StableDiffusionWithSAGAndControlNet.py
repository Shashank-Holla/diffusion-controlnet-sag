import torch
import numpy as np
from PIL import Image

# offload cpu
import accelerate
from diffusers import AutoencoderKL, UniPCMultistepScheduler, UNet2DConditionModel
from transformers import CLIPTextModel, CLIPTokenizer
from tqdm.auto import tqdm


from utils import image_utils

class StableDiffusionWithSAGAndControlNet:
    def __init__(self, controlnet, torch_device, guidance_scale=7.5):
        # Autoencoder- latents into image space
        self.vae = AutoencoderKL.from_pretrained("runwayml/stable-diffusion-v1-5", 
                                            revision="fp16", 
                                            torch_dtype=torch.float16, 
                                            subfolder="vae")

        # Tokenizer and Text encoder to tokenize and encode the prompt
        self.tokenizer = CLIPTokenizer.from_pretrained("runwayml/stable-diffusion-v1-5", 
                                                revision="fp16", 
                                                torch_dtype=torch.float16, 
                                                subfolder="tokenizer")
        
        self.text_encoder = CLIPTextModel.from_pretrained("runwayml/stable-diffusion-v1-5", 
                                                    revision="fp16", 
                                                    torch_dtype=torch.float16, 
                                                    subfolder="text_encoder")

        # UNet- generate latents
        self.unet = UNet2DConditionModel.from_pretrained("runwayml/stable-diffusion-v1-5", 
                                                    revision="fp16", 
                                                    torch_dtype=torch.float16, 
                                                    subfolder="unet")

        # The noise scheduler
        self.scheduler = UniPCMultistepScheduler.from_pretrained("runwayml/stable-diffusion-v1-5", 
                                                            subfolder="scheduler")
        
        # controlnet model- canny or openpose
        self.controlnet = controlnet

        self.torch_device = torch_device

        self.guidance_scale = guidance_scale

    
    def cpu_offload_model(self):
        device = device
        hook = None
        for cpu_offloaded_model in [self.text_encoder, self.unet, self.vae]:
            _, hook = accelerate.cpu_offload_with_hook(cpu_offloaded_model, device, prev_module_hook=hook)

        # control net hook has be manually offloaded as it alternates with unet
        accelerate.cpu_offload_with_hook(self.controlnet, device)

        # final offload vae 
        final_offload_hook = hook
        return final_offload_hook

    def prepare_latents(self, batch_size, num_channels_latents, height, width, embedding_type, device, generator):
        shape = (batch_size, num_channels_latents, height // 8, width // 8)
        latents = torch.randn(shape, generator=generator, dtype=embedding_type)
        latents = latents.to(device)
        latents = latents * self.scheduler.init_noise_sigma
        return latents

    def prepare_image(self, image):
        images = [image]
        # control image needs to be turned RGB
        image = [image_utils.convert_to_rgb(i) for i in images]
        # image resize
        images = [image_utils.resize(image) for image in images]
        # PIL to numpy
        images = [np.array(image).astype(np.float32) / 255.0 for image in images]
        images = np.stack(images, axis=0)
        # numpy to tensor
        if images.ndim == 3:
            images = images[..., None]
        images = torch.from_numpy(images.transpose(0, 3, 1, 2))
        # return normalized image [-1,1]
        images = 2.0 * images - 1.0
        return images
        
    def generateDiffusion(self, prompt, controlnet_image, generator, batch_size, num_inference_steps=20):
        # format align
        control_guidance_start = [0.0]
        control_guidance_end = [1.0]
        controlnet_conditioning_scale = 1.0

        final_offload_hook = self.cpu_offload_model()

        # 1. encode prompt
        text_input = self.tokenizer(prompt, 
                                    padding="max_length", 
                                    max_length=self.tokenizer.model_max_length, 
                                    truncation=True, return_tensors="pt")
        text_input_ids = text_input.input_ids
        with torch.no_grad():
            text_embeddings = self.text_encoder(text_input_ids.to(self.torch_device))[0]
        # to maintain same dtype across inputs to avoid data type mismatch in computation
        embedding_type = self.text_encoder.dtype
        text_embeddings = text_embeddings.to(dtype=embedding_type, device=self.torch_device)

        max_length = text_input.input_ids.shape[-1]
        uncond_input = self.tokenizer([""] * batch_size, 
                                      padding="max_length", 
                                      max_length=max_length, 
                                      truncation=True, return_tensors="pt")
        with torch.no_grad():
            uncond_embeddings = self.text_encoder(uncond_input.input_ids.to(self.torch_device))[0]
        uncond_embeddings = uncond_embeddings.to(dtype=embedding_type, device=self.torch_device)

        text_embeddings = torch.cat([uncond_embeddings, text_embeddings])

        # 2. prepare image
        # convert to resized, normalized tensor
        image = self.prepare_image(controlnet_image).to(dtype=self.controlnet.dtype, device=self.torch_device)
        height, width = image.shape[-2:]

        # 3. prepare timesteps
        self.scheduler.set_timesteps(num_inference_steps, device=self.torch_device)
        timesteps = self.scheduler.timesteps

        # 4. prepare latents
        latents = self.prepare_latents(batch_size, 
                                       self.unet.config.in_channels, 
                                       height, 
                                       width, 
                                       embedding_type, 
                                       self.torch_device, 
                                       generator)

        # new idea
        controlnet_keep = []
        for i in range(len(timesteps)):
            keeps = [
                1.0 - float(i / len(timesteps) < s or (i + 1) / len(timesteps) > e)
                for s, e in zip(control_guidance_start, control_guidance_end)
            ]
            controlnet_keep.append(keeps[0])
        # new idea

        # 5. denoising loop
        for i, t in tqdm(enumerate(timesteps), total=len(timesteps)):
            # expand the latents to avoid doing two forward passes.
            latent_model_input = torch.cat([latents] * 2)
            # Scale the latents (preconditioning):
            latent_model_input = self.scheduler.scale_model_input(latent_model_input, t)

            # controlnet inference
            control_model_input = latent_model_input
            controlnet_prompt_embeds = text_embeddings
            # new idea
            
            if isinstance(controlnet_keep[i], list):
                cond_scale = [c * s for c, s in zip(controlnet_conditioning_scale, controlnet_keep[i])]
            else:
                controlnet_cond_scale = controlnet_conditioning_scale
                if isinstance(controlnet_cond_scale, list):
                    controlnet_cond_scale = controlnet_cond_scale[0]
                cond_scale = controlnet_cond_scale * controlnet_keep[i]
            # new idea

            with torch.no_grad():
                down_block_res_samples, mid_block_res_sample = self.controlnet(
                        control_model_input,
                        t,
                        encoder_hidden_states=controlnet_prompt_embeds,
                        controlnet_cond=image,
                        conditioning_scale=cond_scale,
                        return_dict=False,
                    )

            # predict noise residual
            with torch.no_grad():
                noise_pred = self.unet(latent_model_input,t,
                        encoder_hidden_states=text_embeddings,
                        down_block_additional_residuals=down_block_res_samples,
                        mid_block_additional_residual=mid_block_res_sample,
                        return_dict=False
                    )[0]

            # perform guidance
            noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
            noise_pred = noise_pred_uncond + self.guidance_scale * (noise_pred_text - noise_pred_uncond)

            # compute the previous noisy sample x_t -> x_t-1
            latents = self.scheduler.step(noise_pred, t, latents, return_dict=False)[0]


        # gpu offloading
        self.unet.to("cpu")
        self.controlnet.to("cpu")
        torch.cuda.empty_cache()


        # scale and decode the image latents with vae
        latents = latents / 0.18215
        with torch.no_grad():
            image = self.vae.decode(latents, return_dict=False)[0]

        # Display
        image = (image / 2 + 0.5).clamp(0, 1)
        image = image.detach().cpu().permute(0, 2, 3, 1).float().numpy()
        images = (image * 255).round().astype("uint8")
        pil_images = [Image.fromarray(image) for image in images]
        final_offload_hook.offload()

        return pil_images[0]
