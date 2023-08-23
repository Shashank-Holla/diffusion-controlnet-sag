import torch
import torch.nn.functional as F
import numpy as np
from PIL import Image
import os

# offload cpu
import accelerate
from diffusers import AutoencoderKL, UniPCMultistepScheduler, UNet2DConditionModel, ControlNetModel
from transformers import CLIPTextModel, CLIPTokenizer
from tqdm.auto import tqdm


from utils import image_utils



# processes and stores attention probabilities
class CrossAttnStoreProcessor:
    def __init__(self):
        self.attention_probs = None

    def __call__(
        self,
        attn,
        hidden_states,
        encoder_hidden_states=None,
        attention_mask=None,
    ):
        batch_size, sequence_length, _ = hidden_states.shape
        attention_mask = attn.prepare_attention_mask(attention_mask, sequence_length, batch_size)
        query = attn.to_q(hidden_states)

        if encoder_hidden_states is None:
            encoder_hidden_states = hidden_states
        elif attn.norm_cross:
            encoder_hidden_states = attn.norm_encoder_hidden_states(encoder_hidden_states)

        key = attn.to_k(encoder_hidden_states)
        value = attn.to_v(encoder_hidden_states)

        query = attn.head_to_batch_dim(query)
        key = attn.head_to_batch_dim(key)
        value = attn.head_to_batch_dim(value)

        self.attention_probs = attn.get_attention_scores(query, key, attention_mask)
        hidden_states = torch.bmm(self.attention_probs, value)
        hidden_states = attn.batch_to_head_dim(hidden_states)

        # linear proj
        hidden_states = attn.to_out[0](hidden_states)
        # dropout
        hidden_states = attn.to_out[1](hidden_states)

        return hidden_states



class StableDiffusionWithSAGAndControlNet:
    def __init__(self, torch_device, controlnet=None):
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
        self.controlnet_guidance = False if self.controlnet is None else True
        self.torch_device = torch_device

        # load style weights for textual inversion
        self.load_textual_inversion_style_model(os.path.join("style_model", "learned_embeds.bin"))



    def load_textual_inversion_style_model(self, style_file_path):
        """
        Updates tokenizer and text encoder with the new token and embeddings from the trained style weights
        """
        # new tokenid-embedding pair from trained style weights. This is updated into the text encoder.
        tokenid_embedding_pairs = []
        # load textual inversion model. <text_key>: weights
        state_dict = torch.load(style_file_path, map_location="cpu")
        style_token, style_embed = next(iter(state_dict.items()))
        style_embed = style_embed.to(dtype=self.text_encoder.dtype, device=self.text_encoder.device)
        vocab = self.tokenizer.get_vocab()
        if style_token not in vocab:
            # put into list
            style_tokens, style_embeds = [style_token], [style_embed]

            # update tokenizer
            self.tokenizer.add_tokens(style_tokens)
            style_token_ids = self.tokenizer.convert_tokens_to_ids(style_tokens)
            tokenid_embedding_pairs += zip(style_token_ids, style_embeds)

            # update text encoder
            self.text_encoder.resize_token_embeddings(len(self.tokenizer))
            for token_id, embed in tokenid_embedding_pairs:
                self.text_encoder.get_input_embeddings().weight.data[token_id] = embed
        else:
            print("Style token already present")



    # SAG related method - 1
    def pred_X0(self, sample, model_output, timestep):
        alpha_prod_t = self.scheduler.alphas_cumprod[timestep]
        beta_prod_t = 1 - alpha_prod_t
        pred_original_sample = (sample - beta_prod_t ** (0.5) * model_output) / alpha_prod_t ** (0.5)
        return pred_original_sample
    

    # SAG related method - 2
    def sag_masking(self, original_latents, attn_map, map_size, t, eps):
        # Same masking process as in SAG paper: https://arxiv.org/pdf/2210.00939.pdf
        bh, hw1, hw2 = attn_map.shape
        b, latent_channel, latent_h, latent_w = original_latents.shape
        h = self.unet.config.attention_head_dim
        if isinstance(h, list):
            h = h[-1]

        # Produce attention mask
        attn_map = attn_map.reshape(b, h, hw1, hw2)
        attn_mask = attn_map.mean(1, keepdim=False).sum(1, keepdim=False) > 1.0
        attn_mask = (
            attn_mask.reshape(b, map_size[0], map_size[1])
            .unsqueeze(1)
            .repeat(1, latent_channel, 1, 1)
            .type(attn_map.dtype)
        )
        attn_mask = F.interpolate(attn_mask, (latent_h, latent_w))

        # Blur according to the self-attention mask
        degraded_latents = image_utils.gaussian_blur_2d(original_latents, kernel_size=9, sigma=1.0)
        degraded_latents = degraded_latents * attn_mask + original_latents * (1 - attn_mask)

        # Noise it again to match the noise level
        degraded_latents = self.scheduler.add_noise(degraded_latents, noise=eps, timesteps=t)

        return degraded_latents
    

    
    def cpu_offload_model(self):
        hook = None
        for cpu_offloaded_model in [self.text_encoder, self.unet, self.vae]:
            _, hook = accelerate.cpu_offload_with_hook(cpu_offloaded_model, self.torch_device, prev_module_hook=hook)

        # check if controlnet is available,
        # control net hook has be manually offloaded as it alternates with unet
        if self.controlnet_guidance:            
            accelerate.cpu_offload_with_hook(self.controlnet, self.torch_device)

        # final offload vae 
        final_offload_hook = hook
        return final_offload_hook
    

    
    def encode_prompt(self, prompt, batch_size):
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
        return text_embeddings



    def prepare_latents(self, batch_size, num_channels_latents, height, width, embedding_type, device, generator):
        shape = (batch_size, num_channels_latents, height // 8, width // 8)
        latents = torch.randn(shape, generator=generator, dtype=embedding_type)
        latents = latents.to(device)
        latents = latents * self.scheduler.init_noise_sigma
        return latents

        
    def generateDiffusion(self, 
                          prompt, 
                          controlnet_image, 
                          generator, 
                          batch_size, 
                          num_inference_steps=20, 
                          controlnet_conditioning_scale=1.0,
                          guidance_scale=7.5,
                          sag_scale=0.75):
        
        # format align
        control_guidance_start = [0.0]
        control_guidance_end = [1.0]
        # By default, using unet's config. If control image is provided, it's height and width will be used. 
        height = 512
        width = 512
        

        final_offload_hook = self.cpu_offload_model()

        # 1. encode prompt
        text_embeddings = self.encode_prompt(prompt, batch_size)

        # 2. prepare image
        # convert to resized, normalized tensor
        if self.controlnet_guidance:
            cond_image = image_utils.prepare_control_image(controlnet_image, device=self.torch_device, dtype=self.controlnet.dtype)         
            height, width = cond_image.shape[-2:]

        # 3. prepare timesteps
        self.scheduler.set_timesteps(num_inference_steps, device=self.torch_device)
        timesteps = self.scheduler.timesteps

        # 4. prepare latents
        latents = self.prepare_latents(batch_size, 
                                       self.unet.config.in_channels, 
                                       height, 
                                       width, 
                                       text_embeddings.dtype, 
                                       self.torch_device, 
                                       generator
                                       )

        # to decide controlnet
        if self.controlnet_guidance:
            controlnet_keep = []
            for i in range(len(timesteps)):
                keeps = [
                    1.0 - float(i / len(timesteps) < s or (i + 1) / len(timesteps) > e)
                    for s, e in zip(control_guidance_start, control_guidance_end)
                ]
                controlnet_keep.append(keeps[0] if isinstance(self.controlnet, ControlNetModel) else keeps)

        # 5. denoising loop
        store_processor = CrossAttnStoreProcessor()
        self.unet.mid_block.attentions[0].transformer_blocks[0].attn1.processor = store_processor
        map_size = None

        def get_map_size(module, input, output):
            nonlocal map_size
            map_size = output[0].shape[-2:]

        with self.unet.mid_block.attentions[0].register_forward_hook(get_map_size):
            for i, t in tqdm(enumerate(timesteps), total=len(timesteps)):
                # expand the latents to avoid doing two forward passes.
                latent_model_input = torch.cat([latents] * 2)
                # Scale the latents (preconditioning):
                latent_model_input = self.scheduler.scale_model_input(latent_model_input, t)

                # controlnet guidance
                down_block_res_samples = None
                mid_block_res_sample = None

                if self.controlnet_guidance:
                    control_model_input = latent_model_input
                    controlnet_prompt_embeds = text_embeddings
                    if isinstance(controlnet_keep[i], list):
                        cond_scale = [c * s for c, s in zip(controlnet_conditioning_scale, controlnet_keep[i])]
                    else:
                        controlnet_cond_scale = controlnet_conditioning_scale
                        if isinstance(controlnet_cond_scale, list):
                            controlnet_cond_scale = controlnet_cond_scale[0]
                        cond_scale = controlnet_cond_scale * controlnet_keep[i]

                    with torch.no_grad():
                        down_block_res_samples, mid_block_res_sample = self.controlnet(
                                control_model_input,
                                t,
                                encoder_hidden_states=controlnet_prompt_embeds,
                                controlnet_cond=cond_image,
                                conditioning_scale=cond_scale,
                                return_dict=False,
                            )
                
                # predict the noise residual
                with torch.no_grad():
                    noise_pred = self.unet(latent_model_input,t,
                        encoder_hidden_states=text_embeddings,
                        down_block_additional_residuals=down_block_res_samples,
                        mid_block_additional_residual=mid_block_res_sample,
                        return_dict=False
                        )[0]
                
                # perform guidance
                noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
                noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_text - noise_pred_uncond)

                ## Self-attention guidance from stored attention maps
                pred_x0 = self.pred_X0(latents, noise_pred_uncond, t)

                # get the stored attention maps
                uncond_attn, cond_attn = store_processor.attention_probs.chunk(2)
                # self-attention-based degrading of latents
                degraded_latents = self.sag_masking(pred_x0, uncond_attn, map_size, t, noise_pred_uncond)
                uncond_emb, _ = text_embeddings.chunk(2)
                # forward and give guidance
                with torch.no_grad():
                    degraded_pred = self.unet(degraded_latents, t, encoder_hidden_states=uncond_emb).sample
                noise_pred += sag_scale * (noise_pred_uncond - degraded_pred)

                # compute the previous noisy sample x_t -> x_t-1
                latents = self.scheduler.step(noise_pred, t, latents, return_dict=False)[0]
        
        # End of denoising loop

        # gpu offloading
        self.unet.to("cpu")
        if self.controlnet_guidance:
            self.controlnet.to("cpu")
        torch.cuda.empty_cache()


        # scale and decode the image latents with vae
        latents = latents / 0.18215
        with torch.no_grad():
            image = self.vae.decode(latents, return_dict=False)[0]
        
        # offloading VAE from GPU
        final_offload_hook.offload()

        # post process final generated image
        image = (image / 2 + 0.5).clamp(0, 1)
        image = image.detach().cpu().permute(0, 2, 3, 1).float().numpy()
        images = (image * 255).round().astype("uint8")
        pil_images = [Image.fromarray(image) for image in images]
        final_offload_hook.offload()

        return pil_images[0]
