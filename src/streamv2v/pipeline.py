import glob
import os
import time
from typing import List, Optional, Union, Any, Dict, Tuple, Literal
from collections import deque

import numpy as np
import PIL.Image
import torch
import torch.nn.functional as F
from torchvision.models.optical_flow import raft_small

from diffusers import FlowMatchEulerDiscreteScheduler, StableDiffusion3Pipeline as StableDiffusionPipeline
from diffusers.image_processor import VaeImageProcessor
from diffusers.pipelines.stable_diffusion.pipeline_stable_diffusion_img2img import retrieve_latents

from .image_utils import postprocess_image, forward_backward_consistency_check
from .models.utils import get_nn_latent
from .image_filter import SimilarImageFilter

class StreamV2V:
    def __init__(
        self,
        pipe: StableDiffusionPipeline,
        t_index_list: List[int],
        torch_dtype: torch.dtype = torch.float16,
        width: int = 512,
        height: int = 512,
        do_add_noise: bool = True,
        use_denoising_batch: bool = True,
        frame_buffer_size: int = 1,
        cfg_type: Literal["none", "full", "self", "initialize"] = "self",
    ) -> None:
        self.device = pipe.device
        if self.device.type == "mps":
            self.torchbackend = torch.mps
        elif self.device.type == "cuda":
            if torch.cuda.is_available():
                self.torchbackend = torch.cuda
            else:
                raise RuntimeError("CUDA is not available. Please check your PyTorch installation.")
        
        self.dtype = torch_dtype
        self.generator = None
        self.height = height
        self.width = width
        self.latent_height = int(height // pipe.vae_scale_factor)
        self.latent_width = int(width // pipe.vae_scale_factor)
        self.frame_bff_size = frame_buffer_size
        self.denoising_steps_num = len(t_index_list)
        self.cfg_type = cfg_type
        if use_denoising_batch:
            self.batch_size = self.denoising_steps_num * frame_buffer_size
            if self.cfg_type == "initialize":
                self.trt_unet_batch_size = (self.denoising_steps_num + 1) * self.frame_bff_size
            elif self.cfg_type == "full":
                self.trt_unet_batch_size = 2 * self.denoising_steps_num * self.frame_bff_size
            else:
                self.trt_unet_batch_size = self.denoising_steps_num * frame_buffer_size
        else:
            self.trt_unet_batch_size = self.frame_bff_size
            self.batch_size = frame_buffer_size
        self.t_list = t_index_list
        self.do_add_noise = do_add_noise
        self.use_denoising_batch = use_denoising_batch
        self.similar_image_filter = False
        self.similar_filter = SimilarImageFilter()
        self.prev_image_tensor = None
        self.prev_x_t_latent = None
        self.prev_image_result = None
        self.pipe = pipe
        self.image_processor = VaeImageProcessor(pipe.vae_scale_factor)
        # self.scheduler = FlowMatchEulerDiscreteScheduler.from_config(self.pipe.scheduler.config)
        # new FlowMatch scheduler (for SD-3.5)
        # print(f"Using FlowMatchEulerDiscreteScheduler for SD-3.5 {self.pipe.scheduler.config}")
        # Updated for SD 3.5
        self.scheduler = FlowMatchEulerDiscreteScheduler.from_config(
            self.pipe.scheduler.config
        )
        self.text_encoder = pipe.text_encoder
        # Use the SD3.5 model components
        self.transformer = pipe  # Updated for SD3.5 which uses the entire pipeline
        self.vae = pipe.vae
        self.flow_model = raft_small(pretrained=True, progress=False).to(device=pipe.device).eval()
        self.cached_x_t_latent = deque(maxlen=4)
        self.inference_time_ema = 0
        # Initialize pooled text embeddings attributes
        self.pooled_prompt_embeds: Optional[torch.Tensor] = None
        self.null_pooled_prompt_embeds: Optional[torch.Tensor] = None

        # Initialize alpha and beta tensors for noise scheduling
        self.alpha_prod_t_sqrt = torch.ones(len(t_index_list), device=self.device)
        self.beta_prod_t_sqrt = torch.zeros(len(t_index_list), device=self.device)
        for i, t in enumerate(t_index_list):
            self.alpha_prod_t_sqrt[i] = torch.sqrt(1 - (self.scheduler.sigmas[t] ** 2))
            self.beta_prod_t_sqrt[i] = torch.sqrt(self.scheduler.sigmas[t] ** 2)


    def load_lcm_lora(
        self,
        pretrained_model_name_or_path_or_dict: Union[str, Dict[str, torch.Tensor]] = "latent-consistency/lcm-lora-sdv1-5",
        adapter_name: Optional[Any] = "lcm",
        **kwargs,
    ) -> None:
        self.pipe.load_lora_weights(pretrained_model_name_or_path_or_dict, adapter_name, **kwargs)

    def load_lora(
        self,
        pretrained_lora_model_name_or_path_or_dict: Union[str, Dict[str, torch.Tensor]],
        adapter_name: Optional[Any] = None,
        **kwargs,
    ) -> None:
        self.pipe.load_lora_weights(pretrained_lora_model_name_or_path_or_dict, adapter_name, **kwargs)

    def fuse_lora(
        self,
        fuse_unet: bool = True,
        fuse_text_encoder: bool = True,
        lora_scale: float = 1.0,
        safe_fusing: bool = False,
    ) -> None:
        self.pipe.fuse_lora(
            fuse_unet=fuse_unet,
            fuse_text_encoder=fuse_text_encoder,
            lora_scale=lora_scale,
            safe_fusing=safe_fusing,
        )

    def enable_similar_image_filter(self, threshold: float = 0.98, max_skip_frame: float = 10) -> None:
        self.similar_image_filter = True
        self.similar_filter.set_threshold(threshold)
        self.similar_filter.set_max_skip_frame(max_skip_frame)

    def disable_similar_image_filter(self) -> None:
        self.similar_image_filter = False

    @torch.no_grad()
    def prepare(
        self,
        prompt: str,
        negative_prompt: str = "",
        num_inference_steps: int = 50,
        guidance_scale: float = 1.2,
        delta: float = 1.0,
        generator: Optional[torch.Generator] = torch.Generator(),
        seed: int = 2,
    ) -> None:
        self.generator = generator
        self.generator.manual_seed(seed)
        # initialize x_t_latent (it can be any random tensor)
        if self.denoising_steps_num > 1:
            self.x_t_latent_buffer = torch.zeros(
                ((self.denoising_steps_num - 1) * self.frame_bff_size, 4, self.latent_height, self.latent_width),
                dtype=self.dtype,
                device=self.device,
            )
        else:
            self.x_t_latent_buffer = None
        if self.cfg_type == "none":
            self.guidance_scale = 1.0
        else:
            self.guidance_scale = guidance_scale
        self.delta = delta
        do_classifier_free_guidance = False
        if self.guidance_scale > 1.0:
            do_classifier_free_guidance = True
        # Encode prompt to get text embeddings (and pooled embeddings for SD3)
        print(f"Encoding prompt: {prompt}")
        if isinstance(prompt, list):
            pnms = ['prompt', 'prompt_2','prompt_3']
            promptkwargs = dict(zip(pnms, prompt))
        else:    
            promptkwargs= {"prompt": prompt}
        print(f"Prompt kwargs: {promptkwargs}")
        # Updated for SD3.5 encoding
        # Handle multiple prompts for SD3.5
        if isinstance(prompt, list):
            encoder_output = self.pipe.encode_prompt(
                prompt=promptkwargs["prompt"],
                prompt_2=promptkwargs.get("prompt_2", ""),
                prompt_3=promptkwargs.get("prompt_3", ""),
                device=self.device,
                num_images_per_prompt=1,
                do_classifier_free_guidance=True,
                negative_prompt=negative_prompt,
            )
        else:
            encoder_output = self.pipe.encode_prompt(
                prompt=promptkwargs["prompt"],
                device=self.device,
                num_images_per_prompt=1,
                do_classifier_free_guidance=True,
                negative_prompt=negative_prompt,
            )
        # Handle outputs from encode_prompt (supports SD3.5 which returns 4 values, or older pipelines with 2 values)
        if isinstance(encoder_output, (tuple, list)):
            if len(encoder_output) == 4:
                prompt_embeds, neg_prompt_embeds, pooled_prompt_embeds, neg_pooled_prompt_embeds = encoder_output
            elif len(encoder_output) == 2:
                prompt_embeds, neg_prompt_embeds = encoder_output
                pooled_prompt_embeds = None
                neg_pooled_prompt_embeds = None
            else:
                raise ValueError("Unexpected output format from pipe.encode_prompt")
        else:
            raise ValueError("pipe.encode_prompt did not return a tuple/list as expected")
        # Set up prompt embeddings for diffusion model
        self.prompt_embeds = prompt_embeds.repeat(self.batch_size, 1, 1)
        self.null_prompt_embeds = neg_prompt_embeds
        # Set up pooled text embeddings if provided (for SD3.5 transformer)
        if pooled_prompt_embeds is not None:
            # Ensure both tensors have the same dimensions
            # Ensure both tensors have the same dimensions (3D)
            if neg_pooled_prompt_embeds.dim() == 2:
                # Add dimensions to match pooled_prompt_embeds
                neg_pooled_prompt_embeds = neg_pooled_prompt_embeds.unsqueeze(0).unsqueeze(1)
            elif neg_pooled_prompt_embeds.dim() == 3 and pooled_prompt_embeds.dim() == 3:
                # Already in the correct shape
                pass
            else:
                # Handle unexpected dimensions by reshaping to match expected format
                if neg_pooled_prompt_embeds.dim() == 1:
                    neg_pooled_prompt_embeds = neg_pooled_prompt_embeds.unsqueeze(0).unsqueeze(0)
                elif neg_pooled_prompt_embeds.dim() == 2:
                    neg_pooled_prompt_embeds = neg_pooled_prompt_embeds.unsqueeze(0)
            self.pooled_prompt_embeds = pooled_prompt_embeds.repeat(self.batch_size, 1, 1)
            self.null_pooled_prompt_embeds = neg_pooled_prompt_embeds
        else:
            self.pooled_prompt_embeds = None
            self.null_pooled_prompt_embeds = None
        # Duplicate unconditional embeddings for classifier-free guidance, if needed
        if self.use_denoising_batch and self.cfg_type == "full":
            uncond_prompt_embeds = neg_prompt_embeds.repeat(self.batch_size, 1, 1)
            if pooled_prompt_embeds is not None:
                uncond_pooled_prompt_embeds = neg_pooled_prompt_embeds.repeat(self.batch_size, 1, 1)
        elif self.cfg_type == "initialize":
            uncond_prompt_embeds = neg_prompt_embeds.repeat(self.frame_bff_size, 1, 1)
            if pooled_prompt_embeds is not None:
                uncond_pooled_prompt_embeds = neg_pooled_prompt_embeds.repeat(self.frame_bff_size, 1, 1)
        else:
            uncond_prompt_embeds = None  # not used for cfg_type "self" or "none"
        if self.guidance_scale > 1.0 and (self.cfg_type == "initialize" or self.cfg_type == "full"):
            # Concatenate unconditional and conditional embeddings for CFG
            self.prompt_embeds = torch.cat([uncond_prompt_embeds, self.prompt_embeds], dim=0)
            if self.pooled_prompt_embeds is not None:
                print(f"uncond_pooled_prompt_embeds shape: {uncond_pooled_prompt_embeds.shape}")
                print(f"self.pooled_prompt_embeds shape: {self.pooled_prompt_embeds.shape}")
                self.pooled_prompt_embeds = torch.cat([uncond_pooled_prompt_embeds, self.pooled_prompt_embeds], dim=0)
                print(f"After concat pooled_prompt_embeds shape: {self.pooled_prompt_embeds.shape}")
        # Set scheduler timesteps
        self.scheduler.set_timesteps(num_inference_steps, self.device)
        self.timesteps = self.scheduler.timesteps.to(self.device)

        # make sub timesteps list based on the indices in the t_list list and the values in the timesteps list
        # either convert scalars to Python ints...
        self.sub_timesteps = [ self.timesteps[t].item() for t in self.t_list ]
        sub_timesteps_tensor = torch.tensor(
            self.sub_timesteps, dtype=torch.long, device=self.device
        )

        self.sub_timesteps_tensor = torch.repeat_interleave(
            sub_timesteps_tensor,
            repeats=self.frame_bff_size if self.use_denoising_batch else 1,
            dim=0,
        )


        # -    self.sub_timesteps = []
        # -    for t in self.t_list:
        # -        self.sub_timesteps.append(self.timesteps[t])
        # -
        # -    sub_timesteps_tensor = torch.tensor(
        # -        self.sub_timesteps, dtype=torch.long, device=self.device
        # -    )
        #     # Prepare sub-timesteps from specified indices
        #     self.sub_timesteps = []
        #     for t in self.t_list:
        #         self.sub_timesteps.append(self.timesteps[t])
        #     sub_timesteps_tensor = torch.tensor(self.sub_timesteps, dtype=torch.long, device=self.device)
        #     self.sub_timesteps_tensor = torch.repeat_interleave(
        #         sub_timesteps_tensor,
        #         repeats=self.frame_bff_size if self.use_denoising_batch else 1,
        #         dim=0,
        #     )
    
        # Initialize noise tensors
        self.init_noise = torch.randn(
            (self.batch_size, 4, self.latent_height, self.latent_width),
            generator=generator,
        ).to(device=self.device, dtype=self.dtype)
        self.randn_noise = self.init_noise[:1].clone()
        self.warp_noise = self.init_noise[:1].clone()
        self.stock_noise = torch.zeros_like(self.init_noise)

    @torch.no_grad()
    def update_prompt(self, prompt: str) -> None:
        if isinstance(prompt, list):
            pnms = ['prompt', 'prompt_2','prompt_3']
            promptkwargs = dict(zip(pnms, prompt))
        else:    
            promptkwargs= {"prompt": prompt}
            
        encoder_output = self.pipe.encode_prompt(
            **promptkwargs,
            device=self.device,
            num_images_per_prompt=1,
            do_classifier_free_guidance=False,
        )
        if isinstance(encoder_output, (tuple, list)):
            # Only use the conditional prompt embeddings
            prompt_embeds = encoder_output[0]
        else:
            prompt_embeds = encoder_output
        self.prompt_embeds = prompt_embeds.repeat(self.batch_size, 1, 1)
        # (Optionally, could update pooled embeddings if needed, but omitted for simplicity)

    def add_noise(
        self,
        original_samples: torch.Tensor,
        noise: torch.Tensor,
        t_index: int,
    ) -> torch.Tensor:
        # Ensure alpha_prod_t_sqrt and beta_prod_t_sqrt are broadcasted correctly
        alpha = self.alpha_prod_t_sqrt[t_index]
        beta = self.beta_prod_t_sqrt[t_index]
        noisy_samples = (
        )
        return noisy_samples

    def scheduler_step_batch(
        self,
        model_pred_batch: torch.Tensor,
        x_t_latent_batch: torch.Tensor,
        idx: Optional[int] = None,
    ) -> torch.Tensor:
        if idx is None:
            # Compute the denoised latents for the entire batch (all steps at once)
            F_theta = (x_t_latent_batch - self.beta_prod_t_sqrt * model_pred_batch) / self.alpha_prod_t_sqrt
            denoised_batch = self.c_out * F_theta + self.c_skip * x_t_latent_batch
        else:
            # Compute for a single time step index
            F_theta = (x_t_latent_batch - self.beta_prod_t_sqrt[idx] * model_pred_batch) / self.alpha_prod_t_sqrt[idx]
            denoised_batch = self.c_out[idx] * F_theta + self.c_skip[idx] * x_t_latent_batch
        return denoised_batch

    def unet_step(
        self,
        x_t_latent: torch.Tensor,
        t_list: Union[torch.Tensor, List[int]],
        idx: Optional[int] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        # Prepare latents for unconditional + conditional in one batch if doing classifier-free guidance
        if self.guidance_scale > 1.0 and (self.cfg_type == "initialize"):
            x_t_latent_plus_uc = torch.concat([x_t_latent[0:1], x_t_latent], dim=0)
            t_list = torch.concat([t_list[0:1], t_list], dim=0)
        elif self.guidance_scale > 1.0 and (self.cfg_type == "full"):
            x_t_latent_plus_uc = torch.concat([x_t_latent, x_t_latent], dim=0)
            t_list = torch.concat([t_list, t_list], dim=0)
        else:
            # Ensure the input tensor has the correct number of channels (16)
            if x_t_latent.shape[1] != 16:
                # Add a convolutional layer to adjust the number of channels
                weight = torch.randn(16, 4, 3, 3, dtype=x_t_latent.dtype, device=x_t_latent.device)
                x_t_latent_plus_uc = torch.nn.functional.conv2d(x_t_latent, weight=weight, stride=1, padding=1)
            else:
                x_t_latent_plus_uc = x_t_latent
        # Denoise using the pipeline (SD3.5)
        # Fix dimension mismatch for pooled prompt embeds
        if self.pooled_prompt_embeds is not None and hasattr(self, 'null_pooled_prompt_embeds') and self.null_pooled_prompt_embeds is not None:
            # Ensure both tensors have the same dimensions (3D)
            if self.null_pooled_prompt_embeds.dim() == 2:
                # Add dimensions to match pooled_prompt_embeds
                self.null_pooled_prompt_embeds = self.null_pooled_prompt_embeds.unsqueeze(0).unsqueeze(1)
            elif self.null_pooled_prompt_embeds.dim() == 3 and self.pooled_prompt_embeds.dim() == 3:
                # Already in the correct shape
                pass
            else:
                # Handle unexpected dimensions by reshaping to match expected format
                if self.null_pooled_prompt_embeds.dim() == 1:
                    self.null_pooled_prompt_embeds = self.null_pooled_prompt_embeds.unsqueeze(0).unsqueeze(0)
                elif self.null_pooled_prompt_embeds.dim() == 2:
                    self.null_pooled_prompt_embeds = self.null_pooled_prompt_embeds.unsqueeze(0)

        model_out = self.transformer(
            prompt_embeds=self.prompt_embeds,
            pooled_prompt_embeds=self.pooled_prompt_embeds if self.pooled_prompt_embeds is not None else None,
            latents=x_t_latent_plus_uc,
        ).images
        # Extract the predicted noise (epsilon) from the model output
        if isinstance(model_out, tuple):
            # Handle both SD1.5 and SD3.5 output formats
            if len(model_out) >= 9:
                # For SD3.5, the output is a tensor directly
                model_pred = model_out
        else:
            model_pred = model_out
        # Handle classifier-free guidance mixing
        if self.guidance_scale > 1.0 and (self.cfg_type == "initialize"):
            # In "initialize" mode, first element is unconditional prediction
            noise_pred_text = model_pred[1:]
            # Store unconditional noise for self-guidance
            # Update for SD 3.5: Handle text embeddings and model predictions
            if self.cfg_type == "full":
                # In SD 3.5, model_pred contains [uncond, cond] chunks
                noise_pred_uncond, noise_pred_text = model_pred.chunk(2)
            else:
                noise_pred_text = model_pred

            if self.cfg_type == "self" or self.cfg_type == "initialize":
                # Use stored noise as the self-condition unconditional prediction
                noise_pred_uncond = self.stock_noise * self.delta

            if self.cfg_type != "none":
                # Combine unconditional and text predictions (CFG formula)
                model_pred = noise_pred_uncond + self.guidance_scale * (noise_pred_text - noise_pred_uncond)
            else:
                model_pred = noise_pred_text
        # Compute the denoised latent for current step
        if self.use_denoising_batch:
            denoised_batch = self.scheduler_step_batch(model_pred, x_t_latent, idx)
            if self.cfg_type == "self" or self.cfg_type == "initialize":
                # Apply latent self-conditioning update for next step
                scaled_noise = self.beta_prod_t_sqrt * self.stock_noise
                delta_x = self.scheduler_step_batch(model_pred, scaled_noise, idx)
                alpha_next = torch.concat(
                    [self.alpha_prod_t_sqrt[1:], torch.ones_like(self.alpha_prod_t_sqrt[0:1])],
                    dim=0,
                )
                delta_x = alpha_next * delta_x
                Beta_next = torch.concat(
                    [self.beta_prod_t_sqrt[1:], torch.ones_like(self.beta_prod_t_sqrt[0:1])],
                    dim=0,
                )
                delta_x = delta_x / Beta_next
                init_noise = torch.concat([self.init_noise[1:], self.init_noise[0:1]], dim=0)
                self.stock_noise = init_noise + delta_x
        else:
            # Single-step update (not batch mode)
            denoised_batch = self.scheduler_step_batch(model_pred, x_t_latent, idx)
        return denoised_batch, model_pred

    def norm_noise(self, noise):
        # Compute mean and std of blended noise
        mean = noise.mean()
        std = noise.std()
        # Normalize noise to mean=0 and std=1
        normalized_noise = (noise - mean) / std
        return normalized_noise

    def encode_image(self, image_tensors: torch.Tensor) -> torch.Tensor:
        image_tensors = image_tensors.to(device=self.device, dtype=self.vae.dtype)
        # Encode image to latent space
        img_latent_dist = self.vae.encode(image_tensors)
        img_latent = retrieve_latents(img_latent_dist, self.generator)
        # Apply scaling (and shift if applicable) to match diffusion latent space
        if hasattr(self.vae.config, "shift_factor"):
            img_latent = (img_latent - self.vae.config.shift_factor) * self.vae.config.scaling_factor
        else:
            img_latent = img_latent * self.vae.config.scaling_factor
        # Add initial noise to latents for the first diffusion step
        x_t_latent = self.add_noise(img_latent, self.init_noise[0], 0)
        return x_t_latent

    def decode_image(self, x_0_pred_out: torch.Tensor) -> torch.Tensor:
        # Decode latents to image space
        latents = x_0_pred_out / self.vae.config.scaling_factor
        if hasattr(self.vae.config, "shift_factor"):
            latents = latents + self.vae.config.shift_factor
        output_image = self.vae.decode(latents, return_dict=False)[0]
        return output_image

    def predict_x0_batch(self, x_t_latent: torch.Tensor) -> torch.Tensor:
        prev_latent_batch = self.x_t_latent_buffer
        if self.use_denoising_batch:
            t_list = self.sub_timesteps_tensor
            if self.denoising_steps_num > 1:
                # Ensure prev_latent_batch is a tensor before concatenation
                if prev_latent_batch is not None:
                    # Log the type and shape of x_t_latent for debugging
                    print(f"x_t_latent type: {type(x_t_latent)}")
                    if isinstance(x_t_latent, torch.Tensor):
                        print(f"x_t_latent shape: {x_t_latent.shape}")
                    elif isinstance(x_t_latent, tuple):
                        print(f"x_t_latent length: {len(x_t_latent)}")
                        for i, item in enumerate(x_t_latent):
                            print(f"x_t_latent[{i}] type: {type(item)}")
                            if isinstance(item, torch.Tensor):
                                print(f"x_t_latent[{i}] shape: {item.shape}")
                    # Log the type and shape of prev_latent_batch for debugging
                    print(f"prev_latent_batch type: {type(prev_latent_batch)}")
                    if isinstance(prev_latent_batch, torch.Tensor):
                        print(f"prev_latent_batch shape: {prev_latent_batch.shape}")
                    elif isinstance(prev_latent_batch, tuple):
                        print(f"prev_latent_batch[0] type: {type(prev_latent_batch[0])}")
                        if isinstance(prev_latent_batch[0], torch.Tensor):
                            print(f"prev_latent_batch[0] shape: {prev_latent_batch[0].shape}")
                    # Ensure prev_latent_batch is a tensor before concatenation
                    if isinstance(prev_latent_batch, tuple):
                        prev_latent_batch = prev_latent_batch[0]
                    if not isinstance(prev_latent_batch, torch.Tensor):
                        prev_latent_batch = torch.zeros_like(x_t_latent)
                    # Concatenate current latent with buffered latents from previous steps
                    if isinstance(x_t_latent, tuple) and len(x_t_latent) == 0:
                        x_t_latent = prev_latent_batch
                    else:
                        x_t_latent = torch.cat((x_t_latent, prev_latent_batch), dim=0)
                    # Rotate the stock noise buffer for next iteration
                    self.stock_noise = torch.cat((self.init_noise[0:1], self.stock_noise[:-1]), dim=0)
                else:
                    prev_latent_batch = torch.zeros_like(x_t_latent)
            # Perform a batch denoising step for all timesteps at once
            x_0_pred_batch, _ = self.unet_step(x_t_latent, t_list)
            if self.denoising_steps_num > 1:
                # The last element of x_0_pred_batch corresponds to the final denoised latent
                x_0_pred_out = x_0_pred_batch[-1].unsqueeze(0)
                if self.do_add_noise:
                    # Update latent buffer for next frame's iteration
                    self.x_t_latent_buffer = (
                        self.alpha_prod_t_sqrt[1:] * x_0_pred_batch[:-1]
                        + self.beta_prod_t_sqrt[1:] * self.init_noise[1:]
                    )
                else:
                    self.x_t_latent_buffer = (
                        self.alpha_prod_t_sqrt[1:] * x_0_pred_batch[:-1]
                    )
            else:
                x_0_pred_out = x_0_pred_batch
                self.x_t_latent_buffer = None
        else:
            # Sequential denoising when not using batch optimization
            self.init_noise = x_t_latent
            for idx, t in enumerate(self.sub_timesteps_tensor):
                t_batch = t.view(1,).repeat(self.frame_bff_size)
                x_0_pred, _ = self.unet_step(x_t_latent, t_batch, idx)
                if idx < len(self.sub_timesteps_tensor) - 1:
                    # Add noise for next step if needed
                    if self.do_add_noise:
                        x_t_latent = (
                            self.alpha_prod_t_sqrt[idx + 1] * x_0_pred
                            + self.beta_prod_t_sqrt[idx + 1] * torch.randn_like(x_0_pred, device=self.device, dtype=self.dtype)
                        )
                    else:
                        x_t_latent = self.alpha_prod_t_sqrt[idx + 1] * x_0_pred
            x_0_pred_out = x_0_pred
        return x_0_pred_out

    @torch.no_grad()
    def __call__(self, x: Union[torch.Tensor, PIL.Image.Image, np.ndarray] = None) -> torch.Tensor:
        start = self.torchbackend.Event(enable_timing=True)
        end = self.torchbackend.Event(enable_timing=True)
        start.record()
        if x is not None:
            # Preprocess input image to latent
            x = self.image_processor.preprocess(x, self.height, self.width).to(device=self.device, dtype=self.dtype)
            if self.similar_image_filter:
                x = self.similar_filter(x)
            if x is None:
                # If filter drops the frame, reuse previous output after a short delay
                time.sleep(self.inference_time_ema)
                return self.prev_image_result
            x_t_latent = self.encode_image(x)
        else:
            # If no input is provided, start from random noise (text-to-image mode)
            x_t_latent = torch.randn((1, 4, self.latent_height, self.latent_width)).to(device=self.device, dtype=self.dtype)
        # Run the diffusion process to predict the denoised latent
        x_0_pred_out = self.predict_x0_batch(x_t_latent)
        # Decode latent to image
        x_output = self.decode_image(x_0_pred_out).detach().clone()
        self.prev_image_result = x_output
        end.record()
        self.torchbackend.synchronize()
        inference_time = start.elapsed_time(end) / 1000.0
        # Update exponential moving average of inference time
        self.inference_time_ema = 0.9 * self.inference_time_ema + 0.1 * inference_time
        return x_output
