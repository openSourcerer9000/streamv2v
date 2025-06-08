"""StreamV2V – Stable Diffusion 3.5 Medium‑Turbo version
------------------------------------------------------
A drop‑in replacement for the original StreamV2V pipeline that targets the
`tensorart/stable-diffusion-3.5-medium-turbo` model (and any other SD 3
medium‑class model) via 🤗 Diffusers’ `StableDiffusion3Pipeline`.

Key adaptations
===============
* Detects the latent channel count from the VAE (`4` for SD 1.x, `16` for
  SD 3) and uses it everywhere noise / latent tensors are created.
* Handles the four‑tuple returned by `encode_prompt` (`prompt_embeds`,
  `negative_prompt_embeds`, `pooled_prompt_embeds`,
  `negative_pooled_prompt_embeds`).
* Feeds both **positive** and **negative** (pooled) embeddings to the
  pipeline call instead of manually concatenating unconditional/conditional
  chunks.
* Implements `add_noise` correctly so `x_t_latent` is always a **tensor**.
* Removes the old random‑convolution hack – latents are created with the
  correct channel count from the start.

The public API (constructor, `prepare`, `update_prompt`, `__call__`) remains
unchanged relative to the previous StreamV2V implementation, so existing
calling code continues to work.
"""

from __future__ import annotations

import time
from collections import deque
from typing import Any, Dict, List, Literal, Optional, Tuple, Union

import numpy as np
import PIL.Image
import torch
from torchvision.models.optical_flow import raft_small

from diffusers import (
    FlowMatchEulerDiscreteScheduler,
    StableDiffusion3Pipeline as StableDiffusionPipeline,
)
from diffusers.image_processor import VaeImageProcessor
from diffusers.pipelines.stable_diffusion.pipeline_stable_diffusion_img2img import (
    retrieve_latents,
)

# Project‑local helpers -------------------------------------------------------
from .image_utils import postprocess_image, forward_backward_consistency_check  # noqa: F401 – kept for API parity
from .models.utils import get_nn_latent  # noqa: F401 – kept for API parity
from .image_filter import SimilarImageFilter


def _repeat(t: Optional[torch.Tensor], times: int) -> Optional[torch.Tensor]:
    """Utility: repeat `t` along batch dim `times` if `t` is not None."""
    if t is None:
        return None
    return t.repeat(times, 1, *([] if t.dim() == 2 else [1]))


class StreamV2V:
    """Video‑to‑video streaming pipeline for Stable Diffusion 3.5."""

    # ---------------------------------------------------------------------
    # Construction / state -------------------------------------------------
    # ---------------------------------------------------------------------

    def __init__(
        self,
        pipe: StableDiffusionPipeline,
        t_index_list: List[int],
        *,
        torch_dtype: torch.dtype = torch.float16,
        width: int = 512,
        height: int = 512,
        do_add_noise: bool = True,
        use_denoising_batch: bool = True,
        frame_buffer_size: int = 1,
        cfg_type: Literal["none", "full", "self", "initialize"] = "self",
    ) -> None:
        self.pipe = pipe
        self.device = pipe.device
        if self.device.type == "mps":
            self._tb = torch.mps  # timing backend
        elif self.device.type == "cuda":
            if not torch.cuda.is_available():
                raise RuntimeError("CUDA unavailable – check your PyTorch install.")
            self._tb = torch.cuda
        else:
            raise RuntimeError("StreamV2V requires CUDA or MPS device.")

        # -----------------------------------------------------------------
        # Model‑specific constants
        # -----------------------------------------------------------------
        self.dtype = torch_dtype
        self.height = height
        self.width = width
        self.latent_height = height // pipe.vae_scale_factor
        self.latent_width = width // pipe.vae_scale_factor
        # SD 3 VAE outputs 16‑channel latents; SD 1.x = 4.
        self.latent_channels = getattr(pipe.vae.config, "latent_channels", 4)

        # Denoising step handling
        self.t_index_list = t_index_list
        self.denoising_steps_num = len(t_index_list)
        self.use_denoising_batch = use_denoising_batch
        self.frame_buffer_size = frame_buffer_size
        self.cfg_type = cfg_type
        self.do_add_noise = do_add_noise

        # Effective batch size (steps × frames) when batching timesteps
        if use_denoising_batch:
            self.batch_size = self.denoising_steps_num * frame_buffer_size
        else:
            self.batch_size = frame_buffer_size

        # -----------------------------------------------------------------
        # Diffusion scheduler & helpers
        # -----------------------------------------------------------------
        self.scheduler = FlowMatchEulerDiscreteScheduler.from_config(pipe.scheduler.config)
        # alpha/β pre‑computations for `add_noise`
        self.alpha_prod_t_sqrt = torch.zeros(len(t_index_list), device=self.device)
        self.beta_prod_t_sqrt = torch.zeros_like(self.alpha_prod_t_sqrt)
        for i, t in enumerate(t_index_list):
            sigma_t = self.scheduler.sigmas[t]
            self.alpha_prod_t_sqrt[i] = torch.sqrt(1 - sigma_t**2)
            self.beta_prod_t_sqrt[i] = torch.sqrt(sigma_t**2)
        # Default Euler coefficients (can be overridden externally)
        self.c_out = torch.ones_like(self.alpha_prod_t_sqrt, device=self.device)
        self.c_skip = torch.zeros_like(self.alpha_prod_t_sqrt, device=self.device)

        # Text encoders / VAE / optical‑flow
        self.text_encoder = pipe.text_encoder
        self.vae = pipe.vae
        self.flow_model = raft_small(pretrained=True, progress=False).to(self.device).eval()

        # Misc state
        self.image_processor = VaeImageProcessor(pipe.vae_scale_factor)
        self.similar_filter = SimilarImageFilter()
        self.similar_image_filter = False
        self.cached_x_t_latent: deque[torch.Tensor] = deque(maxlen=4)
        self.inference_time_ema = 0.0

        # Will be filled in `prepare()`
        self.prompt_embeds: Optional[torch.Tensor] = None
        self.null_prompt_embeds: Optional[torch.Tensor] = None
        self.pooled_prompt_embeds: Optional[torch.Tensor] = None
        self.null_pooled_prompt_embeds: Optional[torch.Tensor] = None
        self.generator: Optional[torch.Generator] = None

    # ---------------------------------------------------------------------
    # Public helpers -------------------------------------------------------
    # ---------------------------------------------------------------------

    def enable_similar_image_filter(self, threshold: float = 0.98, max_skip_frame: int = 10) -> None:
        self.similar_image_filter = True
        self.similar_filter.set_threshold(threshold)
        self.similar_filter.set_max_skip_frame(max_skip_frame)

    def disable_similar_image_filter(self) -> None:
        self.similar_image_filter = False

    # ---------------------------------------------------------------------
    # Diffusion preparation ------------------------------------------------
    # ---------------------------------------------------------------------

    @torch.no_grad()
    def prepare(
        self,
        prompt: Union[str, List[str]],
        *,
        negative_prompt: str = "",
        num_inference_steps: int = 50,
        guidance_scale: float = 1.2,
        delta: float = 1.0,
        generator: Optional[torch.Generator] = None,
        seed: int | None = 2,
    ) -> None:
        """Encode prompt text and initialise noise/latent buffers."""

        # RNG for reproducibility ------------------------------------------------
        self.generator = generator or torch.Generator(device=self.device)
        if seed is not None:
            self.generator.manual_seed(seed)

        # Buffers for multi‑step batching --------------------------------------
        if self.denoising_steps_num > 1:
            self.x_t_latent_buffer = torch.zeros(
                (self.denoising_steps_num - 1) * self.frame_buffer_size,
                self.latent_channels,
                self.latent_height,
                self.latent_width,
                dtype=self.dtype,
                device=self.device,
            )
        else:
            self.x_t_latent_buffer = None

        # Classifier‑free guidance config --------------------------------------
        self.guidance_scale = 1.0 if self.cfg_type == "none" else guidance_scale
        self.delta = delta

        # ---------------------------------------------------------------------
        # Encode prompt(s)
        # ---------------------------------------------------------------------
        if isinstance(prompt, list):
            prompt_dict = {
                "prompt": prompt[0],
                "prompt_2": prompt[1] if len(prompt) > 1 else "",
                "prompt_3": prompt[2] if len(prompt) > 2 else "",
            }
        else:
            prompt_dict = {"prompt": prompt}

        encoder_out = self.pipe.encode_prompt(
            **prompt_dict,
            device=self.device,
            num_images_per_prompt=1,
            do_classifier_free_guidance=True,
            negative_prompt=negative_prompt,
        )
        if len(encoder_out) == 4:
            (
                prompt_embeds,
                neg_prompt_embeds,
                pooled_prompt_embeds,
                neg_pooled_prompt_embeds,
            ) = encoder_out
        else:  # fall back (shouldn’t happen with SD 3)
            prompt_embeds, neg_prompt_embeds = encoder_out
            pooled_prompt_embeds = neg_pooled_prompt_embeds = None

        # Repeat embeddings to cover batched timesteps -------------------------
        self.prompt_embeds = _repeat(prompt_embeds, self.batch_size)
        self.null_prompt_embeds = _repeat(neg_prompt_embeds, self.batch_size)

        if pooled_prompt_embeds is not None:
            # Ensure 2‑D tensors [batch, emb_dim]
            if pooled_prompt_embeds.dim() == 3 and pooled_prompt_embeds.shape[1] == 1:
                pooled_prompt_embeds = pooled_prompt_embeds.squeeze(1)
            if neg_pooled_prompt_embeds is not None and neg_pooled_prompt_embeds.dim() == 3 and neg_pooled_prompt_embeds.shape[1] == 1:
                neg_pooled_prompt_embeds = neg_pooled_prompt_embeds.squeeze(1)
            self.pooled_prompt_embeds = _repeat(pooled_prompt_embeds, self.batch_size)
            self.null_pooled_prompt_embeds = _repeat(neg_pooled_prompt_embeds, self.batch_size)
        else:
            self.pooled_prompt_embeds = self.null_pooled_prompt_embeds = None

        # ---------------------------------------------------------------------
        # Scheduler timesteps
        # ---------------------------------------------------------------------
        self.scheduler.set_timesteps(num_inference_steps, self.device)
        self.timesteps = self.scheduler.timesteps.to(self.device)
        self.sub_timesteps = [int(self.timesteps[t]) for t in self.t_index_list]
        self.sub_timesteps_tensor = torch.repeat_interleave(
            torch.tensor(self.sub_timesteps, device=self.device, dtype=torch.long),
            repeats=self.frame_buffer_size if self.use_denoising_batch else 1,
            dim=0,
        )

        # ---------------------------------------------------------------------
        # Noise initialisation
        # ---------------------------------------------------------------------
        self.init_noise = torch.randn(
            self.batch_size,
            self.latent_channels,
            self.latent_height,
            self.latent_width,
            generator=self.generator,
            device=self.device,
            dtype=self.dtype,
        )
        self.stock_noise = self.init_noise.clone()

    # ---------------------------------------------------------------------
    # Prompt update (live)
    # ---------------------------------------------------------------------

    @torch.no_grad()
    def update_prompt(self, prompt: Union[str, List[str]]) -> None:
        if isinstance(prompt, list):
            prompt_dict = {
                "prompt": prompt[0],
                "prompt_2": prompt[1] if len(prompt) > 1 else "",
                "prompt_3": prompt[2] if len(prompt) > 2 else "",
            }
        else:
            prompt_dict = {"prompt": prompt}

        # Only conditional embeds (no CFG here)
        prompt_embeds = self.pipe.encode_prompt(
            **prompt_dict,
            device=self.device,
            num_images_per_prompt=1,
            do_classifier_free_guidance=False,
        )[0]
        self.prompt_embeds = _repeat(prompt_embeds, self.batch_size)

    # ---------------------------------------------------------------------
    # Core math helpers -----------------------------------------------------
    # ---------------------------------------------------------------------

    def add_noise(self, x0: torch.Tensor, noise: torch.Tensor, t_idx: int) -> torch.Tensor:
        """Forward diffusion: q(x_t|x_0)."""
        return self.alpha_prod_t_sqrt[t_idx] * x0 + self.beta_prod_t_sqrt[t_idx] * noise

    def scheduler_step_batch(
        self,
        model_pred: torch.Tensor,
        x_t: torch.Tensor,
        idx: Optional[int] = None,
    ) -> torch.Tensor:
        """Simple Euler–style update using pre‑set c_out / c_skip."""
        if idx is None:
            f_theta = (x_t - self.beta_prod_t_sqrt.view(-1, 1, 1, 1) * model_pred) / self.alpha_prod_t_sqrt.view(
                -1, 1, 1, 1
            )
            return self.c_out.view(-1, 1, 1, 1) * f_theta + self.c_skip.view(-1, 1, 1, 1) * x_t
        # single step
        f_theta = (x_t - self.beta_prod_t_sqrt[idx] * model_pred) / self.alpha_prod_t_sqrt[idx]
        return self.c_out[idx] * f_theta + self.c_skip[idx] * x_t

    # ---------------------------------------------------------------------
    # Transformer (UNet) step wrapper
    # ---------------------------------------------------------------------

    def transformer_step(
        self,
        x_t_latent: torch.Tensor,
        t_batch: torch.Tensor,
    ) -> torch.Tensor:
        """Call StableDiffusion3Pipeline in latent mode and return ε prediction."""
        # Build kwargs depending on guidance mode --------------------------------
        pipe_kwargs: Dict[str, Any] = {
            "prompt_embeds": self.prompt_embeds,
            "latents": x_t_latent,
            "output_type": "latent",
            "guidance_scale": self.guidance_scale,
            "num_inference_steps": self.denoising_steps_num,
        }
        if self.guidance_scale > 1.0 and self.cfg_type != "self":
            pipe_kwargs["negative_prompt_embeds"] = self.null_prompt_embeds
            if self.pooled_prompt_embeds is not None:
                pipe_kwargs["pooled_prompt_embeds"] = self.pooled_prompt_embeds
                pipe_kwargs["negative_pooled_prompt_embeds"] = self.null_pooled_prompt_embeds
        else:  # self‑guidance or no CFG
            if self.pooled_prompt_embeds is not None:
                pipe_kwargs["pooled_prompt_embeds"] = self.pooled_prompt_embeds
        # ---------------------------------------------------------------------
        model_latent = self.pipe(**pipe_kwargs).images  # (B, C, H, W)
        return model_latent

    # ---------------------------------------------------------------------
    # Encode / decode helpers ------------------------------------------------
    # ---------------------------------------------------------------------

    def encode_image(self, img_tensor: torch.Tensor) -> torch.Tensor:
        img_tensor = img_tensor.to(device=self.device, dtype=self.vae.dtype)
        latent_dist = self.vae.encode(img_tensor)
        latent = retrieve_latents(latent_dist, self.generator)
        scale = getattr(self.vae.config, "scaling_factor", 0.18215)
        latent = latent * scale
        return self.add_noise(latent, self.init_noise[0], 0)

    def decode_image(self, latent: torch.Tensor) -> torch.Tensor:
        scale = getattr(self.vae.config, "scaling_factor", 0.18215)
        latent = latent / scale
        img = self.vae.decode(latent, return_dict=False)[0]
        return img

    # ---------------------------------------------------------------------
    # Denoising loop --------------------------------------------------------
    # ---------------------------------------------------------------------

    def predict_x0_batch(self, x_t_latent: torch.Tensor) -> torch.Tensor:
        # For batch mode, glue previous‑step latents for all timesteps -------------
        if self.use_denoising_batch and self.x_t_latent_buffer is not None:
            x_t_latent = torch.cat((x_t_latent, self.x_t_latent_buffer), dim=0)
        # One transformer call gives ε, we convert using scheduler_step_batch
        model_pred = self.transformer_step(x_t_latent, self.sub_timesteps_tensor)
        x_0_pred = self.scheduler_step_batch(model_pred, x_t_latent)
        # Update buffer with intermediate latents for next frame ------------------
        if self.use_denoising_batch and self.denoising_steps_num > 1:
            self.x_t_latent_buffer = (
                self.alpha_prod_t_sqrt[1:].view(-1, 1, 1, 1) * x_0_pred[:-1]
                + self.beta_prod_t_sqrt[1:].view(-1, 1, 1, 1) * self.init_noise[1:]
            )
            return x_0_pred[-1:]
        return x_0_pred
    
    #lora stuff
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

    # ---------------------------------------------------------------------
    # Main call (per frame) -------------------------------------------------
    # ---------------------------------------------------------------------

    @torch.no_grad()
    def __call__(self, x: Union[torch.Tensor, PIL.Image.Image, np.ndarray, None] = None) -> torch.Tensor:
        start = self._tb.Event(enable_timing=True)
        end = self._tb.Event(enable_timing=True)
        start.record()

        # ------------------------------------------------------------------
        # Input processing
        # ------------------------------------------------------------------
        if x is not None:
            x = self.image_processor.preprocess(x, self.height, self.width).to(
                device=self.device, dtype=self.dtype
            )
            if self.similar_image_filter:
                x = self.similar_filter(x)
                if x is None:  # dropped (too similar)
                    time.sleep(self.inference_time_ema)
                    return self.prev_image_result  # type: ignore[arg‑type]
            x_t_latent = self.encode_image(x)
        else:
            x_t_latent = torch.randn(
                1,
                self.latent_channels,
                self.latent_height,
                self.latent_width,
                device=self.device,
                dtype=self.dtype,
            )

        # ------------------------------------------------------------------
        # Diffusion
        # ------------------------------------------------------------------
        x0_pred = self.predict_x0_batch(x_t_latent)
        img_out = self.decode_image(x0_pred).detach().clone()
        self.prev_image_result = img_out

        end.record()
        self._tb.synchronize()
        t = start.elapsed_time(end) / 1000.0  # seconds
        self.inference_time_ema = 0.9 * self.inference_time_ema + 0.1 * t
        return img_out
