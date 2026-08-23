# Adapted from https://github.com/huggingface/diffusers/blob/main/src/diffusers/pipelines/flux/pipeline_flux.py

import inspect
from typing import Any, Callable, Dict, List, Optional, Union

import numpy as np
import torch
import math
from transformers import CLIPTextModel, CLIPTokenizer, T5EncoderModel, T5TokenizerFast

from diffusers.image_processor import VaeImageProcessor
from diffusers.loaders import FluxLoraLoaderMixin, FromSingleFileMixin
from diffusers import AutoencoderKL
from diffusers import FluxTransformer2DModel
from diffusers.schedulers import FlowMatchEulerDiscreteScheduler

from diffusers.utils.torch_utils import randn_tensor
from diffusers.pipelines.pipeline_utils import DiffusionPipeline

from dataclasses import dataclass
from PIL import Image

from diffusers.utils import BaseOutput

from .utils import get_attr, set_attr_raw
from .flux_blocks import TransformerBlock, SingleTransformerBlock

@dataclass
class PipelineOutput(BaseOutput):
    images: Union[List[Image.Image], np.ndarray]

def calculate_shift(
    image_seq_len,
    base_seq_len: int = 256,
    max_seq_len: int = 4096,
    base_shift: float = 0.5,
    max_shift: float = 1.16):

    m = (max_shift - base_shift) / (max_seq_len - base_seq_len)
    b = base_shift - m * base_seq_len
    mu = image_seq_len * m + b
    return mu


def retrieve_timesteps(
    scheduler,
    num_inference_steps: Optional[int] = None,
    device: Optional[Union[str, torch.device]] = None,
    timesteps: Optional[List[int]] = None,
    sigmas: Optional[List[float]] = None,
    **kwargs):

    if timesteps is not None: # Init from timesteps directly
        scheduler.set_timesteps(timesteps=timesteps, device=device, **kwargs)
        timesteps = scheduler.timesteps
        num_inference_steps = len(timesteps)
    elif sigmas is not None: # Init from sigmas
        scheduler.set_timesteps(sigmas=sigmas, device=device, **kwargs)
        timesteps = scheduler.timesteps
        num_inference_steps = len(timesteps)
    else: # Init from number of timesteps
        scheduler.set_timesteps(num_inference_steps, device=device, **kwargs)
        timesteps = scheduler.timesteps
    return timesteps, num_inference_steps

class GenerationPipelineSemantic(DiffusionPipeline, FromSingleFileMixin):
    model_cpu_offload_seq = "text_encoder->text_encoder_2->transformer->vae"
    _optional_components = []
    _callback_tensor_inputs = ["latents", "prompt_embeds"]

    def __init__(self,
                scheduler: FlowMatchEulerDiscreteScheduler,
                vae: AutoencoderKL,
                text_encoder: CLIPTextModel,
                tokenizer: CLIPTokenizer,
                text_encoder_2: T5EncoderModel,
                tokenizer_2: T5TokenizerFast,
                transformer: FluxTransformer2DModel):

        super().__init__()

        # Register Modules
        self.register_modules(
            vae=vae,
            text_encoder=text_encoder,
            text_encoder_2=text_encoder_2,
            tokenizer=tokenizer,
            tokenizer_2=tokenizer_2,
            transformer=transformer,
            scheduler=scheduler
        )

        self.vae_scale_factor = (
            2 ** (len(self.vae.config.block_out_channels) - 1) if hasattr(self, "vae") and self.vae is not None else 8
        )

        self.image_processor = VaeImageProcessor(vae_scale_factor=self.vae_scale_factor)
        self.tokenizer_max_length = (
            self.tokenizer.model_max_length if hasattr(self, "tokenizer") and self.tokenizer is not None else 77
        )

        self.default_sample_size = 128

    def get_t5_context(self,
                       prompt: Union[str, List[str]] = None,
                       num_images_per_prompt: int = 1,
                       max_sequence_length: int = 512,
                       device: Optional[torch.device] = None,
                       dtype: Optional[torch.dtype] = None):

        device = device or self._execution_device
        dtype = dtype or self.text_encoder.dtype

        prompt = [prompt] if isinstance(prompt, str) else prompt
        batch_size = len(prompt)

        text_input_ids = self.tokenizer_2(
            prompt,
            padding="max_length",
            max_length=max_sequence_length,
            truncation=True,
            return_length=False,
            return_overflowing_tokens=False,
            return_tensors="pt"
        ).input_ids

        untruncated_ids = self.tokenizer_2(prompt, padding="longest", return_tensors="pt").input_ids

        if untruncated_ids.shape[-1] >= text_input_ids.shape[-1] and not torch.equal(text_input_ids, untruncated_ids):
            print(f"WARNING T5 - Input truncated for prompt: {prompt}")

        prompt_embeds = self.text_encoder_2(text_input_ids.to(device), output_hidden_states=False)[0].to(dtype=self.text_encoder_2.dtype, device=device)
        _, seq_len, _ = prompt_embeds.shape

        # Duplicate prompt embeds and attention mask
        prompt_embeds = prompt_embeds.repeat(1, num_images_per_prompt, 1)
        prompt_embeds = prompt_embeds.view(batch_size * num_images_per_prompt, seq_len, -1)

        return prompt_embeds

    def get_clip_context(self,
                         prompt: Union[str, List[str]],
                         num_images_per_prompt: int = 1,
                         device: Optional[torch.device] = None):

        device = device or self._execution_device

        prompt = [prompt] if isinstance(prompt, str) else prompt
        batch_size = len(prompt)

        text_input_ids = self.tokenizer(
            prompt,
            padding="max_length",
            max_length=self.tokenizer_max_length,
            truncation=True,
            return_overflowing_tokens=False,
            return_length=False,
            return_tensors="pt"
        ).input_ids

        untruncated_ids = self.tokenizer(prompt, padding="longest", return_tensors="pt").input_ids
        if untruncated_ids.shape[-1] >= text_input_ids.shape[-1] and not torch.equal(text_input_ids, untruncated_ids):
            print(f"WARNING CLIP - Input truncated for prompt: {prompt}")

        # Pooled Embedding for CLIP
        prompt_embeds = self.text_encoder(text_input_ids.to(device), output_hidden_states=False).pooler_output.to(dtype=self.text_encoder.dtype, device=device)

        # Duplicate prompt embeds and attention mask
        prompt_embeds = prompt_embeds.repeat(1, num_images_per_prompt)
        prompt_embeds = prompt_embeds.view(batch_size * num_images_per_prompt, -1)

        return prompt_embeds

    def encode_prompt(self,
                      prompt: Union[str, List[str]],
                      prompt_2: Union[str, List[str]],
                      device: Optional[torch.device] = None,
                      num_images_per_prompt: int = 1,
                      prompt_embeds: Optional[torch.FloatTensor] = None,
                      pooled_prompt_embeds: Optional[torch.FloatTensor] = None,
                      max_sequence_length: int = 512):

        device = device or self._execution_device

        prompt = [prompt] if isinstance(prompt, str) else prompt

        if prompt is not None:
            batch_size = len(prompt)
        else:
            batch_size = prompt_embeds.shape[0]

        if prompt_embeds is None:
            prompt_2 = prompt_2 or prompt
            prompt_2 = [prompt_2] if isinstance(prompt_2, str) else prompt_2

            # Pooled Embeddings from CLIP
            pooled_prompt_embeds = self.get_clip_context(
                prompt=prompt, device=device, num_images_per_prompt=num_images_per_prompt
            )

            # Sequenced Embeddings from T5
            prompt_embeds = self.get_t5_context(
                prompt=prompt_2,
                num_images_per_prompt=num_images_per_prompt, max_sequence_length=max_sequence_length,
                device=device
            )

        dtype = self.text_encoder.dtype if self.text_encoder is not None else self.transformer.dtype
        text_ids = torch.zeros(prompt_embeds.shape[1], 3).to(device=device, dtype=dtype)

        return prompt_embeds, pooled_prompt_embeds, text_ids

    @staticmethod
    def prepare_latent_image_ids(batch_size, height, width, device, dtype):
        latent_image_ids = torch.zeros(height, width, 3)
        latent_image_ids[..., 1] = latent_image_ids[..., 1] + torch.arange(height)[:, None]
        latent_image_ids[..., 2] = latent_image_ids[..., 2] + torch.arange(width)[None, :]

        latent_image_id_height, latent_image_id_width, latent_image_id_channels = latent_image_ids.shape

        latent_image_ids = latent_image_ids.reshape(latent_image_id_height * latent_image_id_width, latent_image_id_channels)

        return latent_image_ids.to(device=device, dtype=dtype)

    @staticmethod
    def pack_latents(latents, batch_size, num_channels_latents, height, width):
        latents = latents.view(batch_size, num_channels_latents, height // 2, 2, width // 2, 2)
        latents = latents.permute(0, 2, 4, 1, 3, 5)
        latents = latents.reshape(batch_size, height // 2 * width // 2, num_channels_latents * 4)
        return latents

    @staticmethod
    def unpack_latents(latents, height, width, vae_scale_factor):
        batch_size, num_patches, channels = latents.shape
        height = height // vae_scale_factor
        width = width // vae_scale_factor

        latents = latents.view(batch_size, height // 2, width // 2, channels // 4, 2, 2)
        latents = latents.permute(0, 3, 1, 4, 2, 5)
        latents = latents.reshape(batch_size, channels // (2 * 2), height, width)
        return latents

    def enable_vae_slicing(self):
        self.vae.enable_slicing()

    def disable_vae_slicing(self):
        self.vae.disable_slicing()

    def enable_vae_tiling(self):
        self.vae.enable_tiling()

    def disable_vae_tiling(self):
        self.vae.disable_tiling()

    def prepare_latents(self,
                        batch_size: int,
                        num_channels_latents: int,
                        height: int,
                        width: int,
                        dtype: torch.dtype,
                        device: torch.device,
                        generator: torch.Generator,
                        latents=None):

        height = int(height) // self.vae_scale_factor
        width = int(width) // self.vae_scale_factor

        shape = (batch_size, num_channels_latents, height, width)

        if latents is not None:
            latent_image_ids = self.prepare_latent_image_ids(batch_size, height, width, device, dtype)
            return latents.to(device=device, dtype=dtype), latent_image_ids

        latents = randn_tensor(shape, generator=generator, device=device, dtype=dtype)
        latents = self.pack_latents(latents, batch_size, num_channels_latents, height, width)

        latent_image_ids = self.prepare_latent_image_ids(batch_size, height // 2, width // 2, device, dtype)

        return latents, latent_image_ids

    @property
    def guidance_scale(self):
        return self._guidance_scale

    @property
    def joint_attention_kwargs(self):
        return self._joint_attention_kwargs

    @property
    def num_timesteps(self):
        return self._num_timesteps

    @property
    def interrupt(self):
        return self._interrupt

    def register_transformer_blocks(self):
        weight_keys = self.transformer.state_dict().keys()
        transformer_modules = []
        single_transformer_modules = []

        for weight_key in weight_keys:
            module_name = ".".join(weight_key.split(sep=".")[:2])
            if weight_key.startswith("single_transformer_blocks"):
                if module_name not in single_transformer_modules:
                    single_transformer_modules.append(module_name)
            elif weight_key.startswith("transformer_blocks"):
                if module_name not in transformer_modules:
                    transformer_modules.append(module_name)


        for single_transformer_module in single_transformer_modules:
            orig_module = get_attr(self.transformer, single_transformer_module)
            unit = SingleTransformerBlock(orig_module, single_transformer_module)
            set_attr_raw(self.transformer, single_transformer_module, unit)

        for transformer_module in transformer_modules:
            orig_module = get_attr(self.transformer, transformer_module)
            unit = TransformerBlock(orig_module, transformer_module)
            set_attr_raw(self.transformer, transformer_module, unit)

    def get_temb_for_projections(self, timestep, pooled_projections, guidance=None):
        tte = self.transformer.time_text_embed
        tte_dev = next(tte.parameters()).device
        out_dev = timestep.device
        timestep = timestep.to(tte_dev)
        pooled_projections = pooled_projections.to(tte_dev)
        if guidance is not None:
            guidance = guidance.to(tte_dev)
        temb = (
            tte(timestep, pooled_projections)
            if guidance is None
            else tte(timestep, guidance, pooled_projections)
        )
        return temb.to(out_dev)

    @torch.no_grad()
    def __call__(
        self,
        prompt: Union[str, List[str]] = None,
        prompt_2: Union[str, List[str]] = None,
        height: Optional[int] = None,
        width: Optional[int] = None,
        num_inference_steps: Optional[int] = 1,
        timesteps: Optional[List[int]] = None,
        guidance_scale: float = 3.5,
        num_images_per_prompt: Optional[int] = 1,
        generator: Optional[Union[torch.Generator, List[torch.Generator]]] = None,
        latents: Optional[torch.FloatTensor] = None,
        prompt_embeds: Optional[torch.FloatTensor] = None,
        pooled_prompt_embeds: Optional[torch.FloatTensor] = None,
        output_type: Optional[str] = "pil",
        return_dict: bool = True,
        joint_attention_kwargs: Optional[Dict[str, Any]] = None,
        max_sequence_length: int = 512,
        edit_prompt: str = None,
        edit_start_iter: int = 0,
        edit_stop_iter: int = None,
        edit_global_scale: float = 0,
        edit_content_scale: float = 0,
        attention_threshold: float = 0.5
    ):

        height = height or self.default_sample_size * self.vae_scale_factor
        width = width or self.default_sample_size * self.vae_scale_factor

        self._guidance_scale = guidance_scale
        self._joint_attention_kwargs = joint_attention_kwargs
        self._interrupt = False

        if prompt is not None and isinstance(prompt, str):
            batch_size = 1
        elif prompt is not None and isinstance(prompt, list):
            batch_size = len(prompt)
        else:
            batch_size = prompt_embeds.shape[0]

        device = self._execution_device

        # Prepare context
        prompt_embeds, pooled_prompt_embeds, text_ids = self.encode_prompt(
            prompt=prompt,
            prompt_2=prompt_2,
            prompt_embeds=prompt_embeds,
            pooled_prompt_embeds=pooled_prompt_embeds,
            max_sequence_length=max_sequence_length,
            num_images_per_prompt=num_images_per_prompt,
            device=device,
        )

        edit_prompt_embeds, edit_pooled_prompt_embeds, edit_text_ids = self.encode_prompt(
            prompt=edit_prompt,
            prompt_2=edit_prompt,
            prompt_embeds=None,
            pooled_prompt_embeds=None,
            max_sequence_length=max_sequence_length,
            num_images_per_prompt=num_images_per_prompt,
            device=device,
        )

        if self._joint_attention_kwargs is None:
            self._joint_attention_kwargs = {}


        # Negative Prompt Embeds
        neg_prompt_embeds, neg_pooled_prompt_embeds, neg_text_ids = self.encode_prompt(
            prompt="",
            prompt_2="",
            prompt_embeds=None,
            pooled_prompt_embeds=None,
            max_sequence_length=max_sequence_length,
            num_images_per_prompt=num_images_per_prompt,
            device=device,
        )

        # Prepare latents
        num_channels_latents = self.transformer.config.in_channels // 4
        latents, latent_image_ids = self.prepare_latents(
            batch_size=batch_size,
            num_channels_latents=num_channels_latents,
            height=height,
            width=width,
            dtype=prompt_embeds.dtype,
            device=device,
            generator=generator,
            latents=latents
        )

        # Prepare timesteps
        sigmas = np.linspace(1.0, 1 / num_inference_steps, num_inference_steps)

        image_sequence_length = latents.shape[1]
        mu = calculate_shift(
            image_seq_len=image_sequence_length,
            base_seq_len=self.scheduler.config.base_image_seq_len,
            max_seq_len=self.scheduler.config.max_image_seq_len,
            base_shift=self.scheduler.config.base_shift,
            max_shift=self.scheduler.config.max_shift
        )

        timesteps, num_inference_steps = retrieve_timesteps(
            scheduler=self.scheduler,
            num_inference_steps=num_inference_steps,
            device=device,
            timesteps=timesteps,
            sigmas=sigmas,
            mu=mu
        )

        num_warmup_steps = max(len(timesteps) - num_inference_steps * self.scheduler.order, 0)
        self._num_timesteps = len(timesteps)

        # Guidance
        if self.transformer.config.guidance_embeds:
            guidance = torch.full([1], guidance_scale, device=device, dtype=torch.float32)
            guidance = guidance.expand(latents.shape[0])
        else:
            guidance = None

        # Set edit_stop_iter to num_inference_steps if not provided
        if edit_stop_iter is None:
            edit_stop_iter = num_inference_steps

        self._joint_attention_kwargs["start_timestep_idx"] = edit_start_iter
        self._joint_attention_kwargs["stop_timestep_idx"] = edit_stop_iter
        self._joint_attention_kwargs["edit_content_scale"] = edit_content_scale
        self._joint_attention_kwargs["attention_threshold"] = attention_threshold

        # Project Embeddings
        emb_product = torch.sum(edit_pooled_prompt_embeds * pooled_prompt_embeds, dim=1, keepdim=True)

        emb_norm_squared = torch.sum(pooled_prompt_embeds * pooled_prompt_embeds, dim=1, keepdim=True)

        emb_projection = (emb_product / (emb_norm_squared + 1e-10)) * pooled_prompt_embeds

        ortho_emb = edit_pooled_prompt_embeds - emb_projection

        assert abs(edit_global_scale) >= 0 and abs(edit_global_scale) <= 1

        edit_pooled_prompt_embeds = (1 - edit_global_scale) * pooled_prompt_embeds + edit_global_scale * (ortho_emb)


        # Denoising Loop
        with self.progress_bar(total=num_inference_steps) as progress_bar:
            for i, t in enumerate(timesteps):
                if self.interrupt:
                    continue

                _ce = self.transformer.context_embedder
                _ce_dev = next(_ce.parameters()).device
                _lat_dev = latents.device
                self._joint_attention_kwargs["edit_prompt_embeds"] = _ce(edit_prompt_embeds.to(_ce_dev)).to(_lat_dev)
                self._joint_attention_kwargs["neg_prompt_embeds"] = _ce(neg_prompt_embeds.to(_ce_dev)).to(_lat_dev)

                timestep = t.expand(latents.shape[0]).to(latents.dtype)

                self._joint_attention_kwargs["current_timestep_idx"] = i

                temb_edit = self.get_temb_for_projections(
                    timestep,
                    pooled_projections=edit_pooled_prompt_embeds,
                    guidance=guidance * 1000
                )

                self._joint_attention_kwargs["temb_edit"] = temb_edit

                noise_pred = self.transformer(
                    hidden_states=latents,
                    timestep=timestep / 1000,
                    guidance=guidance,
                    pooled_projections=pooled_prompt_embeds,
                    encoder_hidden_states=prompt_embeds,
                    txt_ids=text_ids,
                    img_ids=latent_image_ids,
                    joint_attention_kwargs=self.joint_attention_kwargs,
                    return_dict=False
                )[0]

                latents = self.scheduler.step(noise_pred, t, latents, return_dict=False)[0]

                # call the callback, if provided
                if i == len(timesteps) - 1 or ((i + 1) > num_warmup_steps and (i + 1) % self.scheduler.order == 0):
                    progress_bar.update()


        if output_type == "latent":
            image = latents

        else:
            latents = self.unpack_latents(latents, height, width, self.vae_scale_factor)
            latents = (latents / self.vae.config.scaling_factor) + self.vae.config.shift_factor
            image = self.vae.decode(latents, return_dict=False)[0]
            image = self.image_processor.postprocess(image, output_type=output_type)

        self.maybe_free_model_hooks()

        if not return_dict:
            return (image,)

        return PipelineOutput(images=image)
