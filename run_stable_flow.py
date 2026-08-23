import argparse
import os

import torch
from diffusers import FluxPipeline
import numpy as np
from PIL import Image

class StableFlow:
    MULTIMODAL_VITAL_LAYERS = [0, 1, 17, 18]
    SINGLE_MODAL_VITAL_LAYERS = list(np.array([28, 53, 54, 56, 25]) - 19) 

    def __init__(self):
        self._parse_args()
        self._load_pipeline()

    def _parse_args(self):
        parser = argparse.ArgumentParser()
        
        parser.add_argument("--model_path", type=str, default="black-forest-labs/FLUX.1-dev")
        parser.add_argument("--hf_token", type=str, required=True)
        parser.add_argument("--prompts", type=str, nargs="+", required=True)
        parser.add_argument("--output_path", type=str, default="outputs/result.jpg")
        parser.add_argument("--input_img_path", type=str, default=None)
        parser.add_argument("--seed", type=int, default=42)
        parser.add_argument("--cpu_offload", action="store_true")
        parser.add_argument("--device", type=str, default="cuda")

        self.args = parser.parse_args()
        os.makedirs(os.path.dirname(self.args.output_path), exist_ok=True)

    def _load_pipeline(self):
        self.pipe = FluxPipeline.from_pretrained(
            self.args.model_path, 
            torch_dtype=torch.float16,
            visualize_attention=False,
            token=self.args.hf_token
        )

        if self.args.cpu_offload:
            self.pipe.enable_sequential_cpu_offload()
        else:
            self.pipe.to(self.args.device)

    @torch.no_grad()
    def infer_and_save(self, prompts):
        latents = torch.randn(
            (4096, 64), 
            generator=torch.Generator(0).manual_seed(self.args.seed), 
            device=self.args.device, 
            dtype=torch.float16
        ).tile(len(prompts), 1, 1)
        images = self.pipe(
            prompts,
            height=1024,
            width=1024,
            guidance_scale=3.5,
            output_type="pil",
            num_inference_steps=15,
            max_sequence_length=512,
            latents=latents,
            mm_copy_blocks=StableFlow.MULTIMODAL_VITAL_LAYERS,
            single_copy_blocks=StableFlow.SINGLE_MODAL_VITAL_LAYERS,
        ).images
        images = [np.array(img) for img in images]
        res = Image.fromarray(np.hstack((images)))
        res.save(self.args.output_path)

    @torch.no_grad()
    def image2latent(self, image, latent_nudging_scalar = 1.15):
        image = self.pipe.image_processor.preprocess(image).type(self.pipe.vae.dtype).to("cuda")
        latents = self.pipe.vae.encode(image)["latent_dist"].mean
        latents = (latents - self.pipe.vae.config.shift_factor) * self.pipe.vae.config.scaling_factor
        latents = latents * latent_nudging_scalar
        latents = self.pipe._pack_latents(
            latents=latents,
            batch_size=1,
            num_channels_latents=16,
            height=128,
            width=128
        )

        return latents


    @torch.no_grad()
    def invert_and_save(self, prompts):
        inversion_prompt = prompts[0:1]
        # Invert
        inverted_latent_list = self.pipe(
            inversion_prompt,
            height=1024,
            width=1024,
            guidance_scale=1,
            output_type="pil",
            num_inference_steps=50,
            max_sequence_length=512,
            latents=self.image2latent(Image.open(self.args.input_img_path)),
            invert_image=True
        )

        # Edit
        images = self.pipe(
            prompts,
            height=1024,
            width=1024,
            guidance_scale=[1] + [3] * (len(prompts) - 1),
            output_type="pil",
            num_inference_steps=50,
            max_sequence_length=512,
            latents=inverted_latent_list[-1].tile(len(prompts), 1, 1),
            inverted_latent_list=inverted_latent_list,
            mm_copy_blocks=StableFlow.MULTIMODAL_VITAL_LAYERS,
            single_copy_blocks=StableFlow.SINGLE_MODAL_VITAL_LAYERS,
        ).images
        images = [np.array(img) for img in images]
        res = Image.fromarray(np.hstack((images)))
        res.save(self.args.output_path)

if __name__ == "__main__":
    stable_flow = StableFlow()
    if stable_flow.args.input_img_path is None:
        stable_flow.infer_and_save(prompts=stable_flow.args.prompts)
    else:
        stable_flow.invert_and_save(prompts=stable_flow.args.prompts)
