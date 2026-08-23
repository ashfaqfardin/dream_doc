"""
Patches FluxKontextPipeline.prepare_latents to support multiple context images.

By default, diffusers only accepts a single context image. The Kontext paper describes
a 3D RoPE scheme where each context image gets its own temporal index (i=1,2,...,N),
allowing the model to distinguish multiple references. This patch enables that.

Usage:
    python KontextPipeline/patch_diffusers.py           # apply patch
    python KontextPipeline/patch_diffusers.py --check   # verify patch is applied
    python KontextPipeline/patch_diffusers.py --revert  # restore original
"""

import argparse
import importlib
import re
import sys
from pathlib import Path


SENTINEL = "# MULTI-CONTEXT PATCH APPLIED"

OLD_BLOCK = '''\
        image_latents = image_ids = None
        if image is not None:
            image = image.to(device=device, dtype=dtype)
            if image.shape[1] != self.latent_channels:
                image_latents = self._encode_vae_image(image=image, generator=generator)
            else:
                image_latents = image
            if batch_size > image_latents.shape[0] and batch_size % image_latents.shape[0] == 0:
                # expand init_latents for batch_size
                additional_image_per_prompt = batch_size // image_latents.shape[0]
                image_latents = torch.cat([image_latents] * additional_image_per_prompt, dim=0)
            elif batch_size > image_latents.shape[0] and batch_size % image_latents.shape[0] != 0:
                raise ValueError(
                    f"Cannot duplicate `image` of batch size {image_latents.shape[0]} to {batch_size} text prompts."
                )
            else:
                image_latents = torch.cat([image_latents], dim=0)

            image_latent_height, image_latent_width = image_latents.shape[2:]
            image_latents = self._pack_latents(
                image_latents, batch_size, num_channels_latents, image_latent_height, image_latent_width
            )
            image_ids = self._prepare_latent_image_ids(
                batch_size, image_latent_height // 2, image_latent_width // 2, device, dtype
            )
            # image ids are the same as latent ids with the first dimension set to 1 instead of 0
            image_ids[..., 0] = 1'''

NEW_BLOCK = '''\
        image_latents = image_ids = None  ''' + SENTINEL + '''
        if image is not None:
            image = image.to(device=device, dtype=dtype)
            if image.shape[1] != self.latent_channels:
                image_latents_raw = self._encode_vae_image(image=image, generator=generator)
            else:
                image_latents_raw = image

            # Expand single image to fill batch if needed (original behaviour)
            if batch_size > image_latents_raw.shape[0]:
                if batch_size % image_latents_raw.shape[0] != 0:
                    raise ValueError(
                        f"Cannot duplicate `image` of batch size {image_latents_raw.shape[0]} to {batch_size} text prompts."
                    )
                additional_image_per_prompt = batch_size // image_latents_raw.shape[0]
                image_latents_raw = torch.cat([image_latents_raw] * additional_image_per_prompt, dim=0)

            # n_ctx: number of context images per sample in the batch.
            # For a single context image n_ctx=1 (original behaviour).
            # For multiple context images passed as a list, n_ctx>1 and each gets its own
            # 3-D RoPE temporal index (i=1,2,...,n_ctx) as described in the Kontext paper.
            n_ctx = image_latents_raw.shape[0] // batch_size

            all_packed = []
            all_ids = []
            for ctx_idx in range(n_ctx):
                lo, hi = ctx_idx * batch_size, (ctx_idx + 1) * batch_size
                ctx_lat = image_latents_raw[lo:hi]          # (batch_size, C, H, W)
                h, w = ctx_lat.shape[2:]
                packed = self._pack_latents(ctx_lat, batch_size, num_channels_latents, h, w)
                ids = self._prepare_latent_image_ids(batch_size, h // 2, w // 2, device, dtype)
                ids[..., 0] = ctx_idx + 1                   # temporal index: 1, 2, 3, ...
                all_packed.append(packed)
                all_ids.append(ids)

            image_latents = torch.cat(all_packed, dim=1)    # cat along token-sequence dim
            image_ids = torch.cat(all_ids, dim=0)           # cat along sequence dim'''


def find_pipeline_file() -> Path:
    try:
        import diffusers.pipelines.flux.pipeline_flux_kontext as m
        return Path(m.__file__)
    except ImportError:
        sys.exit("diffusers is not installed. Run: pip install diffusers")


def apply_patch(path: Path):
    src = path.read_text(encoding="utf-8")
    if SENTINEL in src:
        print("Patch already applied.")
        return
    if OLD_BLOCK not in src:
        sys.exit(
            "Could not find the target block in pipeline_flux_kontext.py.\n"
            "The diffusers version may have changed. Check the file manually."
        )
    patched = src.replace(OLD_BLOCK, NEW_BLOCK, 1)
    path.write_text(patched, encoding="utf-8")
    print(f"Patch applied: {path}")


def revert_patch(path: Path):
    src = path.read_text(encoding="utf-8")
    if SENTINEL not in src:
        print("Patch is not applied — nothing to revert.")
        return
    if NEW_BLOCK not in src:
        sys.exit("Sentinel found but block doesn't match. File may have been edited manually.")
    reverted = src.replace(NEW_BLOCK, OLD_BLOCK, 1)
    path.write_text(reverted, encoding="utf-8")
    print(f"Patch reverted: {path}")


def check_patch(path: Path):
    src = path.read_text(encoding="utf-8")
    if SENTINEL in src:
        print(f"Patch IS applied: {path}")
    else:
        print(f"Patch is NOT applied: {path}")


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    g = p.add_mutually_exclusive_group()
    g.add_argument("--check",  action="store_true", help="Check whether patch is applied")
    g.add_argument("--revert", action="store_true", help="Revert to original diffusers code")
    args = p.parse_args()

    path = find_pipeline_file()

    if args.check:
        check_patch(path)
    elif args.revert:
        revert_patch(path)
    else:
        apply_patch(path)


if __name__ == "__main__":
    main()
