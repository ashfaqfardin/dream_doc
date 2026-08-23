"""
Orchestrator — one edit = one denoising run with the self-masking schedule.

Runs in a torch/diffusers/GPU environment. Ties together:
  Step 1 packing (via torch_adapters)      Step 2 norm extraction (processor)
  Step 3 repulsion prior (processor)        Step 4 Otsu gate (masking)
  Step 5 memory + occlusion (memory)        + BCG latent blend

This is intentionally a readable reference loop, not a speed-tuned production
path. It calls the model's own pack/unpack helpers where available so token
ordering matches exactly. Names match mainline diffusers FluxPipeline; confirm
against your pinned version.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional
import numpy as np
import torch

from .memory import LayerMemory
from .masking import otsu_gate, refine_mask, soft_prior_fallback_mask, normalize01
from .processor import ExtractionState, attach_processors
from .torch_adapters import (
    occupied_mask_to_token_tensor, norm_vector_to_patch_map, phase_for_step,
)
from .packing import latent_dims_for_image, patch_grid


@dataclass
class EditConfig:
    height: int = 1024
    width: int = 1024
    num_steps: int = 28
    guidance_scale: float = 3.5
    early_frac: float = 0.25
    late_frac: float = 0.70
    otsu_confidence: float = 0.02
    otsu_hard_cutoff_frac: float = 0.40
    penalty: float = 6.0
    collect_blocks: Optional[set] = None      # None -> use a default mid-late band
    bcg_late_floor: float = 0.15              # soft floor for seam-free blending
    coverage: float = 0.5


class SequentialEditor:
    """
    Wraps a loaded diffusers FluxPipeline. Holds the layer memory across edits so
    each new prompt sees the accumulated scene.
    """

    def __init__(self, pipe, config: EditConfig | None = None):
        self.pipe = pipe
        self.cfg = config or EditConfig()
        self.lh, self.lw = latent_dims_for_image(self.cfg.height, self.cfg.width)
        self.ph, self.pw = patch_grid(self.lh, self.lw)
        self.memory = LayerMemory(self.ph, self.pw)
        self.state = ExtractionState(total_steps=self.cfg.num_steps,
                                     penalty=self.cfg.penalty)
        default_band = set(range(8, 15))       # mid-late dual-stream blocks
        attach_processors(pipe.transformer, self.state,
                          collect_blocks=self.cfg.collect_blocks or default_band)
        self._prev_latent = None               # clean latent of committed scene
        self._bcg_noise   = None               # fixed noise realisation for BCG blend

    # ---- text-token span resolution ----------------------------------------

    def _object_token_span(self, prompt: str, object_phrase: str) -> tuple[int, int]:
        """
        Find the text-token index span for the object phrase within the prompt.
        Uses the pipeline tokenizer(s). FLUX uses T5 for the sequence tokens that
        enter the transformer; span-finding is approximate and should be sanity
        checked per-tokenizer. Returns (start, end) exclusive.
        """
        tok = self.pipe.tokenizer_2      # T5 tokenizer in FLUX
        full = tok(prompt, add_special_tokens=True).input_ids
        obj = tok(object_phrase, add_special_tokens=False).input_ids
        # naive contiguous sublist match
        for i in range(len(full) - len(obj) + 1):
            if full[i:i + len(obj)] == obj:
                return (i, i + len(obj))
        # fallback: whole prompt
        return (0, len(full))

    # ---- the core: one edit -------------------------------------------------

    @torch.no_grad()
    def add_object(self, prompt: str, object_phrase: str, seed: int | None = None):
        """
        Add a new object described by `object_phrase` within `prompt`, letting
        FLUX choose placement (repulsed away from occupied regions), harvesting
        the self-derived mask, and committing it to memory.

        Returns dict with the final image and the locked patch mask.
        """

        cfg = self.cfg
        s = self.state
        s.reset_collection()
        s.step_index = 0
        s.obj_token_span = self._object_token_span(prompt, object_phrase)
        self._bcg_noise = None               # fresh noise field per edit

        occupied = self.memory.occupied_union()            # (ph,pw) bool
        device = self.pipe.transformer.device
        s.occupied_token_idx = occupied_mask_to_token_tensor(occupied, device) \
            if occupied.any() else None

        locked_mask: Optional[np.ndarray] = None
        prior_soft = self._build_soft_prior(occupied)      # normalized (ph,pw)

        # ---- manual denoising loop (mirrors FluxPipeline.__call__) ----
        gen = torch.Generator(device=device).manual_seed(seed) if seed is not None else None
        latents, latent_kwargs = self._prepare_latents(prompt, gen)
        timesteps = latent_kwargs["timesteps"]

        # Evolving-mask strategy:
        #   • Start BCG at FIRST Otsu fire (early background protection).
        #   • Whenever a later step fires with fewer tokens, update the BCG mask.
        #   • Commit the final tightest mask when the fire window closes (gap≥2).
        #   • Hard-fallback only if Otsu NEVER fired (best_mask is None).
        best_mask:     Optional[np.ndarray] = None   # tightest raw mask seen
        best_tokens:   int = 10**9
        bcg_mask:      Optional[np.ndarray] = None   # refined mask used for BCG
        last_fired_step: int = -1

        for i, t in enumerate(timesteps):
            s.step_index = i
            phase = phase_for_step(i, cfg.num_steps, cfg.early_frac, cfg.late_frac)
            s.apply_prior = (phase == "early" and s.occupied_token_idx is not None)

            noise_pred = self._transformer_step(latents, t, latent_kwargs)
            latents = self._scheduler_step(noise_pred, t, latents, i)

            # Accumulate fired masks; update BCG mask whenever a tighter one arrives.
            if locked_mask is None:
                nrm = s.aggregate_norm()
                if nrm is not None:
                    pmap = norm_vector_to_patch_map(nrm, self.ph, self.pw)
                    fired, mask, thr, var = otsu_gate(pmap, cfg.otsu_confidence)
                    if fired:
                        n_tok = int(mask.sum())
                        if n_tok < best_tokens:
                            best_mask   = mask
                            best_tokens = n_tok
                            bcg_mask    = refine_mask(mask)   # update BCG immediately
                        last_fired_step = i
                s.reset_collection()

            # Commit the tightest mask once the fire window has closed (2 consecutive
            # non-firing steps). bcg_mask is already refined, reuse it.
            if locked_mask is None and best_mask is not None:
                if i - last_fired_step >= 2:
                    locked_mask = bcg_mask

            # Hard-fallback: Otsu never fired at all past the cutoff.
            if (locked_mask is None
                    and best_mask is None
                    and i >= cfg.otsu_hard_cutoff_frac * cfg.num_steps):
                locked_mask = soft_prior_fallback_mask(prior_soft)
                bcg_mask    = locked_mask

            # BCG: blend from first fire onward (bcg_mask is set earlier than
            # locked_mask, so background protection starts as soon as signal arrives).
            active_mask = locked_mask if locked_mask is not None else bcg_mask
            if active_mask is not None and self._prev_latent is not None:
                if self._bcg_noise is None:
                    self._bcg_noise = torch.randn_like(self._prev_latent)
                latents = self._bcg_blend(latents, active_mask, phase, i)

        image = self._decode(latents)

        # If the loop ended while Otsu was still firing (last step fired, so
        # gap never reached 2), commit whatever best mask we accumulated.
        if locked_mask is None and bcg_mask is not None:
            locked_mask = bcg_mask
        if locked_mask is None:                            # ultimate safety
            locked_mask = soft_prior_fallback_mask(prior_soft)

        span = s.obj_token_span
        self.memory.add_layer(prompt, locked_mask, token_span=span,
                              object_phrase=object_phrase)
        self._prev_latent = latents.detach()
        return {"image": image, "mask": locked_mask,
                "memory_summary": self.memory.summary()}

    # ---- helpers that touch the diffusers pipeline internals ----------------
    # Kept small and named so they can be adapted to a pinned diffusers version.

    def _build_soft_prior(self, occupied_bool: np.ndarray) -> np.ndarray:
        """Free-space prior: high where unoccupied. Used for early nudge target
        and the Otsu hard-fallback mask."""
        free = (~occupied_bool).astype(np.float64)
        if not free.any():
            free = np.ones_like(free)
        return normalize01(free)

    def _prepare_latents(self, prompt, generator):
        """Encode prompt, init latents, get timesteps. Delegates to the pipe's
        own prep so packing/rope match. Returns (latents, kwargs)."""
        # Reference wiring — adapt to pinned diffusers. Pseudocode-level but
        # calls real methods so the intent is unambiguous.
        (prompt_embeds, pooled, text_ids) = self.pipe.encode_prompt(
            prompt=prompt, prompt_2=prompt, device=self.pipe.transformer.device,
            num_images_per_prompt=1,
        )
        num_channels = self.pipe.transformer.config.in_channels // 4
        latents, latent_image_ids = self.pipe.prepare_latents(
            1, num_channels, self.cfg.height, self.cfg.width,
            prompt_embeds.dtype, self.pipe.transformer.device, generator,
        )
        # Newer diffusers requires mu when use_dynamic_shifting=True (FLUX.1-dev default).
        # latents.shape[1] is the packed image-token sequence length.
        sched_kwargs = dict(device=self.pipe.transformer.device)
        if getattr(self.pipe.scheduler.config, "use_dynamic_shifting", False):
            from diffusers.pipelines.flux.pipeline_flux import calculate_shift
            mu = calculate_shift(
                latents.shape[1],
                getattr(self.pipe.scheduler.config, "base_image_seq_len", 256),
                getattr(self.pipe.scheduler.config, "max_image_seq_len", 4096),
                getattr(self.pipe.scheduler.config, "base_shift", 0.5),
                getattr(self.pipe.scheduler.config, "max_shift", 1.16),
            )
            sched_kwargs["mu"] = mu
        self.pipe.scheduler.set_timesteps(self.cfg.num_steps, **sched_kwargs)
        return latents, {
            "prompt_embeds": prompt_embeds, "pooled": pooled,
            "text_ids": text_ids, "latent_image_ids": latent_image_ids,
            "timesteps": self.pipe.scheduler.timesteps,
        }

    def _transformer_step(self, latents, t, kw):
        guidance = torch.full((1,), self.cfg.guidance_scale,
                              device=latents.device, dtype=latents.dtype) \
            if self.pipe.transformer.config.guidance_embeds else None
        ts = t.expand(latents.shape[0]).to(latents.dtype) / 1000
        return self.pipe.transformer(
            hidden_states=latents,
            timestep=ts,
            guidance=guidance,
            pooled_projections=kw["pooled"],
            encoder_hidden_states=kw["prompt_embeds"],
            txt_ids=kw["text_ids"],
            img_ids=kw["latent_image_ids"],
            return_dict=False,
        )[0]

    def _scheduler_step(self, noise_pred, t, latents, i):
        # With cpu_offload the scheduler lives on CPU while noise_pred/latents are on
        # CUDA — move everything to the same device before the step.
        device = latents.device
        return self.pipe.scheduler.step(
            noise_pred.to(device), t.to(device), latents, return_dict=False
        )[0]

    def _bcg_blend(self, latents, patch_mask_2d, phase, step_idx):
        """
        Background Consistency Guidance with soft late floor.

        latents live in packed-token space (b, n_tok, c). The background latent
        must be at the SAME noise level as `latents` or FLUX sees an impossible
        denoising state (clean bg blended with noisy current → blocky seams and
        duplicate objects). We noise `_prev_latent` to match step_idx's sigma
        using a fixed noise realisation stored in `_bcg_noise`.

        Flow-matching forward: x_t = (1-σ) * x_clean + σ * noise, σ∈[1,0].
        """
        from .torch_adapters import occupied_mask_to_token_tensor
        idx = occupied_mask_to_token_tensor(patch_mask_2d, latents.device)
        w = torch.zeros(latents.shape[1], device=latents.device, dtype=latents.dtype)
        w[idx] = 1.0
        if phase == "late":                       # relax: let some new content leak
            w = self.cfg.bcg_late_floor + (1 - self.cfg.bcg_late_floor) * w
        w = w.view(1, -1, 1)

        # Noise the background to the current noise level before blending.
        sigma = self.pipe.scheduler.sigmas[step_idx].to(latents.device, dtype=latents.dtype)
        bg_noise = self._bcg_noise.to(latents.device)
        bg_at_t  = (1.0 - sigma) * self._prev_latent + sigma * bg_noise

        return latents * w + bg_at_t * (1 - w)

    def _decode(self, latents):
        img = self.pipe._unpack_latents(latents, self.cfg.height, self.cfg.width,
                                        self.pipe.vae_scale_factor)
        img = (img / self.pipe.vae.config.scaling_factor) + \
            self.pipe.vae.config.shift_factor
        img = self.pipe.vae.decode(img, return_dict=False)[0]
        return self.pipe.image_processor.postprocess(img, output_type="pil")[0]
