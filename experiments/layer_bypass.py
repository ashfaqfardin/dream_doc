"""
Layer bypass utilities — supports FLUX and SD3 model families.

Model          Pipeline class               Bypass mechanism
-----------    -------------------------    -----------------------------------------
FLUX.1-dev     FluxPipeline (local fork)    mm_skip_blocks / single_skip_blocks (built-in)
FLUX.1-schnell FluxPipeline (local fork)    same, guidance_scale=0.0
FLUX.2-dev     Flux2Pipeline (upstream)     monkey-patch block.forward, bfloat16
SD 3.5 Large   StableDiffusion3Pipeline     monkey-patch block.forward (no built-in skip)

Usage
-----
from experiments.layer_bypass import load_pipeline, get_block_counts, generate_with_bypass

pipe = load_pipeline("black-forest-labs/FLUX.2-dev", hf_token, device)
n_mm, n_single = get_block_counts(pipe)   # (19, 38) for FLUX.2-dev

img = generate_with_bypass(pipe, prompt, seed=0, block_type="mm", bypass_idx=5)
"""

import contextlib
import os
import sys
import warnings
from typing import Optional

import torch
from PIL import Image

# Ensure the local diffusers fork takes precedence over any system install.
# The local fork lives in <repo_root>/src/; this file is in <repo_root>/experiments/.
_LOCAL_DIFFUSERS_SRC = os.path.normpath(
    os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src")
)
if os.path.isdir(_LOCAL_DIFFUSERS_SRC) and _LOCAL_DIFFUSERS_SRC not in sys.path:
    sys.path.insert(0, _LOCAL_DIFFUSERS_SRC)


# ---------------------------------------------------------------------------
# Model detection
# ---------------------------------------------------------------------------

def detect_model_type(pipe) -> str:
    """Return 'flux', 'flux2', or 'sd3' based on pipeline class."""
    cls_name = type(pipe).__name__
    if "Flux2" in cls_name:
        return "flux2"
    if "Flux" in cls_name:
        return "flux"
    if "StableDiffusion3" in cls_name:
        return "sd3"
    # Fallback: inspect transformer
    t = pipe.transformer
    if hasattr(t, "single_transformer_blocks"):
        return "flux"
    return "sd3"


def get_block_counts(pipe) -> tuple[int, int]:
    """
    Return (n_mm_blocks, n_single_blocks).

    FLUX.1-dev / schnell / FLUX.2-dev  → (19, 38)
    SD 3.5 Large                        → (38,  0)
    """
    model_type = detect_model_type(pipe)
    if model_type in ("flux", "flux2"):
        n_mm     = len(pipe.transformer.transformer_blocks)
        n_single = len(pipe.transformer.single_transformer_blocks)
        return n_mm, n_single
    else:  # sd3
        n_mm = len(pipe.transformer.transformer_blocks)
        return n_mm, 0


def _default_guidance(pipe) -> float:
    """Return a sensible default guidance scale for the detected model."""
    model_type = detect_model_type(pipe)
    if model_type == "flux2":
        return 4.0
    if model_type == "sd3":
        # SD3.5 Large: recommended 4.5; SD3-medium: 7.0
        model_id = getattr(pipe, "name_or_path", "") or ""
        return 4.5 if "3.5" in model_id else 7.0
    # FLUX.1: schnell uses 0.0, dev uses 3.5
    model_id = getattr(pipe, "name_or_path", "") or ""
    if "schnell" in model_id.lower():
        return 0.0
    return 3.5


# ---------------------------------------------------------------------------
# SD3 bypass context manager (monkey-patch block.forward)
# ---------------------------------------------------------------------------

@contextlib.contextmanager
def _sd3_bypass_block(pipe, layer_idx: int):
    """
    Temporarily replace block.forward so it passes tensors through unchanged.

    SD3 JointTransformerBlock.forward() signature:
        (hidden_states, encoder_hidden_states, temb, ...) → (encoder_hidden_states, hidden_states)
    The return order is swapped vs the input order — the skip must match it.
    """
    block = pipe.transformer.transformer_blocks[layer_idx]
    original_forward = block.forward

    def _skip(hidden_states, encoder_hidden_states, temb=None, **kwargs):
        return encoder_hidden_states, hidden_states  # match actual return order

    block.forward = _skip
    try:
        yield
    finally:
        block.forward = original_forward


# ---------------------------------------------------------------------------
# FLUX.2 bypass context managers (monkey-patch block.forward)
# ---------------------------------------------------------------------------

@contextlib.contextmanager
def _flux2_bypass_mm_block(pipe, layer_idx: int):
    """
    Bypass a FLUX.2 double-stream (MM) block via monkey-patch.

    FluxTransformerBlock.forward() signature:
        (hidden_states, encoder_hidden_states, temb, ...) → (encoder_hidden_states, hidden_states)
    Note the swapped return order — the skip must preserve it.
    """
    block = pipe.transformer.transformer_blocks[layer_idx]
    original_forward = block.forward

    def _skip(hidden_states, encoder_hidden_states, temb=None, **kwargs):
        return encoder_hidden_states, hidden_states  # preserve return order

    block.forward = _skip
    try:
        yield
    finally:
        block.forward = original_forward


@contextlib.contextmanager
def _flux2_bypass_single_block(pipe, layer_idx: int):
    """
    Bypass a FLUX.2 single-stream block via monkey-patch.

    FluxSingleTransformerBlock.forward() signature:
        (hidden_states, temb, ...) → hidden_states
    """
    block = pipe.transformer.single_transformer_blocks[layer_idx]
    original_forward = block.forward

    def _skip(hidden_states, temb=None, **kwargs):
        return hidden_states

    block.forward = _skip
    try:
        yield
    finally:
        block.forward = original_forward


# ---------------------------------------------------------------------------
# Unified generation function
# ---------------------------------------------------------------------------

@torch.no_grad()
def generate_with_bypass(
    pipe,
    prompt: str,
    seed: int = 0,
    *,
    block_type: str = "mm",       # "mm" or "single"
    bypass_idx: Optional[int] = None,
    height: int = 1024,
    width: int = 1024,
    num_inference_steps: int = 28,
    guidance_scale: Optional[float] = None,
    device: str = "cuda",
) -> Image.Image:
    """
    Generate one image, optionally bypassing a single transformer block.

    Works for FLUX.1 (dev, schnell), FLUX.2-dev, and SD 3.5.

    Parameters
    ----------
    pipe          : loaded pipeline
    prompt        : text prompt
    seed          : RNG seed
    block_type    : "mm" (double-stream / joint) or "single" (FLUX only)
    bypass_idx    : block index to bypass; None = full model
    guidance_scale: override auto-detected default
    """
    model_type = detect_model_type(pipe)
    gs = guidance_scale if guidance_scale is not None else _default_guidance(pipe)

    generator = torch.Generator(device=device).manual_seed(seed)

    if model_type == "flux":
        # FLUX.1: local fork supports mm_skip_blocks / single_skip_blocks natively
        latents = torch.randn(
            (1, 4096, 64),
            generator=generator,
            device=device,
            dtype=pipe.transformer.dtype,
        )
        mm_skip     = [bypass_idx] if (bypass_idx is not None and block_type == "mm")     else None
        single_skip = [bypass_idx] if (bypass_idx is not None and block_type == "single") else None

        result = pipe(
            prompt,
            height=height,
            width=width,
            guidance_scale=gs,
            output_type="pil",
            num_inference_steps=num_inference_steps,
            max_sequence_length=512,
            latents=latents,
            mm_skip_blocks=mm_skip,
            single_skip_blocks=single_skip,
        )
        return result.images[0]

    elif model_type == "flux2":
        # FLUX.2: upstream Flux2Pipeline — use monkey-patch bypass
        if bypass_idx is not None and block_type == "mm":
            ctx = _flux2_bypass_mm_block(pipe, bypass_idx)
        elif bypass_idx is not None and block_type == "single":
            ctx = _flux2_bypass_single_block(pipe, bypass_idx)
        else:
            ctx = contextlib.nullcontext()

        with ctx:
            result = pipe(
                prompt=prompt,
                height=height,
                width=width,
                guidance_scale=gs,
                output_type="pil",
                num_inference_steps=num_inference_steps,
                generator=torch.Generator(device=device).manual_seed(seed),
            )
        return result.images[0]

    else:  # sd3
        # SD3 only has joint (MM-DiT) blocks — single-stream doesn't apply
        if bypass_idx is not None and block_type == "single":
            raise ValueError("SD 3.5 has no single-stream blocks. Use block_type='mm'.")

        ctx = (
            _sd3_bypass_block(pipe, bypass_idx)
            if bypass_idx is not None
            else contextlib.nullcontext()
        )
        with ctx:
            result = pipe(
                prompt,
                height=height,
                width=width,
                guidance_scale=gs,
                output_type="pil",
                num_inference_steps=num_inference_steps,
                generator=torch.Generator(device=device).manual_seed(seed),
            )
        return result.images[0]


# ---------------------------------------------------------------------------
# Pipeline loader
# ---------------------------------------------------------------------------

def _load_with_upstream_diffusers(model_path: str, hf_token: str,
                                   pipeline_class_name: str,
                                   torch_dtype=torch.float16,
                                   cpu_offload: bool = False,
                                   device: str = "cuda",
                                   cache_dir: str = "./models"):
    """
    Load a pipeline using the system (upstream) diffusers installation,
    bypassing the local fork in sys.path.

    Needed for models that require features the local fork doesn't have:
      FLUX.2-dev   — Flux2Pipeline + bfloat16
      SD 3.5 Large — qk_norm / norm_added_k
    Both use monkey-patch bypass so mm_skip_blocks is not needed.

    Device placement is done here, while upstream diffusers modules are still
    active in sys.modules — this avoids 'No module named diffusers.hooks' errors
    that occur when the pipe tries lazy imports after sys.modules is restored.
    On success, sys.path is restored but upstream diffusers stays in sys.modules
    so the pipe's methods can continue doing lazy imports normally.
    """
    # Find every entry in sys.path that has a diffusers install,
    # excluding the local fork. Works in Colab, venvs, and standard installs
    # without relying on site.getsitepackages() which can miss non-standard paths.
    sys_diff_dirs = [
        p for p in sys.path
        if p != _LOCAL_DIFFUSERS_SRC
        and os.path.isfile(os.path.join(p, "diffusers", "__init__.py"))
    ]
    if not sys_diff_dirs:
        raise RuntimeError(
            "System diffusers not found. Install it first:\n"
            "  pip install -U diffusers"
        )

    # Snapshot current state
    saved_path = list(sys.path)
    saved_mods = {k: v for k, v in sys.modules.items()
                  if k == "diffusers" or k.startswith("diffusers.")}

    # Build new sys.path: system diffusers dirs first, local fork removed
    other_paths = [p for p in sys.path
                   if p not in sys_diff_dirs
                   and not os.path.isfile(os.path.join(p, "diffusers", "__init__.py"))]
    sys.path[:] = sys_diff_dirs + other_paths
    for k in list(saved_mods):
        del sys.modules[k]

    try:
        import diffusers as _diffusers_upstream
        try:
            PipelineClass = getattr(_diffusers_upstream, pipeline_class_name)
        except (AttributeError, RuntimeError) as e:
            if "Mistral3ForConditionalGeneration" in str(e):
                raise RuntimeError(
                    "FLUX.2-dev requires transformers>=4.52 (Mistral3/Pixtral text encoder).\n"
                    "Upgrade with:\n"
                    "  pip install -U transformers\n"
                    "Then restart the Colab runtime and re-run."
                ) from e
            raise
        pipe = PipelineClass.from_pretrained(
            model_path,
            torch_dtype=torch_dtype,
            token=hf_token,
            cache_dir=cache_dir,
        )
        # Device placement must happen here, while upstream sys.modules is active.
        # enable_sequential_cpu_offload() does lazy imports from diffusers.hooks
        # which only exists in upstream diffusers — not in the local fork.
        if cpu_offload:
            pipe.enable_sequential_cpu_offload()
        else:
            pipe.to(device)

    except Exception:
        # On failure: fully restore sys.path and sys.modules
        sys.path[:] = saved_path
        for k in list(sys.modules.keys()):
            if k == "diffusers" or k.startswith("diffusers."):
                del sys.modules[k]
        sys.modules.update(saved_mods)
        raise

    # On success: restore sys.path so future imports still resolve to local fork,
    # but keep upstream diffusers in sys.modules so the pipe's lazy imports work.
    sys.path[:] = saved_path
    return pipe


def load_pipeline(model_path: str, hf_token: str, device: str = "cuda",
                  cpu_offload: bool = False, cache_dir: str = "./models"):
    """
    Auto-detect and load the correct pipeline for each model family.

      FLUX.1-dev / schnell  — FluxPipeline (local fork, mm_skip_blocks support)
      FLUX.2-dev            — Flux2Pipeline (upstream diffusers, bfloat16)
      SD 3.5 Large          — StableDiffusion3Pipeline (upstream if qk_norm missing)
    """
    name = model_path.lower()

    if "flux.2" in name or "flux2" in name:
        # FLUX.2-dev: go directly to upstream — device placement handled inside
        print("[load_pipeline] FLUX.2-dev detected — loading Flux2Pipeline via upstream diffusers.")
        return _load_with_upstream_diffusers(
            model_path, hf_token, "Flux2Pipeline",
            torch_dtype=torch.bfloat16, cpu_offload=cpu_offload, device=device,
            cache_dir=cache_dir,
        )

    elif "stable-diffusion-3" in name or "sd3" in name:
        from diffusers import StableDiffusion3Pipeline  # local fork
        # SD3.5 Large uses bfloat16; SD3-medium uses float16
        sd_dtype = torch.bfloat16 if "3.5" in name else torch.float16
        try:
            pipe = StableDiffusion3Pipeline.from_pretrained(
                model_path,
                torch_dtype=sd_dtype,
                token=hf_token,
                cache_dir=cache_dir,
            )
        except (ValueError, AttributeError) as exc:
            if "norm_added_k" not in str(exc) and "qk_norm" not in str(exc) and "no attribute" not in str(exc):
                raise
            print(
                "[load_pipeline] Local fork missing qk_norm support (SD 3.5 Large feature).\n"
                "  Retrying with system/upstream diffusers — monkey-patch bypass unaffected."
            )
            return _load_with_upstream_diffusers(
                model_path, hf_token, "StableDiffusion3Pipeline",
                torch_dtype=sd_dtype, cpu_offload=cpu_offload, device=device,
                cache_dir=cache_dir,
            )

    else:
        from diffusers import FluxPipeline, AutoencoderKL  # local fork
        # FLUX.1-dev, FLUX.1-schnell — use local fork for mm_skip_blocks support
        _w = [("ignore", r".*torch_dtype.*deprecated.*"), ("ignore", r".*Use `dtype` instead.*")]
        try:
            with warnings.catch_warnings():
                for _action, _msg in _w:
                    warnings.filterwarnings(_action, message=_msg)
                pipe = FluxPipeline.from_pretrained(
                    model_path,
                    torch_dtype=torch.float16,
                    token=hf_token,
                    cache_dir=cache_dir,
                )
        except AttributeError as exc:
            exc_str = str(exc)

            if "Mistral3ForConditionalGeneration" in exc_str:
                raise RuntimeError(
                    "FLUX.2-dev requires transformers>=4.52 (Mistral3/Pixtral text encoder).\n"
                    "The current transformers version is too old.\n\n"
                    "Upgrade with:\n"
                    "  pip install -U transformers\n"
                    "Then restart the runtime and re-run.\n\n"
                    "If the latest stable release still fails, try the dev build:\n"
                    "  pip install git+https://github.com/huggingface/transformers"
                ) from exc

            if "AutoencoderKLFlux2" not in exc_str:
                raise

            # AutoencoderKLFlux2 missing from local fork — pre-load VAE as plain AutoencoderKL
            print(
                "[load_pipeline] Local fork missing AutoencoderKLFlux2.\n"
                "  Retrying with VAE pre-loaded as AutoencoderKL (FLUX.2 workaround).\n"
                "  Layer-bypass comparisons remain valid; absolute image quality may differ slightly."
            )
            with warnings.catch_warnings():
                for _action, _msg in _w:
                    warnings.filterwarnings(_action, message=_msg)
                vae = AutoencoderKL.from_pretrained(
                    model_path,
                    subfolder="vae",
                    torch_dtype=torch.float16,
                    token=hf_token,
                    cache_dir=cache_dir,
                )
                pipe = FluxPipeline.from_pretrained(
                    model_path,
                    vae=vae,
                    torch_dtype=torch.float16,
                    token=hf_token,
                    cache_dir=cache_dir,
                )

    if cpu_offload:
        pipe.enable_sequential_cpu_offload()
    else:
        pipe.to(device)

    return pipe
