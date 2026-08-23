"""
UltimateFlux — unified training-free editing on FLUX.1-dev.

Synthesises StableFlow, FluxSpace, FreeFlux, and SVD-Style into a single
dual-branch pipeline with paper-validated layer sets for FLUX.1-dev
(§6–7 of Pipeline_Plan.md).

Supported tasks
---------------
  non_rigid          Change pose/action; preserve appearance (Task 1)
  object_replace     Swap an object in a masked region        (Task 3)
  bg_replace         Regenerate background; keep foreground   (Task 4)
  attr_edit          Disentangled single-attribute edit       (Task 5)
  style              Reference-image style personalization    (Task 7)

Usage — single run
------------------
  python NewWork/UltimateFlux/run_ultimateflux.py \\
      --task non_rigid \\
      --source_prompt "a bird perched on a branch" \\
      --edit_prompt   "a bird flying from the branch" \\
      --seed 42 --device cuda --save_images

  python NewWork/UltimateFlux/run_ultimateflux.py \\
      --task style \\
      --source_prompt "A cat" \\
      --edit_prompt   "A cat" \\
      --style_image   inputs/watercolor_ref.png \\
      --seed 42 --device cuda --save_images

Usage — config file (batch)
----------------------------
  python NewWork/UltimateFlux/run_ultimateflux.py \\
      --config prompts/ultimateflux_demo.json \\
      --hf_token "$HF_TOKEN" --device cuda --save_images
"""

import argparse
import json
import os
import sys
import warnings

# Harmless divide-by-zero from diffusers scheduler at the t=0 boundary timestep.
warnings.filterwarnings(
    "ignore", message="divide by zero encountered in divide",
    category=RuntimeWarning,
)

import torch
from PIL import Image

# Allow running from repo root or from this file's directory.
_REPO_ROOT = os.path.normpath(
    os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..")
)
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from NewWork.UltimateFlux.sampler import (
    load_pipeline, generate_dual_branch, generate_masked_delta_flow,
    TIER_A, TIER_B, N_DOUBLE, N_SINGLE,
)
from NewWork.UltimateFlux.policies import (
    NonRigidPolicy,
    ObjectAdditionPolicy,
    ObjectReplacementPolicy,
    BackgroundReplacePolicy,
    FineGrainedAttrPolicy,
    ColorCtrlPolicy,
    KontextColorPolicy,
    StylePersonalizationPolicy,
)


# ───────────────────────────── Policy factory ─────────────────────────────────

def build_policy(cfg: dict):
    """Construct the right policy from a run config dict."""
    task = cfg.get("task", "non_rigid")

    if task == "non_rigid":
        fg_mask = None
        if cfg.get("fg_mask"):
            fg_mask = Image.open(cfg["fg_mask"]).convert("L")
        return NonRigidPolicy(
            inject_steps_frac=tuple(cfg["inject_steps_frac"]) if cfg.get("inject_steps_frac") else (0.0, 1.0),
            v_blend=cfg.get("v_blend", 0.0),
            v_blend_steps_frac=tuple(cfg["v_blend_steps_frac"]) if cfg.get("v_blend_steps_frac") else (0.0, 1.0),
            preserve_color=cfg.get("preserve_color", False),
            synps=cfg.get("synps", True),
            m_min=cfg.get("m_min", 0.7),
            m_max=cfg.get("m_max", 0.95),
            static_w=cfg.get("static_w", None),
            identity_guidance=cfg.get("identity_guidance", False),
            identity_strength=cfg.get("identity_strength", 0.3),
            identity_steps_frac=tuple(cfg["identity_steps_frac"]) if cfg.get("identity_steps_frac") else (0.0, 0.5),
            low_freq_cutoff=cfg.get("low_freq_cutoff", 0.1),
            fg_mask=fg_mask,
            subject_noun=cfg.get("subject_noun", None),
            use_sam2=cfg.get("use_sam2_nonrigid", False),
            sam2_model_id=cfg.get("sam2_model_id", "facebook/sam2-hiera-large"),
            bg_dilate=cfg.get("bg_dilate", 6),
            inject_all_single=cfg.get("inject_all_single", False),
            bg_steps_frac=tuple(cfg["bg_steps_frac"]) if cfg.get("bg_steps_frac") else (0.0, 1.0),
        )

    if task == "object_add":
        mask = None
        if cfg.get("placement_mask"):
            mask = Image.open(cfg["placement_mask"]).convert("L")
        return ObjectAdditionPolicy(
            added_word=cfg.get("added_word", None),
            placement_mask=mask,
            inject_steps_frac=tuple(cfg["inject_steps_frac"]) if cfg.get("inject_steps_frac") else (0.0, 1.0),
            derive_step=cfg.get("derive_step", 7),
            top_k_frac=cfg.get("top_k_frac", 0.15),
        )

    if task == "object_replace":
        mask = None
        if cfg.get("mask_image"):
            mask = Image.open(cfg["mask_image"]).convert("L")
        return ObjectReplacementPolicy(
            mask=mask,
            inject_steps_frac=tuple(cfg["inject_steps_frac"]) if cfg.get("inject_steps_frac") else (0.0, 0.9),
        )

    if task == "bg_replace":
        fg_mask = None
        if cfg.get("fg_mask_image"):
            fg_mask = Image.open(cfg["fg_mask_image"]).convert("L")
        return BackgroundReplacePolicy(
            fg_mask=fg_mask,
            use_sam2=cfg.get("use_sam2", True),
            sam2_model_id=cfg.get("sam2_model_id", "facebook/sam2-hiera-large"),
        )

    if task == "attr_edit":
        inject_layers_raw = cfg.get("inject_layers", None)
        if inject_layers_raw in ("color", "colour"):
            # Kontext-style colour editing: Q+K injection in all 19 double-stream
            # blocks locks spatial structure (face, car shape); 38 single-stream
            # blocks run fully free so the edit text ("blonde", "blue") drives
            # colour via V in the joint attention.  Uses generate_dual_branch.
            # Optional PFB (svd_alpha > 0): blends source principal features into
            # edit's residual for extra identity anchoring.
            return KontextColorPolicy(
                qk_layers=list(range(N_DOUBLE)),
                k_only_layers=list(range(N_DOUBLE, N_DOUBLE + N_SINGLE)),
                qk_steps_frac=tuple(cfg.get("qk_steps_frac", [0.0, 1.0])),
                k_only_steps_frac=tuple(cfg.get("k_only_steps_frac", [0.0, 1.0])),
                ss_q_steps_frac=tuple(cfg.get("ss_q_steps_frac", [0.0, 0.5])),
                svd_alpha=cfg.get("svd_alpha", 0.0),
                svd_layers=cfg.get("svd_layers", [1]),
                svd_steps_frac=tuple(cfg.get("svd_steps_frac", [0.0, 0.25])),
            )
        elif inject_layers_raw == "double_stream":
            # Lock all 19 double-stream (joint text-image) blocks; 38 single-stream free.
            # key_only=True (default): inject only K into double-stream blocks.
            #   K locks SPATIAL POSITIONS (structure preserved, no forehead drift).
            #   V stays target-conditioned ("blonde"/"blue") → full colour change.
            # key_only=False: inject K+V (stronger structure lock, weaker colour change).
            return FineGrainedAttrPolicy(
                inject_layers=list(range(N_DOUBLE)),  # _DOUBLE_STREAM: 0-18
                inject_steps_frac=tuple(cfg["inject_steps_frac"]) if cfg.get("inject_steps_frac") else (0.0, 1.0),
                key_only=cfg.get("key_only", True),
            )
        elif inject_layers_raw == "tier_a":
            inject_layers = TIER_A          # K+V — breed/shape change
        else:
            inject_layers = None            # default → _PRESERVE_LAYERS, K+V
        return FineGrainedAttrPolicy(
            inject_layers=inject_layers,
            inject_steps_frac=tuple(cfg["inject_steps_frac"]) if cfg.get("inject_steps_frac") else (0.0, 1.0),
        )

    if task == "style":
        style_img   = None
        content_img = None
        if cfg.get("style_image"):
            style_img = Image.open(cfg["style_image"]).convert("RGB")
        if cfg.get("content_image") and os.path.isfile(cfg["content_image"]):
            content_img = Image.open(cfg["content_image"]).convert("RGB")
        # content_strength default: 0.6 when auto-generating source (strong identity),
        # 0.85 when user supplies their own content_image (more style freedom).
        _default_cs = 0.6 if not cfg.get("content_image") else 0.85
        # T2I (no content_image): style V is extracted at sigma≈0.3, which corresponds
        # to roughly the 70-80 % mark of FLUX's shifted schedule for 28 steps at 1024px.
        # Injecting in ALL steps means we inject V from sigma=0.3 into steps where the
        # model is at sigma=1.0, 0.9, … — the feature spaces are incompatible there.
        # Limit to the second half of steps where sigma is close to 0.3.
        # content_image mode: sigma is already matched (denoising starts from sigma_start),
        # so inject from step 0.
        if cfg.get("inject_steps_frac"):
            _inject_frac = tuple(cfg["inject_steps_frac"])
        elif cfg.get("content_image"):
            _inject_frac = (0.0, 1.0)
        else:
            _inject_frac = (0.5, 1.0)
        return StylePersonalizationPolicy(
            style_image             = style_img,
            content_image           = content_img,
            style_strength          = cfg.get("style_strength",          1.0),
            content_strength        = cfg.get("content_strength") or _default_cs,
            inject_steps_frac       = _inject_frac,
            color_transfer_strength = cfg.get("color_transfer_strength", 0.6),
            style_description       = cfg.get("style_description",       ""),
        )

    raise ValueError(
        f"Unknown task '{task}'. "
        "Choose from: non_rigid, object_replace, bg_replace, attr_edit, style"
    )


# ───────────────────────────── Single run ─────────────────────────────────────

def run_single(pipe, cfg: dict, out_dir: str, save_images: bool, device: str):
    name               = cfg.get("name", "output")
    source_prompt      = cfg.get("source_prompt", cfg.get("prompt", ""))
    edit_prompt        = cfg.get("edit_prompt",   cfg.get("prompt", ""))
    seed               = cfg.get("seed", 42)
    num_steps          = cfg.get("num_steps", 28)
    guidance           = cfg.get("guidance_scale", 3.5)
    height             = cfg.get("height", 1024)
    width              = cfg.get("width",  1024)
    max_seq_len        = cfg.get("max_sequence_length", 512)
    explicit_intermediates = cfg.get("save_intermediates", False)
    save_intermediates     = explicit_intermediates or save_images  # steps.png whenever saving
    intermediate_every     = cfg.get("intermediate_every", 4)
    delta_scale        = cfg.get("delta_scale", 2.0)
    delta_start_step   = cfg.get("delta_start_step", None)

    policy = build_policy(cfg)

    # Always create run_dir when saving anything (finals or intermediates).
    run_dir = os.path.join(out_dir, name)
    if save_images or save_intermediates:
        os.makedirs(run_dir, exist_ok=True)

    print(f"\n[UltimateFlux] '{name}' | task={cfg.get('task','non_rigid')} | seed={seed}")
    print(f"  source_prompt: {source_prompt}")
    print(f"  edit_prompt:   {edit_prompt}")

    # save_strips=True only for explicit --save_intermediates; steps.png always saved.
    save_strips = explicit_intermediates

    _stage1_source: Image.Image = None

    # ColorCtrlPolicy (legacy) uses masked delta-flow.
    # All other policies (including KontextColorPolicy) use dual-branch attention injection.
    if isinstance(policy, ColorCtrlPolicy) and not isinstance(policy, KontextColorPolicy):
        print(f"  [route] generate_masked_delta_flow (delta_scale={delta_scale})")
        src_img, edit_img = generate_masked_delta_flow(
            pipe=pipe,
            policy=policy,
            source_prompt=source_prompt,
            edit_prompt=edit_prompt,
            seed=seed,
            num_steps=num_steps,
            guidance_scale=guidance,
            height=height,
            width=width,
            max_sequence_length=max_seq_len,
            device=device,
            delta_scale=delta_scale,
            delta_start_step=delta_start_step,
            save_intermediates=save_intermediates,
            intermediate_out_dir=run_dir if save_intermediates else None,
            intermediate_every=intermediate_every,
            save_strips=save_strips,
        )
    else:
        src_img, edit_img = generate_dual_branch(
            pipe=pipe,
            policy=policy,
            source_prompt=source_prompt,
            edit_prompt=edit_prompt,
            seed=seed,
            num_steps=num_steps,
            guidance_scale=guidance,
            height=height,
            width=width,
            max_sequence_length=max_seq_len,
            device=device,
            save_intermediates=save_intermediates,
            intermediate_out_dir=run_dir if save_intermediates else None,
            intermediate_every=intermediate_every,
            save_strips=save_strips,
        )

    # For two-stage style: source.png = clean stage-1 image (not stage-2 reconstruction).
    _display_src = _stage1_source if _stage1_source is not None else src_img

    if save_images:
        src_path  = os.path.join(run_dir, "source.png")
        edit_path = os.path.join(run_dir, "edit.png")
        _display_src.save(src_path)
        edit_img.save(edit_path)
        print(f"  Saved → {src_path}")
        print(f"  Saved → {edit_path}")

        # Side-by-side comparison.
        # For style task: [style_ref | source | styled] so the reference is visible.
        panels = [_display_src, edit_img]
        style_ref_path = cfg.get("style_image")
        if cfg.get("task") == "style" and style_ref_path:
            ref = Image.open(style_ref_path).convert("RGB").resize(
                (_display_src.width, _display_src.height), Image.LANCZOS)
            panels = [ref, _display_src, edit_img]

        comp_w = sum(p.width for p in panels)
        comp_h = max(p.height for p in panels)
        comp   = Image.new("RGB", (comp_w, comp_h))
        x = 0
        for panel in panels:
            comp.paste(panel, (x, 0))
            x += panel.width
        comp_path = os.path.join(run_dir, "compare.png")
        comp.save(comp_path)
        print(f"  Saved → {comp_path}")

    return src_img, edit_img


# ───────────────────────────── CLI ────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(description="UltimateFlux unified editing pipeline")

    # Model / infra
    p.add_argument("--model_path",  default="black-forest-labs/FLUX.1-dev")
    p.add_argument("--hf_token",    default=os.environ.get("HF_TOKEN"))
    p.add_argument("--device",      default="cuda")
    p.add_argument("--cpu_offload", action="store_true")
    p.add_argument("--cache_dir",   default="./models")
    p.add_argument("--out_dir",     default="results/ultimateflux")
    p.add_argument("--save_images", action="store_true")
    p.add_argument("--save_intermediates", action="store_true", default=False,
                   help="Save source/edit/compare intermediate step grids alongside final images")
    p.add_argument("--intermediate_every", type=int, default=4,
                   help="Capture a frame every N denoising steps (--save_intermediates)")

    # Config file (batch mode)
    p.add_argument("--config", default=None, help="Path to JSON config for batch runs")

    # Single-run overrides
    p.add_argument("--task",          default="non_rigid",
                   choices=["non_rigid", "object_add", "object_replace", "bg_replace", "attr_edit", "style"])
    p.add_argument("--name",          default="output")
    p.add_argument("--source_prompt", default="")
    p.add_argument("--edit_prompt",   default="")
    p.add_argument("--prompt",        default=None, help="Sets both source and edit prompt")
    p.add_argument("--style_image",   default=None)
    p.add_argument("--added_word",    default=None, help="Word for new object (object_add), e.g. 'vase'")
    p.add_argument("--placement_mask",default=None, help="Placement region mask (object_add)")
    p.add_argument("--mask_image",    default=None, help="Object region mask (object_replace)")
    p.add_argument("--fg_mask_image", default=None, help="Foreground mask (bg_replace)")
    p.add_argument("--use_sam2",      action="store_true", default=True,
                   help="Auto-segment foreground with SAM2 when no fg_mask_image given (bg_replace)")
    p.add_argument("--no_sam2",       dest="use_sam2", action="store_false",
                   help="Disable SAM2; fall back to TIER_A global injection (bg_replace)")
    p.add_argument("--sam2_model_id", default="facebook/sam2-hiera-large",
                   help="HuggingFace SAM2 model ID (bg_replace)")
    p.add_argument("--inject_steps_frac", type=float, nargs=2, default=None,
                   metavar=("START", "END"),
                   help="Non-rigid / attr_edit: fraction of denoising steps to apply injection. "
                        "Default depends on task (non_rigid: 0.0 1.0, attr_edit: 0.0 1.0).")
    p.add_argument("--synps", dest="synps", action="store_true", default=True,
                   help="Non-rigid: enable SynPS adaptive w-scaled RoPE injection at TIER_A "
                        "(CVPR 2026, arXiv:2512.14423). Default on. Replaces v_blend for colour preservation.")
    p.add_argument("--no_synps", dest="synps", action="store_false",
                   help="Non-rigid: disable SynPS; fall back to pure FreeFlux K+V at TIER_A "
                        "(no colour preservation unless --v_blend > 0).")
    p.add_argument("--m_min", type=float, default=0.7,
                   help="SynPS: lower M_t threshold — below this, w=1 (full spatial lock). Default 0.7.")
    p.add_argument("--m_max", type=float, default=0.95,
                   help="SynPS: upper M_t threshold — above this, w=0 (position-agnostic). Default 0.95.")
    p.add_argument("--static_w", type=float, default=None,
                   help="SynPS: bypass adaptive M_t; use this fixed w at all TIER_A steps. "
                        "0.0 = fully position-agnostic (max colour retrieval); "
                        "1.0 = full RoPE (FreeFlux baseline). "
                        "Try 0.0 first if identity is not fully preserved.")
    p.add_argument("--fg_mask", default=None,
                   help="Non-rigid: path to foreground mask image (white=subject to edit). "
                        "Activates KV-Edit masked mode. Highest priority.")
    p.add_argument("--subject_noun", default=None,
                   help="Non-rigid: subject word in the source prompt, e.g. 'bird', 'cat', "
                        "'golden retriever'. Enables ConceptAttention masking (arXiv:2502.04320): "
                        "uses FLUX's own attention output projections in layers 8-17 to locate "
                        "the subject — no SAM2 or external model needed. Semantically accurate "
                        "because it finds the named concept, not just the largest blob.")
    p.add_argument("--use_sam2_nonrigid", action="store_true", default=False,
                   help="Non-rigid: auto-segment subject with SAM2 before generation. "
                        "Generates a source preview, runs SAM2, builds a foreground mask. "
                        "Activates KV-Edit masked mode automatically.")
    p.add_argument("--no_use_sam2_nonrigid", dest="use_sam2_nonrigid", action="store_false",
                   help="Non-rigid: disable SAM2 auto-segmentation (default: off).")
    p.add_argument("--bg_dilate", type=int, default=6,
                   help="Non-rigid masked mode: token-space dilation radius for soft bg composite. "
                        "Expands the 'safe fg' zone so the edited pose can extend beyond the "
                        "source mask boundary without ghost traces (default 6 = ~96px at 1024px). "
                        "Increase if ghost traces remain; decrease for tighter bg lock.")
    p.add_argument("--identity_guidance", dest="identity_guidance",
                   action="store_true", default=False,
                   help="Non-rigid: after each denoising step, blend low-frequency FFT components "
                        "of source latent into edit latent to anchor global colour. "
                        "Complementary to SynPS — SynPS acts at attention level, this acts at latent level.")
    p.add_argument("--no_identity_guidance", dest="identity_guidance", action="store_false",
                   help="Non-rigid: disable FFT identity guidance (default).")
    p.add_argument("--identity_strength", type=float, default=0.3,
                   help="Non-rigid: FFT blend strength (0=no effect, 1=full source colour). Default 0.3.")
    p.add_argument("--identity_steps_frac", type=float, nargs=2, default=[0.0, 0.5],
                   metavar=("START", "END"),
                   help="Non-rigid: step window for FFT identity guidance. "
                        "Default first 50%% of steps — colour anchor active early, "
                        "high-freq pose detail refined freely in later steps.")
    p.add_argument("--low_freq_cutoff", type=float, default=0.1,
                   help="Non-rigid: fraction of FFT spatial frequencies treated as low-freq "
                        "(0.1 = lowest 10%% = global colour/brightness). Higher = more structure "
                        "transferred, but may suppress pose change.")
    p.add_argument("--v_blend", type=float, default=0.0,
                   help="Non-rigid: source V blend weight at non-TIER_A layers (default 0.0 with SynPS on). "
                        "V_edit = v_blend*V_src + (1-v_blend)*V_edit_orig. "
                        "Only needed when --no_synps; try 0.3 in that case. "
                        "WARNING: values above 0 at all 44 non-TIER_A layers can suppress pose change.")
    p.add_argument("--v_blend_steps_frac", type=float, nargs=2, default=None,
                   metavar=("START", "END"),
                   help="Non-rigid: step window for non-TIER_A V blend (default: all steps). "
                        "Use '0.0 0.3' to limit V injection to first 15 of 50 steps. "
                        "Colour is established early; later steps refine pose without interference.")
    p.add_argument("--preserve_color", dest="preserve_color", action="store_true", default=False,
                   help="Non-rigid: apply Reinhard LAB colour transfer after generation. "
                        "Off by default (v_blend handles colour during generation). "
                        "Enable as an additional fallback if v_blend alone is insufficient.")
    p.add_argument("--inject_all_single", action="store_true", default=True,
                   help="Non-rigid: also inject K,V at ALL single-stream layers to lock background. Default True.")
    p.add_argument("--no_inject_all_single", dest="inject_all_single", action="store_false",
                   help="Non-rigid: only inject at TIER_A layers (disable background locking).")
    p.add_argument("--bg_steps_frac", type=float, nargs=2, default=None,
                   metavar=("START", "END"),
                   help="Non-rigid: step window for background-lock (all-single-stream) injection. "
                        "Default 0.5 1.0 — starts at halfway so early steps can establish the new pose. "
                        "Lower START for stronger background lock; raise for more pose freedom.")
    p.add_argument("--inject_layers", default=None,
                   choices=["color", "double_stream", "tier_a"],
                   help="attr_edit mode: color=Kontext Q+K injection (colour change), double_stream=K+V in 19 joint blocks, tier_a=breed/shape change")
    p.add_argument("--key_only", action="store_true", default=False,
                   help="double_stream mode: inject K only (not V) — locks spatial layout without locking colour. Default True for double_stream via config.")
    p.add_argument("--reweight_scale", type=float, default=1.0,
                   help="color mode: multiply image→colour-word attention scores by this factor pre-softmax (§3.5). 1.0=disabled; try 3.0–5.0 for strong colour change.")
    p.add_argument("--ds_key_inject", action="store_true", default=False,
                   help="color mode: also inject K in double-stream blocks to prevent face-structure drift when qk_frac=0.")
    p.add_argument("--top_k_frac",  type=float, default=0.2,
                   help="ColorCtrl: fraction of image tokens treated as editing region (default 0.2)")
    p.add_argument("--qk_frac",    type=float, default=1.0,
                   help="ColorCtrl: fraction of steps for structure preservation / v-v score injection")
    p.add_argument("--v_frac",     type=float, default=1.0,
                   help="ColorCtrl: fraction of steps for colour preservation / V masking")
    p.add_argument("--color_word", default=None,
                   help="ColorCtrl: colour word in edit prompt (e.g. 'blonde', 'blue'); "
                        "focuses the editing-region mask on that word's T5 tokens")
    p.add_argument("--chunk_size", type=int, default=4,
                   help="ColorCtrl: heads per chunk in manual attention (reduce to 2/1 for OOM)")
    p.add_argument("--color_sam2", action="store_true", default=False,
                   help="color mode: use SAM2 point-prompted segmentation for the editing mask "
                        "instead of attention topk. Requires: pip install sam2")
    p.add_argument("--color_sam2_model", default="facebook/sam2-hiera-large",
                   help="HuggingFace SAM2 model ID for color mask segmentation")
    p.add_argument("--mask_build_step", type=int, default=5,
                   help="ColorCtrl: denoising step at which to build the editing-region mask "
                        "(steps 0..N-1 run freely; mask built at N, ColorCtrl from N..end). "
                        "Larger = more accurate mask but less structure locked early.")
    p.add_argument("--qk_steps_frac", type=float, nargs=2, default=[0.0, 1.0],
                   help="color mode: [start end] fraction of steps for double-stream Q+K injection.")
    p.add_argument("--k_only_steps_frac", type=float, nargs=2, default=[0.0, 1.0],
                   help="color mode: step fraction for single-stream K injection (always-on position lock).")
    p.add_argument("--ss_q_steps_frac", type=float, nargs=2, default=[0.0, 0.5],
                   help="color mode: step fraction for single-stream Q injection (identity lock). "
                        "Default first 50%% of steps. Raise end toward 1.0 for tighter identity; "
                        "lower end toward 0.0 for stronger colour.")
    p.add_argument("--svd_alpha",   type=float, default=0.0,
                   help="color mode: PFB alpha for optional SVD block hook (0 = disabled). "
                        "Enable (e.g. 1.0) to blend source structural features into edit's residual.")
    p.add_argument("--svd_layers",  type=int, nargs="+", default=[1],
                   help="color mode: block indices for PFB SVD hook (default: [1])")
    p.add_argument("--svd_steps_frac", type=float, nargs=2, default=[0.0, 0.25],
                   help="color mode: step fraction for PFB SVD hook (default first 25%)")
    p.add_argument("--color_structure_frac", type=float, default=0.0,
                   help="Unused; kept for backward compat")
    p.add_argument("--inject_frac", type=float, default=0.5,
                   help="Unused; kept for backward compat")
    p.add_argument("--pfb_step",   type=int,   default=3,
                   help="Unused; kept for backward compat")
    p.add_argument("--pfb_alpha",   type=float, default=1.0,
                   help="Unused; kept for backward compat")
    p.add_argument("--style_strength", type=float, default=1.0,
                   help="Style task: K,V blend weight (0.0=no injection, 1.0=full style). Default 1.0.")
    p.add_argument("--style_description", default="",
                   help="Style task: text description of the style (e.g. 'oil painting'). "
                        "Used for attention-based style-patch selection during extraction. "
                        "Empty = extract all patches uniformly (previous behaviour).")
    p.add_argument("--color_transfer_strength", type=float, default=0.6,
                   help="Style task: LAB histogram matching strength applied as post-processing "
                        "(0.0=off, 1.0=full palette replacement). Default 0.6. "
                        "Transfers style reference colour palette onto edit image after generation.")
    p.add_argument("--content_image", type=str, default=None,
                   help="Style task: source image whose identity to preserve. "
                        "When provided, both branches start from this image's noisy latent "
                        "(img2img mode) instead of random noise.")
    p.add_argument("--content_strength", type=float, default=None,
                   help="Style task: noise fraction added to source latent (0,1). "
                        "Two-stage default 0.6 (strong identity). Explicit --content_image default 0.85. "
                        "Lower = stronger identity; higher = more style freedom.")
    p.add_argument("--q_preservation", type=float, default=1.0,
                   help="Deprecated, ignored.")
    p.add_argument("--delta_scale", type=float, default=2.0,
                   help="Unused; kept for backward compat (legacy masked-delta-flow)")
    p.add_argument("--delta_start_step", type=int, default=None,
                   help="Unused; kept for backward compat (legacy masked-delta-flow)")
    p.add_argument("--seed",          type=int, default=42)
    p.add_argument("--num_steps",     type=int, default=28)
    p.add_argument("--guidance_scale",type=float, default=3.5)
    p.add_argument("--height",        type=int, default=1024)
    p.add_argument("--width",         type=int, default=1024)

    return p.parse_args()


def main():
    args = parse_args()

    # ── Load pipeline ──
    print(f"[UltimateFlux] Loading {args.model_path} …")
    pipe = load_pipeline(
        model_path=args.model_path,
        hf_token=args.hf_token,
        device=args.device,
        cpu_offload=args.cpu_offload,
        cache_dir=args.cache_dir,
    )
    print(f"[UltimateFlux] Model loaded on {args.device}")

    os.makedirs(args.out_dir, exist_ok=True)

    # ── Config file (batch) ──
    if args.config:
        with open(args.config) as f:
            config = json.load(f)

        global_cfg = config.get("global", {})
        runs       = config.get("runs", [])

        for run_cfg in runs:
            merged = {**global_cfg, **run_cfg}
            run_single(pipe, merged, args.out_dir, args.save_images, args.device)

        return

    # ── Single run ──
    source_prompt = args.prompt if args.prompt else args.source_prompt
    edit_prompt   = args.prompt if args.prompt else args.edit_prompt

    single_cfg = {
        "name":          args.name,
        "task":          args.task,
        "source_prompt": source_prompt,
        "edit_prompt":   edit_prompt,
        "style_image":    args.style_image,
        "added_word":     args.added_word,
        "placement_mask": args.placement_mask,
        "mask_image":     args.mask_image,
        "fg_mask_image":  args.fg_mask_image,
        "use_sam2":       args.use_sam2,
        "sam2_model_id":  args.sam2_model_id,
        "inject_steps_frac":    args.inject_steps_frac,
        "fg_mask":              args.fg_mask,
        "subject_noun":         args.subject_noun,
        "use_sam2_nonrigid":    args.use_sam2_nonrigid,
        "bg_dilate":            args.bg_dilate,
        "synps":                args.synps,
        "m_min":                args.m_min,
        "m_max":                args.m_max,
        "static_w":             args.static_w,
        "identity_guidance":    args.identity_guidance,
        "identity_strength":    args.identity_strength,
        "identity_steps_frac":  args.identity_steps_frac,
        "low_freq_cutoff":      args.low_freq_cutoff,
        "v_blend":              args.v_blend,
        "v_blend_steps_frac":   args.v_blend_steps_frac,
        "preserve_color":       args.preserve_color,
        "inject_all_single":    args.inject_all_single,
        "bg_steps_frac":        args.bg_steps_frac,
        "inject_layers":        args.inject_layers,
        "key_only":             args.key_only,
        "reweight_scale":       args.reweight_scale,
        "ds_key_inject":        args.ds_key_inject,
        "top_k_frac":           args.top_k_frac,
        "qk_frac":              args.qk_frac,
        "v_frac":               args.v_frac,
        "color_word":           args.color_word,
        "chunk_size":           args.chunk_size,
        "mask_build_step":      args.mask_build_step,
        "color_sam2":           args.color_sam2,
        "color_sam2_model":     args.color_sam2_model,
        "qk_steps_frac":        args.qk_steps_frac,
        "k_only_steps_frac":    args.k_only_steps_frac,
        "ss_q_steps_frac":      args.ss_q_steps_frac,
        "svd_alpha":            args.svd_alpha,
        "svd_layers":           args.svd_layers,
        "svd_steps_frac":       args.svd_steps_frac,
        "color_structure_frac": args.color_structure_frac,
        "inject_frac":          args.inject_frac,
        "pfb_step":             args.pfb_step,
        "pfb_alpha":            args.pfb_alpha,
        "style_strength":             args.style_strength,
        "style_description":          args.style_description,
        "color_transfer_strength":    args.color_transfer_strength,
        "content_image":              args.content_image,
        "content_strength":           args.content_strength,
        "q_preservation":       args.q_preservation,
        "delta_scale":          args.delta_scale,
        "delta_start_step":     args.delta_start_step,
        "seed":               args.seed,
        "num_steps":          args.num_steps,
        "guidance_scale":     args.guidance_scale,
        "height":             args.height,
        "width":              args.width,
        "save_intermediates": args.save_intermediates,
        "intermediate_every": args.intermediate_every,
    }
    run_single(pipe, single_cfg, args.out_dir, args.save_images, args.device)


if __name__ == "__main__":
    main()
