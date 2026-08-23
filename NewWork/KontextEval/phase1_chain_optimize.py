"""
Phase 1 — Chain Optimization: Preserve Bicycle Identity During Vase Addition

Conclusion from latent composition experiments (phases 1_intuition through 1_clean_mask):
  Latent/pixel-space composition fundamentally fails for this scene:
  - The bicycle and vase overlap in 2D (bike is partially behind the coffee table area)
  - Any hard-copy boundary in latent space decodes as distorted pixels
  - The vase image (generated from base, no bike) doesn't know the bike exists
    → pasting vase pixels into img_bike erases the bike in the overlap region
  - 4 separate masking strategies all failed for the same underlying reason

The chain (base → bike → vase) is the correct architecture.
Kontext step 2 sees the bicycle in its reference image and preserves it by default.

The question now: can we make step 2 change the bicycle LESS while still adding the vase?

Two strategies
--------------
  A. Conservative generation settings in step 2:
     Fewer steps (5-10) and lower guidance (1.0-2.0) = Kontext deviates less
     from the input, adding the vase more minimally.

  B. Stronger preservation prompt in step 2:
     Explicit mention of the bicycle in the vase-addition prompt forces Kontext
     to reason about the bike as a preserved element, not just background.

  C. Combined: strong prompt + conservative settings

Metric: bicycle identity preservation
  For each step-2 result, compute pixel diff vs img_bike in the bicycle region.
  Lower diff = bicycle changed less = better identity preservation.
  The bike region is defined by pixel diff between base and img_bike (threshold=40).

Experiments
-----------
  A. Step count sweep: 5 / 10 / 15 / 28 steps (guidance=2.5, standard prompt)
  B. Guidance sweep: 1.0 / 1.5 / 2.0 / 2.5 (steps=10, standard prompt)
  C. Prompt comparison: standard vs preservation-explicit prompt (steps=10, g=2.0)
  D. Best combination vs original chain baseline
  E. SSIM/pixel-diff table to quantify bicycle preservation numerically

Usage
-----
python NewWork/KontextEval/phase1_chain_optimize.py \\
    --hf_token $HF_TOKEN \\
    --cache_dir ./models \\
    --out_dir results/phase1_chain_opt
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from NewWork.KontextEval.utils.model_utils import load_kontext_pipeline, generate


# ============================================================
# Prompts
# ============================================================

BASE_PROMPT = "A modern living room with a sofa and a wooden coffee table."

BICYCLE_PROMPT = (
    "Add a yellow bicycle leaning against the wall on the left side. "
    "Keep the rest of the room exactly the same."
)

# Standard chain step-2 prompt (what we've been using all along)
VASE_PROMPT_STANDARD = (
    "Add a white ceramic vase with flowers on the coffee table. "
    "Keep the rest of the room exactly the same."
)

# Preservation-explicit prompt: explicitly names the bicycle so Kontext treats it as preserved
VASE_PROMPT_PRESERVE = (
    "Add a white ceramic vase with vivid flowers on the coffee table. "
    "The yellow bicycle leaning against the left wall must stay exactly as it is — "
    "same position, same color, same shape, do not change it at all."
)

# Descriptive prompt: describes the full desired scene (Kontext has full context)
VASE_PROMPT_DESCRIBE = (
    "This room has a sofa, a coffee table, and a yellow bicycle leaning against the left wall. "
    "Now also add a white ceramic vase with vivid flowers placed on the coffee table. "
    "Everything else, especially the yellow bicycle, remains completely unchanged."
)


# ============================================================
# VAE helpers (for measuring bicycle change)
# ============================================================

@torch.no_grad()
def vae_encode(pipe, pil_img: Image.Image) -> torch.Tensor:
    t = pipe.image_processor.preprocess(pil_img).to(pipe.device, pipe.vae.dtype)
    raw = pipe.vae.encode(t).latent_dist.mean
    return ((raw - pipe.vae.config.shift_factor) * pipe.vae.config.scaling_factor).cpu()


def pixel_diff_mask_img(img_a: Image.Image, img_b: Image.Image,
                        threshold: float = 40.0) -> np.ndarray:
    """Binary (H, W) mask: where pixel diff exceeds threshold. Image space."""
    a = np.array(img_a).astype(np.float32)
    b = np.array(img_b).astype(np.float32)
    diff = np.abs(b - a).mean(axis=2)
    return diff >= threshold


def bicycle_change_score(img_bike: Image.Image, result: Image.Image,
                         bike_mask: np.ndarray) -> dict:
    """
    Measure how much the bicycle changed between img_bike and result.
    Lower = bicycle was preserved better.
    """
    a = np.array(img_bike).astype(np.float32)
    b = np.array(result).astype(np.float32)
    diff = np.abs(b - a).mean(axis=2)  # (H, W) per-pixel mean abs diff

    bike_diff = diff[bike_mask].mean() if bike_mask.sum() > 0 else 0.0
    global_diff = diff.mean()
    return {
        "bike_diff":   round(float(bike_diff), 2),   # lower = bike preserved better
        "global_diff": round(float(global_diff), 2), # how much image changed overall
        "bike_pct":    round(float(bike_mask.mean()) * 100, 1),
    }


# ============================================================
# Grid helper
# ============================================================

def save_grid(images, titles, path, ncols=None):
    n = len(images)
    ncols = ncols or n
    nrows = (n + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(5 * ncols, 5 * nrows))
    axes = [axes] if n == 1 else list(axes.flat)
    for ax, img, title in zip(axes, images, titles):
        ax.imshow(img)
        ax.axis("off")
        ax.set_title(title, fontsize=8)
    for ax in axes[n:]:
        ax.axis("off")
    plt.tight_layout()
    plt.savefig(path, dpi=100, bbox_inches="tight")
    plt.close(fig)


# ============================================================
# Args
# ============================================================

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--hf_token",  required=True)
    p.add_argument("--cache_dir", default="./models")
    p.add_argument("--out_dir",   default="results/phase1_chain_opt")
    p.add_argument("--seed",      type=int,   default=42)
    p.add_argument("--height",    type=int,   default=1024)
    p.add_argument("--width",     type=int,   default=1024)
    p.add_argument("--device",    default="cuda")
    return p.parse_args()


# ============================================================
# Main
# ============================================================

def main():
    args = parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    def save(img, name):
        img.save(os.path.join(args.out_dir, f"{name}.png"))

    def run(src, prompt, steps=28, guidance=2.5):
        return generate(
            pipe, prompt, src,
            seed=args.seed,
            num_steps=steps,
            guidance_scale=guidance,
            height=args.height, width=args.width,
        )

    print("Loading FLUX.1-Kontext-dev …")
    pipe = load_kontext_pipeline(
        hf_token=args.hf_token, device=args.device, cache_dir=args.cache_dir,
    )

    # ----------------------------------------------------------
    # Setup: base + step-1 bicycle (shared for all experiments)
    # ----------------------------------------------------------
    print("\n=== Setup ===")
    grey = Image.new("RGB", (args.width, args.height), (200, 200, 190))
    base = run(grey, BASE_PROMPT, steps=28, guidance=2.5)
    save(base, "base")

    img_bike = run(base, BICYCLE_PROMPT, steps=28, guidance=2.5)
    save(img_bike, "step1_bicycle")

    # Bicycle mask from pixel diff (base vs bike image)
    bike_mask_img = pixel_diff_mask_img(base, img_bike, threshold=40.0)
    bike_pct = bike_mask_img.mean() * 100
    print(f"  Bicycle region: {bike_pct:.1f}% of image pixels (threshold=40)")

    # Visualise bike mask
    mask_vis = np.array(img_bike).copy()
    mask_vis[bike_mask_img] = (mask_vis[bike_mask_img] * 0.5 + np.array([255, 120, 0]) * 0.5).clip(0, 255)
    save(Image.fromarray(mask_vis.astype(np.uint8)), "bike_region_overlay")

    # Chain baseline (our existing best): step 1 → step 2 at standard settings
    chain_standard = run(img_bike, VASE_PROMPT_STANDARD, steps=28, guidance=2.5)
    save(chain_standard, "chain_baseline_28steps_g2.5")
    base_scores = bicycle_change_score(img_bike, chain_standard, bike_mask_img)
    print(f"  Chain baseline (28 steps, g=2.5): bike_diff={base_scores['bike_diff']}, "
          f"global_diff={base_scores['global_diff']}")

    scores = {}
    scores["baseline_28s_g2.5"] = base_scores

    # ----------------------------------------------------------
    # Experiment A: Step count sweep (guidance=2.5, standard prompt)
    # ----------------------------------------------------------
    print("\n=== Experiment A: Step count sweep (g=2.5, standard prompt) ===")
    step_counts = [5, 10, 15, 28]
    a_imgs, a_labels = [], []
    for s in step_counts:
        img = run(img_bike, VASE_PROMPT_STANDARD, steps=s, guidance=2.5)
        sc = bicycle_change_score(img_bike, img, bike_mask_img)
        scores[f"steps={s}_g=2.5"] = sc
        save(img, f"expA_steps{s}")
        a_imgs.append(img)
        a_labels.append(f"{s} steps, g=2.5\nbike_diff={sc['bike_diff']}")
        print(f"  {s} steps: bike_diff={sc['bike_diff']}  global_diff={sc['global_diff']}")

    save_grid([img_bike] + a_imgs, ["Step 1\n(bicycle only)"] + a_labels,
              os.path.join(args.out_dir, "expA_step_sweep.png"))
    print("  Saved: expA_step_sweep.png")
    print("  KEY: Does lower step count preserve bicycle better while still adding vase?")

    # ----------------------------------------------------------
    # Experiment B: Guidance sweep (steps=10, standard prompt)
    # ----------------------------------------------------------
    print("\n=== Experiment B: Guidance sweep (steps=10, standard prompt) ===")
    guidances = [1.0, 1.5, 2.0, 2.5, 3.0]
    b_imgs, b_labels = [], []
    for g in guidances:
        img = run(img_bike, VASE_PROMPT_STANDARD, steps=10, guidance=g)
        sc = bicycle_change_score(img_bike, img, bike_mask_img)
        scores[f"steps=10_g={g}"] = sc
        save(img, f"expB_g{g:.1f}")
        b_imgs.append(img)
        b_labels.append(f"10 steps, g={g:.1f}\nbike_diff={sc['bike_diff']}")
        print(f"  g={g:.1f}: bike_diff={sc['bike_diff']}  global_diff={sc['global_diff']}")

    save_grid([img_bike] + b_imgs, ["Step 1\n(bicycle only)"] + b_labels,
              os.path.join(args.out_dir, "expB_guidance_sweep.png"))
    print("  Saved: expB_guidance_sweep.png")
    print("  KEY: Lower guidance = stays closer to input = less bicycle change?")
    print("       Find: lowest guidance where vase IS added AND bike_diff is small.")

    # ----------------------------------------------------------
    # Experiment C: Prompt comparison (steps=10, guidance=2.0)
    # ----------------------------------------------------------
    print("\n=== Experiment C: Prompt comparison (steps=10, g=2.0) ===")
    prompts = [
        ("standard",  VASE_PROMPT_STANDARD),
        ("preserve",  VASE_PROMPT_PRESERVE),
        ("describe",  VASE_PROMPT_DESCRIBE),
    ]
    c_imgs, c_labels = [], []
    for pname, prompt in prompts:
        img = run(img_bike, prompt, steps=10, guidance=2.0)
        sc = bicycle_change_score(img_bike, img, bike_mask_img)
        scores[f"prompt={pname}_10s_g2.0"] = sc
        save(img, f"expC_prompt_{pname}")
        c_imgs.append(img)
        c_labels.append(f"Prompt: {pname}\nbike_diff={sc['bike_diff']}")
        print(f"  {pname}: bike_diff={sc['bike_diff']}  global_diff={sc['global_diff']}")

    save_grid([img_bike] + c_imgs + [chain_standard],
              ["Step 1"] + c_labels + ["Chain baseline\n28s g=2.5"],
              os.path.join(args.out_dir, "expC_prompt_compare.png"))
    print("  Saved: expC_prompt_compare.png")
    print("  KEY: Does explicitly naming the bicycle in the prompt reduce bike_diff?")

    # ----------------------------------------------------------
    # Experiment D: Best combination
    #   From A+B: expect best is steps=10, g=1.5 or g=2.0
    #   Try all 3 prompts at best settings
    # ----------------------------------------------------------
    print("\n=== Experiment D: Best settings + all prompts ===")
    BEST_STEPS, BEST_G = 10, 1.5
    d_imgs, d_labels = [], []
    for pname, prompt in prompts:
        img = run(img_bike, prompt, steps=BEST_STEPS, guidance=BEST_G)
        sc = bicycle_change_score(img_bike, img, bike_mask_img)
        scores[f"best_{pname}"] = sc
        save(img, f"expD_best_{pname}")
        d_imgs.append(img)
        d_labels.append(f"{pname} prompt\n{BEST_STEPS}s g={BEST_G}\nbike_diff={sc['bike_diff']}")
        print(f"  {pname}: bike_diff={sc['bike_diff']}  global_diff={sc['global_diff']}")

    save_grid([img_bike, chain_standard] + d_imgs,
              ["Step 1 (bicycle)", f"Chain baseline\nbike_diff={base_scores['bike_diff']}"] + d_labels,
              os.path.join(args.out_dir, "expD_best_combinations.png"))
    print("  Saved: expD_best_combinations.png")

    # ----------------------------------------------------------
    # Numeric summary table
    # ----------------------------------------------------------
    print(f"\n{'='*65}")
    print("BICYCLE IDENTITY PRESERVATION TABLE")
    print(f"{'='*65}")
    print(f"  {'Config':<35}  {'bike_diff':>9}  {'global_diff':>11}")
    print(f"  {'-'*35}  {'-'*9}  {'-'*11}")
    for k, v in sorted(scores.items(), key=lambda x: x[1]['bike_diff']):
        print(f"  {k:<35}  {v['bike_diff']:>9.2f}  {v['global_diff']:>11.2f}")
    print(f"  {'='*57}")
    print(f"  bike_diff: mean absolute pixel diff (0-255 scale) in bicycle region")
    print(f"  LOWER bike_diff = bicycle was preserved more faithfully")
    print(f"  LOWER global_diff = image changed less overall")
    print()
    print("  TARGET: Find config where:")
    print("    bike_diff << chain baseline  (bicycle identity better preserved)")
    print("    global_diff > 0              (vase WAS actually added)")

    # ----------------------------------------------------------
    # KEY RESULT: side-by-side of best config vs chain
    # ----------------------------------------------------------
    # Use preserve prompt at best steps/guidance
    img_best = run(img_bike, VASE_PROMPT_PRESERVE, steps=BEST_STEPS, guidance=BEST_G)
    sc_best = bicycle_change_score(img_bike, img_best, bike_mask_img)
    save(img_best, "KEY_best_result")

    save_grid(
        [base, img_bike, img_best, chain_standard],
        ["Base scene",
         "Step 1: add bicycle\n(28 steps, g=2.5)",
         f"Step 2: add vase (OPTIMIZED)\n{BEST_STEPS}s g={BEST_G} preserve-prompt\nbike_diff={sc_best['bike_diff']}",
         f"Chain BASELINE\n28s g=2.5 standard-prompt\nbike_diff={base_scores['bike_diff']}"],
        os.path.join(args.out_dir, "KEY_RESULT.png"),
    )
    print(f"\n  Saved: KEY_RESULT.png")
    print(f"  Best: bike_diff={sc_best['bike_diff']} vs baseline: bike_diff={base_scores['bike_diff']}")

    # Save numeric summary to file
    with open(os.path.join(args.out_dir, "scores.txt"), "w") as f:
        f.write("Bicycle Identity Preservation Scores\n")
        f.write("bike_diff: lower = bicycle changed less (better)\n")
        f.write("global_diff: how much the whole image changed\n\n")
        for k, v in sorted(scores.items(), key=lambda x: x[1]['bike_diff']):
            f.write(f"{k:<40} bike={v['bike_diff']:.2f}  global={v['global_diff']:.2f}\n")
    print("  Saved: scores.txt")

    print(f"\n{'='*65}")
    print("WHAT TO CHECK")
    print(f"{'='*65}")
    print(f"""
bike_region_overlay.png
  Orange tint = the bicycle region as measured by pixel diff (thr=40).
  This is the region used to measure bicycle identity preservation.
  If the orange region is too large or too small, the measurement is noisy.

expA_step_sweep.png
  Does reducing step count from 28→5 add the vase less aggressively?
  Lower steps = Kontext deviates less from input = bike should change less.
  But: too few steps = vase may not appear at all.

expB_guidance_sweep.png
  g=1.0: Kontext mostly copies the input, minimal edit → might not add vase
  g=1.5: slight edit → vase appears but bike minimally changed
  g=2.5: standard → vase added but bike changes somewhat
  FIND: lowest guidance where vase is visible and fully formed.

expC_prompt_compare.png (steps=10, g=2.0)
  Compare: standard vs preserve vs describe prompt.
  Does "The yellow bicycle must stay exactly as it is" reduce bike_diff?
  Expected: yes — explicitly naming the preserved element helps Kontext.

KEY_RESULT.png
  Final comparison: optimized chain vs standard chain.
  SUCCESS = vase IS present AND bike_diff is lower than baseline.
  This is "beating the chain" not by compositing but by better Kontext editing.
""")
    print(f"All results → {args.out_dir}/")


if __name__ == "__main__":
    main()
