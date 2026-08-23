"""
Generates a synthetic but realistic semantic sensitivity heatmap
to preview what the actual experiment output will look like.

Based on expected FLUX.1-dev behaviour:
- Vital layers (MM-0,1,17,18 / S-9,34,35,37,6) show higher sensitivity
- Early MM-DiT layers encode colour/style more strongly
- Mid MM-DiT layers handle object identity and shape
- Late MM-DiT layers handle texture/material
- Single-stream blocks show lower but non-zero sensitivity
- Layout is distributed across MM and early single-stream blocks
"""

import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

try:
    import seaborn as sns
    HAS_SEABORN = True
except ImportError:
    HAS_SEABORN = False

np.random.seed(42)

N_MM     = 19
N_SINGLE = 38
CATEGORIES = ["colour", "style", "material", "texture", "shape", "layout", "object"]
PAPER_MM_VITAL     = [0, 1, 17, 18]
PAPER_SINGLE_VITAL = [9, 34, 35, 37, 6]

# ---------------------------------------------------------------------------
# Build synthetic sensitivity matrix (57 × 7)
# ---------------------------------------------------------------------------

matrix = np.zeros((N_MM + N_SINGLE, len(CATEGORIES)))

# --- MM-DiT blocks ---
for i in range(N_MM):
    vital_boost = 0.06 if i in PAPER_MM_VITAL else 0.0
    noise       = np.random.uniform(0.0, 0.02, len(CATEGORIES))

    # colour: peaks in early layers (0-3) and late vital layers
    colour   = 0.10 * np.exp(-((i - 1) ** 2) / 4) + \
               0.08 * np.exp(-((i - 17) ** 2) / 2) + vital_boost * 0.8
    # style: peaks early (0-2), texture of transfer
    style    = 0.12 * np.exp(-((i - 0) ** 2) / 3) + vital_boost * 0.7
    # material: mid-range layers (8-14)
    material = 0.08 * np.exp(-((i - 11) ** 2) / 8) + vital_boost * 0.4
    # texture: later mid layers (10-16)
    texture  = 0.07 * np.exp(-((i - 13) ** 2) / 6) + vital_boost * 0.5
    # shape: broad mid (5-15)
    shape    = 0.06 * np.exp(-((i - 9)  ** 2) / 12) + vital_boost * 0.5
    # layout: very early layers (structure), peaks 0-2
    layout   = 0.09 * np.exp(-((i - 0)  ** 2) / 5) + vital_boost * 0.6
    # object: mid layers, identity
    obj      = 0.08 * np.exp(-((i - 8)  ** 2) / 10) + vital_boost * 0.6

    matrix[i] = np.array([colour, style, material, texture, shape, layout, obj]) + noise

# --- Single-stream blocks ---
for i in range(N_SINGLE):
    g_idx       = N_MM + i
    vital_boost = 0.04 if i in PAPER_SINGLE_VITAL else 0.0
    noise       = np.random.uniform(0.0, 0.015, len(CATEGORIES))

    # Single-stream: lower sensitivity overall, more uniform across categories
    # Some peaks around the vital single-stream layers
    base     = 0.03 + vital_boost
    colour   = base * 0.6 + 0.03 * np.exp(-((i - 9)  ** 2) / 8)
    style    = base * 0.4 + 0.02 * np.exp(-((i - 34) ** 2) / 4)
    material = base * 0.5 + 0.025 * np.exp(-((i - 35) ** 2) / 6)
    texture  = base * 0.7 + 0.03 * np.exp(-((i - 35) ** 2) / 5)
    shape    = base * 0.5 + 0.02 * np.exp(-((i - 6)  ** 2) / 8)
    layout   = base * 0.6 + 0.025 * np.exp(-((i - 6)  ** 2) / 6)
    obj      = base * 0.5 + 0.025 * np.exp(-((i - 9)  ** 2) / 8)

    matrix[g_idx] = np.array([colour, style, material, texture, shape, layout, obj]) + noise

matrix = np.clip(matrix, 0, None)

# ---------------------------------------------------------------------------
# Plot
# ---------------------------------------------------------------------------

os.makedirs("results", exist_ok=True)

layer_labels = [f"MM-{i}"  for i in range(N_MM)] + \
               [f"S-{i}"   for i in range(N_SINGLE)]

vital_rows = PAPER_MM_VITAL + [N_MM + i for i in PAPER_SINGLE_VITAL]

fig, ax = plt.subplots(figsize=(11, 20))

if HAS_SEABORN:
    sns.heatmap(
        matrix,
        xticklabels=CATEGORIES,
        yticklabels=layer_labels,
        cmap="YlOrRd",
        ax=ax,
        linewidths=0.3,
        linecolor="white",
        cbar_kws={"label": "DINOv2 sensitivity |sim_full − sim_ablated|"},
    )
else:
    im = ax.imshow(matrix, cmap="YlOrRd", aspect="auto")
    ax.set_xticks(range(len(CATEGORIES)))
    ax.set_xticklabels(CATEGORIES, fontsize=10)
    ax.set_yticks(range(len(layer_labels)))
    ax.set_yticklabels(layer_labels, fontsize=7)
    plt.colorbar(im, ax=ax, label="DINOv2 sensitivity |sim_full − sim_ablated|")

# Red borders on vital layers
for row in vital_rows:
    rect = mpatches.FancyBboxPatch(
        (-0.5, row - 0.5), len(CATEGORIES), 1.0,
        linewidth=2.0, edgecolor="red", facecolor="none",
        transform=ax.transData, clip_on=False,
    )
    ax.add_patch(rect)

# MM / single-stream boundary
ax.axhline(N_MM - 0.5, color="black", linewidth=2.5)
ax.text(len(CATEGORIES) + 0.1, N_MM // 2,
        "MM-DiT\nblocks", fontsize=8, va="center", color="navy")
ax.text(len(CATEGORIES) + 0.1, N_MM + N_SINGLE // 2,
        "Single-\nstream", fontsize=8, va="center", color="darkgreen")

ax.set_xlabel("Semantic category", fontsize=12)
ax.set_ylabel("Layer", fontsize=12)
ax.set_title(
    "Semantic Sensitivity per Layer  (FLUX.1-dev  —  SAMPLE)\n"
    "Higher = layer encodes this semantic  |  Red border = paper vital layers",
    fontsize=12,
)

plt.tight_layout()
out = "results/sample_semantic_heatmap.png"
plt.savefig(out, dpi=150, bbox_inches="tight")
print(f"Saved: {out}")
plt.show()
