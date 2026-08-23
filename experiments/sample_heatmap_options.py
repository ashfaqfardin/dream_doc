"""
Preview 4 alternative visualisations for the semantic sensitivity matrix.
Saves results/heatmap_option_*.png
"""
import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.gridspec as gridspec

try:
    import seaborn as sns
    HAS_SEABORN = True
except ImportError:
    HAS_SEABORN = False

np.random.seed(42)
N_MM, N_SINGLE = 19, 38
CATEGORIES = ["colour", "style", "material", "texture", "shape", "layout", "object"]
PAPER_MM_VITAL     = [0, 1, 17, 18]
PAPER_SINGLE_VITAL = [9, 34, 35, 37, 6]
os.makedirs("results", exist_ok=True)

# ── Synthetic data (same as sample_heatmap.py) ─────────────────────────────
matrix = np.zeros((N_MM + N_SINGLE, len(CATEGORIES)))
for i in range(N_MM):
    vb = 0.06 if i in PAPER_MM_VITAL else 0.0
    n  = np.random.uniform(0, 0.02, 7)
    matrix[i] = np.array([
        0.10*np.exp(-((i-1)**2)/4)  + 0.08*np.exp(-((i-17)**2)/2) + vb*0.8,
        0.12*np.exp(-((i-0)**2)/3)  + vb*0.7,
        0.08*np.exp(-((i-11)**2)/8) + vb*0.4,
        0.07*np.exp(-((i-13)**2)/6) + vb*0.5,
        0.06*np.exp(-((i-9)**2)/12) + vb*0.5,
        0.09*np.exp(-((i-0)**2)/5)  + vb*0.6,
        0.08*np.exp(-((i-8)**2)/10) + vb*0.6,
    ]) + n
for i in range(N_SINGLE):
    gi = N_MM + i
    vb = 0.04 if i in PAPER_SINGLE_VITAL else 0.0
    n  = np.random.uniform(0, 0.015, 7)
    b  = 0.03 + vb
    matrix[gi] = np.array([
        b*0.6 + 0.03*np.exp(-((i-9)**2)/8),
        b*0.4 + 0.02*np.exp(-((i-34)**2)/4),
        b*0.5 + 0.025*np.exp(-((i-35)**2)/6),
        b*0.7 + 0.03*np.exp(-((i-35)**2)/5),
        b*0.5 + 0.02*np.exp(-((i-6)**2)/8),
        b*0.6 + 0.025*np.exp(-((i-6)**2)/6),
        b*0.5 + 0.025*np.exp(-((i-9)**2)/8),
    ]) + n
matrix = np.clip(matrix, 0, None)

vital_rows = PAPER_MM_VITAL + [N_MM + i for i in PAPER_SINGLE_VITAL]
layer_labels = [f"MM-{i}" for i in range(N_MM)] + [f"S-{i}" for i in range(N_SINGLE)]

def mark_vital(ax, vital_rows, n_cols):
    for r in vital_rows:
        ax.add_patch(mpatches.FancyBboxPatch(
            (-0.5, r-0.5), n_cols, 1.0,
            lw=1.8, edgecolor="red", facecolor="none",
            transform=ax.transData, clip_on=False))

# ═══════════════════════════════════════════════════════════════════════════
# OPTION 1 — Split MM / Single  +  column-normalised  (diverging palette)
# ═══════════════════════════════════════════════════════════════════════════
mm_norm  = matrix[:N_MM]  / (matrix[:N_MM].max(0)  + 1e-8)
sin_norm = matrix[N_MM:]  / (matrix[N_MM:].max(0)  + 1e-8)

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 9),
                                gridspec_kw={"width_ratios": [N_MM, N_SINGLE]})

for ax, data, title, vital_idx in [
    (ax1, mm_norm.T,  "MM-DiT blocks (0–18)",      PAPER_MM_VITAL),
    (ax2, sin_norm.T, "Single-stream blocks (0–37)", [N_MM+i for i in PAPER_SINGLE_VITAL]),
]:
    if HAS_SEABORN:
        sns.heatmap(data, ax=ax, cmap="magma", linewidths=0,
                    xticklabels=[l.split("-")[1] for l in layer_labels[:data.shape[1]]],
                    yticklabels=CATEGORIES if ax is ax1 else [],
                    cbar_kws={"label": "Norm. sensitivity"},
                    vmin=0, vmax=1)
    else:
        im = ax.imshow(data, aspect="auto", cmap="magma", vmin=0, vmax=1)
        ax.set_xticks(range(data.shape[1]))
        ax.set_xticklabels([str(x) for x in range(data.shape[1])], fontsize=6)
        ax.set_yticks(range(len(CATEGORIES)))
        ax.set_yticklabels(CATEGORIES if ax is ax1 else [], fontsize=9)
        plt.colorbar(im, ax=ax, label="Norm. sensitivity")

    # vital column markers (now transposed so rows=categories, cols=layers)
    real_vital = PAPER_MM_VITAL if ax is ax1 else PAPER_SINGLE_VITAL
    for v in real_vital:
        ax.axvline(v - 0.5, color="red", lw=1.2, alpha=0.8)
        ax.axvline(v + 0.5, color="red", lw=1.2, alpha=0.8)
    ax.set_title(title, fontsize=11, fontweight="bold")
    ax.set_xlabel("Layer index", fontsize=9)

fig.suptitle("Option 1 — Split MM/Single, column-normalised, magma palette",
             fontsize=12, fontweight="bold")
plt.tight_layout()
plt.savefig("results/heatmap_option_1.png", dpi=150, bbox_inches="tight")
plt.close()
print("Saved option 1")


# ═══════════════════════════════════════════════════════════════════════════
# OPTION 2 — Bubble / dot chart  (size + colour = sensitivity)
# ═══════════════════════════════════════════════════════════════════════════
fig, axes = plt.subplots(1, 2, figsize=(18, 7),
                          gridspec_kw={"width_ratios": [N_MM, N_SINGLE], "wspace": 0.05})

for ax, rows, offset, title, vital_idx in [
    (axes[0], range(N_MM),    0,    "MM-DiT",       PAPER_MM_VITAL),
    (axes[1], range(N_SINGLE), N_MM, "Single-stream", PAPER_SINGLE_VITAL),
]:
    xs, ys, sz, col = [], [], [], []
    for li in rows:
        for ci, cat in enumerate(CATEGORIES):
            val = matrix[offset + li, ci]
            xs.append(li)
            ys.append(ci)
            sz.append(val * 4000)
            col.append(val)

    sc = ax.scatter(xs, ys, s=sz, c=col, cmap="YlOrRd",
                    vmin=0, vmax=matrix.max(), alpha=0.85, edgecolors="grey", lw=0.3)
    for v in vital_idx:
        ax.axvline(v, color="red", lw=1.5, alpha=0.6, linestyle="--")
    ax.set_xticks(range(len(rows)))
    ax.set_xticklabels([str(i) for i in rows], fontsize=6)
    ax.set_yticks(range(len(CATEGORIES)))
    ax.set_yticklabels(CATEGORIES if ax is axes[0] else [], fontsize=9)
    ax.set_xlabel("Layer index", fontsize=9)
    ax.set_title(title, fontsize=11, fontweight="bold")
    ax.grid(True, alpha=0.2)
    ax.set_xlim(-0.8, len(rows) - 0.2)
    ax.set_ylim(-0.8, len(CATEGORIES) - 0.2)

plt.colorbar(sc, ax=axes[1], label="DINOv2 sensitivity")
fig.suptitle("Option 2 — Bubble chart  (size + colour = sensitivity, red dashes = vital layers)",
             fontsize=12, fontweight="bold")
plt.savefig("results/heatmap_option_2.png", dpi=150, bbox_inches="tight")
plt.close()
print("Saved option 2")


# ═══════════════════════════════════════════════════════════════════════════
# OPTION 3 — Per-category line plots  (all 7 semantics, MM + single)
# ═══════════════════════════════════════════════════════════════════════════
COLORS = plt.cm.tab10(np.linspace(0, 1, len(CATEGORIES)))
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(16, 8), sharex=False)

x_mm  = np.arange(N_MM)
x_sin = np.arange(N_SINGLE)

for ci, (cat, color) in enumerate(zip(CATEGORIES, COLORS)):
    ax1.plot(x_mm,  matrix[:N_MM, ci],  color=color, label=cat, lw=1.8, marker="o", ms=4)
    ax2.plot(x_sin, matrix[N_MM:, ci],  color=color, lw=1.5, marker="o", ms=3)

for ax, vital_idx, n in [(ax1, PAPER_MM_VITAL, N_MM), (ax2, PAPER_SINGLE_VITAL, N_SINGLE)]:
    for v in vital_idx:
        ax.axvspan(v - 0.4, v + 0.4, alpha=0.15, color="red")
    ax.set_xlim(-0.5, n - 0.5)
    ax.set_xticks(range(n))
    ax.set_xticklabels([str(i) for i in range(n)], fontsize=7)
    ax.set_ylabel("Sensitivity", fontsize=9)
    ax.grid(True, alpha=0.25)
    ax.yaxis.set_tick_params(labelsize=8)

ax1.set_title("MM-DiT blocks (0–18)  —  red shading = vital layers", fontsize=11)
ax2.set_title("Single-stream blocks (0–37)", fontsize=11)
ax1.legend(fontsize=8, loc="upper right", ncol=4)
ax2.set_xlabel("Layer index", fontsize=9)

fig.suptitle("Option 3 — Line plot per semantic category", fontsize=12, fontweight="bold")
plt.tight_layout()
plt.savefig("results/heatmap_option_3.png", dpi=150, bbox_inches="tight")
plt.close()
print("Saved option 3")


# ═══════════════════════════════════════════════════════════════════════════
# OPTION 4 — Stacked bar chart  (each layer = one bar, categories stacked)
# ═══════════════════════════════════════════════════════════════════════════
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(18, 9), sharex=False)

for ax, data, n, vital_idx, title in [
    (ax1, matrix[:N_MM],  N_MM,    PAPER_MM_VITAL,     "MM-DiT blocks (0–18)"),
    (ax2, matrix[N_MM:], N_SINGLE, PAPER_SINGLE_VITAL, "Single-stream blocks (0–37)"),
]:
    x      = np.arange(n)
    bottom = np.zeros(n)
    for ci, (cat, color) in enumerate(zip(CATEGORIES, COLORS)):
        ax.bar(x, data[:, ci], bottom=bottom, label=cat,
               color=color, width=0.75, edgecolor="none")
        bottom += data[:, ci]
    for v in vital_idx:
        ax.axvspan(v - 0.45, v + 0.45, alpha=0.15, color="red", zorder=0)
        ax.text(v, bottom[v] + 0.008, "★", ha="center", va="bottom",
                fontsize=9, color="red", fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels([str(i) for i in x], fontsize=7)
    ax.set_ylabel("Total sensitivity", fontsize=9)
    ax.set_title(title, fontsize=11, fontweight="bold")
    ax.grid(axis="y", alpha=0.25)

ax1.legend(fontsize=8, loc="upper right", ncol=4, framealpha=0.9)
ax2.set_xlabel("Layer index", fontsize=9)
fig.suptitle("Option 4 — Stacked bars  (total sensitivity per layer, colour = category)",
             fontsize=12, fontweight="bold")
plt.tight_layout()
plt.savefig("results/heatmap_option_4.png", dpi=150, bbox_inches="tight")
plt.close()
print("Saved option 4")
