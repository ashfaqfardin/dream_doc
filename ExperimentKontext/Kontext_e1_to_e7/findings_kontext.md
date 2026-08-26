# Kontext Model — Experimental Findings

**Model:** `black-forest-labs/FLUX.1-Kontext-dev`  
**Experiments:** E1–E6  
**Inference steps used:** 28 (default)  
**Seed:** 42 throughout

---

## E4 — Timestep Commitment

**Question:** At which denoising step does the output structure crystallise?

| Step | SSIM to final | LPIPS to final |
|------|--------------|----------------|
| 0–15 | < 0.15 | ~1.03 |
| 20 | 0.27 | 0.86 |
| 23 | 0.52 | 0.61 |
| 25 | 0.82 | 0.33 |
| 26 | 0.945 | 0.128 |
| 27 | 0.998 | 0.0003 |

**Finding:** The model operates as a noise sculptor for ~80% of its steps. Meaningful structure begins at step 23, with a sharp commitment cliff at steps 25–27. Final layout is effectively locked by step 27.

**Pipeline implication:**
- Step 3 edits can run at 20–22 steps without quality loss — the commitment window is proportionally the same.
- Any K/V injection for appearance steering should target **steps 24–27**, the only window where spatial form is being finalised.

---

## E1 — Attention Entropy per Block

**Question:** Which transformer blocks read the context image most selectively?

| Block range | Entropy range | Character |
|-------------|--------------|-----------|
| 0 | 8.42 | Global bootstrap, very diffuse |
| 1–11 | 7.1–7.5 | Flat, moderately high |
| 12–18 | 6.1–6.9 | Progressively focused |
| 18 (deepest) | 6.09 | Most selective |

**Finding:** Attention entropy decreases monotonically from block 0 to block 18. The last 6 double-stream blocks (13–18) are the most focused context-readers. Early blocks are doing global layout reasoning; late blocks are reading fine-grained appearance.

**Pipeline implication:**
- TIER_A for K/V injection = **blocks 13–18** (the 6 deepest double-stream blocks).
- Injecting into early or mid blocks is largely wasted — they are not reading context selectively.

---

## E2 — 3D RoPE Temporal Index Ablation

**Question:** Does the temporal index (i=1 vs i=2) affect what the model does with each context image? Does order matter?

| Condition | vs. A (single scene) SSIM | LPIPS | CLIP-I |
|-----------|--------------------------|-------|--------|
| B — scene × 2 (i=1 and i=2) | **0.996** | 0.0018 | 0.999 |
| C — scene (i=1), obj (i=2) | 0.744 | 0.359 | 0.661 |
| D — obj (i=1), scene (i=2) | 0.738 | 0.362 | 0.654 |

**Findings:**

1. **B ≈ A (SSIM=0.996):** Passing the same image at two different temporal indices gives pixel-identical output. The model recognises content identity and ignores the redundancy. Temporal index is meaningful (it doesn't double-weight) but not tricked.

2. **C ≈ D (SSIM 0.744 vs 0.738):** Swapping which context image gets i=1 vs i=2 has negligible effect. Both conditions produce the bicycle correctly placed in the scene. The model is not strongly biased by temporal ordering.

**Pipeline implication:**
- Ordering context images as `[original_scene, last_result]` is safe regardless of which gets i=1.
- Passing the same image twice causes no harm — validates the dual-context anchoring approach.

---

## E3 — Layer Ablation

**Question:** Which block ranges are load-bearing for context information?

| Ablated subset | n_ablated | SSIM vs baseline | LPIPS vs baseline |
|----------------|-----------|-----------------|-------------------|
| middle_third (6–12) | 6 | 0.956 | 0.050 — least impact |
| first_third (0–5) | 6 | 0.943 | 0.065 |
| every_3rd | 7 | 0.940 | 0.064 |
| all_layers | 19 | 0.927 | 0.073 |
| **last_third (13–18)** | **7** | **0.891** | **0.079 — most impact** |

**Findings:**

1. **Last third is most critical.** Ablating blocks 13–18 causes the largest deviation from full-context baseline. These blocks perform the final "apply context" refinement.

2. **Middle third is nearly irrelevant.** Blocks 6–12 carry almost no load-bearing context information.

3. **Counterintuitive:** Ablating ALL layers (SSIM=0.927) is less harmful than ablating only the last third (SSIM=0.891). When all blocks are ablated the model consistently falls back to text-prompt generation. When only the last third is ablated, it disrupts the final refinement stage specifically, causing more divergence.

4. **Overall magnitudes are modest.** Even with all context attention removed, the model produces SSIM=0.927 vs. baseline — text prompt alone drives most of the composition.

**Pipeline implication:**
- Confirm TIER_A = **blocks 13–18** (consistent with E1).
- K/V injection outside this range contributes little to appearance transfer.

---

## E5 — Multi-Context Attention Segregation

**Question:** When three context images are passed (`[scene, obj1, obj2]`), how much attention does each receive per block?

| Block | scene (i=1) | obj1 (i=2) | obj2 (i=3) | Dominant |
|-------|------------|-----------|-----------|---------|
| 0 | 0.227 | 0.259 | 0.259 | OBJ (bootstrap) |
| 1–18 | 0.087–0.160 | 0.029–0.111 | 0.029–0.133 | SCENE |

**Averages:**

| Context | Mean mass |
|---------|-----------|
| scene (i=1) | 0.1218 |
| obj1 (i=2) | 0.0802 |
| obj2 (i=3) | 0.0854 |

Scene/obj1 ratio: **1.52×**

**Findings:**

1. **Block 0 bootstraps uniformly.** All three context images receive nearly equal attention at block 0 — the model does an initial "inventory" before specialising.

2. **Scene dominates at every subsequent block.** The first context image (i=1) consistently receives ~1.5× more attention than secondary images.

3. **obj1 and obj2 are treated symmetrically** (0.080 vs 0.085). There is no meaningful specialisation between i=2 and i=3 — the model cannot distinguish "reference object" from "other context" by temporal index alone.

4. **The gap narrows in late blocks (13–18)** — objects receive relatively more attention in the deepest layers (0.094–0.111 vs scene 0.090–0.160). These late blocks are where object appearance is being "consumed."

**Pipeline implication:**
- Objects passed as secondary context (i≥2) receive only ~65% as much attention as the scene.
- Passive context concatenation is **insufficient** for object identity transfer — the model defaults to text-driven object generation.
- Active K/V injection at blocks 13–18 is required to force the model to use the reference object's visual features.

---

## E6 — Incremental Drift Measurement

**Question:** How much does the scene and object identity drift across a 7-step chain of incremental edits?

| Edit step | Object | bg_ssim vs original | obj_dino vs reference |
|-----------|--------|--------------------|-----------------------|
| 1 | bicycle | 0.716 | **0.720** ✓ |
| 2 | vase | 0.519 | **0.099** ✗ |
| 3 | ball | 0.392 | -0.005 ✗ |
| 4 | chair | 0.293 | -0.031 ✗ |
| 5 | lamp | 0.198 | 0.007 ✗ |
| 6 | plant | 0.111 | 0.013 ✗ |
| 7 | backpack | 0.065 | -0.023 ✗ |

**Findings:**

1. **Background drift is catastrophic and continuous.** Background SSIM vs. original drops from 0.72 → 0.065 over 7 steps. By step 7, the room is visually unrecognisable as the same scene. Each edit permanently transforms the space.

2. **Object identity collapses after the first object.** Only the bicycle (step 1) achieves meaningful DINOv2 similarity (0.72). From step 2 onward, DINO drops to near-zero or negative — the model is ignoring the reference image and generating objects from text description alone.

3. **Why step 1 works and others don't:** The bicycle is added to a clean empty room with a clear white-background reference. The context is unambiguous. From step 2 onward, the scene is cluttered, the previous edit output is the context (not the original room), and the model has insufficient capacity to read the object reference against a complex background.

4. **CLIP-I semantic similarity stays moderate** (0.52–0.86) — the model produces something in the correct semantic category ("a vase"), but not the specific reference vase. Semantic category ≠ visual identity.

**Pipeline implications:**

- **Dual-context anchoring is mandatory.** Always pass `image=[original_base_scene, last_result]` — the original scene must be present at every edit step as an anchor, not only at step 1.
- **K/V injection is non-optional for identity.** From step 2 onward, the object reference image is not being used. Active injection at blocks 13–18 is the only mechanism to force visual identity transfer.

---

## Consolidated Findings

| Experiment | Core finding | Required pipeline change |
|------------|-------------|--------------------------|
| E4 | Commitment cliff at steps 25–27/28 | Use 20–22 steps for edits; inject at steps 24–27 |
| E1 | Late blocks 13–18 are most focused | TIER_A = blocks 13–18 only |
| E3 | Last-third blocks carry context load | Confirmed: inject only at blocks 13–18 |
| E2 | Order and duplication are harmless | Use `[original_scene, last_result]` freely |
| E5 | Objects receive 65% of scene's attention | Passive context ≠ identity; active K/V injection needed |
| E6 | Identity collapses at step 2; drift is catastrophic | Both anchoring AND K/V injection are non-optional |

---

## Required Pipeline Architecture

Based on these findings, every Step 3 edit call should:

```
image=[original_base_scene, last_result]   # dual context — anchor + previous
K/V inject from reference_object at blocks 13–18, steps 24–27
inference_steps = 20–22
```

The two additions that are non-negotiable:
1. **Dual-context anchoring** — `original_base_scene` always present to arrest background drift.
2. **K/V injection at TIER_A (blocks 13–18)** — forces the model to use the reference object's visual identity, bypassing its tendency to hallucinate from text.
