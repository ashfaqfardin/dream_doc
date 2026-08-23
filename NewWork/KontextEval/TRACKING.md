# FLUX.1-Kontext-dev Incremental Editing — Phase Tracker

> Fill in **Results** sections as you run each phase. Paste terminal output directly into the code blocks.

| Phase | Name | Status | Script |
|-------|------|--------|--------|
| 1 | Environment & Baseline | ✅ Done | `phase1_baseline.py` |
| 2 | Architecture Inspection | ✅ Done | `phase2_architecture.py` |
| 3 | Attention Hooking | ✅ Done | `phase3_hooking.py` |
| 4 | Attention Cache | ✅ Done | `phase4_cache.py` |
| 5 | Injection Prototype | ⬜ Pending | `phase5_injection.py` |
| 6 | Layer Ablation | ⬜ Pending | `phase6_ablation.py` |
| 7 | Incremental Pipeline | ⬜ Pending | `phase7_pipeline.py` |
| 8 | Adaptive α | ⬜ Pending | `phase8_adaptive.py` |
| 9 | Full Evaluation | ⬜ Pending | `phase9_eval.py` |

**Status key:** ⬜ Pending · 🔄 In Progress · ✅ Done · ❌ Failed

---

## Setup

```bash
cd /content/cherry_on_top_exp
pip install -r NewWork/KontextEval/requirements.txt
```

---

## Phase 1 — Environment & Baseline

**Status:** ✅ Done

**Run:**
```bash
python NewWork/KontextEval/phase1_baseline.py \
    --hf_token $HF_TOKEN \
    --cache_dir ./models \
    --out_dir results/phase1
```

### Checklist
- [x] Google Colab A100 configured
- [x] FLUX.1-Kontext-dev loads without error
- [x] Baseline edits produce plausible output images
- [x] Deterministic generation verified (same seed → same pixels)

### Results

| Edit step | Action | Output file |
|-----------|--------|-------------|
| Step 0 | Empty grey scene (base) | `step0_empty_scene.png` |
| Step 1 | Add wooden chair | `step1_add_wooden_chair.png` |
| Step 2 | Replace → iron chair | `step2_replace_iron_chair.png` |
| Step 3 | Change color → red | `step3_change_chair_color.png` |
| Step 4 | Style → oil painting | `step4_style_change.png` |

**Deterministic verification:**
- Run 1 hash: `870c378de98ae7e7e57c3e3e37515f5f`
- Run 2 hash: `870c378de98ae7e7e57c3e3e37515f5f`
- **Result: ✅ Identical — same seed produces pixel-identical output**

### Notes
- Model downloaded fresh (29.2 GB / ~1.5 min on Colab)
- Height/width auto-adjusted from 768 → 1024 by the pipeline (model minimum)
- Use `--size 1024` on future runs to avoid the adjustment warning
- Chainwise editing confirmed: each step used the previous step's output as input

---

## Phase 2 — Architecture Inspection

**Status:** ✅ Done

**Run:**
```bash
python NewWork/KontextEval/phase2_architecture.py \
    --hf_token $HF_TOKEN \
    --cache_dir ./models \
    --out_dir results/phase2
```

### Checklist
- [x] Total parameter count documented
- [x] Double-stream block count confirmed
- [x] Single-stream block count confirmed
- [x] Q/K/V projection shapes recorded
- [x] Expected K/V tensor shape at 1024×1024 computed
- [x] `architecture_summary.json` saved

### Results

| Property | Value |
|----------|-------|
| Total params (B) | 11.90B |
| Double-stream blocks | 19 |
| Single-stream blocks | 38 |
| Hidden dim | 3072 |
| Attention heads | 24 |
| Head dim | 128 |
| Guidance embeds | True |
| In channels | 64 |
| VAE scale factor | 8 |
| Latent spatial dim | 128×128 |
| Image tokens / image | 4096 (64×64 packed) |
| Kontext total img tokens | 8192 (ref + gen) |
| Text tokens (T5) | ~256 |
| Joint seq len (double) | ~8448 |
| K/V shape (double, example) | [1, 24, 8448, 128] |
| K/V size MB (bfloat16, per block) | 51.9 MB |

**Q/K/V projection shapes (all blocks identical):**
- Double: Q/K/V/Out each (3072, 3072), add_Q/K/V each (3072, 3072) for text stream
- Single: Q/K/V each (3072, 3072), no separate `to_out` (fused at block level)

### Notes
- All 19 double blocks and all 38 single blocks have identical weight shapes
- Total K/V cache for one generation (all 57 blocks, K+V): 57 × 2 × 51.9 MB ≈ 5.9 GB
- Single-stream `FluxAttention` has no `to_out` — output is fused with MLP at block level

---

## Phase 3 — Attention Hooking

**Status:** ✅ Done

**Run:**
```bash
python NewWork/KontextEval/phase3_hooking.py \
    --hf_token $HF_TOKEN \
    --cache_dir ./models \
    --out_dir results/phase3
```

### Checklist
- [x] CaptureProcessor attaches to all blocks without error
- [x] Output pixel-identical to baseline (same seed, same prompt)
- [x] Q, K, V shapes extracted and documented
- [x] Inference overhead measured

### Results

| Metric | Value |
|--------|-------|
| Pixel-identical | **YES ✓** |
| Time without hooks (s) | 28.6s |
| Time with hooks (s) | 209.8s |
| Overhead (%) | +634.3% |
| Total tensors captured | 171 (Q+K+V × 57 blocks) |
| Total capture size (MB) | 9145 MB (~8.9 GB) |
| `double_0_K` shape | (1, 24, 8704, 128) |
| `double_0_V` shape | (1, 24, 8704, 128) |
| `single_0_K` shape | (1, 24, 8704, 128) |

### Notes
- **Actual seq len is 8704**, not 8448 predicted in Phase 2. Breakdown: 8192 img tokens (ref+gen) + **512** text tokens (T5 max = 512, not 256 as predicted)
- Double and single-stream blocks have identical Q/K/V shapes — Kontext processes the full joint sequence in all 57 blocks
- Overhead of 634% is expected: `detach().cpu()` on 9 GB per denoising step × 28 steps. For Phases 4–9, capture is only stored once (not per-step), so per-generation cost ≈ one copy of 9 GB instead of 28 copies
- Two bugs fixed in `utils/attention_utils.py` during this phase:
  1. `dh = out.shape[-1] // h` → unpacked `b, h, seq, dh = out.shape` before transpose
  2. `attn.to_out[0]` on single-stream blocks → guarded with `hasattr(attn, "to_out")`

---

## Phase 4 — Attention Cache Construction

**Status:** ✅ Done

**Run:**
```bash
python NewWork/KontextEval/phase4_cache.py \
    --hf_token $HF_TOKEN \
    --cache_dir ./models \
    --out_dir results/phase4 \
    --prompt "a modern living room with a sofa"
```

### Checklist
- [x] K/V captured for all blocks
- [x] Cache saved to disk (one `.pt` file per tensor)
- [x] Cache reloaded successfully
- [x] Numerical equality verified (all keys pass)

### Results

| Property | Value |
|----------|-------|
| Tensors saved (K+V only) | 114 |
| Tensors loaded | 114 |
| Verification passed | **114 / 114** |
| Total cache size (MB) | 6096 MB (~6 GB) |
| `double_0_K` shape | (1, 24, 8704, 128) |
| `double_0_V` shape | (1, 24, 8704, 128) |
| dtype | torch.bfloat16 |

### Notes
- Q tensors are captured in memory but NOT saved to disk (not needed for injection)
- Fix applied: `save_cache` called with `store_kv` (K+V only filter) instead of full `store`
- 6 GB cache fits on A100 with room to spare alongside the 33 GB model

---

## Phase 5 — Attention Injection Prototype

**Status:** ⬜ Pending

**Run:**
```bash
python NewWork/KontextEval/phase5_injection.py \
    --hf_token $HF_TOKEN \
    --cache_dir ./models \
    --out_dir results/phase5 \
    --base_prompt "a modern living room with a sofa and a coffee table" \
    --edit_prompt "add a yellow bicycle leaning against the wall"
```

### Checklist
- [ ] Experiment A: K-only injection at α = 0.25 / 0.50 / 0.75 / 1.00
- [ ] Experiment B: V-only injection at α = 0.25 / 0.50 / 0.75 / 1.00
- [ ] Experiment C: K+V injection at α = 0.25 / 0.50 / 0.75 / 1.00
- [ ] Comparison grid saved

### Results

```
# Paste terminal output here
```

#### Visual observations

| Experiment | α | Content preserved? | Edit applied? | Artefacts? |
|-----------|---|-------------------|---------------|------------|
| K_only | 0.25 | — | — | — |
| K_only | 0.50 | — | — | — |
| K_only | 0.75 | — | — | — |
| K_only | 1.00 | — | — | — |
| V_only | 0.25 | — | — | — |
| V_only | 0.50 | — | — | — |
| V_only | 0.75 | — | — | — |
| V_only | 1.00 | — | — | — |
| K_and_V | 0.25 | — | — | — |
| K_and_V | 0.50 | — | — | — |
| K_and_V | 0.75 | — | — | — |
| K_and_V | 1.00 | — | — | — |

**Best experiment / α:** —

### Notes

---

## Phase 6 — Layer Ablation

**Status:** ⬜ Pending

**Run:**
```bash
python NewWork/KontextEval/phase6_ablation.py \
    --hf_token $HF_TOKEN \
    --cache_dir ./models \
    --out_dir results/phase6 \
    --alpha_k 0.5 --alpha_v 0.5
```

### Checklist
- [ ] Early layers (double 0–6) tested
- [ ] Middle layers (double 7–13) tested
- [ ] Late double layers (14–18) tested
- [ ] Late single-stream layers tested
- [ ] All double blocks tested
- [ ] All blocks tested
- [ ] CSV saved

### Results

```
# Paste terminal output here
```

| Layer Group | Blocks | PSNR ↑ | SSIM ↑ | LPIPS ↓ | DINOv2 ↑ |
|-------------|--------|--------|--------|---------|---------|
| early (0–6) | 7 dbl | — | — | — | — |
| middle (7–13) | 7 dbl | — | — | — | — |
| late_double (14–18) | 5 dbl | — | — | — | — |
| late_single | 38 sgl | — | — | — | — |
| all_double | 19 dbl | — | — | — | — |
| all | 19+38 | — | — | — | — |
| **baseline (no inj)** | — | — | — | — | — |

**Best layer group:** — (highest DINOv2 / lowest LPIPS)

### Hypothesis vs Finding
| Hypothesis | Confirmed? |
|------------|------------|
| Early → layout/global structure | — |
| Middle → object identity | — |
| Late double → fine details | — |
| Late single → texture | — |

### Notes

---

## Phase 7 — Incremental Editing Pipeline

**Status:** ⬜ Pending

**Run:**
```bash
python NewWork/KontextEval/phase7_pipeline.py \
    --hf_token $HF_TOKEN \
    --cache_dir ./models \
    --out_dir results/phase7 \
    --alpha_k 0.5 --alpha_v 0.5
```

### Edit Sequence
| Step | Prompt |
|------|--------|
| 0 | A modern living room |
| 1 | Add a bicycle |
| 2 | Add a vase on the table |
| 3 | Replace bicycle with a car |
| 4 | Change car color to red |
| 5 | Remove vase |

### Checklist
- [ ] Baseline run (no injection) completed
- [ ] Method run (K+V injection) completed
- [ ] DINOv2 computed for all steps
- [ ] results.json saved

### Results

```
# Paste terminal output here
```

| Step | Prompt | Baseline DINOv2 | Method DINOv2 | Baseline PSNR | Method PSNR |
|------|--------|----------------|---------------|---------------|-------------|
| 1 | Add a bicycle | — | — | — | — |
| 2 | Add a vase | — | — | — | — |
| 3 | Replace bicycle → car | — | — | — | — |
| 4 | Change car → red | — | — | — | — |
| 5 | Remove vase | — | — | — | — |

**Questions to answer from images:**
- [ ] Does the living room remain consistent across steps?
- [ ] Does the bicycle disappear correctly at step 3?
- [ ] Does the car colour change at step 4?
- [ ] Do errors accumulate noticeably?

### Notes

---

## Phase 8 — Adaptive Attention Preservation

**Status:** ⬜ Pending

**Run:**
```bash
python NewWork/KontextEval/phase8_adaptive.py \
    --hf_token $HF_TOKEN \
    --cache_dir ./models \
    --out_dir results/phase8 \
    --fixed_alpha 0.5 \
    --adaptive_base 0.5 \
    --adaptive_method cosine
```

### Checklist
- [ ] Fixed-α baseline completed
- [ ] Cosine-adaptive run completed
- [ ] Per-layer α values logged
- [ ] Final metrics computed

### Results

```
# Paste terminal output here
```

| Method | PSNR ↑ | LPIPS ↓ | DINOv2 ↑ |
|--------|--------|---------|---------|
| Baseline (no inj) | — | — | — |
| Fixed α = 0.5 | — | — | — |
| Adaptive (cosine) | — | — | — |

**Adaptive α per step (mean over layers):**

| Step | Prompt | Mean α |
|------|--------|--------|
| 1 | Add a bicycle | — |
| 2 | Add a vase | — |
| 3 | Replace bicycle → car | — |
| 4 | Change car → red | — |
| 5 | Remove vase | — |

**Adaptive vs Fixed: improvement?** —

### Notes

---

## Phase 9 — Full Evaluation

**Status:** ⬜ Pending

**Run:**
```bash
python NewWork/KontextEval/phase9_eval.py \
    --hf_token $HF_TOKEN \
    --cache_dir ./models \
    --out_dir results/phase9
```

### Checklist
- [ ] All 5 methods run on full sequence
- [ ] LPIPS computed for all steps/methods
- [ ] DINOv2 computed for all steps/methods
- [ ] PSNR computed for all steps/methods
- [ ] CLIP direction similarity computed
- [ ] CSV exported
- [ ] Final comparison table produced

### Content Preservation (DINOv2 ↑ vs base image)

| Step | Native | K only | V only | K+V fixed | K+V adaptive |
|------|--------|--------|--------|-----------|--------------|
| 1 | — | — | — | — | — |
| 2 | — | — | — | — | — |
| 3 | — | — | — | — | — |
| 4 | — | — | — | — | — |
| 5 | — | — | — | — | — |

### Edit Success (CLIP Direction Similarity ↑)

| Step | Native | K only | V only | K+V fixed | K+V adaptive |
|------|--------|--------|--------|-----------|--------------|
| 1 | — | — | — | — | — |
| 2 | — | — | — | — | — |
| 3 | — | — | — | — | — |
| 4 | — | — | — | — | — |
| 5 | — | — | — | — | — |

### Content Preservation (LPIPS ↓ vs base image)

| Step | Native | K only | V only | K+V fixed | K+V adaptive |
|------|--------|--------|--------|-----------|--------------|
| 1 | — | — | — | — | — |
| 2 | — | — | — | — | — |
| 3 | — | — | — | — | — |
| 4 | — | — | — | — | — |
| 5 | — | — | — | — | — |

### Summary Findings
| Claim | Evidence | Confirmed? |
|-------|----------|------------|
| K injection preserves content better than native | DINOv2 gap at step 5 | — |
| V injection helps less than K injection | DINOv2 K vs V | — |
| K+V beats K-only | DINOv2 K+V vs K | — |
| Adaptive α outperforms fixed α | DINOv2 adaptive vs fixed | — |
| Edit alignment degrades with injection (trade-off) | CLIP dir. similarity | — |

### Results Terminal Output
```
# Paste final terminal output here
```

### Notes

---

## File Structure

```
NewWork/KontextEval/
├── TRACKING.md             ← this file
├── requirements.txt
├── utils/
│   ├── model_utils.py      — pipeline loading, generate()
│   ├── attention_utils.py  — CaptureProcessor, InjectProcessor, AdaptiveInjectProcessor
│   ├── cache_utils.py      — save/load/verify K/V cache
│   └── metrics.py          — LPIPS, PSNR, DINOv2, CLIP
├── phase1_baseline.py
├── phase2_architecture.py
├── phase3_hooking.py
├── phase4_cache.py
├── phase5_injection.py
├── phase6_ablation.py
├── phase7_pipeline.py
├── phase8_adaptive.py
└── phase9_eval.py
```

## Key Design Decisions

| Decision | Rationale |
|----------|-----------|
| CaptureProcessor mirrors FluxAttnProcessor2_0 exactly | Guarantees zero-impact hooking (Phase 3 sanity check) |
| Cache stores K/V on CPU | Avoids occupying GPU VRAM between generations |
| Injection formula: `(1-α)K_curr + α K_cache` | α=0 → no injection, α=1 → full replacement; continuous sweep |
| Adaptive α via cosine similarity | High sim. → small α (allow edit); low sim. → large α (preserve) |
| DINOv2 [CLS] for semantic preservation | Captures object identity better than pixel metrics |
| CLIP direction for edit alignment | Standard metric from InstructPix2Pix, measures edit direction |
