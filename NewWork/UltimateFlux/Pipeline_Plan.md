# UltimateFlux — Unified Editing Pipeline (FLUX.1-dev primary target)

Source papers: **StableFlow** (2411.14430, CVPR'25), **FluxSpace** (2412.09611, CVPR'25),
**FreeFlux** (2503.16153, ICCV'25), **SVD-Style** (2507.04482, arXiv). First three are trained-free
methods built and validated on **FLUX.1-dev**; SVD-Style is built on **Infinity** (autoregressive), not FLUX.
**Primary deployment model: FLUX.1-dev** (all papers validated here; layer sets and hyperparameters are
paper-confirmed). The schnell ablation study in §5 is preserved as a research contribution and can serve
as a starting hypothesis for a fast-inference variant.

> **Critical implementation note — text/image token boundary:**
> In FLUX's attention sequence, the first `max_sequence_length` positions are text tokens (T5 encoding)
> and the remainder are image tokens. K,V injection must target **only image token positions**
> (`txt_len:` for double-stream blocks, `max_sequence_length:` for single-stream blocks where
> `txt_len=0` is passed from the sampler). Injecting text token K,V from the source branch into the edit
> branch overwrites the edit prompt's text conditioning — which is wrong for all tasks where
> source_prompt ≠ edit_prompt. FreeFlux enforces this with `kc_tgt_modified[:,:,512:,:]` (512 = default
> T5 length for dev). Our implementation uses a dynamic `txt_len` / `_txt_len_single` equivalent.

---

## 1. Downstream tasks in scope

| # | Task | Definition |
|---|---|---|
| 1 | Non-rigid editing | Change pose/shape/action while preserving identity, texture, background |
| 2 | Object addition | Insert a new object at a plausible location, leave everything else pixel-identical |
| 3 | Object replacement | Swap one object for another at the same location/pose |
| 4 | Background/scene replacement | Regenerate background completely, preserve foreground pixel-perfectly |
| 5 | Fine-grained local attribute editing | Disentangled single-attribute edits (e.g. add eyeglasses) without touching anything else |
| 6 | Global/coarse style-appearance editing | Shift the whole image's style (photo → comic/oil painting) |
| 7 | Reference-based style personalization | Apply the style of a reference image to new content from a text prompt |
| 8 | Real-image editing (inversion) | Cross-cutting: tasks 1–6 applied to a real photo instead of a freshly generated image |

## 2. Task → paper mapping (from each paper's own benchmarks)

| Task | Best paper | Evidence | Notes |
|---|---|---|---|
| Non-rigid editing | FreeFlux | 74.1% user pref vs StableFlow 11.1% | |
| Object addition | FreeFlux | 50.8% pref; solves the "Suppression Phenomenon" | |
| Object replacement | StableFlow | CLIP_img 0.92, explicitly benchmarked | FreeFlux doesn't test this directly |
| Background replacement | FreeFlux | 81.4% pref, SAM-2 mask + value-only injection | StableFlow weak here (1.4%) |
| Fine-grained attribute edits | FluxSpace | best identity preservation, CLIP-I 0.94 / DINO 0.94 | orthogonal projection + attention mask |
| Global style shift | FluxSpace | coarse-level pooled-CLIP embedding editing | |
| Reference style personalization | SVD-Style | only paper targeting this task | not built on FLUX — requires adaptation |
| Real-image inversion | StableFlow (simplest) | latent nudging, λ=1.15 | FluxSpace uses RF-Inversion; FreeFlux uses inverse Euler ODE — all dev-tuned |

## 3. Why FLUX.1-schnell isn't a drop-in swap for dev

Architecture is identical between dev and schnell (19 double-stream + 38 single-stream blocks), so layer
*indices* are structurally transferable. But schnell is a separately distilled checkpoint:

| Property | dev | schnell | Why it matters |
|---|---|---|---|
| Sampling steps | ~28–50 | 1–4 | Less room to correct injection artifacts; inversion is harder |
| Guidance | guidance-distilled, takes a guidance-scale embedding | guidance largely irrelevant | FluxSpace's null-prompt orthogonal-projection trick assumes a meaningful conditioned-vs-null gap that schnell wasn't trained to preserve |
| License | non-commercial | Apache 2.0 | relevant if there's any deployment angle |

None of the four papers report schnell numbers. Every layer set and hyperparameter borrowed from them is a
**starting hypothesis**, not a validated setting — hence the re-probing described in §4.

## 4. Core architectural insight: one shared skeleton, not three separate systems

StableFlow, FluxSpace, and FreeFlux are the same **dual-branch generation pattern** with a different
injection policy plugged in:

1. Run two parallel denoising branches from the same initial noise: a **source/content branch** (unedited)
   and a **generation/edit branch**.
2. At chosen layers and timesteps, copy some tensor from the source branch into the edit branch.
3. Papers differ only in **which layers**, **what tensor** (attention-output / K,V / Q,K / values-only),
   and **what extra masking/projection** is applied.

| Paper | Tensor injected | Layer selection rule | Extra step |
|---|---|---|---|
| StableFlow | attention-layer output | "vital" layers (ablate-and-measure) | latent nudging for inversion |
| FreeFlux | K,V (position-dep.) / K,V (content-dep.) / V-only (all layers) | RoPE-dependency probing, 3 layer sets | cross-attention + SAM-2 mask for bg task |
| FluxSpace | attention output, orthogonally projected | all joint-attention layers | self-supervised attention mask, null-prompt projection |

**Conclusion**: build one shared dual-branch schnell sampler with a pluggable injection policy per task,
rather than three separate implementations. SVD-Style is structurally different (feature-space SVD
blending, not attention injection) and becomes a fourth module: a content branch + a style branch instead
of content + edit.

## 5. Empirical finding: schnell's own semantic-sensitivity profile

Ran a per-layer ablation study on FLUX.1-schnell: for each layer, ablate it and measure DINOv2
`|sim_full − sim_ablated|`, broken out per semantic attribute (colour, style, material, texture, shape,
layout, object), across both double-stream (0–18) and single-stream (0–37) blocks.

### Findings

1. **Extreme front-loading.** Every attribute except shape spikes at double-stream layers 0–2 (layer 1
   peaks at ~0.55 for "object", ~0.51 texture, ~0.50 style — the highest values on the whole chart), then
   collapses 3–5x by layer 3 and stays flat (~0.05–0.15) through the rest of both streams.
2. **Layer 2 matters more on schnell than on dev.** The StableFlow paper treats layer 2 as borderline/
   removable from the dev vital set. On schnell it's nearly as strong as layer 1 (~0.42 object, ~0.37
   style) — should be included, not optional, for schnell.
3. **Dev's late vital layers don't transfer.** StableFlow's dev vital layers 17–18 (double-stream) show
   **no elevation** on schnell — flat, same as every non-vital layer. The dev-derived vital-layer set does
   not carry over for these layers.
4. **Shape is structurally different from every other attribute.** It doesn't collapse after layer 2 —
   it holds a persistent, noisy ~0.10–0.17 band across nearly all double-stream layers and stays
   comparably elevated across all 38 single-stream layers (including the single highest single-stream
   point, ~0.145 at layer 1). Shape information is smeared across the network, not localized.
5. **Texture has a secondary, later re-emergence.** Beyond its layer-0 peak (~0.41), texture shows real
   secondary bumps at single-stream layers 3–4 (~0.10–0.11), 9 (~0.113), and 34 (~0.09). Layer 34
   corresponds to StableFlow's combined dev index 53 (one of their vital layers) — so this specific
   dev-predicted layer partially validates on schnell, but **only for texture**, not as a general vital
   layer.
6. **Single-stream layer 3 is a genuine structural hotspot.** Layout spikes to ~0.125 there (second-highest
   point in the whole single-stream plot), with material and texture also elevated simultaneously, while
   shape and colour are comparatively low — a distinct compositional layer separate from the early
   "everything" bottleneck.

### Practical consequence

The dev-derived layer sets from all three FLUX papers need to be **replaced, not reused**, for schnell.

## 6. Layer tier assignments

### 6a. Dev-validated layer sets (primary — from paper implementations)

**Tier A — content-similarity-dependent (FreeFlux ICCV 2025, RoPE frequency analysis):**
Combined indices `[0, 7, 8, 9, 10, 18, 25, 28, 37, 42, 45, 50, 56]`
Used for: non-rigid editing (freeze appearance), object replacement (global injection), attribute editing.

**Object-addition hotspot layers (FreeFlux, position-dependent / layout):**
Combined indices `[1, 2, 4, 26, 30, 54, 55]`
Used for: Phase 1 K,V capture in ObjectAdditionPolicy.

**Tier B — all 57 layers:**
Used for: background replacement (value-only), shape-linked attribute edits.

### 6b. Schnell-derived tiers (secondary — from §5 ablation study, kept as research data)

**Schnell Tier A — "Appearance" layers** (schnell ablation):
`double-stream {0, 1, 2}` + `single-stream {3, 9, 34}` (texture/layout secondary taps)

**Schnell Tier B — "Structure" layers** (shape/pose — smeared across all layers):
most/all layers in both streams.

Every task becomes a combination of which tiers get injected, where, and with what masking.

## 7. Updated per-task pipeline

| Task | What to inject | Layers | Reasoning / what changed from the raw dev-paper recipe |
|---|---|---|---|
| **Non-rigid editing** | Image-token K,V injection at content-similarity layers | **Dev (primary):** `{0,7,8,9,10,18,25,28,37,42,45,50,56}` (FreeFlux RoPE analysis). **Schnell hypothesis:** `{0,1,2}` double + `{3,9,34}` single | FreeFlux's dev set is RoPE-validated. Injecting only appearance layers freezes texture/colour while pose deforms freely through the uninjected layers. |
| **Object replacement** | Image-token K,V at appearance layers globally; outside-mask K,V at remaining layers | **Dev:** Tier A global + all-layers masked. **Schnell:** Drop StableFlow `{17,18,25,28,53,54,56}` — no elevation on schnell | Masking Tier B prevents the source object's shape from fighting the new object's replacement shape. |
| **Object addition** | Two-phase: Phase 1 captures K,V at layout-hotspot layers with source-only generation; Phase 2 injects outside placement mask | **Dev (primary):** `{1,2,4,26,30,54,55}` (FreeFlux position-dependent layers). **Schnell hypothesis:** `{0}` double + `{3}` single | FreeFlux's `{1,2,4,26,30,54,55}` is from positional RoPE dependency analysis — paper-validated on dev. Phase 1 guidance_scale must match Phase 2. |
| **Background replacement** | Value-only injection inside foreground mask, all layers | all 57 (both dev and schnell) | Task policy is layer-insensitive by design. Efficiency ablation: trimming to Tier A may lose nothing. |
| **Fine-grained attribute editing** | Orthogonal-projection edit on image-token keys at Tier A | **Dev:** `{0,7,8,9,10,18,25,28,37,42,45,50,56}`. **Schnell:** `{0,1,2}` + `{3,9,34}`. For shape-linked edits use all 57 | FluxSpace uses all 19 joint-attention layers. Orthogonal projection applies to image tokens only — text tokens must be left unchanged. |
| **Global style edit** | Pooled-CLIP coarse conditioning | unchanged mechanism | Style/colour channels peak hardest at early layers; effect likely locks in fast. |
| **Style personalization (SVD-Style-inspired module)** | PFB at double-stream block 1 output (image tokens); SAC (image-token Q,K) at block 1 | **double-stream layer 1** (both dev and schnell) | Layer 1 is where object (0.555), texture (0.51), and style (0.50) all peak simultaneously — the same signature SVD-Style used to identify Infinity's F₃. SAC must only replace image-token Q,K (positions `txt_len:`), not text tokens. |
| **Real-image inversion (latent nudging)** | unaffected by layer assignment | — | StableFlow's λ=1.15 nudging is a dev-derived value; needs validation on schnell. |

## 8. Open question / next experiment

The layer-axis collapse above is convincing, but there's no evidence yet of an equivalent **timestep-axis**
collapse across schnell's 1–4 denoising steps. Proposed next probe: repeat the same ablate-and-measure
DINOv2 methodology, holding layers fixed and instead ablating/freezing individual *timesteps*, using the
same per-attribute breakdown.

If timestep collapse is similarly extreme (content mostly decided in step 1):
- FreeFlux's Reasoning-before-Generation strategy (needs ≥7 steps before its mask-guided restart) is
  fundamentally incompatible with schnell's native budget and needs a real redesign, not just a layer
  swap — likely candidates: force extra steps specifically for object addition, or compress the two-phase
  logic into schnell's available steps and validate it doesn't collapse.
- Latent nudging for inversion may need to concentrate its correction in step 1 rather than being applied
  uniformly across the trajectory.

## 9. Build order

1. **Shared dual-branch schnell sampler.** Generic two-branch sampler (same seed, N steps, hooks to inject
   at arbitrary layers/timesteps) before touching any task-specific policy. Every task below reuses this,
   and the ablate-and-measure probing infrastructure (§5, §8) is built from the same hooks.
2. **StableFlow-style policy — object replacement + inversion baseline.** Simplest single-tensor,
   single-layer-set injection; also establishes the latent-nudging inversion path every real-image task
   needs.
3. **FreeFlux-style policies — non-rigid, object addition, background replacement.** Do background
   replacement first (least step-sensitive, safest on schnell's short budget), then non-rigid, then object
   addition last (needs the two-phase restart, most exposed to the open timestep question in §8).
4. **FluxSpace-style policy — fine-grained attribute + global style.** Validate that the null-prompt
   orthogonal projection still yields a meaningful signal under schnell's guidance distillation before
   trusting dev-tuned `λ_fine`/`λ_coarse` values; recalibrate empirically if the projected edit vector
   collapses toward zero.
5. **SVD-Style-inspired module.** Built directly on the pivotal-layer finding in §7 (double-stream layer
   1). This is the one module with no existing FLUX precedent — genuine adaptation work, not application
   of a published recipe.
6. **Shared eval harness**, running alongside every step above: CLIP-T, CLIP-I, DINO, PSNR (for
   preservation tasks) — so all modules report on identical metrics for a single unified dissertation
   comparison table.
