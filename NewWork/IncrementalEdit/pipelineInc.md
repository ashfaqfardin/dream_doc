# Build spec: Incremental, object-addressable scene editing on FLUX.1-Kontext

## Changelog vs. the original draft (read this first)

The original draft ported the **FLUX.1-dev dual-branch** K/V-injection recipe
(FreeFlux/StableFlow style: run a *second* clean "source" denoising trajectory,
cache its K/V, overwrite the edit branch's K/V from that cache) onto Kontext
almost unchanged. Two verification passes against the real Kontext architecture
(diffusers `pipeline_flux_kontext.py`) and against the user's actual intent
found four problems serious enough to redesign around, not patch:

1. **The "cache a clean pass" stage doesn't apply to Kontext and should be
   deleted, not kept as a stage.** In FLUX.1-dev dual-branch work, the source
   K/V has to come from a *separate denoising trajectory* because there's no
   other way to get "the clean image's K/V at diffusion step t" — you have to
   generate it. Kontext is architecturally different: the reference (canvas)
   image is VAE-encoded **once**, un-noised, and re-concatenated onto the
   noisy generation tokens at **every single step of the same forward pass**
   (`latent_model_input = torch.cat([latents, image_latents], dim=1)`, confirmed
   in `pipeline_flux_kontext.py`). Its K/V at every vital layer are already
   sitting in the same attention call as the generation tokens' K/V — reading
   them costs a tensor slice, not a second trajectory. Building a "Stage 1:
   cache" pass anyway would double the compute for no benefit and, worse,
   would be caching K/V computed under a *different* attention state (no
   generation-token neighbors) than the K/V actually present during the edit
   step — a subtle correctness bug, not just waste. **Fix**: single-branch,
   single-pass. "Caching" becomes a structural assertion that the reference
   token slice is where we expect it, checked once at pipeline setup.

2. **No mechanism for incremental, per-object editing.** The user's actual
   requirement — add a car, later recolor *that same car*, later replace it
   with a cycle, later retint *that cycle's tire* — needs objects to be
   addressable across separate CLI invocations run hours or days apart. A
   single coarse `--edit-region right/left/top/bottom/center` rectangle (the
   original draft's only masking mechanism) cannot express "the car" as a
   region, let alone "the car's tire." **Fix**: a persistent scene manifest
   (§2) that stores one mask per named object, derived semantically (not by
   hand-drawn rectangle) and reusable by every future edit call.

3. **No distinction between the edit verbs.** Recolor, replace, remove, and
   part-edit are not the same operation with different prompts — they need
   *different injection strength inside the target region itself*, not just a
   frozen/free split. Recoloring a car must keep the car's silhouette (lock
   K, free V, inside the mask) or you get a car-shaped color splotch, not a
   red version of the same car. Replacing it must NOT lock K inside the mask,
   or the cycle is forced into the car's K-derived attention pattern and comes
   out car-shaped. The original draft had exactly one knob (`inject-strength`)
   applied uniformly. **Fix**: a three-zone model per edit (§3) that reuses
   UltimateFlux's already-validated K+V-freeze / K-only-freeze / no-injection
   primitives (see `NewWork/UltimateFlux/policies.py`), applied over scene
   geometry instead of layer geometry.

4. **Wrong default model, unvalidated layer-transfer, and one open risk
   flagged instead of hidden.** Canvas generation should default to plain
   `FLUX.1-dev` (Kontext is not trained for from-scratch text→image; it's an
   edit model). And Kontext's RoPE is not identical to dev's: the reference
   image tokens get a distinguishing offset in the *first* RoPE coordinate
   (`image_ids[..., 0] = 1`, i.e. "R-RoPE" per the Kontext tech report,
   arXiv:2506.15742) that dev's txt2img RoPE never had. FreeFlux's TIER_A
   layer set was derived purely from dev's H/W-frequency analysis and has
   never been re-validated against this extra type-dimension. It is used
   here as the **default hypothesis** (matching how `UltimateFlux/Pipeline_Plan.md`
   treats dev→schnell layer transfer — carried over, flagged, not asserted),
   with `--vital-layers all` as the fallback if TIER_A misbehaves under
   Kontext's modified RoPE.

Everything else in the original draft's philosophy — determinism-first,
verify-every-stage-with-a-number, honest about what can't run without a GPU —
is kept and extended to the new stages.

---

## 0. Mental model

**Two models, two roles, loaded one at a time (never together):**

| Stage | Model | Pipeline class |
|---|---|---|
| Canvas generation (once, or full re-roll) | `black-forest-labs/FLUX.1-dev` | `FluxPipeline` |
| Every edit op (add/recolor/replace/remove/part-edit) | `black-forest-labs/FLUX.1-Kontext-dev` | `FluxKontextPipeline` |

**How Kontext actually conditions on the canvas** (confirmed against
`diffusers/pipelines/flux/pipeline_flux_kontext.py`, not assumed):

- The canvas is VAE-encoded once into `image_latents` — clean, never noised.
- At every denoising step, `image_latents` is concatenated onto the evolving
  noisy `latents` along the sequence dimension: image-token sequence =
  `[generation_tokens (N_gen) | reference_tokens (N_ref)]`. Text tokens are
  handled separately by each double-stream block and joined for joint
  attention, same as plain FLUX.
- Reference tokens carry the **same H/W spatial coordinates** as their
  corresponding generation-grid position (so canvas pixel (i,j) and the
  generation token that will eventually decode to (i,j) share RoPE
  coordinates 1 and 2) but a **different value in RoPE coordinate 0**
  (`0` for generation tokens, `1` for reference tokens — this is the R-RoPE
  "type" axis). This is what lets the model tell "the thing I'm drawing" from
  "the thing I'm copying" apart even where they spatially coincide.
- Only the generation-token slice is updated by the scheduler each step; the
  reference slice is re-attached unchanged next step.

**Why inject K/V at all, if Kontext already conditions on the reference?**
Kontext's conditioning is *soft*: generation tokens attend to reference tokens
via ordinary cross-token attention, so preservation is a learned tendency, not
a guarantee — the background is free to drift over a chain of edits (exactly
the drift the user described wanting to avoid across "add car → recolor car →
replace with cycle → retint tire"). Overwriting a frozen generation token's
**own** K/V with the reference token's K/V forces that token's attention
output to be *computed from* the reference content, not merely influenced by
it. This is the same justification FreeFlux gives for FLUX.1-dev; it now
applies within a single Kontext forward pass instead of across two branches.

**Single denoising loop, no dual branch.** Because the reference is already
in-sequence, there is no second "source" trajectory to run in parallel — drop
`generate_dual_branch`'s B=2 pattern entirely for this pipeline. Injection is
applied by reading `k_ref = k[..., N_gen:, :]` (this step's own reference
slice) and selectively overwriting `k_gen = k[..., :N_gen, :]` before SDPA, in
the same forward call, at vital layers.

---

## 1. Scene manifest — the incremental-editing backbone

Everything in §0 explains one edit call. What makes editing *incremental* is
persisting object identity across separate CLI invocations, so `recolor` run
today can find "the car" that `add` created last week.

`<project_dir>/manifest.json`:

```json
{
  "resolution": [1024, 1024],
  "revisions": [
    {"id": 0, "image": "canvas_v0.png", "op": "init",   "prompt": "an empty driveway at dusk", "parent": null},
    {"id": 1, "image": "canvas_v1.png", "op": "add",     "prompt": "a red sports car in the driveway", "parent": 0, "object": "car_1"},
    {"id": 2, "image": "canvas_v2.png", "op": "attribute","prompt": "make the car blue",        "parent": 1, "object": "car_1"},
    {"id": 3, "image": "canvas_v3.png", "op": "replace",  "prompt": "a bicycle in the same spot","parent": 2, "object": "cycle_1", "replaces": "car_1"},
    {"id": 4, "image": "canvas_v4.png", "op": "part_edit","prompt": "a spoked alloy wheel",      "parent": 3, "object": "tire_1",  "parent_object": "cycle_1"}
  ],
  "objects": {
    "car_1":   {"noun": "car",   "mask": "masks/car_1.png",   "created_at": 1, "retired_at": 3, "parent_object": null},
    "cycle_1": {"noun": "cycle", "mask": "masks/cycle_1.png", "created_at": 3, "retired_at": null, "parent_object": null, "replaces": "car_1"},
    "tire_1":  {"noun": "tire",  "mask": "masks/tire_1.png",  "created_at": 4, "retired_at": null, "parent_object": "cycle_1"}
  }
}
```

Rules:
- **Masks are token-resolution PNGs** (`h_lat × w_lat` grid, nearest-upscaled
  for display), stored once per object at the revision it was
  created/last-refined, reused by every later edit that targets that object
  by name. They are only re-derived when the object's own appearance changes
  enough to invalidate them (`replace`, and optionally `attribute` if
  `--refresh-mask` is passed).
- **Replacing an object retires it** (`retired_at` set) rather than deleting
  it — the manifest is an append-only edit history, matching Stage 6's
  drift-tracking need (§6) and letting the user re-render from any revision.
- **Removing an object retires it with no replacement** and clears
  `parent_object` links for any children (a removed car's mirror sub-object,
  if one existed, becomes orphaned — flagged, not silently deleted).
- **Part objects require `parent_object`** and their mask must be derived
  *inside* the parent's mask (§3, part-edit) — this is what stops "tire"
  from matching some other tire-shaped thing elsewhere in the scene.
- A project directory is fully self-contained: `manifest.json`, `canvas_v*.png`,
  `masks/*.png`. Nothing depends on wall-clock ordering beyond the `parent`
  chain, so this is safe to resume in a new process at any time — the actual
  incrementality the user asked for.

---

## 2. Edit verbs and their three-zone injection

Every edit call operates on: `background = ¬(parent object's mask, if any editing an existing object)`, `shell = parent_object mask ∖ target mask` (only nonempty for part-edit), `target = the region actually changing`. All are token masks over the **generation-token slice only** (`[:N_gen]`); the reference slice is never masked, it's always fully-available context.

| Zone | Meaning | Injection |
|---|---|---|
| Background | everything not being touched by this call | **K+V freeze**: `k_gen[bg] = k_ref[bg]`, `v_gen[bg] = v_ref[bg]` at vital layers, all steps (subject to `--inject-cutoff-frac`) |
| Shell | rest of the parent object, when editing one of its parts | **K-only freeze**: `k_gen[shell] = k_ref[shell]`; V left free | 
| Target | the thing actually changing | verb-dependent (below) |

| Verb | Target zone injection | Notes |
|---|---|---|
| `add` | free (no injection) until placement is *known*, then frozen everywhere else — see reasoning sub-pass below | mask doesn't exist yet; must be derived mid-generation |
| `attribute` (recolor / restyle in place) | **K-only freeze**, V free | keeps silhouette, lets appearance change — this is "same car, new color" |
| `replace` | **no injection** | must NOT inherit old object's K, or the new object is forced into the old one's shape |
| `remove` | **no injection**, driven by a background-completion prompt (`--fill-prompt`, auto-derived from the canvas prompt if omitted) | Kontext's own reference conditioning supplies plausible surrounding context; no compositing |
| `part_edit` | **K-only freeze**, V free (same rationale as `attribute`, scoped inside the parent) | shell zone is what makes this different from a plain `attribute` call on the parent |

**`add`'s placement sub-step** (replaces `ObjectAdditionPolicy`'s two-pass
reasoning design from FLUX.1-dev — cheaper here because Kontext's frozen
zones reconstruct the reference "for free" instead of wasting a pass):
run the *same* single-branch loop with background fully frozen everywhere
(nothing can move yet); at a chosen `--derive-step` (default step 7),
capture DAAM-style `Q_img[gen] × K_concept[new noun]` cross-attention
(reusing `_DAAmAttnProcessor`'s logic from `UltimateFlux/policies.py`,
scoped to the generation-token slice) to get a saliency map; threshold to a
token mask; from that step onward, treat that mask as `target` (free) and
everything else as `background` (frozen) for the remainder of the loop. One
continuous forward pass, not two.

**Layer set** for all freeze/K-only ops: `--vital-layers` defaults to
FreeFlux's TIER_A = `{0,7,8,9,10,18,25,28,37,42,45,50,56}` (§4 caveat on why
this is a hypothesis, not a fact, on Kontext). `--vital-layers all` selects
all 57 as the safer-but-more-suppressive fallback.

---

## 3. The suppression trap — unchanged risk, same knobs

Freezing K+V at every vital layer for every step still risks Kontext simply
reproducing the reference with no visible edit (FreeFlux's documented
Suppression Phenomenon, sharper here because the reference is always
in-sequence). Same two knobs as the original draft, now scoped per-zone
(background always fully injected; target/shell zones are governed by the
table in §2, not by these knobs):

- `--inject-cutoff-frac` (default 0.6): background freeze active only for the
  first fraction of steps, then released.
- `--inject-strength` (default 1.0): blend factor for the background freeze.

Expected failure mode if these are too aggressive for a given scene: the
requested change doesn't show up at all, or (for `add`) the new object
appears faint/ghosted. That's the signal to lower them, not a bug.

---

## 4. Layer-transfer caveat (must ship in the README, not buried)

TIER_A `{0,7,8,9,10,18,25,28,37,42,45,50,56}` is validated (FreeFlux,
ICCV 2025) for **FLUX.1-dev's** RoPE, which only varies over H/W. Kontext
adds a third, binary "reference-vs-generation" RoPE coordinate absent from
that analysis. It is plausible the same 13 layers remain content-similarity-
dominated with respect to the new axis too (the H/W-frequency structure that
makes them TIER_A is presumably untouched by fine-tuning a new coordinate
in), but this has not been re-derived for Kontext specifically. Treat TIER_A
here as a carried-over starting point, exactly as `UltimateFlux/Pipeline_Plan.md`
treats dev-derived layer sets on schnell: a hypothesis with a stated fallback
(`--vital-layers all`), not a fact.

---

## 5. Determinism and the attention-processor contract

Unchanged from the original draft and still non-negotiable:

```python
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
torch.use_deterministic_algorithms(True, warn_only=True)
generator = torch.Generator("cuda").manual_seed(seed)
```

The injecting processor must reproduce stock `FluxAttnProcessor` math exactly
in `off` mode (same projections, `norm_q`/`norm_k`, RoPE, output split) — this
is what makes the Stage 1 `MSE==0` canvas-reproducibility check trustworthy,
and it is now also what makes the Stage 2 reference-slice assertion (below)
trustworthy: if `off` mode isn't bit-identical to baseline, a shape assertion
passing doesn't prove the injection math is correct, only that it didn't
crash.

Discover layer count and processor keys at runtime
(`pipe.transformer.attn_processors`); parse indices out of
`transformer_blocks.N.attn.processor` / `single_transformer_blocks.N.attn.processor`
— never hardcode block counts.

---

## 6. The six verification stages (per edit call)

| Stage | Does | Prints (pass condition) |
|---|---|---|
| 1. Canvas | `FLUX.1-dev`, fixed seed, generate + **regenerate** | `latent MSE(runA,runB)` — must be exactly `0.0` |
| 2. Reference-slice assertion | one forward-shape check: `N_gen + N_ref == total image tokens`, `N_ref == h_lat*w_lat` of the canvas, `img_ids[N_gen:, 0] == 1` and `img_ids[:N_gen, 0] == 0` | `reference slice verified (N/N vital layers)` |
| 3. Inject | edit pass, three-zone overwrite per §2, background cutoff/strength applied | injection mode logged per (layer, step) |
| 4. Probe | per step: is background held, is target actually moving | `bg-zone latent MSE` (trend, not monotonic — see caveat below), `attention leakage` target→background |
| 5. Decide | interpret probe → tune knobs | printed guidance in verdict block |
| 6. Commit + drift | save new canvas revision, update/derive mask(s), write manifest | `step PSNR` (vs immediate parent), `cumulative PSNR` (vs revision 0), `LPIPS` if available |

**Probe caveat (corrects the original draft's framing):** unlike the
dual-branch case, the generation-token latent starts from pure noise, not a
noised copy of the canvas — early-step `masked_latent_mse` against the clean
canvas latent will be large almost by construction and is not meaningful
before the injection has had steps to act. Report it as a **trend over the
injection window**, converging by the final active step, not as a value
expected to monotonically fall from step 0.

**Drift is now two numbers, not one**, because edits chain: `step PSNR`
(this edit vs. its immediate parent — was *this* call clean) and
`cumulative PSNR` (current canvas vs. revision 0 — is the *scene* drifting
over a long edit chain, e.g. background degrading over 10 edits even if each
individual step looked fine). Both belong in `run_summary.json`.

**The single most important derived number, per zone that should be frozen
this call:** `preservation gap = background_PSNR - whole_image_PSNR`.
Positive = frozen region held while target changed (PASS). `≤0` = mask
mapping wrong or injection mistuned for this verb.

---

## 7. CLI surface

```
init         --prompt --seed --steps --guidance --resolution --project-dir
add          --project-dir --object <name> --noun <word> --prompt --derive-step --top-k-frac
attribute    --project-dir --object <name> --prompt [--refresh-mask]
replace      --project-dir --object <name> --noun <word> --prompt
remove       --project-dir --object <name> [--fill-prompt]
part_edit    --project-dir --object <parent-name> --part <name> --part-noun <word> --prompt

Shared:
--canvas-model-id     default black-forest-labs/FLUX.1-dev      (init only)
--edit-model-id       default black-forest-labs/FLUX.1-Kontext-dev  (all other verbs)
--seed                default 42
--steps               default 28
--guidance            default 2.5
--dtype               bf16 | fp16
--inject-strength     default 1.0     (background zone only)
--inject-cutoff-frac  default 0.6     (background zone only)
--vital-layers        comma list, overrides the default
--vital-lo/--vital-hi override; unset by default (TIER_A is the default, not a band)
--mask                user-supplied PNG, overrides semantic derivation for this call
--use-sam2            SAM2 fallback if DAAM-style attention derivation is weak
```

---

## 8. Output contract

Per edit call, write `<project_dir>/runs/rev<N>_summary.json` with: full
config, `stage1_latent_mse`, `stage1_reproducible`, `stage2_reference_slice_ok`,
`stage4_final_bg_latent_mse`, per-step probe (step, bg_mse, leak, mode),
`step_psnr`, `cumulative_psnr`, `lpips` (or `"n/a"`). Save `canvas_v<N>.png`
and, if the object's mask changed, `masks/<object>.png`. Update
`manifest.json` last, only after the image is written successfully.

Final **VERDICT** block, same spirit as the original draft:

```
Stage 1 canvas reproducible : PASS/FAIL
Stage 2 reference slice     : PASS/FAIL   (N/N vital layers)
Stage 4 bg zone final MSE   : <value>
Interpret: bg MSE ~0 AND target changed -> injection works for this verb.
           target unchanged -> lower cutoff-frac/strength (suppression) or
                                check the mask isn't accidentally covering it.
           background drifts (step PSNR low) -> raise cutoff-frac/strength,
                                or the mask↔token mapping is wrong.
           cumulative PSNR drops steadily across a long edit chain even
           though each step_psnr looks fine -> compounding drift; consider
           periodically re-rooting (start a fresh `init` from a decoded
           canvas) rather than chaining edits indefinitely.
```

---

## 9. Robustness / honesty rules

- Wrap each verb's `main()` in try/except; on failure print a hint that
  projection names (`to_q`, `add_q_proj`, `norm_added_k`) and the exact
  reference-token concatenation order shift between diffusers releases —
  dump `attn_processors` keys and the observed `N_gen`/`N_ref` split, and
  adapt from there.
- VAE decode degrades gracefully: try `_unpack_latents` → unscale → decode →
  postprocess; on failure skip pixel metrics with a warning, keep latent
  metrics.
- Never let a missing optional dep (`lpips`, `sam2`) crash a run.
- Assert-and-fail-loud on shape mismatches (mask vs. token count, `N_ref` vs.
  expected canvas token count, parent-object mask missing when a part-edit
  references it). Silent coercion here produces confident wrong numbers.
- Manifest writes are last-step and atomic (write to a temp file, then
  rename) — a crash mid-edit must never leave `manifest.json` pointing at an
  image that was never written.

---

## 10. Validate before declaring done

No GPU in the build environment (confirmed: no `torch` install either), so
none of Stages 1–6 can be run for real here. What can and must be validated:

1. `python -c "import ast; ast.parse(open(f).read())"` for every file —
   parses clean.
2. Pure-function unit tests on CPU (numpy/PIL only, no torch — confirmed
   available in this environment):
   - Manifest round-trip: write → read → objects/revisions match, orphaning
     on `remove`/`replace` behaves per §1.
   - Token-mask math: rectangle and intersection (part-in-parent) masks
     return the right frozen fraction and flatten to `h_lat*w_lat`.
   - `region_psnr` with an all-frozen mask equals `image_psnr`;
     `image_psnr(a,a) == inf`.
3. State plainly in the README that Stage 1/2/4/6's actual pass conditions
   (the numbers, not the code paths) can only be confirmed on a GPU with
   `diffusers`+`torch` installed, and give the exact run command to do so.

Do not fabricate metric values or claim any stage "works" from this
environment. The job here is to make sure the code *will* produce correct
numbers on the target machine, and to be explicit about the line between
what was verified now (parses, pure-function correctness, architectural
claims checked against actual diffusers source) and what is unverified until
then (the injection math's numerical behavior, the suppression-trap knobs'
actual defaults, whether TIER_A transfers).

---

## 11. Environment note for the user

```
pip install "diffusers>=0.32" transformers accelerate torch pillow numpy
pip install lpips sam2   # optional
# needs a >=24GB GPU; only one pipeline (dev OR Kontext-dev) loaded at a time.
# Both are gated on Hugging Face — accept the license and `huggingface-cli login`.
```

Reference basis (one line each): **FLUX.1-Kontext** (arXiv:2506.15742,
in-context editing via in-sequence reference conditioning + R-RoPE — this is
what makes single-pass injection possible instead of dual-branch);
**FreeFlux** (arXiv:2503.16153, ICCV 2025 — TIER_A layer set, K-only vs K+V
injection semantics, the Suppression Phenomenon); **StableFlow**
(arXiv:2411.14430, CVPR 2025 — vital-layer ablation methodology, the
principled way to re-derive TIER_A for Kontext if the transfer hypothesis in
§4 fails); **ConceptAttention/DAAM** (arXiv:2502.04320 — Q×K cross-attention
subject saliency, reused here for semantic mask derivation instead of hand
rectangles). The three-zone injection model (§2) and the object-registry
pattern (§1) are this project's own synthesis, built to satisfy the
incremental multi-object editing requirement — not drawn from a single paper.
