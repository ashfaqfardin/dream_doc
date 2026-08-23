# IncrementalEdit

Incremental, object-addressable scene editing on **FLUX.1-Kontext-dev** via
single-branch K/V injection. Build a scene up over separate commands, run at
any later time — add an object, recolor it, replace it, edit one of its
parts — while everything else in the frame stays put.

```
python run_incremental_edit.py init \
    --project-dir runs/driveway --prompt "an empty driveway at dusk" --seed 42

python run_incremental_edit.py add \
    --project-dir runs/driveway --object car_1 --noun car \
    --prompt "a red sports car parked in the driveway"

python run_incremental_edit.py attribute \
    --project-dir runs/driveway --object car_1 --prompt "make the car blue"

python run_incremental_edit.py replace \
    --project-dir runs/driveway --object car_1 --new-object cycle_1 --noun cycle \
    --prompt "a bicycle parked in the same spot"

python run_incremental_edit.py part-edit \
    --project-dir runs/driveway --object cycle_1 --part tire_1 --part-noun tire \
    --prompt "a spoked alloy wheel"

python run_incremental_edit.py remove \
    --project-dir runs/driveway --object cycle_1
```

Each call is a separate process — `runs/driveway/manifest.json` is what
makes `attribute`/`replace`/`part-edit` able to find "the car" (or "the
cycle's tire") days later without re-deriving anything from scratch. See
`pipelineInc.md` for the full design and the reasoning behind it.

**Running on Colab:** open `colab_run.ipynb` (Runtime > GPU, ideally
>=24GB) — it walks the exact sequence above end-to-end with inline image
previews, and is also the fastest way to find out whether the one
unverified guess in this codebase (`assert_reference_slice()`'s diffusers
method name) needs adjusting for your installed diffusers version.

## What this is (and isn't)

Read `pipelineInc.md` first — the changelog at the top explains what was
wrong with the original single-file draft and why this ended up as five
modules instead of one. Short version: FLUX.1-Kontext conditions on a
reference image *in-sequence* (concatenated onto the noisy latent at every
step), not via a second denoising trajectory the way FLUX.1-dev dual-branch
work does — so this pipeline is single-branch, and "K/V injection" means
reading a tensor slice already present in the same forward pass, not caching
one from a separate pass.

The three-zone injection model (background / shell / target) and the
five edit verbs (`add`/`attribute`/`replace`/`remove`/`part-edit`) are this
project's own synthesis. The primitives underneath them — K+V freeze,
K-only freeze (shape locked, appearance free), DAAM-style cross-attention
saliency for mask derivation — are not new; they're reused from
`NewWork/UltimateFlux/policies.py`, where they're already validated on
FLUX.1-dev. What's new here is applying them across a *scene hierarchy*
(background → parent object → part) instead of only a *layer* hierarchy,
and doing it single-pass because Kontext's architecture allows it.

## Files

| File | What | Testable here without a GPU? |
|---|---|---|
| `pipelineInc.md` | Design doc — read this first | n/a |
| `manifest.py` | Scene/object registry, JSON, atomic writes | Yes — pure Python |
| `mask_ops.py` | Token-mask arithmetic (rect/intersect/topk/zone split) | Yes — numpy only |
| `metrics.py` | latent MSE, PSNR, preservation gap, LPIPS (optional) | Yes — numpy only, torch duck-typed |
| `kontext_injection.py` | The actual attention processor + injection + DAAM masking | No — needs torch+diffusers+GPU |
| `run_incremental_edit.py` | CLI wiring the above together | Partial — argparse only |
| `test_cpu.py` | Unit tests for the testable pieces | Run it: `python test_cpu.py` |
| `requirements.txt` | pip deps (core + optional) | n/a |
| `colab_run.ipynb` | End-to-end Colab walkthrough of the example scene | This is where Stages 1–6 actually get their first real GPU run |

## Environment

```
pip install "diffusers>=0.32" transformers accelerate torch pillow numpy
pip install lpips sam2   # optional
```

Needs a >=24GB GPU. `FLUX.1-dev` and `FLUX.1-Kontext-dev` are both gated on
Hugging Face — accept each license and run `huggingface-cli login` first.
Only one pipeline is loaded per process (`init` loads `FLUX.1-dev`; every
other verb loads `FLUX.1-Kontext-dev`), so you don't need 2x the VRAM.

## What's verified vs. what isn't (read before trusting a run)

This was built in an environment with **no GPU and no torch install**.
What was actually checked:

- Every file parses (`ast.parse`) and `test_cpu.py` passes — run it
  yourself: `python test_cpu.py`.
- The Kontext architecture claims in `pipelineInc.md` §0 (in-sequence
  reference concatenation, R-RoPE type-dimension flag, single forward pass
  per step) were checked against the actual diffusers
  `pipeline_flux_kontext.py` source, not assumed.
- The injection primitives (K+V freeze / K-only freeze / DAAM saliency) are
  not new — they're the same math already validated on FLUX.1-dev in
  `NewWork/UltimateFlux/policies.py`.

What is **not** verified, and can't be from this environment:

- Whether TIER_A actually stays content-similarity-dominated under
  Kontext's extra RoPE axis (`pipelineInc.md` §4) — try `--vital-layers all`
  if edits leak into the background or the background won't hold.
- The exact diffusers method name probed in `kontext_injection.py`'s
  `assert_reference_slice` (`pipe._encode_vae_image`) — this is the single
  most likely thing to need adjusting for whatever diffusers version you
  install; the error message it raises on failure dumps
  `attn_processors` keys to help you re-derive the right call.
- Numeric defaults (`--inject-cutoff-frac 0.6`, `--inject-strength 1.0`,
  `--derive-step 7`, `--top-k-frac 0.08/0.35`) are carried over from
  FreeFlux/UltimateFlux's dev-tuned values as starting points, not
  Kontext-specific tuning.

If you hit a suppressed edit (background perfect, requested change didn't
happen) or a leaking one (background drifting), start with
`pipelineInc.md` §3 and §6's VERDICT block — both failure modes have a
specific knob to move, not a code change.

## Known limitations

- Semantic masks (DAAM saliency) can collide for visually similar objects
  in the same scene (two cars). Pass `--mask <path.png>` to override.
- `part-edit`'s mask is `intersect(saliency_for_part_noun, parent_mask)` —
  if the part isn't visually distinguishable from the rest of its parent at
  the working resolution, this can come back empty (the CLI raises rather
  than silently editing nothing).
- Long edit chains compound drift even when each individual step's PSNR
  looks fine — `pipelineInc.md` §6 tracks both `step_psnr` (vs. immediate
  parent) and `cumulative_psnr` (vs. revision 0) for this reason. If
  cumulative PSNR degrades over many edits, consider re-rooting from a
  decoded canvas rather than chaining indefinitely.
