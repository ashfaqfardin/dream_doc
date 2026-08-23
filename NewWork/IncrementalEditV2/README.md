# flux_seqedit — self-masked sequential editing on FLUX.1-dev

Safe-floor implementation of layer-wise-memory sequential editing (Kim et al.,
CVPR 2025) ported to FLUX.1-dev, with the user sketchpad replaced by an
attention-derived mask. Prompt-only: FLUX chooses object placement, repulsed
from occupied regions; the mask is harvested from the model's own attention.

## Build checklist (status)

| # | Component | File | Status |
|---|-----------|------|--------|
| 1 | Mask ↔ packed-token map (+ T1/T2 gate) | `packing.py` | **done, tested** |
| 2 | Output-norm extraction (SDPA-safe) | `processor.py` | **done** (GPU) |
| 3 | Pre-softmax memory-repulsion prior | `processor.py` | **done** (GPU) |
| 4 | Normalized Otsu gate + hard fallback | `masking.py` | **done, tested** |
| 5 | Generation-order occlusion + memory | `memory.py` | **done, tested** |
| — | Denoising-loop orchestration + BCG | `pipeline.py` | **done** (GPU) |
| — | End-to-end session simulation | `tests/` | **done, tested** |

23 tests pass with numpy/scipy only — no GPU required for the logic core.

## What runs where

The **logic core** (packing, masking, Otsu, memory/occlusion) is pure
numpy/scipy and is fully unit-tested in this repo without torch or a GPU. That
is deliberate: it is the highest-risk, most reusable part, and the tests catch
the failure modes that a GPU would only surface after minutes of model loading
(token-order transposes, truncated occlusion cascades, scale-dependent Otsu).

The **FLUX runtime** (`processor.py`, `pipeline.py`) requires torch + diffusers
+ FLUX.1-dev weights. It compiles and is written against the mainline diffusers
API; run it on a GPU box via `demo.py`.

## Run the tests (no GPU)

```bash
pip install numpy scipy pytest
PYTHONPATH=. python -m pytest tests/ -v
```

Key tests:
- `test_packing.py::test_T1_*` — round-trip identity (catches transpose)
- `test_packing.py::test_T2_*` — localization (catches packing-order mismatch)
- `test_masking_memory.py::test_three_layer_cascade_reocclusion` — a late edit
  must re-occlude an early one through an intervening layer (catches a truncated
  `m_j − Σ m_l` cascade)
- `test_integration_sim.py::test_full_three_edit_session` — a full 3-edit session
  driven by synthetic attention, incl. the Otsu hard fallback for diffuse concepts

## Run on GPU

```bash
pip install -r requirements.txt
huggingface-cli login          # FLUX.1-dev is gated
python demo.py
```

## Design decisions (locked, see build spec)

- **Primary mask signal = attention OUTPUT NORM**, not the attention map. FLUX's
  `FluxAttnProcessor2_0` uses fused SDPA and never materializes QK^T; the output
  norm is free, the weight matrix is not. The attention-map route is a
  probe-gated experiment, not part of the floor.
- **Joint attention, not cross-attention.** The signal is the image-token rows of
  one joint `[text+image]` attention; there is no separable cross map.
- **Packing map is step zero** and unit-tested before anything else, because a
  token-order bug is silent and poisons every downstream mask.
- **Occlusion = generation order** (newest on top), the base paper's proven
  `m_j − Σ_{l>j} m_l` cascade, with **raw masks stored** (not visible) so late
  edits re-occlude early ones correctly. Argmax-over-norm occlusion is explicitly
  out of scope (unvalidated saliency≠depth assumption).
- **FLUX.1-dev, not Fill** — Fill's from-step-zero binary mask conditioning would
  fight a dynamic self-derived mask.

## Out of scope here (probe-gated upside)

Do not build these into the floor until the probes pass:
- attention-map route (needs materialized weights),
- argmax-over-norm occlusion (saliency ≠ depth),
- pure free placement with no repulsion prior.

Two probes to run once the GPU path is live (they need the extraction
scaffolding, which now exists):
- **P1 localization timing** — earliest step the Otsu gate reliably fires per
  object; sets `early_frac`.
- **P2 argmax-depth correlation** — whether argmax-over-norm recovers true
  occlusion on size/emphasis-imbalanced overlapping pairs.

## Tuning knobs (`EditConfig`)

`early_frac`, `late_frac` (phase boundaries — flow-schedule dependent, tune don't
assume), `otsu_confidence`, `otsu_hard_cutoff_frac`, `penalty` (repulsion
strength), `collect_blocks` (which dual-stream blocks feed the norm signal —
default mid-late band 8–14), `bcg_late_floor` (seam-free blending).
```
