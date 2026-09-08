# Result Analysis: Final Local Collage Harmonization

## Experimental scope

This report analyses the output of `final_local_collage_harmonization.py` in `QwenMaskFinal_04`. The evaluation contains 10 scenes, three sequential insertions per scene, and therefore 30 editing turns. The experiment uses Qwen-Image-Edit-2509 with the 8-step Lightning adapter, a fixed seed of 42, 1024 x 1024 processing, RMBG-2.0 reference cutouts, and manually supplied placement rectangles.

The evaluated procedure is training-free. At turn \(i\), it (1) extracts a foreground cutout from reference \(O_i\), (2) scales and pastes the cutout into placement rectangle \(L_i\), (3) asks Qwen to harmonize a square local crop, and (4) copies the result back only through an alpha-derived interaction mask \(M_i\):

\[
I_i=(1-M_i)\odot I_{i-1}+M_i\odot \hat I_i.
\]

Consequently, non-disturbance outside \(M_i\) is an explicit property of the compositing algorithm, not an emergent property learned by Qwen.

## Metrics and interpretation

| Requirement | Reported metric | Desired direction | What it measures | Important limitation |
|---|---:|:---:|---|---|
| Object fidelity | DINOv2 cosine similarity | Higher | Semantic/appearance similarity between the reference cutout and localized final object | Can remain high when the pasted reference is retained but is not naturally integrated |
| Object fidelity | Colour-histogram similarity | Higher | Agreement of foreground colour distributions | Insensitive to geometry and spatial arrangement |
| Object fidelity | Edge-structure similarity | Higher | Agreement between edge-magnitude distributions | Distributional score; it does not establish pixel or part correspondence |
| Spatial localization | Object support inside rectangle | Higher | Fraction of pasted alpha support within the requested rectangle | Guaranteed by deterministic placement |
| Spatial localization | Changed pixels inside rectangle | Higher | Fraction of detected changes occurring in the rectangle | A halo may legitimately extend outside the rectangle for shadows or blending |
| Edit extent | Changed fraction of image | Lower, conditionally | Fraction of the full image changed by more than 8 intensity levels | A tiny edit is not necessarily a successful edit |
| Non-disturbance | Outside-mask MAE / changed fraction | Lower | Pixel error outside the permitted interaction mask | Exactly zero is guaranteed by hard RGB composition |
| Cross-turn stability | Prior-object changed fraction / MAE | Lower | Drift of earlier visible object pixels in later turns | Hard restoration and support subtraction strongly constrain this result |

## Aggregate quantitative results

Values are descriptive statistics across all 30 editing turns.

| Metric | Mean | Std. dev. | Minimum | Maximum |
|---|---:|---:|---:|---:|
| DINOv2 identity similarity | 0.9566 | 0.0637 | 0.6573 | 0.9997 |
| Colour-histogram similarity | 0.9588 | 0.0501 | 0.7877 | 0.9989 |
| Edge-structure similarity | 0.8421 | 0.1330 | 0.4603 | 0.9948 |
| Object support inside rectangle | 1.0000 | 0.0000 | 1.0000 | 1.0000 |
| Changed pixels inside interaction mask | 1.0000 | 0.0000 | 1.0000 | 1.0000 |
| Changed pixels inside placement rectangle | 0.9882 | 0.0248 | 0.8798 | 1.0000 |
| Interaction mask inside rectangle | 0.8971 | 0.0863 | 0.5522 | 0.9925 |
| Allowed change fraction of image | 0.0429 | 0.0324 | 0.0118 | 0.1483 |
| Observed changed fraction of image | 0.0247 | 0.0250 | 0.0027 | 0.1150 |
| Outside-mask changed fraction | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| Outside-mask MAE | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| Prior-object changed fraction (20 eligible turns) | 0.00048 | 0.00144 | 0.0000 | 0.00598 |
| Prior-object MAE (20 eligible turns) | 0.0178 | 0.0537 | 0.0000 | 0.2320 |

The identity scores are generally high, with DINOv2 and colour similarity both above 0.95 on average. Edge similarity is lower and substantially more variable, indicating that structural fidelity is the least reliable component. The hard compositing constraint succeeds exactly: no pixel outside the interaction mask changes in any evaluated turn. However, these two findings answer different questions. They demonstrate local appearance retention and deterministic background preservation; they do not demonstrate natural geometric integration.

## Behaviour across editing depth

| Turn | DINO identity | Colour similarity | Edge similarity | Changed image fraction |
|---:|---:|---:|---:|---:|
| 1 | 0.9637 | 0.9614 | 0.8405 | 0.0286 |
| 2 | 0.9683 | 0.9631 | 0.8737 | 0.0315 |
| 3 | 0.9377 | 0.9520 | 0.8122 | 0.0140 |

The third insertion has the weakest mean identity and edge similarity. This is suggestive of an editing-depth effect, but it is not sufficient to establish one: object category and placement difficulty are confounded with turn index, and only ten observations occur at each depth. A controlled order-randomization study would be required before claiming cumulative degradation.

## Per-scene results

| Case | Inserted objects | Mean DINO | Mean colour | Mean edge | Mean changed fraction |
|---:|---|---:|---:|---:|---:|
| 1 | couch, floor lamp, potted plant | 0.9866 | 0.9762 | 0.9464 | 0.0459 |
| 2 | toaster, microwave, oven | 0.8552 | 0.9639 | 0.6960 | 0.0095 |
| 3 | laptop, clock, chair | 0.9787 | 0.9809 | 0.8986 | 0.0314 |
| 4 | dresser, bed, pillow | 0.9978 | 0.9975 | 0.9678 | 0.0557 |
| 5 | teapot, vase, fork | 0.9906 | 0.9835 | 0.9024 | 0.0163 |
| 6 | toothbrush, bucket, potted plant | 0.9581 | 0.9613 | 0.8883 | 0.0143 |
| 7 | ladder, bicycle, hammer | 0.9494 | 0.9295 | 0.6797 | 0.0227 |
| 8 | chair, table, umbrella | 0.9538 | 0.9043 | 0.7816 | 0.0180 |
| 9 | basket, table, chair | 0.9795 | 0.9525 | 0.9046 | 0.0193 |
| 10 | table, chair, book | 0.9162 | 0.9388 | 0.7561 | 0.0139 |

Cases 1, 4, and 5 obtain the strongest localized reference-fidelity scores. Cases 2, 7, and 10 are structurally weaker. The lowest single-object result is the case-2 oven (DINO 0.6573; edge 0.4603). Other diagnostically useful failures include the case-7 ladder (edge 0.5287), case-8 umbrella (colour 0.7877), and case-10 book, which produces the largest measured later-turn disturbance of an earlier object (changed fraction 0.00598; MAE 0.2320).

## Qualitative findings

### What works

1. **Strict edit localization.** Background pixels outside the alpha-derived interaction support are identical to the preceding image. This directly satisfies the hard form of non-disturbance in the problem formulation.
2. **Strong appearance retention.** Most objects retain their reference colour, broad silhouette, and recognizable category. The reference-first collage is more identity preserving than asking the editor to regenerate an object from text alone.
3. **Stable sequential composition.** Previously inserted visible regions exhibit little measured drift because later outputs are copied back only within the current interaction support.
4. **Useful diagnostic trace.** Each turn saves the before image, raw collage, local Qwen output, final composite, alpha, interaction mask, blend mask, and a five-panel summary. This makes failures auditable.

### What remains unsolved

1. **Physical plausibility is inconsistent.** Some objects appear to float, intersect surfaces, or lack convincing contact shadows. The final grids show conspicuous examples in the bedroom, bathroom, garage, garden, and classroom scenes.
2. **Placement rectangles encode location but not scene geometry.** A rectangle does not specify a supporting plane, depth, surface normal, occlusion order, or perspective. Qwen receives only a limited local crop and cannot reliably infer all of these constraints.
3. **Hard preservation limits harmonization.** The same mechanism that guarantees zero outside-mask error prevents the model from creating broad cast shadows, reflections, occlusions, or illumination changes beyond the narrow interaction support. This creates a fundamental fidelity-versus-integration trade-off.
4. **Identity metrics reward collage retention.** The localized metrics use the known pasted alpha support. A nearly unchanged pasted object can score highly even if it appears visually composited. Thus, the scores should be described as *localized reference retention*, not as complete insertion quality.
5. **Scale and contact errors are not measured.** No current metric evaluates support-surface contact, relative scale, perspective consistency, occlusion correctness, or human-rated realism.

## Recommended claims for the paper

The results support the following defensible claims:

- The pipeline enforces exact per-turn non-disturbance outside the specified interaction region.
- Deterministic cutout initialization followed by local harmonization retains reference appearance well under DINOv2, colour, and edge-distribution measures.
- Cross-turn pixel stability is high for visible portions of previously inserted objects.
- Structural fidelity and physical integration remain category- and placement-dependent, especially for articulated, thin, or support-sensitive objects.

The results do **not** support claims of generally natural placement, perfect semantic identity, or superiority to another method because no baseline, human study, multiple-seed evaluation, or significance test is included.

## Paper-ready discussion

Across 30 sequential edits, the proposed local collage harmonization procedure achieved a mean DINOv2 reference similarity of 0.9566, colour-histogram similarity of 0.9588, and edge-structure similarity of 0.8421. The lower and more variable edge score indicates that structure is less reliably retained than colour or high-level appearance. All pasted object support remained inside its assigned placement rectangle. Moreover, hard masked recomposition yielded exactly zero change outside the interaction support in every turn, while previously inserted visible regions changed by only 0.00048 on average at the adopted threshold. These preservation results should be interpreted as guarantees of the compositing design rather than unconstrained generative-editing performance.

Qualitative inspection reveals the remaining limitation: localized reference retention does not ensure physically plausible insertion. Several outputs preserve the object strongly but exhibit implausible scale, weak surface contact, missing or inconsistent shadows, or foreground-like compositing. This behaviour is expected because the placement rectangle constrains two-dimensional support but contains no explicit scene geometry, while hard background restoration prevents the editor from modifying a sufficiently broad area to express global illumination and contact effects. The method therefore provides a strong identity-and-preservation baseline, but spatial plausibility remains the primary open problem.

## Additional evaluation needed for publication

1. Run at least three to five seeds and report confidence intervals rather than a single fixed-seed result.
2. Compare against the raw collage, unconstrained full-image Qwen editing, and mask-only Qwen inpainting using identical scenes and references.
3. Conduct a blinded human study with separate questions for object identity, placement naturalness, and overall realism.
4. Add an independent local perceptual metric (for example, masked LPIPS) and report it alongside DINOv2; avoid relying on three correlated appearance measures alone.
5. Measure boundary quality in a narrow ring around the silhouette, since seams and contact failures occur there rather than in the distant background.
6. Randomize insertion order to isolate true cross-turn degradation from category and placement difficulty.

## Reproducibility artifacts

- Configuration: [`config.json`](config.json)
- Per-turn metrics: [`metrics.csv`](metrics.csv)
- Aggregate metrics: [`metrics_summary.json`](metrics_summary.json)
- Run summary: [`summary.json`](summary.json)
- Qualitative outputs: each `case_XXX` directory contains `FINAL.png`, placement labels, histories, and turn-level diagnostic panels.

