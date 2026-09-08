# Result Analysis: FLUX.1 Kontext Local Collage Harmonization

## Experimental setting

This report analyses `KontextMaskFinal_01`, the FLUX.1 Kontext version of the training-free local collage harmonization pipeline. The run contains 10 scenes, three sequential reference-object insertions per scene, and 30 editing turns in total.

The experiment retains the same deterministic components used in the Qwen evaluation:

- RMBG-2.0 reference cutouts;
- user-provided placement rectangles;
- aspect-ratio-preserving cutout placement;
- alpha-derived interaction masks;
- square local-context crops;
- inward mask feathering;
- hard RGB restoration outside the interaction mask;
- identical localized fidelity, spatial, non-disturbance, and cross-turn metrics.

Only the local generative harmonizer is changed to `black-forest-labs/FLUX.1-Kontext-dev`. The model is evaluated in BF16 with 28 inference steps, guidance scale 2.5, inpainting strength 0.82, resolution 1024 x 1024, and fixed seed 42. No model training, KV injection, or post-generation refinement is used.

At each turn, the final image is obtained using

\[
I_i=(1-M_i)\odot I_{i-1}+M_i\odot\hat I_i,
\]

where \(M_i\) is the alpha-derived interaction mask and \(\hat I_i\) is the locally generated Kontext result. Therefore, pixels outside \(M_i\) are restored directly from the preceding image.

## Aggregate quantitative results

The following statistics are calculated over all 30 editing turns.

| Metric | Mean | Std. dev. | Minimum | Maximum |
|---|---:|---:|---:|---:|
| DINOv2 identity similarity | **0.9814** | 0.0211 | 0.9064 | 0.9996 |
| Colour-histogram similarity | **0.9676** | 0.0273 | 0.8636 | 0.9938 |
| Edge-structure similarity | **0.9141** | 0.0643 | 0.7730 | 0.9968 |
| Object support inside placement rectangle | **1.0000** | 0.0000 | 1.0000 | 1.0000 |
| Changed pixels inside interaction mask | **1.0000** | 0.0000 | 1.0000 | 1.0000 |
| Changed pixels inside placement rectangle | **0.9994** | 0.0017 | 0.9908 | 1.0000 |
| Interaction mask inside placement rectangle | 0.8971 | 0.0863 | 0.5522 | 0.9925 |
| Allowed change fraction of image | 0.0429 | 0.0324 | 0.0118 | 0.1483 |
| Observed changed fraction of image | 0.0232 | 0.0249 | 0.0021 | 0.1150 |
| Outside-mask changed fraction | **0.0000** | 0.0000 | 0.0000 | 0.0000 |
| Outside-mask MAE | **0.0000** | 0.0000 | 0.0000 | 0.0000 |
| Prior-object changed fraction (20 eligible turns) | 0.00044 | 0.00124 | 0.0000 | 0.00479 |
| Prior-object MAE (20 eligible turns) | 0.0215 | 0.0618 | 0.0000 | 0.2631 |

The three reference-fidelity measures are high. DINOv2 similarity reaches 0.9814, while the mean edge-structure score of 0.9141 indicates that most localized outputs retain the reference silhouette and internal structure. Edge similarity remains weaker than DINOv2 and colour similarity, suggesting that geometry is still more vulnerable than broad semantic appearance or palette.

All object support lies inside the requested placement rectangles. Furthermore, the outside-mask pixel error is exactly zero in every turn. These two perfect scores are expected consequences of deterministic placement and hard RGB composition. They validate correct pipeline execution, but they should not be presented as learned capabilities of FLUX.1 Kontext.

## Results across editing depth

| Editing turn | DINO identity | Colour similarity | Edge similarity | Changed image fraction | Changes inside rectangle |
|---:|---:|---:|---:|---:|---:|
| 1 | 0.9804 | 0.9606 | 0.9257 | 0.0272 | 0.9998 |
| 2 | **0.9899** | **0.9758** | **0.9386** | 0.0289 | 0.9998 |
| 3 | 0.9738 | 0.9663 | 0.8782 | 0.0133 | 0.9985 |

The second turn obtains the strongest average fidelity, whereas the third turn is weakest, particularly for edge structure. The third-turn edge score is 0.0604 below that of the second turn. This pattern is consistent with greater difficulty in the third-object set, but it does not by itself prove cumulative degradation. Object category and turn index are confounded because the same object order is used in every run. A controlled experiment should randomize insertion order before attributing the difference to editing depth.

The mean changed fraction decreases to 0.0133 at turn three. This should not automatically be interpreted as better preservation: later objects may simply occupy smaller masks. Preservation is already determined by the explicit composition rule.

## Per-scene analysis

| Case | Inserted objects | Mean DINO | Mean colour | Mean edge | Mean changed fraction |
|---:|---|---:|---:|---:|---:|
| 1 | couch, floor lamp, potted plant | 0.9857 | 0.9659 | 0.9474 | 0.0460 |
| 2 | toaster, microwave, oven | 0.9706 | 0.9626 | 0.8541 | 0.0087 |
| 3 | laptop, clock, chair | 0.9828 | 0.9724 | 0.9408 | 0.0284 |
| 4 | dresser, bed, pillow | 0.9971 | 0.9826 | **0.9715** | 0.0564 |
| 5 | teapot, vase, fork | **0.9925** | 0.9806 | 0.9141 | 0.0157 |
| 6 | toothbrush, bucket, potted plant | 0.9614 | 0.9500 | 0.8878 | 0.0140 |
| 7 | ladder, bicycle, hammer | 0.9917 | 0.9637 | 0.9144 | 0.0165 |
| 8 | chair, table, umbrella | 0.9896 | **0.9847** | 0.9175 | 0.0153 |
| 9 | basket, table, chair | 0.9804 | 0.9417 | 0.9288 | 0.0188 |
| 10 | table, chair, book | 0.9618 | 0.9716 | 0.8651 | 0.0117 |

Cases 4, 5, 7, and 8 obtain particularly high localized reference-retention scores. Cases 2, 6, and 10 contain the weakest structural or semantic results. However, high reference similarity is not always aligned with physical plausibility. Case 4, for example, scores extremely well despite the pillow appearing unnaturally positioned on the floor and the dresser having weak scene contact. The quantitative measures primarily verify that the inserted pixels resemble the reference, not that the result is physically convincing.

## Most informative object-level failures

| Case / turn | Object | DINO | Colour | Edge | Interpretation |
|---|---|---:|---:|---:|---|
| 10 / 3 | book | **0.9064** | 0.9732 | 0.8986 | Lowest semantic identity; the placed result reads more like a generic dark open object than a strongly preserved reference instance |
| 6 / 1 | toothbrush | 0.9459 | 0.9314 | 0.8759 | Small/thin object with limited feature support and weak visual prominence |
| 6 / 3 | potted plant | 0.9474 | 0.9264 | **0.7876** | Simultaneous degradation in semantic, colour, and structural fidelity |
| 9 / 1 | basket | 0.9576 | **0.8636** | 0.9524 | Structure is retained, but colour/appearance shifts substantially |
| 2 / 1 | toaster | 0.9961 | 0.9818 | **0.7730** | High DINO score but low edge agreement, demonstrating disagreement between semantic and structural metrics |
| 7 / 3 | hammer | 0.9914 | 0.9608 | 0.8162 | Identity is recognizable, but the object lacks convincing physical support and local integration |

The toaster result is especially useful methodologically: DINOv2 regards it as highly similar while edge structure is the worst in the dataset. This supports reporting multiple localized metrics rather than treating DINO similarity as a complete identity measure.

## Qualitative assessment

### Successful behaviour

1. **Identity is generally retained.** Most objects remain recognizable and preserve the reference's dominant colour, material, silhouette, and structural cues.
2. **The strongest placements use clear supporting surfaces.** The couch in case 1, laptop and clock in case 3, tabletop objects in case 5, and central tables in cases 8–10 are visually coherent because their placement rectangles correspond to unambiguous scene surfaces.
3. **Background preservation is exact.** No unrelated global restyling, white-background takeover, or full-image reconstruction occurs. This directly addresses the dominant failure of earlier full-image editing experiments.
4. **Previously inserted content remains stable.** Cross-turn changed fractions are close to zero because later edits are spatially restricted.

### Remaining failure modes

1. **Contact and support remain unreliable.** The bedroom pillow, garage hammer, garden umbrella, and foreground basket do not establish convincing physical contact with an appropriate surface.
2. **A two-dimensional rectangle does not encode geometry.** The model is told where an object should occupy the image but is not given depth, surface orientation, occlusion order, or a support-plane constraint.
3. **High fidelity can correspond to visible collage retention.** Several objects preserve reference pixels very strongly yet still appear pasted because local illumination, boundary colour, shadow, or perspective is not sufficiently adapted.
4. **Hard preservation restricts environmental effects.** The narrow interaction region prevents broad cast shadows, reflections, and illumination changes from extending naturally through the scene.
5. **Small and thin objects are difficult to evaluate and harmonize.** The toothbrush, fork, hammer, and book contain fewer stable pixels after resizing and receive relatively weak contextual evidence.
6. **Semantically questionable user placements cannot be repaired reliably.** If a rectangle places a pillow on open floor or an umbrella behind a fence without a plausible pole location, local harmonization cannot infer a different intended relation without violating the placement constraint.

## Comparison with the Qwen backbone

Because the same cutouts, masks, placements, hard composition, and metric implementation are used, the two folders permit a paired descriptive comparison. Nevertheless, the inference configurations are not computationally matched: Kontext uses 28 steps and guidance 2.5, whereas the Qwen run uses an eight-step Lightning adapter. The table should therefore be interpreted as a comparison of the evaluated configurations, not an architecture-only ablation.

| Metric | Qwen (`QwenMaskFinal_04`) | Kontext (`KontextMaskFinal_01`) | Kontext − Qwen |
|---|---:|---:|---:|
| DINOv2 identity | 0.9566 | **0.9814** | **+0.0248** |
| Colour similarity | 0.9588 | **0.9676** | **+0.0087** |
| Edge similarity | 0.8421 | **0.9141** | **+0.0720** |
| Changed image fraction | 0.0247 | **0.0232** | −0.0016 |
| Changes inside placement rectangle | 0.9882 | **0.9994** | +0.0112 |
| Outside-mask changed fraction | 0.0000 | 0.0000 | 0.0000 |
| Prior-object changed fraction | 0.00048 | **0.00044** | −0.00004 |

Kontext has higher mean DINO identity and edge similarity, with the largest gain occurring in structural retention. It improves edge similarity on 22 of 30 paired turns and DINO similarity on 17 of 30. The colour result is more nuanced: Kontext's mean is higher by 0.0087, but it wins only 8 of 30 individual turns. The positive mean is therefore driven by relatively large gains on a smaller subset rather than consistent improvement across the dataset.

The largest case-level Kontext improvements occur in cases 2, 7, 8, and 10. Relative to Qwen, their mean DINO improvements are approximately +0.1154, +0.0423, +0.0357, and +0.0456, respectively. Their mean edge improvements are +0.1582, +0.2347, +0.1359, and +0.1090. Cases 1 and 4 are effectively tied in DINO similarity, indicating a ceiling effect for objects already preserved very strongly by the deterministic collage.

These metric gains should not be translated into an equally strong claim about naturalness. Visual inspection still reveals unsupported and weakly integrated objects. The most defensible conclusion is that, under this pipeline, the evaluated Kontext configuration preserves localized reference structure more reliably than the evaluated Qwen configuration, while both remain constrained by the same two-dimensional placement and hard-composition limitations.

## Interpretation with respect to the problem formulation

### Object fidelity

The run performs strongly under the selected localized metrics. Mean DINOv2, colour, and edge similarities are 0.9814, 0.9676, and 0.9141. Kontext therefore retains the supplied reference appearance better than earlier unconstrained attempts and, descriptively, better than the corresponding Qwen configuration. Object fidelity is not perfect: the book, small bathroom objects, basket colour, and several thin structures remain failure cases.

### Spatial plausibility

Localization is satisfied in its narrow two-dimensional sense: all alpha support falls inside the specified rectangle, and 99.94% of changed pixels lie within it. Physical plausibility is only partially satisfied. The pipeline does not explicitly model supporting planes, depth, perspective, or occlusion, and the current metrics do not quantify these properties.

### Non-disturbance

The strict condition

\[
I_i(p)=I_{i-1}(p),\qquad p\notin M_i,
\]

holds exactly for all 30 turns. This is the strongest verified property of the method, but it follows from hard compositing rather than from Kontext's native editing locality.

### Cross-turn identity stability

The visible portions of earlier objects change by only 0.00044 on average at the adopted threshold. This indicates strong sequential stability. As with background preservation, the result is heavily influenced by the hard spatial restoration mechanism. Overlapping placement masks can still modify or occlude an earlier object, which explains the non-zero maximum.

## Validity limitations

1. The study contains only 10 scenes and 30 objects.
2. Every configuration is evaluated with one fixed seed.
3. Placement rectangles and insertion order are fixed rather than randomized.
4. The Qwen and Kontext configurations use different denoising budgets and guidance mechanisms.
5. DINO, colour, and edge metrics are computed under the known pasted alpha support and can reward insufficiently harmonized collage retention.
6. There is no automatic metric or human rating for physical support, perspective, shadow consistency, or overall naturalness.
7. Exact background preservation and support containment are deterministic pipeline guarantees and must not be treated as independent evidence of generative-model quality.

## Defensible paper claims

The results support the following claims:

- Local collage initialization combined with FLUX.1 Kontext preserves reference-object appearance strongly under localized DINOv2, colour, and edge measures.
- The evaluated Kontext configuration improves localized structural retention over the corresponding Qwen configuration.
- Hard spatial recomposition eliminates all changes outside the authorized interaction region.
- The sequential pipeline substantially limits visible degradation of previously inserted objects.
- Physical integration remains dependent on the placement geometry and is not solved by high reference-fidelity scores alone.

The results do not support claims of universally natural placement, exact reference identity, or a fair efficiency-controlled superiority of Kontext over Qwen.

## Paper-ready results discussion

Across 30 sequential object insertions, the FLUX.1 Kontext variant achieved mean localized DINOv2, colour-histogram, and edge-structure similarities of 0.9814, 0.9676, and 0.9141, respectively. The lower edge score indicates that structural details remain more difficult to retain than high-level identity or colour, although its comparatively small standard deviation shows reasonably consistent performance. All foreground support remained inside its assigned placement rectangle, and 99.94% of changed pixels occurred inside that rectangle. Hard masked recomposition produced exactly zero pixel change outside the interaction region for every turn. Previously inserted visible object regions also remained stable, with a mean later-turn changed fraction of 0.00044.

Relative to the corresponding Qwen configuration, Kontext increased mean DINOv2 similarity by 0.0248 and edge similarity by 0.0720. The structural improvement occurred on 22 of 30 paired turns and was particularly large for the kitchen-appliance, garage, outdoor, and classroom cases. However, quantitative identity retention did not guarantee visually natural integration. Qualitative inspection revealed persistent failures involving physical contact, relative scale, support surfaces, and shadows. These failures arise because the placement rectangle specifies only two-dimensional image support, while the hard preservation mask restricts the spatial extent over which the editor can synthesize environmental interactions. The method is therefore best characterized as a strong training-free baseline for reference retention and non-disturbance, with spatial and physical plausibility remaining the principal unresolved challenge.

## Recommended next evaluation

For a publication-grade comparison, the next experiment should:

1. evaluate at least three seeds per case and report bootstrap confidence intervals;
2. match or explicitly normalize the two backbones' inference budgets;
3. randomize object order to separate turn depth from category difficulty;
4. include masked LPIPS or a comparable localized perceptual distance;
5. measure a narrow boundary ring for seam and harmonization quality;
6. collect blinded human ratings separately for identity, placement naturalness, and background preservation;
7. report identity-localization trade-offs rather than collapsing all requirements into one score.

## Reproducibility artifacts

- Run configuration: [`config.json`](config.json)
- Per-turn metrics: [`metrics.csv`](metrics.csv)
- Aggregate metrics: [`metrics_summary.json`](metrics_summary.json)
- Run summary: [`summary.json`](summary.json)
- Final images and diagnostic panels: the corresponding `case_XXX` directories
- Comparison run: [`../QwenMaskFinal_04/result_analysis.md`](../QwenMaskFinal_04/result_analysis.md)

