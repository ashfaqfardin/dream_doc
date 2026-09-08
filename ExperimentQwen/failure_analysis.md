# ExperimentQwen Project Experiment Failure Analysis: E1--E15

This document records failures observed in experiments implemented for this
project, including architectural findings and important runtime failures. It
maps the evidence to the E1--E15 project sequence and does not treat an
unmeasured hypothesis as a successful result or an internal experiment as an
external baseline.

======== E1 failure ========

The first object, particularly the bicycle, could be inserted successfully.
During later incremental edits, however, Qwen frequently replaced or heavily
reconstructed the accumulated scene using the new reference image. Consequently,
the base image and previously inserted objects were not preserved.

Likely cause: unrestricted multi-image editing provided no explicit distinction
between immutable scene content and the new reference object. Error accumulated
over sequential full-frame generations.

======== E2 failure ========

The SAM collage-and-repaint design became a long, multi-stage pipeline involving
placement estimation, segmentation, collage construction, Qwen repainting, and
scene protection. Placement initially required manual boxes; automated
counterfactual placement remained heuristic. Cutout errors and mask boundaries
could directly damage the final composite or constrain the generated object.

Runtime failure also occurred when the `sam2` package was unavailable. This
showed that SAM alone was not a complete solution: it needs an object prompt or
box and adds a brittle external dependency without solving placement or identity.

======== E3 failure ========

Across the prompt suite, not every requested object survived to `FINAL.png`, and
several objects were placed unnaturally. The immediate Qwen output produced from
the collage often looked good, but subsequent mask-based preservation or
processing degraded it.

Likely cause: every sequential full-scene edit could forget earlier objects,
while hard spatial masks disagreed with the object's newly generated boundary,
pose, shadow, or extent.

======== E4 failure ========

E4 was a diagnostic feature-trajectory lab rather than a working insertion
method. Its 3D PCA, token trajectories, affinity surfaces, and heatmaps were
descriptive but did not yield a reliable causal rule for preserving identity or
placing an object.

The experiment also encountered non-finite recorded features and an SVD
non-convergence error. Although numerical sanitization could make the plots run,
the visualization itself did not establish that a proposed feature intervention
would improve generation.

======== E5 failure ========

Several E5 feature-transplant and spatial-K/V variants failed. Direct feature
transplant produced completely dark outputs. Later K/V or residual routing
produced grid-like images, poor cutouts, excessive base-image influence, and weak
object-identity transfer.

SAM cutouts were initially poor. RMBG-2.0 improved the intended matte source but
introduced dependency and Transformers compatibility failures (`kornia`, missing
`model_type`, and `all_tied_weights_keys`). Even when loading succeeded, passing
collage, object, and base signals together overloaded or confused conditioning.

Likely cause: latent or K/V tensors were combined without a trained correspondence
between source-object tokens, target-scene tokens, and output spatial queries.

======== E6 failure ========

Block-sparse reference attention was mathematically cleaner than raw feature
addition, but the outputs were still poor. Restricting attention connectivity did
not create the missing semantic and geometric correspondence between the object
cutout and its intended output representation.

Likely cause: attention routing controls where information may flow, but it does
not ensure that untrained reference values are compatible with target queries or
that the model will render the object's identity faithfully.

======== E7 failure ========

The paper-aligned collage harmonization experiment first failed with an A100
80 GB out-of-memory error while constructing an inpaint pipeline from an already
loaded planner. The conversion attempted to duplicate or recast large Qwen
components while almost all VRAM was occupied.

After further evaluation, the experiment was judged useless and deliberately
deleted. There is therefore no active E7 implementation in the repository.

======== E8 failure ========

The initial masked-attention version produced a final image nearly identical to
the base image, with no inserted object. After increasing the cutout signal, the
generated object still looked essentially nothing like the reference object.

A runtime error also exposed a missing `object_scale` argument. Methodologically,
masked concatenated K/V supplied appearance features but did not provide a valid
query-to-reference correspondence, so added signal strength did not translate
into identity preservation.

======== E9 failure ========

Placeholder-first semantic value matching failed catastrophically in the
reported example: the final output became a newly generated sofa on a white
background instead of an edited scene containing the reference sofa.

Likely cause: replacing or interpolating attention values destabilized the
model's learned joint representation. The placeholder did not establish a
reliable semantic correspondence, and the reference branch dominated the scene.

======== E10 failure ========

Asymmetric VAE/VL conditioning was unable to preserve the reference object's
semantic identity reliably. Supplying the base through VAE and VL paths while
supplying the object through the VL path gave the model object semantics, but not
an exact spatially grounded structure-and-color constraint.

Increasing the identity-guidance signal could not guarantee identity and risked
overpowering the scene. The experiment demonstrated that seeing an object in the
vision-language encoder is not equivalent to reconstructing that object through
the denoising representation.

======== E11 failure ========

Self-localizing VL K/V injection did not work as expected. Early variants used
K/V in a way that did not respect Qwen's token provenance and spatial alignment.
RMBG was also absent from the localization path. Later changes still caused the
base scene to be replaced by an object-dominated image with a white background.

Likely cause: the difficult correspondence gate remained unresolved. Global or
incorrectly localized reference K/V injection let the isolated reference dominate
joint attention instead of modifying only the intended scene object.

======== E12 failure ========

The small stitched-reference-card observation was not consistently reproduced
by the automated E12 pipeline. The single stitched board gave the model an object
identity cue but no strong target placement signal, and the model could interpret
the card or its background as output content rather than temporary evidence.

Changing `true_cfg_scale` to `4.0` with an empty negative prompt did not fix the
failure. CFG increased adherence to the same ambiguous instruction; it did not
resolve spatial correspondence, identity binding, or scene/reference role
confusion.

======== E13 failure ========

Timestep-aligned masked latent blending produced a visibly broken sofa: the
object was clipped, incomplete, and accompanied by white artifacts near the
floor. The object also formed too close to the image border.

The fixed latent gate did not follow the geometry generated by Qwen. When the
object expanded or shifted outside the original paste alpha, background anchoring
overwrote those parts. Independently encoded base and generated latents were also
not boundary-compatible, so their spatial blend decoded into seams and white
structures. E13 demonstrates that hard latent ownership conflicts with the
geometric freedom required for natural harmonization.

======== E14 failure ========

The training-free paired-difference correspondence experiment failed by
collapsing the complete scene into the reference sofa on a white background.
The input collage itself was correct: it contained both the original room and
the placed sofa. The placement/correspondence heatmap also highlighted a
reasonable sofa region, but this localization did not translate into a valid
edited output. The native control, correlation variant, and propagated
`FINAL.png` all exhibited the same full-frame reference reconstruction.

Because the native control failed without the custom correspondence router, the
primary failure was not caused by Sinkhorn or correlation strength. Passing the
base and collage as simultaneous semantic images caused conditioning-role
ambiguity: Qwen treated the object/reference evidence as the generation target
instead of treating the room as the persistent output canvas. The change
heatmap activated across almost the entire scene, confirming global replacement
rather than a localized edit.

E14 was revised so that the native control received only collage `C`, while the
asymmetric variants exposed only `C` to Qwen2.5-VL and supplied VAE banks
`[C, B]`. This still produced the isolated sofa result. Therefore, separating
the VL and VAE paths at inference time did not recreate the binding learned for
a localized edit. Feature-difference localization can identify where `B` and
`C` differ, but it cannot force the pretrained denoiser to preserve `B` while
rendering the selected identity.

Conclusion: E14 demonstrates that accurate attention localization is not
sufficient for controlled object insertion. A correct heatmap is only a
diagnostic; it does not guarantee causal, spatially restricted generation.
Training-free attention-logit routing could not overcome the model's native
full-frame conditioning collapse, and increasing the object bias would likely
strengthen that failure.

======== E15 failure ========

Native broad-mask collage inpainting preserved enough object evidence to form
the requested object, but the object was not placed naturally in the scene.
Its pose, scale, perspective, support/contact, or illumination did not agree
with the surrounding geometry, so the result still appeared pasted or
physically implausible.

The broad inpaint mask gave Qwen room to redraw boundaries, but it did not
provide the missing scene geometry. A 2D placement proposal plus an object
cutout specifies *where* pixels may change, not the object's ground plane,
depth, orientation, occlusion order, or contact relationship. Enlarging the
mask further would add freedom while weakening background preservation;
tightening it would constrain pose and cause clipping. Therefore this failure
is not primarily a mask-width tuning problem.

Conclusion: E15 confirms that native inpainting can improve local rendering,
but it cannot recover physically plausible placement from an arbitrary 2D
collage without an explicit geometry-aware placement condition.

======== Overall conclusion ========

The repeated failure is not simply insufficient reference strength. Stronger
reference signals often destroy the scene, while stronger base preservation
clips or suppresses the object. The unresolved problem is a trained or otherwise
reliable correspondence between reference-object identity, target location, and
output tokens. SAM/RMBG masks provide foreground support but do not establish
that correspondence, and untrained K/V or latent replacement is not a substitute
for it.
