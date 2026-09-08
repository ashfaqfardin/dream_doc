# Training-Free Local Collage Harmonization Pipeline

## Method overview

We formulate sequential reference-object insertion as a local collage harmonization problem. At editing turn \(i\), the method receives the current scene \(I_{i-1}\), a reference object image \(O_i\), and a user-provided placement constraint \(L_i\). The constraint is represented by a rectangular region indicating where the object should appear. The method produces the next image \(I_i\) without updating the parameters of the underlying generative model.

The central design principle is to separate the three responsibilities of an insertion operation:

1. **Object identity** is supplied by a foreground cutout extracted from \(O_i\).
2. **Object location and scale** are initialized deterministically from \(L_i\).
3. **Appearance harmonization** is performed by a pretrained Qwen image-editing model within a local scene crop.

After local generation, an explicit spatial composition rule restores all pixels outside the permitted interaction region from \(I_{i-1}\). This provides hard per-turn non-disturbance and prevents the generative model from modifying unrelated parts of the scene.

## Inputs and notation

At turn \(i\), the pipeline uses:

- \(I_{i-1}\in\mathbb{R}^{H\times W\times 3}\): the scene resulting from the previous editing turn;
- \(O_i\): the reference image defining the desired object's appearance;
- \(L_i\): the user-specified rectangular placement region;
- \(A_i\in[0,1]^{H\times W}\): the object alpha matte after placement;
- \(M_i\in[0,1]^{H\times W}\): the interaction mask defining where generated pixels may be copied into the scene;
- \(C_i\): the initialized collage containing the placed reference object;
- \(\hat I_i\): the locally harmonized candidate produced by Qwen;
- \(I_i\): the final output of turn \(i\).

The same procedure is applied sequentially for \(i=1,\ldots,N\).

## Stage 1: Reference foreground extraction

The reference image \(O_i\) may contain a background that should not be transferred into the target scene. We therefore apply RMBG-2.0 to estimate a foreground alpha matte:

\[
(R_i,\alpha_i)=\mathcal{S}(O_i),
\]

where \(\mathcal{S}\) denotes the background-removal model, \(R_i\) is the foreground RGB image, and \(\alpha_i\) is its soft alpha matte. The foreground and alpha are cropped to their active support. Cutouts are computed once and cached, avoiding repeated segmentation during multi-scene evaluation.

This stage supplies direct visual evidence of colour, material, texture, and structure. Unlike a textual object description, the cutout provides an explicit appearance initialization and therefore reduces the need for the editor to reconstruct object identity from language alone.

## Stage 2: Placement-constrained collage initialization

The placement annotation is decoded into rectangle

\[
L_i=(x_i^0,y_i^0,x_i^1,y_i^1).
\]

The extracted reference foreground is isotropically scaled to fit inside this rectangle while preserving its aspect ratio. A scale factor slightly below the maximum available size leaves limited contextual space around the object. The scaled cutout is bottom-aligned within the rectangle to provide a simple support-surface prior for common indoor objects.

Let \(T_i\) denote the resulting scale-and-translation transformation. The full-resolution placed foreground and alpha matte are

\[
\tilde R_i=T_i(R_i), \qquad A_i=T_i(\alpha_i).
\]

The initialized collage is formed by standard alpha composition:

\[
C_i=(1-A_i)\odot I_{i-1}+A_i\odot \tilde R_i.
\]

This collage fixes the approximate object position, scale, silhouette, and appearance before generative editing. Qwen is consequently asked to harmonize an existing visual placement rather than infer both identity and location from an unconstrained multi-image prompt.

## Stage 3: Interaction-mask construction

The original placement rectangle specifies the permissible object location but is not used directly as the inpainting mask. Editing the entire rectangle would expose unnecessary background pixels to regeneration and could erase local scene structure.

Instead, a binary silhouette is obtained by thresholding the placed alpha matte:

\[
S_i(p)=\mathbb{1}[A_i(p)>\tau_\alpha].
\]

The silhouette is dilated to include a narrow interaction band around the foreground. A shifted and dilated copy is also included below the object to provide limited capacity for a contact or cast shadow. The resulting mask can be written as

\[
M_i^{\mathrm{hard}}
=\operatorname{Dilate}(S_i,r_o)
\;\cup\;
\operatorname{Shift}_{y}\!\left(\operatorname{Dilate}(S_i,r_s)\right),
\]

where \(r_o\) is the object-boundary dilation radius, \(r_s\) is the shadow dilation radius, and the vertical shift approximates a local contact-shadow region. Thus, \(M_i\) represents where object-scene interaction is permitted, whereas \(A_i\) represents the placed object's support.

## Stage 4: Local context selection

Running a generative edit over the complete image is unnecessary and increases the risk of global scene drift. The method therefore computes the bounding box of \(M_i^{\mathrm{hard}}\), expands it by a context margin, and converts it into a square crop \(B_i\). A minimum crop size is enforced so that Qwen observes sufficient surrounding geometry and illumination.

The local model input is

\[
C_i^{B}=\operatorname{Crop}(C_i,B_i),
\qquad
M_i^{B}=\operatorname{Crop}(M_i^{\mathrm{hard}},B_i).
\]

Both are resized to the model's processing resolution. RGB images use high-quality interpolation, whereas the inpainting mask uses nearest-neighbour interpolation to preserve its binary ownership boundary.

The crop is intentionally larger than the object silhouette: object pixels provide identity information, while the surrounding scene provides local evidence about illumination, texture, supporting surfaces, and perspective.

## Stage 5: Local Qwen harmonization

The local collage and its interaction mask are passed to a frozen Qwen-Image-Edit-2509 inpainting pipeline. The prompt is deliberately short:

> Integrate the pasted [object] naturally with the surrounding scene. Preserve its design, colour, material, position and overall structure.

The local generation operation is

\[
\hat I_i^{B}
=\mathcal{G}_{\theta}
\left(C_i^{B},M_i^{B},P_i;\epsilon_i\right),
\]

where \(\mathcal{G}_{\theta}\) is the frozen Qwen editor, \(P_i\) is the object-specific prompt, and \(\epsilon_i\) denotes the seeded diffusion noise. No fine-tuning, adapter training, feature injection, or key-value modification is performed. In the reported configuration, inference uses the Qwen-Image-Lightning eight-step adapter.

The generated crop is resized to the original crop dimensions and inserted into a full-resolution temporary canvas. At this stage, it remains only a candidate result; pixels outside the authorized interaction region are not accepted.

## Stage 6: Hard spatial preservation

To prevent visible seams without allowing changes outside the interaction region, the hard mask is feathered inward. If \(G_\sigma\) denotes Gaussian smoothing, the blend mask is

\[
M_i
=G_\sigma(M_i^{\mathrm{hard}})
\odot M_i^{\mathrm{hard}}.
\]

Multiplication by the hard mask is essential: it ensures

\[
M_i(p)=0 \quad \forall p\notin M_i^{\mathrm{hard}},
\]

even after feathering. The final image is then formed by explicit RGB composition:

\[
I_i
=(1-M_i)\odot I_{i-1}
+M_i\odot \hat I_i.
\]

It follows directly that

\[
I_i(p)=I_{i-1}(p)
\quad \forall p\notin M_i^{\mathrm{hard}}.
\]

Non-disturbance outside the interaction mask is therefore a deterministic guarantee of the pipeline rather than a probabilistic behaviour expected from the generative model.

## Sequential editing and previous-object preservation

After completing turn \(i\), the output \(I_i\) becomes the input to turn \(i+1\). Each subsequent Qwen invocation receives only a crop around the new object's interaction region. Hard composition restores the remainder of \(I_i\), including previously inserted objects that lie outside the new mask.

When two object supports overlap, the current object is treated as being inserted in front of the overlapped portion. For cross-turn evaluation, the newly occupied support is subtracted from the visible mask of earlier objects. This prevents genuine occlusion from being incorrectly counted as identity degradation.

The recurrence is therefore

\[
I_0\xrightarrow{(O_1,L_1)}I_1
\xrightarrow{(O_2,L_2)}\cdots
\xrightarrow{(O_N,L_N)}I_N,
\]

with a separate interaction mask applied at every turn.

## Evaluation protocol

The implementation evaluates the three requirements in the problem formulation separately.

### Localized object fidelity

The inserted object is localized using its placed alpha support and compared with the corresponding reference foreground. The reported measures are:

- DINOv2 feature cosine similarity for high-level visual identity;
- colour-histogram similarity for appearance and palette retention;
- edge-structure similarity for silhouette and structural retention.

These metrics measure localized reference retention. They should not independently be interpreted as measures of natural scene integration.

### Spatial localization

Spatial measurements quantify:

- the fraction of object support lying inside the requested placement rectangle;
- the fraction of changed pixels lying inside the interaction mask;
- the fraction of changed pixels lying inside the placement rectangle;
- the fraction of the full image modified during the turn.

### Non-disturbance

Pixel MAE, maximum error, changed-pixel fraction, and PSNR are computed between \(I_{i-1}\) and \(I_i\) outside \(M_i^{\mathrm{hard}}\). Under correct execution, outside-mask MAE and changed-pixel fraction are exactly zero, while outside-mask PSNR is infinite.

### Cross-turn stability

For every earlier object, its appearance immediately after insertion is stored as a snapshot. At later turns, the currently visible portion of that object is compared with the snapshot using MAE, PSNR, and the fraction of pixels changing above a fixed threshold. This measures cumulative degradation separately from current-turn background preservation.

## Algorithm summary

**Input:** base image \(I_0\), reference objects \(\{O_i\}_{i=1}^{N}\), and placement rectangles \(\{L_i\}_{i=1}^{N}\).

**For each editing turn \(i=1,\ldots,N\):**

1. Extract reference foreground \((R_i,\alpha_i)\) using RMBG-2.0.
2. Scale and anchor the foreground within \(L_i\).
3. Alpha-composite the placed cutout over \(I_{i-1}\) to obtain \(C_i\).
4. Construct the silhouette-derived interaction mask \(M_i^{\mathrm{hard}}\).
5. Extract a square crop containing the complete interaction mask and local scene context.
6. Apply frozen Qwen inpainting to the local collage crop.
7. Resize the generated crop to its original spatial extent.
8. Feather the interaction mask inward.
9. Composite generated pixels only through \(M_i\), restoring \(I_{i-1}\) everywhere else.
10. Set the composed result as \(I_i\) and continue to the next object.

**Output:** the final multi-object scene \(I_N\).

## Design rationale and limitation

The pipeline intentionally prioritizes reference retention and non-disturbance. Direct cutout initialization supplies a stronger identity signal than text-only generation, local cropping reduces the editing search space, and hard composition eliminates global background drift. These properties make the procedure simple, reproducible, and training-free.

However, the hard preservation constraint also limits the area in which Qwen can create scene-level interactions. Large cast shadows, reflections, complex occlusions, and broader illumination changes cannot be expressed outside \(M_i\). Furthermore, a two-dimensional placement rectangle specifies position but does not explicitly encode depth, supporting-plane geometry, surface orientation, or perspective. The method should therefore be characterized as a strong localized identity-and-preservation baseline; physically natural placement remains dependent on the quality of the placement annotation, reference pose, and local scene geometry.

## Suggested concise paper description

> At each editing turn, we extract the reference foreground using RMBG-2.0 and isotropically fit it inside the user-provided placement rectangle. The transformed foreground is alpha-composited with the current scene to produce an identity-preserving collage initialization. Rather than editing the full image, we construct a silhouette-derived interaction mask, augment it with a narrow region for boundary and contact-shadow harmonization, and crop a square region containing the complete object and nearby scene context. A frozen Qwen-Image-Edit-2509 model then harmonizes this local collage using an eight-step diffusion process. Finally, the generated crop is copied back through an inward-feathered interaction mask, while all pixels outside its hard support are restored from the preceding image. This composition yields exact per-turn non-disturbance outside the authorized region. The resulting image becomes the input to the next editing turn, enabling sequential multi-object insertion without model training.

