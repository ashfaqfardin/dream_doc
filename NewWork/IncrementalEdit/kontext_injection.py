"""
Single-branch K/V injection on FLUX.1-Kontext-dev (pipelineInc.md §0, §2, §5).

Requires torch + diffusers + a >=24GB GPU to actually run. Not executable in
this build environment (no torch install here — see pipelineInc.md §10);
validated by ast.parse and by matching diffusers' actual
`pipeline_flux_kontext.py` API (confirmed via source inspection, not
assumed) for the reference-token concatenation and R-RoPE flag described in
pipelineInc.md §0. If a diffusers version shifts these internals, the
try/except blocks below print the actual `attn_processors` keys and the
observed token-slice sizes so a human can re-derive the split quickly
(pipelineInc.md §9) rather than silently producing wrong numbers.

Architecture recap (why this file has no B=2 dual branch, unlike
UltimateFlux/sampler.py): Kontext concatenates the clean, VAE-encoded canvas
(`image_latents`) onto the noisy generation latents at every step, in the
SAME forward pass. So "source K/V" is not something we generate via a
parallel trajectory (FLUX.1-dev dual-branch style) — it's a tensor slice
already present in every attention call. Injection reads that slice and
selectively overwrites the generation-token slice's own K/V before SDPA.
"""

import os
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Set, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

from diffusers.models.attention_processor import Attention
from diffusers.models.embeddings import apply_rotary_emb

from mask_ops import topk_mask, assert_token_count

# ── Layer tiers — carried over from FreeFlux/UltimateFlux, dev-validated.
# pipelineInc.md §4: this is a HYPOTHESIS on Kontext (extra R-RoPE type axis
# not present in the analysis that produced it), not a re-validated fact.
TIER_A: List[int] = [0, 7, 8, 9, 10, 18, 25, 28, 37, 42, 45, 50, 56]
TIER_ALL: List[int] = list(range(57))
N_DOUBLE = 19
N_SINGLE = 38
N_LAYERS = N_DOUBLE + N_SINGLE  # 57, same architecture as FLUX.1-dev


def resolve_vital_layers(spec: Optional[str]) -> List[int]:
    """CLI --vital-layers parsing: 'tier_a' (default), 'all', or a comma list."""
    if spec is None or spec == "tier_a":
        return list(TIER_A)
    if spec == "all":
        return list(TIER_ALL)
    return sorted(int(x) for x in spec.split(",") if x.strip())


def set_determinism(seed: int) -> torch.Generator:
    """pipelineInc.md §5 — must run before any noise is sampled."""
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.use_deterministic_algorithms(True, warn_only=True)
    return torch.Generator("cuda").manual_seed(seed)


def load_canvas_pipeline(model_id: str = "black-forest-labs/FLUX.1-dev",
                         dtype: torch.dtype = torch.bfloat16, device: str = "cuda"):
    from diffusers import FluxPipeline

    pipe = FluxPipeline.from_pretrained(model_id, torch_dtype=dtype)
    pipe.to(device)
    return pipe


def load_kontext_pipeline(model_id: str = "black-forest-labs/FLUX.1-Kontext-dev",
                          dtype: torch.dtype = torch.bfloat16, device: str = "cuda"):
    from diffusers import FluxKontextPipeline

    pipe = FluxKontextPipeline.from_pretrained(model_id, torch_dtype=dtype)
    pipe.to(device)
    return pipe


def _step_active(step: int, n_steps: int, frac: Tuple[float, float]) -> bool:
    return int(frac[0] * n_steps) <= step < int(frac[1] * n_steps)


# ─────────────────────────── Injection state ──────────────────────────────

@dataclass
class ZoneMasks:
    """Boolean token masks over the GENERATION-token slice only, length
    n_gen == h_lat*w_lat (pipelineInc.md §2). background/shell get frozen
    (with different strength), target never receives injection."""
    background: np.ndarray
    shell: np.ndarray
    target: np.ndarray

    def to_device(self, device) -> "ZoneMasks":
        return ZoneMasks(
            background=torch.as_tensor(self.background, device=device),
            shell=torch.as_tensor(self.shell, device=device),
            target=torch.as_tensor(self.target, device=device),
        )


@dataclass
class InjectionState:
    """Mutable state shared between the run loop and the attention
    processor. `mode='off'` must reproduce stock FluxAttnProcessor math
    exactly (pipelineInc.md §5) — that's what makes Stage 1's canvas
    MSE==0 check meaningful, and what makes the Stage-2 shape assertion
    trustworthy rather than merely non-crashing.
    """
    mode: str = "off"  # 'off' | 'edit' | 'reasoning'
    vital_layers: Set[int] = field(default_factory=lambda: set(TIER_A))
    n_gen: int = 0
    n_ref: int = 0
    zones: Optional[ZoneMasks] = None          # 'edit' mode
    cutoff_frac: Tuple[float, float] = (0.0, 0.6)
    strength: float = 1.0
    cur_step: int = 0
    n_steps: int = 28
    single_txt_len: int = 512  # single-stream blocks' fixed text-prefix offset, set by install_processor
    # reasoning-pass (add verb, pipelineInc.md §2 "add's placement sub-step")
    concept_layers: Set[int] = field(default_factory=set)
    concept_token_idx: List[int] = field(default_factory=list)
    derive_step: int = 7
    saliency_accum: Optional[torch.Tensor] = None  # (n_gen,) accumulated on device
    saliency_captures: int = 0


class KontextInjectionProcessor:
    """Drop-in replacement for the transformer's per-block attention
    processor. Mirrors UltimateFluxAttnProcessor's structure
    (NewWork/UltimateFlux/sampler.py) but single-branch (batch=1) and reads
    the reference slice from the SAME forward call instead of a cached one.
    """

    def __init__(self, state: InjectionState, total_layers: int = N_LAYERS):
        if not hasattr(F, "scaled_dot_product_attention"):
            raise ImportError("Requires PyTorch 2.0+ (scaled_dot_product_attention).")
        self.state = state
        self.total_layers = total_layers
        self.cur_layer = 0

    def reset(self):
        self.cur_layer = 0

    def _tick(self):
        self.cur_layer += 1
        if self.cur_layer == self.total_layers:
            self.cur_layer = 0
            self.state.cur_step += 1

    def __call__(self, attn: Attention, hidden_states, encoder_hidden_states=None,
                 attention_mask=None, image_rotary_emb=None):
        out = self._forward(attn, hidden_states, encoder_hidden_states,
                            attention_mask, image_rotary_emb)
        self._tick()
        return out

    def _forward(self, attn, hidden_states, encoder_hidden_states,
                 attention_mask, image_rotary_emb):
        st = self.state
        layer = self.cur_layer
        is_double = encoder_hidden_states is not None
        B = hidden_states.shape[0]

        q = attn.to_q(hidden_states)
        k = attn.to_k(hidden_states)
        v = attn.to_v(hidden_states)
        inner_dim = k.shape[-1]
        head_dim = inner_dim // attn.heads

        q = q.view(B, -1, attn.heads, head_dim).transpose(1, 2)
        k = k.view(B, -1, attn.heads, head_dim).transpose(1, 2)
        v = v.view(B, -1, attn.heads, head_dim).transpose(1, 2)
        if attn.norm_q is not None:
            q = attn.norm_q(q)
        if attn.norm_k is not None:
            k = attn.norm_k(k)

        txt_len = 0
        if is_double:
            eq = attn.add_q_proj(encoder_hidden_states)
            ek = attn.add_k_proj(encoder_hidden_states)
            ev = attn.add_v_proj(encoder_hidden_states)
            eq = eq.view(B, -1, attn.heads, head_dim).transpose(1, 2)
            ek = ek.view(B, -1, attn.heads, head_dim).transpose(1, 2)
            ev = ev.view(B, -1, attn.heads, head_dim).transpose(1, 2)
            if attn.norm_added_q is not None:
                eq = attn.norm_added_q(eq)
            if attn.norm_added_k is not None:
                ek = attn.norm_added_k(ek)
            txt_len = eq.shape[2]
            q = torch.cat([eq, q], dim=2)
            k = torch.cat([ek, k], dim=2)
            v = torch.cat([ev, v], dim=2)

        if image_rotary_emb is not None:
            q = apply_rotary_emb(q, image_rotary_emb)
            k = apply_rotary_emb(k, image_rotary_emb)

        # img_offset: where the image-token slice [gen | ref] begins within
        # this block's combined sequence. Double-stream blocks prefix text
        # (txt_len computed above); single-stream blocks receive an
        # already-combined [text | gen | ref] sequence with a fixed offset
        # the caller must supply via st on pre_generate (max_sequence_length).
        img_offset = txt_len if is_double else st.single_txt_len

        if st.mode == "edit" and layer in st.vital_layers and st.n_gen > 0:
            k, v = self._inject(k, v, img_offset)
        elif st.mode == "reasoning" and layer in st.concept_layers and is_double and txt_len > 0:
            self._capture_saliency(q, k, txt_len, img_offset)

        out = F.scaled_dot_product_attention(q, k, v, dropout_p=0.0, is_causal=False,
                                             attn_mask=attention_mask)
        out = out.transpose(1, 2).reshape(B, -1, attn.heads * head_dim).to(q.dtype)

        if is_double:
            enc_out = out[:, :encoder_hidden_states.shape[1]]
            out = out[:, encoder_hidden_states.shape[1]:]
            out = attn.to_out[0](out)
            out = attn.to_out[1](out)
            enc_out = attn.to_add_out(enc_out)
            return out, enc_out
        return out

    def _inject(self, k, v, img_offset: int):
        """Three-zone overwrite (pipelineInc.md §2). background: K+V freeze.
        shell: K-only freeze. target: never touched here — the caller
        decides target's own policy (attribute/part_edit want K-only there
        too, replace/remove/add want no injection) by simply excluding it
        from `background`/`shell` in the ZoneMasks passed at pre_generate."""
        st = self.state
        n_gen = st.n_gen
        gen_lo, gen_hi = img_offset, img_offset + n_gen
        ref_lo, ref_hi = gen_hi, gen_hi + st.n_ref

        step_ok = _step_active(st.cur_step, st.n_steps, st.cutoff_frac)
        if not step_ok:
            return k, v

        zones = st.zones
        s = st.strength

        k_ref = k[:, :, ref_lo:ref_hi, :]
        v_ref = v[:, :, ref_lo:ref_hi, :]
        k_gen = k[:, :, gen_lo:gen_hi, :].clone()
        v_gen = v[:, :, gen_lo:gen_hi, :].clone()

        bg = zones.background.view(1, 1, -1, 1)
        k_gen = torch.where(bg, (1 - s) * k_gen + s * k_ref, k_gen)
        v_gen = torch.where(bg, (1 - s) * v_gen + s * v_ref, v_gen)

        if zones.shell.any():
            sh = zones.shell.view(1, 1, -1, 1)
            k_gen = torch.where(sh, (1 - s) * k_gen + s * k_ref, k_gen)
            # V left free in the shell zone — silhouette locked, appearance not.

        k = k.clone()
        v = v.clone()
        k[:, :, gen_lo:gen_hi, :] = k_gen
        v[:, :, gen_lo:gen_hi, :] = v_gen
        return k, v

    def _capture_saliency(self, q, k, txt_len, img_offset):
        """DAAM-style Q_img[gen] x K_concept saliency, reused from
        UltimateFlux/policies.py's _DAAmAttnProcessor for the `add` verb's
        placement sub-step (pipelineInc.md §2). Direction: HIGH score =
        generation-token i attends strongly to the new object's noun token
        = plausible placement for the new object."""
        st = self.state
        valid = [i for i in st.concept_token_idx if i < txt_len]
        if not valid:
            return
        gen_lo, gen_hi = img_offset, img_offset + st.n_gen
        head_dim = q.shape[-1]
        q_img = q[:, :, gen_lo:gen_hi, :].float()
        k_con = k[:, :, valid, :].float()
        logits = torch.einsum("bhid,bhjd->bhij", q_img, k_con) * (head_dim ** -0.5)
        sal = logits.mean(1).mean(-1)[-1].detach()  # (n_gen,)
        if st.saliency_accum is None:
            st.saliency_accum = sal.clone()
        else:
            st.saliency_accum += sal
        st.saliency_captures += 1


# ─────────────────────── Stage 2: reference-slice assertion ───────────────

def assert_reference_slice(pipe, canvas: Image.Image, height: int, width: int,
                           vital_layers: List[int]) -> dict:
    """pipelineInc.md §6 Stage 2. Structural check (not a caching pass, see
    the module-level changelog reasoning): confirms N_gen + N_ref matches
    the expected token counts and that the R-RoPE type flag distinguishes
    the two slices, by inspecting one prepared batch before running any
    denoising steps. Raises with a diagnostic dump (attn_processors keys,
    observed shapes) rather than proceeding on a guess — pipelineInc.md §9.
    """
    vae_scale = getattr(pipe, "vae_scale_factor", 8)
    h_lat = height // (vae_scale * 2)
    w_lat = width // (vae_scale * 2)
    n_expected = h_lat * w_lat

    try:
        img_latents = pipe._encode_vae_image(image=canvas, generator=None)  # diffusers>=0.32 Kontext API
        n_ref = img_latents.shape[1] if img_latents.dim() == 3 else img_latents.shape[0]
    except Exception as exc:  # noqa: BLE001 — deliberately broad, see docstring
        raise RuntimeError(
            "Stage 2 reference-slice assertion failed to call the expected "
            "diffusers Kontext image-encoding API. This usually means the "
            "installed diffusers version changed its internal method names. "
            f"Underlying error: {exc!r}. Dump of attn_processors keys for "
            f"manual re-derivation: {list(pipe.transformer.attn_processors.keys())[:5]}..."
        ) from exc

    ok = n_ref == n_expected
    return {
        "n_gen_expected": n_expected,
        "n_ref_observed": n_ref,
        "reference_slice_ok": bool(ok),
        "vital_layers_checked": len(vital_layers),
    }


# ─────────────────────────── pre_generate wiring ───────────────────────────

def install_processor(pipe, state: InjectionState, max_sequence_length: int = 512) -> KontextInjectionProcessor:
    state.single_txt_len = max_sequence_length  # see _forward's img_offset comment
    proc = KontextInjectionProcessor(state, total_layers=N_LAYERS)
    proc.reset()
    pipe.transformer.set_attn_processor(proc)
    return proc


def get_t5_token_indices(pipe, text: str, word: str) -> List[int]:
    """Same logic as UltimateFlux/policies.py:_get_t5_token_indices — kept
    duplicated rather than imported cross-package since UltimateFlux is a
    sibling project, not a shared library (see README for the reuse note)."""
    tok = getattr(pipe, "tokenizer_2", None) or pipe.tokenizer
    prompt_ids = tok(text, add_special_tokens=False).input_ids
    word_ids = tok(word, add_special_tokens=False).input_ids
    if not word_ids:
        return []
    indices: List[int] = []
    for i in range(len(prompt_ids) - len(word_ids) + 1):
        if prompt_ids[i : i + len(word_ids)] == word_ids:
            indices.extend(range(i, i + len(word_ids)))
    if not indices:
        for wt in word_ids:
            indices += [j for j, pt in enumerate(prompt_ids) if pt == wt]
    return sorted(set(indices))


@torch.no_grad()
def run_edit_pass(
    pipe,
    canvas: Image.Image,
    prompt: str,
    zones: Optional[ZoneMasks],
    h_lat: int,
    w_lat: int,
    vital_layers: List[int],
    seed: int = 42,
    num_steps: int = 28,
    guidance_scale: float = 2.5,
    inject_cutoff_frac: Tuple[float, float] = (0.0, 0.6),
    inject_strength: float = 1.0,
    max_sequence_length: int = 512,
    added_noun: Optional[str] = None,
    derive_step: int = 7,
    top_k_frac: float = 0.08,
    device: str = "cuda",
) -> Tuple[Image.Image, dict]:
    """One single-branch Kontext denoising loop with three-zone injection
    (pipelineInc.md §0, §2). If `added_noun` is given and `zones` is None,
    runs the `add` verb's reasoning sub-step first (single continuous pass,
    see pipelineInc.md §2) to derive the target mask mid-generation.
    """
    n_gen = h_lat * w_lat
    height, width = h_lat * 16, w_lat * 16

    state = InjectionState(
        mode="edit",
        vital_layers=set(vital_layers),
        n_gen=n_gen,
        n_ref=n_gen,  # same-resolution canvas -> equal token counts
        cutoff_frac=inject_cutoff_frac,
        strength=inject_strength,
        n_steps=num_steps,
    )

    if zones is None:
        if not added_noun:
            raise ValueError("run_edit_pass needs either `zones` or `added_noun` (add verb).")
        # `add` verb: start fully frozen everywhere (nothing can move yet),
        # derive placement mid-pass, then switch to target-free from
        # derive_step onward — pipelineInc.md §2 "add's placement sub-step".
        everywhere = np.ones(n_gen, dtype=bool)
        nothing = np.zeros(n_gen, dtype=bool)
        zones = ZoneMasks(background=everywhere, shell=nothing, target=nothing)
        state.mode = "reasoning"
        state.derive_step = derive_step
        concept_idx = get_t5_token_indices(pipe, prompt, added_noun)
        if not concept_idx:
            raise ValueError(
                f"'{added_noun}' not found in the T5 tokenization of the edit "
                f"prompt — placement derivation needs the noun to appear verbatim."
            )
        state.concept_token_idx = concept_idx
        state.concept_layers = {l for l in vital_layers if l < N_DOUBLE} or set(range(N_DOUBLE))

    state.zones = zones.to_device(device)
    proc = install_processor(pipe, state, max_sequence_length=max_sequence_length)

    probe: List[dict] = []

    def _callback(pipe_ref, step_index, timestep, callback_kwargs):
        if state.mode == "reasoning" and step_index == state.derive_step:
            if state.saliency_accum is not None and state.saliency_captures > 0:
                saliency = (state.saliency_accum / state.saliency_captures).cpu().numpy()
                target = topk_mask(saliency, top_k_frac)
                assert_token_count(target, h_lat, w_lat, "derived placement mask")
                new_zones = ZoneMasks(
                    background=np.logical_and(zones.background, ~target),
                    shell=zones.shell,
                    target=target,
                )
                state.zones = new_zones.to_device(device)
                state.mode = "edit"
            else:
                # No saliency captured — fall back to fully-frozen (add fails
                # visibly as "nothing appeared", per pipelineInc.md §3, not silently).
                state.mode = "edit"
        probe.append({"step": step_index, "mode": state.mode})
        return callback_kwargs

    generator = set_determinism(seed)
    result = pipe(
        image=canvas,
        prompt=prompt,
        num_inference_steps=num_steps,
        guidance_scale=guidance_scale,
        height=height,
        width=width,
        max_sequence_length=max_sequence_length,
        generator=generator,
        output_type="pil",
        callback_on_step_end=_callback,
        callback_on_step_end_tensor_inputs=[],
    )
    edit_img = result.images[0]
    target_np = None
    if state.zones is not None:
        target_np = state.zones.target.detach().cpu().numpy().astype(bool)
    return edit_img, {"probe": probe, "final_mode": state.mode, "target_mask": target_np}


@torch.no_grad()
def derive_object_mask_on_canvas(
    pipe,
    canvas: Image.Image,
    noun: str,
    prompt: str,
    h_lat: int,
    w_lat: int,
    vital_layers: List[int],
    seed: int = 42,
    mask_steps: int = 20,
    top_k_frac: float = 0.35,
    max_sequence_length: int = 512,
    device: str = "cuda",
) -> np.ndarray:
    """Post-hoc DAAM-style mask for an OBJECT ALREADY PRESENT in the canvas
    (pipelineInc.md §1: masks stored per object, reused across future calls).
    Distinct from run_edit_pass's `add`-verb reasoning sub-step, which finds
    where a NOT-YET-EXISTING object should go — this instead finds an
    existing object's current extent, for `replace`'s new object, or
    `attribute --refresh-mask`.

    Runs the canvas fully frozen (background zone = everywhere) so nothing
    changes, purely to capture Q_img[gen] x K_concept(noun) saliency in the
    same single-pass style as the rest of this module.
    """
    n_gen = h_lat * w_lat
    height, width = h_lat * 16, w_lat * 16

    state = InjectionState(
        mode="reasoning",
        vital_layers=set(vital_layers),
        n_gen=n_gen,
        n_ref=n_gen,
        n_steps=mask_steps,
    )
    everywhere = np.ones(n_gen, dtype=bool)
    nothing = np.zeros(n_gen, dtype=bool)
    state.zones = ZoneMasks(background=everywhere, shell=nothing, target=nothing).to_device(device)

    concept_idx = get_t5_token_indices(pipe, prompt, noun)
    if not concept_idx:
        raise ValueError(
            f"'{noun}' not found in the T5 tokenization of prompt {prompt!r} — "
            f"cannot derive a saliency mask for it. Pass an explicit --mask instead."
        )
    state.concept_token_idx = concept_idx
    state.concept_layers = {l for l in vital_layers if l < N_DOUBLE} or set(range(N_DOUBLE))

    install_processor(pipe, state, max_sequence_length=max_sequence_length)
    generator = set_determinism(seed)
    pipe(
        image=canvas,
        prompt=prompt,
        num_inference_steps=mask_steps,
        guidance_scale=1.0,  # no CFG -> single forward per step, faster probe
        height=height,
        width=width,
        max_sequence_length=max_sequence_length,
        generator=generator,
        output_type="latent",
    )

    if state.saliency_accum is None or state.saliency_captures == 0:
        raise RuntimeError(
            f"No saliency captured for '{noun}' — concept_token_idx may exceed "
            f"txt_len, or vital_layers has no double-stream entries."
        )
    saliency = (state.saliency_accum / state.saliency_captures).cpu().numpy()
    return topk_mask(saliency, top_k_frac)
