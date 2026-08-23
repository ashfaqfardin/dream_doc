"""
CLI for incremental, object-addressable scene editing on FLUX.1-Kontext-dev
(pipelineInc.md §7). Requires a >=24GB GPU; only one pipeline (FLUX.1-dev for
`init`, FLUX.1-Kontext-dev for everything else) is loaded per invocation.

Usage — build up a scene across separate calls, run at any later time:

  python run_incremental_edit.py init \\
      --project-dir runs/driveway --prompt "an empty driveway at dusk" --seed 42

  python run_incremental_edit.py add \\
      --project-dir runs/driveway --object car_1 --noun car \\
      --prompt "a red sports car parked in the driveway"

  python run_incremental_edit.py attribute \\
      --project-dir runs/driveway --object car_1 --prompt "make the car blue"

  python run_incremental_edit.py replace \\
      --project-dir runs/driveway --object car_1 --noun cycle \\
      --prompt "a bicycle parked in the same spot"

  python run_incremental_edit.py part-edit \\
      --project-dir runs/driveway --object cycle_1 --part tire_1 --part-noun tire \\
      --prompt "a spoked alloy wheel"

  python run_incremental_edit.py remove \\
      --project-dir runs/driveway --object cycle_1 --fill-prompt "an empty driveway"
"""

import argparse
import json
import math
import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

from manifest import SceneManifest, ManifestError  # noqa: E402
import mask_ops  # noqa: E402
import metrics  # noqa: E402


def _lazy_imports():
    """torch/diffusers imports deferred past argparse so `--help` and the
    CPU-only unit tests never need them (pipelineInc.md §10)."""
    import torch
    import kontext_injection as ki

    return torch, ki


# ────────────────────────────── shared helpers ─────────────────────────────

def _latent_grid(resolution):
    width, height = resolution
    h_lat = height // 16
    w_lat = width // 16
    return h_lat, w_lat


def _print_verdict(lines):
    print("\n" + "=" * 60)
    print("VERDICT")
    print("=" * 60)
    for line in lines:
        print(line)
    print("=" * 60 + "\n")


def _write_run_summary(manifest: SceneManifest, rev_id: int, summary: dict):
    path = os.path.join(manifest.project_dir, "runs", f"rev{rev_id}_summary.json")
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, default=str)
    return path


def _load_edit_pipeline_and_check(args, ki, torch, manifest, canvas, vital_layers):
    """Load FLUX.1-Kontext-dev and run pipelineInc.md §6 Stage 2 (reference-
    slice assertion) before any denoising. Prints PASS/FAIL; a FAIL is a
    diagnostic, not a hard stop, since the assertion itself may be probing
    an internal API name that shifted across diffusers versions (§9) — the
    edit still runs so the user isn't blocked by our own assumption."""
    pipe = ki.load_kontext_pipeline(args.edit_model_id, dtype=torch.bfloat16, device=args.device)
    resolution = manifest.resolution
    try:
        stage2 = ki.assert_reference_slice(pipe, canvas, resolution[1], resolution[0], vital_layers)
        status = "PASS" if stage2["reference_slice_ok"] else "FAIL"
        print(f"[stage2] reference slice {status}: {stage2}")
    except Exception as exc:  # noqa: BLE001 — see docstring
        print(f"[stage2] assertion could not run: {exc}")
        stage2 = {"reference_slice_ok": None, "error": str(exc)}
    return pipe, stage2


# ──────────────────────────────── init ─────────────────────────────────────

def cmd_init(args):
    torch, ki = _lazy_imports()

    manifest = SceneManifest.create(args.project_dir, resolution=[args.resolution, args.resolution])
    pipe = ki.load_canvas_pipeline(args.canvas_model_id, dtype=torch.bfloat16, device=args.device)

    generator_a = ki.set_determinism(args.seed)
    out_a = pipe(
        prompt=args.prompt, num_inference_steps=args.steps, guidance_scale=args.guidance,
        height=args.resolution, width=args.resolution, generator=generator_a, output_type="latent",
    )
    generator_b = ki.set_determinism(args.seed)
    out_b = pipe(
        prompt=args.prompt, num_inference_steps=args.steps, guidance_scale=args.guidance,
        height=args.resolution, width=args.resolution, generator=generator_b, output_type="latent",
    )
    stage1_mse = metrics.latent_mse(out_a.images, out_b.images)
    stage1_pass = stage1_mse == 0.0

    generator = ki.set_determinism(args.seed)
    result = pipe(
        prompt=args.prompt, num_inference_steps=args.steps, guidance_scale=args.guidance,
        height=args.resolution, width=args.resolution, generator=generator, output_type="pil",
    )
    canvas_img = result.images[0]

    rev = manifest.add_revision(op="init", prompt=args.prompt, parent=None)
    canvas_img.save(manifest.canvas_path(rev["id"]))
    manifest.save()

    summary = {"config": vars(args), "stage1_latent_mse": stage1_mse, "stage1_reproducible": stage1_pass}
    _write_run_summary(manifest, rev["id"], summary)
    _print_verdict([
        f"Stage 1 canvas reproducible : {'PASS' if stage1_pass else 'FAIL'} (MSE={stage1_mse})",
        f"Canvas saved -> {manifest.canvas_path(rev['id'])}",
    ])


# ─────────────────────────────── shared finalize ───────────────────────────

def _finalize(args, manifest, rev, edit_img, object_name, target_mask, h_lat, w_lat, extra_summary=None):
    canvas_path = manifest.canvas_path(rev["id"])
    edit_img.save(canvas_path)

    if object_name is not None and target_mask is not None:
        mask_img = mask_ops.mask_to_image(target_mask, h_lat, w_lat)
        mask_path = manifest.mask_path(object_name)
        os.makedirs(os.path.dirname(mask_path), exist_ok=True)
        mask_img.save(mask_path)

    import numpy as np
    from PIL import Image

    background_pixel_mask = None
    if target_mask is not None:
        background_pixel_mask = ~mask_ops.upsample_mask(
            np.asarray(target_mask).reshape(h_lat, w_lat), manifest.resolution[1], manifest.resolution[0]
        )
    edit_np = np.array(edit_img.convert("RGB"))
    step_psnr = background_psnr = float("nan")
    lpips_val = None
    if rev["parent"] is not None:
        parent_np = np.array(Image.open(manifest.canvas_path(rev["parent"])).convert("RGB"))
        step_psnr = metrics.image_psnr(parent_np, edit_np)
        if background_pixel_mask is not None:
            background_psnr = metrics.region_psnr(parent_np, edit_np, background_pixel_mask)
        lpips_val = metrics.lpips_dist(parent_np, edit_np)  # None if lpips/torch unavailable
    root_np = np.array(Image.open(manifest.canvas_path(0)).convert("RGB"))
    cumulative_psnr = metrics.image_psnr(root_np, edit_np)
    gap = metrics.preservation_gap(background_psnr, step_psnr) if not math.isnan(background_psnr) else float("nan")

    manifest.save()

    summary = {
        "config": vars(args),
        "step_psnr": step_psnr,
        "cumulative_psnr": cumulative_psnr,
        "background_psnr": background_psnr,
        "preservation_gap": gap,
        "lpips": lpips_val if lpips_val is not None else "n/a",
    }
    if extra_summary:
        summary.update(extra_summary)
    _write_run_summary(manifest, rev["id"], summary)

    verdict = [
        f"Stage 6 step PSNR (vs parent) : {step_psnr:.2f} dB" if not math.isnan(step_psnr) else "Stage 6 step PSNR : n/a (first revision)",
        f"Stage 6 cumulative PSNR (vs v0): {cumulative_psnr:.2f} dB",
    ]
    if not math.isnan(background_psnr):
        verdict.append(f"Background-zone PSNR : {background_psnr:.2f} dB")
        verdict.append(f"Preservation gap : {gap:.2f} dB " + ("(PASS)" if gap > 0 else "(CHECK — mask mapping or injection mistuned)"))
    verdict.append(f"Canvas saved -> {manifest.canvas_path(rev['id'])}")
    _print_verdict(verdict)


# ──────────────────────────────── add ──────────────────────────────────────

def cmd_add(args):
    torch, ki = _lazy_imports()
    from PIL import Image

    manifest = SceneManifest.load(args.project_dir)
    h_lat, w_lat = _latent_grid(manifest.resolution)
    parent_rev = manifest.latest_revision_id()
    canvas = Image.open(manifest.canvas_path(parent_rev)).convert("RGB")

    vital_layers = ki.resolve_vital_layers(args.vital_layers)
    pipe, stage2 = _load_edit_pipeline_and_check(args, ki, torch, manifest, canvas, vital_layers)

    edit_img, run_info = ki.run_edit_pass(
        pipe, canvas, args.prompt, zones=None, h_lat=h_lat, w_lat=w_lat, vital_layers=vital_layers,
        seed=args.seed, num_steps=args.steps, guidance_scale=args.guidance,
        inject_cutoff_frac=(0.0, args.inject_cutoff_frac), inject_strength=args.inject_strength,
        added_noun=args.noun, derive_step=args.derive_step, top_k_frac=args.top_k_frac, device=args.device,
    )

    rev = manifest.add_revision(op="add", prompt=args.prompt, parent=parent_rev, object=args.object)
    edit_img.save(manifest.canvas_path(rev["id"]))

    # Refine the rough placement mask into a proper object mask on the
    # finished canvas (pipelineInc.md §1/§2: the placement heatmap used
    # mid-generation is coarser than a saliency pass on the finished result).
    try:
        refined = ki.derive_object_mask_on_canvas(
            pipe, edit_img, args.noun, args.prompt, h_lat, w_lat, vital_layers,
            seed=args.seed, device=args.device,
        )
        target_mask = refined
    except Exception as exc:  # noqa: BLE001
        print(f"[add] mask refinement failed ({exc!r}) — keeping the placement-derived mask.")
        target_mask = run_info["target_mask"]

    manifest.register_object(args.object, args.noun, created_at=rev["id"])
    _finalize(args, manifest, rev, edit_img, args.object, target_mask, h_lat, w_lat,
             extra_summary={"stage2_reference_slice": stage2, "stage4_probe": run_info["probe"][:5]})


# ────────────────────────────── attribute ───────────────────────────────────

def cmd_attribute(args):
    torch, ki = _lazy_imports()
    from PIL import Image

    manifest = SceneManifest.load(args.project_dir)
    h_lat, w_lat = _latent_grid(manifest.resolution)
    obj = manifest.require_object(args.object)
    parent_rev = manifest.latest_revision_id()
    canvas = Image.open(manifest.canvas_path(parent_rev)).convert("RGB")

    vital_layers = ki.resolve_vital_layers(args.vital_layers)
    pipe, stage2 = _load_edit_pipeline_and_check(args, ki, torch, manifest, canvas, vital_layers)

    if args.mask:
        target = mask_ops.image_to_mask(Image.open(args.mask), h_lat, w_lat)
    elif args.refresh_mask:
        target = ki.derive_object_mask_on_canvas(
            pipe, canvas, obj["noun"], args.prompt, h_lat, w_lat, vital_layers, seed=args.seed, device=args.device,
        )
    else:
        target = mask_ops.image_to_mask(Image.open(os.path.join(manifest.project_dir, obj["mask"])), h_lat, w_lat)
    mask_ops.assert_token_count(target, h_lat, w_lat, f"object '{args.object}' mask")

    everywhere = target.copy()
    everywhere[:] = True
    background, shell, target_zone = mask_ops.zone_masks(everywhere, target, parent=None)
    zones = ki.ZoneMasks(background=background, shell=shell, target=target_zone)

    edit_img, run_info = ki.run_edit_pass(
        pipe, canvas, args.prompt, zones, h_lat, w_lat, vital_layers,
        seed=args.seed, num_steps=args.steps, guidance_scale=args.guidance,
        inject_cutoff_frac=(0.0, args.inject_cutoff_frac), inject_strength=args.inject_strength,
        device=args.device,
    )

    rev = manifest.add_revision(op="attribute", prompt=args.prompt, parent=parent_rev, object=args.object)
    _finalize(args, manifest, rev, edit_img, args.object if args.refresh_mask else None, target, h_lat, w_lat,
             extra_summary={"stage2_reference_slice": stage2, "stage4_probe": run_info["probe"][:5]})


# ─────────────────────────────── replace ───────────────────────────────────

def cmd_replace(args):
    torch, ki = _lazy_imports()
    from PIL import Image

    manifest = SceneManifest.load(args.project_dir)
    h_lat, w_lat = _latent_grid(manifest.resolution)
    obj = manifest.require_object(args.object)
    parent_rev = manifest.latest_revision_id()
    canvas = Image.open(manifest.canvas_path(parent_rev)).convert("RGB")

    vital_layers = ki.resolve_vital_layers(args.vital_layers)
    pipe, stage2 = _load_edit_pipeline_and_check(args, ki, torch, manifest, canvas, vital_layers)

    old_mask = mask_ops.image_to_mask(Image.open(os.path.join(manifest.project_dir, obj["mask"])), h_lat, w_lat)
    everywhere = old_mask.copy()
    everywhere[:] = True
    # replace: target zone gets NO injection (must not inherit the old
    # object's K — pipelineInc.md §2), background = everything outside the
    # old object's known extent.
    background, shell, target_zone = mask_ops.zone_masks(everywhere, old_mask, parent=None)
    zones = ki.ZoneMasks(background=background, shell=shell, target=target_zone)

    edit_img, run_info = ki.run_edit_pass(
        pipe, canvas, args.prompt, zones, h_lat, w_lat, vital_layers,
        seed=args.seed, num_steps=args.steps, guidance_scale=args.guidance,
        inject_cutoff_frac=(0.0, args.inject_cutoff_frac), inject_strength=args.inject_strength,
        device=args.device,
    )

    rev = manifest.add_revision(op="replace", prompt=args.prompt, parent=parent_rev,
                                object=args.new_object, replaces=args.object)
    orphaned = manifest.retire_object(args.object, retired_at=rev["id"])
    if orphaned:
        print(f"[replace] '{args.object}' had child object(s) now orphaned: {orphaned}")

    try:
        new_mask = ki.derive_object_mask_on_canvas(
            pipe, edit_img, args.noun, args.prompt, h_lat, w_lat, vital_layers, seed=args.seed, device=args.device,
        )
    except Exception as exc:  # noqa: BLE001
        print(f"[replace] mask refinement failed ({exc!r}) — reusing the old object's mask as an approximation.")
        new_mask = old_mask

    manifest.register_object(args.new_object, args.noun, created_at=rev["id"], replaces=args.object)
    _finalize(args, manifest, rev, edit_img, args.new_object, new_mask, h_lat, w_lat,
             extra_summary={"stage2_reference_slice": stage2, "stage4_probe": run_info["probe"][:5],
                            "orphaned_children": orphaned})


# ──────────────────────────────── remove ────────────────────────────────────

def cmd_remove(args):
    torch, ki = _lazy_imports()
    from PIL import Image

    manifest = SceneManifest.load(args.project_dir)
    h_lat, w_lat = _latent_grid(manifest.resolution)
    obj = manifest.require_object(args.object)
    parent_rev = manifest.latest_revision_id()
    canvas = Image.open(manifest.canvas_path(parent_rev)).convert("RGB")

    fill_prompt = args.fill_prompt or manifest.revisions[0]["prompt"]

    vital_layers = ki.resolve_vital_layers(args.vital_layers)
    pipe, stage2 = _load_edit_pipeline_and_check(args, ki, torch, manifest, canvas, vital_layers)

    target = mask_ops.image_to_mask(Image.open(os.path.join(manifest.project_dir, obj["mask"])), h_lat, w_lat)
    everywhere = target.copy()
    everywhere[:] = True
    background, shell, target_zone = mask_ops.zone_masks(everywhere, target, parent=None)
    zones = ki.ZoneMasks(background=background, shell=shell, target=target_zone)

    edit_img, run_info = ki.run_edit_pass(
        pipe, canvas, fill_prompt, zones, h_lat, w_lat, vital_layers,
        seed=args.seed, num_steps=args.steps, guidance_scale=args.guidance,
        inject_cutoff_frac=(0.0, args.inject_cutoff_frac), inject_strength=args.inject_strength,
        device=args.device,
    )

    rev = manifest.add_revision(op="remove", prompt=fill_prompt, parent=parent_rev, object=args.object)
    orphaned = manifest.retire_object(args.object, retired_at=rev["id"])
    if orphaned:
        print(f"[remove] '{args.object}' had child object(s) now orphaned: {orphaned}")

    _finalize(args, manifest, rev, edit_img, None, target, h_lat, w_lat,
             extra_summary={"stage2_reference_slice": stage2, "stage4_probe": run_info["probe"][:5],
                            "orphaned_children": orphaned})


# ─────────────────────────────── part-edit ──────────────────────────────────

def cmd_part_edit(args):
    torch, ki = _lazy_imports()
    from PIL import Image

    manifest = SceneManifest.load(args.project_dir)
    h_lat, w_lat = _latent_grid(manifest.resolution)
    parent_obj = manifest.require_object(args.object)
    parent_rev = manifest.latest_revision_id()
    canvas = Image.open(manifest.canvas_path(parent_rev)).convert("RGB")

    vital_layers = ki.resolve_vital_layers(args.vital_layers)
    pipe, stage2 = _load_edit_pipeline_and_check(args, ki, torch, manifest, canvas, vital_layers)

    parent_mask = mask_ops.image_to_mask(Image.open(os.path.join(manifest.project_dir, parent_obj["mask"])), h_lat, w_lat)

    if args.mask:
        part_mask_raw = mask_ops.image_to_mask(Image.open(args.mask), h_lat, w_lat)
    else:
        # Derive the part noun's saliency over the WHOLE canvas, then scope
        # it to the parent object's mask (pipelineInc.md §1/§2) — this is
        # what stops "tire" from matching an unrelated tire elsewhere.
        part_mask_raw = ki.derive_object_mask_on_canvas(
            pipe, canvas, args.part_noun, args.prompt, h_lat, w_lat, vital_layers, seed=args.seed, device=args.device,
        )
    part_mask = mask_ops.intersect(part_mask_raw, parent_mask)
    if not part_mask.any():
        raise RuntimeError(
            f"Derived '{args.part_noun}' mask does not overlap parent object "
            f"'{args.object}' at all — pass --mask explicitly, or the part "
            f"noun may not be visually distinguishable at this resolution."
        )

    everywhere = parent_mask.copy()
    everywhere[:] = True
    background, shell, target_zone = mask_ops.zone_masks(everywhere, part_mask, parent=parent_mask)
    zones = ki.ZoneMasks(background=background, shell=shell, target=target_zone)

    edit_img, run_info = ki.run_edit_pass(
        pipe, canvas, args.prompt, zones, h_lat, w_lat, vital_layers,
        seed=args.seed, num_steps=args.steps, guidance_scale=args.guidance,
        inject_cutoff_frac=(0.0, args.inject_cutoff_frac), inject_strength=args.inject_strength,
        device=args.device,
    )

    rev = manifest.add_revision(op="part_edit", prompt=args.prompt, parent=parent_rev,
                                object=args.part, parent_object=args.object)
    manifest.register_object(args.part, args.part_noun, created_at=rev["id"], parent_object=args.object)
    _finalize(args, manifest, rev, edit_img, args.part, part_mask, h_lat, w_lat,
             extra_summary={"stage2_reference_slice": stage2, "stage4_probe": run_info["probe"][:5]})


# ───────────────────────────────── CLI ──────────────────────────────────────

def build_parser():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="command", required=True)

    def add_shared(sp, needs_prompt=True):
        sp.add_argument("--project-dir", required=True)
        sp.add_argument("--seed", type=int, default=42)
        sp.add_argument("--steps", type=int, default=28)
        sp.add_argument("--guidance", type=float, default=2.5)
        sp.add_argument("--device", default="cuda")
        sp.add_argument("--inject-strength", type=float, default=1.0)
        sp.add_argument("--inject-cutoff-frac", type=float, default=0.6)
        sp.add_argument("--vital-layers", default="tier_a")
        sp.add_argument("--edit-model-id", default="black-forest-labs/FLUX.1-Kontext-dev")
        sp.add_argument("--mask", default=None, help="user-supplied PNG override, see pipelineInc.md §2")
        if needs_prompt:
            sp.add_argument("--prompt", required=True)

    sp_init = sub.add_parser("init")
    sp_init.add_argument("--project-dir", required=True)
    sp_init.add_argument("--prompt", required=True)
    sp_init.add_argument("--seed", type=int, default=42)
    sp_init.add_argument("--steps", type=int, default=28)
    sp_init.add_argument("--guidance", type=float, default=3.5)
    sp_init.add_argument("--resolution", type=int, default=1024)
    sp_init.add_argument("--device", default="cuda")
    sp_init.add_argument("--canvas-model-id", default="black-forest-labs/FLUX.1-dev")
    sp_init.set_defaults(func=cmd_init)

    sp_add = sub.add_parser("add")
    add_shared(sp_add)
    sp_add.add_argument("--object", required=True)
    sp_add.add_argument("--noun", required=True)
    sp_add.add_argument("--derive-step", type=int, default=7)
    sp_add.add_argument("--top-k-frac", type=float, default=0.08)
    sp_add.set_defaults(func=cmd_add)

    sp_attr = sub.add_parser("attribute")
    add_shared(sp_attr)
    sp_attr.add_argument("--object", required=True)
    sp_attr.add_argument("--refresh-mask", action="store_true")
    sp_attr.set_defaults(func=cmd_attribute)

    sp_rep = sub.add_parser("replace")
    add_shared(sp_rep)
    sp_rep.add_argument("--object", required=True, help="the object being replaced")
    sp_rep.add_argument("--new-object", required=True, help="name for the replacement object")
    sp_rep.add_argument("--noun", required=True, help="noun for the replacement (for mask re-derivation)")
    sp_rep.set_defaults(func=cmd_replace)

    sp_rm = sub.add_parser("remove")
    add_shared(sp_rm, needs_prompt=False)
    sp_rm.add_argument("--object", required=True)
    sp_rm.add_argument("--fill-prompt", default=None)
    sp_rm.set_defaults(func=cmd_remove)

    sp_part = sub.add_parser("part-edit")
    add_shared(sp_part)
    sp_part.add_argument("--object", required=True, help="parent object name")
    sp_part.add_argument("--part", required=True, help="new name for the part object")
    sp_part.add_argument("--part-noun", required=True)
    sp_part.set_defaults(func=cmd_part_edit)

    return p


def main():
    parser = build_parser()
    args = parser.parse_args()
    try:
        args.func(args)
    except ManifestError as exc:
        print(f"[manifest] {exc}")
        sys.exit(1)
    except Exception as exc:  # noqa: BLE001 — top-level: print a diagnosable hint, see pipelineInc.md §9
        print(f"[error] {type(exc).__name__}: {exc}")
        print(
            "If this is a shape/attribute error inside kontext_injection.py, "
            "the installed diffusers version likely renamed internal "
            "projection methods (to_q/add_q_proj/norm_added_k) or the "
            "Kontext image-encoding API. Dump pipe.transformer.attn_processors "
            "and pipe's public methods to re-derive the current names."
        )
        raise


if __name__ == "__main__":
    main()
