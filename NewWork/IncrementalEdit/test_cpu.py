"""
CPU-only unit tests (pipelineInc.md §10.2). No torch dependency — run this
directly to validate manifest.py, mask_ops.py, and metrics.py in any
environment, GPU or not:

    python test_cpu.py

kontext_injection.py (the torch-dependent attention processor) is NOT
exercised here; it can only be validated by ast.parse in this environment
(see pipelineInc.md §10.1) plus manual review against the diffusers source.
"""

import math
import os
import shutil
import tempfile
import unittest

import numpy as np
from PIL import Image

import mask_ops
import metrics
from manifest import SceneManifest, ManifestError


class TestMaskOps(unittest.TestCase):
    def test_rect_mask_fractions(self):
        h, w = 8, 8
        self.assertEqual(mask_ops.rect_mask(h, w, "none").sum(), 0)
        self.assertEqual(mask_ops.rect_mask(h, w, "all").sum(), h * w)
        self.assertEqual(mask_ops.rect_mask(h, w, "left").sum(), h * (w // 2))
        self.assertEqual(mask_ops.rect_mask(h, w, "right").sum(), h * (w - w // 2))
        self.assertEqual(mask_ops.rect_mask(h, w, "top").sum(), (h // 2) * w)
        self.assertEqual(mask_ops.rect_mask(h, w, "bottom").sum(), (h - h // 2) * w)

    def test_rect_mask_flattens_to_token_count(self):
        h, w = 13, 17  # deliberately non-square, non-power-of-two
        for region in ["none", "all", "left", "right", "top", "bottom", "center"]:
            m = mask_ops.rect_mask(h, w, region)
            self.assertEqual(m.shape, (h * w,))

    def test_rect_mask_unknown_region_raises(self):
        with self.assertRaises(ValueError):
            mask_ops.rect_mask(4, 4, "diagonal")

    def test_topk_mask_fraction(self):
        n = 1000
        rng = np.random.default_rng(0)
        saliency = rng.random(n)
        m = mask_ops.topk_mask(saliency, 0.1)
        self.assertEqual(m.sum(), 100)
        # The masked tokens must be exactly the top-100 by score.
        top_idx = set(np.argsort(saliency)[-100:].tolist())
        self.assertEqual(set(np.nonzero(m)[0].tolist()), top_idx)

    def test_topk_mask_full_when_frac_geq_one(self):
        saliency = np.arange(10)
        m = mask_ops.topk_mask(saliency, 1.5)
        self.assertTrue(m.all())

    def test_intersect_union_invert_subtract(self):
        a = np.array([True, True, False, False])
        b = np.array([True, False, True, False])
        np.testing.assert_array_equal(mask_ops.intersect(a, b), [True, False, False, False])
        np.testing.assert_array_equal(mask_ops.union(a, b), [True, True, True, False])
        np.testing.assert_array_equal(mask_ops.invert(a), [False, False, True, True])
        np.testing.assert_array_equal(mask_ops.subtract(a, b), [False, True, False, False])

    def test_zone_masks_top_level(self):
        n = 16
        everywhere = np.ones(n, dtype=bool)
        target = np.zeros(n, dtype=bool)
        target[:4] = True
        bg, shell, tgt = mask_ops.zone_masks(everywhere, target, parent=None)
        self.assertEqual(shell.sum(), 0)
        np.testing.assert_array_equal(tgt, target)
        np.testing.assert_array_equal(bg, ~target)
        # background, shell, target must partition `everywhere` exactly.
        union_all = bg | shell | tgt
        np.testing.assert_array_equal(union_all, everywhere)
        self.assertEqual((bg & tgt).sum(), 0)

    def test_zone_masks_part_edit_scoping(self):
        n = 16
        parent = np.zeros(n, dtype=bool)
        parent[:8] = True  # "the car"
        target = np.zeros(n, dtype=bool)
        target[2:4] = True  # "the tire", inside the car
        everywhere = np.ones(n, dtype=bool)
        bg, shell, tgt = mask_ops.zone_masks(everywhere, target, parent=parent)
        # shell = rest of the car minus the tire
        expected_shell = np.zeros(n, dtype=bool)
        expected_shell[[0, 1, 4, 5, 6, 7]] = True
        np.testing.assert_array_equal(shell, expected_shell)
        # background = everything outside the car entirely
        np.testing.assert_array_equal(bg, ~parent)
        np.testing.assert_array_equal(tgt, target)

    def test_zone_masks_part_edit_rejects_non_subset_target(self):
        n = 8
        parent = np.array([True] * 4 + [False] * 4)
        target = np.array([False, False, False, False, True, False, False, False])  # outside parent
        with self.assertRaises(ValueError):
            mask_ops.zone_masks(np.ones(n, dtype=bool), target, parent=parent)

    def test_mask_image_roundtrip(self):
        h, w = 6, 5
        rng = np.random.default_rng(1)
        mask = rng.random(h * w) > 0.5
        img = mask_ops.mask_to_image(mask, h, w)
        recovered = mask_ops.image_to_mask(img, h, w)
        np.testing.assert_array_equal(recovered, mask)

    def test_upsample_mask_shape_and_content(self):
        token_mask = np.array([[True, False], [False, True]])
        up = mask_ops.upsample_mask(token_mask, height=8, width=8)
        self.assertEqual(up.shape, (8, 8))
        self.assertTrue(up[0:4, 0:4].all())
        self.assertFalse(up[0:4, 4:8].any())
        self.assertFalse(up[4:8, 0:4].any())
        self.assertTrue(up[4:8, 4:8].all())

    def test_upsample_mask_rejects_non_multiple(self):
        token_mask = np.array([[True, False], [False, True], [True, True]])  # h_lat=3
        with self.assertRaises(ValueError):
            mask_ops.upsample_mask(token_mask, height=7, width=8)  # 7 % 3 != 0

    def test_assert_token_count_passes_and_fails(self):
        mask_ops.assert_token_count(np.zeros(12, dtype=bool), 3, 4, "ok case")
        with self.assertRaises(ValueError):
            mask_ops.assert_token_count(np.zeros(11, dtype=bool), 3, 4, "bad case")


class TestMetrics(unittest.TestCase):
    def test_latent_mse_zero_for_identical(self):
        a = np.random.default_rng(0).random((2, 100, 16)).astype(np.float32)
        self.assertEqual(metrics.latent_mse(a, a), 0.0)

    def test_latent_mse_known_value(self):
        a = np.zeros((1, 4, 1), dtype=np.float32)
        b = np.ones((1, 4, 1), dtype=np.float32) * 2.0
        self.assertAlmostEqual(metrics.latent_mse(a, b), 4.0, places=5)

    def test_masked_latent_mse_numpy_path(self):
        edit = np.zeros((1, 4, 2), dtype=np.float32)
        ref = np.zeros((1, 4, 2), dtype=np.float32)
        ref[:, 0, :] = 10.0  # only token 0 differs
        mask_all = np.array([True, True, True, True])
        mask_tok0 = np.array([True, False, False, False])
        mask_rest = np.array([False, True, True, True])
        self.assertGreater(metrics.masked_latent_mse(edit, ref, mask_tok0), 0.0)
        self.assertEqual(metrics.masked_latent_mse(edit, ref, mask_rest), 0.0)
        self.assertGreater(metrics.masked_latent_mse(edit, ref, mask_all), 0.0)

    def test_image_psnr_identical_is_inf(self):
        a = np.random.default_rng(0).integers(0, 255, (16, 16, 3)).astype(np.uint8)
        self.assertEqual(metrics.image_psnr(a, a), float("inf"))

    def test_image_psnr_known_value(self):
        a = np.zeros((4, 4, 3), dtype=np.uint8)
        b = np.full((4, 4, 3), 10, dtype=np.uint8)
        expected = 20 * math.log10(255.0) - 10 * math.log10(100.0)
        self.assertAlmostEqual(metrics.image_psnr(a, b), expected, places=4)

    def test_region_psnr_all_mask_equals_image_psnr(self):
        rng = np.random.default_rng(2)
        a = rng.integers(0, 255, (10, 10, 3)).astype(np.uint8)
        b = rng.integers(0, 255, (10, 10, 3)).astype(np.uint8)
        full_mask = np.ones((10, 10), dtype=bool)
        self.assertAlmostEqual(metrics.region_psnr(a, b, full_mask), metrics.image_psnr(a, b), places=6)

    def test_region_psnr_isolates_region(self):
        a = np.zeros((4, 4, 3), dtype=np.uint8)
        b = a.copy()
        b[0, 0, :] = 255  # only one pixel changed
        mask_hit = np.zeros((4, 4), dtype=bool)
        mask_hit[0, 0] = True
        mask_miss = ~mask_hit
        self.assertLess(metrics.region_psnr(a, b, mask_hit), 20.0)  # big diff -> low PSNR
        self.assertEqual(metrics.region_psnr(a, b, mask_miss), float("inf"))  # untouched -> inf

    def test_preservation_gap(self):
        self.assertGreater(metrics.preservation_gap(40.0, 25.0), 0.0)  # PASS case
        self.assertLess(metrics.preservation_gap(20.0, 30.0), 0.0)     # CHECK case
        self.assertEqual(metrics.preservation_gap(float("inf"), float("inf")), 0.0)

    def test_lpips_dist_never_raises(self):
        # Must never raise regardless of whether torch/lpips happen to be
        # installed in the environment this runs in (pipelineInc.md §9) —
        # a too-small image is exactly the kind of failure that must
        # degrade to None rather than crash a real edit run.
        tiny = np.zeros((8, 8, 3), dtype=np.uint8)
        result = metrics.lpips_dist(tiny, tiny)
        self.assertTrue(result is None or isinstance(result, float))

    def test_lpips_dist_identical_images_near_zero_when_available(self):
        # AlexNet's pooling stack needs a real-sized input; use one large
        # enough to work if lpips/torch ARE installed. If they aren't,
        # this degrades to None like every other call site.
        img = np.random.default_rng(0).integers(0, 255, (64, 64, 3)).astype(np.uint8)
        result = metrics.lpips_dist(img, img)
        if result is not None:
            self.assertAlmostEqual(result, 0.0, places=4)


class TestManifest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.mkdtemp(prefix="incedit_test_")

    def tearDown(self):
        shutil.rmtree(self.tmp, ignore_errors=True)

    def test_create_then_load_roundtrip(self):
        m = SceneManifest.create(self.tmp, resolution=[1024, 1024])
        rev = m.add_revision(op="init", prompt="an empty driveway", parent=None)
        m.save()

        m2 = SceneManifest.load(self.tmp)
        self.assertEqual(m2.resolution, [1024, 1024])
        self.assertEqual(len(m2.revisions), 1)
        self.assertEqual(m2.revisions[0]["op"], "init")
        self.assertEqual(rev["id"], 0)

    def test_create_twice_raises(self):
        m = SceneManifest.create(self.tmp, resolution=[512, 512])
        m.add_revision(op="init", prompt="p", parent=None)
        m.save()  # create() only guards against clobbering a SAVED project
        with self.assertRaises(ManifestError):
            SceneManifest.create(self.tmp, resolution=[512, 512])

    def test_load_missing_raises(self):
        with self.assertRaises(ManifestError):
            SceneManifest.load(self.tmp)

    def test_register_and_require_object(self):
        m = SceneManifest.create(self.tmp, resolution=[512, 512])
        m.add_revision(op="init", prompt="p", parent=None)
        rev1 = m.add_revision(op="add", prompt="add car", parent=0, object="car_1")
        m.register_object("car_1", "car", created_at=rev1["id"])

        obj = m.require_object("car_1")
        self.assertEqual(obj["noun"], "car")
        self.assertIsNone(obj["retired_at"])

        with self.assertRaises(ManifestError):
            m.require_object("does_not_exist")

    def test_register_duplicate_active_name_raises(self):
        m = SceneManifest.create(self.tmp, resolution=[512, 512])
        m.add_revision(op="init", prompt="p", parent=None)
        m.register_object("car_1", "car", created_at=0)
        with self.assertRaises(ManifestError):
            m.register_object("car_1", "car", created_at=0)

    def test_retire_object_orphans_children_and_blocks_future_targeting(self):
        m = SceneManifest.create(self.tmp, resolution=[512, 512])
        m.add_revision(op="init", prompt="p", parent=None)
        m.register_object("car_1", "car", created_at=0)
        m.register_object("mirror_1", "mirror", created_at=0, parent_object="car_1")

        rev = m.add_revision(op="replace", prompt="a bicycle", parent=0, object="cycle_1", replaces="car_1")
        orphaned = m.retire_object("car_1", retired_at=rev["id"])

        self.assertEqual(orphaned, ["mirror_1"])
        self.assertIsNone(m.objects["mirror_1"]["parent_object"])
        self.assertEqual(m.objects["mirror_1"]["orphaned_from"], "car_1")
        with self.assertRaises(ManifestError):
            m.require_object("car_1")  # retired objects can't be targeted directly

    def test_register_object_with_missing_parent_raises(self):
        m = SceneManifest.create(self.tmp, resolution=[512, 512])
        m.add_revision(op="init", prompt="p", parent=None)
        with self.assertRaises(ManifestError):
            m.register_object("tire_1", "tire", created_at=0, parent_object="cycle_1")

    def test_active_objects_excludes_retired(self):
        m = SceneManifest.create(self.tmp, resolution=[512, 512])
        m.add_revision(op="init", prompt="p", parent=None)
        m.register_object("car_1", "car", created_at=0)
        rev = m.add_revision(op="remove", prompt="p", parent=0, object="car_1")
        m.retire_object("car_1", retired_at=rev["id"])
        self.assertEqual(m.active_objects(), {})

    def test_atomic_save_leaves_no_tmp_file_and_is_idempotent(self):
        m = SceneManifest.create(self.tmp, resolution=[256, 256])
        m.add_revision(op="init", prompt="p", parent=None)
        m.save()
        m.save()  # second save must not error or leave stray temp files
        import os

        entries = os.listdir(self.tmp)
        self.assertIn("manifest.json", entries)
        self.assertFalse(any(e.startswith(".manifest_") for e in entries))

    def test_mask_path_and_canvas_path(self):
        m = SceneManifest.create(self.tmp, resolution=[256, 256])
        self.assertEqual(m.mask_path("car_1"), os.path.join(self.tmp, "masks", "car_1.png"))
        self.assertEqual(m.canvas_path(3), os.path.join(self.tmp, "canvas_v3.png"))


if __name__ == "__main__":
    unittest.main(verbosity=2)
