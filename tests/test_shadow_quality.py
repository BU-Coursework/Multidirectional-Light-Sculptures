import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from compare_physical_shadow import register_photo
from light_setup import search_light_setup
from optimize_consistency import optimize_silhouettes
from postprocess_prune import fast_projection_prune
from quality_candidates import connected_core_width, select_candidate, tradeoff_options
from render import render_mesh_shadow, render_shadow
from run_pipeline import assess_strict_hull, run_pipeline
from shadow_hull import compute_shadow_hull, connected_components_3d
from shadow_source import build_sources
from voxel_ops import make_voxel_centers


class ShadowQualityTests(unittest.TestCase):
    def test_pruning_preserves_original_coverage_and_connectivity(self):
        size = 24
        centers = make_voxel_centers(size, size, size, 1)
        mask = np.zeros((size, size), dtype=bool)
        mask[3:15, 5:19] = True
        mask[5:9, 8:12] = False
        originals = build_sources([mask, mask], 1)
        hull = compute_shadow_hull(originals, centers)
        pruned, _ = fast_projection_prune(hull, centers, originals, original_sources=originals,
                                           max_passes=6, verbose=False)
        self.assertLess(pruned.sum(), hull.sum())
        self.assertEqual(connected_components_3d(pruned)[1], 1)
        for source in originals:
            np.testing.assert_array_equal(render_shadow(pruned, centers, source), source.image)

    def test_bridge_joins_diagonal_islands_by_faces(self):
        size = 16
        centers = make_voxel_centers(size, size, size, 1)
        mask = np.zeros((size, size), dtype=bool)
        mask[4:8, 4:8] = True
        mask[8:12, 8:12] = True
        source, = build_sources([mask], 1)
        hull = compute_shadow_hull([source], centers)
        self.assertEqual(connected_components_3d(hull)[1], 1)
        actual = render_shadow(hull, centers, source)
        self.assertFalse((mask & ~actual).any())
        self.assertEqual((actual & ~mask).sum(), 1)

    def test_optimizer_keeps_exact_compatible_originals(self):
        size = 20
        centers = make_voxel_centers(size, size, size, 1)
        first = np.zeros((size, size), dtype=bool)
        first[4:16, 4:16] = True
        first[7:13, 8:12] = False
        second = np.zeros_like(first)
        second[4:16, 5:15] = True
        sources = build_sources([first, second], 1)
        optimized = optimize_silhouettes(sources, centers, verbose=False)
        for original, result in zip(sources, optimized):
            np.testing.assert_array_equal(original.image, result.image)

    def test_fit_diagnostics_separate_sampling_and_silhouette_conflicts(self):
        centers = make_voxel_centers(10, 10, 10, 1)
        top = np.zeros((20, 20), dtype=bool)
        top[:8, :] = True
        bottom = top[::-1].copy()
        hull, fit = assess_strict_hull(build_sources([top, bottom], 1), centers)
        self.assertFalse(hull.any())
        self.assertEqual(fit["disjoint_view_pairs"], [[1, 2]])
        for view in fit["views"]:
            self.assertGreater(view["no_center_ray"], 0)
            self.assertGreater(view["blocked_by_other_views"], 0)
            self.assertEqual(view["no_center_ray"] + view["blocked_by_other_views"], view["target_pixels"])

    def test_mesh_shadow_covers_triangle_footprints(self):
        source, = build_sources([np.zeros((9, 9), dtype=bool)], 1)
        corners = np.array([[0, -.25, -.25], [0, .25, -.25], [0, .25, .25], [0, -.25, .25]])
        triangles = corners[[[0, 1, 2], [0, 2, 3]]]
        expected = np.zeros((9, 9), dtype=bool)
        expected[2:7, 2:7] = True
        np.testing.assert_array_equal(render_mesh_shadow(triangles, source), expected)

    def test_pipeline_reports_original_stl_shadow_at_mismatched_resolution(self):
        with tempfile.TemporaryDirectory() as tmp:
            directory = Path(tmp)
            mask = np.zeros((32, 32), dtype=bool)
            mask[7:24, 8:26] = True
            path = directory / "mask.png"
            Image.fromarray(np.where(mask, 0, 255).astype(np.uint8)).save(path)
            result = run_pipeline([path], grid=16, image_size_value=32,
                                  output_dir=directory, log=lambda _: None)
            scores, = result["hull_summaries"]
            diagnostic, = result["fit_diagnostics"]["views"]
            self.assertGreater(diagnostic["no_center_ray"], 0)
            self.assertGreater(scores["iou"], .8)
            preview = np.asarray(Image.open(directory / "sim" / "hull_view0_target.png")) > 127
            np.testing.assert_array_equal(preview, mask)
            quality = result["candidate_quality"]
            self.assertAlmostEqual(quality["candidates"][quality["selected"]]["mean_iou"], scores["iou"])
            self.assertTrue((directory / "candidate_quality.json").is_file())

    def test_setup_search_preserves_pixels_and_first_view(self):
        size = 36
        first = np.zeros((size, size), dtype=bool)
        second = np.zeros_like(first)
        first[8:25, 10:26] = True
        second[12:29, 10:26] = True
        sources = build_sources([first, second], 80)
        adjusted, adjustments, before, after = search_light_setup(
            sources, size, alignment=True, angles=False)
        self.assertGreater(after, before + .1)
        self.assertNotEqual(adjustments[1].vertical, 0)
        for source, original in zip(adjusted, sources):
            np.testing.assert_array_equal(source.image, original.image)
        np.testing.assert_array_equal(adjusted[0].world_center, sources[0].world_center)
        np.testing.assert_array_equal(adjusted[1].direction, sources[1].direction)

    def test_stl_selection_balances_quality_and_exposes_material_tradeoff(self):
        candidates = [
            {"mean_iou": .9, "score": .89, "mesh_volume": 10, "connected_core_width": 1},
            {"mean_iou": .899, "score": .91, "mesh_volume": 9, "connected_core_width": 1},
            {"mean_iou": .87, "score": .92, "mesh_volume": 8, "connected_core_width": 3},
        ]
        self.assertEqual(select_candidate(candidates), 1)
        self.assertEqual(tradeoff_options(candidates, 1), [2])

    def test_core_width_detects_thin_connector(self):
        thin = np.zeros((24, 16, 16), dtype=bool)
        thin[1:8, 4:11, 4:11] = True
        thin[15:22, 4:11, 4:11] = True
        thin[8:15, 7, 7] = True
        thick = thin.copy()
        thick[8:15, 6:9, 6:9] = True
        self.assertEqual(connected_core_width(thin, 1), 1)
        self.assertGreater(connected_core_width(thick, 1), 1)

    def test_photo_registration_recovers_the_calibrated_frame(self):
        photo = np.full((18, 21), 245, dtype=np.uint8)
        photo[7:10, 6:10] = 20
        image = Image.fromarray(photo)
        corners = [(4, 5), (11, 5), (11, 12), (4, 12)]
        result = register_photo(image, (8, 8), corners)
        np.testing.assert_array_equal(np.asarray(result), photo[5:13, 4:12])
        with self.assertRaisesRegex(ValueError, "clockwise"):
            register_photo(image, (8, 8), corners[::-1])


if __name__ == "__main__":
    unittest.main()
