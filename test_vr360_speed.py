import unittest
import json
import tempfile
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
from unittest.mock import patch

import numpy as np
import cv2 as cv

import stage_vr360 as stage
from vr360_contour import ScaleSamples
from vr360_options import style_options
from vr360_union import SYFE_UNION, UNION_MODEL


class SourceGuideTests(unittest.TestCase):
    def test_union_variants_select_the_expected_weights_and_guide_scale(self):
        path = Path(stage.__file__).parent / "vr360_graph.json"
        for variant, weight, scale in (
            ("official", UNION_MODEL, ["union_model", 1]),
            ("syfe", SYFE_UNION, 1.0),
        ):
            graph = json.loads(path.read_text())
            stage.add_guide(graph, "shot", dict(style_options({}), union_variant=variant))
            self.assertEqual(graph["union_model"]["inputs"]["lora_name"], weight)
            self.assertEqual(graph["depth_guide"]["inputs"]["latent_downscale_factor"], scale)

    def test_background_extension_only_writes_unprotected_exterior(self):
        geom = stage.Geometry(960, 540, 110, 50, True)
        ext = stage.BackgroundExtension(geom, style_options({}))
        frame = np.full((540, 960, 3), (120, 105, 70), np.uint8)
        raw = geom.guide(frame)
        expanded = raw.copy()
        ext.apply(expanded, frame, [])
        self.assertTrue(np.array_equal(raw[~geom.invalid], expanded[~geom.invalid]))
        self.assertGreater(np.count_nonzero(expanded[geom.invalid]), 0)
        ext.apply(expanded, frame, [[0, 0, 960, 540]])
        np.testing.assert_array_equal(raw, expanded)

    def test_source_mask_matches_the_tested_geometry_and_all_window_frames(self):
        geom = stage.Geometry(960, 540, 110, 50, True)
        opts = style_options({"front_only": False})
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "guide.npy"
            stage.write_source_mask(path, geom, 9, opts)
            actual = np.load(path, allow_pickle=False)
            stage.write_source_mask(path, geom, 9, dict(opts, front_only=True))
            front = np.load(path, allow_pickle=False)
        coverage = cv.resize((~geom.invalid).astype("f"), (40, 20), interpolation=cv.INTER_AREA)
        np.testing.assert_array_equal(actual, np.broadcast_to(.1 + .9 * coverage, (9, 20, 40)))
        np.testing.assert_array_equal(front, actual[:, :, 8:32])

    def test_source_mask_targets_only_the_original_guide_with_or_without_union(self):
        for union in (False, True):
            graph = json.loads((Path(stage.__file__).parent / "vr360_graph.json").read_text())
            opts = style_options(dict(union_enabled=union, source_mask_enabled=True, source_attention=.8,
                                      joint_audio=False, png_compression=1))
            if union:
                stage.add_guide(graph, "shot", opts)
            stage.apply_inference_controls(graph, "shot", opts)
            self.assertEqual(graph["13"]["class_type"], "LTXAddVideoICLoRAGuideAdvanced")
            self.assertEqual(graph["13"]["inputs"]["attention_mask"], ["source_weights", 0])
            self.assertEqual(graph["13"]["inputs"]["attention_strength"], .8)
            self.assertEqual(graph["12"]["inputs"]["width"], 768)
            self.assertEqual(graph["source_weights"]["inputs"]["width"], 768)
            if union:
                self.assertEqual(graph["depth_guide"]["inputs"]["attention_mask"], ["depth_mask", 0])
                self.assertEqual(graph["depth_mask"]["inputs"]["width"], 768)
            self.assertEqual(graph["20"]["inputs"]["latent_image"], ["depth_guide" if union else "13", 2])
            self.assertEqual(graph["24"]["inputs"]["compression"], 1)
            self.assertNotIn("21", graph)
        graph = json.loads((Path(stage.__file__).parent / "vr360_graph.json").read_text())
        old = json.loads(json.dumps(graph))
        stage.apply_inference_controls(graph, "old", {})
        self.assertEqual(graph, old)


class WindowSchedulingTests(unittest.TestCase):
    def test_prefetch_preserves_cut_and_overlap_order(self):
        edges = [0, 2, 506, 537]
        options = {"generation_fps": 24, "window_frames": 49, "window_overlap": 8}
        expected = [(shot, start, ids) for shot, (begin, end) in enumerate(zip(edges, edges[1:]))
                    for start, _, ids in stage.windows(begin, end, 60, 24, options)]

        def generate(instance, tmp, cap, geom, warp, shot, start, ids, opts):
            return shot, start, ids

        with patch.object(stage, "generate_window", side_effect=generate), ThreadPoolExecutor(1) as pool:
            actual = list(stage.prefetch_windows(pool, None, None, None, None, {}, edges, 60, options))
        self.assertEqual(actual, expected)


class ContourSamplingTests(unittest.TestCase):
    def test_tiled_filter_matches_global_grid_at_edges_and_corners(self):
        rng = np.random.default_rng(103)
        field = rng.uniform(0, 1024, (512, 768)).astype("f")
        positions = (np.array([0, 383, 384, 767]), np.array([2, 40, 251, 511]))
        mask = np.ones_like(field)
        weights = np.ones(4, dtype="f")
        for stride, sigma in [(1, 1.35), (2, 4.), (8, 3.875)]:
            for index in range(len(weights)):
                group = np.array([index])
                local = ScaleSamples(mask, mask, positions, group, weights, sigma, stride, True)
                full = cv.resize(field, (768 // stride, 512 // stride), interpolation=cv.INTER_AREA)
                full = cv.GaussianBlur(full, (0, 0), sigma)
                maps = [(axis[group].astype("f")[:, None] + .5) / stride - .5 for axis in positions]
                expected = cv.remap(full, *maps, cv.INTER_LINEAR, borderMode=cv.BORDER_REPLICATE)
                np.testing.assert_allclose(local.sample(field[local.area]), expected.ravel(), atol=.00025)
