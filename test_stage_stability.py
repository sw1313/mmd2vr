"""A partially tracked shot must not alternate between a generated set and ambient light."""

import tempfile
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np

import stage_background as stage


class StageStabilityTests(unittest.TestCase):
    def test_unmatched_empty_stage_does_not_create_a_new_plate(self):
        image = np.full((90, 160, 3), 150, dtype=np.uint8)
        capture = MagicMock()
        capture.read.return_value = True, image
        library = {}
        seg = stage.plan_segment(
            capture, (0, 3, 1, None, "", "B01", "", 1), 30,
            stage.cv2.SIFT_create(), library, {}, reference_only=True,
        )
        self.assertEqual(seg["action"], stage.AMBIENT)
        self.assertIsNone(seg["plate"])
        self.assertFalse(library)

    def test_sparse_failures_can_be_bridged_but_sustained_loss_cannot(self):
        identity = np.eye(3)

        def samples(missing):
            return [(i, None if i in missing else identity) for i in range(60)]

        self.assertEqual(len(stage.stable_track(samples({30}), 0, 60, 30)), 59)
        self.assertEqual(stage.stable_track(samples(range(30, 35)), 0, 60, 30), [])
        self.assertEqual(stage.stable_track(samples(range(0, 60, 5)), 0, 60, 30), [])
        self.assertEqual(stage.stable_track(samples(set())[:-1], 0, 60, 30), [])

    def test_one_failed_interval_downgrades_the_whole_shot_before_generation(self):
        def candidate(capture, row, rate, sift, library, stored, *, reference_only):
            self.assertFalse(reference_only)
            start, end, _ = row
            key = str(start)
            library[key] = object()
            return {
                "start": start,
                "end": end,
                "ref": start * 30,
                "plate": key,
                "action": "生成",
                "reason": "",
            }

        def measured(clip, segment, rate, anchor, angle, sift):
            start = segment["start"] * 30
            return [
                (i, None if start == 30 and i < 40 else np.eye(3))
                for i in range(start, start + 30)
            ]

        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            with (
                patch.object(stage.engine, "MODELS", root),
                patch.object(stage.cv2, "VideoCapture", return_value=MagicMock()),
                patch.object(stage, "plan_segment", side_effect=candidate),
                patch.object(stage, "track", side_effect=measured),
                patch.object(
                    stage, "shot_palette", return_value=[[[10, 10, 10]]]
                ) as palette,
                patch.object(
                    stage.stage_atlas,
                    "foreground_predictor",
                    side_effect=AssertionError,
                ),
            ):
                segments, tracks, files = stage.plan(
                    root / "clip",
                    [(0, 1, 1), (1, 2, 1)],
                    30,
                    100,
                    None,
                    root,
                    root,
                    2880,
                )
            self.assertEqual([s["action"] for s in segments], [stage.AMBIENT] * 2)
            self.assertFalse(tracks)
            self.assertFalse(files)
            palette.assert_called_once()
