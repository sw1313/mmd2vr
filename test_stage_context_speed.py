import unittest

import numpy as np

import stage_context as context


class CloseupSpeedTests(unittest.TestCase):
    def test_sampled_windows_preserve_cut_endpoints_and_native_timestamps(self):
        # A cut between two non-sampled timestamps and a cut at a full window.
        frames = [(i, np.full((4, 4, 3), color, np.uint8))
                  for i, color in enumerate([0] * 206 + [255] * 5 + [0] * 2)]
        batches = list(context.windows(context.sampled_frames(frames, 3)))
        self.assertEqual([carry for _, carry in batches], [False, True, False, False])
        self.assertEqual([item[0] for item in batches[1][0]], [205])
        self.assertEqual([item[0] for item in batches[2][0]], [206, 209, 210])
        self.assertEqual([item[0] for item in batches[-1][0]], [211, 212])
        restored = []
        for batch, carry in batches:
            begin = restored[-1] + 1 if carry else batch[0][0]
            restored.extend(range(begin, batch[-1][0] + 1))
        self.assertEqual(restored, list(range(len(frames))))

    def test_standard_keeps_every_frame_and_uses_thirty_steps(self):
        thumb = np.zeros((4, 4, 3), np.uint8)
        batches = context.windows(context.sampled_frames(((i, thumb) for i in range(141)), 1))
        self.assertEqual([index for batch, _ in batches for index, _ in batch], list(range(141)))
        self.assertEqual(context.graph("control", "mask", 33)["13"]["inputs"]["steps"], 30)


if __name__ == "__main__":
    unittest.main()
