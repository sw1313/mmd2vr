"""Millisecond Matroska background timestamps must never hold back an SBS frame."""

import json
import subprocess
import tempfile
import unittest
from fractions import Fraction
from pathlib import Path

import numpy as np

from vr180 import projection_args


class StageProjectionTests(unittest.TestCase):
    def test_background_keeps_foreground_frame_order(self):
        for rate in (Fraction(30), Fraction(30000, 1001)):
            with self.subTest(rate=rate), tempfile.TemporaryDirectory() as tmp:
                work = Path(tmp)
                count = 21
                geometry = (128, 64)
                levels = (32 + np.arange(count) * 9).astype("uint8")
                pixels = np.broadcast_to(
                    levels[:, None, None], (count, *geometry[::-1])
                )
                stereo = work / "master.nut"
                # NUT preserves exact rational frame times. The background uses a millisecond time base.
                cmd = [
                    "ffmpeg",
                    "-v",
                    "error",
                    "-f",
                    "rawvideo",
                    "-s",
                    "128x64",
                    "-pix_fmt",
                    "gray",
                    "-r",
                    str(rate),
                    "-i",
                    "-",
                    "-c:v",
                    "ffv1",
                    str(stereo),
                ]
                subprocess.run(cmd, input=pixels.tobytes(), check=True)
                stage = work / "stage.mkv"
                subprocess.run(
                    [
                        "ffmpeg",
                        "-v",
                        "error",
                        "-f",
                        "lavfi",
                        "-i",
                        f"color=white:s=96x96:r={rate}",
                        "-frames:v",
                        str(count),
                        "-c:v",
                        "ffv1",
                        str(stage),
                    ],
                    check=True,
                )
                for curve, zoom in ((0, 1.0), (50, 1.0), (0, 1.10), (100, 1.10)):
                    options = projection_args(
                        64,
                        64,
                        100,
                        96,
                        rate,
                        False,
                        curvature=curve,
                        view_zoom=zoom,
                        keep_center=True,
                        work=work,
                        stage_path=stage,
                    )
                    dest = work / f"projected-{curve}-{zoom}.mp4"
                    subprocess.run(
                        [
                            "ffmpeg",
                            "-v",
                            "error",
                            "-i",
                            str(stereo),
                            "-f",
                            "lavfi",
                            "-i",
                            "anullsrc",
                            *options[: options.index("-c:v")],
                            "-c:v",
                            "libx264",
                            "-qp",
                            "0",
                            str(dest),
                        ],
                        check=True,
                    )
                    raw = subprocess.check_output(
                        [
                            "ffmpeg",
                            "-v",
                            "error",
                            "-i",
                            str(dest),
                            "-pix_fmt",
                            "gray",
                            "-f",
                            "rawvideo",
                            "-",
                        ]
                    )
                    frames = np.frombuffer(raw, np.uint8).reshape(-1, 96, 192)
                    for x in (48, 144):
                        np.testing.assert_allclose(frames[:, 48, x], levels, atol=2)
                    meta = json.loads(
                        subprocess.check_output(
                            [
                                "ffprobe",
                                "-v",
                                "error",
                                "-show_streams",
                                "-of",
                                "json",
                                str(dest),
                            ]
                        )
                    )
                    self.assertEqual(
                        Fraction(meta["streams"][0]["avg_frame_rate"]), rate
                    )


if __name__ == "__main__":
    unittest.main()
