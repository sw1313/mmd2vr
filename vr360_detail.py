"""Anime SR luminance detail from the A reference render."""

import cv2 as cv
import numpy as np
import torch


def detail(im, net, gate):
    rgb = (
        torch.from_numpy(cv.cvtColor(im, cv.COLOR_BGR2RGB))
        .permute(2, 0, 1)
        .unsqueeze(0)
        .to("cuda", torch.float16)
        / 255
    )
    # 2x display detail from the model's native 4x result. The input is the shared FP8 backdrop.
    with torch.inference_mode():
        pred = net(rgb)
        half = torch.nn.functional.avg_pool2d(pred, 2).clamp(0, 1)
        rgb8 = (
            half[0].mul(255).round().byte().permute(1, 2, 0).contiguous().cpu().numpy()
        )
    ref = cv.resize(
        cv.cvtColor(im, cv.COLOR_BGR2GRAY),
        (1280, 1280),
        interpolation=cv.INTER_LANCZOS4,
    ).astype("f")
    lum = cv.cvtColor(rgb8, cv.COLOR_RGB2GRAY).astype("f")
    diff = lum - ref
    extra = 0.72 * (diff - cv.GaussianBlur(diff, (0, 0), 2.0)) + 0.43 * (
        ref - cv.GaussianBlur(ref, (0, 0), 2.3)
    )
    return np.uint8(np.rint(128 + np.clip(extra, -14, 14) * gate))
