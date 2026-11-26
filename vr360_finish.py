"""Apply the selected exterior finish in panoramic coordinates, before projection."""

import cv2 as cv
import numpy as np
from scipy.ndimage import distance_transform_edt, gaussian_filter1d

from vr360_union import actor_mask


class ExteriorFinish:
    def __init__(self, geom, options):
        self.geom = geom
        self.options = options
        outside = cv.distanceTransform(geom.invalid.astype("uint8"), cv.DIST_L2, 5)
        feather = options["finish_feather"]
        alpha8 = (np.rint(255 * np.clip(outside / feather, 0, 1)).astype("uint8")
                  if feather else geom.invalid.astype("uint8") * 255)
        self.alpha = alpha8.astype("f") / 255
        self.interior = (255 - alpha8).astype("f") / 255
        self.lut = None
        if options["palette_enabled"]:
            yy, xx = np.where((outside >= 5) & (outside <= 14))
            self.yy, self.xx = yy[::3], xx[::3]
            distance = cv.distanceTransform(
                (~geom.invalid).astype("uint8"), cv.DIST_L2, 5
            )
            nearest = distance_transform_edt(
                distance < 12, return_distances=False, return_indices=True
            )
            self.iy, self.ix = nearest[:, self.yy, self.xx]

    def fit_palette(self, samples):
        colors, differences = [], []
        for raw, panorama, people in samples:
            original = self.geom.guide(cv.GaussianBlur(raw, (7, 7), 2))
            background = cv.GaussianBlur(panorama, (7, 7), 2)
            blocked = self.geom.guide(actor_mask(raw.shape, people, 4))[
                self.iy, self.ix
            ]

            def lab(values):
                return (
                    cv.cvtColor(values.reshape(-1, 1, 3), cv.COLOR_BGR2LAB)
                    .reshape(-1, 3)
                    .astype("f")
                )

            source = lab(background[self.yy, self.xx])
            delta = lab(original[self.iy, self.ix]) - source
            valid = (
                (blocked == 0)
                & (np.abs(delta[:, 0]) < 55)
                & (np.linalg.norm(delta[:, 1:], axis=1) < 30)
            )
            colors.append(source[valid])
            differences.append(delta[valid])
        source = np.concatenate(colors)
        if len(source) < 100:
            return
        delta = np.concatenate(differences)
        levels = np.arange(256, dtype="f")
        curves = []
        for channel, limit in enumerate((15, 9, 9)):
            offsets = np.zeros(256)
            for value in range(256):
                mask = np.abs(source[:, channel] - value) < 12
                count = mask.sum()
                if count:
                    offsets[value] = (
                        np.clip(np.median(delta[mask, channel]), -limit, limit)
                        * count
                        / (count + 35)
                    )
            mapped = (
                levels
                + gaussian_filter1d(offsets, 4) * self.options["palette_strength"]
            )
            curves.append(np.rint(np.clip(mapped, 0, 255)).astype(np.uint8))
        self.lut = np.stack(curves, axis=1).reshape(256, 1, 3)

    def apply(self, image):
        o = self.options
        if o["texture_enabled"] and o["texture_mix"]:
            radius = o["texture_radius"]
            pad = max(12, 2 * radius)
            padded = cv.copyMakeBorder(image, 0, 0, pad, pad, cv.BORDER_WRAP)
            diameter = 2 * radius + 1
            filtered = cv.bilateralFilter(
                padded, diameter, o["texture_sigma1"], o["texture_space"]
            )
            if o["texture_passes"] == 2:
                filtered = cv.bilateralFilter(
                    filtered, diameter, o["texture_sigma2"], o["texture_space"]
                )
            mixed = cv.addWeighted(
                filtered[:, pad:-pad], o["texture_mix"], image, 1 - o["texture_mix"], 0
            )
            image = cv.blendLinear(image, mixed, self.interior, self.alpha)
        if self.lut is not None and o["palette_strength"]:
            graded = cv.cvtColor(
                cv.LUT(cv.cvtColor(image, cv.COLOR_BGR2LAB), self.lut), cv.COLOR_LAB2BGR
            )
            image = cv.blendLinear(image, graded, self.interior, self.alpha)
        return image
