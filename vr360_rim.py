"""Low-frequency luminance correction outside the observed image."""

import cv2 as cv
import numpy as np
from scipy.ndimage import gaussian_filter, gaussian_filter1d, distance_transform_edt
from scipy.spatial import cKDTree


class Rim:
    def __init__(self, outer):
        self.outer = outer
        inside = np.uint8(~self.outer)
        contour = cv.findContours(inside, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_NONE)[0][0][
            :, 0
        ].astype("f")
        length = np.r_[
            0,
            np.cumsum(
                np.linalg.norm(
                    np.diff(np.vstack((contour, contour[:1])), axis=0), axis=1
                )
            ),
        ]
        arc = np.arange(0, length[-1], 1.5)
        end = np.vstack((contour, contour[:1]))
        xy = np.stack([np.interp(arc, length, end[:, k]) for k in [0, 1]], axis=1)
        xy = gaussian_filter(xy, (1.2, 0), mode=("wrap", "nearest")).astype("f")
        sd = distance_transform_edt(self.outer) - distance_transform_edt(~self.outer)
        gy, gx = np.gradient(cv.GaussianBlur(sd.astype("f"), (0, 0), 2))
        col = np.rint(xy[:, 0]).astype(int)
        row = np.rint(xy[:, 1]).astype(int)
        norm = np.stack((gx[row, col], gy[row, col]), axis=1)
        norm /= np.maximum(0.001, np.linalg.norm(norm, axis=1, keepdims=True))
        radii = np.arange(-22, 99, dtype="f")
        loc = xy[None, :, :] + norm[None, :, :] * radii[:, None, None]
        self.px = loc[:, :, 0].copy()
        self.py = loc[:, :, 1].copy()
        self.r = radii
        y, x = np.where(self.outer & (sd < 97))
        dist, idx = cKDTree(xy).query(np.stack((x, y), axis=1))
        self.y = y
        self.x = x
        self.map_shape = inside.shape
        self.radius = np.clip(dist, 0, 98).astype("f")
        self.angle = idx.astype("int32")
        self.inside = ~self.outer
        self.untouched = ~(self.outer & (sd < 97))
        self.sd = sd

    def polar(self, im):
        return cv.remap(
            im, self.px, self.py, cv.INTER_LINEAR, borderMode=cv.BORDER_REPLICATE
        )

    def field(self, bg, source):
        observed = self.polar(cv.cvtColor(source, cv.COLOR_BGR2YCrCb)).astype("f")
        gen = self.polar(cv.cvtColor(bg, cv.COLOR_BGR2YCrCb)).astype("f")
        # Smooth along the boundary so roof edges and individual window frames
        # cannot imprint themselves on the expansion as a light/dark stripe.
        ref = gaussian_filter(
            observed[(self.r >= -12) & (self.r <= -3)].mean(0),
            (7, 0),
            mode=("wrap", "nearest"),
        )
        ref_far = gaussian_filter(
            observed[(self.r >= -22) & (self.r <= -14), :, 0].mean(0), 7, mode="wrap"
        )
        profile = gaussian_filter(
            gen, (3.5, 7, 0), mode=("nearest", "wrap", "nearest")
        )[22:]
        dist = self.r[22:, None]
        target = ref[None, :, 0] + np.clip((ref[:, 0] - ref_far) / 11, -0.2, 0.2)[
            None, :
        ] * np.minimum(dist, 32)
        chromadist = np.linalg.norm(ref[None, :, 1:] - profile[:, :, 1:], axis=2)
        color_agreement = np.exp(-0.5 * (chromadist / 28) ** 2)
        # A tapered, bounded gain in encoded luma retains texture detail.
        fade = np.clip((96 - dist) / 54, 0, 1)
        fade = fade * fade * (3 - 2 * fade)
        diff = np.clip(target - profile[:, :, 0], -42, 48)
        gain = (
            np.clip(diff / (profile[:, :, 0] + 12), -0.27, 0.49)
            * color_agreement
            * fade
        )
        return gain.astype("f")


class SmoothRim(Rim):
    def __init__(self, outer):
        super().__init__(outer)
        mask = (self.outer & (self.sd < 97)).astype("f")

        def blur(m):
            return cv.GaussianBlur(
                cv.resize(m, (160, 160), interpolation=cv.INTER_AREA), (0, 0), 2.8
            )

        self.blur = blur
        self.divisor = np.maximum(0.0001, blur(mask))
        t = np.clip((96 - self.radius) / 22, 0, 1)
        self.tail = t * t * (3 - 2 * t)

    def field(self, bg, source):
        f = super().field(bg, source)
        # Limit narrow angular corrections from a limb, hair or furniture to the immediate seam.
        smooth = gaussian_filter1d(f, 25, axis=1, mode="wrap")
        t = np.clip((self.r[22:, None] - 2) / 30, 0, 1)
        return (smooth + (f - smooth) * (1 - 0.94 * t)).astype("f")

    def adjust(self, bg, gain):
        lo = self.radius.astype("int32")
        hi = np.minimum(lo + 1, len(gain) - 1)
        frac = self.radius - lo
        raw = gain[lo, self.angle] * (1 - frac) + gain[hi, self.angle] * frac
        field = np.zeros(bg.shape[:2], "f")
        field[self.y, self.x] = raw
        smoothed = cv.resize(
            self.blur(field) / self.divisor, (640, 640), interpolation=cv.INTER_LINEAR
        )[self.y, self.x]
        entry = np.clip((self.radius - 1) / 11, 0, 1)
        entry = entry * entry * (3 - 2 * entry)
        g = smoothed * self.tail * entry
        pix = bg[self.y, self.x]
        Y = pix @ np.array([0.114, 0.587, 0.299], dtype="f")
        delta = np.rint(np.clip(g * (Y + 12), -42, 48)).astype("int16")
        delta = np.clip(
            delta,
            -pix.min(axis=1).astype("int16"),
            255 - pix.max(axis=1).astype("int16"),
        )
        result = bg.copy()
        result[self.y, self.x] = (pix.astype("int16") + delta[:, None]).astype("uint8")
        return result
