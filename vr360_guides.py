"""Match and track registered empty stages for optional VR360 conditioning."""

from bisect import bisect_right
import json
import math
import cv2 as cv
import numpy as np
from scipy.ndimage import distance_transform_edt
import stage_background as stage
import engine
from vr360_union import actor_mask


class BackgroundExtension:
    """Extend smooth background colours into the invalid portion of an ERP guide."""

    def __init__(self, geom, options):
        width = options["guide_extension_width"]
        distance, nearest = distance_transform_edt(geom.invalid, return_indices=True)
        self.y, self.x = np.where((distance > 0) & (distance < width))
        rows, cols = nearest[:, self.y, self.x]
        self.source_x = np.clip(np.rint(geom.px[rows, cols]).astype(int), 0, geom.size[0] - 1)
        self.source_y = np.clip(np.rint(geom.py[rows, cols]).astype(int), 0, geom.size[1] - 1)
        fraction = np.clip((width - distance[self.y, self.x]) / (.7 * width), 0, 1)
        self.fade = fraction**2 * (3 - 2 * fraction)
        self.margin = options["guide_extension_margin"]
        self.contrast = options["guide_extension_contrast"]
        self.blur = options["guide_extension_blur"]
        self.kernel = np.ones((13, 13), np.uint8)

    def apply(self, image, frame, boxes):
        protected = actor_mask(frame.shape, boxes, margin=self.margin)
        gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
        local_range = cv.morphologyEx(gray, cv.MORPH_GRADIENT, self.kernel)
        coords = self.source_y, self.source_x
        confidence = np.clip((self.contrast - local_range[coords].astype("f")) / (self.contrast * 2 / 3), 0, 1)
        confidence[protected[coords] != 0] = 0
        colour = cv.GaussianBlur(frame, (0, 0), self.blur)[coords]
        image[self.y, self.x] = np.rint(colour * (self.fade * confidence)[:, None]).astype(np.uint8)
        return self.fade * confidence


def references(angle):
    library, stored = {}, {}
    catalog = engine.MODELS / "stage-library"
    for metadata in sorted(catalog.glob("*/info.json")):
        directory = metadata.parent
        name = directory.name
        info = json.loads(metadata.read_text(encoding="utf8"))
        if info.get("angle") != angle or not info.get("empty_stage"):
            continue
        if not all((directory / f).is_file() for f in ("anchor.png", "panorama.png")):
            continue
        key = "R-" + name
        stored[key] = (info, directory / "panorama.png")
        library[key] = cv.imread(str(directory / "anchor.png"))
    return library, stored


def choose(path, edges, angle, aspect, rate):
    images, stored = references(angle)
    if not images:
        return {}
    sift = cv.SIFT_create(nfeatures=1600, contrastThreshold=0.017)
    library = {k: stage.small_view(v, -1, sift) for k, v in images.items()}
    cap = cv.VideoCapture(str(path))
    rows = []
    warps = {}
    try:
        for j, (a, b) in enumerate(zip(edges, edges[1:])):
            row = {
                "frames": [a, b],
                "guide": "continuous_source",
                "available_stages": list(library),
            }
            if library:
                segment = stage.plan_segment(
                    cap,
                    [a / rate, b / rate, j, None, None, "—", "", None],
                    rate,
                    sift,
                    library,
                    stored,
                    reference_only=True,
                )
                row["reason"] = segment["reason"]
                plate = segment["plate"]
                if plate:
                    mats = stage.track(path, segment, rate, library[plate], angle, sift)
                    valid = stage.stable_track(mats, a, b, rate)
                    row["tracked"] = len(valid)
                    row["samples"] = len(mats)
                    if valid:
                        info, file = stored[plate]
                        row.update(guide="tracked_stage", stage=plate)
                        warps[j] = TrackedERP(file, info["roll"], valid, angle, aspect)
            rows.append(row)
    finally:
        cap.release()
    for row in rows:
        print("VR360", row, flush=True)
    return warps


class TrackedERP:
    def __init__(self, file, roll, records, angle, aspect):
        self.world = cv.imread(str(file))
        self.roll = roll
        self.records = records
        self.ids = [i for i, _ in records]
        yy, xx = np.mgrid[:640, :1280]
        lon = ((xx + 0.5) / 1280 - 0.5) * 2 * np.pi
        lat = ((yy + 0.5) / 640 - 0.5) * np.pi
        self.rays = np.stack(
            (np.cos(lat) * np.sin(lon), np.sin(lat), np.cos(lat) * np.cos(lon)), axis=-1
        ).astype("f")
        z = self.rays[:, :, 2]
        limit = math.tan(math.radians(angle / 2))
        self.out = (
            (z <= 0)
            | (abs(self.rays[:, :, 0]) > z * limit)
            | (abs(self.rays[:, :, 1]) > z * limit / aspect)
        )

    def apply(self, guide, index):
        j = max(0, bisect_right(self.ids, index) - 1)
        i0, m0 = self.records[j]
        i1, m1 = self.records[min(j + 1, len(self.records) - 1)]
        t = np.clip((index - i0) / max(1, i1 - i0), 0, 1)
        u, v, z = (self.rays @ ((1 - t) * m0 + t * m1).T).transpose(2, 0, 1)
        c, s = math.cos(self.roll), math.sin(self.roll)
        u, v = c * u + s * v, -s * u + c * v
        h, w = self.world.shape[:2]
        mx = ((np.arctan2(u, z) / (2 * np.pi) + 0.5) * w - 0.5).astype("f")
        my = np.clip(
            (np.arctan2(v, np.hypot(u, z)) / np.pi + 0.5) * h - 0.5, 0, h - 1
        ).astype("f")
        patch = cv.remap(self.world, mx, my, cv.INTER_LINEAR, borderMode=cv.BORDER_WRAP)
        guide[self.out] = patch[self.out]
        return guide
