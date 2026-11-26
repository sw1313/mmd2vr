"""Blend low-frequency colour across the exterior of the projected stereo source."""

import json
from contextlib import ExitStack
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import subprocess
import sys

import cv2 as cv
import numpy as np


class ScaleSamples:
    def __init__(self, inner, exterior, positions, indices, weights, sigma, stride, tiled):
        self.indices = indices
        self.weights = weights[self.indices]
        self.sigma = sigma
        h, w = inner.shape
        out_w, out_h = max(1, w // stride), max(1, h // stride)
        x, y = (axis[indices] for axis in positions)
        # Preserve the global downsampling grid and all Gaussian and remap neighbours.
        if tiled:
            halo = int(np.ceil(4 * sigma)) + 2
            x0, y0 = max(0, x.min() // stride - halo), max(0, y.min() // stride - halo)
            x1 = min(out_w, x.max() // stride + halo + 1)
            y1 = min(out_h, y.max() // stride + halo + 1)
            self.area = np.s_[y0*stride:y1*stride, x0*stride:x1*stride]
        else:
            x0, y0, x1, y1 = 0, 0, out_w, out_h
            self.area = np.s_[:, :]
        self.size = (x1 - x0, y1 - y0)
        # OpenCV remap requires each dimension to fit within a signed short.
        pad = (-len(self.indices)) % 512
        self.maps = [
            np.pad((coord.astype("f") + .5) * ratio - .5 - float(origin),
                   (0, pad), mode="edge").reshape(-1, 512)
            for coord, ratio, origin in ((x, out_w / w, x0), (y, out_h / h, y0))
        ]
        self.inner = inner[self.area]
        self.exterior = exterior[self.area]
        self.inner_weight = np.maximum(self.sample(self.inner), .0001)
        self.outer_weight = np.maximum(self.sample(self.exterior), .0001)

    def sample(self, field):
        if self.size != field.shape[::-1]:
            field = cv.resize(field, self.size, interpolation=cv.INTER_AREA)
        blurred = cv.GaussianBlur(field, (0, 0), self.sigma)
        return cv.remap(blurred, *self.maps, cv.INTER_LINEAR,
                        borderMode=cv.BORDER_REPLICATE).ravel()[:len(self.indices)]


class ExteriorBand:
    def __init__(self, inner, protected, subsampling, width):
        self.inner = inner.astype("f")
        self.exterior = (~protected).astype("f")
        distance = cv.distanceTransform(self.exterior.astype("uint8"), cv.DIST_L2, cv.DIST_MASK_PRECISE)
        distance *= subsampling
        self.y, self.x = np.where((~protected) & (distance <= width))
        d = distance[self.y, self.x]
        relative = width / 64
        t = np.clip((d - 1) / (width - 1), 0, 1)
        self.fade = 1 - t*t*(3 - 2*t)
        self.near = np.exp(-d / (4 * relative))
        anchors = np.array([1, 5, 12, 26, 52], dtype="f") * relative
        self.scales = []
        for index, (sigma, step) in enumerate(((1.35, 1), (3.6, 1), (8, 2), (17, 4), (31, 8))):
            weights = np.interp(d, anchors, np.eye(5)[index]).astype("f")
            indices = np.flatnonzero(weights > 0)
            if not indices.size:
                continue
            stride = max(1, int(step * relative / subsampling))
            h, w = inner.shape
            tiled = h % stride == 0 and w % stride == 0
            groups = [indices]
            if tiled:
                cell = 384
                tile = (self.y[indices] // cell) * ((w + cell - 1) // cell) + self.x[indices] // cell
                groups = [indices[tile == key] for key in np.unique(tile)]
            for group in groups:
                self.scales.append(ScaleSamples(
                    self.inner, self.exterior, (self.x, self.y), group, weights,
                    max(.65, sigma * relative / (subsampling * stride)), stride, tiled,
                ))

    def apply(self, plane, strength, ten_bit, luma):
        unit = 4 if ten_bit else 1
        samples = plane[self.y, self.x].astype("f") / unit
        offset = np.zeros_like(samples)
        for scale in self.scales:
            ids = scale.indices
            values = plane[scale.area].astype("f")
            if ten_bit:
                values *= .25
            target = scale.sample(values * scale.inner) / scale.inner_weight
            background = scale.sample(values * scale.exterior) / scale.outer_weight
            texture = np.abs(samples[ids] - background)
            cap = (38 if luma else 24) / (1 + texture / (22 if luma else 14))
            delta = np.clip(target - background, -cap, cap)
            limit = 6 if luma else 3
            halo = np.clip(background - samples[ids], -limit, limit)
            offset[ids] += scale.weights * (delta + .60 * self.near[ids] * halo)
        corrected = (samples + strength * self.fade * offset) * unit
        plane[self.y, self.x] = np.clip(np.rint(corrected), 0, 1023 if ten_bit else 255)


def render(plan):
    if plan["make_mask"]:
        subprocess.run(plan["make_mask"], check=True)
    eye = plan["eye"]
    alpha = cv.imread(plan["mask"], cv.IMREAD_UNCHANGED)
    if alpha is None or alpha.shape != (eye, 2 * eye):
        raise ValueError("投影蒙版尺寸无效")
    left, right = np.hsplit(alpha > 0, 2)
    if not np.array_equal(left, right):
        raise ValueError("左右眼的投影轮廓不一致")
    chroma = left.reshape(eye // 2, 2, eye // 2, 2)
    bands = (ExteriorBand(left, left, 1, plan["width"]),
             ExteriorBand(chroma.all(axis=(1, 3)), chroma.any(axis=(1, 3)), 2, plan["width"]))
    ten = plan["ten_bit"]
    dtype = np.dtype("<u2" if ten else "u1")
    size = eye * eye
    frame_bytes = 3 * size * dtype.itemsize
    with ExitStack() as stack:
        # Kill before waiting if the other end fails during streaming.
        project = stack.enter_context(subprocess.Popen(plan["project"], stdout=subprocess.PIPE))
        stack.callback(project.kill)
        encode = stack.enter_context(subprocess.Popen(plan["encode"], stdin=subprocess.PIPE))
        stack.callback(encode.kill)
        eyes = stack.enter_context(ThreadPoolExecutor(max_workers=2))

        def correct_eye(planes, eye_index):
            for channel, data in enumerate(planes):
                width = eye if channel else 2 * eye
                plane = data.reshape(-1, width)
                start = eye_index * (width // 2)
                view = plane[:, start:start + width // 2]
                bands[bool(channel)].apply(view, plan["strength"], ten, channel == 0)

        def correct(frame):
            planes = np.split(frame, (2 * size, 5 * size // 2))
            tasks = [eyes.submit(correct_eye, planes, index) for index in (0, 1)]
            for task in tasks:
                task.result()
            encode.stdin.write(memoryview(frame).cast("B"))

        # The next projection can fill the other buffer during correction and encoding.
        buffers = [np.empty(3 * size, dtype=dtype) for _ in range(2)]
        pending = None
        with ThreadPoolExecutor(max_workers=1) as pool:
            index = 0
            while True:
                frame = buffers[index % 2]
                received = project.stdout.readinto(memoryview(frame).cast("B"))
                if pending is not None:
                    pending.result()
                if not received:
                    break
                if received != frame_bytes:
                    raise ValueError("投影输出包含不完整帧")
                pending = pool.submit(correct, frame)
                index += 1
        encode.stdin.close()
        for process in (project, encode):
            if process.wait():
                raise subprocess.CalledProcessError(process.returncode, process.args)


if __name__ == "__main__":
    cv.setNumThreads(3)
    render(json.loads(Path(sys.argv[1]).read_text(encoding="utf-8")))
