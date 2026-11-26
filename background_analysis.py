"""Plan stage-background reuse from an existing video. No image synthesis is performed."""

import argparse
import csv
import hashlib
import itertools
import json
import os
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path

import av
import cv2
import numpy as np
from filelock import FileLock

import engine as e

VERSION = "bg-plan-2"
SAMPLE_SECONDS = 0.25
NEW, REUSE, REVIEW = "预计新建", "复用", "需复核"
HEADERS = [
    "开始（秒）",
    "结束（秒）",
    "镜头",
    "触发",
    "建议",
    "背景编号",
    "判断依据",
    "建议参考帧（秒）",
]


def file_signature(path):
    stat = path.stat()
    return {"source": str(path), "size": stat.st_size, "mtime": stat.st_mtime_ns}


def report_path(path):
    ident = json.dumps([VERSION, file_signature(path)], sort_keys=True).encode()
    return (
        e.STATE
        / "background-analysis"
        / (hashlib.sha256(ident).hexdigest()[:20] + ".json")
    )


def save_json(path, value):
    with tempfile.NamedTemporaryFile(
        dir=path.parent, delete=False, mode="w", encoding="utf-8"
    ) as file:
        temp = Path(file.name)
    try:
        temp.write_text(
            json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        os.replace(temp, path)
    finally:
        temp.unlink(missing_ok=True)


def shot_boundaries(path, cache, signature):
    if cache.is_file():
        saved = json.loads(cache.read_text(encoding="utf-8"))
        if all(saved.get(k) == v for k, v in signature.items()):
            return set(saved["pts"])
    import torch

    sys.path.insert(0, str(e.NUNIF))
    from nunif.utils.shot_boundary_detection import detect_boundary

    torch.set_num_threads(4)
    torch.hub.set_dir(str(e.MODELS / "iw3/pretrained_models/hub"))
    points = detect_boundary(str(path), device="cpu", max_fps=None)
    save_json(cache, {**signature, "pts": sorted(points)})
    return points


@dataclass
class View:
    time: float
    xy: np.ndarray
    descriptors: np.ndarray
    thumbnail: np.ndarray
    mask: np.ndarray
    histogram: np.ndarray


def describe(image, timestamp, sift):
    h, w = image.shape[:2]
    mask = np.zeros((h, w), np.uint8)
    # Centre-stage movement is usually the performer. Compare separated areas of scenery.
    mask[h // 16 : -max(1, h // 16), w // 35 : round(w * 0.27)] = 255
    mask[h // 16 : -max(1, h // 16), round(w * 0.73) : -max(1, w // 35)] = 255
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    # Ignore black letterboxing, burned-in corner text alone cannot establish a match.
    mask[gray < 10] = 0
    kp, desc = sift.detectAndCompute(gray, mask)
    xy = np.array([p.pt for p in kp], dtype=np.float32).reshape(-1, 2) / 4
    thumb = cv2.resize(image, None, fx=0.25, fy=0.25, interpolation=cv2.INTER_AREA)
    small = cv2.resize(
        mask, (thumb.shape[1], thumb.shape[0]), interpolation=cv2.INTER_NEAREST
    )
    hist = cv2.calcHist(
        [cv2.cvtColor(thumb, cv2.COLOR_BGR2HSV)],
        [0, 1],
        small,
        [12, 4],
        [0, 180, 0, 256],
    )
    return View(timestamp, xy, desc, thumb, small, cv2.normalize(hist, hist).flatten())


@dataclass
class Match:
    inliers: int
    changed: float
    overlap: float
    brightness: float
    matrix: np.ndarray

    @property
    def stable(self):
        return self.changed < 0.32


def match(a, b):
    if min(len(a.xy), len(b.xy)) < 12:
        return None
    pairs = cv2.BFMatcher().knnMatch(a.descriptors, b.descriptors, k=2)
    good = [m for m, n in pairs if m.distance < 0.72 * n.distance]
    if len(good) < 12:
        return None
    x = a.xy[[m.queryIdx for m in good]]
    y = b.xy[[m.trainIdx for m in good]]
    H, flags = cv2.findHomography(x, y, cv2.RANSAC, 0.85, maxIters=600)
    if H is None:
        return None
    h, w = b.mask.shape
    keep = flags.ravel().astype(bool)
    n = int(keep.sum())
    if n < 12 or n / len(good) < 0.48:
        return None
    for pts in (x[keep], y[keep]):
        if (
            min((pts[:, 0] < w * 0.48).sum(), (pts[:, 0] > w * 0.52).sum()) < 3
            or np.ptp(pts[:, 1]) < h * 0.14
        ):
            return None
    corners = np.float32([[[0, 0], [w, 0], [w, h], [0, h]]])
    poly = cv2.perspectiveTransform(corners, H)[0]
    area = cv2.contourArea(poly, oriented=True)
    if not cv2.isContourConvex(poly) or not 0.30 * w * h < area < 3.4 * w * h:
        return None
    support = cv2.warpPerspective(a.mask, H, (w, h)) > 200
    valid = support & (b.mask > 0)
    if valid.sum() < max(32, 0.12 * np.count_nonzero(b.mask)):
        return None
    warped = cv2.warpPerspective(a.thumbnail, H, (w, h))
    diff = (
        cv2.GaussianBlur(b.thumbnail, (3, 3), 0).astype("float32")
        - cv2.GaussianBlur(warped, (3, 3), 0)
    )[valid]
    shift = np.median(diff, axis=0)
    changed = float((np.max(abs(diff - shift), axis=1) > 29).mean())
    overlap = cv2.intersectConvexConvex(corners[0], poly)[0] / (w * h)
    return Match(
        n, changed, float(np.clip(overlap, 0, 1)), float(np.mean(abs(shift))), H
    )


def split_views(views):
    """Confirm a persistent break on both sides. A single flash must not become a new plate."""
    start = 0
    previous = views[0]
    for i in range(1, len(views)):
        current = views[i]
        fit = match(previous, current)
        if fit and fit.stable:
            previous = current
            continue
        if i + 1 == len(views):
            continue
        following = views[i + 1]
        back = match(previous, following)
        if back and back.stable:
            continue
        onward = match(current, following)
        appearance_jump = float(previous.histogram @ current.histogram) < 0.72
        if (
            onward
            and onward.stable
            and (fit is not None or appearance_jump)
            and current.time - previous.time <= 1.25
        ):
            yield start, i
            start = i
        previous = current
    yield start, len(views)


def representative(views):
    usable = [v for v in views if len(v.xy) >= 18]
    if len(usable) <= 3:
        return usable
    return [usable[round((len(usable) - 1) * f)] for f in (0.12, 0.5, 0.88)]


def find_background(probes, library):
    if not probes:
        return None
    # Colour ranks candidates; only a geometric + photometric match can justify reuse.
    ranked = sorted(
        library,
        key=lambda key: max(
            float(p.histogram @ v.histogram) for p in probes for v in library[key]
        ),
        reverse=True,
    )
    winner = None
    for key in ranked[:16]:
        hits = []
        for p in probes:
            fits = [f for ref in library[key] if (f := match(ref, p)) and f.stable]
            if fits:
                hits.append((p, max(fits, key=lambda x: x.inliers)))
        if len(hits) >= min(2, len(probes)):
            score = (len(hits), sum(f.inliers for _, f in hits))
            if winner is None or score > winner[0]:
                winner = (score, key, hits)
    return winner


def classify(views, shot, trigger, start, end, library):
    probes = representative(views)
    found = find_background(probes, library)
    key = None
    anchor = None
    if found:
        _, key, hits = found
        anchor, fit = max(hits, key=lambda item: item[1].inliers)
        action = REUSE
        reason = f"与 {key} 的外围布景对齐（{fit.inliers} 个内点）"
        if fit.brightness > 18:
            reason += "；亮度或色调有变化"
        if fit.overlap < 0.75:
            reason += "；新视野较大，需检查外围补画"
        refs = [p for p, _ in hits]
    elif end - start < 0.8 or len(probes) < 2:
        action = REVIEW
        reason = "短镜头或外围纹理不足，无法确认背景归属"
    else:
        strength = {}
        stable_count = 0
        for a, b in itertools.pairwise(views):
            fit = match(a, b)
            if fit and fit.stable:
                stable_count += 1
                for v in (a, b):
                    strength[v.time] = strength.get(v.time, 0) + fit.inliers * (
                        1 - fit.changed
                    )
        if stable_count < max(1, (len(views) - 1) * 0.45):
            action = REVIEW
            reason = "连续采样对齐不足，可能有遮挡、快速运镜或动态布景"
        else:
            action = NEW
            key = f"B{len(library) + 1:02d}"
            reason = (
                "首次取得稳定布景"
                if not library
                else "已有背景视角未可靠匹配，可能是同场换机位"
            )
            anchor = max(views, key=lambda v: strength.get(v.time, 0))
            refs = representative([v for v in views if v.time in strength])
            if all(v is not anchor for v in refs):
                refs.append(anchor)
    if key:
        library[key] = library.get(key, []) + refs
        if len(library[key]) > 6:
            library[key] = library[key][:3] + library[key][-3:]
    return [
        round(start, 3),
        round(end, 3),
        shot,
        trigger,
        action,
        key or "—",
        reason,
        round(anchor.time, 3) if anchor is not None else None,
    ]


def camera_coverage(views, start, stop):
    """Split a continuous pan once two overlapping samples have left the anchor view."""
    anchor = start
    for i in range(start + 2, stop - 1):
        if views[i].time - views[anchor].time < 1:
            continue
        direct = match(views[anchor], views[i])
        if direct and direct.stable and direct.overlap > 0.6:
            continue
        next_fit = match(views[anchor], views[i + 1])
        if next_fit and next_fit.stable and next_fit.overlap > 0.6:
            continue
        adjacent = match(views[i], views[i + 1])
        if adjacent and adjacent.stable:
            yield anchor, i
            anchor = i
    yield anchor, stop


def timeline(path, points, *, track_coverage=False):
    sift = cv2.SIFT_create(nfeatures=420, contrastThreshold=0.022)
    library = {}
    rows = []

    def finish(views, start, end, shot):
        intervals = split_views(views)
        if track_coverage:
            intervals = itertools.chain.from_iterable(
                camera_coverage(views, a, b) for a, b in intervals
            )
        for a, b in intervals:
            label = "片头" if shot == 1 else "切镜"
            rows.append(
                classify(
                    views[a:b],
                    shot,
                    label if a == 0 else "镜内持续变化",
                    start if a == 0 else views[a].time,
                    views[b].time if b < len(views) else end,
                    library,
                )
            )

    with av.open(str(path)) as container:
        stream = container.streams.video[0]
        stream.thread_type = "AUTO"
        first = None
        prev_cut = False
        views = []
        shot = 1
        start = 0.0
        for frame in container.decode(stream):
            if frame.pts is None:
                continue
            timestamp = float(frame.pts * frame.time_base)
            if first is None:
                first = timestamp
            t = timestamp - first
            cut = frame.pts in points
            if prev_cut and not cut and views:
                finish(views, start, t, shot)
                shot += 1
                start = t
                views = []
            prev_cut = cut
            if not views or t >= views[-1].time + SAMPLE_SECONDS - 0.001:
                # Integer multiples of four keep feature coordinates consistent with thumbnails.
                width = 480
                height = max(8, round(width * frame.height / frame.width / 4) * 4)
                image = frame.reformat(
                    width=width, height=height, format="bgr24"
                ).to_ndarray()
                views.append(describe(image, t, sift))
            frame_duration = float((frame.duration or 0) * frame.time_base)
            end = t + (frame_duration or 1 / float(stream.average_rate or 30))
        if not views:
            raise ValueError("未解码到带时间戳的视频帧")
        finish(views, start, end, shot)
    return rows


def analyze(path):
    path = path.resolve(strict=True)
    signature = file_signature(path)
    report = report_path(path)
    report.parent.mkdir(exist_ok=True, parents=True)
    with FileLock(str(report) + ".lock"):
        if not report.exists():
            cv2.setNumThreads(2)
            pts = shot_boundaries(path, report.with_suffix(".cuts.json"), signature)
            rows = timeline(path, pts)
            if file_signature(path) != signature:
                raise ValueError("分析期间源文件已变化，请重新分析")
            save_json(
                report,
                {
                    "version": VERSION,
                    **signature,
                    "headers": HEADERS,
                    "rows": rows,
                    "shots": rows[-1][2],
                    "new": sum(row[4] == NEW for row in rows),
                    "reuse": sum(row[4] == REUSE for row in rows),
                    "review": sum(row[4] == REVIEW for row in rows),
                },
            )
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="分析完整视频的镜头和背景复用时间线")
    parser.add_argument("source", help="素材路径")
    parser.add_argument("--csv", type=Path, help="导出 UTF-8 时间线")
    args = parser.parse_args()
    report = analyze(e.source(args.source))
    if args.csv:
        data = json.loads(report.read_text(encoding="utf-8"))
        with args.csv.open("x", encoding="utf-8-sig", newline="") as file:
            writer = csv.writer(file)
            writer.writerow(HEADERS)
            writer.writerows(data["rows"])
    print(report)
