"""Automatic stage planning, reference selection and camera-aware peripheral rendering."""

import hashlib
import itertools
import json
import math
import os
import subprocess
import sys
from collections import OrderedDict
from contextlib import ExitStack
from pathlib import Path

import av
import cv2
import numpy as np

import background_analysis as analysis
import engine
import stage_atlas
import stage_model as model


def small_view(image, time, sift):
    h, w = image.shape[:2]
    small = cv2.resize(
        image, (960, max(8, round(240 * h / w) * 4)), interpolation=cv2.INTER_AREA
    )
    return analysis.describe(small, time, sift)


def estimate_roll(image):
    # Horizontal and vertical architectural lines vote for the same camera roll modulo 90°.
    _h, w = image.shape[:2]
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 55, 140)
    edges[:, int(0.32 * w) : int(0.68 * w)] = 0
    lines = cv2.HoughLinesP(
        edges, 1, np.pi / 180, 22, minLineLength=max(18, w // 20), maxLineGap=5
    )
    if lines is None or len(lines) < 5:
        return 0.0
    x, y, u, v = lines.reshape(-1, 4).T
    lengths = np.hypot(v - y, u - x)
    angles = (np.degrees(np.arctan2(v - y, u - x)) + 45) % 90 - 45
    votes = np.exp(4j * np.radians(angles)) @ lengths / lengths.sum()
    return float(np.angle(votes) / 4) if abs(votes) > 0.63 else 0.0


def frame_bounds(start, end, rate):
    first = max(0, math.ceil(start * rate - 0.04))
    return first, max(first + 1, math.ceil(end * rate - 0.04))


def flow_to_previous(current, previous):
    """Track short motion between neighbouring samples when descriptor matching is ambiguous."""
    h, w = current.shape
    mask = np.zeros_like(current)
    mask[h // 12 : -h // 12, w // 30 : int(0.27 * w)] = 255
    mask[h // 12 : -h // 12, int(0.73 * w) : -w // 30] = 255
    pts = cv2.goodFeaturesToTrack(current, 240, 0.012, 5, mask=mask)
    if pts is None or len(pts) < 18:
        return None
    options = {"winSize": (21, 21), "maxLevel": 3}
    end, good, _ = cv2.calcOpticalFlowPyrLK(current, previous, pts, None, **options)
    if end is None:
        return None
    back, valid, _ = cv2.calcOpticalFlowPyrLK(previous, current, end, None, **options)
    if back is None:
        return None
    reliable = (
        (good.ravel() > 0)
        & (valid.ravel() > 0)
        & (np.linalg.norm((back - pts)[:, 0], axis=1) < 0.75)
    )
    x, y = pts[reliable, 0], end[reliable, 0]
    if len(x) < 16:
        return None
    m, inliers = cv2.findHomography(x, y, cv2.RANSAC, 1.3)
    if m is None:
        return None
    sel = inliers.ravel().astype(bool)
    support = x[sel]
    if (
        sel.mean() < 0.65
        or min((support[:, 0] < 0.3 * w).sum(), (support[:, 0] > 0.7 * w).sum()) < 5
        or np.ptp(support[:, 1]) < 0.2 * h
    ):
        return None
    corners = np.float32([[[0, 0], [w, 0], [w, h], [0, h]]])
    poly = cv2.perspectiveTransform(corners, m)[0]
    area = cv2.contourArea(poly, oriented=True) / (w * h)
    if not cv2.isContourConvex(poly) or not 0.65 < area < 1.55:
        return None
    # Both side strips must agree after the warp, including a robust per-frame exposure shift.
    warped = cv2.warpPerspective(current, m, (w, h))
    overlap = (cv2.warpPerspective(mask, m, (w, h)) > 250) & (mask > 0)
    for side in (slice(0, w // 2), slice(w // 2, w)):
        values = (warped[:, side].astype("float32") - previous[:, side])[
            overlap[:, side]
        ]
        if (
            len(values) < 0.02 * w * h
            or np.percentile(abs(values - np.median(values)), 75) > 24
        ):
            return None
    return m


def registered_frames(capture, start, end, rate, anchor, sift, extra, *, dense=False):
    """Prefer a direct fit. Bridge temporary occlusions via a short chain of adjacent fits."""
    first, stop = frame_bounds(start, end, rate)
    step = 1 if dense else max(1, round(rate / 15))
    probes = sorted(
        {
            first,
            stop - 1,
            *range(first, stop, step),
            *(i for i in extra if first <= i < stop),
        }
    )
    capture.set(cv2.CAP_PROP_POS_FRAMES, first)
    last = first - 1
    records = []
    previous = prev_gray = None
    for i in probes:
        for _ in range(i - last):
            if not capture.grab():
                return records
        last = i
        ok, image = capture.retrieve()
        if not ok:
            raise OSError(f"无法解码跟踪帧 {i}")
        view = small_view(image, i / rate, sift)
        direct = analysis.match(view, anchor)
        matrix = direct.matrix if direct and direct.stable else None
        if abs(view.time - anchor.time) < 0.0001:
            matrix = np.eye(3)
        gray = None
        if dense:
            rel_h, rel_w = view.mask.shape
            gray = cv2.cvtColor(
                cv2.resize(image, (2 * rel_w, 2 * rel_h), interpolation=cv2.INTER_AREA),
                cv2.COLOR_BGR2GRAY,
            )
        relative = None
        if previous is not None and (matrix is None or records[-1][1] is None):
            fit = analysis.match(view, previous)
            if fit and fit.stable:
                relative = fit.matrix
            elif dense:
                motion = flow_to_previous(gray, prev_gray)
                if motion is not None:
                    scale = np.diag([2.0, 2.0, 1.0])
                    relative = np.diag([0.5, 0.5, 1.0]) @ motion @ scale
        records.append([i, matrix, relative])
        previous, prev_gray = view, gray
    # Limit drift on long unmatched runs. No chains across segment boundaries.
    direct_indices = {i for i, m, _ in records if m is not None}
    for direction in (1, -1):
        last_direct = None
        indices = (
            range(len(records)) if direction == 1 else reversed(range(len(records)))
        )
        for k in indices:
            i, m, _ = records[k]
            if m is not None:
                if direction == 1 or i in direct_indices:
                    last_direct = i
                continue
            if last_direct is None or abs(i - last_direct) > rate * 0.5:
                continue
            prev = k - direction
            link = records[k][2] if direction == 1 else records[prev][2]
            if link is not None and records[prev][1] is not None:
                records[k][1] = records[prev][1] @ (
                    link if direction == 1 else np.linalg.inv(link)
                )
    return records


def reference_image(
    capture, frame, view, start, end, rate, angle, roll, sift, work, index, predictor
):
    h, w = frame.shape[:2]
    ox, oy = w // 3, h // 3
    aw, ah = w + 2 * ox, h + 2 * oy
    total = np.zeros((ah, aw, 3), np.float32)
    den = np.zeros((ah, aw), np.float32)
    step = max(1, round(rate * 4 / 30))
    first, stop = frame_bounds(
        max(start, view.time - 1.4), min(end, view.time + 1.4), rate
    )
    selected = set(range(first, stop, step)) | {index, stop - 1}
    records = registered_frames(
        capture, first / rate, stop / rate, rate, view, sift, selected
    )
    scale = np.diag([w / view.mask.shape[1], h / view.mask.shape[0], 1])
    inv = np.linalg.inv(scale)
    offset = np.eye(3)
    offset[:2, 2] = [ox, oy]
    # These measurements are in increasing frame order. Seek once instead of
    # decoding the same GOP from its keyframe for each observation.
    capture.set(cv2.CAP_PROP_POS_FRAMES, first)
    next_frame = first
    for i, m, _ in records:
        if i not in selected or m is None:
            continue
        while next_frame < i:
            if not capture.grab():
                raise OSError(f"无法解码舞台参考帧 {next_frame}")
            next_frame += 1
        ok, image = capture.read()
        if not ok:
            raise OSError(f"无法解码舞台参考帧 {i}")
        next_frame = i + 1
        weights = stage_atlas.scenery_mask(image, predictor)
        transform = offset @ scale @ m @ inv
        den += cv2.warpPerspective(weights, transform, (aw, ah))
        total += cv2.warpPerspective(image * weights[:, :, None], transform, (aw, ah))
    atlas = (total / np.maximum(den[:, :, None], 1e-5)).clip(0, 255).astype("uint8")
    mask = (255 * np.minimum(den, 1)).astype("uint8")
    ref = stage_atlas.project(atlas, mask, w, angle, roll, 1024)
    for name, img in [
        (f"ref-{index}.png", ref),
        (f"atlas-{index}.png", atlas),
        (f"support-{index}.png", mask),
    ]:
        if not cv2.imwrite(str(work / name), img):
            raise OSError(f"无法保存 {name}")


AMBIENT = "稳定环境光"


def stable_track(mats, first, stop, rate):
    """Only bridge isolated gaps within a shot that has reliable camera coverage."""
    if not mats or mats[0][0] != first or mats[-1][0] != stop - 1:
        return []
    good = [(i, m) for i, m in mats if m is not None and np.isfinite(m).all()]
    if len(good) < 0.9 * len(mats):
        return []
    samples = {i for i, _ in good}
    missing = [
        list(run)
        for valid, run in itertools.groupby(mats, key=lambda v: v[0] in samples)
        if not valid
    ]
    if any(run[-1][0] - run[0][0] + 1 > max(1, rate * 0.15) for run in missing):
        return []
    # Rendering interpolates between valid samples and holds the nearest at each end.
    return good


def shot_palette(capture, start, end, rate):
    """Estimate one dim, low-frequency surround from the edges of the complete shot."""
    first, stop = frame_bounds(start, end, rate)
    colors = []
    for i in np.unique(np.linspace(first, stop - 1, 7).round().astype(int)):
        capture.set(cv2.CAP_PROP_POS_FRAMES, int(i))
        ok, frame = capture.read()
        if not ok:
            raise OSError(f"无法读取环境色参考帧 {i}")
        small = cv2.resize(frame, (48, 24), interpolation=cv2.INTER_AREA)
        # Medians suppress moving foreground and one-frame lighting flashes.
        colors.append(
            [
                [
                    np.median(band[:, :6], axis=(0, 1)),
                    np.median(band[:, -6:], axis=(0, 1)),
                ]
                for band in np.array_split(small, 4)
            ]
        )
    color = np.median(colors, axis=0)
    base = np.median(color, axis=(0, 1))
    muted = 0.5 * color + 0.5 * base
    luminance = muted @ np.array([0.114, 0.587, 0.299])
    muted = 0.55 * muted + 0.45 * luminance[:, :, None]
    return np.rint(np.clip((muted - 8) * 0.3, 0, 64)).astype("uint8").tolist()


def plan_segment(capture, row, rate, sift, library, stored, *, reference_only=False):
    start, end, _, _, action, name, reason, timestamp = row
    time = timestamp if timestamp is not None else (start + end) / 2
    first, stop = frame_bounds(start, end, rate)
    index = max(first, min(round(time * rate), stop - 1))
    capture.set(cv2.CAP_PROP_POS_FRAMES, index)
    ok, frame = capture.read()
    if not ok:
        raise ValueError(f"无法读取 {time:.2f}s 参考帧")
    view = small_view(frame, index / rate, sift)
    plate = None
    choice = "生成"
    roll = 0.0
    ranked = sorted(
        (k for k in library if view.mask.shape == library[k].mask.shape),
        key=lambda k: (name == k, float(view.histogram @ library[k].histogram)),
        reverse=True,
    )
    for k in ranked[:16]:
        fit = analysis.match(view, library[k])
        if (
            not fit
            or fit.changed > 0.18
            or fit.inliers < 22
            or fit.overlap < 0.68
            or fit.brightness > 20
        ):
            continue
        # Require independent evidence before borrowing a panorama from another view.
        check = max({first, stop - 1}, key=lambda i: abs(i - index))
        check = index + int(
            np.clip(
                check - index,
                -max(1, round(rate * 0.22)),
                max(1, round(rate * 0.22)),
            )
        )
        capture.set(cv2.CAP_PROP_POS_FRAMES, check)
        ok, neighbor = capture.read()
        verify = (
            analysis.match(small_view(neighbor, check / rate, sift), library[k])
            if ok and check != index
            else None
        )
        if (
            not verify
            or not verify.stable
            or verify.inliers < 18
            or verify.brightness > 22
        ):
            continue
        plate, choice = k, "复用"
        reason = f"两帧布景配准，{fit.inliers}/{verify.inliers} 个内点"
        if k in stored:
            roll = stored[k][0]["roll"]
            reason += " · " + stored[k][0]["label"]
        break
    if plate is None:
        if reference_only:
            choice = AMBIENT
            reason = "当前机位没有两帧可确认的空舞台匹配"
        elif (
            end - start < 0.65
            or np.percentile(view.thumbnail, 95) < 17
            or (action == analysis.REVIEW and len(view.xy) < 18)
        ):
            choice = AMBIENT
            reason = "短暂转场、暗场或外围特征不足"
        else:
            plate = name if name != "—" and name not in library else f"V{index}"
            library[plate] = view
            roll = estimate_roll(cv2.resize(frame, (480, 2 * view.mask.shape[0])))
            reason = f"自动选取 {index / rate:.2f}s 的两侧布景"
    return {
        "start": start,
        "end": end,
        "time": index / rate,
        "ref": index,
        "plate": plate,
        "action": choice,
        "roll": roll,
        "width": frame.shape[1],
        "reason": reason,
    }


def plan(clip, rows, rate, angle, sift, work, cache, eye, *, allowed_stages=None):
    capture = cv2.VideoCapture(str(clip))
    segments, library, stored = [], {}, {}
    predictor = None
    tracks = {}
    try:
        for info in sorted((engine.MODELS / "stage-library").glob("*/info.json")):
            if allowed_stages is not None and info.parent.name not in allowed_stages:
                continue
            meta = json.loads(info.read_text(encoding="utf8"))
            if meta["angle"] != angle:
                continue
            anchor = cv2.imread(str(info.parent / "anchor.png"))
            if anchor is None or not (info.parent / "panorama.png").is_file():
                continue
            key = "R-" + info.parent.name
            library[key] = small_view(anchor, -1, sift)
            stored[key] = (meta, info.parent / "panorama.png")
        for _, group in itertools.groupby(rows, key=lambda row: row[2]):
            shot_library = library.copy()
            shot = [
                plan_segment(
                    capture, row, rate, sift, shot_library, stored,
                    reference_only=allowed_stages is not None,
                )
                for row in group
            ]
            shot_tracks = {}
            for seg in shot:
                if seg["plate"] is None:
                    continue
                mats = track(clip, seg, rate, shot_library[seg["plate"]], angle, sift)
                first, stop = frame_bounds(seg["start"], seg["end"], rate)
                reliable = stable_track(mats, first, stop, rate)
                seg["tracked"] = sum(m is not None for _, m in mats)
                seg["samples"] = len(mats)
                if reliable:
                    shot_tracks[seg["ref"]] = reliable
                else:
                    seg["reason"] = (
                        f"舞台配准 {seg['tracked']}/{len(mats)}，覆盖不足或连续失配"
                    )
            if len(shot_tracks) != len(shot):
                palette = shot_palette(capture, shot[0]["start"], shot[-1]["end"], rate)
                for seg in shot:
                    if seg["ref"] in shot_tracks:
                        seg["reason"] = "同镜头其他区间缺少稳定布景"
                    seg.update(action=AMBIENT, plate=None, palette=palette)
            else:
                library = shot_library
                tracks.update(shot_tracks)
                for seg in shot:
                    index = seg["ref"]
                    if (
                        seg["action"] != "生成"
                        or (cache / f"{index}-stage-{eye}.png").is_file()
                    ):
                        continue
                    if predictor is None:
                        predictor = stage_atlas.foreground_predictor(
                            engine.MODELS / "stage-weights/sam_vit_b_01ec64.pth"
                        )
                    capture.set(cv2.CAP_PROP_POS_FRAMES, index)
                    ok, frame = capture.read()
                    if not ok:
                        raise OSError(f"无法读取舞台参考帧 {index}")
                    reference_image(
                        capture,
                        frame,
                        library[seg["plate"]],
                        seg["start"],
                        seg["end"],
                        rate,
                        angle,
                        seg["roll"],
                        sift,
                        work,
                        index,
                        predictor,
                    )
            for seg in shot:
                print(
                    f"{seg['start']:.2f}–{seg['end']:.2f}s　{seg['action']}　{seg['reason']}",
                    flush=True,
                )
            segments.extend(shot)
    finally:
        capture.release()
        if predictor is not None:
            import torch

            del predictor
            torch.cuda.empty_cache()
    return segments, tracks, {k: path for k, (_, path) in stored.items()}


def match_colors(generated, reference, dest):
    synth, ref = cv2.imread(str(generated)), cv2.imread(str(reference))
    mask = np.any(ref != (0, 255, 0), axis=2).astype("uint8")
    known = cv2.erode(mask, np.ones((3, 3), np.uint8)) > 0
    if known.sum() < 32:
        os.replace(generated, dest)
        return
    delta = np.clip(ref.astype("f") - synth, -64, 64)
    base = np.clip(np.median(delta[known], axis=0), -36, 36)
    density = cv2.GaussianBlur(known.astype("f"), (0, 0), 20)
    shift = cv2.GaussianBlur(
        (delta - base) * known[:, :, None], (0, 0), 20
    ) / np.maximum(density[:, :, None], 0.001)
    corrected = np.clip(
        synth + base + np.clip(shift, -35, 35) * np.minimum(1, density[:, :, None] * 6),
        0,
        255,
    )
    matched = np.rint(corrected).astype("uint8")
    if not cv2.imwrite(str(generated), matched):
        raise OSError("无法保存舞台匹配结果")
    os.replace(generated, dest)


def generate_plates(segments, cache, work, eye, angle, files):
    upscale = []
    with ExitStack() as stack:
        instance = None
        for seg in segments:
            if seg["action"] != "生成":
                continue
            index = seg["ref"]
            finished = cache / f"{index}-stage-{eye}.png"
            files[seg["plate"]] = finished
            if finished.is_file():
                continue
            ref = work / f"ref-{index}.png"
            raw = cache / f"{index}.png"
            if not raw.exists():
                if instance is None:
                    instance = stack.enter_context(model.server(work))
                print(f"生成背景 {seg['plate']}，参考帧 {seg['time']:.2f}s", flush=True)
                seed = int(hashlib.sha256(ref.read_bytes()).hexdigest()[:8], 16)
                match_colors(model.generate(instance, ref, seed), ref, raw)
            else:
                print(f"读取已生成的背景 {seg['plate']}", flush=True)
            if eye > 1024:
                path = cache / f"{index}-{eye}.png"
                if not path.exists():
                    upscale.append((str(raw), str(work / path.name), eye))

    if upscale:
        print(f"分块超分 {len(upscale)} 张舞台背景", flush=True)
        manifest = work / "upscale.json"
        manifest.write_text(json.dumps(upscale), encoding="utf8")
        subprocess.run(
            [model.PYTHON, str(Path(model.__file__)), str(manifest)], check=True
        )
        for _, dest, _ in upscale:
            os.replace(dest, cache / Path(dest).name)
    for seg in segments:
        if seg["action"] != "生成" or files[seg["plate"]].is_file():
            continue
        index = seg["ref"]
        raw = cache / (f"{index}-{eye}.png" if eye > 1024 else f"{index}.png")
        world = cv2.imread(str(raw))
        atlas = cv2.imread(str(work / f"atlas-{index}.png"))
        mask = cv2.imread(str(work / f"support-{index}.png"), 0)
        stage_atlas.bake(world, atlas, mask, seg["width"], angle, seg["roll"])
        temp = work / f"stage-{index}.png"
        if not cv2.imwrite(str(temp), world):
            raise OSError("无法保存舞台背景")
        os.replace(temp, files[seg["plate"]])
    return files


def ray_grid(n, angle, curvature, keep):
    # Continue the same screen geometry used by vr180.curvature_maps into the periphery.
    a = math.radians(angle / 2)
    tan = math.tan(a)
    t = ((np.arange(n, dtype="f") + 0.5) / n - 0.5) * np.pi
    theta = t
    scale = 1
    if curvature:
        fraction = curvature / 100
        arc = tan * fraction if keep else a * fraction
        radius = (
            1 / fraction
            if keep
            else math.sin(a) / (2 * math.sin(arc / 2) * math.cos(a - arc / 2))
        )
        edge = math.atan2(
            radius * math.sin(arc), 1 - 2 * radius * math.sin(arc / 2) ** 2
        )
        az = np.clip(t, -edge, edge)
        phi = az + np.arcsin((1 / radius - 1) * np.sin(az))
        z = (1 - 2 * radius * np.sin(phi / 2) ** 2) / np.cos(az)
        blend = np.clip((abs(t) - edge) / (np.pi / 2 - edge), 0, 1)
        theta = np.where(
            abs(t) <= edge,
            np.arctan(phi / arc * tan),
            np.sign(t) * (a + (np.pi / 2 - a) * blend),
        )
        scale = (1 - blend) * np.cos(theta) * z * tan / (radius * arc) + blend
    elev = t[:, None]
    return np.stack(
        np.broadcast_arrays(
            np.cos(elev) * np.sin(theta),
            np.sin(elev) * scale,
            np.cos(elev) * np.cos(theta),
        ),
        axis=-1,
    )


def track(clip, seg, rate, anchor, angle, sift):
    capture = cv2.VideoCapture(str(clip))
    try:
        args = (capture, seg["start"], seg["end"], rate, anchor, sift, [seg["ref"]])
        records = registered_frames(*args)
        if any(m is None for _, m, _ in records):
            # Recheck difficult footage at its full frame rate before falling back.
            records = registered_frames(*args, dense=True)
        print(
            f"舞台配准 {seg['start']:.2f}–{seg['end']:.2f}s：{sum(m is not None for _, m, _ in records)}/{len(records)} 个跟踪点",
            flush=True,
        )
    finally:
        capture.release()
    h, w = anchor.mask.shape
    focal = w / 2 / math.tan(math.radians(angle / 2))
    k = np.array([[focal, 0, (w - 1) / 2], [0, focal, (h - 1) / 2], [0, 0, 1]])
    inv = np.linalg.inv(k)
    mats = []
    for i, m, _ in records:
        if m is not None:
            m = inv @ m @ k
            m = (m / m[2, 2]).astype("f")
        mats.append((i, m))
    return mats


def render(clip, work, segments, tracks, files, rate, eye, angle, curve, keep):
    rate_num = float(rate)
    size = eye
    rays = ray_grid(min(384, size), angle, curve, keep)
    space = engine.video(engine.probe(clip)).get("color_space")
    matrix = {
        "bt470bg": "bt601",
        "smpte170m": "bt601",
        "bt2020nc": "bt2020",
        "smpte240m": "smpte240m",
        "fcc": "fcc",
    }.get(space, "bt709")
    cmd = [
        "ffmpeg",
        "-v",
        "error",
        "-y",
        "-f",
        "rawvideo",
        "-pixel_format",
        "bgr24",
        "-video_size",
        f"{size}x{size}",
        "-framerate",
        str(rate),
        "-i",
        "pipe:0",
        "-vf",
        f"scale=out_color_matrix={matrix}:out_range=tv,format=yuv420p",
        "-colorspace",
        space
        if space in ("bt709", "smpte170m", "bt470bg", "bt2020nc", "smpte240m", "fcc")
        else "bt709",
        "-color_range",
        "tv",
        "-an",
        "-c:v",
        "hevc_nvenc",
        "-preset",
        "p4",
        "-cq",
        "15",
        "-b:v",
        "0",
        "-pix_fmt",
        "yuv420p",
        str(work / "stage.mkv"),
    ]
    proc = subprocess.Popen(cmd, stdin=subprocess.PIPE)
    current_palette = None
    last_seg = -1
    rolls = {
        s["plate"]: s["roll"]
        for s in segments
        if s["action"] == "生成" or (s["plate"] or "").startswith("R-")
    }
    idx = 0
    worlds = OrderedDict()
    try:
        with av.open(str(clip)) as container:
            for i, frame in enumerate(container.decode(video=0)):
                while (
                    idx + 1 < len(segments)
                    and (i + 0.04) / rate_num >= segments[idx + 1]["start"]
                ):
                    idx += 1
                seg = segments[idx]
                if idx != last_seg:
                    plate = seg["plate"]
                    world = None
                    previous_palette = current_palette
                    current_palette = None
                    fade_start = i
                    if plate:
                        if plate not in worlds:
                            if len(worlds) >= 2:
                                worlds.popitem(last=False)
                            worlds[plate] = cv2.imread(str(files[plate]))
                        worlds.move_to_end(plate)
                        world = worlds[plate]
                        if world is None:
                            raise ValueError("舞台缓存图片不可读")
                        mats = tracks[seg["ref"]]
                        pos = 0
                    else:
                        target_palette = np.array(seg["palette"], dtype="float32")
                        steady_image = None
                    last_seg = idx
                if world is None:
                    current_palette = target_palette
                    mix = (i - fade_start + 1) / max(1, rate_num * 0.30)
                    if mix < 1 and previous_palette is not None and target_palette.any():
                        current_palette = (
                            1 - mix
                        ) * previous_palette + mix * target_palette
                        img = cv2.resize(current_palette.round().astype("uint8"), (size, size))
                    else:
                        if steady_image is None:
                            steady_image = cv2.resize(target_palette.round().astype("uint8"), (size, size))
                        img = steady_image
                else:
                    while pos + 1 < len(mats) and i >= mats[pos + 1][0]:
                        pos += 1
                    ia, a = mats[pos]
                    ib, b = mats[min(pos + 1, len(mats) - 1)]
                    frac = np.clip((i - ia) / max(1, ib - ia), 0, 1)
                    t = a * (1 - frac) + b * frac
                    u, v, z = (rays @ t.T).transpose(2, 0, 1)
                    # Unwrap longitude before interpolating across the ERP seam.
                    cr, sr = math.cos(rolls[plate]), math.sin(rolls[plate])
                    u, v = cr * u + sr * v, -sr * u + cr * v
                    lon = np.unwrap(np.arctan2(u, z), axis=1)
                    lat = np.arctan2(v, np.hypot(u, z))
                    wh, ww = world.shape[:2]
                    mx = cv2.resize(
                        ((lon / (2 * np.pi) + 0.5) * ww - 0.5).astype("f"),
                        (size, size),
                    )
                    my = cv2.resize(
                        np.clip((lat / np.pi + 0.5) * wh - 0.5, 0, wh - 1).astype("f"),
                        (size, size),
                    )
                    img = cv2.remap(
                        world, mx, my, cv2.INTER_LINEAR, borderMode=cv2.BORDER_WRAP
                    )
                    thumb = frame.reformat(width=32, height=24, format="bgr24").to_ndarray()
                    exposure = min(1, np.percentile(thumb, 90) / 18)
                    if exposure < 1:
                        img = cv2.convertScaleAbs(img, alpha=exposure)
                proc.stdin.write(memoryview(img).cast("B"))
                if i % max(1, round(rate_num)) == 0:
                    print(f"舞台跟踪 frame={i + 1}", flush=True)
        proc.stdin.close()
        if proc.wait():
            raise RuntimeError("舞台背景视频编码失败")
    finally:
        if proc.poll() is None:
            proc.kill()
            proc.wait()


def main(work, key, eye, angle, rate):
    from fractions import Fraction

    rate = Fraction(rate)
    clip = work / "clip.mkv"
    with engine.db() as db:
        spec = json.loads(
            db.execute("SELECT spec FROM jobs WHERE id=?", (key,)).fetchone()[0]
        )
    # Keep measured and generated plates across tracking-only revisions.
    ident = ["atlas-v3", "klein-stage-3.1"] + [
        spec[k]
        for k in (
            "rel",
            "mtime",
            "size",
            "start",
            "length",
            "width",
            "mode",
        )
    ]
    cache = (
        engine.STATE
        / "stage-cache"
        / hashlib.sha256(json.dumps(ident).encode()).hexdigest()[:24]
    )
    cache.mkdir(parents=True, exist_ok=True)
    timeline = cache / "timeline.json"
    cv2.setNumThreads(3)
    if not timeline.exists():
        print("检测镜头与持续的布景变化", flush=True)
        points = analysis.shot_boundaries(
            clip, work / "cuts.json", analysis.file_signature(clip)
        )
        analysis.save_json(
            timeline, analysis.timeline(clip, points, track_coverage=True)
        )
    sift = cv2.SIFT_create(nfeatures=1600, contrastThreshold=0.017)
    allowed = None
    if spec.get("stage_version") == engine.REFERENCE_STAGE:
        catalog = engine.empty_stages(angle)
        allowed = {k for k, v in spec["empty_stages"].items() if catalog.get(k) == v}
    segments, tracks, files = plan(
        clip,
        json.loads(timeline.read_text(encoding="utf8")),
        float(rate),
        angle,
        sift,
        work,
        cache,
        eye,
        allowed_stages=allowed,
    )
    analysis.save_json(engine.STATE / "logs" / f"{key}.stage.json", segments)
    if allowed is None:
        files = generate_plates(segments, cache, work, eye, angle, files)
    render(
        clip,
        work,
        segments,
        tracks,
        files,
        rate,
        eye,
        angle,
        spec.get("curvature", 0),
        spec.get("keep_center", False),
    )


if __name__ == "__main__":
    main(
        Path(sys.argv[1]), sys.argv[2], int(sys.argv[3]), int(sys.argv[4]), sys.argv[5]
    )
