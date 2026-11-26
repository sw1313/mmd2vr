"""Project real stereo observations and add bounded detail outside their support."""

from collections import deque
from concurrent.futures import ThreadPoolExecutor
from contextlib import ExitStack

import cv2 as cv
import numpy as np

import engine
from stage_model import UPSCALER
from vr360_observed import align_one, W, H, pad


def detail_frames(cap, first, stop, rate, net, gate, grid, detail_fps=10):
    from vr360_detail import detail
    from stage_vr360 import read

    stride = max(1, round(rate / detail_fps))
    prev = last_gray = None
    last = first
    for index in range(first, stop):
        if (index - first) % stride and index != stop - 1:
            if not cap.grab():
                raise OSError("细节视频提前结束")
            continue
        frame = read(cap)
        now = detail(frame, net, gate)
        gray = cv.resize(
            cv.cvtColor(frame, cv.COLOR_BGR2GRAY),
            (320, 320),
            interpolation=cv.INTER_AREA,
        )
        if prev is not None and index - last > 1:
            flow = cv.DISOpticalFlow_create(cv.DISOPTICAL_FLOW_PRESET_FAST)
            fw = cv.resize(flow.calc(last_gray, gray, None), (1280, 1280)) * 4
            bw = cv.resize(flow.calc(gray, last_gray, None), (1280, 1280)) * 4
            for offset in range(1, index - last):
                amt = offset / (index - last)
                left = cv.remap(
                    prev,
                    grid - amt * fw,
                    None,
                    cv.INTER_LINEAR,
                    borderMode=cv.BORDER_REPLICATE,
                )
                right = cv.remap(
                    now,
                    grid - (1 - amt) * bw,
                    None,
                    cv.INTER_LINEAR,
                    borderMode=cv.BORDER_REPLICATE,
                )
                yield cv.addWeighted(left, 1 - amt, right, amt, 0)
        yield now
        prev, last_gray, last = now, gray, index


def observations(pool, cap, first, stop, radius, count, source_size):
    from stage_vr360 import read

    donors = {}
    sources = {}
    loaded = first
    pending = deque()
    for index in range(first, stop):
        last = min(stop - 1, index + radius)
        while loaded <= last:
            original = read(cap)
            sources[loaded] = [cv.resize(eye, source_size, interpolation=cv.INTER_AREA)
                               for eye in np.hsplit(original, 2)]
            small = cv.resize(original, (2 * W, H), interpolation=cv.INTER_AREA)
            eyes = np.hsplit(small, 2)
            gray = np.hsplit(cv.cvtColor(small, cv.COLOR_BGR2GRAY), 2)
            donors[loaded] = (eyes, gray)
            loaded += 1
        donors.pop(index - radius - 1, None)
        ids = np.unique(np.rint(np.linspace(max(first, index - radius), last, count)).astype(int))
        target, gray_target = donors[index]
        # Each future owns references to its donors until both eyes are matched.
        pending.append((sources.pop(index), [
            pool.submit(align_one, target[eye], [donors[k][0][eye] for k in ids],
                        index, ids, gray_target[eye], [donors[k][1][eye] for k in ids])
            for eye in range(2)
        ]))
        if len(pending) >= 2:
            source, pair = pending.popleft()
            yield source, [job.result() for job in pair]
    for source, pair in pending:
        yield source, [job.result() for job in pair]


def finish(work, mono, geom, edges, rate, opts):
    from stage_vr360 import capture, read, writer
    import subprocess

    inside = ~geom.outer
    ax = ((geom.mx + 0.5) * W / geom.size[0] - 0.5 + pad).astype("f")
    ay = ((geom.my + 0.5) * H / geom.size[1] - 0.5 + pad).astype("f")
    radius = max(1, round(opts.get("donor_radius",65) * rate / 60))
    color = engine.video(engine.probe(work / "stereo.mp4")).get("color_space")
    color = color if color in ("bt709", "bt470bg", "smpte170m", "bt2020nc") else "bt709"
    matrix = {"bt470bg": "bt601", "smpte170m": "bt601", "bt2020nc": "bt2020"}.get(
        color, "bt709"
    )
    options = [
        "-filter_threads",
        "2",
        "-vf",
        f"scale=iw:ih:out_color_matrix={matrix},format=yuv420p",
        "-colorspace",
        color,
        "-color_range",
        "tv",
        "-c:v",
        "libx264",
        "-qp",
        "0",
        "-threads",
        "3",
        "-preset",
        "ultrafast",
    ]
    with ExitStack() as stack:
        cap = stack.enter_context(capture(work / "stereo.mp4"))
        pool = stack.enter_context(ThreadPoolExecutor(max_workers=2)) if opts["observed_edges"] else None
        bgcap = stack.enter_context(capture(mono))
        out = stack.enter_context(writer(work / "stage.mkv", 1280, 640, rate, options))
        if opts["anime_detail"]:
            import torch
            from spandrel import ModelLoader

            torch.set_num_threads(2)
            net = ModelLoader().load_from_file(str(UPSCALER)).cuda().half().eval()
            gate = cv.resize(
                np.clip(
                    (
                        cv.distanceTransform(geom.outer.astype("uint8"), cv.DIST_L2, 5)
                        - 10
                    )
                    / 12,
                    0,
                    1,
                ),
                (1280, 1280),
            )
            grid = np.dstack(
                np.meshgrid(np.arange(1280, dtype="f"), np.arange(1280, dtype="f"))
            )
            highcap = stack.enter_context(capture(mono))
            detail_out = stack.enter_context(
                subprocess.Popen(
                    [
                        "ffmpeg",
                        "-v",
                        "error",
                        "-y",
                        "-f",
                        "rawvideo",
                        "-pix_fmt",
                        "yuv420p",
                        "-s",
                        "1280x1280",
                        "-r",
                        str(rate),
                        "-i",
                        "pipe:0",
                        "-c:v",
                        "ffv1",
                        "-threads",
                        "3",
                        "-color_range",
                        "tv",
                        str(work / "vr360-detail.mkv"),
                    ],
                    stdin=subprocess.PIPE,
                )
            )
            uv = bytes([128]) * (1280 * 640)
        for first, stop in zip(edges, edges[1:]):
            if opts["anime_detail"]:
                detail = detail_frames(highcap, first, stop, rate, net, gate, grid, opts.get("detail_fps",10))
            if pool is not None:
                matches = observations(pool, cap, first, stop, radius, opts.get("donor_count", 12), geom.size)
            for i in range(first, stop):
                if pool is not None:
                    sources, matched = next(matches)
                else:
                    sources = [cv.resize(eye, geom.size, interpolation=cv.INTER_AREA)
                               for eye in np.hsplit(read(cap), 2)]
                    matched = None
                bg = read(bgcap)
                seen = inside.copy()
                eyes = []
                for eye in range(2):
                    frame = bg.copy()
                    if matched is not None:
                        patch, mask = matched[eye]
                        alpha = mask.astype("f") * (opts.get("observed_strength",1.0) / 255)
                        weight = cv.remap(alpha, ax, ay, cv.INTER_LINEAR)
                        real = cv.remap(
                            patch.astype("f") * alpha[:, :, None],
                            ax,
                            ay,
                            cv.INTER_LINEAR,
                        )
                        region = geom.outer & (weight > 0)
                        frame[region] = np.uint8(
                            np.rint(
                                np.clip(
                                    bg[region] * (1 - weight[region, None])
                                    + real[region],
                                    0,
                                    255,
                                )
                            )
                        )
                        seen |= region
                    frame[inside] = geom.project(sources[eye])[inside]
                    eyes.append(frame)
                out.write(np.hstack(eyes).tobytes())
                if opts["anime_detail"]:
                    field = next(detail)
                    dist = cv.distanceTransform((~seen).astype("uint8"), cv.DIST_L2, 3)
                    mask = cv.resize(np.clip((dist - 2) / 5, 0, 1), (1280, 1280))
                    detail_out.stdin.write(
                        np.uint8(
                            np.rint(128 + (field.astype("f") - 128) * mask * opts.get("detail_strength",1.0))
                        ).tobytes()
                    )
                    detail_out.stdin.write(uv)
        if opts["anime_detail"]:
            detail_out.stdin.close()
            if detail_out.wait():
                raise RuntimeError("外围细节编码失败")
