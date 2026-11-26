"""Condition VACE on a complete shot and render the forward hemisphere for iw3."""

import json
import math
import os
import subprocess
import sys
import tempfile
import time
from contextlib import contextmanager
from fractions import Fraction
from pathlib import Path

import av
import cv2
import numpy as np
import requests

import engine
import stage_model
from stage_background import ray_grid

SIZE = 640
WINDOW = 69
OVERLAP = 5
WEIGHTS = {
    "diffusion_models": "wan2.1_vace_1.3B_fp16.safetensors",
    "text_encoders": "umt5_xxl_fp8_e4m3fn_scaled.safetensors",
    "vae": "wan_2.1_vae.safetensors",
}


def model_dirs():
    flat = os.getenv("MMD_STAGE_WEIGHTS")
    local = engine.ROOT / ".runtime/vace-pilot/models"
    result = {}
    for kind, name in WEIGHTS.items():
        folder = Path(flat) if flat else local / kind
        if not (folder / name).is_file():
            raise FileNotFoundError(f"缺少 VACE 权重 {folder / name}")
        result[kind] = str(folder.resolve())
    return result


def projection(width, height, angle, curve, keep):
    rays = ray_grid(SIZE, angle, curve, keep)
    half = math.tan(math.radians(angle / 2))
    u = rays[:, :, 0] / rays[:, :, 2] / half
    v = rays[:, :, 1] / rays[:, :, 2] / (half * height / width)
    outer = (np.abs(u) > 1) | (np.abs(v) > 1)
    mx = ((u + 1) * width / 2 - 0.5).astype("float32")
    my = ((v + 1) * height / 2 - 0.5).astype("float32")
    return mx, my, outer


def is_cut(previous, current):
    diff = cv2.absdiff(previous, current)
    return float(diff.mean()) > 38 and float((diff.max(axis=2) > 35).mean()) > 0.55


def sampled_frames(frames, stride):
    """Keep both endpoints of each shot. Look for cuts at the original frame rate."""
    last_thumb = None
    pending = None
    for index, (image, thumb) in enumerate(frames):
        cut = last_thumb is None or is_cut(last_thumb, thumb)
        if cut and pending is not None:
            yield pending
            pending = None
        if cut:
            first = index
        if (index - first) % stride == 0:
            yield index, image, cut
            pending = None
        else:
            pending = index, image, False
        last_thumb = thumb
    if pending is not None:
        yield pending


def windows(frames):
    batch, carry = [], False
    for index, image, cut in frames:
        if batch and (cut or len(batch) == WINDOW - (OVERLAP if carry else 0)):
            yield batch, carry
            batch, carry = [], not cut
        batch.append((index, image))
    if batch:
        yield batch, carry


def graph(control, mask, length, steps=30):
    def node(class_type, **inputs):
        return {"class_type": class_type, "inputs": inputs}

    return {
        "1": node(
            "UNETLoader", unet_name=WEIGHTS["diffusion_models"], weight_dtype="default"
        ),
        "2": node(
            "CLIPLoader",
            clip_name=WEIGHTS["text_encoders"],
            type="wan",
            device="default",
        ),
        "3": node("VAELoader", vae_name=WEIGHTS["vae"]),
        "4": node(
            "CLIPTextEncode",
            clip=["2", 0],
            text=(
                "Extend the existing shot outward as a continuous wide view of this same scene. "
                "Continue the visible surfaces, edges, materials, colors and illumination at the "
                "frame boundaries. Preserve the camera perspective, layout and movement of the source."
            ),
        ),
        "5": node(
            "CLIPTextEncode",
            clip=["2", 0],
            text=(
                "duplicate characters, extra figures, extra faces, disconnected limbs, flicker, "
                "warped scenery, frame border, separate inset picture, text, watermark"
            ),
        ),
        "6": node("LoadVideo", file=control),
        "7": node("GetVideoComponents", video=["6", 0]),
        "8": node("LoadVideo", file=mask),
        "9": node("GetVideoComponents", video=["8", 0]),
        "10": node("ImageToMask", image=["9", 0], channel="red"),
        "11": node(
            "WanVaceToVideo",
            positive=["4", 0],
            negative=["5", 0],
            vae=["3", 0],
            width=SIZE,
            height=SIZE,
            length=length,
            batch_size=1,
            strength=1.0,
            control_video=["7", 0],
            control_masks=["10", 0],
        ),
        "12": node("ModelSamplingSD3", model=["1", 0], shift=8.0),
        "13": node(
            "KSampler",
            model=["12", 0],
            positive=["11", 0],
            negative=["11", 1],
            latent_image=["11", 2],
            seed=726391,
            steps=steps,
            cfg=5.0,
            sampler_name="uni_pc",
            scheduler="simple",
            denoise=1.0,
        ),
        "14": node("VAEDecode", samples=["13", 0], vae=["3", 0]),
        "15": node("SaveImage", images=["14", 0], filename_prefix="outpaint"),
    }


@contextmanager
def encoder(path, rate, options):
    args = [
        "ffmpeg",
        "-v",
        "error",
        "-y",
        "-f",
        "rawvideo",
        "-pix_fmt",
        "bgr24",
        "-s",
        f"{SIZE}x{SIZE}",
        "-framerate",
        str(rate),
        "-i",
        "pipe:0",
        "-an",
        *options,
        str(path),
    ]
    with subprocess.Popen(args, stdin=subprocess.PIPE) as proc:
        try:
            yield proc.stdin
        finally:
            proc.stdin.close()
        if proc.wait():
            raise RuntimeError(f"编码失败：{path.name}")


def generate(instance, folder, frames, prior, outer, rate, steps):
    length = max(5, 1 + 4 * math.ceil((len(prior) + len(frames) - 1) / 4))
    opts = ["-c:v", "libx264rgb", "-crf", "0", "-preset", "ultrafast", "-threads", "2"]
    blank = np.zeros((SIZE, SIZE, 3), np.uint8)
    mask = blank.copy()
    mask[outer] = 255
    with (
        encoder(folder / "control.mp4", rate, opts) as control,
        encoder(folder / "mask.mp4", rate, opts) as masks,
    ):
        for image in prior:
            control.write(image.tobytes())
            masks.write(blank.tobytes())
        for i in range(length - len(prior)):
            control.write(frames[min(i, len(frames) - 1)].tobytes())
            masks.write(mask.tobytes())
    url, proc = instance
    response = requests.post(
        url + "/prompt",
        json={"prompt": graph("control.mp4", "mask.mp4", length, steps)},
        timeout=30,
    )
    if not response.ok:
        raise RuntimeError(f"VACE 工作流提交失败：{response.text[:1200]}")
    uid = response.json()["prompt_id"]
    deadline = time.monotonic() + 3600
    while proc.poll() is None and time.monotonic() < deadline:
        item = requests.get(url + "/history/" + uid, timeout=20).json().get(uid)
        if item:
            if item["status"]["status_str"] != "success":
                raise RuntimeError(str(item["status"]["messages"])[-1800:])
            paths = [
                folder / f["subfolder"] / f["filename"]
                for f in item["outputs"]["15"]["images"]
            ]
            if len(paths) != length:
                raise RuntimeError("VACE 输出帧数不匹配")
            try:
                for path in paths[len(prior) : len(prior) + len(frames)]:
                    im = cv2.imread(str(path))
                    if im is None or im.shape[:2] != (SIZE, SIZE):
                        raise RuntimeError("VACE 输出帧损坏")
                    yield im
            finally:
                for path in paths:
                    path.unlink(missing_ok=True)
            return
        time.sleep(1)
    raise RuntimeError("VACE 服务退出或生成超时")


def intermediate_frames(left, right, distance, grid):
    if distance <= 1:
        return
    # Estimate motion at half size on the completed frames. Warp both endpoints
    # toward each intermediate timestamp before blending to mitigate double edges.
    small = [cv2.resize(cv2.cvtColor(im, cv2.COLOR_BGR2GRAY), None, fx=.5, fy=.5,
                        interpolation=cv2.INTER_AREA) for im in (left, right)]
    flow = cv2.DISOpticalFlow_create(cv2.DISOPTICAL_FLOW_PRESET_FAST)
    forward, backward = [
        cv2.resize(flow.calc(a, b, None), (SIZE, SIZE)) * 2
        for a, b in (small, small[::-1])
    ]
    for offset in range(1, distance):
        t = offset / distance
        a = cv2.remap(left, grid - t * forward, None, cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
        b = cv2.remap(right, grid - (1 - t) * backward, None, cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
        yield cv2.addWeighted(a, 1 - t, b, t, 0)


def render(work, key, eye, angle, rate):
    weights = model_dirs()
    with engine.db() as conn:
        spec = json.loads(
            conn.execute("SELECT spec FROM jobs WHERE id=?", (key,)).fetchone()[0]
        )
    cv2.setNumThreads(2)
    balanced = spec.get("vace_speed") == "balanced-v1"
    stride = max(1, math.ceil(rate / 10)) if balanced else 1
    steps = 16 if balanced else 30
    grid = np.dstack(np.meshgrid(np.arange(SIZE, dtype="f"), np.arange(SIZE, dtype="f"))) if stride > 1 else None
    clip = work / "clip.mkv"
    stream = engine.video(engine.probe(clip))
    w, h = stream["width"], stream["height"]
    scale = min(1.0, 960 / max(w, h))
    w, h = max(1, round(w * scale)), max(1, round(h * scale))
    mx, my, outer = projection(
        w, h, angle, spec.get("curvature", 0), spec.get("keep_center", False)
    )
    # Correct low-frequency color mismatch only within a narrow band of the join.
    known = (~outer).astype("float32")

    def blur(image):
        return cv2.GaussianBlur(image, (0, 0), 5)

    norm = blur(known)
    band = ((norm > 0.015) & (norm < 0.985))[:, :, None]

    # Estimate colors on each side independently: the generated pixels inside
    # the mask can match the source while the adjacent exterior still differs.
    def edge_blur(image):
        return cv2.GaussianBlur(image, (0, 0), SIZE / 96)

    inside_mass = edge_blur(known)[:, :, None]
    outside_mass = 1 - inside_mass
    edge_weight = np.clip(inside_mass / 0.035, 0, 1) * np.clip(
        outside_mass / 0.35, 0, 1
    )

    def frames(container):
        for frame in container.decode(video=0):
            small = cv2.resize(
                frame.to_ndarray(format="bgr24"), (w, h), interpolation=cv2.INTER_AREA
            )
            projected = cv2.remap(
                small, mx, my, cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE
            )
            projected[outer] = 127
            yield projected, cv2.resize(small, (96, 64), interpolation=cv2.INTER_AREA)

    matrix = {"bt470bg": "bt601", "smpte170m": "bt601", "bt2020nc": "bt2020"}.get(
        stream.get("color_space"), "bt709"
    )
    opts = [
        "-filter_threads",
        "4",
        "-vf",
        f"scale={eye}:{eye}:flags=lanczos:out_color_matrix={matrix}:out_range=tv,format=yuv420p",
        "-colorspace",
        stream.get("color_space")
        if stream.get("color_space") in {"bt470bg", "smpte170m", "bt2020nc", "bt709"}
        else "bt709",
        "-color_range",
        "tv",
        "-c:v",
        "hevc_nvenc",
        "-preset",
        "p4",
        "-cq",
        "15",
        "-b:v",
        "0",
    ]
    with tempfile.TemporaryDirectory(dir=work) as tmp:
        folder = Path(tmp)
        with (
            stage_model.server(folder, model_dirs=weights) as instance,
            av.open(str(clip)) as cap,
            encoder(work / "stage.mkv", rate, opts) as video,
        ):
            history = []
            done = 0
            sampled = sampled_frames(frames(cap), stride)
            for idx, (batch, carry) in enumerate(windows(sampled), 1):
                print(f"VACE 窗口 {idx} · 已完成 {done} 帧 · 推理 {len(batch)} 帧 × {steps} 步", flush=True)
                prior = history if carry else []
                history = []
                if not carry:
                    previous = None
                for (index, src), gen in zip(
                    batch,
                    generate(instance, folder, [image for _, image in batch], prior, outer, rate / stride, steps),
                    strict=True,
                ):
                    delta = (src.astype("float32") - gen) * known[:, :, None]
                    correction = (
                        np.clip(
                            blur(delta) / np.maximum(norm[:, :, None], 0.015), -14, 14
                        )
                        * band
                    )
                    corrected = gen.astype("float32") + correction
                    if spec.get("stage_edge_version", 1) >= 2:
                        source_color = edge_blur(src * known[:, :, None]) / np.maximum(
                            inside_mass, 0.001
                        )
                        outer_color = edge_blur(
                            corrected * outer[:, :, None]
                        ) / np.maximum(outside_mass, 0.001)
                        corrected += (
                            np.clip(source_color - outer_color, -32, 32) * edge_weight
                        )
                    gen = np.uint8(np.clip(corrected, 0, 255))
                    if previous is not None and stride > 1:
                        for middle in intermediate_frames(previous, gen, index - done + 1, grid):
                            video.write(middle.tobytes())
                    video.write(gen.tobytes())
                    context = gen.copy()
                    context[~outer] = src[~outer]
                    previous = gen
                    done = index + 1
                    history.append(context)
                    if len(history) > OVERLAP:
                        history.pop(0)
            if not done:
                raise ValueError("片段没有可用视频帧")
            print(f"含角色扩画完成 frame={done}", flush=True)


if __name__ == "__main__":
    render(
        Path(sys.argv[1]),
        sys.argv[2],
        int(sys.argv[3]),
        int(sys.argv[4]),
        Fraction(sys.argv[5]),
    )
