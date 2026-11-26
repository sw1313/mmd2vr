"""LTX-2.3 VR360 A outpainting, shared by the local and NAS workers."""

import bisect
from concurrent.futures import ThreadPoolExecutor
import contextlib
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import time
from fractions import Fraction

import cv2 as cv
import numpy as np
import requests
from scipy.ndimage import gaussian_filter1d

import engine
from stage_background import ray_grid
from stage_context import is_cut, projection
import stage_model
from vr360_guides import BackgroundExtension, choose
from vr360_rim import SmoothRim
from vr360_options import DEFAULT_PROMPT, LTX25_ASSETS, LTX25_AUDIO_VAE, LTX_FRONT_WIDTH, ltx25_available, ltx25_path
from vr360_options import (
    CUSTOM_ASSET_DIRS, custom_model_files, gguf_loader_path, model_file_versions, syfe_union_path,
)
from vr360_finish import ExteriorFinish
from vr360_union import People, add_guide, load_depth, write_depth


def model_dirs():
    local = engine.ROOT / ".runtime/ltx23-ab"
    root = Path(os.environ.get("MMD_VR360_WEIGHTS", engine.MODELS / "vr360-a"))
    if not root.exists() and os.name == "nt":
        root = local / "weights"
    paths = {
        name: root / name for name in ("diffusion_models", "vae", "loras", "embeddings")
    }
    if root == local / "weights":
        paths["embeddings"] = local / "base/models/embeddings"
    paths["text_encoders"] = root / "text_encoders"
    return {key: str(path.resolve()) for key, path in paths.items()}


class Geometry:
    def __init__(self, width, height, angle, curve, keep):
        self.background_extension = None
        self.size = (width, height)
        self.mx, self.my, self.outer = projection(width, height, angle, curve, keep)
        lon, lat = np.meshgrid(
            ((np.arange(1280) + 0.5) / 1280 - 0.5) * 2 * np.pi,
            ((np.arange(640) + 0.5) / 640 - 0.5) * np.pi,
        )
        z = np.cos(lat) * np.cos(lon)
        den = np.maximum(z, 1e-6) * math.tan(math.radians(angle / 2))
        self.px = (
            ((np.cos(lat) * np.sin(lon) / den + 1) * width / 2 - 0.5)
            .clip(-1e4, 1e4)
            .astype("f")
        )
        self.py = (
            ((np.sin(lat) / den * width / height + 1) * height / 2 - 0.5)
            .clip(-1e4, 1e4)
            .astype("f")
        )
        self.invalid = (
            (z <= 0)
            | (self.px < -0.5)
            | (self.px >= width - 0.5)
            | (self.py < -0.5)
            | (self.py >= height - 0.5)
        )
        u, v, z = ray_grid(640, angle, curve, keep).transpose(2, 0, 1)
        self.sx = ((np.arctan2(u, z) / (2 * np.pi) + 0.5) * 1280 - 0.5).astype("f")
        self.sy = np.clip(
            (np.arctan2(v, np.hypot(u, z)) / np.pi + 0.5) * 640 - 0.5, 0, 639
        ).astype("f")

    def guide(self, frame):
        image = cv.remap(
            frame, self.px, self.py, cv.INTER_LINEAR, borderMode=cv.BORDER_REPLICATE
        )
        image[self.invalid] = 0
        return image

    def project(self, frame):
        return cv.remap(
            frame, self.mx, self.my, cv.INTER_LINEAR, borderMode=cv.BORDER_REPLICATE
        )


@contextlib.contextmanager
def capture(path):
    cap = cv.VideoCapture(str(path))
    try:
        if not cap.isOpened():
            raise OSError(f"无法打开 {path}")
        yield cap
    finally:
        cap.release()


def read(cap):
    ok, frame = cap.read()
    if not ok:
        raise OSError("视频提前结束")
    return frame


@contextlib.contextmanager
def writer(path, width, height, rate, options):
    args = [
        "ffmpeg",
        "-nostdin",
        "-v",
        "error",
        "-y",
        "-f",
        "rawvideo",
        "-pix_fmt",
        "bgr24",
        "-s",
        f"{width}x{height}",
        "-r",
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
            raise RuntimeError(f"视频编码失败 {path.name}")


LOSSLESS = ["-c:v", "ffv1", "-threads", "3"]


def small_clip(work, split_shots=True):
    clip = work / "clip.mkv"
    stream = engine.video(engine.probe(clip))
    w, h = stream["width"], stream["height"]
    size = tuple(max(2, round(v * 960 / max(w, h))) for v in (w, h))
    path = work / "guide-source.mkv"
    subprocess.run(
        [
            "ffmpeg",
            "-v",
            "error",
            "-y",
            "-i",
            str(clip),
            "-map",
            "0:v:0",
            "-an",
            "-vf",
            f"scale={size[0]}:{size[1]}:flags=area",
            *LOSSLESS,
            str(path),
        ],
        check=True,
    )
    edges = [0]
    previous = None
    with capture(path) as cap:
        i = 0
        while True:
            ok, image = cap.read()
            if not ok:
                break
            thumb = cv.resize(image, (96, 64), interpolation=cv.INTER_AREA)
            if split_shots and previous is not None and is_cut(previous, thumb):
                edges.append(i)
            previous = thumb
            i += 1
    if not i:
        raise ValueError("视频没有可用帧")
    return path, size, edges + [i]


def write_source_mask(path, geom, frames, options):
    coverage = cv.resize((~geom.invalid).astype("f"), (40, 20), interpolation=cv.INTER_AREA)
    inside, outside = options["source_mask_inside"], options["source_mask_outside"]
    weights = outside + (inside - outside) * coverage
    if options.get("front_only"):
        weights = weights[:, 8:32]
    np.save(path, np.broadcast_to(weights, (frames, *weights.shape)))


def apply_inference_controls(graph, uid, options):
    source = graph["13"]
    masked = options.get("source_mask_enabled", False)
    attention = options.get("source_attention", 1.0)
    if masked or attention != 1.0:
        source["class_type"] = "LTXAddVideoICLoRAGuideAdvanced"
        source["inputs"]["attention_strength"] = attention
    if masked:
        graph["source_weights"] = {"class_type": "VR360DepthMask", "inputs": {"name": uid + "-source.npy"}}
        source["inputs"]["attention_mask"] = ["source_weights", 0]
    if not options.get("joint_audio", True):
        graph["20"]["inputs"]["latent_image"] = graph["15"]["inputs"]["video_latent"]
        graph["22"]["inputs"]["latent"] = ["20", 0]
        for key in ("4", "14", "15", "21"):
            del graph[key]
    if "png_compression" in options:
        graph["24"]["class_type"] = "VR360SaveFrames"
        graph["24"]["inputs"]["compression"] = options["png_compression"]
    if options.get("front_only"):
        graph["12"]["inputs"]["width"] = LTX_FRONT_WIDTH
        for key in ("source_weights", "depth_mask"):
            if key in graph:
                graph[key]["inputs"]["width"] = LTX_FRONT_WIDTH


def configure_ltx25(graph, prompt):
    graph["1"] = {"class_type": "UnetLoaderGGUF",
                  "inputs": {"unet_name": LTX25_ASSETS["diffusion_models"]}}
    graph["3"]["inputs"]["vae_name"] = LTX25_ASSETS["vae"]
    graph["4"]["inputs"]["vae_name"] = LTX25_AUDIO_VAE
    graph["clip"] = {"class_type": "CLIPLoader", "inputs": {
        "clip_name": LTX25_ASSETS["text_encoders"], "type": "ltxv", "device": "default",
    }}
    graph["7"] = {"class_type": "CLIPTextEncode", "inputs": {"clip": ["clip", 0], "text": prompt}}


def configure_custom_ltx(graph, manifest, prompt):
    assets = manifest["assets"]
    checkpoint = assets["checkpoint"]
    if checkpoint.lower().endswith(".gguf"):
        graph["1"] = {"class_type": "UnetLoaderGGUF", "inputs": {"unet_name": checkpoint}}
    else:
        graph["1"]["inputs"]["unet_name"] = checkpoint
    graph["2"]["inputs"]["lora_name"] = assets["outpaint_lora"]
    graph["3"]["inputs"]["vae_name"] = assets["video_vae"]
    if "audio_vae" in assets:
        graph["4"]["inputs"]["vae_name"] = assets["audio_vae"]
    if manifest["family"] == "ltx25":
        loader = "CLIPLoader"
        names = {"clip_name": assets["text_encoder"]}
    else:
        loader = "DualCLIPLoader"
        names = {"clip_name1": assets["text_encoder"], "clip_name2": assets["text_projection"]}
    graph["clip"] = {"class_type": loader, "inputs": {**names, "type": "ltxv", "device": "default"}}
    graph["7"] = {"class_type": "CLIPTextEncode", "inputs": {"clip": ["clip", 0], "text": prompt}}


def infer(instance, tmp, guide, uid, frames, fps, seed, checkpoint, options=None):
    url, proc = instance
    graph = json.loads((engine.ROOT / "vr360_graph.json").read_text())
    graph["2"]["inputs"]["lora_name"] = checkpoint
    graph["10"]["inputs"]["file"] = guide.name
    graph["12"]["inputs"]["length"] = graph["14"]["inputs"]["frames_number"] = frames
    graph["9"]["inputs"]["frame_rate"] = graph["14"]["inputs"]["frame_rate"] = fps
    graph["19"]["inputs"]["noise_seed"] = seed
    graph["24"]["inputs"]["filename_prefix"] = uid + "/image"
    if options and "steps" in options:
        graph["2"]["inputs"]["strength_model"] = options["outpaint_weight"]
        graph["13"]["inputs"]["strength"] = options["source_strength"]
        graph["16"]["inputs"]["cfg"] = options["cfg"]
        graph["18"]["inputs"]["sampler_name"] = options["sampler"]
        if options["steps"] != 8:
            schedule = np.fromstring(graph["17"]["inputs"]["sigmas"], sep=",")
            sigmas = np.interp(np.linspace(0,8,options["steps"]+1), np.arange(9), schedule)
            graph["17"]["inputs"]["sigmas"] = ",".join(f"{v:.8g}" for v in sigmas)
        graph["13"]["inputs"].update(use_tiled_encode=options["vae_tiled"],tile_size=options["vae_tile"])
        if options["vae_tiled"]:
            graph["23"]["inputs"]["tile_size"] = options["vae_tile"]
        else:
            graph["23"] = {"class_type":"VAEDecode","inputs":{"samples":["22",2],"vae":["3",0]}}
        if manifest := options.get("custom_model"):
            configure_custom_ltx(graph, manifest, options["prompt"])
        elif options.get("outpaint_model") == "ltx25":
            configure_ltx25(graph, options["prompt"])
        elif options["prompt"] != DEFAULT_PROMPT or not (Path(model_dirs()["embeddings"]) / "vr.safetensors").is_file():
            graph["clip"] = {"class_type":"DualCLIPLoader","inputs":{
                "clip_name1":"gemma_3_12B_it_fp4_mixed.safetensors",
                "clip_name2":"ltx-2.3_text_projection_bf16.safetensors","type":"ltxv","device":"default"}}
            graph["7"] = {"class_type":"CLIPTextEncode","inputs":{"clip":["clip",0],"text":options["prompt"]}}
    if options and options.get("union_enabled"):
        add_guide(graph, uid, options)
    if options:
        apply_inference_controls(graph, uid, options)
    response = requests.post(url + "/prompt", json={"prompt": graph}, timeout=30)
    if not response.ok:
        raise RuntimeError(response.text[:3000])
    job = response.json()["prompt_id"]
    start = time.monotonic()
    while proc.poll() is None and time.monotonic() - start < 3600:
        item = requests.get(url + "/history/" + job, timeout=25).json().get(job)
        if item:
            if item["status"]["status_str"] != "success":
                raise RuntimeError(str(item["status"]["messages"])[-2500:])
            files = [
                tmp / f["subfolder"] / f["filename"]
                for f in item["outputs"]["24"]["images"]
            ]
            if len(files) != frames:
                raise ValueError(f"VR360 输出 {len(files)} 帧，预期 {frames}")
            return files
        time.sleep(1)
    raise RuntimeError("VR360 推理服务退出或超时")


def windows(first, stop, rate, fps, opts):
    total = math.ceil((stop - first) * fps / rate)
    start = 0
    while start < total:
        end = min(start + opts.get("window_frames", 121), total)
        count = max(9, 1 + 8 * math.ceil((end - start - 1) / 8))
        if opts.get("outpaint_model") == "h3":
            count = end - start
        ids = [min(stop - 1, first + round((start + k) * rate / fps)) for k in range(count)]
        yield start, end, ids
        if end == total:
            return
        start = end - opts.get("window_overlap", 8)


def source_frames(cap, ids):
    cap.set(cv.CAP_PROP_POS_FRAMES, ids[0])
    index = ids[0] - 1
    for target in ids:
        if index < target:
            while index < target - 1:
                if not cap.grab():
                    raise OSError("无法读取条件帧")
                index += 1
            frame = read(cap)
            index += 1
        yield frame


def prepare_controls(path, tmp, geom, edges, rate, opts, matched_shots=()):
    depth = opts.get("union_enabled") and opts.get("depth_enabled", True)
    extend = opts.get("guide_extension_enabled", False)
    palette = opts.get("palette_enabled")
    if not (depth or palette or (extend and len(matched_shots) < len(edges) - 1)):
        return
    detector = People(Path(model_dirs()["loras"]).parent / "detectors/person.onnx", opts.get("detector_cuda", False))
    model = load_depth() if depth else None
    fps = opts["generation_fps"]
    with capture(path) as cap:
        for shot, (first, stop) in enumerate(zip(edges, edges[1:])):
            needs_background = extend and shot not in matched_shots
            previous_boxes = {}
            for start, _, ids in windows(first, stop, rate, fps, opts):
                if not (depth or needs_background or (palette and start == 0)):
                    break
                uid = f"shot-{shot}-{start}"
                frames = list(source_frames(cap,ids))
                boxes = []
                window_boxes = {}
                for frame_id, frame in zip(ids, frames, strict=True):
                    if frame_id not in previous_boxes:
                        previous_boxes[frame_id] = detector.boxes(frame, opts["person_threshold"])
                    window_boxes[frame_id] = previous_boxes[frame_id]
                    boxes.append(window_boxes[frame_id])
                previous_boxes = window_boxes
                if needs_background or (palette and start == 0):
                    (tmp / (uid + "-people.json")).write_text(json.dumps(boxes))
                if model is not None:
                    write_depth(tmp, uid, frames, boxes, fps, geom, model, opts)
                print(f"VR360 外围条件 {uid}", flush=True)
    if model is not None:
        import torch
        del model
        torch.cuda.empty_cache()


def generate_window(instance, tmp, cap, geom, warp, shot, start, ids, opts):
    if opts.get("outpaint_model") == "h3":
        from vr360_h3 import generate
        return generate(instance, tmp, cap, geom, warp, shot, start, ids, opts)
    fps = opts["generation_fps"]
    uid = f"shot-{shot}-{start}"
    guide = tmp / (uid + ".mkv")
    source = []
    width = LTX_FRONT_WIDTH if opts.get("front_only") else 1280
    left = (1280 - width) // 2
    extend = opts.get("guide_extension_enabled", False) and warp is None
    if extend:
        if geom.background_extension is None:
            geom.background_extension = BackgroundExtension(geom, opts)
        boxes = json.loads((tmp / (uid + "-people.json")).read_text())
        if len(boxes) != len(ids):
            raise ValueError("背景延续的人物框与引导帧数不匹配")
    with writer(guide, width, 640, fps, LOSSLESS) as enc:
        for i, (frame, target) in enumerate(zip(source_frames(cap,ids),ids,strict=True)):
            image = geom.guide(frame)
            if warp:
                image = warp.apply(image, target)
            elif extend:
                nearby = [box for adjacent in boxes[max(0, i-1):i+2] for box in adjacent]
                geom.background_extension.apply(image, frame, nearby)
            enc.write(image[:, left:left + width].tobytes())
            source.append(geom.project(frame))
    if opts.get("source_mask_enabled"):
        write_source_mask(tmp / (uid + "-source.npy"), geom, len(ids), opts)
    files = infer(instance, tmp, guide, uid, len(ids), fps, opts["seed"] + (shot if opts.get("seed_per_shot",True) else 0), opts["outpaint_checkpoint"], opts)
    if opts.get("source_mask_enabled"):
        (tmp / (uid + "-source.npy")).unlink()
    guide.unlink()
    if opts.get("union_enabled") and opts.get("depth_enabled", True):
        (tmp / (uid + "-depth.mkv")).unlink()
        (tmp / (uid + "-mask.npy")).unlink()
    return files, source


def prefetch_windows(pool, instance, tmp, cap, geom, warps, edges, rate, opts):
    # Keep exactly one inference in flight while the preceding window is composited.
    pending = None
    for shot, (first, stop) in enumerate(zip(edges, edges[1:])):
        for start, _, ids in windows(first, stop, rate, opts["generation_fps"], opts):
            ready = pending.result() if pending is not None else None
            pending = pool.submit(generate_window, instance, tmp, cap, geom, warps.get(shot),
                                  shot, start, ids, opts)
            if ready is not None:
                yield ready
    if pending is not None:
        yield pending.result()


def render_shot(
    instance, tmp, cap, out, geom, rim, warp, shot, first, stop, rate, opts, prepared=None
):
    fps = opts["generation_fps"]
    total = math.ceil((stop - first) * fps / rate)
    finish = ExteriorFinish(geom, opts) if opts.get("texture_enabled") or opts.get("palette_enabled") else None
    previous = {}
    overlap = opts.get("window_overlap",8)
    next_output = first
    verify_center = opts.get("outpaint_model") == "h3" and os.environ.get("MMD_H3_PROFILE") == "1"
    front_only = opts.get("front_only", False)
    margin = (1280 - LTX_FRONT_WIDTH) // 2
    expected_width = LTX_FRONT_WIDTH if front_only else 1280

    def read_panorama(file):
        image = cv.imread(str(file))
        if image is None or image.shape[:2] != (640, expected_width):
            raise ValueError(f"VR360 输出损坏 {file.name}")
        if front_only:
            image = cv.copyMakeBorder(image, 0, 0, margin, margin, cv.BORDER_CONSTANT)
        return image

    for start, end, ids in windows(first, stop, rate, fps, opts):
        count = len(ids)
        uid = f"shot-{shot}-{start}"
        if prepared is None:
            files, source = generate_window(instance, tmp, cap, geom, warp, shot, start, ids, opts)
        else:
            files, source = next(prepared)
        if opts.get("palette_enabled") and start == 0:
            people = json.loads((tmp / (uid + "-people.json")).read_text())
            picks = list(range(0, end-start, 5))
            originals = source_frames(cap, [ids[i] for i in picks])
            finish.fit_palette((raw, read_panorama(files[i]), people[i]) for i,raw in zip(picks, originals))
        bgs = []
        for k, file in enumerate(files):
            image = read_panorama(file)
            if finish is not None:
                image = finish.apply(image)
            bg = cv.remap(
                image, geom.sx, geom.sy, cv.INTER_CUBIC, borderMode=cv.BORDER_WRAP
            )
            if k + start in previous:
                alpha = (k + 1) / (overlap + 1)
                bg = cv.addWeighted(previous[k + start], 1 - alpha, bg, alpha, 0)
            bgs.append(bg)
            file.unlink()
        previous = {
            start + k: bgs[k] for k in range(max(0, end - start - overlap), end - start)
        }
        fields = (
            gaussian_filter1d(
                np.stack([rim.field(bg, ref) for bg, ref in zip(bgs, source)]),
                1.05,
                axis=0,
                mode="nearest",
            )
            if rim
            else None
        )
        limit = stop if end == total else min(stop, first + round((end - overlap) * rate / fps))
        cap.set(cv.CAP_PROP_POS_FRAMES, next_output)
        for index in range(next_output, limit):
            hi = min(count - 1, bisect.bisect_right(ids, index))
            lo = max(0, hi - 1)
            t = np.clip((index - ids[lo]) / max(1, ids[hi] - ids[lo]), 0, 1)
            bg = cv.addWeighted(bgs[lo], 1 - t, bgs[hi], t, 0)
            ref = geom.project(read(cap))
            bg[~geom.outer] = ref[~geom.outer]
            if rim:
                bg = rim.adjust(bg, (fields[lo] * (1 - t) + fields[hi] * t) * opts.get("seam_strength",1.0))
            if verify_center and not np.array_equal(bg[~geom.outer], ref[~geom.outer]):
                raise RuntimeError(f"H3 原片中心在合成时发生变化：{index}")
            out.write(bg.tobytes())
        next_output = limit
        print(f"VR360 A {uid} frame={limit}", flush=True)
        if end == total:
            break


def render(work, key, eye, angle, rate):
    from vr360_stereo import finish

    with engine.db() as db:
        spec = json.loads(
            db.execute("SELECT spec FROM jobs WHERE id=?", (key,)).fetchone()[0]
        )
    opts = spec["vr360"]
    # Jobs submitted before checkpoint selection used step 7000.
    h3 = opts.get("outpaint_model") == "h3"
    ltx25 = opts.get("outpaint_model") == "ltx25"
    directories = model_dirs()
    gguf_nodes = None
    manifest = opts.get("custom_model")
    custom_paths = {}
    if manifest:
        custom_paths = custom_model_files(manifest["id"], manifest["assets"])
        if model_file_versions(custom_paths) != manifest["versions"]:
            raise ValueError("自定义模型权重已在入队后变更，请重新提交任务")
    if h3:
        import vr360_h3
        opts = dict(opts, window_frames=opts["h3_window"], window_overlap=opts["h3_overlap"])
        directories = vr360_h3.model_dirs()
        if opts["h3_union"] and not stage_model.h3_union_table().is_file():
            raise FileNotFoundError("缺少 H3 Union 时间表。请按 docs/models.md 提取 fl-union-table.npy。")
        print(f"H3 FL2VA {opts['h3_size']} px · Union 2.0 {opts['h3_union']}", flush=True)
    else:
        if opts.get("union_enabled") and opts.get("union_variant") == "syfe":
            syfe_weights = syfe_union_path()
            if syfe_weights is None:
                raise RuntimeError("本服务缺少 SyFe Union 权重")
            directories["loras"] += "\n" + str(syfe_weights.parent)
            print("SyFe Union · 1.0× 参考分辨率", flush=True)
        if ltx25 and not manifest:
            if not ltx25_available():
                raise RuntimeError("LTX-2.5 Q4_K_M 权重或 GGUF 加载器未安装")
            home = ltx25_path()
            for key in LTX25_ASSETS:
                directories[key] = str(home / "models" / key)
            gguf_nodes = home / "gguf_nodes"
            print("LTX-2.5 Distilled Q4_K_M", flush=True)
        if manifest:
            opts = dict(opts, outpaint_checkpoint=manifest["assets"]["outpaint_lora"])
        checkpoint = opts.setdefault("outpaint_checkpoint", "360vroutpaint_v2_step07000.safetensors")
        print(f"VR360 A 扩画权重 {checkpoint}", flush=True)
        if opts.get("front_only"):
            print(f"LTX 前半球推理 {LTX_FRONT_WIDTH}×640（左右各留 18°）", flush=True)
    if manifest:
        custom_dirs = {CUSTOM_ASSET_DIRS[role]: str(path.parent) for role, path in custom_paths.items()}
        if opts.get("union_enabled"):
            custom_dirs["loras"] += "\n" + directories["loras"]
        directories.update(custom_dirs)
        if custom_paths["checkpoint"].suffix.lower() == ".gguf":
            loader = gguf_loader_path()
            if not loader.is_file():
                raise FileNotFoundError("自定义 LTX 模型缺少 GGUF 加载器")
            gguf_nodes = loader.parent
        print(f"自定义扩画模型 {manifest['label']}", flush=True)
    cv.setNumThreads(2)
    profile = h3 and os.environ.get("MMD_H3_PROFILE") == "1"
    began = time.perf_counter()
    path, size, edges = small_clip(work, opts.get("split_shots", True))
    geom = Geometry(
        *size, angle, spec.get("curvature", 0), spec.get("keep_center", False)
    )
    if h3:
        geom.fisheye = vr360_h3.Fisheye(geom, angle, opts["h3_size"])
    warps = (
        choose(path, edges, angle, size[0] / size[1], float(rate))
        if opts["match_stage"]
        else {}
    )
    rim = SmoothRim(geom.outer) if opts["seam_luma"] else None
    mono = work / "vr360-mono.mkv"
    with tempfile.TemporaryDirectory(dir=work) as temp:
        tmp = Path(temp)
        prepare_controls(path, tmp, geom, edges, rate, opts, warps)
        controls_ready = time.perf_counter()
        with (
            stage_model.server(tmp, model_dirs=directories, vr360=True, offload_streams=opts.get("offload_streams"),
                               h3=h3, **({"gguf_nodes": gguf_nodes} if gguf_nodes else {})) as instance,
            capture(path) as cap,
            capture(path) as guide_cap,
            ThreadPoolExecutor(max_workers=1) as pool,
            writer(mono, 640, 640, rate, LOSSLESS) as out,
        ):
            prepared = prefetch_windows(pool, instance, tmp, guide_cap, geom, warps, edges, rate, opts)
            for shot, (first, stop) in enumerate(zip(edges, edges[1:])):
                render_shot(
                    instance,
                    tmp,
                    cap,
                    out,
                    geom,
                    rim,
                    warps.get(shot),
                    shot,
                    first,
                    stop,
                    rate,
                    opts,
                    prepared,
                )
    mono_ready = time.perf_counter()
    finish(work, mono, geom, edges, rate, opts)
    if profile:
        print("H3_STAGE " + json.dumps(dict(
            guides=controls_ready-began, inference_and_mono=mono_ready-controls_ready,
            stereo_and_detail=time.perf_counter()-mono_ready, shot_edges=edges,
        )), flush=True)


if __name__ == "__main__":
    render(
        Path(sys.argv[1]),
        sys.argv[2],
        int(sys.argv[3]),
        int(sys.argv[4]),
        Fraction(sys.argv[5]),
    )
