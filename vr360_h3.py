"""Project continuous source frames to a protected fisheye canvas for H3 FL2VA."""
import hashlib
import json
import os
from pathlib import Path
import tempfile
import time

import cv2 as cv
import numpy as np
import requests
from scipy.ndimage import distance_transform_edt

import engine
from vr360_guides import BackgroundExtension
from vr360_options import H3_CHECKPOINTS


def model_dirs():
    root = engine.MODELS / "h3"
    if not root.is_dir() and os.name == "nt":
        root = Path(os.environ["LOCALAPPDATA"]) / "mmd2vr/pano-compare/h3"
    root = Path(os.environ.get("MMD_H3_WEIGHTS", root))
    return {k: str(root / k) for k in ("diffusion_models", "vae", "text_encoders", "model_patches")}


class Fisheye:
    def __init__(self, geom, angle, size):
        width, height = geom.size
        self.size = size
        self.source_size = (size, max(1, round(size * height / width)))
        sw, sh = self.source_size
        # A 512 canvas contains the validated central 448-pixel delivery square.
        radius = 219.52 * size / 512
        coord = np.arange(size) + .5 - size / 2
        x, y = np.meshgrid(coord, coord)
        r = np.hypot(x, y)
        theta = np.pi/2 * r/radius
        dx, dy, dz = x/r*np.sin(theta), y/r*np.sin(theta), np.cos(theta)
        focal = sw/2 / np.tan(np.deg2rad(angle/2))
        self.mx = (sw/2-.5 + focal*dx/np.maximum(dz, 1e-5)).astype("f")
        self.my = (sh/2-.5 + focal*dy/np.maximum(dz, 1e-5)).astype("f")
        self.valid = (r < radius) & (abs(self.mx-(sw/2-.5)) <= sw/2) & (abs(self.my-(sh/2-.5)) <= sh/2)
        self.nearest = tuple(distance_transform_edt(~self.valid, return_distances=False, return_indices=True))
        lon = np.arctan2(dx, dz)
        self.ex = ((lon/(2*np.pi)+.5)*1280-.5).astype("f")
        self.ey = ((np.arcsin(dy)/np.pi+.5)*640-.5).astype("f")
        self.allow = (np.maximum(abs(x), abs(y)) < size/2-size//16) & (abs(lon) < np.deg2rad(84)) & ~self.valid
        border = 32
        ypad = (size//8)//32*32
        rows, cols = np.where(self.valid)
        self.patch = (min(border, cols.min()//32*32), min(ypad, rows.min()//32*32),
                      max(size-border, (cols.max()//32+1)*32), max(size-ypad, (rows.max()//32+1)*32))
        lon, lat = np.meshgrid(((np.arange(1280)+.5)/1280-.5)*2*np.pi,
                               ((np.arange(640)+.5)/640-.5)*np.pi)
        dx, dy, dz = np.cos(lat)*np.sin(lon), np.sin(lat), np.cos(lat)*np.cos(lon)
        scale = radius*2/np.pi * np.minimum(np.arccos(dz), np.pi/2)/np.hypot(dx, dy)
        self.u = (size/2-.5+dx*scale).astype("f")
        self.v = (size/2-.5+dy*scale).astype("f")

    def project(self, frame, geom, warp, index, boxes, opts):
        small = cv.resize(frame, self.source_size, interpolation=cv.INTER_AREA)
        raw = cv.remap(small, self.mx, self.my, cv.INTER_LINEAR, borderMode=cv.BORDER_REPLICATE)
        image = raw[self.nearest]
        weight = np.zeros(self.valid.shape, "f")
        if warp or opts["guide_extension_enabled"]:
            guide = geom.guide(frame)
            if warp:
                guide = warp.apply(guide, index)
                alpha = geom.invalid.astype("f")
            else:
                if geom.background_extension is None:
                    geom.background_extension = BackgroundExtension(geom, opts)
                extend = geom.background_extension
                confidence = extend.apply(guide, frame, boxes)
                alpha = np.zeros(geom.invalid.shape, "f")
                alpha[extend.y, extend.x] = confidence
            weight = cv.remap(alpha, self.ex, self.ey, cv.INTER_LINEAR)
            weight[~self.allow] = 0
            guide[~geom.invalid] = 0
            colour = cv.remap(guide, self.ex, self.ey, cv.INTER_LINEAR)
            image = np.where(self.allow[:, :, None],
                             np.rint(image*(1-weight[:, :, None])+colour).clip(0, 255), image).astype("uint8")
        image[self.valid] = raw[self.valid]
        x0, y0, x1, y1 = self.patch
        # Even when the optional soft reference is off, FL2VA needs its source footprint.
        mask = (np.where(self.valid, opts["source_mask_inside"], weight*opts["source_mask_outside"])
                if opts["source_mask_enabled"] else self.valid.astype("f"))
        mask[:y0] = mask[y1:] = 0
        mask[:, :x0] = mask[:, x1:] = 0
        coverage = cv.resize(mask.astype("f"), (self.size//16,)*2, interpolation=cv.INTER_AREA)
        return cv.cvtColor(image[y0:y1, x0:x1], cv.COLOR_BGR2RGB), coverage


def graph(opts, window):
    nodes = json.loads((engine.ROOT / "h3_graph.json").read_bytes())
    checkpoint = opts.get("h3_checkpoint", H3_CHECKPOINTS["h3"])
    if checkpoint not in H3_CHECKPOINTS.values():
        raise ValueError(f"不支持的 H3 FL2VA 权重 {checkpoint}")
    nodes["8"]["inputs"]["unet_name"] = checkpoint
    if os.environ.get("MMD_H3_PROFILE") == "1":
        for key in ("6", "7", "8", "12", "14"):
            nodes[key]["class_type"] = "H3Timed" + nodes[key]["class_type"]
    nodes["12"]["inputs"]["text"] = opts["h3_prompt"]
    nodes["14"]["inputs"]["text"] = opts["h3_negative"]
    nodes["13"]["inputs"]["window"] = window
    if opts["h3_union"]:
        nodes["15"] = {"class_type": "ModelPatchLoader", "inputs": {
            "name": "minimax_h3_fun_controlnet_union_2.0_pruned_int8_convrot.safetensors"}}
        nodes["13"]["inputs"]["union"] = ["15", 0]
    return nodes


def generate(instance, tmp, cap, geom, warp, shot, start, ids, opts):
    from stage_vr360 import source_frames

    uid = f"shot-{shot}-{start}"
    fish = geom.fisheye
    people = tmp / (uid + "-people.json")
    boxes = json.loads(people.read_bytes()) if people.exists() else [[] for _ in ids]
    rgb, weights, source = [], [], []
    for i, frame in enumerate(source_frames(cap, ids)):
        nearby = [b for group in boxes[max(0, i-1):i+2] for b in group]
        image, mask = fish.project(frame, geom, warp, ids[i], nearby, opts)
        rgb.append(image)
        weights.append(mask)
        source.append(geom.project(frame))
    n = len(ids)
    aligned = max(22, n) + (5-max(22, n)) % 17
    tokens = (aligned-5)//17*5+2
    with tempfile.TemporaryDirectory(prefix=uid+"-", dir=tmp) as directory:
        work = Path(directory)
        pix = np.stack(rgb+[rgb[-1]]*(aligned-n))
        np.save(work/"projected.npy", pix)
        np.save(work/"holes.npy", (~fish.valid).astype("f"))
        np.save(work/"coverage.npy", np.stack([weights[min(n-1, round(i*(aligned-1)/(tokens-1)))] for i in range(tokens)]))
        cfg = dict(size=fish.size, patch=list(map(int, fish.patch)), n=n, aligned=aligned,
                   seed=opts["seed"]+(shot if opts["seed_per_shot"] else 0),
                   sha=hashlib.sha256(memoryview(pix)).hexdigest(), fps=opts["generation_fps"],
                   steps=opts["h3_steps"], cfg=opts["h3_cfg"], sampler=opts["h3_sampler"],
                   union_weight=opts["h3_strength"], compression=opts["png_compression"])
        (work/"params.json").write_text(json.dumps(cfg))
        del pix, rgb
        url, proc = instance
        submitted = time.perf_counter()
        reply = requests.post(url+"/prompt", json={"prompt": graph(opts, work.name)}, timeout=30)
        if not reply.ok:
            raise RuntimeError(reply.text[:3000])
        job = reply.json()["prompt_id"]
        while proc.poll() is None:
            item = requests.get(url+"/history/"+job, timeout=30).json().get(job)
            if item:
                if item["status"]["status_str"] != "success":
                    raise RuntimeError(str(item["status"]["messages"])[-3500:])
                if os.environ.get("MMD_H3_PROFILE") == "1":
                    cached = [v.get("nodes", []) for k, v in item["status"]["messages"] if k == "execution_cached"]
                    print("H3_GRAPH " + json.dumps(dict(window=work.name, seconds=time.perf_counter()-submitted,
                                                       cached=cached)), flush=True)
                break
            time.sleep(1)
        else:
            raise RuntimeError("H3 推理进程已退出")
        files = []
        for i in range(n):
            disk = cv.imread(str(work/f"{i:03}.png"))
            if disk is None or disk.shape[:2] != (fish.size, fish.size):
                raise ValueError(f"H3 窗口第 {i} 帧解码失败")
            erp = cv.remap(disk, fish.u, fish.v, cv.INTER_CUBIC, borderMode=cv.BORDER_REPLICATE)
            out = tmp / f"{uid}-{i}.png"
            if not cv.imwrite(str(out), erp, [cv.IMWRITE_PNG_COMPRESSION, opts["png_compression"]]):
                raise OSError("无法保存扩画帧")
            files.append(out)
    return files, source
