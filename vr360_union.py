"""Background-only depth control for the LTX Union adapter."""

import sys

import cv2 as cv
import numpy as np
from scipy.optimize import least_squares

import engine
from vr360_options import LTX_FRONT_WIDTH, SYFE_UNION

UNION_MODEL = "ltx-2.3-22b-ic-lora-union-control-ref0.5.safetensors"


def add_guide(graph, uid, options):
    syfe = options.get("union_variant") == "syfe"
    graph["union_model"] = {
        "class_type": "LTXICLoRALoaderModelOnly",
        "inputs": {
            "model": ["2", 0],
            "lora_name": SYFE_UNION if syfe else UNION_MODEL,
            "strength_model": options["union_weight"],
        },
    }
    graph["16"]["inputs"]["model"] = ["union_model", 0]
    if not options["depth_enabled"]:
        return
    graph["read_depth"] = {
        "class_type": "LoadVideo",
        "inputs": {"file": uid + "-depth.mkv"},
    }
    graph["depth_frames"] = {
        "class_type": "GetVideoComponents",
        "inputs": {"video": ["read_depth", 0]},
    }
    graph["depth_mask"] = {
        "class_type": "VR360DepthMask",
        "inputs": {"name": uid + "-mask.npy"},
    }
    for key in ("15", "16", "22"):
        for name, value in graph[key]["inputs"].items():
            if isinstance(value, list) and value[0] == "13":
                graph[key]["inputs"][name] = ["depth_guide", value[1]]
    graph["13"]["class_type"] = "LTXAddVideoICLoRAGuideAdvanced"
    graph["13"]["inputs"]["attention_strength"] = 1.0
    graph["depth_guide"] = {
        "class_type": graph["13"]["class_type"],
        "inputs": {
            **graph["13"]["inputs"],
            "positive": ["13", 0],
            "negative": ["13", 1],
            "latent": ["13", 2],
            "image": ["depth_frames", 0],
            "latent_downscale_factor": 1.0 if syfe else ["union_model", 1],
            "attention_mask": ["depth_mask", 0],
            "attention_strength": options["depth_attention"],
            "strength": options["depth_strength"],
        },
    }


class People:
    def __init__(self, path, use_cuda=False):
        import onnxruntime as ort

        settings = ort.SessionOptions()
        settings.intra_op_num_threads = 4
        settings.inter_op_num_threads = 1
        providers = ["CPUExecutionProvider"]
        if use_cuda and "CUDAExecutionProvider" in ort.get_available_providers():
            import torch
            if torch.cuda.is_available():
                if hasattr(ort, "preload_dlls"):
                    ort.preload_dlls()
                providers.insert(0, ("CUDAExecutionProvider", {"use_tf32": 0}))
        self.session = ort.InferenceSession(str(path), sess_options=settings, providers=providers)
        print(f"人物检测设备 {self.session.get_providers()[0]}", flush=True)

    def boxes(self, frame, threshold):
        h, w = frame.shape[:2]
        scale = 640 / max(h, w)
        width, height = max(1, round(w * scale)), max(1, round(h * scale))
        x, y = (640 - width) // 2, (640 - height) // 2
        square = np.full((640, 640, 3), 114, np.uint8)
        square[y : y + height, x : x + width] = cv.cvtColor(
            cv.resize(frame, (width, height)), cv.COLOR_BGR2RGB
        )
        rotations = (0, 1, 3)
        batch = np.stack([np.rot90(square, k) for k in rotations]).transpose(0, 3, 1, 2)
        pred = self.session.run(
            None, {"images": np.ascontiguousarray(batch, dtype=np.float32) / 255}
        )[0]
        boxes, scores = [], []
        for rotation, detections in zip(rotations, pred, strict=True):
            for cx, cy, bw, bh, score in detections.T[detections[4] >= threshold]:
                for _ in range(rotation):
                    cx, cy, bw, bh = 640 - cy, cx, bh, bw
                x0, x1 = np.clip(
                    (np.array([cx - bw / 2, cx + bw / 2]) - x) / scale, 0, w
                )
                y0, y1 = np.clip(
                    (np.array([cy - bh / 2, cy + bh / 2]) - y) / scale, 0, h
                )
                if min(x1 - x0, y1 - y0) >= 4:
                    boxes.append([int(x0), int(y0), int(x1 - x0), int(y1 - y0)])
                    scores.append(float(score))
        return [
            boxes[int(i)]
            for i in np.asarray(
                cv.dnn.NMSBoxes(boxes, scores, threshold, 0.4)
            ).flatten()
        ]


def actor_mask(shape, boxes, margin=0, edge_only=False):
    h, w = shape[:2]
    mask = np.zeros((h, w), np.uint8)
    for x, y, bw, bh in boxes:
        if edge_only and min(x, y, w - x - bw, h - y - bh) > 16:
            continue
        cv.rectangle(
            mask,
            (x - margin, y - margin),
            (x + bw - 1 + margin, y + bh - 1 + margin),
            1,
            -1,
        )
    return mask


def load_depth():
    import torch

    hub = engine.MODELS / "iw3/pretrained_models/hub"
    sys.path.insert(0, str(hub / "nagadomi_Video-Depth-Anything_iw3_main"))
    from video_depth_anything.video_depth import VideoDepthAnything

    model = VideoDepthAnything(
        encoder="vits", features=64, out_channels=[48, 96, 192, 384]
    )
    model.load_state_dict(
        torch.load(
            hub / "checkpoints/video_depth_anything_vits.pth",
            map_location="cpu",
            weights_only=True,
        )
    )
    return model.eval().cuda()


def write_depth(tmp, uid, frames, people, fps, geom, model, opts):
    from stage_vr360 import writer, LOSSLESS

    rgb = np.stack([cv.cvtColor(im, cv.COLOR_BGR2RGB) for im in frames])
    depth, _ = model.infer_video_depth(
        rgb, target_fps=fps, input_size=opts["depth_size"], device="cuda"
    )
    low, high = np.percentile(depth[::3, ::8, ::8], [1, 99])
    depth = np.clip((depth - low) / max(1e-6, high - low), 0, 1)
    if not np.isfinite(depth).all():
        raise ValueError("深度模型输出了非有限值")
    lon, lat = np.meshgrid(
        ((np.arange(1280) + 0.5) / 1280 - 0.5) * 2 * np.pi,
        ((np.arange(640) + 0.5) / 640 - 0.5) * np.pi,
    )
    rays = np.stack(
        [np.cos(lat) * np.sin(lon), np.sin(lat), np.cos(lat) * np.cos(lon)], axis=-1
    ).astype("f")
    inner = ~geom.invalid
    alpha = np.clip(
        cv.distanceTransform(inner.astype("uint8"), cv.DIST_L2, 5) / 18, 0, 1
    )
    outside = cv.distanceTransform(geom.invalid.astype("uint8"), cv.DIST_L2, 5)
    base_gate = np.clip((outside - opts["depth_margin"]) / opts["depth_feather"], 0, 1)
    # Plane fitting uses every sixth projected sample. Remap at those exact
    # positions instead of constructing two full panoramas for each fit.
    px, py = np.ascontiguousarray(geom.px[::6, ::6]), np.ascontiguousarray(geom.py[::6, ::6])
    fits, masks = [], []
    for dep, boxes in zip(depth, people, strict=True):
        erp = cv.remap(dep, px, py, cv.INTER_LINEAR, borderMode=cv.BORDER_REPLICATE)
        exclude = cv.remap(actor_mask(dep.shape, boxes, 6), px, py, cv.INTER_LINEAR,
                           borderMode=cv.BORDER_REPLICATE)
        valid = inner[::6, ::6] & (exclude == 0)
        ray = rays[::6, ::6][valid]
        vals = erp[valid]
        if len(vals) >= 4:
            fit = least_squares(
                lambda v: np.maximum(v[3], ray @ v[:3]) - vals,
                [0.0, 1.0, 0.15, 0.03],
                loss="soft_l1",
                f_scale=0.05,
                bounds=([-2, 0, -1, 0], [2, 3, 1, 0.7]),
            )
            fits.append(fit.x)
        else:
            fits.append(fits[-1] if fits else np.array([0.0, 1.0, 0.15, 0.03]))
        gate = base_gate
        if opts["protect_actors"]:
            edge = geom.guide(actor_mask(dep.shape, boxes, edge_only=True))
            clearance = cv.distanceTransform(1 - edge, cv.DIST_L2, 5)
            gate = gate * np.clip((clearance - 55) / 64, 0, 1)
        masks.append(gate.reshape(20, 32, 40, 32).min(axis=(1, 3)))
    # The video VAE packs the first frame separately, then groups of eight.
    gate = [masks[0]]
    for k in range(1, len(masks), 8):
        gate.extend([np.min(masks[k : k + 8], axis=0)] * len(masks[k : k + 8]))
    width = LTX_FRONT_WIDTH if opts.get("front_only") else 1280
    left = (1280 - width) // 2
    weights = np.stack(gate)[:, :, left // 32:(left + width) // 32]
    np.save(tmp / (uid + "-mask.npy"), weights.astype("float32"))
    with writer(tmp / (uid + "-depth.mkv"), width, 640, fps, LOSSLESS) as stream:
        for i, dep in enumerate(depth):
            radius = opts["depth_smoothing"]
            v = np.median(fits[max(0, i - radius) : i + radius + 1], axis=0)
            surface = np.clip(np.maximum(v[3], rays @ v[:3]), 0, 1)
            erp = cv.remap(
                dep, geom.px, geom.py, cv.INTER_LINEAR, borderMode=cv.BORDER_REPLICATE
            )
            image = np.rint(255 * (erp * alpha + surface * (1 - alpha))).astype("uint8")
            stream.write(cv.cvtColor(image[:, left:left + width], cv.COLOR_GRAY2BGR).tobytes())
