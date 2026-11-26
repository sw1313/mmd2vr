"""Isolated FLUX.2 stage generation and tiled Real-ESRGAN upscaling."""

import contextlib
import json
import os
import shutil
import socket
import subprocess
import sys
import time
from pathlib import Path

import requests

ROOT = Path(__file__).resolve().parent
LOCAL = ROOT / ".runtime/klein-panorama"
COMFY = Path(os.getenv("MMD_COMFY", LOCAL / "ComfyUI"))
PYTHON = os.getenv(
    "MMD_STAGE_PYTHON",
    str(LOCAL / "env/Scripts/python.exe") if os.name == "nt" else sys.executable,
)
WEIGHTS = os.getenv("MMD_STAGE_WEIGHTS")
UPSCALER = Path(WEIGHTS or LOCAL) / "RealESRGAN_x4plus_anime_6B.pth"


def h3_union_table():
    default = Path(os.getenv("NUNIF_HOME", ROOT / ".runtime/models")) / "h3" / "fl-union-table.npy"
    return Path(os.getenv("MMD_H3_UNION_TABLE", default))


def graph(filename, seed):
    def node(kind, **inputs):
        return {"class_type": kind, "inputs": inputs}

    return {
        "1": node("LoadImage", image=filename),
        "2": node(
            "UNETLoader",
            unet_name="flux-2-klein-base-4b.safetensors",
            weight_dtype="default",
        ),
        "3": node("VAELoader", vae_name="flux2-vae.safetensors"),
        "4": node(
            "CLIPLoader", clip_name="qwen_3_4b.safetensors", type="flux2", device="cpu"
        ),
        "5": node(
            "LoraLoaderModelOnly",
            model=["2", 0],
            lora_name="flux-2-klein-4B-360-erp-outpaint-lora_V1.safetensors",
            strength_model=0.9,
        ),
        "6": node(
            "CLIPTextEncode",
            clip=["4", 0],
            text="Fill the green spaces according to the image. Outpaint as a seamless 360 equirectangular panorama (2:1). Keep the horizon level. Match left and right edges. Continue the same empty stage, materials, perspective, colors and lighting visible in the reference. Scenery only.",
        ),
        "7": node(
            "CLIPTextEncode",
            clip=["4", 0],
            text="people, figures, text, duplicate objects, fluorescent green patches, distortion",
        ),
        "8": node("VAEEncode", pixels=["1", 0], vae=["3", 0]),
        "9": node("ReferenceLatent", conditioning=["6", 0], latent=["8", 0]),
        "10": node("ReferenceLatent", conditioning=["7", 0], latent=["8", 0]),
        "11": node(
            "KSampler",
            model=["5", 0],
            positive=["9", 0],
            negative=["10", 0],
            latent_image=["8", 0],
            seed=seed,
            steps=20,
            cfg=5,
            sampler_name="euler",
            scheduler="simple",
            denoise=1,
        ),
        "12": node("VAEDecode", samples=["11", 0], vae=["3", 0]),
        "13": node("SaveImage", images=["12", 0], filename_prefix="stage"),
    }


@contextlib.contextmanager
def server(work, *, model_dirs=None, vr360=False, offload_streams=None, h3=False, gguf_nodes=None):
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        port = sock.getsockname()[1]
    nodes = "h3_nodes" if h3 else "vr360_nodes"
    whitelist = [nodes] if vr360 else []
    if vr360:
        shutil.copytree(ROOT / "comfy_nodes" / nodes, work / "custom_nodes" / nodes)
        if h3 and (table := h3_union_table()).is_file():
            shutil.copyfile(table, work / "custom_nodes" / nodes / "fl-union-table.npy")
    if gguf_nodes is not None:
        shutil.copytree(gguf_nodes, work / "custom_nodes" / "mmd_gguf",
                        ignore=shutil.ignore_patterns(".git", "__pycache__"))
        whitelist.append("mmd_gguf")
    python = PYTHON
    if h3 and os.name == "nt":
        python = os.getenv("MMD_H3_PYTHON", str(ROOT / ".runtime/pano-compare/t8env/Scripts/python.exe"))
    user = work / "user"
    user.mkdir(exist_ok=True)
    args = [
        python,
        str(COMFY / "main.py"),
        "--listen",
        "127.0.0.1",
        "--port",
        str(port),
        "--input-directory",
        str(work),
        "--output-directory",
        str(work),
        "--user-directory",
        str(user),
        "--temp-directory",
        str(work / "temp"),
        *([] if vr360 else [
            "--disable-cuda-malloc", "--disable-dynamic-vram", "--reserve-vram", "1",
            *(["--cache-none"] if model_dirs else ["--lowvram", "--cache-lru", "8"]),
        ]),
        "--disable-all-custom-nodes",
        *(["--whitelist-custom-nodes", *whitelist] if whitelist else []),
        "--offline",
        "--disable-auto-launch",
    ]
    if h3:
        args.extend(["--disable-dynamic-vram", "--reserve-vram", "4.5"])
    if gguf_nodes is not None:
        args.extend(["--disable-dynamic-vram", "--reserve-vram", "3.0"])
    if WEIGHTS or model_dirs:
        args.extend(["--base-directory", str(work)])
        config = work / "models.yaml"
        config.write_text(
            json.dumps(
                {
                    "stage": model_dirs or {
                        k: WEIGHTS
                        for k in ("diffusion_models", "text_encoders", "vae", "loras")
                    }
                }
            ),
            encoding="utf8",
        )
        args.extend(["--extra-model-paths-config", str(config)])
    if vr360 and offload_streams is not None:
        args.extend(["--async-offload", str(offload_streams)])
    url = f"http://127.0.0.1:{port}"
    proc = subprocess.Popen(args, cwd=COMFY)
    try:
        deadline = time.monotonic() + 240
        while time.monotonic() < deadline:
            if proc.poll() is not None:
                raise RuntimeError("舞台生成服务启动失败，请检查上方模型和依赖日志")
            try:
                if requests.get(url + "/system_stats", timeout=2).ok:
                    break
            except requests.ConnectionError:
                pass
            time.sleep(1)
        else:
            raise TimeoutError("舞台生成服务启动超时")
        yield url, proc
    finally:
        if proc.poll() is None:
            proc.terminate()
            try:
                proc.wait(15)
            except subprocess.TimeoutExpired:
                proc.kill()
                proc.wait()


def generate(instance, ref, seed):
    url, proc = instance
    reply = requests.post(
        url + "/prompt", json={"prompt": graph(ref.name, seed)}, timeout=30
    )
    if not reply.ok:
        raise RuntimeError(reply.text[:3000])
    uid = reply.json()["prompt_id"]
    deadline = time.monotonic() + 3600
    while proc.poll() is None and time.monotonic() < deadline:
        history = requests.get(url + "/history/" + uid, timeout=20).json().get(uid)
        if history:
            if history["status"]["status_str"] != "success":
                raise RuntimeError(str(history["status"]["messages"])[-2500:])
            image = history["outputs"]["13"]["images"][0]
            return ref.parent / image["subfolder"] / image["filename"]
        time.sleep(2)
    raise RuntimeError("舞台生成服务已退出或单张背景生成超时")


def upscale(jobs):
    import numpy as np
    import torch
    from PIL import Image
    from spandrel import ModelLoader

    model = ModelLoader().load_from_file(str(UPSCALER)).to("cuda").half().eval()
    with torch.inference_mode():
        for source, dest, size in jobs:
            im = np.asarray(Image.open(source).convert("RGB"))
            h, w = im.shape[:2]
            pad, tile = 20, 192
            ext = np.pad(im, ((0, 0), (pad, pad), (0, 0)), mode="wrap")
            ext = np.pad(ext, ((pad, pad), (0, 0), (0, 0)), mode="reflect")
            out = np.empty((4 * h, 4 * w, 3), np.uint8)
            for y in range(0, h, tile):
                for x in range(0, w, tile):
                    dh, dw = min(tile, h - y), min(tile, w - x)
                    patch = np.ascontiguousarray(
                        ext[y : y + dh + 2 * pad, x : x + dw + 2 * pad]
                    )
                    tensor = (
                        torch.from_numpy(patch)
                        .permute(2, 0, 1)[None]
                        .to(device="cuda", dtype=torch.float16)
                        / 255
                    )
                    pred = model(tensor)[
                        0, :, 4 * pad : 4 * (pad + dh), 4 * pad : 4 * (pad + dw)
                    ]
                    out[4 * y : 4 * (y + dh), 4 * x : 4 * (x + dw)] = (
                        pred.clamp(0, 1)
                        .mul(255)
                        .round()
                        .byte()
                        .permute(1, 2, 0)
                        .cpu()
                        .numpy()
                    )
            Image.fromarray(out).resize(
                (2 * size, size), Image.Resampling.LANCZOS
            ).save(dest, compress_level=1)


if __name__ == "__main__":
    upscale(json.loads(Path(sys.argv[1]).read_text(encoding="utf8")))
