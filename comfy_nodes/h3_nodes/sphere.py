"""One FL2VA denoising trajectory per window, optionally conditioned by H3 Fun Control."""
import contextlib
import json
import os
import time
from pathlib import Path
from types import MethodType, SimpleNamespace
from unittest.mock import patch

import comfy.sample
from comfy.nested_tensor import NestedTensor
import cv2 as cv
import folder_paths
import numpy as np
import torch

from .minimax_lowmem import minimax_attn_lowmem_forward, MiniMaxFFNChunkPatch
from .video_outpaint_plan import build_outpaint_plan
from .video_outpaint_decode import iter_decode_outpaint_shot


@contextlib.contextmanager
def union_condition(model, union, vae, pixels, cfg, shape, work, counts=None):
    from comfy.ldm.minimax.model import MiniMaxH3Model
    from comfy_extras.nodes_minimax_h3 import MiniMaxH3FunControlPatch

    net = union.model
    if not net.inpaint_post_norm or net.control_in_dim != 49:
        raise ValueError("H3 Union 权重与 FL2VA 外围重绘条件不匹配")
    size = cfg["size"]
    x0, y0, x1, y1 = cfg["patch"]
    rgb = torch.zeros(len(pixels), size, size, 3)
    rgb[:, y0:y1, x0:x1] = pixels
    hole = torch.from_numpy(np.load(work / "holes.npy")).expand(len(pixels), -1, -1)
    modified = model.clone()
    control = MiniMaxH3FunControlPatch(union, vae, None, hole, rgb.movedim(-1, 1),
                                      cfg["union_weight"], float("inf"), 0.)
    control.prepare_control_latent(shape)
    control.register(modified)

    # The compact FL2VA stores full adaLN parameters. Fun Control consumes the
    # eight FL2VA curve coefficients for the very same diffusion timestep.
    table = torch.from_numpy(np.load(Path(__file__).with_name("fl-union-table.npy")))
    basis = SimpleNamespace(use_adaln_curves=True, adaln_t_table=table)
    native_time = MiniMaxH3Model._time_embedding
    time_embedding = None
    def capture(this, timestep, device, dtype):
        nonlocal time_embedding
        time_embedding = native_time(basis, timestep, device, dtype)
        return native_time(this, timestep, device, dtype)

    init_stream, step = net.init_stream, net.step
    def init(h, z, layout, t):
        return init_stream(h, z, layout, time_embedding)

    def advance(i, c, t, *args, **kwargs):
        if counts is not None:
            counts[i] += 1
        return step(i, c, time_embedding, *args, **kwargs)

    with contextlib.ExitStack() as hooks:
        hooks.enter_context(patch.object(MiniMaxH3Model, "_time_embedding", capture))
        hooks.enter_context(patch.object(net, "init_stream", init))
        hooks.enter_context(patch.object(net, "step", advance))
        for block in net.control_blocks:
            hooks.enter_context(patch.object(block.attn, "forward", MethodType(minimax_attn_lowmem_forward, block.attn)))
            hooks.enter_context(patch.object(block.mlp, "forward", MiniMaxFFNChunkPatch(8, 4096).__get__(block.mlp, type(block.mlp))))
        yield modified


class H3SphereOutpaint:
    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {"window": ("STRING",), "model": ("MODEL",), "vae": ("VAE",),
                             "positive": ("CONDITIONING",), "negative": ("CONDITIONING",)},
                "optional": {"union": ("MODEL_PATCH",)}}
    RETURN_TYPES = ()
    OUTPUT_NODE = True
    FUNCTION = "run"
    CATEGORY = "MMD2VR"

    def run(self, window, model, vae, positive, negative, union=None):
        root = Path(folder_paths.get_output_directory())
        if Path(window).name != window or not window.startswith("shot-"):
            raise ValueError("无效的扩画窗口")
        work = root / window
        cfg = json.loads((work / "params.json").read_bytes())
        profile = os.environ.get("MMD_H3_PROFILE") == "1"
        def stamp():
            if profile:
                torch.cuda.synchronize()
                return time.perf_counter()
            return 0.
        if profile:
            torch.cuda.reset_peak_memory_stats()
        start = stamp()
        counts = [0] * 10 if profile and union is not None else None
        size = cfg["size"]
        x0, y0, x1, y1 = cfg["patch"]
        pixels = torch.from_numpy(np.load(work / "projected.npy")).float().div_(255)
        cov = torch.from_numpy(np.load(work / "coverage.npy"))[None, None]
        latent = torch.zeros(1, 24, cov.shape[2], size // 16, size // 16)
        latent[:, :, :, y0//16:y1//16, x0//16:x1//16] = vae.encode(pixels).cpu().float()
        latent.mul_(cov > 0)
        mask = (1 - cov).contiguous()
        audio_len = round(cfg["aligned"] * 40 / cfg["fps"])
        av = NestedTensor((latent, torch.zeros(1, 32, 2, audio_len)))
        noise_mask = NestedTensor((mask, torch.zeros(1, 1, 2, audio_len)))
        encoded = stamp()
        condition = (union_condition(model, union, vae, pixels, cfg, tuple(latent.shape), work, counts)
                     if union is not None else contextlib.nullcontext(model))
        with condition as conditioned:
            prepared = stamp()
            sample = comfy.sample.sample(conditioned, comfy.sample.prepare_noise(av, cfg["seed"]),
                                         cfg["steps"], cfg["cfg"], cfg["sampler"], "simple",
                                         positive, negative, av, noise_mask=noise_mask, seed=cfg["seed"])
            sampled = stamp()
        generated, _ = sample.unbind()
        z = torch.where(mask == 0, latent, generated.cpu()).contiguous()
        plan = build_outpaint_plan(source_sha256=cfg["sha"], width=size, height=size,
                                   frame_count=max(6, cfg["n"]), aspect="custom",
                                   generation_megapixels=size**2/1e6, window_frames=90)
        for first, frames, _ in iter_decode_outpaint_shot(vae, lambda a, b: z[:, :, a:b], plan):
            for i, rgb in enumerate((frames.clamp(0, 1)*255).round().byte().numpy()):
                if first+i >= cfg["n"]:
                    break
                if not cv.imwrite(str(work / f"{first+i:03}.png"), cv.cvtColor(rgb, cv.COLOR_RGB2BGR),
                                  [cv.IMWRITE_PNG_COMPRESSION, cfg["compression"]]):
                    raise OSError("无法写入 H3 解码帧")
        if profile:
            print("H3_PROFILE " + json.dumps(dict(window=window, frames=cfg["n"], aligned=cfg["aligned"],
                  cfg=cfg["cfg"], steps=cfg["steps"], encode=encoded-start, control=prepared-encoded,
                  sample=sampled-prepared, decode_and_write=stamp()-sampled, union_calls=counts,
                  peak_allocated=torch.cuda.max_memory_allocated(), peak_reserved=torch.cuda.max_memory_reserved())), flush=True)
        return {}
