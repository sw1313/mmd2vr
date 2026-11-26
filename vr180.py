"""Project a full-SBS master onto the forward hemisphere and tag the HEVC MP4."""

import json
import math
import struct
import sys
from pathlib import Path

import numpy as np


def vertical_fov(width, height, horizontal):
    return math.degrees(
        2 * math.atan(math.tan(math.radians(horizontal / 2)) * height / width)
    )


def curvature_maps(
    directory, w, h, angle, eye, curvature, keep_center=False, mask=False, hard_mask=False,
    view_zoom=1.0,
):
    """Map delivery rays directly to the original screen and optional background."""
    half_fov = math.radians(angle / 2)
    fraction = curvature / 100
    if not curvature:
        half_height = math.tan(half_fov) * h / w
    elif keep_center:
        # Bend the original screen without shortening its arc or changing its height.
        radius = 1 / fraction
        arc = math.tan(half_fov) / radius
        half_fov = math.atan2(
            radius * math.sin(arc), 1 - 2 * radius * math.sin(arc / 2) ** 2
        )
    else:
        arc = half_fov * fraction
        # The midpoint stays at z=1. Keep both endpoints at ±half_fov.
        radius = math.sin(half_fov) / (
            2 * math.sin(arc / 2) * math.cos(half_fov - arc / 2)
        )
    theta = ((np.arange(eye) + 0.5) / eye - 0.5) * np.pi
    longitude = np.arctan2(np.sin(theta), view_zoom * np.cos(theta)) if view_zoom != 1 else theta
    vertical_scale = np.hypot(view_zoom * np.cos(theta), np.sin(theta)) if view_zoom != 1 else 1.0
    # Longitude covers only the intended field; clamp before solving the intersection.
    azimuth = np.clip(longitude, -half_fov, half_fov)
    if curvature:
        phi = azimuth + np.arcsin((1 / radius - 1) * np.sin(azimuth))
        distance = (1 - 2 * radius * np.sin(phi / 2) ** 2) / np.cos(azimuth)
        half_height = radius * arc * h / w
        screen_x = phi / arc
    else:
        distance = 1 / np.cos(azimuth)
        screen_x = np.tan(azimuth) / math.tan(half_fov)
    # remap takes integer sample locations. Sample a double-resolution Lanczos master.
    sx = np.clip(np.rint((screen_x + 1) * w - 0.5), 0, 2 * w - 1)
    paths = [directory / "curve_x.pgm", directory / "curve_y.pgm"]
    header = f"P5\n{2 * eye} {eye}\n65535\n".encode("ascii")
    from contextlib import ExitStack

    with ExitStack() as stack:
        xf, yf = [stack.enter_context(path.open("wb")) for path in paths]
        if mask:
            paths.append(directory / "curve_alpha.pgm")
            alpha = stack.enter_context(paths[-1].open("wb"))
            alpha.write(f"P5\n{2 * eye} {eye}\n255\n".encode("ascii"))
            if view_zoom != 1:
                paths.extend(directory / name for name in ("zoom_bg_x.pgm", "zoom_bg_y.pgm"))
                bx, by = [stack.enter_context(path.open("wb")) for path in paths[-2:]]
                bx.write(header)
                by.write(header)
                bg_x = np.clip(np.rint((longitude / np.pi + .5) * eye - .5), 0, eye - 1)
                bg_x = np.concatenate((bg_x, bg_x + eye)).astype(">u2")
        xf.write(header)
        yf.write(header)
        for start in range(0, eye, 64):
            elevation = theta[start : start + 64, None]
            v = np.tan(elevation) * distance / (half_height * vertical_scale)
            valid = (np.abs(longitude) <= half_fov) & (np.abs(v) <= 1)
            if mask:
                # A narrow seam inside the support keeps the original center fully opaque.
                margin = np.minimum(
                    (half_fov - np.abs(longitude)) * eye / np.pi,
                    (1 - np.abs(v)) * half_height * eye / np.pi,
                )
                alpha.write(
                    np.tile(
                        (255 * (valid if hard_mask else np.clip(margin / max(3, eye * 0.015), 0, 1))).astype(
                            "uint8"
                        ),
                        (1, 2),
                    ).tobytes()
                )
            sy = np.clip(np.rint((v + 1) * h - 0.5), 0, 2 * h - 1)
            x = np.concatenate(
                (np.where(valid, sx, 65535), np.where(valid, sx + 2 * w, 65535)), axis=1
            )
            y = np.tile(np.where(valid, sy, 65535), (1, 2))
            xf.write(x.astype(">u2").tobytes())
            yf.write(y.astype(">u2").tobytes())
            if mask and view_zoom != 1:
                latitude = np.arctan(np.tan(elevation) / vertical_scale)
                bg_y = np.clip(np.rint((latitude / np.pi + .5) * eye - .5), 0, eye - 1)
                bx.write(bg_x.tobytes() * len(elevation))
                by.write(np.tile(bg_y.astype(">u2"), (1, 2)).tobytes())
    return paths


def flat_projection(w, h, angle, eye):
    return (
        f"v360=input=flat:output=hequirect:in_stereo=sbs:out_stereo=sbs:"
        f"ih_fov={angle}:iv_fov={vertical_fov(w, h, angle):.8f}:"
        f"w={eye}:h={eye}:interp=lanczos:alpha_mask=1"
    )


def projection_args(
    w,
    h,
    angle,
    eye,
    rate,
    ten_bit,
    *,
    curvature=0,
    keep_center=False,
    work=None,
    map_input=2,
    stage_path=None,
    stereo_stage=False,
    detail_path=None,
    colorspace=None,
    view_zoom=1.0,
):
    alpha = "yuva420p10le" if ten_bit else "yuva420p"
    overlay = "yuv420p10" if ten_bit else "yuv420"
    background = "yuv420p10le" if ten_bit else "yuv420p"
    graph = (
        f"[0:v:0]format={alpha},{flat_projection(w, h, angle, eye)}[projected];"
        f"color=black:s={eye * 2}x{eye}:r={rate},format={background}[bg];"
        f"[bg][projected]overlay=shortest=1:format={overlay},setsar=1[out]"
    )
    inputs = []
    bg_filter = ""
    if stage_path:
        inputs.extend(["-i", str(stage_path)])
        bg_filter = f"[{map_input}:v]settb=1/({rate}),setpts=N,format={background},split[bl][br];[bl][br]hstack[bg];"
        if stereo_stage:
            bg_filter = f"[{map_input}:v]settb=1/({rate}),setpts=N,scale={2*eye}:{eye}:flags=lanczos,format={background}[bg];"
        if detail_path:
            inputs.extend(["-i", str(detail_path)])
            # Add the original 8-bit luma residual before converting the backdrop to the output pixel format.
            bg_filter = (
                f"[{map_input}:v]settb=1/({rate}),setpts=N,scale={2*eye}:{eye}:flags=lanczos,format=yuv420p[base];"
                f"[{map_input+1}:v]settb=1/({rate}),setpts=N,scale={eye}:{eye}:flags=bilinear,format=yuv420p,split[dl][dr];"
                "[dl][dr]hstack[detail];[base][detail]lut2=c0='clip(x+y-128,0,255)':c1=x:c2=x,"
                f"format={background}[bg];"
            )
            map_input += 1
        graph = graph.replace(
            f"color=black:s={eye * 2}x{eye}:r={rate},format={background}[bg];",
            bg_filter,
        )
        map_input += 1
    if curvature or view_zoom != 1:
        paths = curvature_maps(
            work, w, h, angle, eye, curvature, keep_center, bool(stage_path), stereo_stage,
            view_zoom=view_zoom,
        )
        for path in paths:
            inputs.extend(["-framerate", str(rate), "-i", str(path)])
        planar = "yuv444p10le" if ten_bit else "yuv444p"
        graph = (
            f"[0:v:0]scale={4 * w}:{2 * h}:flags=lanczos,format={planar}[master];"
            f"[master][{map_input}:v][{map_input + 1}:v]remap=fill=black,"
            f"format={background},setsar=1[out]"
        )
        if stage_path:
            if view_zoom != 1:
                bg_filter = bg_filter.removesuffix("[bg];") + (
                    f",format={planar}[unzoomed];"
                    f"[unzoomed][{map_input+3}:v][{map_input+4}:v]remap=fill=black,format={background}[bg];"
                )
            mask_input = f"[{map_input + 2}:v]"
            merge_alpha = f"[center]{mask_input}alphamerge[fg];"
            if ten_bit:
                # alphamerge accepts 8-bit color only. Join the original Main10 planes directly.
                merge_alpha = (
                    f"{mask_input}scale=in_range=full:out_range=full,format=gray10le,setsar=1,loop=loop=-1:size=1:start=0[mask];"
                    "[center][mask]mergeplanes=0x00010210:yuva420p10le[fg];"
                )
            graph = bg_filter + graph.replace(
                f"format={background},setsar=1[out]",
                f"format={background},setsar=1[center];{merge_alpha}"
                f"[bg][fg]overlay=shortest=1:format={overlay},setsar=1[out]",
            )
    if stage_path and colorspace in (
        "bt709",
        "bt470bg",
        "smpte170m",
        "bt2020nc",
        "smpte240m",
        "fcc",
    ):
        # overlay inherits its background color metadata. Keep the source YUV coefficients
        # so final encoding does not convert the already-projected foreground a second time.
        graph = graph.removesuffix("[out]") + f",setparams=colorspace={colorspace}[out]"
    return inputs + [
        "-filter_complex_threads",
        "12",
        "-filter_complex",
        graph,
        "-map",
        "[out]",
        *(["-r:v", str(rate), "-fps_mode", "cfr"] if stage_path else []),
        "-c:v",
        "hevc_nvenc",
        "-preset",
        "p5",
        "-rc",
        "vbr",
        "-cq",
        "18",
        "-b:v",
        "30M",
        "-maxrate",
        "45M",
        "-bufsize",
        "60M",
        "-pix_fmt",
        "p010le" if ten_bit else "yuv420p",
    ]


def contour_command(
    cmd, work, stereo, clip, w, h, angle, eye, rate, ten, curvature, metadata, strength, width,
    view_zoom=1.0,
):
    """Stream the projected planes through the exterior filter before the sole HEVC encode."""
    pixel_format = "yuv420p10le" if ten else "yuv420p"
    split = cmd.index("-c:v")
    color_args = []
    for flag, field in (("-colorspace", "color_space"), ("-color_range", "color_range"),
                        ("-color_trc", "color_transfer"), ("-color_primaries", "color_primaries")):
        value = metadata.get(field)
        if value not in (None, "unknown", "unspecified"):
            color_args.extend([flag, value])
    project = [*cmd[:split], "-pix_fmt", pixel_format, *color_args, "-f", "rawvideo", "pipe:1"]
    encode = [
        "ffmpeg", "-nostdin", "-hide_banner", "-loglevel", "error", "-stats", "-y",
        "-f", "rawvideo", "-pix_fmt", pixel_format, "-s", f"{2 * eye}x{eye}",
        "-framerate", str(rate), *color_args, "-i", "pipe:0", "-i", clip,
        "-map", "0:v:0", *cmd[split:],
    ]
    mask = work / "curve_alpha.pgm"
    make_mask = []
    if not curvature and view_zoom == 1:
        # Extract the same v360 alpha used by the flat overlay, including its edge pixels.
        alpha = "yuva420p10le" if ten else "yuva420p"
        make_mask = [
            "ffmpeg", "-nostdin", "-v", "error", "-y", "-i", stereo,
            "-vf", f"format={alpha},{flat_projection(w, h, angle, eye)},alphaextract",
            "-frames:v", "1", "-pix_fmt", "gray", mask,
        ]
    plan = work / "contour.json"
    plan.write_text(json.dumps(dict(
        project=project, encode=encode, make_mask=make_mask, mask=mask, eye=eye,
        ten_bit=ten, strength=strength, width=width,
    ), default=str), encoding="utf-8")
    return [sys.executable, Path(__file__).with_name("vr360_contour.py"), plan]


def box(kind, payload):
    return struct.pack(">I4s", 8 + len(payload), kind) + payload


def add_projection_metadata(path):
    # Modify only the terminal moov of our own fresh (non-faststart) MP4.
    # Skipping mdat also handles multi-GB jobs without loading media into memory.
    version = bytes(4)
    bounds = struct.pack(">IIII", 0, 0, 1 << 30, 1 << 30)
    proj = box(b"prhd", bytes(16)) + box(b"equi", version + bounds)
    tags = box(b"st3d", version + b"\x02") + box(
        b"sv3d", box(b"svhd", version + b"MMD2VR\x00") + box(b"proj", proj)
    )
    tracks = 0

    def edit(data):
        nonlocal tracks
        pos = 0
        result = bytearray()
        while pos < len(data):
            size, kind = struct.unpack_from(">I4s", data, pos)
            if size < 8 or pos + size > len(data):
                raise ValueError("MP4 元数据长度无效")
            payload = data[pos + 8 : pos + size]
            if kind in (b"trak", b"mdia", b"minf", b"stbl"):
                payload = edit(payload)
            elif kind == b"stsd":
                payload = payload[:8] + edit(payload[8:])
            elif kind == b"hvc1":
                payload += tags
                tracks += 1
            result.extend(box(kind, payload))
            pos += size
        return bytes(result)

    end = path.stat().st_size
    with path.open("r+b") as f:
        while f.tell() < end:
            pos = f.tell()
            size, kind = struct.unpack(">I4s", f.read(8))
            if size == 1:
                size = struct.unpack(">Q", f.read(8))[0]
            if size < f.tell() - pos or pos + size > end:
                raise ValueError("MP4 顶层结构无效")
            if kind == b"moov" and pos + size == end:
                tagged = box(kind, edit(f.read(end - f.tell())))
                if tracks != 1:
                    raise ValueError("未找到唯一的 HEVC 视频轨")
                f.seek(pos)
                f.write(tagged)
                return
            f.seek(pos + size)
    raise ValueError("无法写入 VR180 投影标记")
