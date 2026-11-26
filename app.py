import io
import json
import os
import re
import subprocess
import sys
import time
from functools import partial
from html import escape
from pathlib import Path

os.environ.setdefault("GRADIO_ANALYTICS_ENABLED", "False")
import gradio as gr
from PIL import Image

import engine as e
from vr360_options import H3_CHECKPOINTS, H3_FIELDS, H3_INFO, H3_PROMPT, H3_NEGATIVE, LTX_ONLY
from vr360_options import available_model_choices
from vr360_options import BASE_FIELDS, CHOICES, CONTROLS, CONTROL_INFO, DEFAULT_PROMPT, GROUPS, SPEED_PRESETS
from vr360_options import DEFAULT_VIEW_ZOOM, LTX_DEFAULTS, UNION_LABELS, VIEW_ZOOM_RANGE, syfe_union_path

STATUS = {
    "queued": "排队中",
    "running": "处理中",
    "done": "已完成",
    "failed": "失败",
    "cancelled": "已取消",
    "interrupted": "已中断",
}

GENERATION_PROFILE_CSS = """
#generation-profile {
    --profile-surface: var(--background-fill-primary);
    --profile-border: var(--border-color-primary);
    --profile-muted: color-mix(in srgb, var(--body-text-color) 76%, var(--profile-surface));
    max-width: 66rem;
    width: 100%;
    box-sizing: border-box;
    background: var(--profile-surface);
    border: 1px solid var(--profile-border);
    border-radius: 14px;
    padding: 16px;
    gap: 12px;
}
#generation-profile .profile-heading {
    display: flex;
    align-items: flex-start;
    justify-content: space-between;
    gap: 12px;
    flex-wrap: wrap;
}
#generation-profile h3, #generation-profile h4 {
    color: var(--body-text-color);
    margin: 0 0 4px;
    line-height: 1.35;
}
#generation-profile h3 { font-size: 1.05rem; font-weight: 650; }
#generation-profile h4 { font-size: 1rem; font-weight: 650; }
#generation-profile p {
    color: var(--profile-muted, var(--body-text-color));
    font-size: .875rem;
    line-height: 1.55;
    margin: 0;
}
#generation-profile .profile-state {
    border: 1px solid var(--profile-border);
    background: var(--background-fill-secondary);
    color: var(--body-text-color);
    border-radius: 6px;
    padding: 3px 8px;
    font-size: .8rem;
    font-weight: 600;
}
#generation-profile .profile-state[data-matched="true"] {
    border-color: var(--color-accent);
    border-color: color-mix(in srgb, var(--color-accent) 40%, var(--profile-border));
    background: color-mix(in srgb, var(--color-accent) 8%, var(--profile-surface));
}
#generation-profile .profile-card {
    background: var(--background-fill-secondary);
    border: 1px solid var(--profile-border);
    border-radius: 10px;
    padding: 12px;
    gap: 10px;
}
#generation-profile .profile-card:has(button.primary) {
    border-color: var(--color-accent);
    background: color-mix(in srgb, var(--color-accent) 6%, var(--profile-surface));
}
#generation-profile .profile-card button { width: 100%; }
#generation-profile button.primary {
    border-color: var(--body-text-color);
    background: var(--body-text-color);
    color: var(--profile-surface);
}
#generation-profile button.primary:hover {
    background: color-mix(in srgb, var(--body-text-color) 87%, var(--profile-surface));
}
#generation-profile .profile-union { font-weight: 600; color: var(--body-text-color); }
#generation-profile-actions { justify-content: flex-end; align-items: center; gap: 10px; }
#generation-profile-actions button { flex: 0 1 auto; min-width: 0; padding-inline: 12px; }
#generation-profile-reset {
    background: transparent;
    border-color: transparent;
    color: var(--profile-muted, var(--body-text-color));
}
#generation-profile-reset:hover {
    background: var(--background-fill-secondary);
    color: var(--body-text-color);
}
@media (max-width: 560px) {
    #generation-profile { padding: 12px; }
    #generation-profile-actions { justify-content: flex-start; }
}
"""


def scan_directory(folder, *, base=None):
    base = (base or e.INPUT).resolve(strict=True)
    if not folder.is_dir():
        raise gr.Error("目录不存在")
    rel = folder.relative_to(base)
    dirs = [("整个输入根目录", ".")]
    for parent in [*reversed(rel.parents), rel]:
        if parent.parts:
            dirs.append((parent.as_posix(), parent.as_posix()))
    files = []
    pending = [folder]
    skipped = 0
    while pending:
        current = pending.pop()
        try:
            with os.scandir(current) as entries:
                for entry in entries:
                    if entry.name in ("@eaDir", "#recycle", "@Recycle"):
                        continue
                    path = Path(entry.path)
                    if entry.is_dir(follow_symlinks=False) and not entry.is_junction():
                        if current == folder:
                            child = path.relative_to(base).as_posix()
                            dirs.append((child, child))
                        pending.append(path)
                    elif (
                        path.suffix.lower() in e.EXT
                        and entry.is_file(follow_symlinks=False)
                    ):
                        files.append([path.relative_to(folder).as_posix()])
        except (PermissionError, FileNotFoundError) as exc:
            if current == folder:
                raise gr.Error(f"读取目录失败：{exc}") from exc
            skipped += 1
        except OSError as exc:
            raise gr.Error(f"读取目录失败：{exc}") from exc
    if skipped:
        gr.Warning(f"有 {skipped} 个子目录无法读取，已列出其余视频。")
    dirs.sort(key=lambda choice: (len(Path(choice[1]).parts), choice[1].casefold()))
    files.sort(key=lambda row: row[0].casefold())
    return dirs, files


def scope_folder(root, relative):
    if not relative:
        raise ValueError("请选择本次处理目录")
    folder = e.media_path(root / relative, e.INPUT)
    if not root.is_dir() or not folder.is_dir() or not folder.is_relative_to(root):
        raise ValueError("请在输入根目录内选择本次处理目录")
    return folder


def browse(relative=".", input_root=None, *, remember_choice=False):
    try:
        root = e.media_path(input_root or e.INPUT, e.INPUT)
        folder = scope_folder(root, relative)
    except (OSError, ValueError) as exc:
        raise gr.Error(str(exc)) from exc
    dirs, files = scan_directory(folder, base=root)
    nav = gr.Dropdown(choices=dirs, value=folder.relative_to(root).as_posix())
    display = e.display_path(root, e.INPUT)
    if remember_choice:
        remember(input_root=display, current=folder.relative_to(root).as_posix(), single_video="")
    label = f"视频文件（{len(files)}）" if files else "此目录及其子目录中没有视频"
    return (
        nav,
        gr.Dataframe(value=files, headers=["视频路径（相对于当前目录）"], label=label),
        display,
        None,
        "",
    )


def select_video(input_root, folder, event: gr.SelectData):
    if not event.value:
        return ""
    scope = scope_folder(e.media_path(input_root, e.INPUT), folder)
    path = e.display_path(e.source(scope / event.value), e.INPUT)
    remember(single_video=path)
    return path


def preview(path, seconds):
    result = subprocess.run(
        [
            "ffmpeg",
            "-nostdin",
            "-v",
            "error",
            "-ss",
            str(float(seconds)),
            "-i",
            str(path),
            "-map",
            "0:V:0",
            "-frames:v",
            "1",
            "-vf",
            "scale='min(1200,iw)':-2",
            "-f",
            "image2pipe",
            "-vcodec",
            "mjpeg",
            "pipe:1",
        ],
        capture_output=True,
        timeout=55,
        check=False,
    )
    if result.returncode or not result.stdout:
        raise gr.Error("未能读取此时间点的画面")
    with Image.open(io.BytesIO(result.stdout)) as im:
        return im.copy()


def preview_source(rel):
    if not rel:
        raise gr.Error("请先选择视频")
    if e.media_path(rel, e.INPUT).is_dir():
        raise gr.Error("请点击列表中的一个视频进行预览")
    return preview(e.source(rel), 0)


def dimensions(rel, width, mode):
    if not rel:
        return ""
    try:
        path = e.media_path(rel, e.INPUT)
        if path.is_dir():
            return "将递归提交此目录中的视频，每个视频单独计算输出尺寸。"
        path = e.source(path)
    except (ValueError, OSError):
        return ""
    stream = e.video(e.probe(path))
    size = e.output_size(stream, int(width), mode)
    w, h = size.packed
    return (
        f"片源 {stream['width']}×{stream['height']}　·　"
        f"每眼 {size.eye_width}×{size.eye_height}　·　成片 {w}×{h}"
    )


def submission_paths(rel, listing_root, single_video, output_path, keep_folders):
    root = e.media_path(rel, e.INPUT)
    if root.is_dir():
        selected = scope_folder(root, listing_root)
        if single_video:
            video = e.source(single_video)
            if not video.is_relative_to(selected):
                raise ValueError("单个视频必须位于本次处理目录内")
            selected = video
    else:
        # Older API clients submit the full video path in the first argument.
        selected = e.source(root)
        root = e.INPUT.resolve()
    dest = e.media_path(output_path or e.OUTPUT, e.OUTPUT)
    if keep_folders:
        parent = selected if selected.is_dir() else selected.parent
        dest = dest / parent.relative_to(root)
    return root, selected, dest


def path_preview(rel, listing_root, single_video, output_path, keep_folders):
    try:
        _, selected, dest = submission_paths(rel, listing_root, single_video, output_path, keep_folders)
    except (OSError, ValueError) as exc:
        return str(exc)
    scope = "递归转制目录" if selected.is_dir() else "仅转制此视频"
    return f"{scope}　{e.display_path(selected, e.INPUT)}\n成片保存目录　{e.display_path(dest, e.OUTPUT)}"


def submit(
    rel,
    level,
    width,
    model,
    mode=e.CINEMA,
    curvature=0,
    keep_center=False,
    output_path=None,
    keep_folders=True,
    listing_root=".",
    auto_stage=False,
    stage_method=None,
    stage_preset=None,
    empty_only=False,
    single_video=None,
    fast_closeup=False,
    match_stage=True, seam_luma=True, generation_fps=24, seed=726391,
    observed_edges=True, anime_detail=True,
    union_enabled=True,
    union_weight=1.0,
    depth_attention=1.0,
    texture_enabled=True,
    texture_mix=0.9,
    texture_radius=4,
    texture_sigma1=32,
    texture_sigma2=28,
    texture_space=5,
    finish_feather=12,
    palette_enabled=False,
    palette_strength=0.85,
    outpaint_checkpoint='360vroutpaint_v2_step09000.safetensors',
    outpaint_weight=1.0,
    source_strength=1.0,
    prompt=DEFAULT_PROMPT,
    sampler='euler_ancestral',
    steps=8,
    cfg=1.0,
    window_frames=121,
    window_overlap=8,
    split_shots=True,
    seed_per_shot=True,
    vae_tiled=True,
    vae_tile=512,
    depth_enabled=True,
    depth_strength=1.0,
    depth_size=392,
    depth_smoothing=2,
    protect_actors=True,
    person_threshold=0.307,
    depth_margin=12,
    depth_feather=40,
    texture_passes=2,
    seam_strength=1.0,
    observed_strength=1.0,
    donor_radius=65,
    donor_count=12,
    detail_strength=1.0,
    detail_fps=10,
    contour_enabled=True,
    contour_strength=1.0,
    contour_width=64,
    source_mask_enabled=CONTROLS["source_mask_enabled"][1],
    source_mask_inside=1.0,
    source_mask_outside=0.10,
    source_attention=1.0,
    detector_cuda=True,
    offload_streams=2,
    joint_audio=True,
    iw3_batch=1,
    png_compression=0,
    guide_extension_enabled=True,
    guide_extension_width=40,
    guide_extension_margin=15,
    guide_extension_contrast=45,
    guide_extension_blur=3.0,
    outpaint_model="ltx2", h3_size=512, h3_union=True, h3_strength=1.0, h3_steps=20,
    h3_cfg=1.25, h3_sampler="res_multistep", h3_prompt=H3_PROMPT, h3_negative=H3_NEGATIVE,
    h3_window=141, h3_overlap=8,
    front_only=CONTROLS["front_only"][1], view_zoom=DEFAULT_VIEW_ZOOM, union_variant="official",
):
    if not rel:
        raise gr.Error("请选择视频或素材目录")
    if (
        model == "VDA_Metric_L"
        and not (
            e.MODELS
            / "iw3/pretrained_models/hub/checkpoints/metric_video_depth_anything_vitl.pth"
        ).is_file()
    ):
        raise gr.Error("当前 worker 尚未配置 VDA_Metric_L 权重")
    try:
        root, selected, dest = submission_paths(rel, listing_root, single_video, output_path, keep_folders)
        folder = selected.is_dir()
        if folder:
            _, files = scan_directory(selected)
            if not files:
                raise gr.Error("此目录及其子目录中没有支持的视频文件")
            paths = [selected / filename for (filename,) in files]
        else:
            paths = [selected]
        enqueue = e.enqueue_batch if folder else e.enqueue_many
        jobs = enqueue(
            paths,
            profile=level,
            width=int(width),
            model=model,
            mode=mode,
            curvature=curvature,
            keep_center=keep_center,
            view_zoom=view_zoom,
            output_path=output_path,
            keep_folders=keep_folders,
            listing_root=root,
            auto_stage=auto_stage,
            stage_method=stage_method,
            stage_preset=None if stage_method else stage_preset,
            empty_only=empty_only,
            fast_closeup=fast_closeup, match_stage=match_stage, seam_luma=seam_luma,
            generation_fps=generation_fps, seed=seed, observed_edges=observed_edges, anime_detail=anime_detail,
            union_enabled=union_enabled,
            union_weight=union_weight,
            union_variant=union_variant,
            depth_attention=depth_attention,
            texture_enabled=texture_enabled,
            texture_mix=texture_mix,
            texture_radius=texture_radius,
            texture_sigma1=texture_sigma1,
            texture_sigma2=texture_sigma2,
            texture_space=texture_space,
            finish_feather=finish_feather,
            palette_enabled=palette_enabled,
            palette_strength=palette_strength,
            outpaint_checkpoint=outpaint_checkpoint,
            outpaint_weight=outpaint_weight,
            source_strength=source_strength,
            prompt=prompt,
            sampler=sampler,
            steps=steps,
            cfg=cfg,
            window_frames=window_frames,
            window_overlap=window_overlap,
            split_shots=split_shots,
            seed_per_shot=seed_per_shot,
            vae_tiled=vae_tiled,
            vae_tile=vae_tile,
            depth_enabled=depth_enabled,
            depth_strength=depth_strength,
            depth_size=depth_size,
            depth_smoothing=depth_smoothing,
            protect_actors=protect_actors,
            person_threshold=person_threshold,
            depth_margin=depth_margin,
            depth_feather=depth_feather,
            texture_passes=texture_passes,
            seam_strength=seam_strength,
            observed_strength=observed_strength,
            donor_radius=donor_radius,
            donor_count=donor_count,
            detail_strength=detail_strength,
            detail_fps=detail_fps,
            contour_enabled=contour_enabled,
            contour_strength=contour_strength,
            contour_width=contour_width,
            source_mask_enabled=source_mask_enabled,
            source_mask_inside=source_mask_inside,
            source_mask_outside=source_mask_outside,
            source_attention=source_attention,
            detector_cuda=detector_cuda,
            offload_streams=offload_streams,
            joint_audio=joint_audio,
            iw3_batch=iw3_batch,
            png_compression=png_compression,
            guide_extension_enabled=guide_extension_enabled,
            guide_extension_width=guide_extension_width,
            guide_extension_margin=guide_extension_margin,
            guide_extension_contrast=guide_extension_contrast,
            guide_extension_blur=guide_extension_blur,
            front_only=front_only,
            outpaint_model=outpaint_model, h3_size=h3_size, h3_union=h3_union,
            h3_strength=h3_strength, h3_steps=h3_steps, h3_cfg=h3_cfg, h3_sampler=h3_sampler,
            h3_prompt=h3_prompt, h3_negative=h3_negative, h3_window=h3_window, h3_overlap=h3_overlap,
        )
    except (TypeError, ValueError, OSError) as exc:
        raise gr.Error(str(exc)) from exc
    if folder:
        keys, queued, skipped = jobs
        summary = f"目录中共 {len(paths)} 个视频，本次入队 {queued} 个，已有成片跳过 {skipped} 个，已有任务 {len(keys) - queued} 个。"
        gr.Info(summary)
        return f"{summary}\n成片保存目录　{e.display_path(dest, e.OUTPUT)}", keys[0] if keys else None
    keys, _ = jobs
    key = keys[0]
    spec = json.loads(get_job(key)["spec"])
    dest = e.display_path(e.job_output_folder(spec), e.OUTPUT)
    return f"任务 {key}\n输出目录　{dest}", key


def speed_preset_values(name):
    return tuple(SPEED_PRESETS[name].values())


def mode_label(spec):
    mode = spec.get("mode", e.CINEMA)
    if mode == e.CINEMA:
        return mode
    label = f"{mode} · 曲率 {spec.get('curvature', 0)}%"
    if spec.get("view_zoom", 1.0) != 1.0:
        label += f" · 正前方 {spec['view_zoom']:.2f}×"
    if spec.get("curvature"):
        label += " · " + (
            "保持中心人物大小" if spec.get("keep_center") else "固定水平视场"
        )
    if spec.get("stage_version"):
        if spec.get("stage_preset"):
            label += " · " + spec["stage_preset"]
        label += " · " + {
            e.VR360_A: {"h3": "外围扩画", "ltx25": "LTX-2.5 VR360 A"}.get(
                spec.get("vr360", {}).get("outpaint_model"), "LTX-2.3 VR360 A"
            ),
            e.CONTEXT_STAGE: "含角色扩画",
            e.REFERENCE_STAGE: "仅用空舞台",
        }.get(spec["stage_version"], "固定舞台扩画")
        if spec.get("vr360"):
            opts = spec["vr360"]
            label += f" · 外围 {opts['generation_fps']} fps · 种子 {opts['seed']}"
            if opts.get("outpaint_model") == "h3":
                quant = "NVFP4 HQ" if opts.get("h3_checkpoint") == H3_CHECKPOINTS["h3_hq"] else "NVFP4"
                label += f" · H3 FL2VA {quant} · {opts['h3_size']}×{opts['h3_size']} · {opts['h3_steps']} 步"
                label += f" · H3 Union 2.0 ×{opts['h3_strength']:g}" if opts["h3_union"] else " · 无 Union"
            elif "steps" in opts:
                label += f" · {opts['steps']} 步"
                if opts.get("front_only"):
                    label += " · 前半球"
            if opts.get("detector_cuda"):
                label += " · GPU 人物检测"
            if opts.get("union_enabled"):
                union_name = UNION_LABELS[opts.get("union_variant", "official")]
                label += f" · {union_name} · 权重 {opts['union_weight']:g} / 深度 {opts['depth_attention']:g}"
            if opts.get("texture_enabled"):
                label += f" · 纹理 {opts['texture_mix']:g}"
            if opts.get("contour_enabled"):
                label += f" · 二维接缝 {opts['contour_strength']:g} / {opts['contour_width']} px"
            if opts.get("palette_enabled"):
                label += f" · 配色 {opts['palette_strength']:g}"
            for key, title in (("match_stage", "舞台匹配"), ("seam_luma", "接缝校正"), ("observed_edges", "原片边缘补全"), ("anime_detail", "外围细节")):
                if opts[key]:
                    label += " · " + title
        if spec.get("vace_speed") == "balanced-v1":
            label += " · 均衡提速"
    return label


def queue():
    result = []
    for r in e.rows():
        s = json.loads(r["spec"])
        result.append(
            [
                r["id"],
                STATUS[r["status"]],
                r["stage"],
                s["pilot"] or Path(s["rel"]).stem,
                f"{e.divergence(s['profile']):.1f}",
                mode_label(s),
                "整片"
                if not s["length"]
                else f"{s['start']:g}–{s['start'] + s['length']:g} s",
            ]
        )
    return result


def selected(table, event: gr.SelectData):
    return str(table[event.index[0]][0])


def get_job(key):
    with e.db() as c:
        row = c.execute("SELECT * FROM jobs WHERE id=?", (key,)).fetchone()
    if not row:
        raise gr.Error("请先在列表中点选任务")
    return row


def detail(key):
    row = get_job(key)
    log = e.STATE / "logs" / f"{row['id']}.log"
    tail = ""
    if log.is_file():
        with log.open("rb") as f:
            f.seek(max(0, log.stat().st_size - 3500))
            tail = f.read().decode("utf8", errors="replace").replace("\r", "\n")
    spec = json.loads(row["spec"])
    source_name = str(Path(e.INPUT_DISPLAY) / spec["rel"])
    folder = e.display_path(e.job_output_folder(spec), e.OUTPUT)
    result = (
        str(Path(folder) / Path(row["result"]).name) if row["result"] else row["error"]
    )
    decisions = e.STATE / "logs" / f"{row['id']}.stage.json"
    if spec.get("stage_version") and decisions.is_file():
        segments = json.loads(decisions.read_text(encoding="utf8"))
        result += "\n" + "\n".join(
            f"{seg['start']:.2f}–{seg['end']:.2f}s　{seg['action']}　{seg['plate'] or ''}　{seg['reason']}"
            for seg in segments
        )
    return (
        f"{STATUS[row['status']]} · {row['stage']}\n输入　{source_name}\n输出目录　{folder}\n{spec['model']} · 立体强度 {e.divergence(spec['profile']):.1f} · {mode_label(spec)} · 分辨率 {spec['width'] or '自动'}\n{result}",
        tail,
    )


def preview_job(key):
    row = get_job(key)
    if row["status"] != "done":
        raise gr.Error("完成后可预览")
    return preview(Path(row["result"]), 0.5)


def stereo_output_progress(key, stage_started, anime_detail):
    # A retry may leave an older work directory for the same job.
    for movie in e.STATE.glob(f"work-{key}-*/stage.mkv"):
        try:
            stereo = movie.stat()
            if stereo.st_mtime < stage_started:
                continue
            outputs = [("双眼视频", stereo)]
            if anime_detail:
                detail = movie.with_name("vr360-detail.mkv")
                if detail.is_file():
                    outputs.append(("动漫细节", detail.stat()))
        except FileNotFoundError:
            continue
        sizes = "，".join(f"{name} {stat.st_size / 2**20:.1f} MiB" for name, stat in outputs)
        age = max(0, int(time.time() - max(stat.st_mtime for _, stat in outputs)))
        return f"<p>扩画帧已生成，正在进行双眼边缘合成。已写入{sizes}，最近写入在 {age} 秒前。</p>"
    return None


def live_progress():
    with e.db() as conn:
        row = conn.execute("""SELECT j.*, p.step, p.frames, p.log_offset FROM jobs j
            LEFT JOIN job_progress p ON p.job=j.id ORDER BY
            CASE j.status WHEN 'running' THEN 0 WHEN 'queued' THEN 1 ELSE 2 END,
            CASE WHEN j.status='queued' THEN j.created END ASC, j.updated DESC LIMIT 1""").fetchone()
        waiting = conn.execute(
            "SELECT count(*) FROM jobs WHERE status='queued'"
        ).fetchone()[0]
    if row is None:
        return "<p>队列空闲</p>", ""
    key, state = row["id"], row["status"]
    path = e.STATE / "logs" / f"{key}.log"
    tail = stage_log = ""
    if path.is_file():
        with path.open("rb") as f:
            end = f.seek(0, 2)
            start = max(0, end - 8000)
            f.seek(start)
            data = f.read()
        tail = data.decode("utf8", errors="replace").replace("\r", "\n").strip()
        if row["log_offset"] is not None:
            stage_log = data[max(0, row["log_offset"] - start) :].decode(
                "utf8", errors="replace"
            )
    spec = json.loads(row["spec"])
    steps = 6 if spec.get("stage_version") else 5
    completed = steps if state == "done" else max(0, (row["step"] or 1) - 1)
    if state == "queued":
        completed, tail = 0, ""
    stage = row["stage"] or STATUS[state]
    title = "正在取消" if state == "running" and row["cancel"] else STATUS[state]
    lines = [
        f"<p><strong>{title}</strong>　{escape(spec['rel'])}　·　排队 {waiting} 个</p>",
        f"<p>任务 <code>{key}</code>　·　{escape(stage)}</p>",
        f'<label>已完成阶段 {completed}/{steps} <progress value="{completed}" max="{steps}"></progress></label>',
    ]
    if state == "running" and row["frames"]:
        label = "阶段进度"
        unit = "帧"
        total = row["frames"]
        if row["step"] == 2:
            matches = list(re.finditer(r"([^\r\n]*?)\b(\d+)/(\d+)\s*\[", stage_log))
            current = int(matches[-1][2]) if matches else None
            if matches:
                total = int(matches[-1][3])
                label = (
                    "镜头检测"
                    if "Scene Boundary Detection" in matches[-1][1]
                    else "双眼合成"
                )
        else:
            matches = re.findall(r"\bframe=\s*(\d+)", stage_log)
            current = int(matches[-1]) if matches else None
            if spec.get("stage_version") == e.VR360_A and row["step"] == 3:
                activity = stereo_output_progress(key, row["updated"], spec.get("vr360", {}).get("anime_detail", False))
                if activity:
                    lines.append(activity)
                    current = None
                elif current is not None:
                    label = "已生成扩画帧"
            elif spec.get("stage_version") == e.CONTEXT_STAGE and row["step"] == 3:
                windows = list(re.finditer(r"VACE 窗口 (\d+) · 已完成 (\d+) 帧", stage_log))
                if windows:
                    window = windows[-1]
                    current = int(window[2])
                    label = f"第 {window[1]} 个窗口 · 已合成"
                    count = 16 if spec.get("vace_speed") == "balanced-v1" else 30
                    diffusion = re.findall(rf"\b(\d+)/{count}\s*\[", stage_log[window.end():])
                    if diffusion:
                        label += f" · 本窗口 {diffusion[-1]}/{count} 步"
            elif spec.get("stage_version") and row["step"] == 3 and current is None:
                decisions = e.STATE / "logs" / f"{key}.stage.json"
                if decisions.is_file():
                    segments = json.loads(decisions.read_text(encoding="utf8"))
                    plates = [s["plate"] for s in segments if s["action"] == "生成"]
                    lines.append(
                        f"<p>共 {len(segments)} 段，共用 {len({s['plate'] for s in segments if s['plate']})} 张舞台背景，本次计划生成 {len(plates)} 张</p>"
                    )
                    active = re.findall(r"生成背景 (\w+)，参考帧", stage_log)
                    if active and active[-1] in plates:
                        label = (
                            f"第 {plates.index(active[-1]) + 1}/{len(plates)} 张背景"
                        )
                if "分块超分" in stage_log:
                    lines.append("<p>正在放大舞台背景</p>")
                else:
                    diffusion = re.findall(
                        r"\b(\d+)/(20)\s*\[", stage_log.rsplit("生成背景 ", 1)[-1]
                    )
                    if diffusion:
                        current, total = map(int, diffusion[-1])
                        unit = "步"
        if current is not None and total:
            pct = min(100, round(100 * current / total))
            lines.append(
                f"<p><label>{label}　{current}/{total} {unit} · {pct}% "
                f'<progress value="{pct}" max="100"></progress></label></p>'
            )
        elif not (spec.get("stage_version") and row["step"] == 3):
            lines.append(f"<p>{label}　等候首个帧数报告</p>")
    if row["error"]:
        lines.append(f"<p>{escape(row['error'])}</p>")
    if state == "cancelled" and row["step"] and row["stage"] != "排队时取消":
        lines.append("<p>转制已停止，任务进程已退出。</p>")
    if state == "done" and row["result"]:
        saved = Path(e.OUTPUT_DISPLAY) / Path(row["result"]).relative_to(
            e.OUTPUT.resolve()
        )
        lines.append(f"<p>成片　{escape(str(saved))}</p>")
    return "".join(lines), tail


def initial_tab():
    with e.db() as conn:
        active = conn.execute(
            "SELECT 1 FROM jobs WHERE status IN ('queued','running') LIMIT 1"
        ).fetchone()
    return gr.Tabs(selected="monitor") if active else gr.skip()


def remember(**changes):
    with e.db() as conn:
        conn.executemany(
            """INSERT INTO form_preferences(name, value) VALUES(?, ?)
               ON CONFLICT(name) DO UPDATE SET value=excluded.value
               WHERE value != excluded.value""",
            [
                (name, json.dumps(value, ensure_ascii=False))
                for name, value in changes.items()
                if value is not None
            ],
        )


def remember_field(name, value, previous):
    # Restoring a form also fires Gradio change events. Only persist actual edits.
    if previous is None or value is None or value == previous.get(name):
        return gr.skip()
    remember(**{name: value})
    return {**previous, name: value}


def reset_generation_settings(defaults, selection):
    if selection in ("ltx2", "ltx25"):
        defaults = {**defaults, **LTX_DEFAULTS, "outpaint_model": selection}
    remember(**defaults)
    return tuple(defaults.values())


def open_input(value):
    return browse(".", value, remember_choice=True)


def open_scope(relative, input_root=None):
    return browse(relative, input_root, remember_choice=True)


def open_output(value):
    try:
        folder = e.media_path(value, e.OUTPUT)
        existing = next(p for p in (folder, *folder.parents) if p.exists())
        if not existing.is_dir():
            raise ValueError("输出根目录必须是目录")
        path = e.display_path(folder, e.OUTPUT)
    except (OSError, ValueError) as exc:
        raise gr.Error(str(exc)) from exc
    remember(output=path)
    return path


def restore_form(defaults):
    with e.db() as conn:
        saved = {
            k: json.loads(v)
            for k, v in conn.execute("SELECT name,value FROM form_preferences")
        }
    current = saved.get("current", ".")
    try:
        nav, files, root, _, _ = browse(current, saved.get("input_root"))
    except gr.Error:
        # A vanished saved folder must not turn a narrow batch into a library-wide job.
        root = saved.get("input_root") or e.INPUT_DISPLAY
        current = None
        nav = gr.Dropdown(choices=[("整个输入根目录", ".")], value=None)
        files = gr.Dataframe(value=[], headers=["视频路径"], label="请重新选择处理目录")
    output = saved.get("output", e.OUTPUT_DISPLAY)
    single = saved.get("single_video", saved.get("src", ""))
    if single:
        try:
            submission_paths(root, current, single, output, False)
        except (OSError, ValueError):
            single = ""
    options = {name: saved.get(name, default) for name, default in defaults.items()}
    if "outpaint_model" in options and options["outpaint_model"] not in {
        value for _, value in available_model_choices()
    }:
        options["outpaint_model"] = defaults["outpaint_model"]
    if options.get("union_variant") == "syfe" and syfe_union_path() is None:
        options["union_variant"] = defaults["union_variant"]
    controls = dict(options)
    for name in ("curvature", "keep_center", "view_zoom", "auto_stage"):
        if name in controls:
            controls[name] = gr.update(value=options[name], visible=options.get("mode") != e.CINEMA)
    return (
        nav,
        files,
        root,
        output,
        *controls.values(),
        single,
        {"input_root": root, "single_video": single, "output": output, **options},
    )


def analyze_background(rel):
    import background_analysis as bg

    try:
        path = e.source(rel)
        report = bg.report_path(path)
        if not report.exists():
            result = subprocess.run(
                [
                    sys.executable,
                    "-X",
                    "utf8",
                    str(e.ROOT / "background_analysis.py"),
                    str(path),
                ],
                capture_output=True,
                check=False,
                encoding="utf-8",
                errors="replace",
            )
            if result.returncode:
                raise ValueError(
                    result.stderr.strip().splitlines()[-1]
                    if result.stderr.strip()
                    else "背景分析未完成"
                )
        data = json.loads(report.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        raise gr.Error(str(exc)) from exc
    return (
        (
            f"{path.name} · {data['shots']} 个镜头 · 预计新建 {data['new']} 个背景视角 · "
            f"复用 {data['reuse']} 段 · 需复核 {data['review']} 段"
        ),
        data["rows"],
    )


def parameter_help(value, description):
    if isinstance(value, bool):
        value = "开启" if value else "关闭"
    elif value is None or value == "":
        value = "留空"
    return f"默认值为 {value}。{description}"


def generation_profile(level, selection):
    values = dict(SPEED_PRESETS[level])
    if selection in H3_CHECKPOINTS:
        selection = "h3_hq" if level == "质量" else "h3"
        values.update({key: field[1] for key, field in H3_FIELDS.items()
                       if key not in ("h3_prompt", "h3_negative")})
        values["h3_union"] = level == "质量"
    else:
        values.update(LTX_DEFAULTS)
        if level == "质量":
            values["union_variant"] = "syfe"
    return selection, values


def profile_match(selection, settings):
    for level in ("快速", "质量"):
        model, target = generation_profile(level, selection)
        if model == selection and all(settings.get(key) == value for key, value in target.items()):
            return level
    return None


def profile_header(selection, settings, match):
    applied = f"{match}档" if match else "自定义参数"
    if selection in H3_CHECKPOINTS:
        canvas = int(settings["h3_size"])
        union = "Union 开启" if settings["h3_union"] else "Union 关闭"
        summary = f"{canvas} × {canvas} · {settings['generation_fps']} fps · {int(settings['h3_steps'])} 步 · {union}"
    else:
        canvas = "前半球" if settings["front_only"] else "全景画布"
        union = UNION_LABELS[settings["union_variant"]] if settings["union_enabled"] else "Union 关闭"
        summary = f"{settings['generation_fps']} fps · {int(settings['steps'])} 步 · {canvas} · {union}"
    return f"""
        <div class="profile-heading">
            <div><h3>外围生成预设</h3><p>{escape(summary)}</p></div>
            <span role="status" class="profile-state" data-matched="{str(bool(match)).lower()}">
                当前 · {escape(applied)}
            </span>
        </div>
    """


def profile_description(level, selection, has_syfe):
    model, settings = generation_profile(level, selection)
    fps = settings["generation_fps"]
    if model in H3_CHECKPOINTS:
        name = "NVFP4 HQ" if model == "h3_hq" else "NVFP4"
        union = "开启单遍 Union 2.0" if settings["h3_union"] else "关闭 Union 2.0"
        size = settings["h3_size"]
        description = f'{name} · <span class="profile-union">{union}</span>'
        detail = f"{size} × {size} · {fps} fps · {settings['h3_steps']} 步"
    else:
        description = f'<span class="profile-union">{UNION_LABELS[settings["union_variant"]]} 参考分辨率</span>'
        detail = f"{fps} fps · {settings['steps']} 步 · 前半球"
        if settings["union_variant"] == "syfe" and not has_syfe:
            detail = "需要安装 SyFe Union 权重"
    return f"""
        <h4>{level}档</h4>
        <p>{description}</p>
        <p>{detail}</p>
    """


def build():
    has_syfe = syfe_union_path() is not None
    union_choices = [(label, value) for value, label in UNION_LABELS.items() if value == "official" or has_syfe]
    with e.db() as conn:
        conn.execute(
            "CREATE TABLE IF NOT EXISTS form_preferences(name TEXT PRIMARY KEY, value TEXT NOT NULL)"
        )
    with gr.Blocks(title="MMD 立体转制 · Quest 2") as app:
        gr.Markdown("# MMD 立体转制\nQuest 2 · HEVC")
        tabs = gr.Tabs()
        with tabs, gr.Tab("素材与转制", id="convert"):
            with gr.Row():
                input_root = gr.Textbox(
                    label="输入根目录",
                    value=e.INPUT_DISPLAY,
                    interactive=True,
                    info="保留目录结构时，从这里计算相对路径。修改后按回车浏览。",
                )
                output = gr.Textbox(
                    label="输出根目录",
                    value=e.OUTPUT_DISPLAY,
                    interactive=True,
                    info="可输入现有目录或新目录。新目录会在转制时创建。",
                )
            with gr.Row():
                current = gr.Dropdown(
                    label="本次处理目录（含子目录）",
                    choices=[("整个输入根目录", ".")],
                    value=".",
                    interactive=True,
                    filterable=True,
                    scale=5,
                )
            keep_folders = gr.Checkbox(
                label="保留相对于输入根目录的文件夹结构",
                value=True,
                info="例如选择「舞蹈素材」，成片写入输出根目录下的同名目录。",
            )
            files = gr.Dataframe(
                headers=["视频路径（相对于当前目录）"],
                value=[],
                row_count=0,
                column_count=1,
                datatype="str",
                type="array",
                interactive=False,
                label="视频文件",
                max_height=320,
                show_search="filter",
                buttons=[],
            )
            src = gr.Textbox(
                label="单个视频（可选）", interactive=True, value="",
                info="点击列表或粘贴完整视频路径。留空时处理上方选定的整个目录。",
            )
            with gr.Row():
                mode = gr.Radio(list(e.MODES), value=LTX_DEFAULTS["mode"], label="播放模式")
                strength = gr.Slider(
                    minimum=0.0,
                    maximum=e.MAX_DIVERGENCE,
                    value=3.0,
                    step=0.1,
                    precision=1,
                    interactive=True,
                    label="立体强度",
                    info="可拖动或直接输入数值。轻 1.0 · 标准 2.0 · 重 3.0",
                )
            with gr.Row():
                curvature = gr.Slider(
                    0, 100, value=LTX_DEFAULTS["curvature"], step=1, precision=0,
                    label="横向曲率（%）", info="0% 平面透视，100% 圆柱面。",
                    interactive=True, visible=False,
                )
                view_zoom = gr.Slider(
                    *VIEW_ZOOM_RANGE, value=DEFAULT_VIEW_ZOOM, step=0.01, precision=2,
                    label="原片视角倍率",
                    info="1.00× 沿用原构图。调高后原片占据更多正前方视野，成片仍为双眼 VR180。",
                    interactive=True, visible=False,
                )
                keep_center = gr.Checkbox(
                    label="保持中心人物大小",
                    info="勾选后，曲率越高，画面向两侧展开得越宽。关闭时固定水平视场。",
                    value=True, interactive=True, visible=False,
                )
            with gr.Row():
                width = gr.Radio(
                    [("自动", 0), *e.WIDTHS[1:]], value=0, label="分辨率（单眼上限）"
                )
                model = gr.Dropdown(
                    sorted(e.MODELS_ALLOWED), value="VDA_Metric_L", label="视频深度模型"
                )
            sizes = gr.Textbox(label="预计输出尺寸", interactive=False)
            auto_stage = gr.Checkbox(
                label="启用外围扩画",
                value=True,
                visible=False,
                interactive=True,
                info="开启后可选择外围生成方式。",
            )
            backend = gr.Dropdown(
                available_model_choices(), value="h3_hq", label="外围生成模型", interactive=True, filterable=False,
                info="默认值为 H3 FL2VA · NVFP4 HQ。LTX-2.3 和 LTX-2.5 共用采样、参考掩码和 Union 深度设置。启用外围扩画后生效。",
            )
            with gr.Column(elem_id="generation-profile") as preset_group:
                initial_model, initial_profile = generation_profile("质量", "h3_hq")
                preset_note = gr.HTML(profile_header(initial_model, initial_profile, "质量"), padding=False)
                with gr.Row():
                    with gr.Column(elem_classes="profile-card", min_width=220):
                        quick_card = gr.HTML(profile_description("快速", initial_model, has_syfe), padding=False)
                        quick = gr.Button("应用快速档", size="sm", variant="secondary")
                    with gr.Column(elem_classes="profile-card", min_width=220):
                        quality_card = gr.HTML(profile_description("质量", initial_model, has_syfe), padding=False)
                        quality = gr.Button("应用质量档", size="sm", variant="primary")
                with gr.Row(elem_id="generation-profile-actions"):
                    reset_baseline = gr.Button("恢复全部生成设置", size="sm", variant="secondary",
                                               elem_id="generation-profile-reset")
            match_stage = gr.Checkbox(value=True, label="优先匹配已有空舞台", render=False,
                                      info="仅在机位跟踪稳定时使用舞台参考，其余镜头使用含人物的连续原帧。")
            seam_luma = gr.Checkbox(value=True, label="校正外围接缝亮度", render=False)
            observed_edges = gr.Checkbox(value=True, label="用同镜头双眼原帧补全边缘", render=False)
            anime_detail = gr.Checkbox(value=True, label="增强外围动漫细节", render=False)
            generation_fps = gr.Dropdown([12, 16, 24], value=LTX_DEFAULTS["generation_fps"],
                                         label="外围生成帧率（fps）", render=False)
            seed = gr.Number(value=726391, precision=0, minimum=0, maximum=2147483647,
                             label="起始随机种子", render=False)
            base_controls = {
                "match_stage": match_stage, "seam_luma": seam_luma, "generation_fps": generation_fps,
                "seed": seed, "observed_edges": observed_edges, "anime_detail": anime_detail,
            }
            style_controls = {}
            h3_controls = {}
            model_groups = []
            def add_group(title, names, family=None):
                with gr.Accordion(title, open=False, visible=family != "ltx2") as group:
                    if family:
                        model_groups.append((family, group))
                    for name in names:
                        if name in base_controls:
                            base_controls[name].render()
                            continue
                        settings, help_text, target = ((H3_FIELDS, H3_INFO, h3_controls)
                                                       if name in H3_FIELDS else (CONTROLS, CONTROL_INFO, style_controls))
                        label, default, low, high, step = settings[name]
                        info = parameter_help(default, help_text[name])
                        if name == "union_variant":
                            component = gr.Dropdown(choices=union_choices, value=default, label=label,
                                                    info=parameter_help(UNION_LABELS[default], help_text[name]),
                                                    filterable=False)
                        elif name in CHOICES:
                            component = gr.Dropdown(choices=CHOICES[name], value=default, label=label, info=info)
                        elif isinstance(default, str):
                            component = gr.Textbox(value=default, label=label, info=info, lines=2, max_length=4096)
                        elif isinstance(default, bool):
                            component = gr.Checkbox(value=default, label=label, info=info)
                        else:
                            component = gr.Slider(low, high, value=default, step=step, label=label, info=info)
                        target[name] = component

            with gr.Group() as style_group:
                add_group("H3 · 画布与采样", [k for k in H3_FIELDS if k not in ("h3_union", "h3_strength")], "h3")
                add_group("H3 · Union 2.0", ["h3_union", "h3_strength"], "h3")
                add_group("LTX-2 · 扩画与采样", [k for k in GROUPS["扩画与采样"] if k in LTX_ONLY], "ltx2")
                add_group("LTX-2 · 原片引导与 Union 深度", [k for k in
                          (*GROUPS["原片参考与掩码"], *GROUPS["Union 外围深度"], "joint_audio") if k in LTX_ONLY], "ltx2")
                for title, names in GROUPS.items():
                    shared = [k for k in names if k not in LTX_ONLY and k != "person_threshold"]
                    if title == "生成前外围背景延续":
                        shared.append("person_threshold")
                    if shared:
                        add_group("镜头与帧率" if title == "扩画与采样" else title, shared)

            preset_keys = [*LTX_DEFAULTS, "detail_fps", "detector_cuda", *H3_FIELDS]
            preset_fields = {
                "mode": mode, "curvature": curvature, "view_zoom": view_zoom,
                **base_controls, **style_controls, **h3_controls,
            }
            preset_controls = [preset_fields[key] for key in preset_keys]
            def apply_profile(level, selection):
                selection, values = generation_profile(level, selection)
                if values.get("union_variant") == "syfe" and not has_syfe:
                    raise gr.Error("本服务缺少 SyFe Union 权重")
                return [selection, *(values.get(k, gr.skip()) for k in preset_keys)]
            def profile_state(selection, *values):
                current = dict(zip(preset_keys, values))
                match = profile_match(selection, current)
                return (
                    profile_header(selection, current, match),
                    profile_description("快速", selection, has_syfe),
                    profile_description("质量", selection, has_syfe),
                    gr.update(variant="primary" if match == "快速" else "secondary"),
                    gr.update(variant="primary" if match == "质量" else "secondary",
                              interactive=selection in H3_CHECKPOINTS or has_syfe),
                )
            def model_visibility(selection):
                family = "h3" if selection in H3_CHECKPOINTS else "ltx2"
                return [gr.update(visible=family == group_family) for group_family, _ in model_groups]
            group_outputs = [g for _, g in model_groups]
            backend.change(model_visibility, backend, group_outputs, api_name=False)
            state_inputs = [backend, *preset_controls]
            profile_outputs = [preset_note, quick_card, quality_card, quick, quality]
            def reflect(trigger):
                trigger.then(profile_state, state_inputs, profile_outputs, api_name=False, show_progress="hidden")
            for button, level in ((quick, "快速"), (quality, "质量")):
                reflect(button.click(partial(apply_profile, level), inputs=backend, outputs=state_inputs, api_name=False))
            gr.on(triggers=[backend.change, *(c.input for c in preset_controls)], fn=profile_state,
                  inputs=state_inputs, outputs=profile_outputs, api_name=False, show_progress="hidden",
                  trigger_mode="always_last")
            visibility_controls = [preset_group, style_group]
            # Hidden positional inputs preserve the existing submit API.
            stage_preset = gr.Dropdown([e.DANCE, e.CLOSEUP], value=None, visible=False)
            empty_only = gr.Checkbox(False, visible=False)
            fast_closeup = gr.Checkbox(False, visible=False)
            # Keep the positional stage_method argument available to existing API clients.
            legacy_method = gr.Dropdown(list(e.STAGE_METHODS), value=None, visible=False)
            mode.change(
                lambda m: [gr.update(visible=m != e.CINEMA) for _ in range(4)],
                mode, [curvature, keep_center, view_zoom, auto_stage], api_name=False,
            )
            gr.on(
                triggers=[mode.change, auto_stage.change],
                fn=lambda m, on: [gr.update(visible=m != e.CINEMA and on) for _ in visibility_controls],
                inputs=[mode, auto_stage], outputs=visibility_controls,
                api_name=False,
            )
            paths = gr.Textbox(label="本次转制路径", lines=2, interactive=False)
            with gr.Row():
                see = gr.Button("预览原片")
                send = gr.Button("加入整片转制队列", variant="primary")
            message = gr.Textbox(label="提交结果", interactive=False)
            img = gr.Image(label="原片预览", interactive=False)
            if (e.ROOT / "background_analysis.py").is_file() and (
                e.NUNIF / "nunif"
            ).is_dir():
                import background_analysis as bg

                with gr.Accordion("背景变化检测", open=False):
                    analyze = gr.Button("分析所选视频的背景")
                    summary = gr.Textbox(label="分析结果", interactive=False)
                    timeline = gr.Dataframe(
                        headers=bg.HEADERS,
                        datatype=["str"] * len(bg.HEADERS),
                        interactive=False,
                        label="背景复用时间线",
                    )
                    analyze.click(
                        analyze_background,
                        src,
                        [summary, timeline],
                        api_name="analyze_background",
                        concurrency_limit=1,
                    )
        with tabs, gr.Tab("运行监控", id="monitor"):
            live = gr.HTML()
            live_log = gr.Textbox(
                label="实时任务日志",
                lines=10,
                max_lines=10,
                autoscroll=True,
                interactive=False,
            )
            with gr.Row():
                sync = gr.Button("立即同步")
                release = gr.Button("停止当前及排队任务并释放 GPU", variant="stop")
            release_message = gr.Textbox(label="停止结果", interactive=False)
        with tabs, gr.Tab("任务队列", id="queue"):
            table = gr.Dataframe(
                headers=["任务 ID", "状态", "阶段", "视频", "强度", "播放模式", "范围"],
                datatype=["str"] * 7,
                type="array",
                interactive=False,
                label="最近 200 个任务",
            )
            key = gr.Textbox(label="选中任务 ID", interactive=True)
            with gr.Row():
                refresh = gr.Button("刷新列表")
                info = gr.Button("查看详情")
                see_out = gr.Button("预览双眼画面")
                cancel = gr.Button("取消任务")
                retry = gr.Button("重试任务")
            report = gr.Textbox(label="任务详情与输出位置", lines=4, interactive=False)
            logs = gr.Textbox(label="运行日志", lines=6, interactive=False)
            result = gr.Image(label="双眼画面预览", interactive=False)
        # Read each initial value from its component, before restoring the user's preferences.
        for field, description in (
            (keep_folders, keep_folders.info),
            (mode, "选择头显中的投影形式。VR180 的角度指定原片覆盖的水平视场，影院模式使用平面银幕。"),
            (strength, "调节左右眼视差与立体感。轻度为 1.0，标准为 2.0，重立体为 3.0。"),
            (curvature, curvature.info),
            (keep_center, keep_center.info),
            (view_zoom, view_zoom.info),
            (auto_stage, "为 VR180 原片以外的区域生成连续背景，开启后显示扩画和接缝参数。"),
            (match_stage, match_stage.info),
            (seam_luma, "根据原片边缘校正外围明暗，减轻原片与生成区域之间的亮度跳变。"),
            (observed_edges, "使用同镜头邻帧中可对齐的真实内容，补全左右眼各自的原片边缘。"),
            (anime_detail, "在生成的外围叠加动漫细节层，可调整细节强度和采样频率。"),
            (generation_fps, "每秒送入扩画模型的帧数。降低可节省推理时间，也可能改变布景和动作连续性。"),
            (seed, "生成随机噪声的起始编号。相同参数可用于对照，按镜头递增的开关在扩画与采样中。"),
            (model, "为原片计算深度和左右眼视差。模型的尺寸与深度表示会影响立体效果和处理时间。"),
            (key, "点击队列中的任务或填写任务 ID，可查看详情、预览、取消或重试该任务。"),
        ):
            field.info = parameter_help(field.value, description)
        current.info = "选择本次扫描的位置，包含它的所有子目录。"
        width.info = parameter_help("自动（0）", "按素材和播放模式计算尺寸。手动选择时限制单眼尺寸，预计输出尺寸会随之更新。")
        browse_outputs = [current, files, input_root, img, src]
        options = {
            "keep_folders": keep_folders,
            "mode": mode,
            "strength": strength,
            "curvature": curvature,
            "keep_center": keep_center,
            "view_zoom": view_zoom,
            "width": width,
            "model": model,
            "auto_stage": auto_stage,
            "match_stage": match_stage, "seam_luma": seam_luma,
            "generation_fps": generation_fps, "seed": seed,
            "observed_edges": observed_edges, "anime_detail": anime_detail,
            **style_controls, "outpaint_model": backend, **h3_controls,
        }
        generation_controls = {name: control for name, control in options.items() if name != "keep_folders"}
        generation_defaults = {name: control.value for name, control in generation_controls.items()}
        reset = reset_baseline.click(partial(reset_generation_settings, generation_defaults),
                                    inputs=backend, outputs=list(generation_controls.values()), api_name=False)
        reflect(reset)
        previous = gr.State()
        loaded = app.load(
            partial(
                restore_form,
                {name: component.value for name, component in options.items()},
            ),
            outputs=[
                current,
                files,
                input_root,
                output,
                *options.values(),
                src,
                previous,
            ],
            api_name="restore_form",
        )
        loaded.success(lambda m, on: [gr.update(visible=m != e.CINEMA and on) for _ in visibility_controls],
                       [mode, auto_stage], visibility_controls, api_name=False)
        loaded.success(model_visibility, backend, group_outputs, api_name=False)
        reflect(loaded)
        loaded.success(dimensions, [src, width, mode], sizes, api_name=False)
        path_inputs = [input_root, current, src, output, keep_folders]
        loaded.success(path_preview, path_inputs, paths, api_name=False)
        gr.on(
            triggers=[c.change for c in path_inputs], fn=path_preview,
            inputs=path_inputs, outputs=paths, api_name=False,
            show_progress="hidden", trigger_mode="always_last",
        )
        for name, component in {"input_root": input_root, "single_video": src, "output": output, **options}.items():
            component.change(
                partial(remember_field, name),
                [component, previous],
                previous,
                api_name=False,
                show_progress="hidden",
                trigger_mode="always_last",
                concurrency_id="save_form",
                concurrency_limit=1,
            )
        gr.on(
            triggers=[current.input],
            fn=open_scope,
            inputs=[current, input_root],
            outputs=browse_outputs,
            api_name="browse",
            trigger_mode="always_last",
        )
        input_root.submit(open_input, input_root, browse_outputs, api_name="open_input_path")
        output.submit(open_output, output, output, api_name="open_output_path")
        pick = files.select(select_video, [input_root, current], src, api_name=False)
        pick.then(dimensions, [src, width, mode], sizes, api_name=False)
        gr.on(
            triggers=[src.blur, src.submit, width.change, mode.change, current.change],
            fn=dimensions,
            inputs=[src, width, mode],
            outputs=sizes,
            api_name="dimensions",
            trigger_mode="always_last",
        )
        see.click(preview_source, src, img)
        submitted = send.click(
            submit,
            [
                input_root,
                strength,
                width,
                model,
                mode,
                curvature,
                keep_center,
                output,
                keep_folders,
                current,
                auto_stage,
                legacy_method,
                stage_preset,
                empty_only,
                src,
                fast_closeup,
                *(base_controls[name] for name in BASE_FIELDS),
                *(style_controls[name] for name in CONTROLS if name not in ("front_only", "union_variant")),
                backend, *(h3_controls[k] for k in H3_FIELDS), style_controls["front_only"], view_zoom,
                style_controls["union_variant"],
            ],
            [message, key],
            api_name="submit",
        )
        submitted.success(
            lambda: gr.Tabs(selected="monitor"), outputs=tabs, api_name=False
        )
        app.load(initial_tab, outputs=tabs, queue=False, api_name=False)
        release.click(
            e.stop_queue, outputs=release_message, queue=False, api_name="stop_queue"
        ).then(live_progress, outputs=[live, live_log], queue=False, api_name=False)
        app.load(queue, outputs=table)
        app.load(live_progress, outputs=[live, live_log], queue=False)
        gr.Timer(2).tick(
            live_progress,
            outputs=[live, live_log],
            queue=False,
            show_progress="hidden",
            api_name=False,
        )
        sync.click(
            live_progress, outputs=[live, live_log], queue=False, api_name="live"
        )
        gr.Timer(5).tick(queue, outputs=table, show_progress="hidden", api_name=False)
        refresh.click(queue, outputs=table, api_name="jobs")
        table.select(selected, table, key).then(detail, key, [report, logs])
        info.click(detail, key, [report, logs], api_name="detail")
        see_out.click(preview_job, key, result)
        cancel.click(lambda k: e.control(k, "cancel"), key, report).then(
            queue, outputs=table
        )
        retry.click(lambda k: e.control(k, "retry"), key, report).then(
            queue, outputs=table
        )
    return app


if __name__ == "__main__":
    e.initialize()
    build().launch(
        server_name=os.getenv("MMD_HOST", "127.0.0.1"),
        server_port=int(os.getenv("MMD_PORT", "7860")),
        css=GENERATION_PROFILE_CSS,
        share=False,
        show_error=True,
    )
