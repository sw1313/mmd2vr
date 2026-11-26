"""Controls shared by the VR360 submission API and the WebUI."""

import json
import logging
import math
import os
from pathlib import Path
import re

DEFAULT_PROMPT = "360 degree equirectangular panorama, high quality, detailed video"
DEFAULT_VIEW_ZOOM = 1.0
LTX_DEFAULT_MODE = "VR180 110°"
LTX_DEFAULT_CURVATURE = 100
LTX_DEFAULT_FPS = 24
VIEW_ZOOM_RANGE = (1.0, 1.25)
LTX_FRONT_WIDTH = 768
SYFE_UNION = "syfe-union-step05000.safetensors"
UNION_LABELS = {"official": "官方 Union · 0.5×", "syfe": "SyFe Union · 1.0×"}
H3_PROMPT = "Forward view. Continue the same animated scene and motion. High quality, detailed video. Match the original 3D animation, surface materials, texture detail, lighting direction, and soft shadows. Preserve each visible character's proportions and the continuity of their movements."
H3_NEGATIVE = 'extra limbs, missing limbs, fused limbs, disconnected arms or legs, duplicated bodies, malformed hands, extra fingers, flickering textures, blurry details, smeared surface textures, flat painted backdrops'
CHECKPOINTS = [
    "360vroutpaint_v2_step09000.safetensors",
    "360vroutpaint_v2_step07000.safetensors",
]
CHOICES = {
    "union_variant": tuple(UNION_LABELS),
    "outpaint_checkpoint": CHECKPOINTS,
    "sampler": ["euler_ancestral", "euler", "heun"],
    "window_frames": [9, 17, 33, 49, 65, 89, 121],
    "depth_size": [280, 392, 518],
    "vae_tile": [256, 384, 512, 640],
    "offload_streams": [2, 4],
    "iw3_batch": [1, 2, 4],
    "png_compression": [0, 1, 4],
}

# Label, default, minimum, maximum and slider step. Boolean values use a checkbox.
CONTROLS = {
    "union_enabled": ("Union 外围深度控制", True, None, None, None),
    "union_variant": ("LTX Union 模型", "official", None, None, None),
    "union_weight": ("Union 模型权重", 1.0, 0, 1, 0.05),
    "depth_attention": ("深度引导强度", 1.0, 0, 1, 0.05),
    "texture_enabled": ("简化外围纹理（保留轮廓）", True, None, None, None),
    "texture_mix": ("纹理简化混合强度", 0.90, 0, 1, 0.05),
    "texture_radius": ("双边滤波半径（像素）", 4, 1, 8, 1),
    "texture_sigma1": ("第一遍颜色差阈值（σ）", 32, 1, 100, 1),
    "texture_sigma2": ("第二遍颜色差阈值（σ）", 28, 1, 100, 1),
    "texture_space": ("双边滤波空间范围（σ，像素）", 5, 1, 16, 1),
    "finish_feather": ("纹理和配色的外侧过渡宽度（像素）", 12, 0, 64, 1),
    "palette_enabled": ("按同镜头原片背景调整外围配色", False, None, None, None),
    "palette_strength": ("原片配色强度", 0.85, 0, 1, 0.05),
    "outpaint_checkpoint": ("VR360 扩画权重文件", CHECKPOINTS[0], None, None, None),
    "outpaint_weight": ("扩画 LoRA 权重", 1.0, 0, 1.5, 0.05),
    "source_strength": ("原片参考引导强度", 1.0, 0, 1, 0.05),
    "prompt": ("生成提示词", DEFAULT_PROMPT, None, None, None),
    "sampler": ("采样器", "euler_ancestral", None, None, None),
    "steps": ("扩画采样步数", 8, 2, 32, 1),
    "cfg": ("提示词引导（CFG）", 1.0, 1, 3, 0.1),
    "window_frames": ("单次生成窗口上限（帧）", 121, None, None, None),
    "window_overlap": ("长镜头窗口重叠（帧）", 8, 0, 32, 1),
    "split_shots": ("自动按镜头切分", True, None, None, None),
    "seed_per_shot": ("按镜头递增种子", True, None, None, None),
    "vae_tiled": ("分块编码和解码", True, None, None, None),
    "vae_tile": ("VAE 分块大小（像素）", 512, None, None, None),
    "depth_enabled": ("使用外围深度图引导", True, None, None, None),
    "depth_strength": ("深度参考帧强度", 1.0, 0, 1, 0.05),
    "depth_size": ("深度模型输入尺寸", 392, None, None, None),
    "depth_smoothing": ("深度时序中值半径（帧）", 2, 0, 8, 1),
    "protect_actors": ("保护出框人物附近的扩画区域", True, None, None, None),
    "person_threshold": ("人物检测置信度", 0.307, 0.05, 0.95, 0.001),
    "depth_margin": ("深度引导距原片的留白（像素）", 12, 0, 64, 1),
    "depth_feather": ("深度引导外围过渡宽度（像素）", 40, 1, 128, 1),
    "texture_passes": ("纹理滤波遍数", 2, 1, 2, 1),
    "seam_strength": ("接缝亮度校正强度", 1.0, 0, 1, 0.05),
    "observed_strength": ("双眼原帧补边混合强度", 1.0, 0, 1, 0.05),
    "donor_radius": ("双眼补边前后搜索范围（60 fps 等效帧）", 65, 5, 120, 1),
    "donor_count": ("双眼补边候选帧数", 12, 3, 24, 1),
    "detail_strength": ("外围动漫细节强度", 1.0, 0, 1, 0.05),
    "detail_fps": ("外围细节采样帧率", 10, 1, 30, 1),
    "contour_enabled": ("二维外围接缝融合", True, None, None, None),
    "contour_strength": ("二维接缝融合强度", 1.0, 0, 1, 0.05),
    "contour_width": ("二维接缝外侧宽度（输出像素）", 64, 8, 128, 1),
    "source_mask_enabled": ("使用原片有效区域参考掩码", True, None, None, None),
    "source_mask_inside": ("原片有效区域权重", 1.0, 0, 1, 0.05),
    "source_mask_outside": ("原片以外区域权重", 0.10, 0, 1, 0.05),
    "source_attention": ("原片参考注意力强度", 1.0, 0, 1, 0.05),
    "detector_cuda": ("使用 GPU 检测人物", True, None, None, None),
    "offload_streams": ("模型权重传输流数", 2, None, None, None),
    "joint_audio": ("扩画时联合推算音频和画面", True, None, None, None),
    "iw3_batch": ("原片立体估深批大小", 1, None, None, None),
    "png_compression": ("扩画中间帧 PNG 压缩等级", 0, None, None, None),
    "guide_extension_enabled": ("生成前延续外围背景", True, None, None, None),
    "guide_extension_width": ("背景延续宽度（全景引导图像素）", 40, 4, 96, 1),
    "guide_extension_margin": ("人物框外保护距离（原帧引导图像素）", 15, 0, 64, 1),
    "guide_extension_contrast": ("背景取色的局部亮度差上限", 45, 5, 100, 1),
    "guide_extension_blur": ("背景取色平滑半径（高斯 σ）", 3.0, 0.5, 8, 0.5),
    "front_only": ("只生成前半球", True, None, None, None),
}

H3_FIELDS = {
    "h3_size": ("H3 生成画布（每边像素）", 512, None, None, None),
    "h3_union": ("使用 H3 Union 2.0 · INT8 ConvRot", True, None, None, None),
    "h3_strength": ("H3 Union 控制强度", 1.0, 0.05, 2.0, 0.05),
    "h3_steps": ("H3 采样步数", 20, 2, 40, 1),
    "h3_cfg": ("H3 提示词引导（CFG）", 1.25, 1, 4, 0.05),
    "h3_sampler": ("H3 采样器", "res_multistep", None, None, None),
    "h3_prompt": ("H3 正向提示词", H3_PROMPT, None, None, None),
    "h3_negative": ("H3 负向提示词", H3_NEGATIVE, None, None, None),
    "h3_window": ("H3 单窗口上限（帧）", 141, None, None, None),
    "h3_overlap": ("H3 窗口重叠（帧）", 8, 0, 32, 1),
}
CHOICES.update(h3_size=[448, 512, 640], h3_window=[39, 73, 107, 141], h3_sampler=["res_multistep", "euler"])
H3_CHECKPOINTS = {
    "h3": "MiniMax-H3_FL2VA-NVFP4.safetensors",
    "h3_hq": "MiniMax-H3_FL2VA-NVFP4-HQ.safetensors",
}
MODEL_CHOICES = [("H3 FL2VA · NVFP4", "h3"), ("H3 FL2VA · NVFP4 HQ", "h3_hq"),
                 ("LTX-2.3 · VR360", "ltx2"), ("LTX-2.5 · Distilled Q4_K_M", "ltx25")]
LTX25_ASSETS = {
    "diffusion_models": "LTX-2.5-Distilled-Q4_K_M.gguf",
    "text_encoders": "gemma4-12b-with-proj-ltx-2.5-nvfp4.safetensors",
    "vae": "ltx-2.5-video-vae-conv-bf16.safetensors",
}
LTX25_AUDIO_VAE = "ltx-2.5-audio-vae-bf16.safetensors"


def ltx25_path():
    if explicit := os.environ.get("MMD_LTX25_HOME"):
        return Path(explicit)
    return Path(__file__).resolve().parent / ".runtime" / "models" / "ltx25"


def ltx25_available():
    home = ltx25_path()
    return bool(home and all((home / "models" / directory / name).is_file()
                             for directory, name in LTX25_ASSETS.items())
                and (home / "models" / "vae" / LTX25_AUDIO_VAE).is_file()
                and (home / "gguf_nodes" / "__init__.py").is_file())


def syfe_union_path():
    home = ltx25_path()
    if home:
        path = home / "models" / "loras" / SYFE_UNION
        if path.is_file() and path.stat().st_size == 1711612856:
            return path
    return None


def available_model_choices():
    builtins = list(MODEL_CHOICES) if ltx25_available() else [item for item in MODEL_CHOICES if item[1] != "ltx25"]
    root = custom_models_root()
    if not root.is_dir():
        return builtins
    for folder in sorted(root.iterdir()):
        if not (folder / "model.json").is_file():
            continue
        selection = CUSTOM_PREFIX + folder.name
        try:
            manifest = custom_model(selection)
            style_options({"outpaint_model": selection}, manifest)
        except (OSError, ValueError) as exc:
            logging.getLogger(__name__).warning("跳过自定义扩画模型 %s：%s", folder.name, exc)
            continue
        builtins.append((f"{manifest['label']} · 自定义", selection))
    return builtins


LTX_ONLY = {
    "outpaint_checkpoint", "outpaint_weight", "prompt", "sampler", "steps", "cfg",
    "window_frames", "window_overlap", "vae_tiled", "vae_tile", "source_strength",
    "source_attention", "union_enabled", "union_variant", "union_weight", "depth_enabled", "depth_strength",
    "depth_attention", "depth_size", "depth_smoothing", "protect_actors", "depth_margin",
    "depth_feather", "joint_audio",
    "front_only",
}

CUSTOM_PREFIX = "custom:"
CUSTOM_ID = re.compile(r"[a-z0-9][a-z0-9_-]{0,47}")
CUSTOM_ASSET_DIRS = {
    "checkpoint": "diffusion_models", "video_vae": "vae", "audio_vae": "vae",
    "text_encoder": "text_encoders", "text_projection": "text_encoders", "outpaint_lora": "loras",
}
# Absent capabilities disable both the operation and its related controls.
CUSTOM_FEATURES = {
    "union": {
        "union_enabled": False, "union_variant": "official", "union_weight": 0.0,
        "depth_enabled": False, "depth_strength": 0.0, "depth_attention": 0.0,
        "depth_size": 392, "depth_smoothing": 0, "protect_actors": False,
        "depth_margin": 0, "depth_feather": 1,
        "h3_union": False, "h3_strength": 1.0,
    },
    "source_mask": {
        "source_mask_enabled": False, "source_mask_inside": 1.0, "source_mask_outside": 0.0,
        "source_attention": 1.0,
    },
    "vae_tiling": {"vae_tiled": False, "vae_tile": 512},
    "joint_audio": {"joint_audio": False},
    "front_only": {"front_only": False},
    "background_extension": {
        "guide_extension_enabled": False, "guide_extension_width": 40,
        "guide_extension_margin": 15, "guide_extension_contrast": 45, "guide_extension_blur": 3.0,
    },
    "palette": {"palette_enabled": False, "palette_strength": 0.0},
}
CUSTOM_FAMILY_FEATURES = {
    "ltx2": set(CUSTOM_FEATURES), "ltx25": set(CUSTOM_FEATURES),
    "h3": {"union", "source_mask", "background_extension", "palette"},
}


def custom_models_root():
    return (Path(os.getenv("NUNIF_HOME", Path(__file__).resolve().parent / ".runtime/models"))
            / "outpaint").resolve()


def custom_model_folder(selection):
    model_id = selection.removeprefix(CUSTOM_PREFIX)
    if not selection.startswith(CUSTOM_PREFIX) or not CUSTOM_ID.fullmatch(model_id):
        raise ValueError("无效的自定义扩画模型 ID")
    root = custom_models_root()
    folder = (root / model_id).resolve()
    if folder.parent != root or folder.name != model_id:
        raise ValueError("自定义扩画模型目录须位于模型根目录内")
    return folder


def custom_model_files(selection, assets):
    folder = custom_model_folder(selection)
    paths = {}
    for role, name in assets.items():
        suffixes = (".safetensors", ".gguf") if role == "checkpoint" else (".safetensors",)
        if (role not in CUSTOM_ASSET_DIRS or not isinstance(name, str) or name != name.strip()
                or not name.isprintable() or any(c in name for c in "/\\:\r\n") or Path(name).suffix.lower() not in suffixes):
            raise ValueError(f"自定义扩画模型的 {role} 须为模型文件名")
        path = folder / CUSTOM_ASSET_DIRS[role] / name
        if not path.resolve().is_relative_to(folder) or not path.is_file():
            raise ValueError(f"自定义扩画模型 {folder.name} 缺少有效的 {role} 文件")
        paths[role] = path
    return paths


def unsupported_custom_settings(manifest):
    features = set(manifest["features"])
    settings = {
        key: neutral
        for feature, controls in CUSTOM_FEATURES.items()
        if feature not in features
        for key, neutral in controls.items()
    }
    if not features.intersection({"union", "background_extension", "palette"}):
        settings.update(detector_cuda=False, person_threshold=CONTROLS["person_threshold"][1])
    return settings


def model_file_versions(paths):
    revisions = {}
    for role, path in paths.items():
        info = path.stat()
        revisions[role] = [info.st_size, info.st_mtime_ns]
    return revisions


def custom_model(selection):
    folder = custom_model_folder(selection)
    path = folder / "model.json"
    if not path.is_file() or path.resolve().parent != folder or path.stat().st_size > 65536:
        raise ValueError(f"缺少可读取的自定义模型配置 {folder.name}/model.json")
    try:
        manifest = json.loads(path.read_text(encoding="utf-8-sig"))
    except (UnicodeError, json.JSONDecodeError) as exc:
        raise ValueError(f"自定义扩画模型 {folder.name} 的 model.json 格式无效") from exc
    if not isinstance(manifest, dict) or manifest.keys() - {"label", "family", "assets", "features", "settings"}:
        raise ValueError("自定义扩画模型配置包含未知字段")
    label, family = manifest.get("label"), manifest.get("family")
    if not isinstance(label, str) or not 1 <= len(label.strip()) <= 80 or not label.isprintable():
        raise ValueError("自定义扩画模型的 label 须为 1–80 个字符")
    if not isinstance(family, str) or family not in CUSTOM_FAMILY_FEATURES:
        raise ValueError("自定义模型须使用 ltx2、ltx25 或 h3 工作流")
    features = manifest.get("features", [])
    if (not isinstance(features, list) or any(not isinstance(item, str) for item in features)
            or set(features) - CUSTOM_FAMILY_FEATURES[family]):
        raise ValueError("自定义模型声明了这个工作流不支持的功能")
    assets = manifest.get("assets")
    required = {"checkpoint", "video_vae", "text_encoder"}
    if family != "h3":
        required.add("outpaint_lora")
    if family == "ltx2":
        required.add("text_projection")
    if "joint_audio" in features:
        required.add("audio_vae")
    if not isinstance(assets, dict) or assets.keys() != required:
        raise ValueError(f"自定义模型的 assets 须填写 {', '.join(sorted(required))}")
    paths = custom_model_files(selection, assets)
    if paths["checkpoint"].suffix.lower() == ".gguf" and (family == "h3" or not gguf_loader_path().is_file()):
        raise ValueError("H3 自定义工作流使用 safetensors，LTX 的 GGUF 需要安装 GGUF 加载器")
    settings = manifest.get("settings", {})
    allowed = set(CONTROLS) - {"outpaint_checkpoint"}
    if family == "h3":
        allowed = (allowed - LTX_ONLY) | H3_FIELDS.keys()
    result = dict(label=label.strip(), family=family, assets=assets, features=sorted(set(features)))
    if not isinstance(settings, dict) or settings.keys() - (allowed - unsupported_custom_settings(result).keys()):
        raise ValueError("自定义模型的 settings 包含未知、禁用或由权重文件指定的参数")
    return {**result, "settings": settings, "versions": model_file_versions(paths)}


def gguf_loader_path():
    return ltx25_path() / "gguf_nodes" / "__init__.py"


# The submit endpoint takes these six fields before the extended controls.
BASE_FIELDS = ("match_stage", "seam_luma", "generation_fps", "seed", "observed_edges", "anime_detail")

GROUPS = {
    "扩画与采样": (
        "match_stage",
        "front_only",
        "outpaint_checkpoint",
        "outpaint_weight",
        "prompt",
        "generation_fps",
        "sampler",
        "steps",
        "cfg",
        "window_frames",
        "window_overlap",
        "split_shots",
        "seed",
        "seed_per_shot",
        "vae_tiled",
        "vae_tile",
    ),
    "原片参考与掩码": (
        "source_strength", "source_attention", "source_mask_enabled",
        "source_mask_inside", "source_mask_outside",
    ),
    "生成前外围背景延续": (
        "guide_extension_enabled", "guide_extension_width", "guide_extension_margin",
        "guide_extension_contrast", "guide_extension_blur",
    ),
    "Union 外围深度": (
        "union_enabled",
        "union_variant",
        "union_weight",
        "depth_enabled",
        "depth_strength",
        "depth_attention",
        "depth_size",
        "depth_smoothing",
        "protect_actors",
        "person_threshold",
        "depth_margin",
        "depth_feather",
    ),
    "外围纹理和配色": (
        "texture_enabled",
        "texture_mix",
        "texture_passes",
        "texture_radius",
        "texture_sigma1",
        "texture_sigma2",
        "texture_space",
        "palette_enabled",
        "palette_strength",
        "finish_feather",
    ),
    "接缝亮度与轮廓融合": (
        "seam_luma",
        "seam_strength",
        "contour_enabled",
        "contour_strength",
        "contour_width",
    ),
    "双眼原帧补边": (
        "observed_edges",
        "observed_strength",
        "donor_radius",
        "donor_count",
    ),
    "外围动漫细节": (
        "anime_detail",
        "detail_strength",
        "detail_fps",
    ),
    "推理与中间帧": (
        "detector_cuda", "offload_streams", "joint_audio", "iw3_batch", "png_compression",
    ),
}

CONTROL_INFO = {
    "front_only": "使用 768×640 前向画布，覆盖前方 180° 并在左右各留 18° 余量。关闭时生成完整 360°，交付仍为双眼 VR180。",
    "guide_extension_enabled": "无匹配舞台时，在原片轮廓外延续平滑背景颜色，减轻扩画模型沿边界生成黑框的倾向。",
    "guide_extension_width": "在 1280×640 全景引导图上向外延伸的距离。颜色向远端逐渐淡出。",
    "guide_extension_margin": "在当前和相邻引导帧的人物检测框外额外留白，减少将头发或衣服颜色延伸到外围。",
    "guide_extension_contrast": "局部亮度差达到该值时跳过取色。调低会更严格地避开轮廓和纹理。",
    "guide_extension_blur": "从原帧边缘取色时的高斯平滑尺度。调大可减少延伸到外围的细碎纹理。",
    "outpaint_checkpoint": "选择用于补全全景的 LoRA 训练检查点。step09000 和 step07000 会生成不同的外围。",
    "outpaint_weight": "调整扩画 LoRA 对基础模型的影响。调高会更偏向该检查点学到的全景构图。",
    "prompt": "描述希望生成的外围场景。修改文字可能改变布景和画风，自定义文字需加载文本编码器。",
    "sampler": "选择每一步更新画面的算法。更换算法后，即使种子相同，外围也可能改变。",
    "steps": "每个窗口的去噪迭代次数。增加步数通常延长推理时间，可能改变外围细节。",
    "cfg": "控制生成时对提示词的偏重。高于 1 会加强文本引导，可能改变原片场景的延续。",
    "window_frames": "长镜头分窗口推理。较大的窗口一次能参考更长的动作，也需要更多显存。",
    "window_overlap": "相邻窗口共用的生成帧数。增加重叠可延长衔接过渡，也会重复计算更多帧。",
    "split_shots": "检测画面切换并从新镜头重新生成外围，减少上一个镜头的布景混入。",
    "seed_per_shot": "按镜头编号给起始种子加一。关闭时每个镜头使用同一个起始种子。",
    "vae_tiled": "将编码或解码的画面拆成小块以降低显存需求。关闭会改变显存占用和生成结果。",
    "vae_tile": "启用分块时每块的边长。较大的块需要更多显存，较小的块需要处理更多分块。",
    "source_strength": "控制连续原帧作为视觉引导输入的强度。调低后外围更容易偏离原片。",
    "source_attention": "调节生成时对原帧引导的整体注意力，与有效区和外围的局部权重配合使用。",
    "source_mask_enabled": "按原片覆盖范围给视觉参考分配权重，可帮助外围延续原场景。最终合成仍会贴回原片中心。",
    "source_mask_inside": "开启参考掩码时生效。提高此值会保留更多对原片有效区域的参考，黑色物体也属于有效区。",
    "source_mask_outside": "开启参考掩码时生效。调低会弱化对原片外填充区域的参考，也会弱化该处已有的舞台参考。",
    "union_enabled": "加载 Union 控制 LoRA，使扩画可以使用外围深度条件。关闭后跳过这条引导。",
    "union_variant": "官方 Union 的参考图宽高各缩为 0.5×，SyFe 使用 1.0×。所选模型在同一遍扩画采样中生效。",
    "union_weight": "调节 Union LoRA 对基础模型的影响。降低后对外围结构和材质的约束可能减弱。",
    "depth_enabled": "启用 Union 时，将估计的场景深度作为额外视频参考，帮助模型延续空间布局。",
    "depth_strength": "调整深度视频参考帧作为生成条件的强度。需要同时开启 Union 和外围深度图。",
    "depth_attention": "调整模型对深度参考的注意力。调低会减弱深度图对扩画布局的影响。",
    "depth_size": "外围引导估深时的输入尺寸。增大可保留更细的空间变化，也会增加计算量。",
    "depth_smoothing": "用前后相邻帧的中值平滑背景深度拟合。设为 0 时仅使用当前帧。",
    "protect_actors": "在靠近原片边界的人物周围减弱背景深度引导，为出框肢体留出空间。",
    "person_threshold": "筛选人物检测框的最低置信度。调低会保留更多候选框，也可能误判背景。",
    "depth_margin": "原片边界外暂不施加深度引导的距离。增大可给接缝和出框物体留下更多空间。",
    "depth_feather": "留白外逐渐增强深度引导的距离。增大可使从弱到强的过渡更缓。",
    "texture_enabled": "用保边滤波简化外围的细碎材质，尽量保留轮廓，使生成区更接近动漫画面。",
    "texture_mix": "外围使用滤波结果的比例。0 保留生成纹理，1 使用完整滤波结果。",
    "texture_radius": "滤波时搜索周围像素的半径。半径越大，会综合更大范围内的纹理。",
    "texture_passes": "控制保边滤波运行一遍或两遍。第二遍可以进一步简化小纹理。",
    "texture_sigma1": "第一遍滤波允许混合的颜色差异。数值越大，色差较大的像素也能被平滑。",
    "texture_sigma2": "第二遍滤波的颜色差异尺度，仅在选择两遍滤波时生效。",
    "texture_space": "滤波对邻近像素距离的容许程度。增大后较远的像素也能影响结果。",
    "finish_feather": "在原片边缘外逐渐引入纹理简化和配色。增加宽度使这两种处理过渡得更缓。",
    "palette_enabled": "从同镜头原片背景取样，用其颜色校正生成的外围。开启后可能改变外围色调。",
    "palette_strength": "控制背景配色修正的幅度。0 保留生成色彩，较高值更接近原片背景样本。",
    "seam_strength": "开启接缝亮度校正后，调整原片边缘外围的明暗补偿幅度。",
    "contour_enabled": "在最终投影的原片轮廓外融合亮度和颜色，减轻竖边和拐角处的框感。",
    "contour_strength": "调节最终二维接缝融合的修正幅度。设为 0 时保留融合前的像素。",
    "contour_width": "最终输出上沿原片轮廓向外融合的距离。较宽的范围会影响更远的外围。",
    "observed_strength": "找到同镜头可对齐的原帧像素时，控制它们覆盖生成边缘的比例。",
    "donor_radius": "在当前帧前后多远寻找可用于补边的同镜头画面。实际搜索范围按原片帧率换算。",
    "donor_count": "同镜头补边时最多比较的候选帧数。增加候选可扩大选择，也需要更多匹配计算。",
    "detail_strength": "开启动漫细节增强后，调节外围细节层的混合强度。越高，细节处理越明显。",
    "detail_fps": "每秒计算外围动漫细节的次数。降低可减少细节处理量，快速运动时细节更新也变慢。",
    "detector_cuda": "优先使用 GPU 检测人物，为深度引导和取色提供人物框。关闭时使用 CPU。",
    "offload_streams": "模型显存卸载时并行传输权重的 CUDA 流数。支持 2 或 4 条，实际速度取决于设备。",
    "joint_audio": "在扩画推理时联合计算音频和画面，可能影响外围内容。成片音轨取自原片。",
    "iw3_batch": "原片立体估深每批处理的数量。增大批次会改变显存需求，也可能改变视差。",
    "png_compression": "扩画临时 PNG 的无损压缩等级。0 通常写入更快，较高等级可节省临时磁盘空间。",
}


# Values are copied into the editable form and into each new job.
SPEED_PRESETS = {
    name: {"generation_fps": LTX_DEFAULT_FPS, "detail_fps": 10, "detector_cuda": True}
    for name in ("质量", "快速")
}
LTX_DEFAULTS = {
    "mode": LTX_DEFAULT_MODE,
    "curvature": LTX_DEFAULT_CURVATURE,
    "view_zoom": DEFAULT_VIEW_ZOOM,
    "generation_fps": LTX_DEFAULT_FPS,
    **{name: CONTROLS[name][1] for name in (
        "steps", "front_only", "union_enabled", "union_variant", "union_weight", "depth_enabled", "depth_strength",
        "depth_attention",
    )},
}


def style_options(values, manifest=None):
    values = dict(values)
    selection = values.pop("outpaint_model", "ltx2")
    if not isinstance(selection, str):
        raise ValueError("无效的扩画模型")
    if selection.startswith(CUSTOM_PREFIX):
        manifest = custom_model(selection) if manifest is None else manifest
        backend = manifest["family"]
        values.update(manifest["settings"])
        values.update(unsupported_custom_settings(manifest))
        values.pop("outpaint_checkpoint", None)
        values.pop("h3_checkpoint", None)
    else:
        backend = selection
    if backend not in (*H3_CHECKPOINTS, "ltx2", "ltx25"):
        raise ValueError("无效的扩画模型")
    if backend == "ltx25" and manifest is None and not ltx25_available():
        raise ValueError("本服务缺少 LTX-2.5 Q4_K_M、配套权重或 GGUF 加载器")
    controls = dict(CONTROLS)
    checkpoint = values.pop("h3_checkpoint", H3_CHECKPOINTS.get(backend))
    if backend in H3_CHECKPOINTS:
        if checkpoint not in H3_CHECKPOINTS.values() or (backend == "h3_hq" and checkpoint != H3_CHECKPOINTS[backend]):
            raise ValueError("无效的 H3 FL2VA 权重")
        for key in LTX_ONLY:
            values.pop(key, None)
            controls.pop(key)
        controls.update(H3_FIELDS)
    else:
        for key in H3_FIELDS:
            values.pop(key, None)
    unknown = values.keys() - controls.keys()
    if unknown:
        raise ValueError(f"未知扩画参数 {sorted(unknown)}")
    result = {"outpaint_model": "h3"} if backend in H3_CHECKPOINTS else {}
    if backend == "ltx25":
        result["outpaint_model"] = "ltx25"
    if backend in H3_CHECKPOINTS and checkpoint != H3_CHECKPOINTS["h3"]:
        result["h3_checkpoint"] = checkpoint
    for key, (label, default, low, high, _) in controls.items():
        value = values.get(key, default)
        if key in CHOICES:
            if type(value) is bool or value not in CHOICES[key]:
                raise ValueError(f"无效的{label}")
            value = type(default)(value)
        elif key in ("prompt", "h3_prompt", "h3_negative"):
            if not isinstance(value, str) or not value.strip() or len(value) > 4096:
                raise ValueError("提示词须为 1–4096 个字符")
            value = value.strip()
        elif type(default) is bool:
            if type(value) is not bool:
                raise ValueError(f"{label}须为开关")
        else:
            if (
                type(value) not in (int, float)
                or not math.isfinite(value)
                or not low <= value <= high
            ):
                raise ValueError(f"{label}须在 {low}–{high} 内")
            if type(default) is int and int(value) != value:
                raise ValueError(f"{label}须为整数")
            value = type(default)(value)
        result[key] = value
    if (result.get("h3_overlap", 0) >= result.get("h3_window", 1)
            or result.get("window_overlap", 0) >= result.get("window_frames", 1)):
        raise ValueError("窗口重叠帧数必须小于窗口上限")
    if manifest:
        result["custom_model"] = {"id": selection, **manifest}
    if result.get("union_variant") == "syfe" and (not manifest or result["union_enabled"]):
        if syfe_union_path() is None:
            raise ValueError("本服务缺少 SyFe Union 权重")
    else:
        # Preserve IDs and behavior for official Union jobs and previously saved tasks.
        result.pop("union_variant", None)
    return result

H3_INFO = {
    "h3_size": "等距鱼眼画布的边长，包含保护边。调整它会改变推理耗时、显存占用和外围细节。头显输出尺寸由单眼上限控制。",
    "h3_union": "在同一次 H3 采样中用原片和遮罩控制外围。快速档关闭，质量档开启。",
    "h3_strength": "开启 H3 Union 后生效。调整外围对可见原片轮廓的跟随程度。",
    "h3_steps": CONTROL_INFO["steps"], "h3_cfg": CONTROL_INFO["cfg"],
    "h3_prompt": "描述希望向外围延续的动作、材质和光照。",
    "h3_negative": "描述希望避免的错误。CFG 大于 1 时生效。",
    "h3_sampler": CONTROL_INFO["sampler"], "h3_window": CONTROL_INFO["window_frames"],
    "h3_overlap": CONTROL_INFO["window_overlap"],
}
