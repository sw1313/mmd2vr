# 模型配置

本页的路径相对于模型根目录。Windows 默认根目录为 `.runtime/models`，可用 `NUNIF_HOME` 覆盖。Compose 将宿主机模型目录挂载到 `/models`。各模型及量化权重从发布者处获取，并适用各自的使用条款。

## 基础立体转制

iw3 从模型根目录的 `iw3/pretrained_models/hub` 加载模型缓存。在新机器首次运行时，iw3 会按所选深度模型下载所需文件。如果使用 VDA_Metric_L，也可以把 `metric_video_depth_anything_vitl.pth` 放到该缓存的 `checkpoints` 中。[iw3](https://github.com/nagadomi/nunif/tree/92c294e6624802dd78042ebd47370a127a6d6f96/iw3) · [Metric Video Depth Anything Large](https://huggingface.co/depth-anything/Metric-Video-Depth-Anything-Large)

## ComfyUI 运行环境

Docker worker 会安装项目使用的 ComfyUI 提交。Windows 可另行准备同一提交及其 CUDA PyTorch 和 Python 依赖，然后在启动前设置 `MMD_COMFY` 为 ComfyUI 目录，`MMD_STAGE_PYTHON` 为其 Python 解释器。H3 在 Windows 使用 `MMD_H3_PYTHON`，可以指向同一个环境。该环境还需安装 `onnxruntime-gpu`、`opencv-python-headless` 和 `gguf`。[ComfyUI 固定提交](https://github.com/Comfy-Org/ComfyUI/tree/0072b8680dd2db0b6d68c06f718fa33bd92a728c)

## LTX 共享文件

默认目录为 `vr360-a`，可通过 `MMD_VR360_WEIGHTS` 覆盖。

| 相对路径 | 用途 |
| --- | --- |
| `loras/360vroutpaint_v2_step09000.safetensors` | 默认全景扩画。可另放 step07000 并在网页选择。 |
| `loras/ltx-2.3-22b-ic-lora-union-control-ref0.5.safetensors` | 官方 Union，供 LTX 快速档使用。 |
| `detectors/person.onnx` | 人物检测。输入 `images` 为动态批次 RGB NCHW，640×640，范围 0–1。输出形状为 `[batch, 5, anchors]`，5 个通道为中心 xy、宽高、人物分数。 |

外围深度引导使用 iw3 缓存下的 `checkpoints/video_depth_anything_vits.pth`，并读取同一 hub 目录下的 Video Depth Anything iw3 代码。可先使用 iw3 的 Video Depth Anything Small 模型初始化其缓存。[全景扩画权重](https://huggingface.co/TheBurgstall/VR-360-Outpaint-LTX2.3-IC-LoRA) · [官方 Union](https://huggingface.co/Lightricks/LTX-2.3-22b-IC-LoRA-Union-Control)

### LTX‑2.3

在 `vr360-a` 下补齐：

- `diffusion_models/ltx-2.3-22b-distilled-1.1_transformer_only_mxfp8_block32.safetensors`
- `vae/LTX23_video_vae_bf16.safetensors` 和 `vae/LTX23_audio_vae_bf16.safetensors`
- `text_encoders/gemma_3_12B_it_fp4_mixed.safetensors` 和 `text_encoders/ltx-2.3_text_projection_bf16.safetensors`

有自备的 `embeddings/vr.safetensors` 时，默认提示词可以读取缓存。缺少缓存或修改提示词时，使用文本编码器。[ComfyUI 分拆权重](https://huggingface.co/Kijai/LTX2.3_comfy/tree/main)

### LTX‑2.5 和 SyFe

Windows 默认目录为 `.runtime/models/ltx25`，容器为 `/models/ltx25`。用 `MMD_LTX25_HOME` 指定该目录，并保留下面的子目录结构。

| 相对路径 | 来源 |
| --- | --- |
| `models/diffusion_models/LTX-2.5-Distilled-Q4_K_M.gguf` | [Abiray Q4_K_M](https://huggingface.co/Abiray/LTX-2.5-Distilled-GGUF/blob/main/LTX-2.5-Distilled-Q4_K_M.gguf) |
| `models/text_encoders/gemma4-12b-with-proj-ltx-2.5-nvfp4.safetensors` | [Gemma 4 NVFP4](https://huggingface.co/Deadshot699/ltx-2.5-gemma4-12b-comfy-nvfp4/tree/main) |
| `models/vae/ltx-2.5-video-vae-conv-bf16.safetensors` 和 `models/vae/ltx-2.5-audio-vae-bf16.safetensors` | [LTX‑2.5 VAE](https://huggingface.co/Lightricks/LTX-2.5/tree/main/vae) |
| `gguf_nodes/` | [ComfyUI-GGUF 固定提交](https://github.com/city96/ComfyUI-GGUF/tree/6ea2651e7df66d7585f6ffee804b20e92fb38b8a)。该目录内应直接包含 `__init__.py`。 |
| `models/loras/syfe-union-step05000.safetensors` | 下载 [SyFe 5000 步权重](https://huggingface.co/SyFeee/LTX-2.3-SyFe-Union-Control/tree/main/runs/union_control_bal/checkpoints) 后用此文件名保存。LTX‑2.3 和 2.5 共用，用于质量档。 |

### MiniMax H3

默认目录为模型根目录下的 `h3`，可用 `MMD_H3_WEIGHTS` 覆盖。

- `diffusion_models/MiniMax-H3_FL2VA-NVFP4.safetensors` 或 `diffusion_models/MiniMax-H3_FL2VA-NVFP4-HQ.safetensors`。[FL2VA 量化权重](https://huggingface.co/DmitryDB/MiniMax-H3-ComfyUI-Quants/tree/main/FL2VA)
- `text_encoders/qwen3vl_32b_minimax_h3_nvfp4_awq.safetensors` 和 `vae/minimax_h3_video_vae_fp16.safetensors`。[ComfyUI H3 模型](https://huggingface.co/Comfy-Org/MiniMax-H3/tree/main)
- 质量档需要 `model_patches/minimax_h3_fun_controlnet_union_2.0_pruned_int8_convrot.safetensors` 和一份从裁剪版 FL2VA 提取的时间表。[H3 Union 权重](https://huggingface.co/Comfy-Org/MiniMax-H3/blob/main/model_patches/minimax_h3_fun_controlnet_union_2.0_pruned_int8_convrot.safetensors) · [时间表来源](https://huggingface.co/Comfy-Org/MiniMax-H3/blob/e5eb578a89295337b8ff433a035929ce0279e0b6/diffusion_models/minimax_h3_fl2va_pruned_int8_convrot.safetensors)

在装有 NumPy 的环境运行提取脚本，把生成文件放在模型根目录的 `h3/fl-union-table.npy`。脚本只从固定版本读取文件头和所需系数，可用 `--checkpoint` 改读本机已有的同款裁剪权重。Windows 也可设置 `MMD_H3_UNION_TABLE` 指向生成的文件。

```sh
python tools/extract_h3_union_table.py models/h3/fl-union-table.npy
```

H3 使用同一个人物检测器路径 `vr360-a/detectors/person.onnx`。快速档可以只配置普通 NVFP4、VAE 和编码器。

## 导入自己的外围扩画模型

在模型根目录下新建 `outpaint/<模型 ID>/`，然后放入 `model.json` 和配套权重。Docker 中对应 `/models/outpaint/`。Windows 使用 `NUNIF_HOME` 指定的模型目录，默认是项目下的 `.runtime/models`。模型 ID 使用小写英文字母、数字、下划线或连字符，首字符为字母或数字。

按下面的目录放文件。配置里填写文件名，文件名可自行更换。

| 目录 | 配置键 | 用途 |
| --- | --- | --- |
| `diffusion_models/` | `checkpoint` | 基础模型。LTX 支持适配的 safetensors 或 GGUF，H3 使用兼容 FL2VA 工作流的 safetensors。 |
| `vae/` | `video_vae` | 与底模匹配的视频 VAE。 |
| `vae/` | `audio_vae` | LTX 声画联合推理开启时填写。 |
| `text_encoders/` | `text_encoder` | 与底模匹配的文本编码器。LTX‑2.5 使用含投影的一体权重。 |
| `text_encoders/` | `text_projection` | LTX‑2.3 需要单独填写文本投影权重。 |
| `loras/` | `outpaint_lora` | LTX 需要与模型和 VR360 扩画工作流兼容的 IC-LoRA。 |

例如 `outpaint/my-ltx25/model.json` 可以写成：

```json
{
  "label": "我的 LTX-2.5 扩画",
  "family": "ltx25",
  "assets": {
    "checkpoint": "my-ltx25.gguf",
    "video_vae": "my-video-vae.safetensors",
    "text_encoder": "my-text-encoder.safetensors",
    "outpaint_lora": "my-outpaint-lora.safetensors"
  },
  "features": ["source_mask", "front_only"],
  "settings": {
    "steps": 8,
    "cfg": 1.0
  }
}
```

`family` 选择 `ltx2`（LTX‑2.3）、`ltx25`（LTX‑2.5）或 `h3`（H3 FL2VA）。目录导入会复用选定的工作流，其他架构需要另写适配。LTX 使用 GGUF 时仍需按上文安装加载器。

`features` 声明权重兼容的功能。省略或留空时，表内功能均关闭，WebUI 隐藏相应控件。启用的功能仍可通过页面或 `settings` 调整。Union 使用当前工作流对应的控制权重，人物检测和深度图沿用已配置的公共依赖。

| 功能 | 可用于 |
| --- | --- |
| `union`、`source_mask`、`background_extension`、`palette` | LTX 与 H3 |
| `vae_tiling`、`front_only`、`joint_audio` | LTX |

`settings` 固定指定的参数，例如 LTX 的 `steps`、`sampler`，或 H3 的 `h3_size`、`h3_steps`。固定参数通过配置文件修改，其他兼容参数继续用 WebUI。快速档和质量档按钮用于内置模型。

重启 WebUI 后，在「外围生成模型」下拉框选择带「自定义」标记的模型。缺少文件或配置无效时，该条目会被跳过，原因写入 WebUI 日志。入队任务使用提交时的配置。更换同名权重后重新提交；批处理中已用相同模型配置和权重版本完成的成片会跳过。
