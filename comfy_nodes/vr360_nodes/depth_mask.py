from pathlib import Path
import numpy as np
import torch
from torch.nn.functional import interpolate
import folder_paths


class VR360DepthMask:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {"name": ("STRING",)},
            "optional": {"width": ("INT", {"default": 1280, "min": 64, "max": 1280, "step": 32})},
        }

    RETURN_TYPES = ("MASK",)
    FUNCTION = "load"
    CATEGORY = "VR360"

    def load(self, name, width=1280):
        if Path(name).name != name or not name.endswith(".npy"):
            raise ValueError("无效掩码文件名")
        masks = np.load(
            Path(folder_paths.get_input_directory()) / name, allow_pickle=False
        )
        tensor = torch.from_numpy(masks)[:, None]
        return (interpolate(tensor, (640, width), mode="nearest")[:, 0],)
