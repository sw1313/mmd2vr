from nodes import SaveImage
from .iclora import (
    LTXAddVideoICLoRAGuide,
    LTXAddVideoICLoRAGuideAdvanced,
    LTXICLoRALoaderModelOnly,
)
from .depth_mask import VR360DepthMask

class VR360SaveFrames(SaveImage):
    @classmethod
    def INPUT_TYPES(cls):
        inputs = super().INPUT_TYPES()
        inputs.setdefault("optional", {})["compression"] = ("INT", {"default": 0, "min": 0, "max": 9})
        return inputs

    def save_images(self, images, filename_prefix="ComfyUI", prompt=None, extra_pnginfo=None, compression=0):
        self.compress_level = compression
        return super().save_images(images, filename_prefix, prompt, extra_pnginfo)


NODE_CLASS_MAPPINGS = {
    node.__name__: node
    for node in (
        LTXAddVideoICLoRAGuide,
        LTXAddVideoICLoRAGuideAdvanced,
        LTXICLoRALoaderModelOnly,
        VR360DepthMask,
        VR360SaveFrames,
    )
}
