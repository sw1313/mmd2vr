from .minimax_lowmem import MiniMaxLowVRAMAttention, MiniMaxChunkFeedForward
from .sphere import H3SphereOutpaint
import os

NODE_CLASS_MAPPINGS = {cls.__name__: cls for cls in
                       (MiniMaxLowVRAMAttention, MiniMaxChunkFeedForward, H3SphereOutpaint)}

if os.environ.get("MMD_H3_PROFILE") == "1":
    import json
    import time
    import torch
    from nodes import CLIPLoader, CLIPTextEncode, UNETLoader, VAELoader

    def timed_loader(cls):
        execute = getattr(cls, cls.FUNCTION)
        def measured(self, *args, **kwargs):
            torch.cuda.synchronize()
            start = time.perf_counter()
            result = execute(self, *args, **kwargs)
            torch.cuda.synchronize()
            print("H3_NODE " + json.dumps(dict(node=cls.__name__, seconds=time.perf_counter()-start)), flush=True)
            return result
        return type("H3Timed" + cls.__name__, (cls,), {cls.FUNCTION: measured})

    for loader in (CLIPLoader, CLIPTextEncode, UNETLoader, VAELoader):
        timed = timed_loader(loader)
        NODE_CLASS_MAPPINGS[timed.__name__] = timed
