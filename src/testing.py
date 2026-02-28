import os
import platform

# Manually load c10.dll to bypass the PyTorch 2.9 WinError 1114 bug
if platform.system() == "Windows":
    import ctypes
    from importlib.util import find_spec
    try:
        if (spec := find_spec("torch")) and spec.origin and os.path.exists(
            dll_path := os.path.join(os.path.dirname(spec.origin), "lib", "c10.dll")
        ):
            ctypes.CDLL(os.path.normpath(dll_path))
    except Exception:
        pass

import torch
import yaml
import geoai
import rasterio
from pathlib import Path
from pathlib import Path


test_image = "G:\\data\\test\\image\\tile_000009.tif"
model_path = "G:\\data\\models\\best_model.pth"
masks_path = "G:\\data\\models\\test_9.tif"



geoai.semantic_segmentation(
    input_path=test_image,
    output_path=masks_path,
    model_path=model_path,
    architecture="unet",
    encoder_name="resnet34",
    num_channels=50,
    num_classes=2,
    window_size=512,
    overlap=256,
    batch_size=4,
)
