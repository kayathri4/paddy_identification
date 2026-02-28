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

hard_path = Path("G:/data")
def run_training_pipeline():
    # 1. Setup Paths
    ROOT_DIR = Path(__file__).resolve().parent.parent
    CONFIG_PATH = ROOT_DIR / "config.yaml"

    with open(CONFIG_PATH, 'r') as f:
        config = yaml.safe_load(f)

    # Resolve paths from config
    train_raster = ROOT_DIR / config['data']['norm_output'].strip()
    label_geojson = ROOT_DIR / config['data']['label_geojson'].strip()
    out_tile_folder = hard_path / "data" / "paddy_instance"
    model_output_dir = hard_path / config['paths']['output_model_dir'].strip()

    # Create directories
    out_tile_folder.mkdir(parents=True, exist_ok=True)
    model_output_dir.mkdir(parents=True, exist_ok=True)

    # 2. Determine Channel Count from Raster
    with rasterio.open(train_raster) as src:
        actual_channels = src.count
    print(f"Detected {actual_channels} channels in input stack.")

    # 3. Export Tiff Tiles (Patching)
    # print("Generating training tiles...")
    # tiles = geoai.export_geotiff_tiles(
    #     in_raster=str(train_raster),
    #     out_folder=str(out_tile_folder),
    #     in_class_data=str(label_geojson),
    #     tile_size=config['training']['tile_size'],
    #     stride=config['training']['stride'],
    #     buffer_radius=0,
    # )

    # # 4. Train Segmentation Model
    print(f"Starting {config['training']['architecture']} training...")
    geoai.train_segmentation_model(
        images_dir=f"{out_tile_folder}/images",
        labels_dir=f"{out_tile_folder}/labels",
        output_dir=str(model_output_dir),
        architecture=config['training']['architecture'],
        encoder_name=config['training']['encoder'],
        encoder_weights=config['training']['encoder_weights'],
        num_classes=2,  # Background + Paddy
        num_channels=actual_channels,
        batch_size=config['training']['batch_size'],
        num_epochs=config['training']['epochs'],
        learning_rate=config['training']['learning_rate'],
        val_split=0.2,
        visualize=True,
        verbose=True,
    )


if __name__ == "__main__":
    run_training_pipeline()