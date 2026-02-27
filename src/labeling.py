import os
import yaml
import glob
import numpy as np
import rasterio
import geopandas as gpd
from pathlib import Path
from rasterio.merge import merge
from rasterio.features import shapes
from shapely.geometry import shape


def process_labels():
    # 1. Setup Paths
    ROOT_DIR = Path(__file__).resolve().parent.parent
    CONFIG_PATH = ROOT_DIR / "config.yaml"
    with open(CONFIG_PATH, 'r') as f:
        config = yaml.safe_load(f)

    label_dir = ROOT_DIR / config['data']['label_src_dir']
    merged_path = ROOT_DIR / config['data']['label_merged_tif']
    binary_path = ROOT_DIR / config['data']['label_binary_tif']
    geojson_path = ROOT_DIR / config['data']['label_geojson']

    # Ensure output directories exist
    merged_path.parent.mkdir(parents=True, exist_ok=True)

    # 2. Merge tiles if necessary
    tile_files = glob.glob(str(label_dir / "*.tif"))
    if not tile_files:
        print(f"No label tiles found in {label_dir}")
        return

    if len(tile_files) > 1:
        print(f"Merging {len(tile_files)} tiles...")
        src_files_to_mosaic = [rasterio.open(f) for f in tile_files]
        mosaic, transform = merge(src_files_to_mosaic)

        meta = src_files_to_mosaic[0].meta.copy()
        meta.update({
            "height": mosaic.shape[1],
            "width": mosaic.shape[2],
            "transform": transform
        })

        with rasterio.open(merged_path, "w", **meta) as dst:
            dst.write(mosaic)
        for src in src_files_to_mosaic: src.close()
        current_label_source = merged_path
    else:
        print("Single tile detected, skipping merge.")
        current_label_source = tile_files[0]

    # 3. Convert to Binary Class
    print(f"Converting {current_label_source} to binary...")
    with rasterio.open(current_label_source) as src:
        data = src.read(1)
        meta = src.meta.copy()
        nodata = src.nodata

    # Create binary mask (Class 3 = 1, others = 0)
    binary = np.zeros_like(data, dtype=np.uint8)
    binary[data == 3] = 1

    if nodata is not None:
        binary[data == nodata] = 0  # Ensure nodata is clean

    meta.update(dtype="uint8", count=1, nodata=0)
    with rasterio.open(binary_path, "w", **meta) as dst:
        dst.write(binary, 1)

    # 4. Export to GeoJSON
    print("Vectorizing binary mask to GeoJSON...")
    with rasterio.open(binary_path) as src:
        image = src.read(1)
        mask = (image == 1)
        results = (
            {'properties': {'class': 1}, 'geometry': s}
            for i, (s, v) in enumerate(shapes(image, mask=mask, transform=src.transform))
        )
        gdf = gpd.GeoDataFrame.from_features(list(results), crs=src.crs)
        gdf.to_file(geojson_path, driver="GeoJSON")

    print(f"Labeling complete. GeoJSON saved at: {geojson_path}")


if __name__ == "__main__":
    process_labels()