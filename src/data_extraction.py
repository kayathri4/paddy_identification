import fiona
import geopandas as gpd
import rioxarray
from shapely.geometry import mapping
import yaml
from pathlib import Path
import os
import glob
from pyroSAR.snap import geocode
from pyroSAR import identify

def crop_sar_to_roi():

    # Load configuration
    ROOT_DIR = Path(__file__).resolve().parent.parent
    CONFIG_PATH = ROOT_DIR / "config.yaml"
    with open(CONFIG_PATH, 'r') as f:
        config = yaml.safe_load(f)
        print(config)

    # Extract paths from YAML
    raw_dir = os.path.join(ROOT_DIR,config['data']['raw_zip_dir'])
    output_dir = os.path.join(ROOT_DIR,config['data']['processed_dir'])
    kml_path = os.path.join(ROOT_DIR,config['data']['roi_kml'])
    if not os.path.exists(output_dir):
        print(f"Creating output directory: {output_dir}")
        os.makedirs(output_dir, exist_ok=True)
    print(kml_path)

    # 1. Setup Environment
    fiona.drvsupport.supported_drivers['KML'] = 'rw'
    roi_gdf = gpd.read_file(kml_path, driver='KML')

    safe_folders = [f for f in glob.glob(os.path.join(raw_dir, "*")) if os.path.isdir(f)]
    print(f"Found {len(safe_folders)} potential SAR folders in {raw_dir}")

    for safe_folder in safe_folders:
        folder_name = os.path.basename(safe_folder)
        print(f"Processing directory: {folder_name}...")
        raster_files = glob.glob(os.path.join(safe_folder, "**", "*.tif"), recursive=True)

        for raster_path in raster_files:
            filename = os.path.basename(raster_path)
            output_filename = f"{folder_name}_{filename}"
            output_path = os.path.join(output_dir, output_filename)
            print(output_path)

            try:
                with rioxarray.open_rasterio(raster_path) as src:
                    # Match CRS to the raster
                    roi_projected = roi_gdf.to_crs(src.rio.crs)
                    # Crop to ROI
                    cropped = src.rio.clip(
                        roi_projected.geometry.apply(mapping),
                        roi_projected.crs
                    )
                    cropped.rio.to_raster(output_path)
                    print(f"   [Saved]: {output_filename}")

            except Exception as e:
                print(f"   [Error] skipping {filename}: {e}")

    print("\n--- ROI Cropping Complete for all folders ---")




def process_s1_batch( gpt_path=None):
    """
    Processes all Sentinel-1 scenes in a folder using pyroSAR and SNAP.
    """

    # 0.Load configuration
    ROOT_DIR = Path(__file__).resolve().parent.parent
    CONFIG_PATH = ROOT_DIR / "config.yaml"
    with open(CONFIG_PATH, 'r') as f:
        config = yaml.safe_load(f)
        print(config)

    output_folder = os.path.join(ROOT_DIR, config['data']['processed_dir'])
    input_folder = os.path.join(ROOT_DIR, config['data']['raw_zip_dir'])
    kml_path = os.path.join(ROOT_DIR, config['data']['roi_kml'])
    # 1. Environment Setup

    if gpt_path:
        os.environ['SNAP_GPT_EXECUTABLE'] = gpt_path

    if not os.path.exists(f"{output_folder}/preprocessed"):
        os.makedirs(f"{output_folder}/preprocessed")

    # 2. Identify all zip and SAFE files
    input_bundles = glob.glob(os.path.join(input_folder, "S1*.zip")) + \
                    glob.glob(os.path.join(input_folder, "S1*.SAFE"))

    print(f"Total scenes found: {len(input_bundles)}")

    # 3. Execution Loop
    for bundle in input_bundles:
        try:
            scene = identify(bundle)
            scene_name = scene.outname_base()

            # Check if processing is already done (Optional but recommended)
            # This looks for the directory or file starting with the scene name
            if any(scene_name in f for f in os.listdir(output_folder)):
                print(f"Skipping {scene_name} - Output already exists.")
                continue

            print(f"\n>>> Processing: {scene_name}")
            geocode(
                infile=bundle,
                outdir=output_folder,
                speckleFilter='Refined Lee',
                t_srs=4326,
                spacing=10,
                scaling='db',
                shapefile=kml_path,
                removeS1BorderNoise=True,
                removeS1BorderNoiseMethod='pyroSAR',
                removeS1ThermalNoise=True,
                demResamplingMethod='BILINEAR_INTERPOLATION',
                demName='SRTM 1Sec HGT',
                cleanup=True  # Set to True to save disk space after each run
            )
            print(f"Done: {scene_name}")

        except Exception as e:
            print(f"Error processing {bundle}: {e}")




if __name__ == "__main__":
    crop_sar_to_roi()

    Fix PROJ_LIB for GDAL
    proj_lib = r"D:\cv_project\sar_prj\venv\Lib\site-packages\osgeo\data\proj"
    if os.path.exists(proj_lib):
        os.environ['PROJ_LIB'] = proj_lib

    definin path of SNAP tool
    GPT = r"C:\Program Files\esa-snap\bin\gpt.exe"
    process_s1_batch(gpt_path=GPT)