import os
import glob
import re
import yaml
import rasterio
from pathlib import Path
from collections import defaultdict
import numpy as np


def stack_sar_timeseries():
    # 1. Load configuration
    ROOT_DIR = Path(__file__).resolve().parent.parent
    CONFIG_PATH = ROOT_DIR / "config.yaml"

    with open(CONFIG_PATH, 'r') as f:
        config = yaml.safe_load(f)

    # Extract paths from YAML (using .strip() to avoid newline issues)
    input_dir = ROOT_DIR / config['data']['processed_dir'].strip()
    output_file = ROOT_DIR / config['data']['stack_output'].strip()

    # Ensure the parent directory for the output stack exists
    output_file.parent.mkdir(parents=True, exist_ok=True)

    print(f"Searching for TIFs in: {input_dir}")

    # 2. Group files by date
    # Look for VV and VH pairs in the processed directory
    all_tifs = list(input_dir.glob("*.tif"))
    date_map = defaultdict(dict)

    for f_path in all_tifs:
        filename = f_path.name
        # Regex to find YYYYMMDD
        match = re.search(r'\d{8}', filename)
        if not match:
            continue

        date = match.group(0)
        if "_VV_" in filename:
            date_map[date]['vv'] = str(f_path)
        elif "_VH_" in filename:
            date_map[date]['vh'] = str(f_path)

    sorted_dates = sorted(date_map.keys())
    if not sorted_dates:
        print("No valid VV/VH pairs found. Check your file naming and input directory.")
        return

    # 3. Initialize Metadata from the first valid VV file
    first_vv = date_map[sorted_dates[0]]['vv']
    with rasterio.open(first_vv) as src:
        meta = src.meta.copy()
        # Each date adds 3 bands: VV, VH, and VV-VH (Ratio/Difference)
        total_bands = len(sorted_dates) * 3
        meta.update(
            count=total_bands,
            dtype='float32',
            compress='lzw',
            nodata=0
        )

    # 4. Sequential Stacking
    print(f"Creating stack with {total_bands} bands at: {output_file}")

    with rasterio.open(output_file, 'w', **meta) as dst:
        band_idx = 1

        for date in sorted_dates:
            vv_path = date_map[date].get('vv')
            vh_path = date_map[date].get('vh')

            if not vv_path or not vh_path:
                print(f"Skipping {date}: Incomplete pair.")
                band_idx += 3
                continue

            print(f"   [Stacking] {date}...")
            with rasterio.open(vv_path) as vv_src, rasterio.open(vh_path) as vh_src:
                vv_data = vv_src.read(1).astype('float32')
                vh_data = vh_src.read(1).astype('float32')

                # Calculate Ratio (Difference in dB space is equivalent to ratio in linear)
                # Ensure we handle nodata/zeros if necessary
                ratio_data = vv_data - vh_data

                # Write Bands
                dst.write(vv_data, band_idx)
                dst.write(vh_data, band_idx + 1)
                dst.write(ratio_data, band_idx + 2)

                # Set Metadata Descriptions
                dst.set_band_description(band_idx, f"{date}_VV")
                dst.set_band_description(band_idx + 1, f"{date}_VH")
                dst.set_band_description(band_idx + 2, f"{date}_Ratio")

            band_idx += 3

    print(f"\n--- Stacking Complete: {output_file} ---")



def generate_stack_statistics(config):
    # 1. Setup Paths
    ROOT_DIR = Path(__file__).resolve().parent.parent
    CONFIG_PATH = ROOT_DIR / config

    with open(CONFIG_PATH, 'r') as f:
        config = yaml.safe_load(f)

    stack_path = ROOT_DIR / config['data']['stack_output'].strip()
    print(stack_path)
    log_path = stack_path.with_suffix('.stats.log')
    print(log_path)

    if not stack_path.exists():
        print(f"Error: Stacked file not found at {stack_path}")
        return

    print(f"Calculating statistics for: {stack_path.name}...")

    # 2. Compute Stats per Band
    with rasterio.open(stack_path) as src:
        band_names = src.descriptions
        with open(log_path, 'w') as log:
            log.write(f"Statistical Summary for {stack_path.name}\n")
            log.write("=" * 50 + "\n\n")

            for i in range(1, src.count + 1):
                band_data = src.read(i)
                band_name = band_names[i - 1] if band_names[i - 1] else f"Band_{i}"

                # Filter out NoData (assuming 0 or use src.nodata)
                nodata_val = src.nodata if src.nodata is not None else 0
                valid_data = band_data[band_data != nodata_val]

                if valid_data.size == 0:
                    log.write(f"Band {i} ({band_name}): No valid data found.\n\n")
                    continue

                # Calculate stats
                stats = {
                    "MINIMUM": np.min(valid_data),
                    "MAXIMUM": np.max(valid_data),
                    "MEAN": np.mean(valid_data),
                    "STDDEV": np.std(valid_data),
                    "VALID_PERCENT": (valid_data.size / band_data.size) * 100
                }

                # 3. Write in the requested format
                log.write(f"--- Band {i}: {band_name} ---\n")
                for key, value in stats.items():
                    log.write(f"STATISTICS_{key}={value:.12f}\n")
                log.write("\n")

    print(f"Success! Statistics saved to: {log_path}")


def normalize_sar_stack():
    # 1. Setup Paths
    ROOT_DIR = Path(__file__).resolve().parent.parent
    CONFIG_PATH = ROOT_DIR / "config.yaml"

    with open(CONFIG_PATH, 'r') as f:
        config = yaml.safe_load(f)

    input_path = ROOT_DIR / config['data']['stack_output'].strip()
    output_path = ROOT_DIR / config['data']['norm_output'].strip()

    if not input_path.exists():
        print(f"Error: Input stack not found at {input_path}")
        exit(1)

    # Ensure output directory exists (crucial for DVC)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # 2. Processing
    with rasterio.open(input_path) as src:
        profile = src.profile
        profile.update(dtype='float32', nodata=0, compress='lzw')

        with rasterio.open(output_path, 'w', **profile) as dst:
            for i in range(1, src.count + 1):
                band_name = src.descriptions[i - 1] if src.descriptions else f"Band_{i}"
                print(f"Normalizing {band_name}...")

                data = src.read(i).astype('float32')
                mask = (data != 0) & (~np.isnan(data))

                if np.any(mask):
                    # Robust scaling using percentiles
                    band_min = np.percentile(data[mask], 2)
                    band_max = np.percentile(data[mask], 98)

                    # Avoid division by zero if a band is constant
                    if band_max == band_min:
                        norm_data = np.zeros_like(data)
                    else:
                        # Min-Max Scaling: $x_{norm} = \frac{x - min}{max - min}$
                        norm_data = np.clip(data, band_min, band_max)
                        norm_data = (norm_data - band_min) / (band_max - band_min)

                    # Re-apply mask to preserve NoData
                    norm_data[~mask] = 0
                    dst.write(norm_data, i)
                else:
                    dst.write(data, i)

                dst.set_band_description(i, f"Norm_{band_name}")

    print(f"Normalization complete. File saved to: {output_path}")



if __name__ == "__main__":
    stack_sar_timeseries()
    generate_stack_statistics(config="config.yaml")
    normalize_sar_stack()

