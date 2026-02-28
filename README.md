# Paddy Identification using SAR Data and GeoAI

This project implements a deep learning pipeline for identifying paddy fields from Synthetic Aperture Radar (SAR) imagery. It leverages the `GeoAI` and `TorchGeo` libraries to process geospatial data and train semantic segmentation models.

## Project Overview

The core objective is to segment paddy fields from satellite data. The pipeline uses a UNet architecture with a ResNet34 encoder.

- **Architecture**: UNet
- **Encoder**: ResNet34
- **Input**: SAR Time-Series Data (Normalized/Filtered)
- **Output**: Segmentation Masks (Binary: Background vs. Paddy)

## Installation & Setup

### 1. Prerequisites
- Python 3.10 or 3.11
- NVIDIA GPU with CUDA support (Recommended)

### 2. Environment Setup
Construct a virtual environment and install the required dependencies:

```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
pip install -r requirements.txt
```

### 3. Critical Windows Fix (DLL Error)
On Windows systems, you might encounter `OSError: [WinError 1114]` when importing `torch`. This is resolved by manually loading `c10.dll`. The following snippet is included in the project scripts:

```python
import os
import platform
import ctypes
from importlib.util import find_spec

if platform.system() == "Windows":
    try:
        if (spec := find_spec("torch")) and spec.origin:
            dll_path = os.path.join(os.path.dirname(spec.origin), "lib", "c10.dll")
            if os.path.exists(dll_path):
                ctypes.CDLL(os.path.normpath(dll_path))
    except Exception:
        pass
```

### 4. Dependency Versions
We use specific stable versions to ensure compatibility on Windows:
- `torch==2.5.1+cu121`
- `torchvision==0.20.1+cu121`
- `torchgeo==0.6.2`

## Project Structure

- `src/`: Source code for the pipeline stages.
  - `training.py`: Handles tiling and model training.
  - `testing.py`: Runs inference on test images and generates visualizations.
  - `labeling.py`: Prepares binary labels from external data.
- `data/`: Data directory (standardized structure).
  - `raw/`: Unprocessed input data.
  - `processed/`: Features, stacks, and labels.
  - `external/`: External shapefiles and validation data.
- `config.yaml`: Centralized configuration for paths and training parameters.
- `dvc.yaml`: DVC pipeline definition.

## Pipeline Flow (DVC)

The project uses DVC to manage the workflow:

1. **Prepare Labels**: `python src/labeling.py`
   - Merges source TIFs and creates binary masks/geojson.
2. **Train Model**: `python src/training.py`
   - Generates tiles (patches) from input rasters and labels.
   - Trains the segmentation model.
3. **Inference (Experimental)**: `python src/testing.py`
   - Performs semantic segmentation on test images.
   - Orthogonalizes results and generates split-map visualizations.

## Configuration

Modify `config.yaml` to adjust training parameters:
- `tile_size`: Size of the input patches (default: 512).
- `epochs`: Number of training iterations.
- `batch_size`: Number of samples per training step.

## Data Storage

By default, large data files and models are stored in:
- Project root (`data/`, `models/`)
- External drive (if configured in `training.py`, e.g., `G:\data`)

## Usage

To run the full pipeline through DVC:
```bash
dvc repro
```

To run individual steps:
```bash
python src/training.py
python src/testing.py
```
