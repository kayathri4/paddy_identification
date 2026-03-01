# 🌾 Paddy Field Instance Segmentation using Multi-Temporal SAR Time Series

## 📖 Project Overview
Monitoring rice paddy fields is critical for food security, water management, and methane emission tracking. However, traditional optical satellite imagery (like Sentinel-2) is often unusable in tropical and subtropical regions due to persistent cloud cover during the monsoon cropping season.

This project implements an end-to-end Deep Learning pipeline to automatically segment paddy fields using Sentinel-1 SAR (Synthetic Aperture Radar) Ground Range Detected (GRD) data. By leveraging the unique backscatter temporal signature of rice during its flooding and growth stages, the model achieves high-precision mapping regardless of weather conditions.


## 🛠️ Key Technical Features
* **Multi-Temporal Fusion**: Processes a 60-band data stack representing a 20-date cropping season. Each date includes VV, VH, and VV/VH ratio polarizations to capture the "double-bounce" scattering effect characteristic of rice stems in water.
* **Geospatial Preprocessing**: Integration of ESA SNAP (pyroSAR) for specialized SAR calibration, thermal noise removal, and terrain correction (orthorectification).
* **MLOps Architecture**:
    * **DVC (Data Version Control)**: Manages heavy 60-band GeoTIFF stacks and model weights without bloating the Git repository.
    * **MLflow**: Tracks hyperparameter experiments, loss curves, and evaluation metrics (IoU, F1-Score).
* **High Performance**: Optimized a U-Net + ResNet34 architecture to achieve a 0.85 IoU, successfully filtering "speckle noise" inherent in SAR data.

## 🛰️ Data Sources
* **Sentinel-1 SAR**: Multi-temporal C-band data (20 dates).
* **JAXA LULC Map**: Used for automated ground-truth label generation (Rice vs. Non-Rice).
* **ROI**: Niigata, Japan (High-intensity rice production region).

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
