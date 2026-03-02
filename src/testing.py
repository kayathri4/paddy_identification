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
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, classification_report

def load_config():
    """Finds the config.yaml in the project root."""
    # Assumes script is in /src, so parent of parent is root
    root_dir = Path(__file__).resolve().parent.parent
    config_path = root_dir / "config.yaml"
    
    with open(config_path, "r") as f:
        return yaml.safe_load(f)

def run_inference(config=None):
    """Executes semantic segmentation using parameters from config."""
    if config is None:
        config = load_config()

    # Extract sections for readability
    inf_cfg = config['inference']
    train_cfg = config['training']
    data_cfg = config['data']

    print(f"🚀 Starting inference on: {inf_cfg['test_image']}")
    
    # Run geoai semantic segmentation
    geoai.semantic_segmentation(
        input_path=inf_cfg['test_image'],
        output_path=inf_cfg['masks_path'],
        model_path=inf_cfg['model_path'],
        architecture=train_cfg['architecture'],
        encoder_name=train_cfg['encoder'],
        num_channels=data_cfg['num_channels'],
        num_classes=2,
        window_size=inf_cfg['window_size'],
        overlap=inf_cfg['overlap'],
        batch_size=inf_cfg['batch_size'],
    )
    
    print(f"✅ Prediction saved to: {inf_cfg['masks_path']}")

def calculate_metrics(ground_truth_path, prediction_path):
    """
    Generates Confusion Matrix and Accuracy Report
    """
    with rasterio.open(ground_truth_path) as gt_src, rasterio.open(prediction_path) as pred_src:
        y_true = gt_src.read(1).flatten()
        y_pred = pred_src.read(1).flatten()

    # Filter out NoData if necessary (assuming 0 or 255 might be nodata)
    mask = (y_true <= 1) & (y_pred <= 1) 
    y_true = y_true[mask]
    y_pred = y_pred[mask]

    # Generate Confusion Matrix
    cm = confusion_matrix(y_true, y_pred)
    
    # Plotting
    plt.figure(figsize=(8,6))
    plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Greens)
    plt.title("Confusion Matrix: Rice vs Non-Rice")
    plt.colorbar()
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.savefig("plots/test_confusion_matrix.png")
    
    # Save Text Report
    report = classification_report(y_true, y_pred, target_names=['Non-Rice', 'Rice'])
    with open("plots/test_report.txt", "w") as f:
        f.write(report)
    
    print("📊 Evaluation Metrics Saved to /plots directory.")


    

if __name__ == "__main__":    
    # Execute
    run_inference()
    # If you have a corresponding label for tile_000009, run this:
    calculate_metrics("G:/data/test/labels/tile_000009.tif", config['inference']['output_mask_path'])
