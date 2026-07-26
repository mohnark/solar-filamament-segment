import os
 
import numpy as np
import torch
from torch.utils.data import DataLoader
from PIL import Image
from pycocotools import mask as mask_utils
from scipy import ndimage
import pandas as pd
 
from dataset import SolarFilamentDataset
from transform import get_val_transform
from model import build_model
from tiling import get_tile_coordinates, stitch_predictions
 
 
def load_model(checkpoint_path, device):
    model = build_model()
    model.load_state_dict(torch.load(checkpoint_path, map_location=device))
    model.to(device)
    model.eval()
    return model
 
 
def predict_tiles(model, loader, device):
    """
    Run the model on every test tile, and collect predictions grouped
    by file_name, so they can be stitched back together per image.
 
    Returns: dict {file_name: {"tiles": [...], "coords": [...]}}
    """
    results = {}
 
    with torch.no_grad():
        for images, file_names, ys, xs in loader:
            images = images.to(device)
            preds = model(images)
            preds = torch.sigmoid(preds).cpu().numpy()  # (B, 1, H, W)
 
            for i in range(len(file_names)):
                file_name = file_names[i]
                y = ys[i].item()
                x = xs[i].item()
                pred_tile = preds[i, 0]  # (H, W)
 
                if file_name not in results:
                    results[file_name] = {"tiles": [], "coords": []}
 
                results[file_name]["tiles"].append(pred_tile)
                results[file_name]["coords"].append((y, x))
 
    return results
 
 
def clean_mask(binary_mask, min_area=20):
    """
    Remove tiny isolated specks from the predicted mask.
    This helps with the fragmentation penalty mentioned in the evaluation rubric.
    """
    labeled, num_features = ndimage.label(binary_mask)
    cleaned = np.zeros_like(binary_mask)
 
    for label_id in range(1, num_features + 1):
        component = labeled == label_id
        if component.sum() >= min_area:
            cleaned[component] = 1
 
    return cleaned.astype(np.uint8)
 
 
def split_into_instances(binary_mask):
    """
    Split one binary mask into separate connected-component instances,
    since the submission expects one row per filament, not one row per image.
    """
    labeled, num_features = ndimage.label(binary_mask)
    instances = []
 
    for label_id in range(1, num_features + 1):
        instance_mask = (labeled == label_id).astype(np.uint8)
        instances.append(instance_mask)
 
    return instances
 
 
def mask_to_rle_string(binary_mask):
    """Convert a binary mask to an RLE counts string, per the submission format."""
    rle = mask_utils.encode(np.asfortranarray(binary_mask))
    counts = rle["counts"]
    if isinstance(counts, bytes):
        counts = counts.decode("utf-8")
    return counts
 
 
def build_submission(results, threshold, min_area, output_csv, tile_size):
    rows = []
 
    for file_name, data in results.items():
        pred_tiles = data["tiles"]
        coords = data["coords"]
 
        # infer full image size from the max tile extents
        max_y = max(y for (y, x) in coords) + tile_size
        max_x = max(x for (y, x) in coords) + tile_size
 
        stitched = stitch_predictions(pred_tiles, coords, max_y, max_x, tile_size)
        binary_mask = (stitched > threshold).astype(np.uint8)
        binary_mask = clean_mask(binary_mask, min_area=min_area)
 
        instances = split_into_instances(binary_mask)
 
        base_name = os.path.splitext(file_name)[0]
        for idx, instance_mask in enumerate(instances, start=1):
            filament_id = f"{base_name}_{idx}"
            rle_string = mask_to_rle_string(instance_mask)
            rows.append({"filament_id": filament_id, "segmentation_rle": rle_string})
 
    df = pd.DataFrame(rows)
    df.to_csv(output_csv, index=False)
    print(f"Saved submission with {len(df)} rows to {output_csv}")