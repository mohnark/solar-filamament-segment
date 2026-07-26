import os
import numpy as np
import torch
from pycocotools import mask as mask_utils
from scipy import ndimage
import pandas as pd
from .model import build_model
from .tiling import stitch_predictions


def load_model(checkpoint_path, device):
    # encoder_weights=None: checkpoint overwrites these anyway, and the
    # imagenet download hard-fails in a no-internet Kaggle inference kernel
    model = build_model(encoder_weights=None)
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


def label_and_filter(binary_mask, min_area=1):
    """
    Label connected components once and drop tiny specks by area (helps with
    the fragmentation penalty mentioned in the evaluation rubric), returning
    both the cleaned binary mask and one instance mask per surviving
    component. Replaces separate clean_mask() + split_into_instances() calls,
    which used to label the same mask twice and, in clean_mask's case, did a
    full-image boolean compare per component instead of a single bincount.

    Returns: (cleaned_binary_mask, [instance_mask, ...])
    """
    labeled, num_features = ndimage.label(binary_mask)
    if num_features == 0:
        return np.zeros_like(binary_mask, dtype=np.uint8), []

    areas = np.bincount(labeled.ravel(), minlength=num_features + 1)
    keep_ids = np.nonzero(areas[1:] >= min_area)[0] + 1

    cleaned = np.isin(labeled, keep_ids).astype(np.uint8)
    instances = [(labeled == label_id).astype(np.uint8) for label_id in keep_ids]

    return cleaned, instances


def mask_to_rle_string(binary_mask):
    """Convert a binary mask to an RLE counts string, per the submission format."""
    rle = mask_utils.encode(np.asfortranarray(binary_mask))
    counts = rle["counts"]
    if isinstance(counts, bytes):
        counts = counts.decode("utf-8")
    return counts


def build_submission(results, threshold, min_area, output_csv, tile_size, image_height, image_width):
    rows = []

    for file_name, data in results.items():
        pred_tiles = data["tiles"]
        coords = data["coords"]

        stitched = stitch_predictions(pred_tiles, coords, image_height, image_width, tile_size)
        binary_mask = (stitched > threshold).astype(np.uint8)
        _, instances = label_and_filter(binary_mask, min_area=min_area)

        base_name = os.path.splitext(file_name)[0]
        for idx, instance_mask in enumerate(instances, start=1):
            filament_id = f"{base_name}_{idx}"
            rle_string = mask_to_rle_string(instance_mask)
            rows.append({"filament_id": filament_id, "segmentation_rle": rle_string})

    df = pd.DataFrame(rows)
    df.to_csv(output_csv, index=False)
    print(f"Saved submission with {len(df)} rows to {output_csv}")
