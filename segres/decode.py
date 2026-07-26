import json
import numpy as np
from pycocotools import mask as mask_utils


def load_annotations(json_path):
    with open(json_path, "r") as f:
        data = json.load(f)
    return data


def build_lookup_tables(data):
    """Build fast lookup dicts for images and annotations by image_id."""
    images_by_id = {img["id"]: img for img in data["images"]}

    anns_by_image_id = {}
    for ann in data["annotations"]:
        anns_by_image_id.setdefault(ann["image_id"], []).append(ann)

    return images_by_id, anns_by_image_id


def polygon_to_mask(segmentation, height, width):
    """
    Convert a single polygon segmentation to a binary mask.

    segmentation: list of floats [x0, y0, x1, y1, ..., x0, y0]
    height, width: image dimensions
    """
    rle = mask_utils.frPyObjects(segmentation, height, width)

    # merge in case frPyObjects returns multiple RLEs (e.g. multi-part polygon)
    if isinstance(rle, list):
        rle = mask_utils.merge(rle)

    binary_mask = mask_utils.decode(rle)
    return binary_mask


def build_file_to_annotations(data):
    """
    Map file_name -> annotations from a single annotator.

    The raw annotation file lists the same physical image under multiple
    image_ids when several annotators independently traced it (id suffix
    01/02/03 = annotator pass). Take only the first pass (lowest image_id)
    per file_name instead of merging all annotators' polygons together,
    which would union disagreeing traces into one inflated mask.
    """
    images_by_id, anns_by_image_id = build_lookup_tables(data)

    file_to_image_id = {}
    for image_id, image_info in sorted(images_by_id.items()):
        file_name = image_info["file_name"]
        file_to_image_id.setdefault(file_name, image_id)

    return {
        file_name: anns_by_image_id.get(image_id, [])
        for file_name, image_id in file_to_image_id.items()
    }


DEFAULT_CATEGORY_IDS = frozenset({1, 2})  # 1=Left, 2=Right; excludes 3=Unidentifiable, 4=Ambiguous


def build_combined_mask(annotations, height, width, category_ids=DEFAULT_CATEGORY_IDS):
    """
    Combine filament masks for one image into a single binary mask.

    category_ids: which annotation categories count as foreground. Default
    excludes category 3 "Unidentifiable" (37.5% of annotations) and 4
    "Ambiguous" (unused in this dataset) — deliberately, since those are
    cases the annotator themself couldn't confidently call a filament, and
    burning them into the mask unconditionally trains the model on uncertain
    labels. Pass category_ids=None to include everything (old behavior).
    Chirality (Left/Right, category 1 vs 2) is still collapsed into one
    binary mask here — this only decides what counts as foreground at all.
    """
    combined = np.zeros((height, width), dtype=np.uint8)

    for ann in annotations:
        if category_ids is not None and ann["category_id"] not in category_ids:
            continue
        seg = ann["segmentation"]
        m = polygon_to_mask(seg, height, width)
        combined = np.logical_or(combined, m).astype(np.uint8)

    return combined


def get_image_and_masks(json_path, image_id):
    """
    Convenience function. Loads annotations and returns everything needed
    to visualize one image: file name, height, width, and combined mask.
    """
    data = load_annotations(json_path)
    images_by_id, anns_by_image_id = build_lookup_tables(data)

    if image_id not in images_by_id:
        raise ValueError(f"image_id '{image_id}' not found in annotations.")

    image_info = images_by_id[image_id]
    height = image_info["height"]
    width = image_info["width"]
    file_name = image_info["file_name"]

    annotations = anns_by_image_id.get(image_id, [])
    print(f"Found {len(annotations)} filament annotation(s) for image {image_id}")

    if len(annotations) == 0:
        return file_name, height, width, None

    combined_mask = build_combined_mask(annotations, height, width)
    return file_name, height, width, combined_mask