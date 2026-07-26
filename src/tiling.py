import numpy as np
 
 
def get_tile_coordinates(height, width, tile_size=512, overlap=64):
    """
    Compute top-left (y, x) coordinates for tiles covering the full image,
    with some overlap so filaments crossing tile borders aren't cut cleanly
    without any shared context.
 
    Returns a list of (y, x) tuples.
    """
    stride = tile_size - overlap
    coords = []
 
    y = 0
    while y < height:
        x = 0
        while x < width:
            # clamp so the last tile in each row/column stays inside bounds
            y_clamped = min(y, height - tile_size) if height > tile_size else 0
            x_clamped = min(x, width - tile_size) if width > tile_size else 0
            coords.append((y_clamped, x_clamped))
 
            if x_clamped + tile_size >= width:
                break
            x += stride
 
        if y_clamped + tile_size >= height:
            break
        y += stride
 
    # remove duplicates (can happen at clamped edges)
    coords = sorted(set(coords))
    return coords
 
 
def extract_tile(array, y, x, tile_size=512):
    """Extract a single tile from a 2D array (image or mask).
 
    np.ascontiguousarray is important here: slicing alone returns a view,
    and OpenCV-based operations (used internally by Albumentations) can
    fail silently on non-contiguous arrays.
    """
    tile = array[y:y + tile_size, x:x + tile_size]
    return np.ascontiguousarray(tile)
 
 
def tile_image_and_mask(image, mask, tile_size=512, overlap=64):
    """
    Split one image and its matching mask into a list of tile pairs.
 
    image, mask: 2D numpy arrays, same height/width
    Returns: list of (image_tile, mask_tile, y, x) tuples
    """
    height, width = image.shape[:2]
    coords = get_tile_coordinates(height, width, tile_size, overlap)
 
    tiles = []
    for (y, x) in coords:
        image_tile = extract_tile(image, y, x, tile_size)
        mask_tile = extract_tile(mask, y, x, tile_size)
        tiles.append((image_tile, mask_tile, y, x))
 
    return tiles
 
 
def stitch_predictions(pred_tiles, coords, height, width, tile_size=512):
    """
    Reassemble predicted tiles into a full-resolution mask.
 
    pred_tiles: list of 2D numpy arrays (model output per tile), same order as coords
    coords: list of (y, x) tuples, matching get_tile_coordinates output
    height, width: full image dimensions
 
    Overlapping regions are averaged, not just overwritten, so predictions
    near tile borders blend smoothly instead of showing seams.
    """
    full_pred = np.zeros((height, width), dtype=np.float32)
    count_map = np.zeros((height, width), dtype=np.float32)
 
    for pred_tile, (y, x) in zip(pred_tiles, coords):
        full_pred[y:y + tile_size, x:x + tile_size] += pred_tile
        count_map[y:y + tile_size, x:x + tile_size] += 1.0
 
    # avoid division by zero, though every pixel should be covered at least once
    count_map[count_map == 0] = 1.0
    full_pred = full_pred / count_map
 
    return full_pred
 