import numpy as np
 
 
def _axis_coordinates(size, tile_size, stride):
    """
    Top-left coordinates along one axis, evenly spaced to exactly cover
    [0, size) with `tile_size`-wide tiles.

    A naive fixed-stride walk that clamps the last tile to fit inside bounds
    leaves a short final stride wherever (size - tile_size) isn't a multiple
    of stride — e.g. 2048/512/overlap=64 gives strides {448, 448, 448, 448,
    192}, so the last band is ~57% redundant with its neighbor while the
    others aren't. Spacing all tiles evenly via linspace spreads that
    leftover across every gap instead of dumping it all on one edge.
    """
    if size <= tile_size:
        return [0]

    n_tiles = int(np.ceil((size - tile_size) / stride)) + 1
    positions = np.linspace(0, size - tile_size, n_tiles)
    # round + dedup: linspace can produce repeats when n_tiles is large
    # relative to the span, though not for any tile/overlap combo used here
    return sorted({int(round(p)) for p in positions})


def get_tile_coordinates(height, width, tile_size=512, overlap=64):
    """
    Compute top-left (y, x) coordinates for tiles covering the full image,
    with some overlap so filaments crossing tile borders aren't cut cleanly
    without any shared context. Tiles are evenly spaced per axis (see
    _axis_coordinates) rather than fixed-stride-then-clamp, so overlap is
    uniform instead of concentrated in one edge band.

    Returns a list of (y, x) tuples.
    """
    stride = tile_size - overlap
    ys = _axis_coordinates(height, tile_size, stride)
    xs = _axis_coordinates(width, tile_size, stride)
    return [(y, x) for y in ys for x in xs]
 
 
def extract_tile(array, y, x, tile_size=512):
    """Extract a single tile from a 2D array (image or mask).
 
    np.ascontiguousarray is important here: slicing alone returns a view,
    and OpenCV-based operations (used internally by Albumentations) can
    fail silently on non-contiguous arrays.
    """
    tile = array[y:y + tile_size, x:x + tile_size]
    return np.ascontiguousarray(tile)
 
 
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
 