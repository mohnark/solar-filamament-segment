import numpy as np
from scipy import ndimage


def _otsu_threshold(img_uint8):
    """256-bin Otsu threshold: the intensity that maximizes between-class variance."""
    hist, _ = np.histogram(img_uint8, bins=256, range=(0, 256))
    hist = hist.astype(np.float64)
    total = hist.sum()
    sum_all = np.dot(np.arange(256), hist)

    sum_bg = 0.0
    w_bg = 0.0
    best_thresh = 0
    best_var = -1.0
    for t in range(256):
        w_bg += hist[t]
        if w_bg == 0:
            continue
        w_fg = total - w_bg
        if w_fg == 0:
            break
        sum_bg += t * hist[t]
        mean_bg = sum_bg / w_bg
        mean_fg = (sum_all - sum_bg) / w_fg
        between_var = w_bg * w_fg * (mean_bg - mean_fg) ** 2
        if between_var > best_var:
            best_var = between_var
            best_thresh = t
    return best_thresh


def flatten_disk(img, median_window=9):
    """
    Corrects the radial limb-darkening gradient in a full-disk H-alpha image
    and standardizes per-image exposure, so the network isn't relearning
    "dark relative to local background" from scratch at every radius and
    every day's exposure level under the fixed ImageNet constants in
    transform.py.

    img: 2D uint8 grayscale array.
    Returns (flattened, cy, cx, radius): a float32 array of the same shape,
    plus the fitted disk geometry (pixel coordinates + effective radius) so
    callers can later zero out predictions past ~0.98 * radius (see
    mask_outside_radius) where filament annotations essentially never occur.

    This is the pure-Python/numpy reference implementation. cpp/preprocess.cpp
    (exposed as ccl_cpp.flatten_disk) implements the identical algorithm for
    the same reason ccl_cpp.label_and_filter exists next to
    segres.predict.label_and_filter: a from-scratch, benchmarked C++ path for
    the same result.
    """
    img = np.asarray(img)
    img_f = img.astype(np.float64)
    thresh = _otsu_threshold(img.astype(np.uint8))
    disk = img_f > thresh

    if not disk.any():
        # no disk found (e.g. blank/corrupt frame): fall back to plain
        # per-image standardization rather than dividing by a meaningless profile
        flat = (img_f - img_f.mean()) / (img_f.std() + 1e-6)
        return flat.astype(np.float32), 0.0, 0.0, 0.0

    ys, xs = np.nonzero(disk)
    cy, cx = float(ys.mean()), float(xs.mean())
    radius = float(np.sqrt(disk.sum() / np.pi))  # effective radius from disk area

    yy, xx = np.ogrid[: img.shape[0], : img.shape[1]]
    r = np.hypot(yy - cy, xx - cx)

    r_disk = r[disk]
    pix_disk = img_f[disk]
    n_bins = int(r_disk.max()) + 1
    bin_idx = r_disk.astype(np.int64)

    # median intensity per integer radius bin, forward/backward-filled where
    # a bin happened to catch no disk pixels
    order = np.argsort(bin_idx)
    sorted_bins = bin_idx[order]
    sorted_pix = pix_disk[order]
    boundaries = np.searchsorted(sorted_bins, np.arange(n_bins + 1))

    profile = np.full(n_bins, np.nan)
    for i in range(n_bins):
        seg = sorted_pix[boundaries[i] : boundaries[i + 1]]
        if seg.size:
            profile[i] = np.median(seg)

    valid = ~np.isnan(profile)
    idx = np.arange(n_bins)
    profile = np.interp(idx, idx[valid], profile[valid])

    profile = ndimage.median_filter(profile, size=median_window, mode="nearest")

    bin_centers = np.arange(n_bins, dtype=np.float64)
    divisor = np.interp(r.ravel(), bin_centers, profile).reshape(r.shape)
    divisor = np.clip(divisor, 1e-3, None)

    flat = img_f / divisor
    flat[~disk] = 1.0  # neutral value near the post-division disk mean, not an outlier

    disk_vals = flat[disk]
    mean, std = disk_vals.mean(), disk_vals.std()
    flat = (flat - mean) / (std + 1e-6)

    return flat.astype(np.float32), cy, cx, radius


def mask_outside_radius(mask, cy, cx, radius, frac=0.98):
    """
    Zeroes predicted-mask pixels beyond frac * radius from the disk center.
    Model predictions spuriously hug the limb, where filament annotations
    essentially never occur (filaments are on-disk features) -- this cuts
    that failure mode without touching the model itself.
    """
    yy, xx = np.ogrid[: mask.shape[0], : mask.shape[1]]
    r = np.hypot(yy - cy, xx - cx)
    out = mask.copy()
    out[r > frac * radius] = 0
    return out
