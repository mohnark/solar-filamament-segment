#pragma once

#include <cstdint>
#include <vector>

// Radial limb-darkening correction + per-image standardization for a
// full-disk H-alpha frame. See segres/preprocess.py::flatten_disk for the
// algorithm description; this is the from-scratch C++ implementation of the
// same steps (Otsu disk segmentation, radial median profile, division,
// on-disk standardization).
struct FlattenDiskResult {
    std::vector<float> flattened;  // size height*width
    double cy;
    double cx;
    double radius;
};

FlattenDiskResult flatten_disk(const uint8_t *img, int height, int width, int median_window);

// Zeroes mask pixels farther than frac * radius from (cy, cx), in place.
void mask_outside_radius(uint8_t *mask, int height, int width, double cy, double cx, double radius, double frac);
