#include "preprocess.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <limits>

namespace {

// The intensity threshold that maximizes between-class variance, i.e. the
// disk-vs-background split point for a uint8 histogram.
int otsu_threshold(const uint8_t *img, size_t n) {
    std::array<int64_t, 256> hist{};
    for (size_t i = 0; i < n; ++i) hist[img[i]]++;

    const double total = static_cast<double>(n);
    double sum_all = 0.0;
    for (int t = 0; t < 256; ++t) sum_all += static_cast<double>(t) * static_cast<double>(hist[t]);

    double sum_bg = 0.0, w_bg = 0.0;
    double best_var = -1.0;
    int best_t = 0;
    for (int t = 0; t < 256; ++t) {
        w_bg += static_cast<double>(hist[t]);
        if (w_bg == 0.0) continue;
        const double w_fg = total - w_bg;
        if (w_fg <= 0.0) break;
        sum_bg += static_cast<double>(t) * static_cast<double>(hist[t]);
        const double mean_bg = sum_bg / w_bg;
        const double mean_fg = (sum_all - sum_bg) / w_fg;
        const double diff = mean_bg - mean_fg;
        const double between_var = w_bg * w_fg * diff * diff;
        if (between_var > best_var) {
            best_var = between_var;
            best_t = t;
        }
    }
    return best_t;
}

double median_of(std::vector<float> &v) {
    const size_t mid = v.size() / 2;
    std::nth_element(v.begin(), v.begin() + static_cast<long>(mid), v.end());
    double m = v[mid];
    if (v.size() % 2 == 0) {
        std::nth_element(v.begin(), v.begin() + static_cast<long>(mid - 1), v.begin() + static_cast<long>(mid));
        m = (m + v[mid - 1]) / 2.0;
    }
    return m;
}

// 1D median filter, edge-replicated (matches scipy.ndimage.median_filter's
// mode="nearest" used by the Python reference implementation).
std::vector<double> median_filter_1d(const std::vector<double> &in, int window) {
    const int half = window / 2;
    const int n = static_cast<int>(in.size());
    std::vector<double> out(static_cast<size_t>(n));
    std::vector<double> buf;
    buf.reserve(static_cast<size_t>(window));
    for (int i = 0; i < n; ++i) {
        buf.clear();
        for (int k = -half; k <= half; ++k) {
            const int j = std::clamp(i + k, 0, n - 1);
            buf.push_back(in[static_cast<size_t>(j)]);
        }
        const size_t mid = buf.size() / 2;
        std::nth_element(buf.begin(), buf.begin() + static_cast<long>(mid), buf.end());
        out[static_cast<size_t>(i)] = buf[mid];
    }
    return out;
}

}  // namespace

FlattenDiskResult flatten_disk(const uint8_t *img, int height, int width, int median_window) {
    const size_t n = static_cast<size_t>(height) * static_cast<size_t>(width);
    FlattenDiskResult result;
    result.flattened.assign(n, 0.0f);

    const int thresh = otsu_threshold(img, n);

    std::vector<uint8_t> disk(n);
    double sum_y = 0.0, sum_x = 0.0;
    int64_t disk_count = 0;
    for (int y = 0; y < height; ++y) {
        for (int x = 0; x < width; ++x) {
            const size_t i = static_cast<size_t>(y) * static_cast<size_t>(width) + static_cast<size_t>(x);
            const bool on = img[i] > thresh;
            disk[i] = on ? 1 : 0;
            if (on) {
                sum_y += y;
                sum_x += x;
                ++disk_count;
            }
        }
    }

    if (disk_count == 0) {
        // no disk found: fall back to plain per-image standardization
        double mean = 0.0;
        for (size_t i = 0; i < n; ++i) mean += img[i];
        mean /= static_cast<double>(n);
        double var = 0.0;
        for (size_t i = 0; i < n; ++i) {
            const double d = static_cast<double>(img[i]) - mean;
            var += d * d;
        }
        const double std_ = std::sqrt(var / static_cast<double>(n)) + 1e-6;
        for (size_t i = 0; i < n; ++i) {
            result.flattened[i] = static_cast<float>((static_cast<double>(img[i]) - mean) / std_);
        }
        result.cy = result.cx = result.radius = 0.0;
        return result;
    }

    const double cy = sum_y / static_cast<double>(disk_count);
    const double cx = sum_x / static_cast<double>(disk_count);
    const double radius = std::sqrt(static_cast<double>(disk_count) / M_PI);

    // per-pixel radius, and the largest integer bin any on-disk pixel needs
    std::vector<float> r(n);
    int max_bin = 0;
    for (int y = 0; y < height; ++y) {
        for (int x = 0; x < width; ++x) {
            const size_t i = static_cast<size_t>(y) * static_cast<size_t>(width) + static_cast<size_t>(x);
            const double dy = y - cy, dx = x - cx;
            const float ri = static_cast<float>(std::sqrt(dy * dy + dx * dx));
            r[i] = ri;
            if (disk[i]) {
                const int b = static_cast<int>(ri);
                if (b > max_bin) max_bin = b;
            }
        }
    }
    const int n_bins = max_bin + 1;

    // bucket on-disk intensities by integer radius bin, then take each
    // bucket's median -- this is the radial brightness (limb-darkening) profile
    std::vector<std::vector<float>> buckets(static_cast<size_t>(n_bins));
    for (size_t i = 0; i < n; ++i) {
        if (!disk[i]) continue;
        const int b = static_cast<int>(r[i]);
        buckets[static_cast<size_t>(b)].push_back(static_cast<float>(img[i]));
    }

    std::vector<double> profile(static_cast<size_t>(n_bins), std::numeric_limits<double>::quiet_NaN());
    for (int b = 0; b < n_bins; ++b) {
        auto &bucket = buckets[static_cast<size_t>(b)];
        if (!bucket.empty()) profile[static_cast<size_t>(b)] = median_of(bucket);
    }

    // forward-fill then back-fill empty bins (radii the disk mask never touched)
    double last_valid = profile[0];
    for (int b = 0; b < n_bins; ++b) {
        if (std::isnan(profile[static_cast<size_t>(b)])) {
            profile[static_cast<size_t>(b)] = last_valid;
        } else {
            last_valid = profile[static_cast<size_t>(b)];
        }
    }
    for (int b = n_bins - 1; b >= 0; --b) {
        if (!std::isnan(profile[static_cast<size_t>(b)])) {
            last_valid = profile[static_cast<size_t>(b)];
        } else {
            profile[static_cast<size_t>(b)] = last_valid;
        }
    }

    profile = median_filter_1d(profile, median_window);

    // divide out the radial profile, linearly interpolating between bin
    // centers so the correction doesn't step discontinuously pixel to pixel
    std::vector<float> flat(n);
    for (size_t i = 0; i < n; ++i) {
        const double ri = r[i];
        const int b0 = std::clamp(static_cast<int>(ri), 0, n_bins - 1);
        const int b1 = std::min(b0 + 1, n_bins - 1);
        const double frac = ri - b0;
        double divisor = profile[static_cast<size_t>(b0)] * (1.0 - frac) + profile[static_cast<size_t>(b1)] * frac;
        divisor = std::max(divisor, 1e-3);
        flat[i] = static_cast<float>(static_cast<double>(img[i]) / divisor);
    }
    for (size_t i = 0; i < n; ++i) {
        if (!disk[i]) flat[i] = 1.0f;  // neutral value near the post-division disk mean, not an outlier
    }

    double mean = 0.0;
    for (size_t i = 0; i < n; ++i) {
        if (disk[i]) mean += flat[i];
    }
    mean /= static_cast<double>(disk_count);
    double var = 0.0;
    for (size_t i = 0; i < n; ++i) {
        if (disk[i]) {
            const double d = flat[i] - mean;
            var += d * d;
        }
    }
    const double std_ = std::sqrt(var / static_cast<double>(disk_count)) + 1e-6;

    for (size_t i = 0; i < n; ++i) {
        result.flattened[i] = static_cast<float>((flat[i] - mean) / std_);
    }

    result.cy = cy;
    result.cx = cx;
    result.radius = radius;
    return result;
}

void mask_outside_radius(uint8_t *mask, int height, int width, double cy, double cx, double radius, double frac) {
    const double limit_sq = (frac * radius) * (frac * radius);
    for (int y = 0; y < height; ++y) {
        for (int x = 0; x < width; ++x) {
            const size_t i = static_cast<size_t>(y) * static_cast<size_t>(width) + static_cast<size_t>(x);
            if (!mask[i]) continue;
            const double dy = y - cy, dx = x - cx;
            if (dy * dy + dx * dx > limit_sq) mask[i] = 0;
        }
    }
}
