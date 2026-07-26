#include <algorithm>
#include <vector>
#include <cstdint>

#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>

namespace py = pybind11;


namespace {
class UnionFind {
public:
    UnionFind() { parent_.push_back(0); } // unused padding

    // Create a brand new group containing just one element, return its id.
    int32_t make_set() {
        const int32_t id = static_cast<int32_t>(parent_.size());
        parent_.push_back(id); // a fresh group points to itself
        return id;
    }

    // Follow parent pointers to the root of x's group.
    int32_t find(int32_t x) {
        while (parent_[x] != x) {
            parent_[x] = parent_[parent_[x]];
            x = parent_[x];
        }
        return x;
    }

    // Merge the groups containing a and b.
    void unite(int32_t a, int32_t b) {
        a = find(a);
        b = find(b);
        if (a != b) {
            parent_[std::max(a, b)] = std::min(a, b);
        }
    }

    size_t size() const { return parent_.size(); }

private:
    std::vector<int32_t> parent_;
};

} 

// Labels a 2D binary mask. 
struct LabelResult {
    std::vector<int32_t> pixel_labels; // size height*width, 0 = background
    int32_t num_blobs;
};

LabelResult label_image(const uint8_t* mask, int height, int width) {
    const size_t n = static_cast<size_t>(height) * static_cast<size_t>(width);
    std::vector<int32_t> provisional(n, 0); // first-guess labels, may get merged later
    UnionFind uf;

    auto at = [width](int y, int x) -> size_t {
        return static_cast<size_t>(y) * static_cast<size_t>(width) + static_cast<size_t>(x);
    };

    // Single pass, left-to-right then top-to-bottom. 
    for (int y = 0; y < height; ++y) {
        for (int x = 0; x < width; ++x) {
            const size_t i = at(y, x);
            if (!mask[i]) continue;

            const int32_t up = (y > 0 && mask[at(y - 1, x)]) ? provisional[at(y - 1, x)] : 0;
            const int32_t left = (x > 0 && mask[at(y, x - 1)]) ? provisional[i - 1] : 0;

            if (up == 0 && left == 0) {
                provisional[i] = uf.make_set();
            } else if (left == 0) {
                provisional[i] = up;
            } else if (up == 0) {
                provisional[i] = left;
            } else {
                provisional[i] = std::min(up, left);
                if (up != left) uf.unite(up, left);
            }
        }
    }

    // Second pass: some provisional labels got merged after being assigned
    std::vector<int32_t> root_to_final(uf.size(), 0);
    int32_t num_blobs = 0;
    LabelResult result;
    result.pixel_labels.resize(n);

    for (size_t i = 0; i < n; ++i) {
        if (provisional[i] == 0) continue;
        const int32_t root = uf.find(provisional[i]);
        if (root_to_final[root] == 0) {
            root_to_final[root] = ++num_blobs;
        }
        result.pixel_labels[i] = root_to_final[root];
    }

    result.num_blobs = num_blobs;
    return result;
}

// Labels the mask, then drops any blob smaller than min_area 
struct LabelAndFilterResult {
    std::vector<uint8_t> cleaned;                 // size height*width, 0/1
    std::vector<std::vector<uint8_t>> instances;  // one height*width 0/1 mask per surviving blob
};

LabelAndFilterResult label_and_filter(const uint8_t* mask, int height, int width, int min_area) {
    const size_t n = static_cast<size_t>(height) * static_cast<size_t>(width);
    LabelResult labeled = label_image(mask, height, width);

    LabelAndFilterResult out;
    out.cleaned.assign(n, 0);
    if (labeled.num_blobs == 0) {
        return out;
    }

    // Count how many pixels each blob has.
    std::vector<int64_t> areas(static_cast<size_t>(labeled.num_blobs) + 1, 0);
    for (size_t i = 0; i < n; ++i) {
        areas[static_cast<size_t>(labeled.pixel_labels[i])] += 1;
    }

    // Decide which blobs survive the size filter, and give survivors a
    // fresh 1-based index into the output instance list (0 = dropped).
    std::vector<int32_t> keep_index(areas.size(), 0);
    int32_t kept = 0;
    for (int32_t blob_id = 1; blob_id <= labeled.num_blobs; ++blob_id) {
        if (areas[static_cast<size_t>(blob_id)] >= min_area) {
            keep_index[static_cast<size_t>(blob_id)] = ++kept;
        }
    }

    // One more pass over every pixel
    out.instances.assign(static_cast<size_t>(kept), std::vector<uint8_t>(n, 0));
    for (size_t i = 0; i < n; ++i) {
        const int32_t blob_id = labeled.pixel_labels[i];
        if (blob_id == 0) continue;
        const int32_t inst = keep_index[static_cast<size_t>(blob_id)];
        if (inst == 0) continue;
        out.cleaned[i] = 1;
        out.instances[static_cast<size_t>(inst - 1)][i] = 1;
    }

    return out;
}


namespace {

using ByteArray = py::array_t<uint8_t, py::array::c_style | py::array::forcecast>;

void check_2d(const ByteArray& mask) {
    if (mask.ndim() != 2) {
        throw std::invalid_argument("mask must be a 2D array");
    }
}

py::tuple label_py(ByteArray mask) {
    check_2d(mask);
    const int height = static_cast<int>(mask.shape(0));
    const int width = static_cast<int>(mask.shape(1));

    LabelResult result = label_image(mask.data(), height, width);

    py::array_t<int32_t> labels_out({height, width});
    std::copy(result.pixel_labels.begin(), result.pixel_labels.end(), labels_out.mutable_data());

    return py::make_tuple(labels_out, result.num_blobs);
}

py::tuple label_and_filter_py(ByteArray mask, int min_area) {
    check_2d(mask);
    const int height = static_cast<int>(mask.shape(0));
    const int width = static_cast<int>(mask.shape(1));

    LabelAndFilterResult result = label_and_filter(mask.data(), height, width, min_area);

    py::array_t<uint8_t> cleaned_out({height, width});
    std::copy(result.cleaned.begin(), result.cleaned.end(), cleaned_out.mutable_data());

    py::list instances_out;
    for (const auto& inst : result.instances) {
        py::array_t<uint8_t> inst_out({height, width});
        std::copy(inst.begin(), inst.end(), inst_out.mutable_data());
        instances_out.append(std::move(inst_out));
    }

    return py::make_tuple(cleaned_out, instances_out);
}

} // namespace

PYBIND11_MODULE(ccl_cpp, m) {
    m.doc() = "Connected-component labeling (4-connectivity), a from-scratch "
              "replacement for the scipy.ndimage.label + numpy pattern used "
              "in segres/predict.py::label_and_filter.";

    m.def("label", &label_py, py::arg("mask"),
          "label(mask) -> (labels: int32 HxW array, num_blobs: int)");

    m.def("label_and_filter", &label_and_filter_py, py::arg("mask"), py::arg("min_area") = 1,
          "label_and_filter(mask, min_area) -> (cleaned: uint8 HxW array, instances: list of uint8 HxW arrays)");
}
