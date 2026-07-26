"""Benchmark: ccl_cpp.label_and_filter vs segres.predict.label_and_filter
(scipy.ndimage.label + np.bincount), at the sizes/sparsity this pipeline
actually produces:
  - 512x512, ~27% foreground: a single training tile's ground-truth mask
  - 2048x2048, sparse blobs: a stitched full-image prediction, the shape
    validate() and build_submission() both call label_and_filter on

Run from anywhere with the venv python: `venv/bin/python cpp/benchmark.py`
"""
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(__file__))
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

import ccl_cpp
from segres.predict import label_and_filter as label_and_filter_py


def make_sparse_mask(size, n_blobs, min_area):
    rng = np.random.default_rng(42)
    mask = np.zeros((size, size), dtype=np.uint8)
    for _ in range(n_blobs):
        y, x = rng.integers(0, size - 1, size=2)
        h, w = rng.integers(5, 150, size=2)
        h, w = min(h, size - y), min(w, size - x)
        mask[y : y + h, x : x + w] = rng.random((h, w)) < 0.5
    return mask


def bench(fn, mask, min_area, reps):
    # one warmup call (page faults, allocator warmup), excluded from timing
    fn(mask, min_area=min_area)
    t0 = time.perf_counter()
    for _ in range(reps):
        fn(mask, min_area=min_area)
    return (time.perf_counter() - t0) / reps


def cpp_call(mask, min_area):
    return ccl_cpp.label_and_filter(mask, min_area)


def run_case(name, mask, min_area, reps):
    t_py = bench(label_and_filter_py, mask, min_area, reps)
    t_cpp = bench(cpp_call, mask, min_area, reps)
    print(f"{name:45s} scipy={t_py*1e3:8.3f} ms   ccl_cpp={t_cpp*1e3:8.3f} ms   speedup={t_py/t_cpp:5.1f}x")


def main():
    rng = np.random.default_rng(0)

    tile = (rng.random((512, 512)) < 0.27).astype(np.uint8)
    run_case("512x512 tile, dense random (~27% fg)", tile, min_area=20, reps=50)

    full = make_sparse_mask(2048, n_blobs=40, min_area=200)
    run_case("2048x2048 stitched, sparse blobs", full, min_area=200, reps=20)

    empty = np.zeros((2048, 2048), dtype=np.uint8)
    run_case("2048x2048, all-background (empty image)", empty, min_area=200, reps=20)

    dense = (rng.random((2048, 2048)) < 0.5).astype(np.uint8)
    run_case("2048x2048, dense random (~50% fg, worst case)", dense, min_area=200, reps=5)

    # segres/predict.py's Python path rebuilds each surviving instance with
    # `(labeled == label_id)` -- an O(N) full-image compare per instance, so
    # it's O(N * num_instances) overall. ccl_cpp writes every instance in one
    # O(N) pass. This is where the two implementations should diverge most:
    # a mask with many small, sparse, surviving components.
    # Each surviving instance is materialized as its own full HxW mask (both
    # impls), so instance count directly drives peak RAM: ~12k instances at
    # 0.003 density was ~48GB/call (12k * 2048*2048 bytes). Keep density low
    # enough to stay representative of "many sparse instances" without that
    # blowup.
    many = (rng.random((2048, 2048)) < 0.00004).astype(np.uint8)  # isolated pixels, ~170 survive at min_area=1
    run_case("2048x2048, sparse isolated pixels, min_area=1 (many instances)", many, min_area=1, reps=5)


def count_instances():
    rng = np.random.default_rng(0)
    many = (rng.random((2048, 2048)) < 0.00004).astype(np.uint8)
    _, instances = cpp_call(many, min_area=1)
    print(f"\n(many-instances case surfaced {len(instances)} surviving components)")


if __name__ == "__main__":
    main()
    count_instances()
