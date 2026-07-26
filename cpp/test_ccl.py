"""Correctness check: ccl_cpp.label_and_filter vs segres.predict.label_and_filter
(scipy.ndimage.label + np.bincount) on random and adversarial masks.

Run from anywhere with the venv python: `venv/bin/python cpp/test_ccl.py`
"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(__file__))  # cpp/ccl_cpp*.so
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))  # repo root, for segres

import ccl_cpp
from segres.predict import label_and_filter as label_and_filter_py


def component_set(cleaned, instances):
    """Represent a labeling as a set of frozensets of flat pixel indices,
    so two labelings that assign different numeric ids to the same
    components still compare equal."""
    return frozenset(frozenset(np.flatnonzero(inst)) for inst in instances)


def check(mask, min_area, name):
    mask = mask.astype(np.uint8)

    py_cleaned, py_instances = label_and_filter_py(mask, min_area=min_area)
    cpp_cleaned, cpp_instances = ccl_cpp.label_and_filter(mask, min_area)

    assert np.array_equal(py_cleaned, cpp_cleaned), f"{name}: cleaned mask differs"
    assert component_set(py_cleaned, py_instances) == component_set(cpp_cleaned, cpp_instances), (
        f"{name}: component sets differ (py={len(py_instances)} insts, cpp={len(cpp_instances)} insts)"
    )
    print(f"OK  {name}: {len(cpp_instances)} instance(s), {cpp_cleaned.sum()} px")


def main():
    rng = np.random.default_rng(0)

    check(np.zeros((64, 64), dtype=np.uint8), min_area=1, name="all-background")
    check(np.ones((32, 40), dtype=np.uint8), min_area=1, name="all-foreground")

    single = np.zeros((16, 16), dtype=np.uint8)
    single[7, 7] = 1
    check(single, min_area=1, name="single-pixel")
    check(single, min_area=2, name="single-pixel dropped by min_area")

    # Checkerboard: every foreground pixel touches only background under
    # 4-connectivity, so each is its own component -- this is the case that
    # stresses the union-find (worst-case label count == pixel count).
    yy, xx = np.meshgrid(np.arange(20), np.arange(20), indexing="ij")
    checker = ((yy + xx) % 2 == 0).astype(np.uint8)
    check(checker, min_area=1, name="checkerboard (diagonal-only touches must NOT merge)")

    # A plus-shape and a diagonally-adjacent-only square: must stay separate
    # components under 4-connectivity (regression test for connectivity type).
    diag = np.zeros((10, 10), dtype=np.uint8)
    diag[2, 2] = 1
    diag[3, 3] = 1  # touches [2,2] only diagonally
    check(diag, min_area=1, name="diagonal-adjacent pixels stay separate (4-conn)")

    for trial in range(20):
        h, w = rng.integers(1, 130, size=2)
        p = rng.choice([0.01, 0.05, 0.2, 0.5, 0.8])
        mask = (rng.random((h, w)) < p).astype(np.uint8)
        min_area = int(rng.choice([1, 2, 5, 20, 200]))
        check(mask, min_area=min_area, name=f"random#{trial} ({h}x{w}, p={p}, min_area={min_area})")

    # Realistic scale + sparsity, matching a stitched 2048x2048 prediction
    # mask with a small filament footprint.
    big = np.zeros((2048, 2048), dtype=np.uint8)
    for _ in range(30):
        y, x = rng.integers(0, 2000, size=2)
        h, w = rng.integers(5, 200, size=2)
        h, w = min(h, 2048 - y), min(w, 2048 - x)
        big[y : y + h, x : x + w] = rng.random((h, w)) < 0.6
    check(big, min_area=200, name="realistic 2048x2048 (config.json min_area)")

    print("\nAll checks passed.")


if __name__ == "__main__":
    main()
