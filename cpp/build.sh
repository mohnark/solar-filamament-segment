#!/usr/bin/env bash
# Builds the ccl_cpp pybind11 extension against the project's venv, and
# copies the resulting .so next to this script so `import ccl_cpp` works
# from cpp/ without installing anything.
set -euo pipefail
cd "$(dirname "$0")"

PYTHON="${PYTHON:-../venv/bin/python}"

cmake -S . -B build \
  -DCMAKE_BUILD_TYPE=Release \
  -DPython3_EXECUTABLE="$("$PYTHON" -c 'import sys; print(sys.executable)')" \
  -Dpybind11_DIR="$("$PYTHON" -c 'import pybind11; print(pybind11.get_cmake_dir())')"

cmake --build build -j"$(nproc)"

cp build/ccl_cpp*.so .
echo "Built: $(ls ccl_cpp*.so)"
