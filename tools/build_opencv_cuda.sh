#!/bin/bash
# SPDX-FileCopyrightText: 2025 The unpaper authors
# SPDX-License-Identifier: GPL-2.0-only
#
# Build OpenCV with CUDA support from source
# This script downloads and builds OpenCV with CUDA modules enabled
#
# Prerequisites:
#   - CUDA toolkit installed (nvcc available)
#   - CMake 3.16+
#   - Build tools (gcc, g++, make)
#   - Various dev libraries (see below)
#
# Usage:
#   ./tools/build_opencv_cuda.sh [install_prefix]
#   Default install_prefix: /usr/local

set -e

OPENCV_VERSION="4.10.0"
INSTALL_PREFIX="${1:-/usr/local}"
BUILD_DIR="/tmp/opencv_cuda_build"
JOBS=$(nproc)

echo "=== Building OpenCV ${OPENCV_VERSION} with CUDA support ==="
echo "Install prefix: ${INSTALL_PREFIX}"
echo "Build directory: ${BUILD_DIR}"
echo "Parallel jobs: ${JOBS}"

# Check for CUDA
if ! command -v nvcc &> /dev/null; then
    echo "ERROR: nvcc not found. Please install CUDA toolkit first."
    exit 1
fi

CUDA_PATH=$(dirname $(dirname $(which nvcc)))
echo "CUDA path: ${CUDA_PATH}"

# Install dependencies
echo ""
echo "=== Installing build dependencies ==="
sudo apt-get update
sudo apt-get install -y \
    build-essential cmake git pkg-config \
    libjpeg-dev libpng-dev libtiff-dev \
    libavcodec-dev libavformat-dev libswscale-dev \
    libv4l-dev libxvidcore-dev libx264-dev \
    libgtk-3-dev libatlas-base-dev gfortran \
    python3-dev python3-numpy

# Create build directory
mkdir -p "${BUILD_DIR}"
cd "${BUILD_DIR}"

# Download OpenCV and opencv_contrib
if [ ! -d "opencv" ]; then
    echo ""
    echo "=== Downloading OpenCV ${OPENCV_VERSION} ==="
    git clone --depth 1 --branch ${OPENCV_VERSION} https://github.com/opencv/opencv.git
fi

if [ ! -d "opencv_contrib" ]; then
    echo ""
    echo "=== Downloading OpenCV contrib modules ==="
    git clone --depth 1 --branch ${OPENCV_VERSION} https://github.com/opencv/opencv_contrib.git
fi

# Create build directory
mkdir -p opencv/build
cd opencv/build

# Configure with CMake
echo ""
echo "=== Configuring OpenCV with CUDA ==="
cmake \
    -D CMAKE_BUILD_TYPE=Release \
    -D CMAKE_INSTALL_PREFIX="${INSTALL_PREFIX}" \
    -D OPENCV_EXTRA_MODULES_PATH="${BUILD_DIR}/opencv_contrib/modules" \
    -D WITH_CUDA=ON \
    -D CUDA_TOOLKIT_ROOT_DIR="${CUDA_PATH}" \
    -D CUDA_ARCH_BIN="7.5 8.0 8.6 8.9 9.0" \
    -D CUDA_ARCH_PTX="9.0" \
    -D WITH_CUDNN=OFF \
    -D OPENCV_DNN_CUDA=OFF \
    -D ENABLE_FAST_MATH=ON \
    -D CUDA_FAST_MATH=ON \
    -D WITH_CUBLAS=ON \
    -D WITH_OPENGL=OFF \
    -D WITH_QT=OFF \
    -D WITH_GTK=ON \
    -D BUILD_opencv_python3=OFF \
    -D BUILD_opencv_python2=OFF \
    -D BUILD_EXAMPLES=OFF \
    -D BUILD_TESTS=OFF \
    -D BUILD_PERF_TESTS=OFF \
    -D BUILD_DOCS=OFF \
    -D OPENCV_GENERATE_PKGCONFIG=ON \
    ..

# Build
echo ""
echo "=== Building OpenCV (this may take a while) ==="
make -j${JOBS}

# Install
echo ""
echo "=== Installing OpenCV ==="
sudo make install
sudo ldconfig

echo ""
echo "=== OpenCV with CUDA installed successfully ==="
echo ""
echo "To verify, run:"
echo "  pkg-config --modversion opencv4"
echo "  python3 -c \"import cv2; print(cv2.cuda.getCudaEnabledDeviceCount())\""
echo ""
echo "You may need to rebuild unpaper with:"
echo "  rm -rf builddir-cuda-opencv"
echo "  meson setup builddir-cuda-opencv -Dcuda=enabled -Dopencv=enabled"
echo "  meson compile -C builddir-cuda-opencv"
