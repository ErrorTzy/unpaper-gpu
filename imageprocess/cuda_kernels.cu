// SPDX-FileCopyrightText: 2025 The unpaper authors
//
// SPDX-License-Identifier: GPL-2.0-only

// Master CUDA kernel file - includes all kernel modules
// This file is compiled to PTX and embedded in the binary
//
// Split structure:
// - cuda_kernels_common.cuh: Device helper functions (pixel I/O, interpolation)
// - cuda_kernels_blit.cu:    Rectangle operations (wipe, copy, mirror,
// rotate90, stretch)
// - cuda_kernels_deskew.cu:  Rotation detection and transform kernels
// - cuda_kernels_masks.cu:   Mask application and statistics kernels
// - cuda_kernels_filters.cu: Filter kernels (noisefilter, blackfilter,
// blurfilter, grayfilter, format conversion)

#include "imageprocess/cuda_kernels_blit.cu"
#include "imageprocess/cuda_kernels_deskew.cu"
#include "imageprocess/cuda_kernels_filters.cu"
#include "imageprocess/cuda_kernels_masks.cu"
