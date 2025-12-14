// SPDX-FileCopyrightText: 2025 The unpaper authors
//
// SPDX-License-Identifier: GPL-2.0-only

#pragma once

#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>

#include "imageprocess/cuda_runtime.h"

#ifdef __cplusplus
extern "C" {
#endif

typedef struct {
  int label_count;
  int removed_components;
} UnpaperOpencvCclStats;

typedef struct {
  uint64_t device_ptr;
  int width;
  int height;
  size_t pitch_bytes;
  bool opencv_allocated; // true if allocated via OpenCV/cudart, false via unpaper
} UnpaperOpencvMask;

bool unpaper_opencv_enabled(void);
bool unpaper_opencv_cuda_supported(void);
bool unpaper_opencv_ccl_supported(void);

bool unpaper_opencv_cuda_ccl(uint64_t mask_device, int width, int height,
                             size_t pitch_bytes, uint8_t foreground_value,
                             uint32_t max_component_size,
                             UnpaperCudaStream *stream,
                             UnpaperOpencvCclStats *stats_out);

bool unpaper_opencv_extract_dark_mask(uint64_t src_device, int src_width,
                                      int src_height, size_t src_pitch_bytes,
                                      int src_format, uint8_t min_white_level,
                                      UnpaperCudaStream *stream,
                                      UnpaperOpencvMask *mask_out);

void unpaper_opencv_mask_free(UnpaperOpencvMask *mask);

#ifdef __cplusplus
}
#endif
