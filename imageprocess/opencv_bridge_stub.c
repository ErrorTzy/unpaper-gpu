// SPDX-FileCopyrightText: 2025 The unpaper authors
//
// SPDX-License-Identifier: GPL-2.0-only

#include "imageprocess/opencv_bridge.h"

#include <string.h>

bool unpaper_opencv_enabled(void) { return false; }

bool unpaper_opencv_cuda_supported(void) { return false; }

bool unpaper_opencv_cuda_ccl(uint64_t mask_device, int width, int height,
                             size_t pitch_bytes, uint8_t foreground_value,
                             uint32_t max_component_size,
                             UnpaperCudaStream *stream,
                             UnpaperOpencvCclStats *stats_out) {
  (void)mask_device;
  (void)width;
  (void)height;
  (void)pitch_bytes;
  (void)foreground_value;
  (void)max_component_size;
  (void)stream;
  if (stats_out != NULL) {
    memset(stats_out, 0, sizeof(*stats_out));
  }
  return false;
}

bool unpaper_opencv_extract_dark_mask(uint64_t src_device, int src_width,
                                      int src_height, size_t src_pitch_bytes,
                                      int src_format, uint8_t min_white_level,
                                      UnpaperCudaStream *stream,
                                      UnpaperOpencvMask *mask_out) {
  (void)src_device;
  (void)src_width;
  (void)src_height;
  (void)src_pitch_bytes;
  (void)src_format;
  (void)min_white_level;
  (void)stream;
  if (mask_out != NULL) {
    memset(mask_out, 0, sizeof(*mask_out));
  }
  return false;
}

void unpaper_opencv_mask_free(UnpaperOpencvMask *mask) {
  if (mask != NULL) {
    memset(mask, 0, sizeof(*mask));
  }
}
