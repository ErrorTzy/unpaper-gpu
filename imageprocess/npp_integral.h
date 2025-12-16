// SPDX-FileCopyrightText: 2025 The unpaper authors
//
// SPDX-License-Identifier: GPL-2.0-only

#pragma once

#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>

#include "imageprocess/cuda_runtime.h"
#include "imageprocess/npp_wrapper.h"

#ifdef __cplusplus
extern "C" {
#endif

// Result of integral image computation
typedef struct {
  uint64_t device_ptr; // GPU memory pointer to integral image (int32)
  int width;           // Width of integral image (input_width + 1 for padding)
  int height;         // Height of integral image (input_height + 1 for padding)
  size_t step_bytes;  // Bytes per row (pitch)
  size_t total_bytes; // Total buffer size
} UnpaperNppIntegral;

// Compute integral image on GPU using NPP
// Input: 8-bit single-channel image on GPU
// Output: 32-bit signed integral image on GPU
//
// Note: The output is padded to (width+1) x (height+1) to support the standard
// integral sum formula for tiles at the image boundary. The input is
// automatically zero-padded before computing the integral.
//
// Parameters:
//   src_device   - GPU pointer to source image (8-bit)
//   src_width    - Source image width
//   src_height   - Source image height
//   src_step     - Source image row pitch in bytes
//   ctx          - NPP context (for stream-aware async execution)
//   result       - Output integral image info (caller manages memory)
//
// Returns true on success, false on error
bool unpaper_npp_integral_8u32s(uint64_t src_device, int src_width,
                                int src_height, size_t src_step,
                                UnpaperNppContext *ctx,
                                UnpaperNppIntegral *result);

// Calculate required buffer size for integral image output
// Returns the number of bytes needed for the integral buffer
size_t unpaper_npp_integral_buffer_size(int width, int height,
                                        size_t *step_out);

// Allocate integral image buffer on GPU
// Returns GPU device pointer, or 0 on failure
uint64_t unpaper_npp_integral_alloc(int width, int height, size_t *step_out);

// Free integral image buffer
void unpaper_npp_integral_free(uint64_t device_ptr);

// Convenience: compute sum of rectangle from integral image
// This performs a D2H copy of just the 4 corner values needed
// Coordinates are in the original image space (not integral space)
//
// Note: This function synchronizes the stream to get the result.
// For batch processing, use the GPU scan kernel instead.
int64_t unpaper_npp_integral_rect_sum(const UnpaperNppIntegral *integral,
                                      int x0, int y0, int x1, int y1,
                                      UnpaperCudaStream *stream);

// CPU-side integral image for fast iterative algorithms
// This avoids repeated GPU syncs by downloading the entire integral once
typedef struct {
  int32_t *data;   // CPU memory pointer to integral image
  int width;       // Width of integral image (input_width + 1)
  int height;      // Height of integral image (input_height + 1)
  size_t step_i32; // Elements (int32) per row
} UnpaperCpuIntegral;

// Download integral image from GPU to CPU memory
// This performs a single D2H transfer of the entire integral
// Returns true on success, false on failure
// The caller must free the CPU integral with unpaper_cpu_integral_free()
bool unpaper_npp_integral_download(const UnpaperNppIntegral *gpu_integral,
                                   UnpaperCpuIntegral *cpu_integral,
                                   UnpaperCudaStream *stream);

// Free CPU integral image memory
void unpaper_cpu_integral_free(UnpaperCpuIntegral *integral);

// Compute rectangle sum from CPU integral (no GPU sync required)
// Coordinates are in the original image space (not integral space)
// This is O(1) and completely CPU-side
static inline int64_t
unpaper_cpu_integral_rect_sum(const UnpaperCpuIntegral *integral, int x0,
                              int y0, int x1, int y1) {
  if (integral == NULL || integral->data == NULL) {
    return 0;
  }

  // Clamp to image bounds
  int orig_w = integral->width - 1;
  int orig_h = integral->height - 1;
  if (x0 < 0)
    x0 = 0;
  if (y0 < 0)
    y0 = 0;
  if (x1 >= orig_w)
    x1 = orig_w - 1;
  if (y1 >= orig_h)
    y1 = orig_h - 1;
  if (x0 > x1 || y0 > y1) {
    return 0;
  }

  // Integral formula: sum = I[y1+1,x1+1] - I[y0,x1+1] - I[y1+1,x0] + I[y0,x0]
  size_t step = integral->step_i32;
  int64_t br = integral->data[(size_t)(y1 + 1) * step + (size_t)(x1 + 1)];
  int64_t tr = integral->data[(size_t)y0 * step + (size_t)(x1 + 1)];
  int64_t bl = integral->data[(size_t)(y1 + 1) * step + (size_t)x0];
  int64_t tl = integral->data[(size_t)y0 * step + (size_t)x0];

  return br - tr - bl + tl;
}

#ifdef __cplusplus
}
#endif
