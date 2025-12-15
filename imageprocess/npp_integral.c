// SPDX-FileCopyrightText: 2025 The unpaper authors
//
// SPDX-License-Identifier: GPL-2.0-only

#include "imageprocess/npp_integral.h"

#include <cuda_runtime.h>
#include <nppdefs.h>
#include <nppi_statistics_functions.h>
#include <stdbool.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "imageprocess/cuda_mempool.h"
#include "lib/logging.h"

// Alignment for NPP buffers (typically 512 bytes for best performance)
#define NPP_BUFFER_ALIGNMENT 512

// Round up to alignment
static size_t align_up(size_t value, size_t alignment) {
  return (value + alignment - 1) & ~(alignment - 1);
}

size_t unpaper_npp_integral_buffer_size(int width, int height,
                                         size_t *step_out) {
  if (width <= 0 || height <= 0) {
    if (step_out != NULL) {
      *step_out = 0;
    }
    return 0;
  }

  // Integral output is int32 (4 bytes per pixel)
  // NPP requires aligned row pitch
  size_t row_bytes = (size_t)width * sizeof(int32_t);
  size_t step = align_up(row_bytes, NPP_BUFFER_ALIGNMENT);

  if (step_out != NULL) {
    *step_out = step;
  }

  return step * (size_t)height;
}

uint64_t unpaper_npp_integral_alloc(int width, int height, size_t *step_out) {
  size_t step = 0;
  size_t total_bytes = unpaper_npp_integral_buffer_size(width, height, &step);

  if (total_bytes == 0) {
    if (step_out != NULL) {
      *step_out = 0;
    }
    return 0;
  }

  // Use integral pool if available for better batch performance
  uint64_t ptr = cuda_mempool_integral_global_acquire(total_bytes);
  if (ptr == 0) {
    errOutput("NPP integral: allocation failed for %zu bytes", total_bytes);
    if (step_out != NULL) {
      *step_out = 0;
    }
    return 0;
  }

  if (step_out != NULL) {
    *step_out = step;
  }

  return ptr;
}

void unpaper_npp_integral_free(uint64_t device_ptr) {
  if (device_ptr != 0) {
    // Release to integral pool if available
    cuda_mempool_integral_global_release(device_ptr);
  }
}

bool unpaper_npp_integral_8u32s(uint64_t src_device, int src_width,
                                 int src_height, size_t src_step,
                                 UnpaperNppContext *ctx,
                                 UnpaperNppIntegral *result) {
  if (result == NULL) {
    return false;
  }
  memset(result, 0, sizeof(*result));

  if (src_device == 0 || src_width <= 0 || src_height <= 0) {
    errOutput("NPP integral: invalid input parameters");
    return false;
  }

  // Allocate output buffer
  size_t dst_step = 0;
  uint64_t dst_device = unpaper_npp_integral_alloc(src_width, src_height, &dst_step);
  if (dst_device == 0) {
    return false;
  }

  // Get NPP stream context
  NppStreamContext *npp_ctx = NULL;
  if (ctx != NULL) {
    npp_ctx = (NppStreamContext *)unpaper_npp_context_get_raw(ctx);
  }

  // Set up ROI
  NppiSize roi = {src_width, src_height};

  // Call NPP integral function
  // Note: Only the _Ctx version is available in modern NPP
  NppStatus status;

  // If no context provided, create a temporary one with default stream
  NppStreamContext temp_ctx;
  if (npp_ctx == NULL) {
    // Initialize a default context
    memset(&temp_ctx, 0, sizeof(temp_ctx));
    temp_ctx.hStream = NULL;  // Default stream

    // Get device properties for the context
    UnpaperNppDeviceProps props;
    if (unpaper_npp_get_device_props(&props)) {
      temp_ctx.nCudaDeviceId = props.device_id;
      temp_ctx.nMultiProcessorCount = props.multiprocessor_count;
      temp_ctx.nMaxThreadsPerMultiProcessor = props.max_threads_per_multiprocessor;
      temp_ctx.nMaxThreadsPerBlock = props.max_threads_per_block;
      temp_ctx.nSharedMemPerBlock = props.shared_mem_per_block;
      temp_ctx.nCudaDevAttrComputeCapabilityMajor = props.compute_capability_major;
      temp_ctx.nCudaDevAttrComputeCapabilityMinor = props.compute_capability_minor;
    }
    npp_ctx = &temp_ctx;
  }

  status = nppiIntegral_8u32s_C1R_Ctx(
      (const Npp8u *)src_device, (int)src_step, (Npp32s *)dst_device,
      (int)dst_step, roi, 0, *npp_ctx);

  if (status != NPP_SUCCESS) {
    errOutput("NPP integral: nppiIntegral_8u32s_C1R failed: %s",
              unpaper_npp_status_string(status));
    unpaper_npp_integral_free(dst_device);
    return false;
  }

  // Fill result
  result->device_ptr = dst_device;
  result->width = src_width;
  result->height = src_height;
  result->step_bytes = dst_step;
  result->total_bytes = dst_step * (size_t)src_height;

  return true;
}

int64_t unpaper_npp_integral_rect_sum(const UnpaperNppIntegral *integral,
                                       int x0, int y0, int x1, int y1,
                                       UnpaperCudaStream *stream) {
  if (integral == NULL || integral->device_ptr == 0) {
    return 0;
  }

  // Clamp coordinates to valid range
  if (x0 < 0) x0 = 0;
  if (y0 < 0) y0 = 0;
  if (x1 >= integral->width) x1 = integral->width - 1;
  if (y1 >= integral->height) y1 = integral->height - 1;

  if (x0 > x1 || y0 > y1) {
    return 0;
  }

  // NPP integral format: I[y,x] = sum of pixels from (0,0) to (x-1,y-1)
  // First row and first column are always zero.
  //
  // To compute sum of rectangle (x0,y0) to (x1,y1) inclusive:
  //   sum = I[y1+1,x1+1] - I[y0,x1+1] - I[y1+1,x0] + I[y0,x0]
  //
  // Note: NPP output is width×height, so indices (y1+1, x1+1) might be
  // out of bounds if the rectangle touches the image boundary.
  // For rectangles ending at (width-2, height-2) or smaller, all accesses
  // are valid.

  const int32_t *base = (const int32_t *)integral->device_ptr;
  size_t step_i32 = integral->step_bytes / sizeof(int32_t);

  // We need 4 corner values from the integral image
  int32_t corners[4] = {0, 0, 0, 0};  // br, tr, bl, tl
  size_t offsets[4];
  bool valid[4] = {false, false, false, false};

  // Check which corners are within bounds
  // Bottom-right: I[y1+1, x1+1]
  if (y1 + 1 < integral->height && x1 + 1 < integral->width) {
    offsets[0] = (size_t)(y1 + 1) * step_i32 + (size_t)(x1 + 1);
    valid[0] = true;
  }

  // Top-right: I[y0, x1+1]
  if (x1 + 1 < integral->width) {
    offsets[1] = (size_t)y0 * step_i32 + (size_t)(x1 + 1);
    valid[1] = true;
  }

  // Bottom-left: I[y1+1, x0]
  if (y1 + 1 < integral->height) {
    offsets[2] = (size_t)(y1 + 1) * step_i32 + (size_t)x0;
    valid[2] = true;
  }

  // Top-left: I[y0, x0] - always valid since y0,x0 >= 0
  offsets[3] = (size_t)y0 * step_i32 + (size_t)x0;
  valid[3] = true;

  // Synchronize stream if provided
  if (stream != NULL) {
    unpaper_cuda_stream_synchronize_on(stream);
  }

  // Download corner values
  for (int i = 0; i < 4; i++) {
    if (valid[i]) {
      cudaError_t err = cudaMemcpy(&corners[i], base + offsets[i],
                                    sizeof(int32_t), cudaMemcpyDeviceToHost);
      if (err != cudaSuccess) {
        errOutput("NPP integral rect_sum: cudaMemcpy failed: %s",
                  cudaGetErrorString(err));
        return 0;
      }
    }
  }

  // Compute sum: br - tr - bl + tl
  int64_t br = corners[0];
  int64_t tr = corners[1];
  int64_t bl = corners[2];
  int64_t tl = corners[3];

  return br - tr - bl + tl;
}
