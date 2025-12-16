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

  // Integral output is (width+1) x (height+1) to support boundary tile access
  // NPP requires aligned row pitch
  size_t row_bytes = (size_t)(width + 1) * sizeof(int32_t);
  size_t step = align_up(row_bytes, NPP_BUFFER_ALIGNMENT);

  if (step_out != NULL) {
    *step_out = step;
  }

  return step * (size_t)(height + 1);
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

  // Padded dimensions for extended integral (allows boundary tile access)
  int padded_width = src_width + 1;
  int padded_height = src_height + 1;

  // Allocate padded input buffer with zeros in the extra row/column
  // Use scratch pool to avoid cudaMalloc/cudaFree which serialize all streams
  size_t padded_src_step = align_up((size_t)padded_width, NPP_BUFFER_ALIGNMENT);
  size_t padded_src_bytes = padded_src_step * (size_t)padded_height;
  uint64_t padded_src_device =
      cuda_mempool_scratch_global_acquire(padded_src_bytes);
  if (padded_src_device == 0) {
    errOutput("NPP integral: failed to allocate padded input buffer");
    return false;
  }
  cudaError_t err;

  // Get cuda stream from context (if provided) to avoid serializing all streams
  // Using synchronous operations on default stream would block ALL other
  // streams!
  cudaStream_t cuda_stream = NULL;
  if (ctx != NULL) {
    NppStreamContext *npp_ctx =
        (NppStreamContext *)unpaper_npp_context_get_raw(ctx);
    if (npp_ctx != NULL) {
      cuda_stream = npp_ctx->hStream;
    }
  }

  // Initialize padded buffer to zero (for the padding)
  // Use async memset on our stream to avoid blocking other streams
  err = cudaMemsetAsync((void *)padded_src_device, 0, padded_src_bytes,
                        cuda_stream);
  if (err != cudaSuccess) {
    errOutput("NPP integral: failed to zero padded input buffer");
    cuda_mempool_scratch_global_release(padded_src_device);
    return false;
  }

  // Copy original image to padded buffer (row by row)
  // Use async copy on our stream to avoid blocking other streams
  err = cudaMemcpy2DAsync(
      (void *)padded_src_device, padded_src_step, // dest, dest pitch
      (const void *)src_device, src_step,         // src, src pitch
      (size_t)src_width, (size_t)src_height,      // width, height in bytes
      cudaMemcpyDeviceToDevice, cuda_stream);
  if (err != cudaSuccess) {
    errOutput("NPP integral: failed to copy to padded buffer: %s",
              cudaGetErrorString(err));
    cuda_mempool_scratch_global_release(padded_src_device);
    return false;
  }

  // Allocate output buffer (padded dimensions)
  size_t dst_step = 0;
  uint64_t dst_device =
      unpaper_npp_integral_alloc(src_width, src_height, &dst_step);
  if (dst_device == 0) {
    cuda_mempool_scratch_global_release(padded_src_device);
    return false;
  }

  // Get NPP stream context
  NppStreamContext *npp_ctx = NULL;
  if (ctx != NULL) {
    npp_ctx = (NppStreamContext *)unpaper_npp_context_get_raw(ctx);
  }

  // Set up ROI with padded dimensions
  NppiSize roi = {padded_width, padded_height};

  // If no context provided, create a temporary one with default stream
  NppStreamContext temp_ctx;
  if (npp_ctx == NULL) {
    // Initialize a default context
    memset(&temp_ctx, 0, sizeof(temp_ctx));
    temp_ctx.hStream = NULL; // Default stream

    // Get device properties for the context
    UnpaperNppDeviceProps props;
    if (unpaper_npp_get_device_props(&props)) {
      temp_ctx.nCudaDeviceId = props.device_id;
      temp_ctx.nMultiProcessorCount = props.multiprocessor_count;
      temp_ctx.nMaxThreadsPerMultiProcessor =
          props.max_threads_per_multiprocessor;
      temp_ctx.nMaxThreadsPerBlock = props.max_threads_per_block;
      temp_ctx.nSharedMemPerBlock = props.shared_mem_per_block;
      temp_ctx.nCudaDevAttrComputeCapabilityMajor =
          props.compute_capability_major;
      temp_ctx.nCudaDevAttrComputeCapabilityMinor =
          props.compute_capability_minor;
    }
    npp_ctx = &temp_ctx;
  }

  // Call NPP integral on the padded input
  NppStatus status = nppiIntegral_8u32s_C1R_Ctx(
      (const Npp8u *)padded_src_device, (int)padded_src_step,
      (Npp32s *)dst_device, (int)dst_step, roi, 0, *npp_ctx);

  // Free padded input buffer (return to pool)
  cuda_mempool_scratch_global_release(padded_src_device);

  if (status != NPP_SUCCESS) {
    errOutput("NPP integral: nppiIntegral_8u32s_C1R failed: %s",
              unpaper_npp_status_string(status));
    unpaper_npp_integral_free(dst_device);
    return false;
  }

  // Fill result with padded dimensions
  // Note: width/height are the PADDED dimensions (original + 1)
  // This allows the standard integral formula to work for boundary tiles
  result->device_ptr = dst_device;
  result->width = padded_width;
  result->height = padded_height;
  result->step_bytes = dst_step;
  result->total_bytes = dst_step * (size_t)padded_height;

  return true;
}

int64_t unpaper_npp_integral_rect_sum(const UnpaperNppIntegral *integral,
                                      int x0, int y0, int x1, int y1,
                                      UnpaperCudaStream *stream) {
  if (integral == NULL || integral->device_ptr == 0) {
    return 0;
  }

  // The integral is padded to (orig_w+1) x (orig_h+1), so:
  // - Original image max coords: orig_w-1 = integral->width-2, orig_h-1 =
  // integral->height-2
  // - Clamp to original image bounds
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

  // NPP integral format: I[y,x] = sum of pixels from (0,0) to (x-1,y-1)
  // To compute sum of rectangle (x0,y0) to (x1,y1) inclusive:
  //   sum = I[y1+1,x1+1] - I[y0,x1+1] - I[y1+1,x0] + I[y0,x0]
  //
  // With padded integral, all accesses are valid:
  // - y1+1 <= orig_h = integral->height - 1 < integral->height
  // - x1+1 <= orig_w = integral->width - 1 < integral->width

  const int32_t *base = (const int32_t *)integral->device_ptr;
  size_t step_i32 = integral->step_bytes / sizeof(int32_t);

  // Compute offsets for 4 corners (all valid with padded integral)
  size_t offsets[4];
  offsets[0] = (size_t)(y1 + 1) * step_i32 + (size_t)(x1 + 1); // Bottom-right
  offsets[1] = (size_t)y0 * step_i32 + (size_t)(x1 + 1);       // Top-right
  offsets[2] = (size_t)(y1 + 1) * step_i32 + (size_t)x0;       // Bottom-left
  offsets[3] = (size_t)y0 * step_i32 + (size_t)x0;             // Top-left

  // Synchronize stream if provided
  if (stream != NULL) {
    unpaper_cuda_stream_synchronize_on(stream);
  }

  // Download corner values (all valid with padded integral)
  int32_t corners[4] = {0, 0, 0, 0};
  for (int i = 0; i < 4; i++) {
    cudaError_t err = cudaMemcpy(&corners[i], base + offsets[i],
                                 sizeof(int32_t), cudaMemcpyDeviceToHost);
    if (err != cudaSuccess) {
      errOutput("NPP integral rect_sum: cudaMemcpy failed: %s",
                cudaGetErrorString(err));
      return 0;
    }
  }

  // Compute sum: br - tr - bl + tl
  int64_t br = corners[0];
  int64_t tr = corners[1];
  int64_t bl = corners[2];
  int64_t tl = corners[3];

  return br - tr - bl + tl;
}

bool unpaper_npp_integral_download(const UnpaperNppIntegral *gpu_integral,
                                   UnpaperCpuIntegral *cpu_integral,
                                   UnpaperCudaStream *stream) {
  if (gpu_integral == NULL || cpu_integral == NULL) {
    return false;
  }
  if (gpu_integral->device_ptr == 0) {
    return false;
  }

  // Calculate CPU buffer size (may be smaller due to different alignment)
  size_t step_i32 = gpu_integral->step_bytes / sizeof(int32_t);
  size_t total_elements = step_i32 * (size_t)gpu_integral->height;
  size_t total_bytes = total_elements * sizeof(int32_t);

  // Allocate CPU memory
  int32_t *data = (int32_t *)malloc(total_bytes);
  if (data == NULL) {
    return false;
  }

  // Use async copy on the specific stream to avoid blocking other streams
  // This is critical for stream parallelism!
  cudaError_t err;
  if (stream != NULL) {
    cudaStream_t cuda_stream =
        (cudaStream_t)unpaper_cuda_stream_get_raw_handle(stream);
    err = cudaMemcpyAsync(data, (const void *)gpu_integral->device_ptr,
                          total_bytes, cudaMemcpyDeviceToHost, cuda_stream);
    if (err != cudaSuccess) {
      free(data);
      return false;
    }
    // Sync only this stream to wait for the copy to complete
    unpaper_cuda_stream_synchronize_on(stream);
  } else {
    // No stream provided - use blocking copy (will sync all streams)
    err = cudaMemcpy(data, (const void *)gpu_integral->device_ptr, total_bytes,
                     cudaMemcpyDeviceToHost);
    if (err != cudaSuccess) {
      free(data);
      return false;
    }
  }

  cpu_integral->data = data;
  cpu_integral->width = gpu_integral->width;
  cpu_integral->height = gpu_integral->height;
  cpu_integral->step_i32 = step_i32;

  return true;
}

void unpaper_cpu_integral_free(UnpaperCpuIntegral *integral) {
  if (integral != NULL && integral->data != NULL) {
    free(integral->data);
    integral->data = NULL;
  }
}
