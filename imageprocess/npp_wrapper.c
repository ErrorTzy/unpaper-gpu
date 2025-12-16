// SPDX-FileCopyrightText: 2025 The unpaper authors
//
// SPDX-License-Identifier: GPL-2.0-only

#include "imageprocess/npp_wrapper.h"

#include <cuda_runtime.h>
#include <nppdefs.h>
#include <stdbool.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "lib/logging.h"

// NPP context wrapping NppStreamContext
struct UnpaperNppContext {
  NppStreamContext npp_ctx;
};

// Cached device properties
static bool device_props_cached = false;
static UnpaperNppDeviceProps cached_props;
static bool npp_initialized = false;

bool unpaper_npp_init(void) {
  if (npp_initialized) {
    return true;
  }

  // Cache device properties on first init
  if (!unpaper_npp_get_device_props(&cached_props)) {
    return false;
  }

  npp_initialized = true;
  return true;
}

bool unpaper_npp_available(void) { return npp_initialized; }

bool unpaper_npp_get_device_props(UnpaperNppDeviceProps *props) {
  if (props == NULL) {
    return false;
  }

  if (device_props_cached) {
    *props = cached_props;
    return true;
  }

  // Get current device
  int device_id = 0;
  cudaError_t err = cudaGetDevice(&device_id);
  if (err != cudaSuccess) {
    errOutput("NPP: cudaGetDevice failed: %s", cudaGetErrorString(err));
    return false;
  }

  // Get device properties
  struct cudaDeviceProp dev_prop;
  err = cudaGetDeviceProperties(&dev_prop, device_id);
  if (err != cudaSuccess) {
    errOutput("NPP: cudaGetDeviceProperties failed: %s",
              cudaGetErrorString(err));
    return false;
  }

  props->device_id = device_id;
  props->multiprocessor_count = dev_prop.multiProcessorCount;
  props->max_threads_per_multiprocessor = dev_prop.maxThreadsPerMultiProcessor;
  props->max_threads_per_block = dev_prop.maxThreadsPerBlock;
  props->shared_mem_per_block = dev_prop.sharedMemPerBlock;
  props->compute_capability_major = dev_prop.major;
  props->compute_capability_minor = dev_prop.minor;

  // Cache for future calls
  cached_props = *props;
  device_props_cached = true;

  return true;
}

UnpaperNppContext *unpaper_npp_context_create(UnpaperCudaStream *stream) {
  if (!npp_initialized) {
    if (!unpaper_npp_init()) {
      return NULL;
    }
  }

  UnpaperNppContext *ctx =
      (UnpaperNppContext *)calloc(1, sizeof(UnpaperNppContext));
  if (ctx == NULL) {
    return NULL;
  }

  // Get the CUDA stream handle
  cudaStream_t cuda_stream = NULL;
  if (stream != NULL) {
    cuda_stream = (cudaStream_t)unpaper_cuda_stream_get_raw_handle(stream);
  }

  // Fill in NppStreamContext
  ctx->npp_ctx.hStream = cuda_stream;
  ctx->npp_ctx.nCudaDeviceId = cached_props.device_id;
  ctx->npp_ctx.nMultiProcessorCount = cached_props.multiprocessor_count;
  ctx->npp_ctx.nMaxThreadsPerMultiProcessor =
      cached_props.max_threads_per_multiprocessor;
  ctx->npp_ctx.nMaxThreadsPerBlock = cached_props.max_threads_per_block;
  ctx->npp_ctx.nSharedMemPerBlock = cached_props.shared_mem_per_block;
  ctx->npp_ctx.nCudaDevAttrComputeCapabilityMajor =
      cached_props.compute_capability_major;
  ctx->npp_ctx.nCudaDevAttrComputeCapabilityMinor =
      cached_props.compute_capability_minor;

  // Get stream flags if we have a stream
  if (cuda_stream != NULL) {
    unsigned int flags = 0;
    cudaError_t err = cudaStreamGetFlags(cuda_stream, &flags);
    if (err == cudaSuccess) {
      ctx->npp_ctx.nStreamFlags = flags;
    }
  }

  return ctx;
}

void unpaper_npp_context_destroy(UnpaperNppContext *ctx) {
  if (ctx != NULL) {
    free(ctx);
  }
}

void *unpaper_npp_context_get_raw(UnpaperNppContext *ctx) {
  if (ctx == NULL) {
    return NULL;
  }
  return &ctx->npp_ctx;
}

const char *unpaper_npp_status_string(int status) {
  // NPP status codes from nppdefs.h
  switch (status) {
  case NPP_SUCCESS:
    return "NPP_SUCCESS";
  case NPP_NOT_SUPPORTED_MODE_ERROR:
    return "NPP_NOT_SUPPORTED_MODE_ERROR";
  case NPP_INVALID_HOST_POINTER_ERROR:
    return "NPP_INVALID_HOST_POINTER_ERROR";
  case NPP_INVALID_DEVICE_POINTER_ERROR:
    return "NPP_INVALID_DEVICE_POINTER_ERROR";
  case NPP_LUT_PALETTE_BITSIZE_ERROR:
    return "NPP_LUT_PALETTE_BITSIZE_ERROR";
  case NPP_ZC_MODE_NOT_SUPPORTED_ERROR:
    return "NPP_ZC_MODE_NOT_SUPPORTED_ERROR";
  case NPP_NOT_SUFFICIENT_COMPUTE_CAPABILITY:
    return "NPP_NOT_SUFFICIENT_COMPUTE_CAPABILITY";
  case NPP_TEXTURE_BIND_ERROR:
    return "NPP_TEXTURE_BIND_ERROR";
  case NPP_WRONG_INTERSECTION_ROI_ERROR:
    return "NPP_WRONG_INTERSECTION_ROI_ERROR";
  case NPP_HAAR_CLASSIFIER_PIXEL_MATCH_ERROR:
    return "NPP_HAAR_CLASSIFIER_PIXEL_MATCH_ERROR";
  case NPP_MEMFREE_ERROR:
    return "NPP_MEMFREE_ERROR";
  case NPP_MEMSET_ERROR:
    return "NPP_MEMSET_ERROR";
  case NPP_MEMCPY_ERROR:
    return "NPP_MEMCPY_ERROR";
  case NPP_ALIGNMENT_ERROR:
    return "NPP_ALIGNMENT_ERROR";
  case NPP_CUDA_KERNEL_EXECUTION_ERROR:
    return "NPP_CUDA_KERNEL_EXECUTION_ERROR";
  case NPP_ROUND_MODE_NOT_SUPPORTED_ERROR:
    return "NPP_ROUND_MODE_NOT_SUPPORTED_ERROR";
  case NPP_QUALITY_INDEX_ERROR:
    return "NPP_QUALITY_INDEX_ERROR";
  case NPP_RESIZE_NO_OPERATION_ERROR:
    return "NPP_RESIZE_NO_OPERATION_ERROR";
  case NPP_OVERFLOW_ERROR:
    return "NPP_OVERFLOW_ERROR";
  case NPP_NOT_EVEN_STEP_ERROR:
    return "NPP_NOT_EVEN_STEP_ERROR";
  case NPP_HISTOGRAM_NUMBER_OF_LEVELS_ERROR:
    return "NPP_HISTOGRAM_NUMBER_OF_LEVELS_ERROR";
  case NPP_LUT_NUMBER_OF_LEVELS_ERROR:
    return "NPP_LUT_NUMBER_OF_LEVELS_ERROR";
  case NPP_CORRUPTED_DATA_ERROR:
    return "NPP_CORRUPTED_DATA_ERROR";
  case NPP_CHANNEL_ORDER_ERROR:
    return "NPP_CHANNEL_ORDER_ERROR";
  case NPP_ZERO_MASK_VALUE_ERROR:
    return "NPP_ZERO_MASK_VALUE_ERROR";
  case NPP_QUADRANGLE_ERROR:
    return "NPP_QUADRANGLE_ERROR";
  case NPP_RECTANGLE_ERROR:
    return "NPP_RECTANGLE_ERROR";
  case NPP_COEFFICIENT_ERROR:
    return "NPP_COEFFICIENT_ERROR";
  case NPP_NUMBER_OF_CHANNELS_ERROR:
    return "NPP_NUMBER_OF_CHANNELS_ERROR";
  case NPP_COI_ERROR:
    return "NPP_COI_ERROR";
  case NPP_DIVISOR_ERROR:
    return "NPP_DIVISOR_ERROR";
  case NPP_CHANNEL_ERROR:
    return "NPP_CHANNEL_ERROR";
  case NPP_STRIDE_ERROR:
    return "NPP_STRIDE_ERROR";
  case NPP_ANCHOR_ERROR:
    return "NPP_ANCHOR_ERROR";
  case NPP_MASK_SIZE_ERROR:
    return "NPP_MASK_SIZE_ERROR";
  case NPP_RESIZE_FACTOR_ERROR:
    return "NPP_RESIZE_FACTOR_ERROR";
  case NPP_INTERPOLATION_ERROR:
    return "NPP_INTERPOLATION_ERROR";
  case NPP_MIRROR_FLIP_ERROR:
    return "NPP_MIRROR_FLIP_ERROR";
  case NPP_MOMENT_00_ZERO_ERROR:
    return "NPP_MOMENT_00_ZERO_ERROR";
  case NPP_THRESHOLD_NEGATIVE_LEVEL_ERROR:
    return "NPP_THRESHOLD_NEGATIVE_LEVEL_ERROR";
  case NPP_THRESHOLD_ERROR:
    return "NPP_THRESHOLD_ERROR";
  case NPP_CONTEXT_MATCH_ERROR:
    return "NPP_CONTEXT_MATCH_ERROR";
  case NPP_FFT_FLAG_ERROR:
    return "NPP_FFT_FLAG_ERROR";
  case NPP_FFT_ORDER_ERROR:
    return "NPP_FFT_ORDER_ERROR";
  case NPP_STEP_ERROR:
    return "NPP_STEP_ERROR";
  case NPP_SCALE_RANGE_ERROR:
    return "NPP_SCALE_RANGE_ERROR";
  case NPP_DATA_TYPE_ERROR:
    return "NPP_DATA_TYPE_ERROR";
  case NPP_OUT_OFF_RANGE_ERROR:
    return "NPP_OUT_OFF_RANGE_ERROR";
  case NPP_DIVIDE_BY_ZERO_ERROR:
    return "NPP_DIVIDE_BY_ZERO_ERROR";
  case NPP_MEMORY_ALLOCATION_ERR:
    return "NPP_MEMORY_ALLOCATION_ERR";
  case NPP_NULL_POINTER_ERROR:
    return "NPP_NULL_POINTER_ERROR";
  case NPP_RANGE_ERROR:
    return "NPP_RANGE_ERROR";
  case NPP_SIZE_ERROR:
    return "NPP_SIZE_ERROR";
  case NPP_BAD_ARGUMENT_ERROR:
    return "NPP_BAD_ARGUMENT_ERROR";
  case NPP_NO_MEMORY_ERROR:
    return "NPP_NO_MEMORY_ERROR";
  case NPP_NOT_IMPLEMENTED_ERROR:
    return "NPP_NOT_IMPLEMENTED_ERROR";
  case NPP_ERROR:
    return "NPP_ERROR";
  case NPP_ERROR_RESERVED:
    return "NPP_ERROR_RESERVED";
  default: {
    static char buf[64];
    snprintf(buf, sizeof(buf), "NPP_UNKNOWN_ERROR(%d)", status);
    return buf;
  }
  }
}
