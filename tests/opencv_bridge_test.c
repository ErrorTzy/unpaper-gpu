// SPDX-FileCopyrightText: 2025 The unpaper authors
//
// SPDX-License-Identifier: GPL-2.0-only

#include <assert.h>
#include <stdbool.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "imageprocess/cuda_kernels_format.h"
#include "imageprocess/cuda_runtime.h"
#include "imageprocess/opencv_bridge.h"

static void test_opencv_availability(void) {
  bool enabled = unpaper_opencv_enabled();
#ifdef UNPAPER_WITH_OPENCV
  assert(enabled);
  printf("  OpenCV enabled: true\n");
#else
  assert(!enabled);
  printf("  OpenCV enabled: false (stub)\n");
#endif
}

static void test_stream_raw_handle(void) {
  UnpaperCudaStream *stream = unpaper_cuda_stream_create();
  if (stream == NULL) {
    stream = unpaper_cuda_stream_get_default();
  }
  assert(stream != NULL);

  void *raw = unpaper_cuda_stream_get_raw_handle(stream);
  printf("  Raw stream handle: %p\n", raw);
  assert(raw != NULL);

  if (stream != unpaper_cuda_stream_get_default()) {
    unpaper_cuda_stream_destroy(stream);
  }
}

// Forward declaration of cudart functions to avoid including cuda_runtime.h in C
// These will be resolved at link time since we link cudart
typedef int cudaError_t;
extern cudaError_t cudaMalloc(void **devPtr, size_t size);
extern cudaError_t cudaFree(void *devPtr);
extern cudaError_t cudaMemcpy(void *dst, const void *src, size_t count, int kind);
#define cudaMemcpyHostToDevice 1
#define cudaMemcpyDeviceToHost 2

static void test_mask_extraction_gray8(void) {
  const int width = 16;
  const int height = 16;
  const size_t pitch = width;
  const size_t bytes = pitch * height;

  uint8_t *host_buf = malloc(bytes);
  assert(host_buf != NULL);

  for (int y = 0; y < height; y++) {
    for (int x = 0; x < width; x++) {
      if (x < 8 && y < 8) {
        host_buf[y * pitch + x] = 32;
      } else if (x >= 8 && y >= 8) {
        host_buf[y * pitch + x] = 64;
      } else {
        host_buf[y * pitch + x] = 200;
      }
    }
  }

  // Use cudaMalloc (runtime API) for compatibility with OpenCV CUDA
  void *cuda_ptr = NULL;
  cudaError_t err = cudaMalloc(&cuda_ptr, bytes);
  assert(err == 0 && cuda_ptr != NULL);
  uint64_t src_dptr = (uint64_t)cuda_ptr;
  err = cudaMemcpy(cuda_ptr, host_buf, bytes, cudaMemcpyHostToDevice);
  assert(err == 0);

  UnpaperOpencvMask mask = {0};
  const uint8_t min_white_level = 128;

  bool ok = unpaper_opencv_extract_dark_mask(
      src_dptr, width, height, pitch, (int)UNPAPER_CUDA_FMT_GRAY8,
      min_white_level, NULL, &mask);

#ifdef UNPAPER_WITH_OPENCV
  if (!unpaper_opencv_cuda_supported()) {
    printf("  OpenCV CUDA not supported, skipping mask extraction test\n");
    unpaper_cuda_free(src_dptr);
    free(host_buf);
    return;
  }
  assert(ok);
  assert(mask.device_ptr != 0);
  assert(mask.width == width);
  assert(mask.height == height);
  printf("  Mask extracted: %dx%d, pitch=%zu\n", mask.width, mask.height,
         mask.pitch_bytes);

  uint8_t *mask_host = malloc(mask.pitch_bytes * mask.height);
  assert(mask_host != NULL);
  // Use cudaMemcpy (runtime API) for compatibility
  err = cudaMemcpy(mask_host, (void *)mask.device_ptr,
                   mask.pitch_bytes * mask.height, cudaMemcpyDeviceToHost);
  assert(err == 0);

  int dark_count = 0;
  int bright_count = 0;
  for (int y = 0; y < height; y++) {
    for (int x = 0; x < width; x++) {
      uint8_t val = mask_host[y * mask.pitch_bytes + x];
      if (val != 0) {
        dark_count++;
      } else {
        bright_count++;
      }
    }
  }
  printf("  Dark pixels (mask=255): %d, Bright pixels (mask=0): %d\n",
         dark_count, bright_count);
  assert(dark_count == 8 * 8 + 8 * 8);
  assert(bright_count == 2 * 8 * 8);

  free(mask_host);
  unpaper_opencv_mask_free(&mask);
  assert(mask.device_ptr == 0);
#else
  assert(!ok);
  printf("  Mask extraction skipped (OpenCV not enabled)\n");
#endif

  cudaFree((void *)src_dptr);
  free(host_buf);
}

static void test_mask_round_trip_determinism(void) {
  const int width = 32;
  const int height = 32;
  const size_t pitch = width;
  const size_t bytes = pitch * height;

  uint8_t *host_buf = malloc(bytes);
  assert(host_buf != NULL);

  for (size_t i = 0; i < bytes; i++) {
    host_buf[i] = (uint8_t)((i * 37 + 17) % 256);
  }

  // Use cudaMalloc (runtime API) for compatibility with OpenCV CUDA
  void *cuda_ptr = NULL;
  cudaError_t err = cudaMalloc(&cuda_ptr, bytes);
  assert(err == 0 && cuda_ptr != NULL);
  uint64_t src_dptr = (uint64_t)cuda_ptr;
  err = cudaMemcpy(cuda_ptr, host_buf, bytes, cudaMemcpyHostToDevice);
  assert(err == 0);

  const uint8_t min_white_level = 128;

#ifdef UNPAPER_WITH_OPENCV
  if (!unpaper_opencv_cuda_supported()) {
    printf("  OpenCV CUDA not supported, skipping determinism test\n");
    cudaFree((void *)src_dptr);
    free(host_buf);
    return;
  }

  uint8_t *first_run = NULL;
  size_t first_pitch = 0;

  for (int run = 0; run < 3; run++) {
    UnpaperOpencvMask mask = {0};
    bool ok = unpaper_opencv_extract_dark_mask(
        src_dptr, width, height, pitch, (int)UNPAPER_CUDA_FMT_GRAY8,
        min_white_level, NULL, &mask);
    assert(ok);

    uint8_t *mask_host = malloc(mask.pitch_bytes * mask.height);
    assert(mask_host != NULL);
    err = cudaMemcpy(mask_host, (void *)mask.device_ptr,
                     mask.pitch_bytes * mask.height, cudaMemcpyDeviceToHost);
    assert(err == 0);

    if (run == 0) {
      first_run = mask_host;
      first_pitch = mask.pitch_bytes;
    } else {
      for (int y = 0; y < height; y++) {
        for (int x = 0; x < width; x++) {
          assert(mask_host[y * mask.pitch_bytes + x] ==
                 first_run[y * first_pitch + x]);
        }
      }
      free(mask_host);
    }

    unpaper_opencv_mask_free(&mask);
  }

  free(first_run);
  printf("  Determinism verified across 3 runs\n");
#else
  printf("  Determinism test skipped (OpenCV not enabled)\n");
#endif

  cudaFree((void *)src_dptr);
  free(host_buf);
}

int main(void) {
  printf("Test: OpenCV bridge availability\n");
  test_opencv_availability();

#ifdef UNPAPER_WITH_OPENCV
  // When OpenCV CUDA is enabled, we use CUDA Runtime API (cudaMalloc, etc.)
  // instead of unpaper's Driver API to avoid context conflicts.
  // Skip the stream test as it requires unpaper's Driver API context.
  printf("Test: stream raw handle\n");
  printf("  (skipped - using CUDA Runtime API for OpenCV compatibility)\n");
#else
  UnpaperCudaInitStatus st = unpaper_cuda_try_init();
  if (st != UNPAPER_CUDA_INIT_OK) {
    printf("CUDA not available: %s\n", unpaper_cuda_init_status_string(st));
    printf("Skipping CUDA-dependent tests\n");
    return 0;
  }

  printf("Test: stream raw handle\n");
  test_stream_raw_handle();
#endif

  printf("Test: mask extraction GRAY8\n");
  test_mask_extraction_gray8();

  printf("Test: mask round-trip determinism\n");
  test_mask_round_trip_determinism();

  printf("All OpenCV bridge tests passed!\n");
  return 0;
}
