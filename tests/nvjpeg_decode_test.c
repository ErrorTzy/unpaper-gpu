// SPDX-FileCopyrightText: 2025 The unpaper authors
//
// SPDX-License-Identifier: GPL-2.0-only

#include <cuda_runtime.h>
#include <pthread.h>
#include <stdbool.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "imageprocess/cuda_runtime.h"
#include "imageprocess/nvjpeg_decode.h"

static char test_jpeg_path[4096];

static void init_test_paths(void) {
  const char *imgsrc_dir = getenv("TEST_IMGSRC_DIR");
  if (imgsrc_dir != NULL) {
    snprintf(test_jpeg_path, sizeof(test_jpeg_path), "%stest_jpeg.jpg", imgsrc_dir);
  } else {
    snprintf(test_jpeg_path, sizeof(test_jpeg_path), "tests/source_images/test_jpeg.jpg");
  }
}

static void test_context_init(void) {
  printf("Test: nvjpeg_context_init... ");

  // Initialize with 4 stream states
  bool result = nvjpeg_context_init(4);
  if (!result) {
    printf("FAILED (init returned false)\n");
    exit(1);
  }

  // Check availability
  if (!nvjpeg_is_available()) {
    printf("FAILED (not available after init)\n");
    exit(1);
  }

  // Get stats
  NvJpegStats stats = nvjpeg_get_stats();
  if (stats.stream_state_count != 4) {
    printf("FAILED (wrong stream count: %zu)\n", stats.stream_state_count);
    exit(1);
  }

  printf("PASSED\n");
}

static void test_stream_state_acquire_release(void) {
  printf("Test: stream state acquire/release... ");

  // Acquire all 4 states
  NvJpegStreamState *states[4];
  for (int i = 0; i < 4; i++) {
    states[i] = nvjpeg_acquire_stream_state();
    if (states[i] == NULL) {
      printf("FAILED (acquire %d returned NULL)\n", i);
      exit(1);
    }
  }

  // Check stats
  NvJpegStats stats = nvjpeg_get_stats();
  if (stats.current_in_use != 4) {
    printf("FAILED (wrong in_use count: %zu)\n", stats.current_in_use);
    exit(1);
  }

  // Try to acquire one more (should fail)
  NvJpegStreamState *extra = nvjpeg_acquire_stream_state();
  if (extra != NULL) {
    printf("FAILED (acquired more than pool size)\n");
    exit(1);
  }

  // Release all
  for (int i = 0; i < 4; i++) {
    nvjpeg_release_stream_state(states[i]);
  }

  // Check stats after release
  stats = nvjpeg_get_stats();
  if (stats.current_in_use != 0) {
    printf("FAILED (in_use not 0 after release: %zu)\n", stats.current_in_use);
    exit(1);
  }

  // Peak should be 4
  if (stats.concurrent_peak != 4) {
    printf("FAILED (wrong peak: %zu)\n", stats.concurrent_peak);
    exit(1);
  }

  printf("PASSED\n");
}

static void test_image_info(void) {
  printf("Test: nvjpeg_get_image_info... ");

  // Read test JPEG file
  FILE *f = fopen(test_jpeg_path, "rb");
  if (f == NULL) {
    printf("SKIPPED (test JPEG not found)\n");
    return;
  }

  fseek(f, 0, SEEK_END);
  long file_size = ftell(f);
  fseek(f, 0, SEEK_SET);

  uint8_t *jpeg_data = malloc((size_t)file_size);
  if (jpeg_data == NULL) {
    fclose(f);
    printf("FAILED (malloc failed)\n");
    exit(1);
  }

  size_t bytes_read = fread(jpeg_data, 1, (size_t)file_size, f);
  fclose(f);

  if (bytes_read != (size_t)file_size) {
    free(jpeg_data);
    printf("FAILED (read incomplete)\n");
    exit(1);
  }

  int width = 0, height = 0, channels = 0;
  bool result =
      nvjpeg_get_image_info(jpeg_data, (size_t)file_size, &width, &height, &channels);
  free(jpeg_data);

  if (!result) {
    printf("FAILED (get_image_info returned false)\n");
    exit(1);
  }

  if (width <= 0 || height <= 0) {
    printf("FAILED (invalid dimensions: %dx%d)\n", width, height);
    exit(1);
  }

  if (channels != 1 && channels != 3) {
    printf("FAILED (invalid channels: %d)\n", channels);
    exit(1);
  }

  printf("PASSED (%dx%d, %d channels)\n", width, height, channels);
}

static void test_decode_grayscale(void) {
  printf("Test: decode to grayscale... ");

  // Read test JPEG file
  FILE *f = fopen(test_jpeg_path, "rb");
  if (f == NULL) {
    printf("SKIPPED (test JPEG not found)\n");
    return;
  }

  fseek(f, 0, SEEK_END);
  long file_size = ftell(f);
  fseek(f, 0, SEEK_SET);

  uint8_t *jpeg_data = malloc((size_t)file_size);
  if (jpeg_data == NULL) {
    fclose(f);
    printf("FAILED (malloc failed)\n");
    exit(1);
  }

  size_t bytes_read = fread(jpeg_data, 1, (size_t)file_size, f);
  fclose(f);

  // Acquire stream state
  NvJpegStreamState *state = nvjpeg_acquire_stream_state();
  if (state == NULL) {
    free(jpeg_data);
    printf("FAILED (no stream state)\n");
    exit(1);
  }

  // Decode to grayscale
  NvJpegDecodedImage out = {0};
  bool result = nvjpeg_decode_to_gpu(jpeg_data, (size_t)file_size, state, NULL,
                                     NVJPEG_FMT_GRAY8, &out);

  free(jpeg_data);
  nvjpeg_release_stream_state(state);

  if (!result) {
    printf("FAILED (decode returned false)\n");
    exit(1);
  }

  if (out.gpu_ptr == NULL) {
    printf("FAILED (gpu_ptr is NULL)\n");
    exit(1);
  }

  if (out.width <= 0 || out.height <= 0) {
    printf("FAILED (invalid dimensions)\n");
    cudaFree(out.gpu_ptr);
    exit(1);
  }

  if (out.channels != 1) {
    printf("FAILED (wrong channels: %d)\n", out.channels);
    cudaFree(out.gpu_ptr);
    exit(1);
  }

  // Free GPU memory
  cudaFree(out.gpu_ptr);

  printf("PASSED (%dx%d)\n", out.width, out.height);
}

static void test_decode_rgb(void) {
  printf("Test: decode to RGB... ");

  // Read test JPEG file
  FILE *f = fopen(test_jpeg_path, "rb");
  if (f == NULL) {
    printf("SKIPPED (test JPEG not found)\n");
    return;
  }

  fseek(f, 0, SEEK_END);
  long file_size = ftell(f);
  fseek(f, 0, SEEK_SET);

  uint8_t *jpeg_data = malloc((size_t)file_size);
  if (jpeg_data == NULL) {
    fclose(f);
    printf("FAILED (malloc failed)\n");
    exit(1);
  }

  size_t bytes_read = fread(jpeg_data, 1, (size_t)file_size, f);
  fclose(f);

  // Acquire stream state
  NvJpegStreamState *state = nvjpeg_acquire_stream_state();
  if (state == NULL) {
    free(jpeg_data);
    printf("FAILED (no stream state)\n");
    exit(1);
  }

  // Decode to RGB
  NvJpegDecodedImage out = {0};
  bool result = nvjpeg_decode_to_gpu(jpeg_data, (size_t)file_size, state, NULL,
                                     NVJPEG_FMT_RGB, &out);

  free(jpeg_data);
  nvjpeg_release_stream_state(state);

  if (!result) {
    printf("FAILED (decode returned false)\n");
    exit(1);
  }

  if (out.gpu_ptr == NULL) {
    printf("FAILED (gpu_ptr is NULL)\n");
    exit(1);
  }

  if (out.channels != 3) {
    printf("FAILED (wrong channels: %d)\n", out.channels);
    cudaFree(out.gpu_ptr);
    exit(1);
  }

  // Verify we can read back some data
  size_t test_size = 1024;
  if (out.pitch * (size_t)out.height < test_size) {
    test_size = out.pitch * (size_t)out.height;
  }
  uint8_t *host_data = malloc(test_size);
  if (host_data != NULL) {
    cudaError_t err =
        cudaMemcpy(host_data, out.gpu_ptr, test_size, cudaMemcpyDeviceToHost);
    if (err != cudaSuccess) {
      printf("FAILED (cudaMemcpy failed)\n");
      free(host_data);
      cudaFree(out.gpu_ptr);
      exit(1);
    }
    free(host_data);
  }

  cudaFree(out.gpu_ptr);

  printf("PASSED (%dx%d, pitch=%zu)\n", out.width, out.height, out.pitch);
}

static void test_decode_file(void) {
  printf("Test: nvjpeg_decode_file_to_gpu... ");

  NvJpegDecodedImage out = {0};
  bool result =
      nvjpeg_decode_file_to_gpu(test_jpeg_path, NULL, NVJPEG_FMT_RGB, &out);

  if (!result) {
    printf("SKIPPED (decode failed, file may not exist)\n");
    return;
  }

  if (out.gpu_ptr == NULL) {
    printf("FAILED (gpu_ptr is NULL)\n");
    exit(1);
  }

  cudaFree(out.gpu_ptr);

  printf("PASSED\n");
}

// Thread function for concurrent decode test
typedef struct {
  int thread_id;
  int num_decodes;
  bool success;
} ConcurrentTestArg;

static void *concurrent_decode_thread(void *arg) {
  ConcurrentTestArg *targ = (ConcurrentTestArg *)arg;
  targ->success = true;

  // Read test JPEG file
  FILE *f = fopen(test_jpeg_path, "rb");
  if (f == NULL) {
    targ->success = false;
    return NULL;
  }

  fseek(f, 0, SEEK_END);
  long file_size = ftell(f);
  fseek(f, 0, SEEK_SET);

  uint8_t *jpeg_data = malloc((size_t)file_size);
  if (jpeg_data == NULL) {
    fclose(f);
    targ->success = false;
    return NULL;
  }

  size_t bytes_read = fread(jpeg_data, 1, (size_t)file_size, f);
  fclose(f);

  if (bytes_read != (size_t)file_size) {
    free(jpeg_data);
    targ->success = false;
    return NULL;
  }

  for (int i = 0; i < targ->num_decodes; i++) {
    NvJpegStreamState *state = nvjpeg_acquire_stream_state();
    if (state == NULL) {
      // Pool exhausted, retry
      continue;
    }

    NvJpegDecodedImage out = {0};
    bool result = nvjpeg_decode_to_gpu(jpeg_data, (size_t)file_size, state,
                                       NULL, NVJPEG_FMT_GRAY8, &out);

    nvjpeg_release_stream_state(state);

    if (result && out.gpu_ptr != NULL) {
      cudaFree(out.gpu_ptr);
    }
  }

  free(jpeg_data);
  return NULL;
}

static void test_concurrent_decode(void) {
  printf("Test: concurrent decode... ");

  // Use the existing context (initialized with 4 states)
  // Each thread will acquire/release states as needed

  const int num_threads = 4;
  const int decodes_per_thread = 3;

  pthread_t threads[4];
  ConcurrentTestArg args[4];

  for (int i = 0; i < num_threads; i++) {
    args[i].thread_id = i;
    args[i].num_decodes = decodes_per_thread;
    args[i].success = false;
    pthread_create(&threads[i], NULL, concurrent_decode_thread, &args[i]);
  }

  for (int i = 0; i < num_threads; i++) {
    pthread_join(threads[i], NULL);
  }

  // Check results
  int failures = 0;
  for (int i = 0; i < num_threads; i++) {
    if (!args[i].success) {
      failures++;
    }
  }

  if (failures > 0) {
    printf("FAILED (%d threads failed)\n", failures);
    exit(1);
  }

  // Check statistics
  NvJpegStats stats = nvjpeg_get_stats();
  if (stats.total_decodes == 0) {
    printf("FAILED (no decodes recorded)\n");
    exit(1);
  }

  printf("PASSED (total decodes: %zu, peak concurrent: %zu)\n",
         stats.total_decodes, stats.concurrent_peak);
}

static void test_cleanup(void) {
  printf("Test: nvjpeg_context_cleanup... ");

  nvjpeg_context_cleanup();

  if (nvjpeg_is_available()) {
    printf("FAILED (still available after cleanup)\n");
    exit(1);
  }

  printf("PASSED\n");
}

int main(void) {
  printf("nvJPEG Decode Unit Tests\n");
  printf("========================\n\n");

  // Initialize test paths from environment
  init_test_paths();

  // Check CUDA availability
  UnpaperCudaInitStatus st = unpaper_cuda_try_init();
  if (st != UNPAPER_CUDA_INIT_OK) {
    printf("Skipping tests: %s\n", unpaper_cuda_init_status_string(st));
    return 77;  // Skip return code
  }

  // Run tests
  test_context_init();
  test_stream_state_acquire_release();
  test_image_info();
  test_decode_grayscale();
  test_decode_rgb();
  test_decode_file();
  test_concurrent_decode();
  test_cleanup();

  printf("\nAll tests passed!\n");
  return 0;
}
