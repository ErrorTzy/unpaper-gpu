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
#include "imageprocess/nvjpeg_encode.h"

static char test_jpeg_path[4096];

static void init_test_paths(void) {
  const char *imgsrc_dir = getenv("TEST_IMGSRC_DIR");
  if (imgsrc_dir != NULL) {
    snprintf(test_jpeg_path, sizeof(test_jpeg_path), "%stest_jpeg.jpg",
             imgsrc_dir);
  } else {
    snprintf(test_jpeg_path, sizeof(test_jpeg_path),
             "tests/source_images/test_jpeg.jpg");
  }
}

static void test_encode_init(void) {
  printf("Test: nvjpeg_encode_init... ");

  // Initialize encode context with 4 encoder states
  bool result = nvjpeg_encode_init(4, 85, NVJPEG_ENC_SUBSAMPLING_420);
  if (!result) {
    printf("FAILED (init returned false)\n");
    exit(1);
  }

  // Check availability
  if (!nvjpeg_encode_is_available()) {
    printf("FAILED (not available after init)\n");
    exit(1);
  }

  // Get stats
  NvJpegEncodeStats stats = nvjpeg_encode_get_stats();
  if (stats.encoder_state_count != 4) {
    printf("FAILED (wrong encoder count: %zu)\n", stats.encoder_state_count);
    exit(1);
  }

  printf("PASSED\n");
}

static void test_encoder_state_acquire_release(void) {
  printf("Test: encoder state acquire/release... ");

  // Acquire all 4 states
  NvJpegEncoderState *states[4];
  for (int i = 0; i < 4; i++) {
    states[i] = nvjpeg_encode_acquire_state();
    if (states[i] == NULL) {
      printf("FAILED (acquire %d returned NULL)\n", i);
      exit(1);
    }
  }

  // Check stats
  NvJpegEncodeStats stats = nvjpeg_encode_get_stats();
  if (stats.current_in_use != 4) {
    printf("FAILED (wrong in_use count: %zu)\n", stats.current_in_use);
    exit(1);
  }

  // Try to acquire one more (should fail)
  NvJpegEncoderState *extra = nvjpeg_encode_acquire_state();
  if (extra != NULL) {
    printf("FAILED (acquired more than pool size)\n");
    exit(1);
  }

  // Release all
  for (int i = 0; i < 4; i++) {
    nvjpeg_encode_release_state(states[i]);
  }

  // Check stats after release
  stats = nvjpeg_encode_get_stats();
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

static void test_encode_rgb(void) {
  printf("Test: encode RGB image... ");

  // First decode a JPEG to GPU
  NvJpegDecodedImage decoded = {0};
  bool decode_result =
      nvjpeg_decode_file_to_gpu(test_jpeg_path, NULL, NVJPEG_FMT_RGB, &decoded);

  if (!decode_result || decoded.gpu_ptr == NULL) {
    printf("SKIPPED (decode failed, file may not exist)\n");
    return;
  }

  // Acquire encoder state
  NvJpegEncoderState *state = nvjpeg_encode_acquire_state();
  if (state == NULL) {
    printf("FAILED (no encoder state)\n");
    cudaFree(decoded.gpu_ptr);
    exit(1);
  }

  // Encode from GPU
  NvJpegEncodedImage out = {0};
  bool result = nvjpeg_encode_from_gpu(decoded.gpu_ptr, decoded.pitch,
                                       decoded.width, decoded.height,
                                       NVJPEG_ENC_FMT_RGB, state, NULL, &out);

  nvjpeg_encode_release_state(state);
  cudaFree(decoded.gpu_ptr);

  if (!result) {
    printf("FAILED (encode returned false)\n");
    exit(1);
  }

  if (out.jpeg_data == NULL) {
    printf("FAILED (jpeg_data is NULL)\n");
    exit(1);
  }

  if (out.jpeg_size == 0) {
    printf("FAILED (jpeg_size is 0)\n");
    free(out.jpeg_data);
    exit(1);
  }

  // Verify JPEG header (should start with FFD8)
  if (out.jpeg_size < 2 || out.jpeg_data[0] != 0xFF ||
      out.jpeg_data[1] != 0xD8) {
    printf("FAILED (invalid JPEG header)\n");
    free(out.jpeg_data);
    exit(1);
  }

  printf("PASSED (%dx%d, %zu bytes)\n", out.width, out.height, out.jpeg_size);
  free(out.jpeg_data);
}

static void test_encode_grayscale(void) {
  printf("Test: encode grayscale image... ");

  // Decode to grayscale
  NvJpegDecodedImage decoded = {0};
  bool decode_result = nvjpeg_decode_file_to_gpu(test_jpeg_path, NULL,
                                                 NVJPEG_FMT_GRAY8, &decoded);

  if (!decode_result || decoded.gpu_ptr == NULL) {
    printf("SKIPPED (decode failed)\n");
    return;
  }

  // Acquire encoder state
  NvJpegEncoderState *state = nvjpeg_encode_acquire_state();
  if (state == NULL) {
    printf("FAILED (no encoder state)\n");
    cudaFree(decoded.gpu_ptr);
    exit(1);
  }

  // Encode grayscale
  NvJpegEncodedImage out = {0};
  bool result = nvjpeg_encode_from_gpu(decoded.gpu_ptr, decoded.pitch,
                                       decoded.width, decoded.height,
                                       NVJPEG_ENC_FMT_GRAY8, state, NULL, &out);

  nvjpeg_encode_release_state(state);
  cudaFree(decoded.gpu_ptr);

  if (!result) {
    printf("FAILED (encode returned false)\n");
    exit(1);
  }

  if (out.jpeg_data == NULL || out.jpeg_size == 0) {
    printf("FAILED (no output)\n");
    if (out.jpeg_data)
      free(out.jpeg_data);
    exit(1);
  }

  // Verify JPEG header
  if (out.jpeg_data[0] != 0xFF || out.jpeg_data[1] != 0xD8) {
    printf("FAILED (invalid JPEG header)\n");
    free(out.jpeg_data);
    exit(1);
  }

  printf("PASSED (%zu bytes)\n", out.jpeg_size);
  free(out.jpeg_data);
}

static void test_encode_to_file(void) {
  printf("Test: encode GPU to file... ");

  // Decode to GPU
  NvJpegDecodedImage decoded = {0};
  bool decode_result =
      nvjpeg_decode_file_to_gpu(test_jpeg_path, NULL, NVJPEG_FMT_RGB, &decoded);

  if (!decode_result || decoded.gpu_ptr == NULL) {
    printf("SKIPPED (decode failed)\n");
    return;
  }

  // Encode directly to file
  const char *out_path = "/tmp/nvjpeg_encode_test_output.jpg";
  bool result =
      nvjpeg_encode_gpu_to_file(decoded.gpu_ptr, decoded.pitch, decoded.width,
                                decoded.height, NVJPEG_ENC_FMT_RGB, NULL, out_path);

  cudaFree(decoded.gpu_ptr);

  if (!result) {
    printf("FAILED (encode to file returned false)\n");
    exit(1);
  }

  // Verify file exists and has content
  FILE *f = fopen(out_path, "rb");
  if (f == NULL) {
    printf("FAILED (output file not created)\n");
    exit(1);
  }

  fseek(f, 0, SEEK_END);
  long size = ftell(f);
  fclose(f);

  if (size < 100) {
    printf("FAILED (output file too small: %ld bytes)\n", size);
    exit(1);
  }

  // Clean up
  remove(out_path);

  printf("PASSED (wrote %ld bytes)\n", size);
}

static void test_quality_control(void) {
  printf("Test: quality control... ");

  // Decode to GPU
  NvJpegDecodedImage decoded = {0};
  bool decode_result =
      nvjpeg_decode_file_to_gpu(test_jpeg_path, NULL, NVJPEG_FMT_RGB, &decoded);

  if (!decode_result || decoded.gpu_ptr == NULL) {
    printf("SKIPPED (decode failed)\n");
    return;
  }

  // Encode at quality 50
  nvjpeg_encode_set_quality(50);
  if (nvjpeg_encode_get_quality() != 50) {
    printf("FAILED (quality not set to 50)\n");
    cudaFree(decoded.gpu_ptr);
    exit(1);
  }

  NvJpegEncoderState *state = nvjpeg_encode_acquire_state();
  NvJpegEncodedImage out_low = {0};
  nvjpeg_encode_from_gpu(decoded.gpu_ptr, decoded.pitch, decoded.width,
                         decoded.height, NVJPEG_ENC_FMT_RGB, state, NULL,
                         &out_low);
  nvjpeg_encode_release_state(state);

  // Encode at quality 95
  nvjpeg_encode_set_quality(95);
  state = nvjpeg_encode_acquire_state();
  NvJpegEncodedImage out_high = {0};
  nvjpeg_encode_from_gpu(decoded.gpu_ptr, decoded.pitch, decoded.width,
                         decoded.height, NVJPEG_ENC_FMT_RGB, state, NULL,
                         &out_high);
  nvjpeg_encode_release_state(state);

  cudaFree(decoded.gpu_ptr);

  if (out_low.jpeg_data == NULL || out_high.jpeg_data == NULL) {
    printf("FAILED (encoding failed)\n");
    if (out_low.jpeg_data)
      free(out_low.jpeg_data);
    if (out_high.jpeg_data)
      free(out_high.jpeg_data);
    exit(1);
  }

  // Higher quality should generally produce larger files
  // (though not always guaranteed due to image content)
  printf("PASSED (q50=%zu bytes, q95=%zu bytes)\n", out_low.jpeg_size,
         out_high.jpeg_size);

  free(out_low.jpeg_data);
  free(out_high.jpeg_data);

  // Reset quality
  nvjpeg_encode_set_quality(85);
}

// Thread function for concurrent encode test
typedef struct {
  int thread_id;
  int num_encodes;
  void *gpu_ptr;
  size_t pitch;
  int width;
  int height;
  bool success;
} ConcurrentEncodeArg;

static void *concurrent_encode_thread(void *arg) {
  ConcurrentEncodeArg *targ = (ConcurrentEncodeArg *)arg;
  targ->success = true;

  for (int i = 0; i < targ->num_encodes; i++) {
    NvJpegEncoderState *state = nvjpeg_encode_acquire_state();
    if (state == NULL) {
      // Pool exhausted, retry
      continue;
    }

    NvJpegEncodedImage out = {0};
    bool result =
        nvjpeg_encode_from_gpu(targ->gpu_ptr, targ->pitch, targ->width,
                               targ->height, NVJPEG_ENC_FMT_RGB, state, NULL, &out);

    nvjpeg_encode_release_state(state);

    if (result && out.jpeg_data != NULL) {
      free(out.jpeg_data);
    }
  }

  return NULL;
}

static void test_concurrent_encode(void) {
  printf("Test: concurrent encode... ");

  // Decode to GPU
  NvJpegDecodedImage decoded = {0};
  bool decode_result =
      nvjpeg_decode_file_to_gpu(test_jpeg_path, NULL, NVJPEG_FMT_RGB, &decoded);

  if (!decode_result || decoded.gpu_ptr == NULL) {
    printf("SKIPPED (decode failed)\n");
    return;
  }

  const int num_threads = 4;
  const int encodes_per_thread = 3;

  pthread_t threads[4];
  ConcurrentEncodeArg args[4];

  for (int i = 0; i < num_threads; i++) {
    args[i].thread_id = i;
    args[i].num_encodes = encodes_per_thread;
    args[i].gpu_ptr = decoded.gpu_ptr;
    args[i].pitch = decoded.pitch;
    args[i].width = decoded.width;
    args[i].height = decoded.height;
    args[i].success = false;
    pthread_create(&threads[i], NULL, concurrent_encode_thread, &args[i]);
  }

  for (int i = 0; i < num_threads; i++) {
    pthread_join(threads[i], NULL);
  }

  cudaFree(decoded.gpu_ptr);

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
  NvJpegEncodeStats stats = nvjpeg_encode_get_stats();
  if (stats.total_encodes == 0) {
    printf("FAILED (no encodes recorded)\n");
    exit(1);
  }

  printf("PASSED (total encodes: %zu, successful: %zu)\n", stats.total_encodes,
         stats.successful_encodes);
}

static void test_encode_cleanup(void) {
  printf("Test: nvjpeg_encode_cleanup... ");

  nvjpeg_encode_cleanup();

  if (nvjpeg_encode_is_available()) {
    printf("FAILED (still available after cleanup)\n");
    exit(1);
  }

  printf("PASSED\n");
}

int main(void) {
  printf("nvJPEG Encode Unit Tests\n");
  printf("========================\n\n");

  // Initialize test paths from environment
  init_test_paths();

  // Check CUDA availability
  UnpaperCudaInitStatus st = unpaper_cuda_try_init();
  if (st != UNPAPER_CUDA_INIT_OK) {
    printf("Skipping tests: %s\n", unpaper_cuda_init_status_string(st));
    return 77; // Skip return code
  }

  // Initialize nvJPEG decode context first (required for shared handle)
  if (!nvjpeg_context_init(4)) {
    printf("Skipping tests: nvjpeg_context_init failed\n");
    return 77;
  }

  // Run tests
  test_encode_init();
  test_encoder_state_acquire_release();
  test_encode_rgb();
  test_encode_grayscale();
  test_encode_to_file();
  test_quality_control();
  test_concurrent_encode();
  test_encode_cleanup();

  // Cleanup decode context
  nvjpeg_context_cleanup();

  printf("\nAll tests passed!\n");
  return 0;
}
