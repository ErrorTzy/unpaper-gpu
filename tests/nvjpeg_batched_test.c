// SPDX-FileCopyrightText: 2025 The unpaper authors
//
// SPDX-License-Identifier: GPL-2.0-only

// Unit tests for nvJPEG batched decode API (PR36A)

#include <cuda_runtime.h>
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
    snprintf(test_jpeg_path, sizeof(test_jpeg_path), "%stest_jpeg.jpg",
             imgsrc_dir);
  } else {
    snprintf(test_jpeg_path, sizeof(test_jpeg_path),
             "tests/source_images/test_jpeg.jpg");
  }
}

// Helper to read JPEG file into memory
static uint8_t *read_jpeg_file(const char *path, size_t *size_out) {
  FILE *f = fopen(path, "rb");
  if (f == NULL) {
    return NULL;
  }

  fseek(f, 0, SEEK_END);
  long file_size = ftell(f);
  fseek(f, 0, SEEK_SET);

  if (file_size <= 0 || file_size > (long)(100 * 1024 * 1024)) {
    fclose(f);
    return NULL;
  }

  uint8_t *data = malloc((size_t)file_size);
  if (data == NULL) {
    fclose(f);
    return NULL;
  }

  size_t bytes_read = fread(data, 1, (size_t)file_size, f);
  fclose(f);

  if (bytes_read != (size_t)file_size) {
    free(data);
    return NULL;
  }

  *size_out = (size_t)file_size;
  return data;
}

static void test_batched_init(void) {
  printf("Test: nvjpeg_batched_init... ");

  // Should fail before context init
  bool result = nvjpeg_batched_init(4, 4000, 6000, NVJPEG_FMT_RGB);
  if (result) {
    printf("FAILED (should fail before context init)\n");
    exit(1);
  }

  // Initialize base context first
  result = nvjpeg_context_init(4);
  if (!result) {
    printf("FAILED (context init failed)\n");
    exit(1);
  }

  // Now batched init should work
  result = nvjpeg_batched_init(8, 4000, 6000, NVJPEG_FMT_RGB);
  if (!result) {
    printf("FAILED (batched init returned false)\n");
    exit(1);
  }

  // Check state
  if (!nvjpeg_batched_is_ready()) {
    printf("FAILED (not ready after init)\n");
    exit(1);
  }

  if (nvjpeg_batched_get_max_batch_size() != 8) {
    printf("FAILED (wrong batch size: %d)\n",
           nvjpeg_batched_get_max_batch_size());
    exit(1);
  }

  if (nvjpeg_batched_get_format() != NVJPEG_FMT_RGB) {
    printf("FAILED (wrong format)\n");
    exit(1);
  }

  printf("PASSED\n");
}

static void test_batched_reinit(void) {
  printf("Test: batched reinit with same config... ");

  // Reinit with same config should succeed (no-op)
  bool result = nvjpeg_batched_init(8, 4000, 6000, NVJPEG_FMT_RGB);
  if (!result) {
    printf("FAILED (reinit same config failed)\n");
    exit(1);
  }

  printf("PASSED\n");
}

static void test_decode_batch_single(void) {
  printf("Test: decode batch (single image)... ");

  size_t jpeg_size = 0;
  uint8_t *jpeg_data = read_jpeg_file(test_jpeg_path, &jpeg_size);
  if (jpeg_data == NULL) {
    printf("SKIPPED (test JPEG not found)\n");
    return;
  }

  // Create batch with single image
  const uint8_t *data_ptrs[1] = {jpeg_data};
  size_t sizes[1] = {jpeg_size};
  NvJpegDecodedImage outputs[1] = {0};

  int decoded = nvjpeg_decode_batch(data_ptrs, sizes, 1, outputs);

  free(jpeg_data);

  if (decoded != 1) {
    printf("FAILED (decoded %d images, expected 1)\n", decoded);
    exit(1);
  }

  if (outputs[0].gpu_ptr == NULL) {
    printf("FAILED (output gpu_ptr is NULL)\n");
    exit(1);
  }

  if (outputs[0].width <= 0 || outputs[0].height <= 0) {
    printf("FAILED (invalid dimensions: %dx%d)\n", outputs[0].width,
           outputs[0].height);
    exit(1);
  }

  if (outputs[0].channels != 3) {
    printf("FAILED (wrong channels: %d)\n", outputs[0].channels);
    exit(1);
  }

  printf("PASSED (%dx%d)\n", outputs[0].width, outputs[0].height);
}

static void test_decode_batch_multiple(void) {
  printf("Test: decode batch (4 copies)... ");

  size_t jpeg_size = 0;
  uint8_t *jpeg_data = read_jpeg_file(test_jpeg_path, &jpeg_size);
  if (jpeg_data == NULL) {
    printf("SKIPPED (test JPEG not found)\n");
    return;
  }

  // Create batch with 4 copies of same image
  const int batch_size = 4;
  const uint8_t *data_ptrs[4];
  size_t sizes[4];
  NvJpegDecodedImage outputs[4] = {0};

  for (int i = 0; i < batch_size; i++) {
    data_ptrs[i] = jpeg_data;
    sizes[i] = jpeg_size;
  }

  int decoded = nvjpeg_decode_batch(data_ptrs, sizes, batch_size, outputs);

  if (decoded != batch_size) {
    printf("FAILED (decoded %d images, expected %d)\n", decoded, batch_size);
    free(jpeg_data);
    exit(1);
  }

  // Verify all outputs
  for (int i = 0; i < batch_size; i++) {
    if (outputs[i].gpu_ptr == NULL) {
      printf("FAILED (output[%d].gpu_ptr is NULL)\n", i);
      free(jpeg_data);
      exit(1);
    }

    if (outputs[i].width != outputs[0].width ||
        outputs[i].height != outputs[0].height) {
      printf("FAILED (dimension mismatch at index %d)\n", i);
      free(jpeg_data);
      exit(1);
    }
  }

  // Verify we can read back data from each buffer
  size_t test_size = 1024;
  if (outputs[0].pitch * (size_t)outputs[0].height < test_size) {
    test_size = outputs[0].pitch * (size_t)outputs[0].height;
  }

  uint8_t *host_data = malloc(test_size);
  if (host_data != NULL) {
    for (int i = 0; i < batch_size; i++) {
      cudaError_t err = cudaMemcpy(host_data, outputs[i].gpu_ptr, test_size,
                                   cudaMemcpyDeviceToHost);
      if (err != cudaSuccess) {
        printf("FAILED (cudaMemcpy failed for output[%d])\n", i);
        free(host_data);
        free(jpeg_data);
        exit(1);
      }
    }
    free(host_data);
  }

  free(jpeg_data);

  printf("PASSED (4 images, each %dx%d)\n", outputs[0].width,
         outputs[0].height);
}

static void test_decode_batch_with_nulls(void) {
  printf("Test: decode batch with NULL entries... ");

  size_t jpeg_size = 0;
  uint8_t *jpeg_data = read_jpeg_file(test_jpeg_path, &jpeg_size);
  if (jpeg_data == NULL) {
    printf("SKIPPED (test JPEG not found)\n");
    return;
  }

  // Create batch with some NULL entries
  const int batch_size = 4;
  const uint8_t *data_ptrs[4] = {jpeg_data, NULL, jpeg_data, NULL};
  size_t sizes[4] = {jpeg_size, 0, jpeg_size, 0};
  NvJpegDecodedImage outputs[4] = {0};

  int decoded = nvjpeg_decode_batch(data_ptrs, sizes, batch_size, outputs);

  if (decoded != 2) {
    printf("FAILED (decoded %d images, expected 2)\n", decoded);
    free(jpeg_data);
    exit(1);
  }

  // Verify valid entries have data
  if (outputs[0].gpu_ptr == NULL || outputs[2].gpu_ptr == NULL) {
    printf("FAILED (valid entries have NULL gpu_ptr)\n");
    free(jpeg_data);
    exit(1);
  }

  // Verify NULL entries are marked as failed
  if (outputs[1].gpu_ptr != NULL || outputs[3].gpu_ptr != NULL) {
    printf("FAILED (NULL entries should have NULL gpu_ptr)\n");
    free(jpeg_data);
    exit(1);
  }

  free(jpeg_data);

  printf("PASSED\n");
}

static void test_decode_batch_grayscale(void) {
  printf("Test: decode batch grayscale... ");

  // Need to reinit with grayscale format
  nvjpeg_batched_cleanup();

  bool result = nvjpeg_batched_init(4, 4000, 6000, NVJPEG_FMT_GRAY8);
  if (!result) {
    printf("FAILED (reinit with gray format failed)\n");
    exit(1);
  }

  if (nvjpeg_batched_get_format() != NVJPEG_FMT_GRAY8) {
    printf("FAILED (wrong format after reinit)\n");
    exit(1);
  }

  size_t jpeg_size = 0;
  uint8_t *jpeg_data = read_jpeg_file(test_jpeg_path, &jpeg_size);
  if (jpeg_data == NULL) {
    printf("SKIPPED (test JPEG not found)\n");
    return;
  }

  const uint8_t *data_ptrs[2] = {jpeg_data, jpeg_data};
  size_t sizes[2] = {jpeg_size, jpeg_size};
  NvJpegDecodedImage outputs[2] = {0};

  int decoded = nvjpeg_decode_batch(data_ptrs, sizes, 2, outputs);

  if (decoded != 2) {
    printf("FAILED (decoded %d images, expected 2)\n", decoded);
    free(jpeg_data);
    exit(1);
  }

  for (int i = 0; i < 2; i++) {
    if (outputs[i].channels != 1) {
      printf("FAILED (wrong channels for grayscale: %d)\n",
             outputs[i].channels);
      free(jpeg_data);
      exit(1);
    }
  }

  free(jpeg_data);

  // Reinit back to RGB for remaining tests
  nvjpeg_batched_cleanup();
  nvjpeg_batched_init(8, 4000, 6000, NVJPEG_FMT_RGB);

  printf("PASSED\n");
}

static void test_batch_stats(void) {
  printf("Test: batch statistics... ");

  NvJpegBatchStats stats = nvjpeg_batched_get_stats();

  // We've done several batch calls by now
  if (stats.total_batch_calls == 0) {
    printf("FAILED (no batch calls recorded)\n");
    exit(1);
  }

  if (stats.total_images_decoded == 0) {
    printf("FAILED (no images decoded recorded)\n");
    exit(1);
  }

  printf("PASSED (calls=%zu, images=%zu, failed=%zu, max_batch=%zu)\n",
         stats.total_batch_calls, stats.total_images_decoded,
         stats.failed_decodes, stats.max_batch_size_used);
}

static void test_output_pixel_verification(void) {
  printf("Test: output pixel verification... ");

  size_t jpeg_size = 0;
  uint8_t *jpeg_data = read_jpeg_file(test_jpeg_path, &jpeg_size);
  if (jpeg_data == NULL) {
    printf("SKIPPED (test JPEG not found)\n");
    return;
  }

  // Decode using per-image API
  NvJpegStreamState *state = nvjpeg_acquire_stream_state();
  if (state == NULL) {
    printf("SKIPPED (no stream state available)\n");
    free(jpeg_data);
    return;
  }

  NvJpegDecodedImage single_out = {0};
  bool single_ok = nvjpeg_decode_to_gpu(jpeg_data, jpeg_size, state, NULL,
                                        NVJPEG_FMT_RGB, &single_out);
  nvjpeg_release_stream_state(state);

  if (!single_ok) {
    printf("SKIPPED (single decode failed)\n");
    free(jpeg_data);
    return;
  }

  // Decode using batch API
  const uint8_t *data_ptrs[1] = {jpeg_data};
  size_t sizes[1] = {jpeg_size};
  NvJpegDecodedImage batch_out[1] = {0};

  int decoded = nvjpeg_decode_batch(data_ptrs, sizes, 1, batch_out);

  if (decoded != 1) {
    printf("FAILED (batch decode failed)\n");
    cudaFree(single_out.gpu_ptr);
    free(jpeg_data);
    exit(1);
  }

  // Compare dimensions
  if (single_out.width != batch_out[0].width ||
      single_out.height != batch_out[0].height) {
    printf("FAILED (dimension mismatch: single=%dx%d, batch=%dx%d)\n",
           single_out.width, single_out.height, batch_out[0].width,
           batch_out[0].height);
    cudaFree(single_out.gpu_ptr);
    free(jpeg_data);
    exit(1);
  }

  // Copy both to host and compare first row of pixels
  size_t row_size = (size_t)single_out.width * 3; // RGB
  uint8_t *single_row = malloc(row_size);
  uint8_t *batch_row = malloc(row_size);

  if (single_row != NULL && batch_row != NULL) {
    cudaMemcpy(single_row, single_out.gpu_ptr, row_size,
               cudaMemcpyDeviceToHost);
    cudaMemcpy(batch_row, batch_out[0].gpu_ptr, row_size,
               cudaMemcpyDeviceToHost);

    // Compare pixels (allow small tolerance for different decode paths)
    int max_diff = 0;
    for (size_t i = 0; i < row_size; i++) {
      int diff = abs((int)single_row[i] - (int)batch_row[i]);
      if (diff > max_diff) {
        max_diff = diff;
      }
    }

    free(single_row);
    free(batch_row);

    // nvJPEG should produce identical output for single and batched
    if (max_diff > 1) {
      printf("FAILED (pixel difference too large: %d)\n", max_diff);
      cudaFree(single_out.gpu_ptr);
      free(jpeg_data);
      exit(1);
    }
  }

  cudaFree(single_out.gpu_ptr);
  free(jpeg_data);

  printf("PASSED (outputs match)\n");
}

static void test_batched_cleanup(void) {
  printf("Test: nvjpeg_batched_cleanup... ");

  nvjpeg_batched_cleanup();

  if (nvjpeg_batched_is_ready()) {
    printf("FAILED (still ready after cleanup)\n");
    exit(1);
  }

  if (nvjpeg_batched_get_max_batch_size() != 0) {
    printf("FAILED (max_batch_size not 0 after cleanup)\n");
    exit(1);
  }

  // Double cleanup should be safe
  nvjpeg_batched_cleanup();

  printf("PASSED\n");
}

static void test_context_cleanup(void) {
  printf("Test: context cleanup... ");

  nvjpeg_context_cleanup();

  if (nvjpeg_is_available()) {
    printf("FAILED (still available after cleanup)\n");
    exit(1);
  }

  printf("PASSED\n");
}

int main(void) {
  printf("nvJPEG Batched Decode Unit Tests (PR36A)\n");
  printf("========================================\n\n");

  // Initialize test paths from environment
  init_test_paths();

  // Check CUDA availability
  UnpaperCudaInitStatus st = unpaper_cuda_try_init();
  if (st != UNPAPER_CUDA_INIT_OK) {
    printf("Skipping tests: %s\n", unpaper_cuda_init_status_string(st));
    return 77; // Skip return code
  }

  // Run tests
  test_batched_init();
  test_batched_reinit();
  test_decode_batch_single();
  test_decode_batch_multiple();
  test_decode_batch_with_nulls();
  test_batch_stats(); // Run before grayscale test which resets stats
  test_decode_batch_grayscale();
  test_output_pixel_verification();
  test_batched_cleanup();
  test_context_cleanup();

  printf("\nAll tests passed!\n");
  return 0;
}
