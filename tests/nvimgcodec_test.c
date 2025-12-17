// SPDX-FileCopyrightText: 2025 The unpaper authors
//
// SPDX-License-Identifier: GPL-2.0-only

#include <stdbool.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#ifdef UNPAPER_WITH_CUDA
#include "imageprocess/cuda_runtime.h"
#include "imageprocess/nvimgcodec.h"
#include <cuda_runtime.h>
#endif

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

// ============================================================================
// Format Detection Tests
// ============================================================================

static void test_format_detection_jpeg(void) {
  printf("Test: format detection (JPEG)... ");

#ifdef UNPAPER_WITH_CUDA
  // JPEG magic bytes: FFD8FF
  uint8_t jpeg_header[] = {0xFF, 0xD8, 0xFF, 0xE0, 0x00, 0x10, 0x4A, 0x46};
  NvImgCodecFormat fmt =
      nvimgcodec_detect_format(jpeg_header, sizeof(jpeg_header));

  if (fmt != NVIMGCODEC_FORMAT_JPEG) {
    printf("FAILED (expected JPEG, got %d)\n", fmt);
    exit(1);
  }

  printf("PASSED\n");
#else
  printf("SKIPPED (CUDA not available)\n");
#endif
}

static void test_format_detection_jp2(void) {
  printf("Test: format detection (JP2 codestream)... ");

#ifdef UNPAPER_WITH_CUDA
  // JP2 codestream magic bytes: FF4FFF51
  uint8_t jp2_header[] = {0xFF, 0x4F, 0xFF, 0x51, 0x00, 0x00, 0x00, 0x00};
  NvImgCodecFormat fmt =
      nvimgcodec_detect_format(jp2_header, sizeof(jp2_header));

  if (fmt != NVIMGCODEC_FORMAT_JPEG2000) {
    printf("FAILED (expected JP2, got %d)\n", fmt);
    exit(1);
  }

  printf("PASSED\n");
#else
  printf("SKIPPED (CUDA not available)\n");
#endif
}

static void test_format_detection_jp2_file(void) {
  printf("Test: format detection (JP2 file format)... ");

#ifdef UNPAPER_WITH_CUDA
  // JP2 file format signature
  uint8_t jp2_sig[] = {0x00, 0x00, 0x00, 0x0C, 0x6A, 0x50,
                       0x20, 0x20, 0x0D, 0x0A, 0x87, 0x0A};
  NvImgCodecFormat fmt = nvimgcodec_detect_format(jp2_sig, sizeof(jp2_sig));

  if (fmt != NVIMGCODEC_FORMAT_JPEG2000) {
    printf("FAILED (expected JP2, got %d)\n", fmt);
    exit(1);
  }

  printf("PASSED\n");
#else
  printf("SKIPPED (CUDA not available)\n");
#endif
}

static void test_format_detection_unknown(void) {
  printf("Test: format detection (unknown)... ");

#ifdef UNPAPER_WITH_CUDA
  // Random bytes
  uint8_t unknown[] = {0x12, 0x34, 0x56, 0x78, 0x9A, 0xBC};
  NvImgCodecFormat fmt = nvimgcodec_detect_format(unknown, sizeof(unknown));

  if (fmt != NVIMGCODEC_FORMAT_UNKNOWN) {
    printf("FAILED (expected UNKNOWN, got %d)\n", fmt);
    exit(1);
  }

  printf("PASSED\n");
#else
  printf("SKIPPED (CUDA not available)\n");
#endif
}

static void test_format_detection_null(void) {
  printf("Test: format detection (NULL input)... ");

#ifdef UNPAPER_WITH_CUDA
  NvImgCodecFormat fmt = nvimgcodec_detect_format(NULL, 0);

  if (fmt != NVIMGCODEC_FORMAT_UNKNOWN) {
    printf("FAILED (expected UNKNOWN for NULL, got %d)\n", fmt);
    exit(1);
  }

  printf("PASSED\n");
#else
  printf("SKIPPED (CUDA not available)\n");
#endif
}

// ============================================================================
// Initialization Tests
// ============================================================================

static void test_init(void) {
  printf("Test: nvimgcodec_init... ");

#ifdef UNPAPER_WITH_CUDA
  // Initialize with 4 stream states
  bool result = nvimgcodec_init(4);
  if (!result) {
    printf("FAILED (init returned false)\n");
    exit(1);
  }

  // Check availability
  if (!nvimgcodec_any_available()) {
    printf("FAILED (not available after init)\n");
    nvimgcodec_cleanup();
    exit(1);
  }

  printf("PASSED (nvimgcodec: %s, jp2: %s)\n",
         nvimgcodec_is_available() ? "yes" : "no",
         nvimgcodec_jp2_supported() ? "yes" : "no");
#else
  printf("SKIPPED (CUDA not available)\n");
#endif
}

static void test_decode_state_acquire_release(void) {
  printf("Test: decode state acquire/release... ");

#ifdef UNPAPER_WITH_CUDA
  if (!nvimgcodec_any_available()) {
    printf("SKIPPED (not initialized)\n");
    return;
  }

  // Acquire a state
  NvImgCodecDecodeState *state = nvimgcodec_acquire_decode_state();
  if (state == NULL) {
    printf("FAILED (acquire returned NULL)\n");
    exit(1);
  }

  // Release it
  nvimgcodec_release_decode_state(state);

  printf("PASSED\n");
#else
  printf("SKIPPED (CUDA not available)\n");
#endif
}

static void test_encode_state_acquire_release(void) {
  printf("Test: encode state acquire/release... ");

#ifdef UNPAPER_WITH_CUDA
  if (!nvimgcodec_any_available()) {
    printf("SKIPPED (not initialized)\n");
    return;
  }

  // Acquire a state
  NvImgCodecEncodeState *state = nvimgcodec_acquire_encode_state();
  if (state == NULL) {
    printf("FAILED (acquire returned NULL)\n");
    exit(1);
  }

  // Release it
  nvimgcodec_release_encode_state(state);

  printf("PASSED\n");
#else
  printf("SKIPPED (CUDA not available)\n");
#endif
}

// ============================================================================
// Decode Tests
// ============================================================================

static void test_decode_jpeg(void) {
  printf("Test: decode JPEG to GPU... ");

#ifdef UNPAPER_WITH_CUDA
  if (!nvimgcodec_any_available()) {
    printf("SKIPPED (not initialized)\n");
    return;
  }

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

  // Verify format detection
  NvImgCodecFormat fmt = nvimgcodec_detect_format(jpeg_data, (size_t)file_size);
  if (fmt != NVIMGCODEC_FORMAT_JPEG) {
    free(jpeg_data);
    printf("FAILED (format detection failed)\n");
    exit(1);
  }

  // Acquire decode state
  NvImgCodecDecodeState *state = nvimgcodec_acquire_decode_state();
  if (state == NULL) {
    free(jpeg_data);
    printf("FAILED (no decode state)\n");
    exit(1);
  }

  // Decode to GPU
  NvImgCodecDecodedImage out = {0};
  bool result = nvimgcodec_decode(jpeg_data, (size_t)file_size, state, NULL,
                                  NVIMGCODEC_OUT_GRAY8, &out);

  free(jpeg_data);
  nvimgcodec_release_decode_state(state);

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

  if (out.source_fmt != NVIMGCODEC_FORMAT_JPEG) {
    printf("FAILED (wrong source format: %d)\n", out.source_fmt);
    cudaFree(out.gpu_ptr);
    exit(1);
  }

  // Free GPU memory
  cudaFree(out.gpu_ptr);

  printf("PASSED (%dx%d, %d channels)\n", out.width, out.height, out.channels);
#else
  printf("SKIPPED (CUDA not available)\n");
#endif
}

static void test_decode_file(void) {
  printf("Test: nvimgcodec_decode_file... ");

#ifdef UNPAPER_WITH_CUDA
  if (!nvimgcodec_any_available()) {
    printf("SKIPPED (not initialized)\n");
    return;
  }

  NvImgCodecDecodedImage out = {0};
  bool result =
      nvimgcodec_decode_file(test_jpeg_path, NULL, NVIMGCODEC_OUT_RGB, &out);

  if (!result) {
    printf("SKIPPED (decode failed, file may not exist)\n");
    return;
  }

  if (out.gpu_ptr == NULL) {
    printf("FAILED (gpu_ptr is NULL)\n");
    exit(1);
  }

  if (out.channels != 3) {
    printf("FAILED (wrong channels for RGB: %d)\n", out.channels);
    cudaFree(out.gpu_ptr);
    exit(1);
  }

  cudaFree(out.gpu_ptr);

  printf("PASSED\n");
#else
  printf("SKIPPED (CUDA not available)\n");
#endif
}

// ============================================================================
// Batch Decode Tests
// ============================================================================

static void test_decode_batch(void) {
  printf("Test: nvimgcodec_decode_batch... ");

#ifdef UNPAPER_WITH_CUDA
  if (!nvimgcodec_any_available()) {
    printf("SKIPPED (not initialized)\n");
    return;
  }

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

  // Create a batch of 3 copies of the same JPEG
  const int batch_size = 3;
  const uint8_t *data_ptrs[3] = {jpeg_data, jpeg_data, jpeg_data};
  size_t sizes[3] = {(size_t)file_size, (size_t)file_size, (size_t)file_size};
  NvImgCodecDecodedImage outputs[3] = {0};

  // Batch decode
  int decoded = nvimgcodec_decode_batch(data_ptrs, sizes, batch_size,
                                        NVIMGCODEC_OUT_GRAY8, outputs);

  free(jpeg_data);

  if (decoded != batch_size) {
    printf("FAILED (decoded %d of %d)\n", decoded, batch_size);
    // Clean up any successful decodes
    for (int i = 0; i < batch_size; i++) {
      if (outputs[i].gpu_ptr != NULL) {
        cudaFree(outputs[i].gpu_ptr);
      }
    }
    exit(1);
  }

  // Verify all outputs
  for (int i = 0; i < batch_size; i++) {
    if (outputs[i].gpu_ptr == NULL) {
      printf("FAILED (output[%d].gpu_ptr is NULL)\n", i);
      for (int j = 0; j < batch_size; j++) {
        if (outputs[j].gpu_ptr != NULL) {
          cudaFree(outputs[j].gpu_ptr);
        }
      }
      exit(1);
    }

    if (outputs[i].width <= 0 || outputs[i].height <= 0) {
      printf("FAILED (output[%d] has invalid dimensions)\n", i);
      for (int j = 0; j < batch_size; j++) {
        if (outputs[j].gpu_ptr != NULL) {
          cudaFree(outputs[j].gpu_ptr);
        }
      }
      exit(1);
    }
  }

  // Clean up GPU memory
  for (int i = 0; i < batch_size; i++) {
    cudaFree(outputs[i].gpu_ptr);
  }

  printf("PASSED (%d images decoded)\n", decoded);
#else
  printf("SKIPPED (CUDA not available)\n");
#endif
}

static void test_decode_batch_null_safety(void) {
  printf("Test: nvimgcodec_decode_batch (null safety)... ");

#ifdef UNPAPER_WITH_CUDA
  if (!nvimgcodec_any_available()) {
    printf("SKIPPED (not initialized)\n");
    return;
  }

  // Test with NULL pointers - should return 0 without crashing
  int result =
      nvimgcodec_decode_batch(NULL, NULL, 0, NVIMGCODEC_OUT_GRAY8, NULL);

  if (result != 0) {
    printf("FAILED (expected 0 for NULL input, got %d)\n", result);
    exit(1);
  }

  printf("PASSED\n");
#else
  printf("SKIPPED (CUDA not available)\n");
#endif
}

static void test_decode_batch_partial(void) {
  printf("Test: nvimgcodec_decode_batch (partial batch)... ");

#ifdef UNPAPER_WITH_CUDA
  if (!nvimgcodec_any_available()) {
    printf("SKIPPED (not initialized)\n");
    return;
  }

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

  // Create a batch with one valid and one NULL entry
  const int batch_size = 3;
  const uint8_t *data_ptrs[3] = {jpeg_data, NULL, jpeg_data};
  size_t sizes[3] = {(size_t)file_size, 0, (size_t)file_size};
  NvImgCodecDecodedImage outputs[3] = {0};

  // Batch decode - should skip the NULL entry
  int decoded = nvimgcodec_decode_batch(data_ptrs, sizes, batch_size,
                                        NVIMGCODEC_OUT_RGB, outputs);

  free(jpeg_data);

  // Should have decoded 2 (first and third)
  if (decoded != 2) {
    printf("FAILED (expected 2 decoded, got %d)\n", decoded);
    for (int i = 0; i < batch_size; i++) {
      if (outputs[i].gpu_ptr != NULL) {
        cudaFree(outputs[i].gpu_ptr);
      }
    }
    exit(1);
  }

  // Verify correct entries were decoded
  if (outputs[0].gpu_ptr == NULL) {
    printf("FAILED (output[0] should be valid)\n");
    for (int i = 0; i < batch_size; i++) {
      if (outputs[i].gpu_ptr != NULL) {
        cudaFree(outputs[i].gpu_ptr);
      }
    }
    exit(1);
  }

  if (outputs[1].gpu_ptr != NULL) {
    printf("FAILED (output[1] should be NULL)\n");
    for (int i = 0; i < batch_size; i++) {
      if (outputs[i].gpu_ptr != NULL) {
        cudaFree(outputs[i].gpu_ptr);
      }
    }
    exit(1);
  }

  if (outputs[2].gpu_ptr == NULL) {
    printf("FAILED (output[2] should be valid)\n");
    for (int i = 0; i < batch_size; i++) {
      if (outputs[i].gpu_ptr != NULL) {
        cudaFree(outputs[i].gpu_ptr);
      }
    }
    exit(1);
  }

  // Clean up GPU memory
  cudaFree(outputs[0].gpu_ptr);
  cudaFree(outputs[2].gpu_ptr);

  printf("PASSED\n");
#else
  printf("SKIPPED (CUDA not available)\n");
#endif
}

// ============================================================================
// Encode Tests
// ============================================================================

static void test_encode_jpeg(void) {
  printf("Test: encode JPEG from GPU... ");

#ifdef UNPAPER_WITH_CUDA
  if (!nvimgcodec_any_available()) {
    printf("SKIPPED (not initialized)\n");
    return;
  }

  // Create a test GPU buffer (100x100 grayscale)
  int width = 100;
  int height = 100;
  int channels = 1;
  size_t pitch = ((size_t)width + 255) & ~(size_t)255; // Align to 256
  size_t buffer_size = pitch * (size_t)height;

  void *gpu_ptr = NULL;
  cudaError_t cuda_err = cudaMalloc(&gpu_ptr, buffer_size);
  if (cuda_err != cudaSuccess) {
    printf("FAILED (cudaMalloc failed)\n");
    exit(1);
  }

  // Initialize with a simple pattern
  uint8_t *host_data = malloc(buffer_size);
  if (host_data == NULL) {
    cudaFree(gpu_ptr);
    printf("FAILED (malloc failed)\n");
    exit(1);
  }

  for (int y = 0; y < height; y++) {
    for (int x = 0; x < width; x++) {
      host_data[y * pitch + x] = (uint8_t)((x + y) % 256);
    }
  }

  cudaMemcpy(gpu_ptr, host_data, buffer_size, cudaMemcpyHostToDevice);
  free(host_data);

  // Acquire encode state
  NvImgCodecEncodeState *state = nvimgcodec_acquire_encode_state();
  if (state == NULL) {
    cudaFree(gpu_ptr);
    printf("FAILED (no encode state)\n");
    exit(1);
  }

  // Encode to JPEG
  NvImgCodecEncodedImage out = {0};
  bool result =
      nvimgcodec_encode_jpeg(gpu_ptr, pitch, width, height,
                             NVIMGCODEC_ENC_FMT_GRAY8, 85, state, NULL, &out);

  cudaFree(gpu_ptr);
  nvimgcodec_release_encode_state(state);

  if (!result) {
    printf("FAILED (encode returned false)\n");
    exit(1);
  }

  if (out.data == NULL) {
    printf("FAILED (output data is NULL)\n");
    exit(1);
  }

  if (out.size == 0) {
    printf("FAILED (output size is 0)\n");
    free(out.data);
    exit(1);
  }

  // Verify JPEG header
  if (out.size < 3 || out.data[0] != 0xFF || out.data[1] != 0xD8) {
    printf("FAILED (invalid JPEG header)\n");
    free(out.data);
    exit(1);
  }

  printf("PASSED (size: %zu bytes)\n", out.size);
  free(out.data);
#else
  printf("SKIPPED (CUDA not available)\n");
#endif
}

static void test_jp2_support_status(void) {
  printf("Test: JP2 support status... ");

#ifdef UNPAPER_WITH_CUDA
  if (!nvimgcodec_any_available()) {
    printf("SKIPPED (not initialized)\n");
    return;
  }

  bool jp2_supported = nvimgcodec_jp2_supported();
  bool is_full_nvimgcodec = nvimgcodec_is_available();

  // JP2 should only be supported with full nvImageCodec
  if (jp2_supported && !is_full_nvimgcodec) {
    printf("FAILED (JP2 claimed supported without nvImageCodec)\n");
    exit(1);
  }

  printf("PASSED (JP2: %s)\n", jp2_supported ? "supported" : "not supported");
#else
  printf("SKIPPED (CUDA not available)\n");
#endif
}

// ============================================================================
// Statistics Test
// ============================================================================

static void test_statistics(void) {
  printf("Test: statistics... ");

#ifdef UNPAPER_WITH_CUDA
  if (!nvimgcodec_any_available()) {
    printf("SKIPPED (not initialized)\n");
    return;
  }

  NvImgCodecStats stats = nvimgcodec_get_stats();

  // After our tests, we should have some decodes
  if (stats.total_decodes == 0) {
    printf("FAILED (no decodes recorded)\n");
    exit(1);
  }

  printf("PASSED (decodes: %zu, encodes: %zu)\n", stats.total_decodes,
         stats.total_encodes);
#else
  printf("SKIPPED (CUDA not available)\n");
#endif
}

// ============================================================================
// Cleanup Test
// ============================================================================

static void test_cleanup(void) {
  printf("Test: nvimgcodec_cleanup... ");

#ifdef UNPAPER_WITH_CUDA
  nvimgcodec_cleanup();

  printf("PASSED\n");
#else
  printf("SKIPPED (CUDA not available)\n");
#endif
}

// ============================================================================
// Main
// ============================================================================

int main(void) {
  printf("nvImageCodec Wrapper Unit Tests\n");
  printf("================================\n\n");

  init_test_paths();

#ifdef UNPAPER_WITH_CUDA
  // Check CUDA availability
  UnpaperCudaInitStatus st = unpaper_cuda_try_init();
  if (st != UNPAPER_CUDA_INIT_OK) {
    printf("Skipping tests: %s\n", unpaper_cuda_init_status_string(st));
    return 77; // Skip return code
  }
#endif

  // Format detection tests (don't require initialization)
  test_format_detection_jpeg();
  test_format_detection_jp2();
  test_format_detection_jp2_file();
  test_format_detection_unknown();
  test_format_detection_null();

  // Initialization test
  test_init();

  // State management tests
  test_decode_state_acquire_release();
  test_encode_state_acquire_release();

  // Decode tests
  test_decode_jpeg();
  test_decode_file();

  // Batch decode tests
  test_decode_batch();
  test_decode_batch_null_safety();
  test_decode_batch_partial();

  // Encode tests
  test_encode_jpeg();

  // JP2 status test
  test_jp2_support_status();

  // Statistics test
  test_statistics();

  // Cleanup test
  test_cleanup();

  printf("\nAll tests passed!\n");
  return 0;
}
