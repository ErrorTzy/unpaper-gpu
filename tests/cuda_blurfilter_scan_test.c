// SPDX-FileCopyrightText: 2025 The unpaper authors
//
// SPDX-License-Identifier: GPL-2.0-only

#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "imageprocess/cuda_runtime.h"
#include "imageprocess/npp_wrapper.h"
#include "imageprocess/npp_integral.h"

// PTX for kernels
extern const char unpaper_cuda_kernels_ptx[];

// Test image dimensions (must be divisible by block size)
// Use width=64 to match NPP integral test that works
#define TEST_WIDTH 64
#define TEST_HEIGHT 48
#define BLOCK_W 8
#define BLOCK_H 8

// Output structure must match kernel definition
typedef struct {
  int x;
  int y;
} BlurfilterBlock;

// Maximum blocks to output
#define MAX_BLOCKS 1024

static void *cuda_module = NULL;
static void *k_blurfilter_scan = NULL;

static void ensure_kernel_loaded(void) {
  if (cuda_module != NULL) {
    return;
  }
  cuda_module = unpaper_cuda_module_load_ptx(unpaper_cuda_kernels_ptx);
  k_blurfilter_scan =
      unpaper_cuda_module_get_function(cuda_module, "unpaper_blurfilter_scan");
}

// Create a dark mask with known isolated blocks
// Set specific pixels to 255 (dark in the mask) and rest to 0
static void create_test_dark_mask(uint8_t *mask, int width, int height) {
  memset(mask, 0, (size_t)width * (size_t)height);

  // With 64x48 image and 8x8 blocks, we have 8x6 blocks
  // Interior blocks: bx in [1,6], by in [1,4]
  //
  // Block layout (64x48 / 8x8 = 8x6 blocks):
  //   bx: 0  1  2  3  4  5  6  7
  //   by: 0  .  .  .  .  .  .  .
  //       1  .  .  .  .  .  .  .
  //       2  .  .  .  .  .  .  .
  //       3  .  .  .  .  .  .  .
  //       4  .  .  .  .  .  .  .
  //       5  .  .  .  .  .  .  .

  // Create an isolated dark block at (16, 16) - block (2, 2)
  // Its diagonal neighbors are (1,1), (3,1), (1,3), (3,3) - all empty
  for (int y = 16; y < 16 + BLOCK_H; y++) {
    for (int x = 16; x < 16 + BLOCK_W; x++) {
      mask[y * width + x] = 255;  // Dark pixel
    }
  }

  // Create a non-isolated dark block at (40, 24) - block (5, 3)
  // Its diagonal neighbors are (4,2), (6,2), (4,4), (6,4)
  // We add dark content to (4, 2) to make it non-isolated
  for (int y = 24; y < 24 + BLOCK_H; y++) {
    for (int x = 40; x < 40 + BLOCK_W; x++) {
      mask[y * width + x] = 255;  // Dark pixel
    }
  }
  // Add dark content to upper-left diagonal (block (4, 2)) to make (5,3) non-isolated
  for (int y = 16; y < 16 + BLOCK_H; y++) {
    for (int x = 32; x < 32 + BLOCK_W; x++) {
      mask[y * width + x] = 255;  // Dark pixel - makes block (5,3) non-isolated
    }
  }
}

// CPU reference: compute which blocks should be isolated
// Returns number of isolated blocks found
static int compute_expected_isolated_blocks(const uint8_t *mask, int width,
                                             int height, int block_w,
                                             int block_h, float intensity,
                                             BlurfilterBlock *out_blocks,
                                             int max_blocks) {
  int blocks_per_row = width / block_w;
  int blocks_per_col = height / block_h;
  int64_t total_pixels = (int64_t)block_w * (int64_t)block_h;
  int count = 0;

  // Compute dark counts for all blocks
  int *dark_counts = (int *)calloc((size_t)blocks_per_row * (size_t)blocks_per_col,
                                    sizeof(int));
  if (dark_counts == NULL) {
    return 0;
  }

  for (int by = 0; by < blocks_per_col; by++) {
    for (int bx = 0; bx < blocks_per_row; bx++) {
      int dark_count = 0;
      int x0 = bx * block_w;
      int y0 = by * block_h;
      for (int y = y0; y < y0 + block_h; y++) {
        for (int x = x0; x < x0 + block_w; x++) {
          if (mask[y * width + x] >= 128) {  // Dark threshold
            dark_count++;
          }
        }
      }
      dark_counts[by * blocks_per_row + bx] = dark_count;
    }
  }

  // Find isolated blocks
  for (int by = 0; by < blocks_per_col; by++) {
    for (int bx = 0; bx < blocks_per_row; bx++) {
      int self_count = dark_counts[by * blocks_per_row + bx];
      if (self_count == 0) {
        continue;  // No dark pixels, not a candidate
      }

      // Check 4 diagonal neighbors
      // Missing boundary neighbors are treated as having max density (100%)
      int64_t max_neighbor = 0;

      // Upper-left
      if (bx > 0 && by > 0) {
        int n = dark_counts[(by - 1) * blocks_per_row + (bx - 1)];
        if (n > max_neighbor) max_neighbor = n;
      } else {
        max_neighbor = total_pixels;
      }
      // Upper-right
      if (bx < blocks_per_row - 1 && by > 0) {
        int n = dark_counts[(by - 1) * blocks_per_row + (bx + 1)];
        if (n > max_neighbor) max_neighbor = n;
      } else {
        max_neighbor = total_pixels;
      }
      // Lower-left
      if (bx > 0 && by < blocks_per_col - 1) {
        int n = dark_counts[(by + 1) * blocks_per_row + (bx - 1)];
        if (n > max_neighbor) max_neighbor = n;
      } else {
        max_neighbor = total_pixels;
      }
      // Lower-right
      if (bx < blocks_per_row - 1 && by < blocks_per_col - 1) {
        int n = dark_counts[(by + 1) * blocks_per_row + (bx + 1)];
        if (n > max_neighbor) max_neighbor = n;
      } else {
        max_neighbor = total_pixels;
      }

      float ratio = (float)max_neighbor / (float)total_pixels;
      if (ratio <= intensity) {
        if (count < max_blocks) {
          out_blocks[count].x = bx * block_w;
          out_blocks[count].y = by * block_h;
        }
        count++;
      }
    }
  }

  free(dark_counts);
  return count;
}

static void test_blurfilter_scan_basic(void) {
  printf("test_blurfilter_scan_basic: ");

  ensure_kernel_loaded();

  // Create test dark mask
  uint8_t *mask_host = (uint8_t *)malloc(TEST_WIDTH * TEST_HEIGHT);
  if (mask_host == NULL) {
    fprintf(stderr, "malloc failed\n");
    exit(1);
  }
  create_test_dark_mask(mask_host, TEST_WIDTH, TEST_HEIGHT);

  // Upload mask to GPU
  uint64_t mask_device = unpaper_cuda_malloc(TEST_WIDTH * TEST_HEIGHT);
  if (mask_device == 0) {
    fprintf(stderr, "GPU malloc failed\n");
    exit(1);
  }
  unpaper_cuda_memcpy_h2d(mask_device, mask_host, TEST_WIDTH * TEST_HEIGHT);

  // Initialize NPP
  if (!unpaper_npp_init()) {
    fprintf(stderr, "NPP init failed\n");
    exit(1);
  }

  // Compute GPU integral
  UnpaperNppIntegral integral;
  if (!unpaper_npp_integral_8u32s(mask_device, TEST_WIDTH, TEST_HEIGHT,
                                   TEST_WIDTH, NULL, &integral)) {
    fprintf(stderr, "NPP integral failed\n");
    exit(1);
  }

  // Allocate output buffers on GPU
  uint64_t out_blocks_device = unpaper_cuda_malloc(MAX_BLOCKS * sizeof(BlurfilterBlock));
  uint64_t out_count_device = unpaper_cuda_malloc(sizeof(int));
  if (out_blocks_device == 0 || out_count_device == 0) {
    fprintf(stderr, "GPU malloc for output failed\n");
    exit(1);
  }

  // Initialize count to 0
  int zero = 0;
  unpaper_cuda_memcpy_h2d(out_count_device, &zero, sizeof(int));

  // Run kernel
  float intensity = 0.05f;  // 5% threshold
  int integral_step = (int)integral.step_bytes;
  int img_w = TEST_WIDTH;
  int img_h = TEST_HEIGHT;
  int block_w = BLOCK_W;
  int block_h = BLOCK_H;
  int max_blocks = MAX_BLOCKS;

  void *args[] = {
      &integral.device_ptr,
      &integral_step,
      &img_w,
      &img_h,
      &block_w,
      &block_h,
      &intensity,
      &out_blocks_device,
      &out_count_device,
      &max_blocks,
  };

  // Calculate grid/block dimensions
  int blocks_per_row = TEST_WIDTH / BLOCK_W;
  int blocks_per_col = TEST_HEIGHT / BLOCK_H;
  int threads_per_block = 16;
  int grid_x = (blocks_per_row + threads_per_block - 1) / threads_per_block;
  int grid_y = (blocks_per_col + threads_per_block - 1) / threads_per_block;

  unpaper_cuda_launch_kernel(k_blurfilter_scan, grid_x, grid_y, 1,
                             threads_per_block, threads_per_block, 1, args);
  unpaper_cuda_stream_synchronize();

  // Download results
  int gpu_count = 0;
  unpaper_cuda_memcpy_d2h(&gpu_count, out_count_device, sizeof(int));

  BlurfilterBlock *gpu_blocks = (BlurfilterBlock *)malloc(gpu_count * sizeof(BlurfilterBlock));
  if (gpu_blocks == NULL && gpu_count > 0) {
    fprintf(stderr, "malloc failed\n");
    exit(1);
  }
  if (gpu_count > 0) {
    unpaper_cuda_memcpy_d2h(gpu_blocks, out_blocks_device,
                            gpu_count * sizeof(BlurfilterBlock));
  }

  // Compute expected results
  BlurfilterBlock *expected_blocks = (BlurfilterBlock *)malloc(MAX_BLOCKS * sizeof(BlurfilterBlock));
  if (expected_blocks == NULL) {
    fprintf(stderr, "malloc failed\n");
    exit(1);
  }
  int expected_count = compute_expected_isolated_blocks(
      mask_host, TEST_WIDTH, TEST_HEIGHT, BLOCK_W, BLOCK_H, intensity,
      expected_blocks, MAX_BLOCKS);

  // Verify count matches
  if (gpu_count != expected_count) {
    fprintf(stderr, "FAILED: count mismatch GPU=%d expected=%d\n", gpu_count,
            expected_count);

    printf("GPU blocks found:\n");
    for (int i = 0; i < gpu_count && i < 10; i++) {
      printf("  (%d, %d)\n", gpu_blocks[i].x, gpu_blocks[i].y);
    }
    printf("Expected blocks:\n");
    for (int i = 0; i < expected_count && i < 10; i++) {
      printf("  (%d, %d)\n", expected_blocks[i].x, expected_blocks[i].y);
    }
    exit(1);
  }

  // Verify all expected blocks are found (order may differ due to parallel execution)
  for (int i = 0; i < expected_count; i++) {
    int found = 0;
    for (int j = 0; j < gpu_count; j++) {
      if (gpu_blocks[j].x == expected_blocks[i].x &&
          gpu_blocks[j].y == expected_blocks[i].y) {
        found = 1;
        break;
      }
    }
    if (!found) {
      fprintf(stderr, "FAILED: expected block (%d, %d) not found in GPU output\n",
              expected_blocks[i].x, expected_blocks[i].y);
      exit(1);
    }
  }

  // Clean up
  free(gpu_blocks);
  free(expected_blocks);
  unpaper_npp_integral_free(integral.device_ptr);
  unpaper_cuda_free(out_blocks_device);
  unpaper_cuda_free(out_count_device);
  unpaper_cuda_free(mask_device);
  free(mask_host);

  printf("OK (found %d isolated blocks)\n", gpu_count);
}

static void test_blurfilter_scan_empty(void) {
  printf("test_blurfilter_scan_empty: ");

  ensure_kernel_loaded();

  // Create completely white mask (no dark pixels)
  uint8_t *mask_host = (uint8_t *)calloc(TEST_WIDTH * TEST_HEIGHT, 1);
  if (mask_host == NULL) {
    fprintf(stderr, "malloc failed\n");
    exit(1);
  }

  // Upload mask to GPU
  uint64_t mask_device = unpaper_cuda_malloc(TEST_WIDTH * TEST_HEIGHT);
  unpaper_cuda_memcpy_h2d(mask_device, mask_host, TEST_WIDTH * TEST_HEIGHT);

  // Compute GPU integral
  UnpaperNppIntegral integral;
  if (!unpaper_npp_integral_8u32s(mask_device, TEST_WIDTH, TEST_HEIGHT,
                                   TEST_WIDTH, NULL, &integral)) {
    fprintf(stderr, "NPP integral failed\n");
    exit(1);
  }

  // Allocate output buffers
  uint64_t out_blocks_device = unpaper_cuda_malloc(MAX_BLOCKS * sizeof(BlurfilterBlock));
  uint64_t out_count_device = unpaper_cuda_malloc(sizeof(int));
  int zero = 0;
  unpaper_cuda_memcpy_h2d(out_count_device, &zero, sizeof(int));

  // Run kernel
  float intensity = 0.05f;
  int integral_step = (int)integral.step_bytes;
  int img_w = TEST_WIDTH;
  int img_h = TEST_HEIGHT;
  int block_w = BLOCK_W;
  int block_h = BLOCK_H;
  int max_blocks = MAX_BLOCKS;

  void *args[] = {
      &integral.device_ptr, &integral_step, &img_w, &img_h,
      &block_w, &block_h, &intensity, &out_blocks_device,
      &out_count_device, &max_blocks,
  };

  int blocks_per_row = TEST_WIDTH / BLOCK_W;
  int blocks_per_col = TEST_HEIGHT / BLOCK_H;
  int threads = 16;
  unpaper_cuda_launch_kernel(k_blurfilter_scan,
                             (blocks_per_row + threads - 1) / threads,
                             (blocks_per_col + threads - 1) / threads, 1,
                             threads, threads, 1, args);
  unpaper_cuda_stream_synchronize();

  // Download count
  int gpu_count = 0;
  unpaper_cuda_memcpy_d2h(&gpu_count, out_count_device, sizeof(int));

  if (gpu_count != 0) {
    fprintf(stderr, "FAILED: expected 0 blocks, got %d\n", gpu_count);
    exit(1);
  }

  // Clean up
  unpaper_npp_integral_free(integral.device_ptr);
  unpaper_cuda_free(out_blocks_device);
  unpaper_cuda_free(out_count_device);
  unpaper_cuda_free(mask_device);
  free(mask_host);

  printf("OK\n");
}

static void test_blurfilter_scan_all_dark(void) {
  printf("test_blurfilter_scan_all_dark: ");

  ensure_kernel_loaded();

  // Create mask with all dark pixels (no isolated blocks expected)
  uint8_t *mask_host = (uint8_t *)malloc(TEST_WIDTH * TEST_HEIGHT);
  if (mask_host == NULL) {
    fprintf(stderr, "malloc failed\n");
    exit(1);
  }
  memset(mask_host, 255, TEST_WIDTH * TEST_HEIGHT);

  // Upload mask to GPU
  uint64_t mask_device = unpaper_cuda_malloc(TEST_WIDTH * TEST_HEIGHT);
  unpaper_cuda_memcpy_h2d(mask_device, mask_host, TEST_WIDTH * TEST_HEIGHT);

  // Compute GPU integral
  UnpaperNppIntegral integral;
  if (!unpaper_npp_integral_8u32s(mask_device, TEST_WIDTH, TEST_HEIGHT,
                                   TEST_WIDTH, NULL, &integral)) {
    fprintf(stderr, "NPP integral failed\n");
    exit(1);
  }

  // Allocate output buffers
  uint64_t out_blocks_device = unpaper_cuda_malloc(MAX_BLOCKS * sizeof(BlurfilterBlock));
  uint64_t out_count_device = unpaper_cuda_malloc(sizeof(int));
  int zero = 0;
  unpaper_cuda_memcpy_h2d(out_count_device, &zero, sizeof(int));

  // Run kernel
  float intensity = 0.05f;
  int integral_step = (int)integral.step_bytes;
  int img_w = TEST_WIDTH;
  int img_h = TEST_HEIGHT;
  int block_w = BLOCK_W;
  int block_h = BLOCK_H;
  int max_blocks = MAX_BLOCKS;

  void *args[] = {
      &integral.device_ptr, &integral_step, &img_w, &img_h,
      &block_w, &block_h, &intensity, &out_blocks_device,
      &out_count_device, &max_blocks,
  };

  int blocks_per_row = TEST_WIDTH / BLOCK_W;
  int blocks_per_col = TEST_HEIGHT / BLOCK_H;
  int threads = 16;
  unpaper_cuda_launch_kernel(k_blurfilter_scan,
                             (blocks_per_row + threads - 1) / threads,
                             (blocks_per_col + threads - 1) / threads, 1,
                             threads, threads, 1, args);
  unpaper_cuda_stream_synchronize();

  // Download count - with all blocks full, none should be isolated (neighbors all have 100%)
  int gpu_count = 0;
  unpaper_cuda_memcpy_d2h(&gpu_count, out_count_device, sizeof(int));

  if (gpu_count != 0) {
    fprintf(stderr, "FAILED: expected 0 isolated blocks (all dark), got %d\n", gpu_count);
    exit(1);
  }

  // Clean up
  unpaper_npp_integral_free(integral.device_ptr);
  unpaper_cuda_free(out_blocks_device);
  unpaper_cuda_free(out_count_device);
  unpaper_cuda_free(mask_device);
  free(mask_host);

  printf("OK\n");
}

int main(void) {
  // Initialize CUDA
  UnpaperCudaInitStatus st = unpaper_cuda_try_init();
  if (st != UNPAPER_CUDA_INIT_OK) {
    fprintf(stderr, "skipping: %s\n", unpaper_cuda_init_status_string(st));
    return 77;
  }

  // Initialize NPP
  if (!unpaper_npp_init()) {
    fprintf(stderr, "skipping: NPP init failed\n");
    return 77;
  }

  // Run tests
  test_blurfilter_scan_basic();
  test_blurfilter_scan_empty();
  test_blurfilter_scan_all_dark();

  printf("All blurfilter scan tests passed!\n");
  return 0;
}
