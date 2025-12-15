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

// Test image dimensions
#define TEST_WIDTH 64
#define TEST_HEIGHT 48
#define TILE_W 8
#define TILE_H 8
#define STEP_X 8
#define STEP_Y 8

// Thresholds
#define BLACK_THRESHOLD 32   // Pixels <= this are "dark"
#define GRAY_THRESHOLD 10    // inverse_lightness must be < this to wipe

// Output structure must match kernel definition
typedef struct {
  int x;
  int y;
} GrayfilterTile;

// Maximum tiles to output
#define MAX_TILES 1024

static void *cuda_module = NULL;
static void *k_grayfilter_scan = NULL;

static void ensure_kernel_loaded(void) {
  if (cuda_module != NULL) {
    return;
  }
  cuda_module = unpaper_cuda_module_load_ptx(unpaper_cuda_kernels_ptx);
  k_grayfilter_scan =
      unpaper_cuda_module_get_function(cuda_module, "unpaper_grayfilter_scan");
}

// Create a test grayscale image with:
// - Light gray tiles (average ~250, inverse_lightness ~5)
// - Mixed tiles with dark pixels
// - Darker tiles (average ~200, inverse_lightness ~55)
static void create_test_images(uint8_t *gray, uint8_t *dark_mask,
                                int width, int height) {
  // Initialize everything to white (255) and no dark pixels (0 in mask)
  memset(gray, 255, (size_t)width * (size_t)height);
  memset(dark_mask, 0, (size_t)width * (size_t)height);

  // Tile (0,0) at (0,0): Very light gray, no dark pixels -> SHOULD BE WIPED
  // Average lightness = 250, inverse = 5 < gray_threshold(10)
  for (int y = 0; y < TILE_H; y++) {
    for (int x = 0; x < TILE_W; x++) {
      gray[y * width + x] = 250;
    }
  }

  // Tile (1,0) at (8,0): Light gray but has dark pixel -> should NOT be wiped
  for (int y = 0; y < TILE_H; y++) {
    for (int x = TILE_W; x < 2 * TILE_W; x++) {
      gray[y * width + x] = 250;
    }
  }
  // Add a dark pixel
  gray[2 * width + 10] = 20;  // Dark pixel
  dark_mask[2 * width + 10] = 255;  // Mark as dark in mask

  // Tile (2,0) at (16,0): Darker gray, no dark pixels -> should NOT be wiped
  // Average lightness = 200, inverse = 55 > gray_threshold(10)
  for (int y = 0; y < TILE_H; y++) {
    for (int x = 2 * TILE_W; x < 3 * TILE_W; x++) {
      gray[y * width + x] = 200;
    }
  }

  // Tile (0,1) at (0,8): Very light gray, no dark pixels -> SHOULD BE WIPED
  for (int y = TILE_H; y < 2 * TILE_H; y++) {
    for (int x = 0; x < TILE_W; x++) {
      gray[y * width + x] = 252;  // inverse = 3 < threshold
    }
  }

  // Tile (1,1) at (8,8): Medium light gray, no dark pixels -> should NOT be wiped
  // Average lightness = 240, inverse = 15 > gray_threshold(10)
  for (int y = TILE_H; y < 2 * TILE_H; y++) {
    for (int x = TILE_W; x < 2 * TILE_W; x++) {
      gray[y * width + x] = 240;
    }
  }

  // Tile (3,3) at (24,24): Perfect white -> SHOULD BE WIPED
  // Average = 255, inverse = 0 < threshold
  // (already white from memset)
}

// CPU reference: compute which tiles should be wiped
static int compute_expected_tiles(const uint8_t *gray, const uint8_t *dark_mask,
                                   int width, int height, int tile_w, int tile_h,
                                   int step_x, int step_y, uint8_t gray_threshold,
                                   GrayfilterTile *out_tiles, int max_tiles) {
  int tiles_per_row = (width - tile_w) / step_x + 1;
  int tiles_per_col = (height - tile_h) / step_y + 1;
  int64_t tile_pixels = (int64_t)tile_w * (int64_t)tile_h;
  int count = 0;

  for (int ty = 0; ty < tiles_per_col; ty++) {
    for (int tx = 0; tx < tiles_per_row; tx++) {
      int x0 = tx * step_x;
      int y0 = ty * step_y;

      // Count dark pixels
      int dark_count = 0;
      int64_t lightness_sum = 0;
      for (int y = y0; y < y0 + tile_h; y++) {
        for (int x = x0; x < x0 + tile_w; x++) {
          if (dark_mask[y * width + x] >= 128) {
            dark_count++;
          }
          lightness_sum += gray[y * width + x];
        }
      }

      if (dark_count == 0) {
        uint8_t avg_lightness = (uint8_t)(lightness_sum / tile_pixels);
        uint8_t inverse_lightness = 255 - avg_lightness;

        if (inverse_lightness < gray_threshold) {
          if (count < max_tiles) {
            out_tiles[count].x = x0;
            out_tiles[count].y = y0;
          }
          count++;
        }
      }
    }
  }

  return count;
}

static void test_grayfilter_scan_basic(void) {
  printf("test_grayfilter_scan_basic: ");

  ensure_kernel_loaded();

  // Create test images
  uint8_t *gray_host = (uint8_t *)malloc(TEST_WIDTH * TEST_HEIGHT);
  uint8_t *dark_mask_host = (uint8_t *)malloc(TEST_WIDTH * TEST_HEIGHT);
  if (gray_host == NULL || dark_mask_host == NULL) {
    fprintf(stderr, "malloc failed\n");
    exit(1);
  }
  create_test_images(gray_host, dark_mask_host, TEST_WIDTH, TEST_HEIGHT);

  // Upload to GPU
  uint64_t gray_device = unpaper_cuda_malloc(TEST_WIDTH * TEST_HEIGHT);
  uint64_t dark_mask_device = unpaper_cuda_malloc(TEST_WIDTH * TEST_HEIGHT);
  if (gray_device == 0 || dark_mask_device == 0) {
    fprintf(stderr, "GPU malloc failed\n");
    exit(1);
  }
  unpaper_cuda_memcpy_h2d(gray_device, gray_host, TEST_WIDTH * TEST_HEIGHT);
  unpaper_cuda_memcpy_h2d(dark_mask_device, dark_mask_host, TEST_WIDTH * TEST_HEIGHT);

  // Initialize NPP
  if (!unpaper_npp_init()) {
    fprintf(stderr, "NPP init failed\n");
    exit(1);
  }

  // Compute GPU integrals
  UnpaperNppIntegral gray_integral;
  UnpaperNppIntegral dark_integral;
  if (!unpaper_npp_integral_8u32s(gray_device, TEST_WIDTH, TEST_HEIGHT,
                                   TEST_WIDTH, NULL, &gray_integral)) {
    fprintf(stderr, "NPP gray integral failed\n");
    exit(1);
  }
  if (!unpaper_npp_integral_8u32s(dark_mask_device, TEST_WIDTH, TEST_HEIGHT,
                                   TEST_WIDTH, NULL, &dark_integral)) {
    fprintf(stderr, "NPP dark integral failed\n");
    exit(1);
  }

  // Allocate output buffers on GPU
  uint64_t out_tiles_device = unpaper_cuda_malloc(MAX_TILES * sizeof(GrayfilterTile));
  uint64_t out_count_device = unpaper_cuda_malloc(sizeof(int));
  if (out_tiles_device == 0 || out_count_device == 0) {
    fprintf(stderr, "GPU malloc for output failed\n");
    exit(1);
  }

  // Initialize count to 0
  int zero = 0;
  unpaper_cuda_memcpy_h2d(out_count_device, &zero, sizeof(int));

  // Run kernel
  int gray_step = (int)gray_integral.step_bytes;
  int dark_step = (int)dark_integral.step_bytes;
  int img_w = TEST_WIDTH;
  int img_h = TEST_HEIGHT;
  int tile_w = TILE_W;
  int tile_h = TILE_H;
  int step_x = STEP_X;
  int step_y = STEP_Y;
  int gray_threshold = GRAY_THRESHOLD;  // Use int for CUDA alignment
  int max_tiles = MAX_TILES;

  void *args[] = {
      &gray_integral.device_ptr,
      &dark_integral.device_ptr,
      &gray_step,
      &dark_step,
      &img_w,
      &img_h,
      &tile_w,
      &tile_h,
      &step_x,
      &step_y,
      &gray_threshold,
      &out_tiles_device,
      &out_count_device,
      &max_tiles,
  };

  // Calculate grid/block dimensions
  int tiles_per_row = (TEST_WIDTH - TILE_W) / STEP_X + 1;
  int tiles_per_col = (TEST_HEIGHT - TILE_H) / STEP_Y + 1;
  int threads_per_block = 16;
  int grid_x = (tiles_per_row + threads_per_block - 1) / threads_per_block;
  int grid_y = (tiles_per_col + threads_per_block - 1) / threads_per_block;

  unpaper_cuda_launch_kernel(k_grayfilter_scan, grid_x, grid_y, 1,
                             threads_per_block, threads_per_block, 1, args);
  unpaper_cuda_stream_synchronize();

  // Download results
  int gpu_count = 0;
  unpaper_cuda_memcpy_d2h(&gpu_count, out_count_device, sizeof(int));

  GrayfilterTile *gpu_tiles = (GrayfilterTile *)malloc(gpu_count * sizeof(GrayfilterTile));
  if (gpu_tiles == NULL && gpu_count > 0) {
    fprintf(stderr, "malloc failed\n");
    exit(1);
  }
  if (gpu_count > 0) {
    unpaper_cuda_memcpy_d2h(gpu_tiles, out_tiles_device,
                            gpu_count * sizeof(GrayfilterTile));
  }

  // Compute expected results
  GrayfilterTile *expected_tiles = (GrayfilterTile *)malloc(MAX_TILES * sizeof(GrayfilterTile));
  if (expected_tiles == NULL) {
    fprintf(stderr, "malloc failed\n");
    exit(1);
  }
  int expected_count = compute_expected_tiles(
      gray_host, dark_mask_host, TEST_WIDTH, TEST_HEIGHT, TILE_W, TILE_H,
      STEP_X, STEP_Y, GRAY_THRESHOLD, expected_tiles, MAX_TILES);

  // Verify count matches
  if (gpu_count != expected_count) {
    fprintf(stderr, "FAILED: count mismatch GPU=%d expected=%d\n", gpu_count,
            expected_count);

    printf("GPU tiles found:\n");
    for (int i = 0; i < gpu_count && i < 10; i++) {
      printf("  (%d, %d)\n", gpu_tiles[i].x, gpu_tiles[i].y);
    }
    printf("Expected tiles:\n");
    for (int i = 0; i < expected_count && i < 10; i++) {
      printf("  (%d, %d)\n", expected_tiles[i].x, expected_tiles[i].y);
    }
    exit(1);
  }

  // Verify all expected tiles are found
  for (int i = 0; i < expected_count; i++) {
    int found = 0;
    for (int j = 0; j < gpu_count; j++) {
      if (gpu_tiles[j].x == expected_tiles[i].x &&
          gpu_tiles[j].y == expected_tiles[i].y) {
        found = 1;
        break;
      }
    }
    if (!found) {
      fprintf(stderr, "FAILED: expected tile (%d, %d) not found in GPU output\n",
              expected_tiles[i].x, expected_tiles[i].y);
      exit(1);
    }
  }

  // Clean up
  free(gpu_tiles);
  free(expected_tiles);
  unpaper_npp_integral_free(gray_integral.device_ptr);
  unpaper_npp_integral_free(dark_integral.device_ptr);
  unpaper_cuda_free(out_tiles_device);
  unpaper_cuda_free(out_count_device);
  unpaper_cuda_free(gray_device);
  unpaper_cuda_free(dark_mask_device);
  free(gray_host);
  free(dark_mask_host);

  printf("OK (found %d tiles to wipe)\n", gpu_count);
}

static void test_grayfilter_scan_no_light_tiles(void) {
  printf("test_grayfilter_scan_no_light_tiles: ");

  ensure_kernel_loaded();

  // Create dark gray image with no dark pixels
  // Average = 200, inverse = 55 > threshold
  uint8_t *gray_host = (uint8_t *)malloc(TEST_WIDTH * TEST_HEIGHT);
  uint8_t *dark_mask_host = (uint8_t *)calloc(TEST_WIDTH * TEST_HEIGHT, 1);
  if (gray_host == NULL || dark_mask_host == NULL) {
    fprintf(stderr, "malloc failed\n");
    exit(1);
  }
  memset(gray_host, 200, TEST_WIDTH * TEST_HEIGHT);

  // Upload to GPU
  uint64_t gray_device = unpaper_cuda_malloc(TEST_WIDTH * TEST_HEIGHT);
  uint64_t dark_mask_device = unpaper_cuda_malloc(TEST_WIDTH * TEST_HEIGHT);
  unpaper_cuda_memcpy_h2d(gray_device, gray_host, TEST_WIDTH * TEST_HEIGHT);
  unpaper_cuda_memcpy_h2d(dark_mask_device, dark_mask_host, TEST_WIDTH * TEST_HEIGHT);

  // Compute GPU integrals
  UnpaperNppIntegral gray_integral, dark_integral;
  unpaper_npp_integral_8u32s(gray_device, TEST_WIDTH, TEST_HEIGHT,
                              TEST_WIDTH, NULL, &gray_integral);
  unpaper_npp_integral_8u32s(dark_mask_device, TEST_WIDTH, TEST_HEIGHT,
                              TEST_WIDTH, NULL, &dark_integral);

  // Allocate output buffers
  uint64_t out_tiles_device = unpaper_cuda_malloc(MAX_TILES * sizeof(GrayfilterTile));
  uint64_t out_count_device = unpaper_cuda_malloc(sizeof(int));
  int zero = 0;
  unpaper_cuda_memcpy_h2d(out_count_device, &zero, sizeof(int));

  // Run kernel
  int gray_step = (int)gray_integral.step_bytes;
  int dark_step = (int)dark_integral.step_bytes;
  int img_w = TEST_WIDTH;
  int img_h = TEST_HEIGHT;
  int tile_w = TILE_W;
  int tile_h = TILE_H;
  int step_x = STEP_X;
  int step_y = STEP_Y;
  int gray_threshold = GRAY_THRESHOLD;
  int max_tiles = MAX_TILES;

  void *args[] = {
      &gray_integral.device_ptr, &dark_integral.device_ptr,
      &gray_step, &dark_step, &img_w, &img_h,
      &tile_w, &tile_h, &step_x, &step_y, &gray_threshold,
      &out_tiles_device, &out_count_device, &max_tiles,
  };

  int tiles_per_row = (TEST_WIDTH - TILE_W) / STEP_X + 1;
  int tiles_per_col = (TEST_HEIGHT - TILE_H) / STEP_Y + 1;
  int threads = 16;
  unpaper_cuda_launch_kernel(k_grayfilter_scan,
                             (tiles_per_row + threads - 1) / threads,
                             (tiles_per_col + threads - 1) / threads, 1,
                             threads, threads, 1, args);
  unpaper_cuda_stream_synchronize();

  // Download count
  int gpu_count = 0;
  unpaper_cuda_memcpy_d2h(&gpu_count, out_count_device, sizeof(int));

  if (gpu_count != 0) {
    fprintf(stderr, "FAILED: expected 0 tiles, got %d\n", gpu_count);
    exit(1);
  }

  // Clean up
  unpaper_npp_integral_free(gray_integral.device_ptr);
  unpaper_npp_integral_free(dark_integral.device_ptr);
  unpaper_cuda_free(out_tiles_device);
  unpaper_cuda_free(out_count_device);
  unpaper_cuda_free(gray_device);
  unpaper_cuda_free(dark_mask_device);
  free(gray_host);
  free(dark_mask_host);

  printf("OK\n");
}

static void test_grayfilter_scan_all_dark_pixels(void) {
  printf("test_grayfilter_scan_all_dark_pixels: ");

  ensure_kernel_loaded();

  // Create white image but with dark pixels everywhere
  uint8_t *gray_host = (uint8_t *)malloc(TEST_WIDTH * TEST_HEIGHT);
  uint8_t *dark_mask_host = (uint8_t *)malloc(TEST_WIDTH * TEST_HEIGHT);
  if (gray_host == NULL || dark_mask_host == NULL) {
    fprintf(stderr, "malloc failed\n");
    exit(1);
  }
  memset(gray_host, 255, TEST_WIDTH * TEST_HEIGHT);  // Very light
  memset(dark_mask_host, 255, TEST_WIDTH * TEST_HEIGHT);  // But all marked dark

  // Upload to GPU
  uint64_t gray_device = unpaper_cuda_malloc(TEST_WIDTH * TEST_HEIGHT);
  uint64_t dark_mask_device = unpaper_cuda_malloc(TEST_WIDTH * TEST_HEIGHT);
  unpaper_cuda_memcpy_h2d(gray_device, gray_host, TEST_WIDTH * TEST_HEIGHT);
  unpaper_cuda_memcpy_h2d(dark_mask_device, dark_mask_host, TEST_WIDTH * TEST_HEIGHT);

  // Compute GPU integrals
  UnpaperNppIntegral gray_integral, dark_integral;
  unpaper_npp_integral_8u32s(gray_device, TEST_WIDTH, TEST_HEIGHT,
                              TEST_WIDTH, NULL, &gray_integral);
  unpaper_npp_integral_8u32s(dark_mask_device, TEST_WIDTH, TEST_HEIGHT,
                              TEST_WIDTH, NULL, &dark_integral);

  // Allocate output buffers
  uint64_t out_tiles_device = unpaper_cuda_malloc(MAX_TILES * sizeof(GrayfilterTile));
  uint64_t out_count_device = unpaper_cuda_malloc(sizeof(int));
  int zero = 0;
  unpaper_cuda_memcpy_h2d(out_count_device, &zero, sizeof(int));

  // Run kernel
  int gray_step = (int)gray_integral.step_bytes;
  int dark_step = (int)dark_integral.step_bytes;
  int img_w = TEST_WIDTH;
  int img_h = TEST_HEIGHT;
  int tile_w = TILE_W;
  int tile_h = TILE_H;
  int step_x = STEP_X;
  int step_y = STEP_Y;
  int gray_threshold = GRAY_THRESHOLD;
  int max_tiles = MAX_TILES;

  void *args[] = {
      &gray_integral.device_ptr, &dark_integral.device_ptr,
      &gray_step, &dark_step, &img_w, &img_h,
      &tile_w, &tile_h, &step_x, &step_y, &gray_threshold,
      &out_tiles_device, &out_count_device, &max_tiles,
  };

  int tiles_per_row = (TEST_WIDTH - TILE_W) / STEP_X + 1;
  int tiles_per_col = (TEST_HEIGHT - TILE_H) / STEP_Y + 1;
  int threads = 16;
  unpaper_cuda_launch_kernel(k_grayfilter_scan,
                             (tiles_per_row + threads - 1) / threads,
                             (tiles_per_col + threads - 1) / threads, 1,
                             threads, threads, 1, args);
  unpaper_cuda_stream_synchronize();

  // Download count - no tiles should be wiped because all have dark pixels
  int gpu_count = 0;
  unpaper_cuda_memcpy_d2h(&gpu_count, out_count_device, sizeof(int));

  if (gpu_count != 0) {
    fprintf(stderr, "FAILED: expected 0 tiles (all have dark pixels), got %d\n", gpu_count);
    exit(1);
  }

  // Clean up
  unpaper_npp_integral_free(gray_integral.device_ptr);
  unpaper_npp_integral_free(dark_integral.device_ptr);
  unpaper_cuda_free(out_tiles_device);
  unpaper_cuda_free(out_count_device);
  unpaper_cuda_free(gray_device);
  unpaper_cuda_free(dark_mask_device);
  free(gray_host);
  free(dark_mask_host);

  printf("OK\n");
}

static void test_grayfilter_scan_all_white(void) {
  printf("test_grayfilter_scan_all_white: ");

  ensure_kernel_loaded();

  // Create perfectly white image with no dark pixels
  // All tiles should be wiped (inverse_lightness = 0 < any threshold)
  uint8_t *gray_host = (uint8_t *)malloc(TEST_WIDTH * TEST_HEIGHT);
  uint8_t *dark_mask_host = (uint8_t *)calloc(TEST_WIDTH * TEST_HEIGHT, 1);
  if (gray_host == NULL || dark_mask_host == NULL) {
    fprintf(stderr, "malloc failed\n");
    exit(1);
  }
  memset(gray_host, 255, TEST_WIDTH * TEST_HEIGHT);

  // Debug: print expected tile count
  int tiles_per_row_expected = (TEST_WIDTH - TILE_W) / STEP_X + 1;
  int tiles_per_col_expected = (TEST_HEIGHT - TILE_H) / STEP_Y + 1;
  printf("\n  Debug: img=%dx%d, tile=%dx%d, step=%dx%d\n",
         TEST_WIDTH, TEST_HEIGHT, TILE_W, TILE_H, STEP_X, STEP_Y);
  printf("  Debug: tiles_per_row=%d, tiles_per_col=%d, total=%d\n",
         tiles_per_row_expected, tiles_per_col_expected,
         tiles_per_row_expected * tiles_per_col_expected);

  // Upload to GPU
  uint64_t gray_device = unpaper_cuda_malloc(TEST_WIDTH * TEST_HEIGHT);
  uint64_t dark_mask_device = unpaper_cuda_malloc(TEST_WIDTH * TEST_HEIGHT);
  unpaper_cuda_memcpy_h2d(gray_device, gray_host, TEST_WIDTH * TEST_HEIGHT);
  unpaper_cuda_memcpy_h2d(dark_mask_device, dark_mask_host, TEST_WIDTH * TEST_HEIGHT);

  // Compute GPU integrals
  UnpaperNppIntegral gray_integral, dark_integral;
  unpaper_npp_integral_8u32s(gray_device, TEST_WIDTH, TEST_HEIGHT,
                              TEST_WIDTH, NULL, &gray_integral);
  unpaper_npp_integral_8u32s(dark_mask_device, TEST_WIDTH, TEST_HEIGHT,
                              TEST_WIDTH, NULL, &dark_integral);

  // Allocate output buffers
  uint64_t out_tiles_device = unpaper_cuda_malloc(MAX_TILES * sizeof(GrayfilterTile));
  uint64_t out_count_device = unpaper_cuda_malloc(sizeof(int));
  int zero = 0;
  unpaper_cuda_memcpy_h2d(out_count_device, &zero, sizeof(int));

  // Run kernel
  int gray_step = (int)gray_integral.step_bytes;
  int dark_step = (int)dark_integral.step_bytes;
  int img_w = TEST_WIDTH;
  int img_h = TEST_HEIGHT;
  int tile_w = TILE_W;
  int tile_h = TILE_H;
  int step_x = STEP_X;
  int step_y = STEP_Y;
  int gray_threshold = GRAY_THRESHOLD;
  int max_tiles = MAX_TILES;

  printf("  Debug: gray_step=%d, dark_step=%d\n", gray_step, dark_step);

  void *args[] = {
      &gray_integral.device_ptr, &dark_integral.device_ptr,
      &gray_step, &dark_step, &img_w, &img_h,
      &tile_w, &tile_h, &step_x, &step_y, &gray_threshold,
      &out_tiles_device, &out_count_device, &max_tiles,
  };

  int tiles_per_row = (TEST_WIDTH - TILE_W) / STEP_X + 1;
  int tiles_per_col = (TEST_HEIGHT - TILE_H) / STEP_Y + 1;
  int threads = 16;
  int grid_x = (tiles_per_row + threads - 1) / threads;
  int grid_y = (tiles_per_col + threads - 1) / threads;
  printf("  Debug: grid=(%d,%d), threads=(%d,%d)\n", grid_x, grid_y, threads, threads);

  unpaper_cuda_launch_kernel(k_grayfilter_scan, grid_x, grid_y, 1,
                             threads, threads, 1, args);
  unpaper_cuda_stream_synchronize();

  // Download count - all tiles should be wiped
  int gpu_count = 0;
  unpaper_cuda_memcpy_d2h(&gpu_count, out_count_device, sizeof(int));

  int expected_count = tiles_per_row * tiles_per_col;
  printf("  Debug: gpu_count=%d, expected=%d\n", gpu_count, expected_count);
  if (gpu_count != expected_count) {
    // Download tiles to see which are found
    GrayfilterTile *gpu_tiles = (GrayfilterTile *)malloc(gpu_count * sizeof(GrayfilterTile));
    if (gpu_tiles != NULL && gpu_count > 0) {
      unpaper_cuda_memcpy_d2h(gpu_tiles, out_tiles_device, gpu_count * sizeof(GrayfilterTile));
      // Print found tiles
      int grid[6][8] = {0};
      for (int i = 0; i < gpu_count; i++) {
        int tx = gpu_tiles[i].x / STEP_X;
        int ty = gpu_tiles[i].y / STEP_Y;
        if (tx < 8 && ty < 6) grid[ty][tx] = 1;
      }
      printf("  Tile grid (1=found):\n");
      for (int y = 0; y < 6; y++) {
        printf("    row %d: ", y);
        for (int x = 0; x < 8; x++) {
          printf("%d ", grid[y][x]);
        }
        printf("\n");
      }
      free(gpu_tiles);
    }
    fprintf(stderr, "FAILED: expected %d tiles, got %d\n", expected_count, gpu_count);
    exit(1);
  }

  // Clean up
  unpaper_npp_integral_free(gray_integral.device_ptr);
  unpaper_npp_integral_free(dark_integral.device_ptr);
  unpaper_cuda_free(out_tiles_device);
  unpaper_cuda_free(out_count_device);
  unpaper_cuda_free(gray_device);
  unpaper_cuda_free(dark_mask_device);
  free(gray_host);
  free(dark_mask_host);

  printf("OK (all %d tiles wiped)\n", gpu_count);
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

  // Run simpler tests first to verify basic functionality
  test_grayfilter_scan_all_white();
  test_grayfilter_scan_no_light_tiles();
  test_grayfilter_scan_all_dark_pixels();
  test_grayfilter_scan_basic();

  printf("All grayfilter scan tests passed!\n");
  return 0;
}
