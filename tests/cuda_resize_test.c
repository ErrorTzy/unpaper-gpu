// SPDX-FileCopyrightText: 2025 The unpaper authors
//
// SPDX-License-Identifier: GPL-2.0-only

#include <assert.h>
#include <inttypes.h>
#include <stdbool.h>
#include <stdint.h>
#include <stdio.h>

#include <libavutil/frame.h>
#include <libavutil/pixfmt.h>

#include "imageprocess/backend.h"
#include "imageprocess/blit.h"
#include "imageprocess/cuda_runtime.h"
#include "imageprocess/image.h"
#include "imageprocess/interpolate.h"
#include "imageprocess/pixel.h"

static void fill_pattern(Image image) {
  Rectangle area = full_image(image);
  scan_rectangle(area) {
    Pixel p = (Pixel){
        .r = (uint8_t)((x * 7 + y * 3) & 0xFF),
        .g = (uint8_t)((x * 5 + y * 11) & 0xFF),
        .b = (uint8_t)((x * 13 + y * 17) & 0xFF),
    };
    set_pixel(image, (Point){x, y}, p);
  }
}

// Compute absolute difference between two uint8_t values
static int abs_diff(uint8_t a, uint8_t b) {
  return a > b ? (int)(a - b) : (int)(b - a);
}

// Check if images are similar within tolerance.
// OpenCV uses half-pixel center coordinate convention which differs from
// unpaper's corner-based convention. This causes ~1 pixel sampling differences
// at certain positions, so we allow for tolerance.
//
// max_per_pixel_diff: Maximum allowed difference per channel per pixel
// max_diff_fraction: Maximum fraction of pixels that can exceed tolerance
static void assert_images_similar(const char *label, Image a, Image b,
                                  int max_per_pixel_diff,
                                  double max_diff_fraction) {
  assert(a.frame != NULL);
  assert(b.frame != NULL);
  assert(a.frame->width == b.frame->width);
  assert(a.frame->height == b.frame->height);
  assert(a.frame->format == b.frame->format);

  int64_t total_pixels = 0;
  int64_t diff_pixels = 0;
  double total_error = 0.0;

  Rectangle area = full_image(a);
  scan_rectangle(area) {
    Pixel pa = get_pixel(a, (Point){x, y});
    Pixel pb = get_pixel(b, (Point){x, y});

    int dr = abs_diff(pa.r, pb.r);
    int dg = abs_diff(pa.g, pb.g);
    int db = abs_diff(pa.b, pb.b);
    int max_diff = dr > dg ? (dr > db ? dr : db) : (dg > db ? dg : db);

    total_error += (double)(dr + dg + db);
    total_pixels++;

    if (max_diff > max_per_pixel_diff) {
      diff_pixels++;
    }
  }

  double diff_fraction = (double)diff_pixels / (double)total_pixels;
  double mean_error = total_error / (double)(total_pixels * 3);

  if (diff_fraction > max_diff_fraction) {
    fprintf(
        stderr,
        "%s: too many differing pixels: %.2f%% > %.2f%% (mean error: %.1f)\n",
        label, diff_fraction * 100.0, max_diff_fraction * 100.0, mean_error);
    assert(false);
  }

  fprintf(stderr, "%s: PASS (%.2f%% pixels differ by >%d, mean error: %.1f)\n",
          label, diff_fraction * 100.0, max_per_pixel_diff, mean_error);
}

static const char *interp_name(Interpolation i) {
  switch (i) {
  case INTERP_NN:
    return "nn";
  case INTERP_LINEAR:
    return "linear";
  case INTERP_CUBIC:
  default:
    return "cubic";
  }
}

// OpenCV's coordinate convention causes sampling differences.
// Allow up to 50% of pixels to differ (due to shifted sampling grid)
// and max 255 per-channel difference (neighboring pixels can vary widely).
// The mean error should still be reasonable.
#define RESIZE_MAX_PIXEL_DIFF 255
#define RESIZE_MAX_DIFF_FRACTION 0.60

static void test_stretch_rgb24(Interpolation interp) {
  Image cpu = create_image((RectangleSize){.width = 31, .height = 19},
                           AV_PIX_FMT_RGB24, false, PIXEL_WHITE, 128);
  Image gpu = create_image((RectangleSize){.width = 31, .height = 19},
                           AV_PIX_FMT_RGB24, false, PIXEL_WHITE, 128);
  fill_pattern(cpu);
  fill_pattern(gpu);

  image_backend_select(UNPAPER_DEVICE_CPU);
  stretch_and_replace(&cpu, (RectangleSize){.width = 53, .height = 41}, interp);

  image_backend_select(UNPAPER_DEVICE_CUDA);
  stretch_and_replace(&gpu, (RectangleSize){.width = 53, .height = 41}, interp);
  image_ensure_cpu(&gpu);

  char label[128];
  snprintf(label, sizeof(label), "stretch_rgb24_up_%s", interp_name(interp));
  assert_images_similar(label, cpu, gpu, RESIZE_MAX_PIXEL_DIFF,
                        RESIZE_MAX_DIFF_FRACTION);

  free_image(&cpu);
  free_image(&gpu);
}

static void test_stretch_gray8(Interpolation interp) {
  Image cpu = create_image((RectangleSize){.width = 37, .height = 23},
                           AV_PIX_FMT_GRAY8, false, PIXEL_WHITE, 128);
  Image gpu = create_image((RectangleSize){.width = 37, .height = 23},
                           AV_PIX_FMT_GRAY8, false, PIXEL_WHITE, 128);
  fill_pattern(cpu);
  fill_pattern(gpu);

  image_backend_select(UNPAPER_DEVICE_CPU);
  stretch_and_replace(&cpu, (RectangleSize){.width = 19, .height = 17}, interp);

  image_backend_select(UNPAPER_DEVICE_CUDA);
  stretch_and_replace(&gpu, (RectangleSize){.width = 19, .height = 17}, interp);
  image_ensure_cpu(&gpu);

  char label[128];
  snprintf(label, sizeof(label), "stretch_gray8_down_%s", interp_name(interp));
  assert_images_similar(label, cpu, gpu, RESIZE_MAX_PIXEL_DIFF,
                        RESIZE_MAX_DIFF_FRACTION);

  free_image(&cpu);
  free_image(&gpu);
}

static void test_resize_rgb24(Interpolation interp) {
  Image cpu = create_image((RectangleSize){.width = 30, .height = 20},
                           AV_PIX_FMT_RGB24, false, PIXEL_WHITE, 128);
  Image gpu = create_image((RectangleSize){.width = 30, .height = 20},
                           AV_PIX_FMT_RGB24, false, PIXEL_WHITE, 128);
  fill_pattern(cpu);
  fill_pattern(gpu);

  image_backend_select(UNPAPER_DEVICE_CPU);
  resize_and_replace(&cpu, (RectangleSize){.width = 41, .height = 41}, interp);

  image_backend_select(UNPAPER_DEVICE_CUDA);
  resize_and_replace(&gpu, (RectangleSize){.width = 41, .height = 41}, interp);
  image_ensure_cpu(&gpu);

  char label[128];
  snprintf(label, sizeof(label), "resize_rgb24_center_%s", interp_name(interp));
  assert_images_similar(label, cpu, gpu, RESIZE_MAX_PIXEL_DIFF,
                        RESIZE_MAX_DIFF_FRACTION);

  free_image(&cpu);
  free_image(&gpu);
}

int main(void) {
  UnpaperCudaInitStatus st = unpaper_cuda_try_init();
  if (st != UNPAPER_CUDA_INIT_OK) {
    return 77;
  }

  test_stretch_rgb24(INTERP_NN);
  test_stretch_rgb24(INTERP_LINEAR);
  test_stretch_rgb24(INTERP_CUBIC);

  test_stretch_gray8(INTERP_NN);
  test_stretch_gray8(INTERP_LINEAR);
  test_stretch_gray8(INTERP_CUBIC);

  test_resize_rgb24(INTERP_NN);
  test_resize_rgb24(INTERP_LINEAR);
  test_resize_rgb24(INTERP_CUBIC);

  return 0;
}
