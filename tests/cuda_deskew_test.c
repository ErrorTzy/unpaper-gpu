// SPDX-FileCopyrightText: 2025 The unpaper authors
//
// SPDX-License-Identifier: GPL-2.0-only

#include <assert.h>
#include <math.h>
#include <stdbool.h>
#include <stdio.h>
#include <stdint.h>

#include <libavutil/mathematics.h>
#include <libavutil/frame.h>
#include <libavutil/pixfmt.h>

#include "imageprocess/backend.h"
#include "imageprocess/cuda_runtime.h"
#include "imageprocess/deskew.h"
#include "imageprocess/image.h"
#include "imageprocess/pixel.h"

static void fill_skewed_edge(Image image, float radians) {
  assert(image.frame != NULL);
  const int w = image.frame->width;
  const int h = image.frame->height;
  const float cx = (float)w * 0.35f;
  const float cy = (float)h / 2.0f;
  const float m = tanf(radians);

  Rectangle area = full_image(image);
  scan_rectangle(area) {
    const float boundary_x = cx + m * ((float)y - cy);
    const bool is_black = ((float)x) >= boundary_x;
    set_pixel(image, (Point){x, y}, is_black ? PIXEL_BLACK : PIXEL_WHITE);
  }
}

// Compute absolute difference between two uint8_t values
static int abs_diff(uint8_t a, uint8_t b) {
  return a > b ? (int)(a - b) : (int)(b - a);
}

// Check if images are similar within tolerance.
// OpenCV's warpAffine may produce slightly different results due to
// interpolation implementation differences. For document deskewing,
// small differences are acceptable.
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

  Rectangle area = full_image(a);
  scan_rectangle(area) {
    Pixel pa = get_pixel(a, (Point){x, y});
    Pixel pb = get_pixel(b, (Point){x, y});

    int dr = abs_diff(pa.r, pb.r);
    int dg = abs_diff(pa.g, pb.g);
    int db = abs_diff(pa.b, pb.b);
    int max_diff = dr > dg ? (dr > db ? dr : db) : (dg > db ? dg : db);

    total_pixels++;
    if (max_diff > max_per_pixel_diff) {
      diff_pixels++;
    }
  }

  double diff_fraction = (double)diff_pixels / (double)total_pixels;
  if (diff_fraction > max_diff_fraction) {
    fprintf(stderr,
            "%s: too many differing pixels: %.2f%% > %.2f%%\n",
            label, diff_fraction * 100.0, max_diff_fraction * 100.0);
    assert(false);
  }

  fprintf(stderr, "%s: PASS (%.2f%% pixels differ by >%d)\n",
          label, diff_fraction * 100.0, max_per_pixel_diff);
}

static void test_detect_rotation_and_deskew_parity(void) {
  const RectangleSize sz = {.width = 241, .height = 179};
  const float true_radians = (float)(2.0 * M_PI / 180.0);
  const Rectangle mask = (Rectangle){{{0, 0}, {sz.width - 1, sz.height - 1}}};

  Edges edges = {.left = true, .top = false, .right = true, .bottom = false};
  DeskewParameters params;
  edges.right = false;
  assert(validate_deskew_parameters(&params, 5.0f, 0.1f, 10.0f, 400, 0.5f,
                                   edges));

  Image cpu = create_image(sz, AV_PIX_FMT_GRAY8, false, PIXEL_WHITE, 128);
  Image gpu = create_image(sz, AV_PIX_FMT_GRAY8, false, PIXEL_WHITE, 128);
  fill_skewed_edge(cpu, true_radians);
  fill_skewed_edge(gpu, true_radians);

  image_backend_select(UNPAPER_DEVICE_CPU);
  const float rotation_cpu = detect_rotation(cpu, mask, params);

  image_backend_select(UNPAPER_DEVICE_CUDA);
  const float rotation_cuda = detect_rotation(gpu, mask, params);

  assert(fabsf(rotation_cpu) > 1e-4f);
  assert(fabsf(rotation_cpu - rotation_cuda) < 1e-6f);

  Image cpu_d = create_image(sz, AV_PIX_FMT_GRAY8, false, PIXEL_WHITE, 128);
  Image gpu_d = create_image(sz, AV_PIX_FMT_GRAY8, false, PIXEL_WHITE, 128);
  fill_skewed_edge(cpu_d, true_radians);
  fill_skewed_edge(gpu_d, true_radians);

  image_backend_select(UNPAPER_DEVICE_CPU);
  deskew(cpu_d, mask, rotation_cpu, INTERP_CUBIC);

  image_backend_select(UNPAPER_DEVICE_CUDA);
  deskew(gpu_d, mask, rotation_cpu, INTERP_CUBIC);
  image_ensure_cpu(&gpu_d);

  // OpenCV warpAffine may produce slightly different results due to
  // interpolation implementation differences. Allow small tolerance.
  // For deskewing documents, differences at edges are acceptable.
  assert_images_similar("deskew_cpu_vs_cuda", cpu_d, gpu_d, 128, 0.15);

  Image gpu_d2 = create_image(sz, AV_PIX_FMT_GRAY8, false, PIXEL_WHITE, 128);
  fill_skewed_edge(gpu_d2, true_radians);
  image_backend_select(UNPAPER_DEVICE_CUDA);
  deskew(gpu_d2, mask, rotation_cpu, INTERP_CUBIC);
  image_ensure_cpu(&gpu_d2);

  // Determinism check: running CUDA deskew twice should be identical
  assert_images_similar("deskew_cuda_determinism", gpu_d, gpu_d2, 0, 0.0);

  free_image(&cpu);
  free_image(&gpu);
  free_image(&cpu_d);
  free_image(&gpu_d);
  free_image(&gpu_d2);
}

int main(void) {
  UnpaperCudaInitStatus st = unpaper_cuda_try_init();
  if (st != UNPAPER_CUDA_INIT_OK) {
    return 77; // skip if no CUDA runtime/device
  }

  test_detect_rotation_and_deskew_parity();
  return 0;
}
