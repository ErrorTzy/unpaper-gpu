// SPDX-FileCopyrightText: 2025 The unpaper authors
//
// SPDX-License-Identifier: GPL-2.0-only

#include <assert.h>
#include <inttypes.h>
#include <stdbool.h>
#include <stdio.h>
#include <stdint.h>

#include <libavutil/frame.h>
#include <libavutil/pixfmt.h>

#include "imageprocess/backend.h"
#include "imageprocess/blit.h"
#include "imageprocess/cuda_runtime.h"
#include "imageprocess/filters.h"
#include "imageprocess/image.h"
#include "imageprocess/pixel.h"

static void assert_images_equal(const char *label, Image a, Image b) {
  assert(a.frame != NULL);
  assert(b.frame != NULL);
  assert(a.frame->width == b.frame->width);
  assert(a.frame->height == b.frame->height);
  assert(a.frame->format == b.frame->format);

  Rectangle area = full_image(a);
  scan_rectangle(area) {
    Pixel pa = get_pixel(a, (Point){x, y});
    Pixel pb = get_pixel(b, (Point){x, y});
    if (compare_pixel(pa, pb) != 0) {
      fprintf(stderr,
              "mismatch (%s) at (%" PRId32 ",%" PRId32 "): a=(%u,%u,%u) "
              "b=(%u,%u,%u) fmt=%d\n",
              label, x, y, pa.r, pa.g, pa.b, pb.r, pb.g, pb.b, a.frame->format);
      assert(false);
    }
  }
}

static void fill_noise_pattern(Image image) {
  wipe_rectangle(image, full_image(image), PIXEL_WHITE);

  // Single-pixel cluster (should be removed at small intensity).
  set_pixel(image, (Point){5, 5}, PIXEL_BLACK);

  // Larger cluster (should generally remain).
  for (int32_t y = 10; y <= 12; y++) {
    for (int32_t x = 15; x <= 17; x++) {
      set_pixel(image, (Point){x, y}, PIXEL_BLACK);
    }
  }
}

static void test_noisefilter_gray8(void) {
  const RectangleSize sz = {.width = 32, .height = 24};
  const uint8_t abs_black_threshold = 64;
  const uint64_t intensity = 2;
  const uint8_t min_white_level = 200;

  image_backend_select(UNPAPER_DEVICE_CPU);
  Image cpu = create_image(sz, AV_PIX_FMT_GRAY8, true, PIXEL_WHITE,
                           abs_black_threshold);
  Image gpu = create_image(sz, AV_PIX_FMT_GRAY8, true, PIXEL_WHITE,
                           abs_black_threshold);
  fill_noise_pattern(cpu);
  fill_noise_pattern(gpu);

  image_backend_select(UNPAPER_DEVICE_CPU);
  noisefilter(cpu, intensity, min_white_level);

  image_backend_select(UNPAPER_DEVICE_CUDA);
  noisefilter(gpu, intensity, min_white_level);
  image_ensure_cpu(&gpu);

  assert_images_equal("noisefilter_gray8", cpu, gpu);

  // Determinism: run the exact same CUDA invocation twice on identical inputs.
  Image gpu1 = create_image(sz, AV_PIX_FMT_GRAY8, true, PIXEL_WHITE,
                            abs_black_threshold);
  Image gpu2 = create_image(sz, AV_PIX_FMT_GRAY8, true, PIXEL_WHITE,
                            abs_black_threshold);
  fill_noise_pattern(gpu1);
  fill_noise_pattern(gpu2);

  image_backend_select(UNPAPER_DEVICE_CUDA);
  noisefilter(gpu1, intensity, min_white_level);
  noisefilter(gpu2, intensity, min_white_level);
  image_ensure_cpu(&gpu1);
  image_ensure_cpu(&gpu2);
  assert_images_equal("noisefilter_determinism", gpu1, gpu2);

  free_image(&cpu);
  free_image(&gpu);
  free_image(&gpu1);
  free_image(&gpu2);
}

static void fill_blackfilter_pattern(Image image) {
  wipe_rectangle(image, full_image(image), PIXEL_WHITE);

  // A solid black vertical bar spanning the image height.
  Rectangle bar = (Rectangle){{{8, 0}, {11, 23}}};
  wipe_rectangle(image, bar, PIXEL_BLACK);
}

static void test_blackfilter_gray8(void) {
  const RectangleSize sz = {.width = 32, .height = 24};
  const uint8_t abs_black_threshold = 128;

  Rectangle exclusions[4];
  BlackfilterParameters params;
  (void)validate_blackfilter_parameters(
      &params,
      /*scan_size=*/(RectangleSize){.width = 4, .height = 4},
      /*scan_step=*/(Delta){.horizontal = 2, .vertical = 2},
      /*scan_depth_h=*/(uint32_t)sz.width,
      /*scan_depth_v=*/(uint32_t)sz.height,
      /*scan_direction=*/DIRECTION_BOTH,
      /*threshold=*/0.9f,
      /*intensity=*/2,
      /*exclusions_count=*/0,
      /*exclusions=*/exclusions);

  image_backend_select(UNPAPER_DEVICE_CPU);
  Image cpu = create_image(sz, AV_PIX_FMT_GRAY8, true, PIXEL_WHITE,
                           abs_black_threshold);
  Image gpu = create_image(sz, AV_PIX_FMT_GRAY8, true, PIXEL_WHITE,
                           abs_black_threshold);
  fill_blackfilter_pattern(cpu);
  fill_blackfilter_pattern(gpu);

  image_backend_select(UNPAPER_DEVICE_CPU);
  blackfilter(cpu, params);

  image_backend_select(UNPAPER_DEVICE_CUDA);
  blackfilter(gpu, params);
  image_ensure_cpu(&gpu);

  assert_images_equal("blackfilter_gray8", cpu, gpu);

  free_image(&cpu);
  free_image(&gpu);
}

static void fill_grayfilter_pattern(Image image) {
  // Two 4x4 blocks:
  // - top-left: light gray (eligible for wipe if threshold low enough)
  // - top-right: darker gray (should remain).
  wipe_rectangle(image, full_image(image), PIXEL_WHITE);
  wipe_rectangle(image, (Rectangle){{{0, 0}, {3, 3}}}, (Pixel){150, 150, 150});
  wipe_rectangle(image, (Rectangle){{{4, 0}, {7, 3}}}, (Pixel){80, 80, 80});
}

static void test_grayfilter_gray8(void) {
  const RectangleSize sz = {.width = 8, .height = 8};
  const uint8_t abs_black_threshold = 50;

  GrayfilterParameters params;
  (void)validate_grayfilter_parameters(
      &params,
      /*scan_size=*/(RectangleSize){.width = 4, .height = 4},
      /*scan_step=*/(Delta){.horizontal = 4, .vertical = 4},
      /*threshold=*/0.5f);

  image_backend_select(UNPAPER_DEVICE_CPU);
  Image cpu = create_image(sz, AV_PIX_FMT_GRAY8, true, PIXEL_WHITE,
                           abs_black_threshold);
  Image gpu = create_image(sz, AV_PIX_FMT_GRAY8, true, PIXEL_WHITE,
                           abs_black_threshold);
  fill_grayfilter_pattern(cpu);
  fill_grayfilter_pattern(gpu);

  image_backend_select(UNPAPER_DEVICE_CPU);
  grayfilter(cpu, params);

  image_backend_select(UNPAPER_DEVICE_CUDA);
  grayfilter(gpu, params);
  image_ensure_cpu(&gpu);

  assert_images_equal("grayfilter_gray8", cpu, gpu);

  free_image(&cpu);
  free_image(&gpu);
}

static void fill_blurfilter_pattern(Image image) {
  wipe_rectangle(image, full_image(image), PIXEL_WHITE);

  // Keep the top scan row all-white to avoid depending on undefined contents of
  // the CPU blurfilter's first-iteration buffers.

  // A sparse block in the 2nd scan row (y = 4..7).
  set_pixel(image, (Point){1, 5}, PIXEL_BLACK);
  set_pixel(image, (Point){2, 5}, PIXEL_BLACK);
  set_pixel(image, (Point){1, 6}, PIXEL_BLACK);

  // A dense block in the 2nd scan row (y = 4..7).
  for (int32_t y = 4; y < 8; y++) {
    for (int32_t x = 8; x < 12; x++) {
      set_pixel(image, (Point){x, y}, PIXEL_BLACK);
    }
  }
}

static void test_blurfilter_gray8(void) {
  const RectangleSize sz = {.width = 16, .height = 16};
  const uint8_t abs_black_threshold = 64;
  const uint8_t abs_white_threshold = 200;

  BlurfilterParameters params;
  (void)validate_blurfilter_parameters(
      &params,
      /*scan_size=*/(RectangleSize){.width = 4, .height = 4},
      /*scan_step=*/(Delta){.horizontal = 2, .vertical = 2},
      /*intensity=*/0.5f);

  image_backend_select(UNPAPER_DEVICE_CPU);
  Image cpu = create_image(sz, AV_PIX_FMT_GRAY8, true, PIXEL_WHITE,
                           abs_black_threshold);
  Image gpu = create_image(sz, AV_PIX_FMT_GRAY8, true, PIXEL_WHITE,
                           abs_black_threshold);
  fill_blurfilter_pattern(cpu);
  fill_blurfilter_pattern(gpu);

  image_backend_select(UNPAPER_DEVICE_CPU);
  blurfilter(cpu, params, abs_white_threshold);

  image_backend_select(UNPAPER_DEVICE_CUDA);
  blurfilter(gpu, params, abs_white_threshold);
  image_ensure_cpu(&gpu);

  assert_images_equal("blurfilter_gray8", cpu, gpu);

  free_image(&cpu);
  free_image(&gpu);
}

int main(void) {
  UnpaperCudaInitStatus st = unpaper_cuda_try_init();
  if (st != UNPAPER_CUDA_INIT_OK) {
    return 77;
  }

  test_noisefilter_gray8();
  test_blackfilter_gray8();
  test_grayfilter_gray8();
  test_blurfilter_gray8();
  return 0;
}
