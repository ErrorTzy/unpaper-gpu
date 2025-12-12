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
              "b=(%u,%u,%u) fmt=%d\\n",
              label, x, y, pa.r, pa.g, pa.b, pb.r, pb.g, pb.b, a.frame->format);
      assert(false);
    }
  }
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

static void test_stretch_rgb24(Interpolation interp) {
  Image cpu =
      create_image((RectangleSize){.width = 31, .height = 19}, AV_PIX_FMT_RGB24,
                   false, PIXEL_WHITE, 128);
  Image gpu =
      create_image((RectangleSize){.width = 31, .height = 19}, AV_PIX_FMT_RGB24,
                   false, PIXEL_WHITE, 128);
  fill_pattern(cpu);
  fill_pattern(gpu);

  image_backend_select(UNPAPER_DEVICE_CPU);
  stretch_and_replace(&cpu, (RectangleSize){.width = 53, .height = 41}, interp);

  image_backend_select(UNPAPER_DEVICE_CUDA);
  stretch_and_replace(&gpu, (RectangleSize){.width = 53, .height = 41}, interp);
  image_ensure_cpu(&gpu);

  char label[128];
  snprintf(label, sizeof(label), "stretch_rgb24_up_%s", interp_name(interp));
  assert_images_equal(label, cpu, gpu);

  free_image(&cpu);
  free_image(&gpu);
}

static void test_stretch_gray8(Interpolation interp) {
  Image cpu =
      create_image((RectangleSize){.width = 37, .height = 23}, AV_PIX_FMT_GRAY8,
                   false, PIXEL_WHITE, 128);
  Image gpu =
      create_image((RectangleSize){.width = 37, .height = 23}, AV_PIX_FMT_GRAY8,
                   false, PIXEL_WHITE, 128);
  fill_pattern(cpu);
  fill_pattern(gpu);

  image_backend_select(UNPAPER_DEVICE_CPU);
  stretch_and_replace(&cpu, (RectangleSize){.width = 19, .height = 17}, interp);

  image_backend_select(UNPAPER_DEVICE_CUDA);
  stretch_and_replace(&gpu, (RectangleSize){.width = 19, .height = 17}, interp);
  image_ensure_cpu(&gpu);

  char label[128];
  snprintf(label, sizeof(label), "stretch_gray8_down_%s", interp_name(interp));
  assert_images_equal(label, cpu, gpu);

  free_image(&cpu);
  free_image(&gpu);
}

static void test_resize_rgb24(Interpolation interp) {
  Image cpu =
      create_image((RectangleSize){.width = 30, .height = 20}, AV_PIX_FMT_RGB24,
                   false, PIXEL_WHITE, 128);
  Image gpu =
      create_image((RectangleSize){.width = 30, .height = 20}, AV_PIX_FMT_RGB24,
                   false, PIXEL_WHITE, 128);
  fill_pattern(cpu);
  fill_pattern(gpu);

  image_backend_select(UNPAPER_DEVICE_CPU);
  resize_and_replace(&cpu, (RectangleSize){.width = 41, .height = 41}, interp);

  image_backend_select(UNPAPER_DEVICE_CUDA);
  resize_and_replace(&gpu, (RectangleSize){.width = 41, .height = 41}, interp);
  image_ensure_cpu(&gpu);

  char label[128];
  snprintf(label, sizeof(label), "resize_rgb24_center_%s", interp_name(interp));
  assert_images_equal(label, cpu, gpu);

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
