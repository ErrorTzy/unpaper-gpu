// SPDX-FileCopyrightText: 2025 The unpaper authors
//
// SPDX-License-Identifier: GPL-2.0-only

#include <assert.h>
#include <inttypes.h>
#include <stdbool.h>
#include <stdint.h>
#include <stdio.h>
#include <string.h>

#include <libavutil/frame.h>
#include <libavutil/pixfmt.h>

#include "imageprocess/backend.h"
#include "imageprocess/blit.h"
#include "imageprocess/cuda_runtime.h"
#include "imageprocess/image.h"
#include "imageprocess/masks.h"
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
              "b=(%u,%u,%u) fmt=%d\n",
              label, x, y, pa.r, pa.g, pa.b, pb.r, pb.g, pb.b, a.frame->format);
      assert(false);
    }
  }
}

static void test_apply_masks_rgb24(void) {
  image_backend_select(UNPAPER_DEVICE_CPU);
  const RectangleSize sz = {.width = 37, .height = 23};
  Image cpu = create_image(sz, AV_PIX_FMT_RGB24, false, PIXEL_WHITE, 128);
  Image gpu = create_image(sz, AV_PIX_FMT_RGB24, false, PIXEL_WHITE, 128);
  fill_pattern(cpu);
  fill_pattern(gpu);

  Rectangle masks[2] = {
      (Rectangle){{{5, 4}, {20, 12}}},
      (Rectangle){{{0, 0}, {3, 3}}},
  };
  const Pixel color = (Pixel){.r = 10, .g = 20, .b = 30};

  image_backend_select(UNPAPER_DEVICE_CPU);
  apply_masks(cpu, masks, 2, color);

  image_backend_select(UNPAPER_DEVICE_CUDA);
  apply_masks(gpu, masks, 2, color);
  image_ensure_cpu(&gpu);

  assert_images_equal("apply_masks_rgb24", cpu, gpu);
  free_image(&cpu);
  free_image(&gpu);
}

static void test_apply_wipes_mono(void) {
  image_backend_select(UNPAPER_DEVICE_CPU);
  const RectangleSize sz = {.width = 41, .height = 17};
  Image cpu = create_image(sz, AV_PIX_FMT_MONOBLACK, false, PIXEL_WHITE, 128);
  Image gpu = create_image(sz, AV_PIX_FMT_MONOBLACK, false, PIXEL_WHITE, 128);

  Rectangle area = full_image(cpu);
  scan_rectangle(area) {
    Pixel p = ((x + y) % 5 == 0) ? PIXEL_BLACK : PIXEL_WHITE;
    set_pixel(cpu, (Point){x, y}, p);
    set_pixel(gpu, (Point){x, y}, p);
  }

  Wipes wipes = {.count = 2};
  wipes.areas[0] = (Rectangle){{{2, 2}, {12, 6}}};
  wipes.areas[1] = (Rectangle){{{20, 0}, {35, 10}}};

  image_backend_select(UNPAPER_DEVICE_CPU);
  apply_wipes(cpu, wipes, PIXEL_BLACK);

  image_backend_select(UNPAPER_DEVICE_CUDA);
  apply_wipes(gpu, wipes, PIXEL_BLACK);
  image_ensure_cpu(&gpu);

  assert_images_equal("apply_wipes_mono", cpu, gpu);
  free_image(&cpu);
  free_image(&gpu);
}

static void test_apply_border_gray8(void) {
  image_backend_select(UNPAPER_DEVICE_CPU);
  const RectangleSize sz = {.width = 53, .height = 29};
  Image cpu = create_image(sz, AV_PIX_FMT_GRAY8, false, PIXEL_WHITE, 128);
  Image gpu = create_image(sz, AV_PIX_FMT_GRAY8, false, PIXEL_WHITE, 128);
  fill_pattern(cpu);
  fill_pattern(gpu);

  const Border border = (Border){.left = 3, .top = 2, .right = 5, .bottom = 4};

  image_backend_select(UNPAPER_DEVICE_CPU);
  apply_border(cpu, border, PIXEL_BLACK);

  image_backend_select(UNPAPER_DEVICE_CUDA);
  apply_border(gpu, border, PIXEL_BLACK);
  image_ensure_cpu(&gpu);

  assert_images_equal("apply_border_gray8", cpu, gpu);
  free_image(&cpu);
  free_image(&gpu);
}

static void test_align_mask_rgb24(void) {
  image_backend_select(UNPAPER_DEVICE_CPU);
  const RectangleSize sz = {.width = 60, .height = 40};
  Image cpu = create_image(sz, AV_PIX_FMT_RGB24, false, PIXEL_WHITE, 128);
  Image gpu = create_image(sz, AV_PIX_FMT_RGB24, false, PIXEL_WHITE, 128);

  Rectangle area = full_image(cpu);
  scan_rectangle(area) {
    set_pixel(cpu, (Point){x, y}, PIXEL_WHITE);
    set_pixel(gpu, (Point){x, y}, PIXEL_WHITE);
  }

  const Rectangle inside = (Rectangle){{{10, 8}, {29, 19}}};
  scan_rectangle(inside) {
    set_pixel(cpu, (Point){x, y}, PIXEL_BLACK);
    set_pixel(gpu, (Point){x, y}, PIXEL_BLACK);
  }

  MaskAlignmentParameters params = {
      .alignment = (Edges){.left = true, .top = true},
      .margin = (Delta){.horizontal = 2, .vertical = 3},
  };

  image_backend_select(UNPAPER_DEVICE_CPU);
  align_mask(cpu, inside, full_image(cpu), params);

  image_backend_select(UNPAPER_DEVICE_CUDA);
  align_mask(gpu, inside, full_image(gpu), params);
  image_ensure_cpu(&gpu);

  assert_images_equal("align_mask_rgb24", cpu, gpu);
  free_image(&cpu);
  free_image(&gpu);
}

static void test_detect_border_and_masks_match(void) {
  image_backend_select(UNPAPER_DEVICE_CPU);
  const RectangleSize sz = {.width = 96, .height = 64};
  Image cpu = create_image(sz, AV_PIX_FMT_RGB24, false, PIXEL_WHITE, 128);
  Image gpu = create_image(sz, AV_PIX_FMT_RGB24, false, PIXEL_WHITE, 128);

  Rectangle area = full_image(cpu);
  scan_rectangle(area) {
    set_pixel(cpu, (Point){x, y}, PIXEL_WHITE);
    set_pixel(gpu, (Point){x, y}, PIXEL_WHITE);
  }

  const Rectangle content = (Rectangle){{{20, 14}, {75, 49}}};
  scan_rectangle(content) {
    set_pixel(cpu, (Point){x, y}, PIXEL_BLACK);
    set_pixel(gpu, (Point){x, y}, PIXEL_BLACK);
  }

  BorderScanParameters border_params = {
      .scan_size = (RectangleSize){.width = 5, .height = 5},
      .scan_step = (Delta){.horizontal = 5, .vertical = 5},
      .scan_threshold = {.horizontal = 5, .vertical = 5},
      .scan_direction = DIRECTION_BOTH,
  };

  MaskDetectionParameters mask_params = {
      .scan_size = (RectangleSize){.width = 11, .height = 11},
      .scan_step = (Delta){.horizontal = 5, .vertical = 5},
      .scan_depth = {.horizontal = -1, .vertical = -1},
      .scan_direction = DIRECTION_BOTH,
      .scan_threshold = {.horizontal = 0.1f, .vertical = 0.1f},
      .minimum_width = 1,
      .maximum_width = sz.width,
      .minimum_height = 1,
      .maximum_height = sz.height,
  };

  Point points[1] = {{.x = sz.width / 2, .y = sz.height / 2}};
  Rectangle cpu_masks[1];
  Rectangle gpu_masks[1];

  image_backend_select(UNPAPER_DEVICE_CPU);
  Border cpu_border = detect_border(cpu, border_params, full_image(cpu));
  size_t cpu_count = detect_masks(cpu, mask_params, points, 1, cpu_masks);

  image_backend_select(UNPAPER_DEVICE_CUDA);
  Border gpu_border = detect_border(gpu, border_params, full_image(gpu));
  Border gpu_border2 = detect_border(gpu, border_params, full_image(gpu));
  size_t gpu_count = detect_masks(gpu, mask_params, points, 1, gpu_masks);

  assert(memcmp(&cpu_border, &gpu_border, sizeof(Border)) == 0);
  assert(memcmp(&gpu_border, &gpu_border2, sizeof(Border)) == 0);
  assert(cpu_count == gpu_count);
  assert(memcmp(&cpu_masks[0], &gpu_masks[0], sizeof(Rectangle)) == 0);

  free_image(&cpu);
  free_image(&gpu);
}

int main(void) {
  UnpaperCudaInitStatus st = unpaper_cuda_try_init();
  if (st != UNPAPER_CUDA_INIT_OK) {
    return 77; // skip if no CUDA runtime/device
  }

  test_apply_masks_rgb24();
  test_apply_wipes_mono();
  test_apply_border_gray8();
  test_align_mask_rgb24();
  test_detect_border_and_masks_match();
  return 0;
}
