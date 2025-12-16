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

static void test_wipe_rectangle_rgb24(void) {
  image_backend_select(UNPAPER_DEVICE_CPU);
  const RectangleSize sz = {.width = 31, .height = 19};
  Image cpu = create_image(sz, AV_PIX_FMT_RGB24, false, PIXEL_WHITE, 128);
  Image gpu = create_image(sz, AV_PIX_FMT_RGB24, false, PIXEL_WHITE, 128);
  fill_pattern(cpu);
  fill_pattern(gpu);

  const Rectangle r = (Rectangle){{{5, 4}, {20, 12}}};
  const Pixel color = (Pixel){.r = 10, .g = 20, .b = 30};

  wipe_rectangle(cpu, r, color);

  image_backend_select(UNPAPER_DEVICE_CUDA);
  wipe_rectangle(gpu, r, color);
  image_ensure_cpu(&gpu);

  assert_images_equal("wipe_rectangle_rgb24", cpu, gpu);

  free_image(&cpu);
  free_image(&gpu);
}

static void test_copy_rectangle_gray_to_rgb(void) {
  image_backend_select(UNPAPER_DEVICE_CPU);
  const RectangleSize src_sz = {.width = 29, .height = 17};
  const RectangleSize dst_sz = {.width = 40, .height = 30};

  Image src_cpu =
      create_image(src_sz, AV_PIX_FMT_GRAY8, false, PIXEL_WHITE, 128);
  Image src_gpu =
      create_image(src_sz, AV_PIX_FMT_GRAY8, false, PIXEL_WHITE, 128);
  Image dst_cpu =
      create_image(dst_sz, AV_PIX_FMT_RGB24, true, PIXEL_WHITE, 128);
  Image dst_gpu =
      create_image(dst_sz, AV_PIX_FMT_RGB24, true, PIXEL_WHITE, 128);

  fill_pattern(src_cpu);
  fill_pattern(src_gpu);

  const Rectangle src_area = (Rectangle){{{3, 2}, {18, 10}}};
  const Point dst_origin = (Point){7, 5};

  image_backend_select(UNPAPER_DEVICE_CPU);
  copy_rectangle(src_cpu, dst_cpu, src_area, dst_origin);

  image_backend_select(UNPAPER_DEVICE_CUDA);
  copy_rectangle(src_gpu, dst_gpu, src_area, dst_origin);
  image_ensure_cpu(&dst_gpu);

  assert_images_equal("copy_rectangle_gray_to_rgb", dst_cpu, dst_gpu);

  free_image(&src_cpu);
  free_image(&src_gpu);
  free_image(&dst_cpu);
  free_image(&dst_gpu);
}

static void test_center_image_rgb24(void) {
  image_backend_select(UNPAPER_DEVICE_CPU);
  Image src_cpu = create_image((RectangleSize){.width = 19, .height = 11},
                               AV_PIX_FMT_RGB24, false, PIXEL_WHITE, 128);
  Image src_gpu = create_image((RectangleSize){.width = 19, .height = 11},
                               AV_PIX_FMT_RGB24, false, PIXEL_WHITE, 128);
  fill_pattern(src_cpu);
  fill_pattern(src_gpu);

  Image dst_cpu = create_image((RectangleSize){.width = 40, .height = 30},
                               AV_PIX_FMT_RGB24, true, PIXEL_WHITE, 128);
  Image dst_gpu = create_image((RectangleSize){.width = 40, .height = 30},
                               AV_PIX_FMT_RGB24, true, PIXEL_WHITE, 128);

  image_backend_select(UNPAPER_DEVICE_CPU);
  center_image(src_cpu, dst_cpu, POINT_ORIGIN, size_of_image(dst_cpu));

  image_backend_select(UNPAPER_DEVICE_CUDA);
  center_image(src_gpu, dst_gpu, POINT_ORIGIN, size_of_image(dst_gpu));
  image_ensure_cpu(&dst_gpu);

  assert_images_equal("center_image_rgb24", dst_cpu, dst_gpu);

  free_image(&src_cpu);
  free_image(&src_gpu);
  free_image(&dst_cpu);
  free_image(&dst_gpu);
}

static void test_mirror_shift_rotate_mono(void) {
  image_backend_select(UNPAPER_DEVICE_CPU);
  const RectangleSize sz = {.width = 23, .height = 15};
  Image cpu = create_image(sz, AV_PIX_FMT_MONOWHITE, false, PIXEL_WHITE, 128);
  Image gpu = create_image(sz, AV_PIX_FMT_MONOWHITE, false, PIXEL_WHITE, 128);

  Rectangle area = full_image(cpu);
  scan_rectangle(area) {
    Pixel p = ((x + y) % 3 == 0) ? PIXEL_BLACK : PIXEL_WHITE;
    set_pixel(cpu, (Point){x, y}, p);
    set_pixel(gpu, (Point){x, y}, p);
  }

  image_backend_select(UNPAPER_DEVICE_CPU);
  mirror(cpu, DIRECTION_HORIZONTAL);
  shift_image(&cpu, (Delta){.horizontal = 2, .vertical = -1});
  flip_rotate_90(&cpu, ROTATE_CLOCKWISE);

  image_backend_select(UNPAPER_DEVICE_CUDA);
  mirror(gpu, DIRECTION_HORIZONTAL);
  shift_image(&gpu, (Delta){.horizontal = 2, .vertical = -1});
  flip_rotate_90(&gpu, ROTATE_CLOCKWISE);
  image_ensure_cpu(&gpu);

  assert_images_equal("mirror_shift_rotate_mono", cpu, gpu);

  free_image(&cpu);
  free_image(&gpu);
}

int main(void) {
  UnpaperCudaInitStatus st = unpaper_cuda_try_init();
  if (st != UNPAPER_CUDA_INIT_OK) {
    return 77; // skip if no CUDA runtime/device
  }

  test_wipe_rectangle_rgb24();
  test_copy_rectangle_gray_to_rgb();
  test_center_image_rgb24();
  test_mirror_shift_rotate_mono();
  return 0;
}
