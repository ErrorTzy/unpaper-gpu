// SPDX-FileCopyrightText: 2025 The unpaper authors
//
// SPDX-License-Identifier: GPL-2.0-only

#include "imageprocess/backend.h"

#include <inttypes.h>

#include <libavutil/frame.h>
#include <libavutil/mem.h>
#include <libavutil/pixfmt.h>

#include "imageprocess/cuda_kernels_format.h"
#include "imageprocess/cuda_runtime.h"
#include "lib/logging.h"
#include "lib/math_util.h"

typedef struct {
  uint64_t dptr;
  size_t bytes;
  int format;
  int width;
  int height;
  int linesize;
  bool cpu_dirty;
  bool cuda_dirty;
} ImageCudaState;

extern const char unpaper_cuda_kernels_ptx[];

static void *cuda_module;
static void *k_wipe_rect_bytes;
static void *k_wipe_rect_mono;
static void *k_copy_rect_to_bytes;
static void *k_copy_rect_to_mono;
static void *k_mirror_bytes;
static void *k_mirror_mono;
static void *k_rotate90_bytes;
static void *k_rotate90_mono;
static void *k_stretch_bytes;
static void *k_stretch_mono;
static void *k_count_brightness_range;
static void *k_sum_lightness_rect;
static void *k_sum_darkness_inverse_rect;
static void *k_noisefilter;
static void *k_blackfilter_floodfill_rect;

static void cuda_unimplemented(const char *op_name) {
  errOutput("CUDA backend selected, but it is not implemented yet (%s).",
            op_name);
}

static void ensure_kernels_loaded(void) {
  if (cuda_module != NULL) {
    return;
  }

  cuda_module = unpaper_cuda_module_load_ptx(unpaper_cuda_kernels_ptx);
  k_wipe_rect_bytes =
      unpaper_cuda_module_get_function(cuda_module, "unpaper_wipe_rect_bytes");
  k_wipe_rect_mono =
      unpaper_cuda_module_get_function(cuda_module, "unpaper_wipe_rect_mono");
  k_copy_rect_to_bytes =
      unpaper_cuda_module_get_function(cuda_module, "unpaper_copy_rect_to_bytes");
  k_copy_rect_to_mono =
      unpaper_cuda_module_get_function(cuda_module, "unpaper_copy_rect_to_mono");
  k_mirror_bytes =
      unpaper_cuda_module_get_function(cuda_module, "unpaper_mirror_bytes");
  k_mirror_mono =
      unpaper_cuda_module_get_function(cuda_module, "unpaper_mirror_mono");
  k_rotate90_bytes =
      unpaper_cuda_module_get_function(cuda_module, "unpaper_rotate90_bytes");
  k_rotate90_mono =
      unpaper_cuda_module_get_function(cuda_module, "unpaper_rotate90_mono");
  k_stretch_bytes =
      unpaper_cuda_module_get_function(cuda_module, "unpaper_stretch_bytes");
  k_stretch_mono =
      unpaper_cuda_module_get_function(cuda_module, "unpaper_stretch_mono");
  k_count_brightness_range = unpaper_cuda_module_get_function(
      cuda_module, "unpaper_count_brightness_range");
  k_sum_lightness_rect = unpaper_cuda_module_get_function(
      cuda_module, "unpaper_sum_lightness_rect");
  k_sum_darkness_inverse_rect = unpaper_cuda_module_get_function(
      cuda_module, "unpaper_sum_darkness_inverse_rect");
  k_noisefilter =
      unpaper_cuda_module_get_function(cuda_module, "unpaper_noisefilter");
  k_blackfilter_floodfill_rect = unpaper_cuda_module_get_function(
      cuda_module, "unpaper_blackfilter_floodfill_rect");
}

static inline uint8_t pixel_grayscale(Pixel pixel) {
  return (uint8_t)(((uint32_t)pixel.r + (uint32_t)pixel.g + (uint32_t)pixel.b) /
                   3u);
}

static UnpaperCudaFormat cuda_format_from_av(int fmt) {
  switch (fmt) {
  case AV_PIX_FMT_GRAY8:
    return UNPAPER_CUDA_FMT_GRAY8;
  case AV_PIX_FMT_Y400A:
    return UNPAPER_CUDA_FMT_Y400A;
  case AV_PIX_FMT_RGB24:
    return UNPAPER_CUDA_FMT_RGB24;
  case AV_PIX_FMT_MONOWHITE:
    return UNPAPER_CUDA_FMT_MONOWHITE;
  case AV_PIX_FMT_MONOBLACK:
    return UNPAPER_CUDA_FMT_MONOBLACK;
  default:
    return UNPAPER_CUDA_FMT_INVALID;
  }
}

static int bytes_per_pixel_from_av(int fmt) {
  switch (fmt) {
  case AV_PIX_FMT_GRAY8:
    return 1;
  case AV_PIX_FMT_Y400A:
    return 2;
  case AV_PIX_FMT_RGB24:
    return 3;
  default:
    return 0;
  }
}

static ImageCudaState *image_cuda_state(Image image) {
  if (image.frame == NULL || image.frame->opaque_ref == NULL) {
    return NULL;
  }
  return (ImageCudaState *)image.frame->opaque_ref->data;
}

static bool rect_empty(Rectangle area) {
  return (area.vertex[0].x > area.vertex[1].x) ||
         (area.vertex[0].y > area.vertex[1].y);
}

static unsigned long long cuda_rect_count_brightness_range(
    Image image, Rectangle input_area, uint8_t min_brightness,
    uint8_t max_brightness) {
  Rectangle area = clip_rectangle(image, input_area);
  if (rect_empty(area)) {
    return 0;
  }
  const int rect_w = area.vertex[1].x - area.vertex[0].x + 1;
  const int rect_h = area.vertex[1].y - area.vertex[0].y + 1;
  if (rect_w <= 0 || rect_h <= 0) {
    return 0;
  }

  ensure_kernels_loaded();
  image_ensure_cuda(&image);
  ImageCudaState *st = image_cuda_state(image);
  if (st == NULL || st->dptr == 0) {
    errOutput("CUDA image state missing for count_brightness_range.");
  }

  const UnpaperCudaFormat fmt = cuda_format_from_av(image.frame->format);
  if (fmt == UNPAPER_CUDA_FMT_INVALID) {
    errOutput("CUDA count_brightness_range: unsupported pixel format.");
  }

  uint64_t out_dptr = unpaper_cuda_malloc(sizeof(unsigned long long));
  unpaper_cuda_memset_d8(out_dptr, 0, sizeof(unsigned long long));

  const int src_fmt = (int)fmt;
  const int src_w = image.frame->width;
  const int src_h = image.frame->height;
  const int x0 = area.vertex[0].x;
  const int y0 = area.vertex[0].y;
  const int x1 = area.vertex[1].x;
  const int y1 = area.vertex[1].y;

  const unsigned long long total =
      (unsigned long long)rect_w * (unsigned long long)rect_h;
  const uint32_t block_x = 256;
  uint32_t grid_x = (uint32_t)((total + block_x - 1) / block_x);
  if (grid_x == 0) {
    grid_x = 1;
  }
  if (grid_x > 1024) {
    grid_x = 1024;
  }

  void *params[] = {
      &st->dptr, &st->linesize, &src_fmt, &src_w, &src_h, &x0, &y0, &x1, &y1,
      &min_brightness, &max_brightness, &out_dptr,
  };
  unpaper_cuda_launch_kernel(k_count_brightness_range, grid_x, 1, 1, block_x, 1,
                             1, params);

  unsigned long long out = 0;
  unpaper_cuda_memcpy_d2h(&out, out_dptr, sizeof(out));
  unpaper_cuda_free(out_dptr);
  return out;
}

static unsigned long long cuda_rect_sum_lightness(Image image,
                                                  Rectangle input_area) {
  Rectangle area = clip_rectangle(image, input_area);
  if (rect_empty(area)) {
    return 0;
  }
  const int rect_w = area.vertex[1].x - area.vertex[0].x + 1;
  const int rect_h = area.vertex[1].y - area.vertex[0].y + 1;
  if (rect_w <= 0 || rect_h <= 0) {
    return 0;
  }

  ensure_kernels_loaded();
  image_ensure_cuda(&image);
  ImageCudaState *st = image_cuda_state(image);
  if (st == NULL || st->dptr == 0) {
    errOutput("CUDA image state missing for sum_lightness_rect.");
  }

  const UnpaperCudaFormat fmt = cuda_format_from_av(image.frame->format);
  if (fmt == UNPAPER_CUDA_FMT_INVALID) {
    errOutput("CUDA sum_lightness_rect: unsupported pixel format.");
  }

  uint64_t out_dptr = unpaper_cuda_malloc(sizeof(unsigned long long));
  unpaper_cuda_memset_d8(out_dptr, 0, sizeof(unsigned long long));

  const int src_fmt = (int)fmt;
  const int src_w = image.frame->width;
  const int src_h = image.frame->height;
  const int x0 = area.vertex[0].x;
  const int y0 = area.vertex[0].y;
  const int x1 = area.vertex[1].x;
  const int y1 = area.vertex[1].y;

  const unsigned long long total =
      (unsigned long long)rect_w * (unsigned long long)rect_h;
  const uint32_t block_x = 256;
  uint32_t grid_x = (uint32_t)((total + block_x - 1) / block_x);
  if (grid_x == 0) {
    grid_x = 1;
  }
  if (grid_x > 1024) {
    grid_x = 1024;
  }

  void *params[] = {
      &st->dptr, &st->linesize, &src_fmt, &src_w, &src_h, &x0, &y0, &x1, &y1,
      &out_dptr,
  };
  unpaper_cuda_launch_kernel(k_sum_lightness_rect, grid_x, 1, 1, block_x, 1, 1,
                             params);

  unsigned long long out = 0;
  unpaper_cuda_memcpy_d2h(&out, out_dptr, sizeof(out));
  unpaper_cuda_free(out_dptr);
  return out;
}

static unsigned long long cuda_rect_sum_darkness_inverse(Image image,
                                                         Rectangle input_area) {
  Rectangle area = clip_rectangle(image, input_area);
  if (rect_empty(area)) {
    return 0;
  }
  const int rect_w = area.vertex[1].x - area.vertex[0].x + 1;
  const int rect_h = area.vertex[1].y - area.vertex[0].y + 1;
  if (rect_w <= 0 || rect_h <= 0) {
    return 0;
  }

  ensure_kernels_loaded();
  image_ensure_cuda(&image);
  ImageCudaState *st = image_cuda_state(image);
  if (st == NULL || st->dptr == 0) {
    errOutput("CUDA image state missing for sum_darkness_inverse_rect.");
  }

  const UnpaperCudaFormat fmt = cuda_format_from_av(image.frame->format);
  if (fmt == UNPAPER_CUDA_FMT_INVALID) {
    errOutput("CUDA sum_darkness_inverse_rect: unsupported pixel format.");
  }

  uint64_t out_dptr = unpaper_cuda_malloc(sizeof(unsigned long long));
  unpaper_cuda_memset_d8(out_dptr, 0, sizeof(unsigned long long));

  const int src_fmt = (int)fmt;
  const int src_w = image.frame->width;
  const int src_h = image.frame->height;
  const int x0 = area.vertex[0].x;
  const int y0 = area.vertex[0].y;
  const int x1 = area.vertex[1].x;
  const int y1 = area.vertex[1].y;

  const unsigned long long total =
      (unsigned long long)rect_w * (unsigned long long)rect_h;
  const uint32_t block_x = 256;
  uint32_t grid_x = (uint32_t)((total + block_x - 1) / block_x);
  if (grid_x == 0) {
    grid_x = 1;
  }
  if (grid_x > 1024) {
    grid_x = 1024;
  }

  void *params[] = {
      &st->dptr, &st->linesize, &src_fmt, &src_w, &src_h, &x0, &y0, &x1, &y1,
      &out_dptr,
  };
  unpaper_cuda_launch_kernel(k_sum_darkness_inverse_rect, grid_x, 1, 1, block_x,
                             1, 1, params);

  unsigned long long out = 0;
  unpaper_cuda_memcpy_d2h(&out, out_dptr, sizeof(out));
  unpaper_cuda_free(out_dptr);
  return out;
}

static uint8_t cuda_inverse_lightness_rect(Image image, Rectangle input_area) {
  Rectangle area = clip_rectangle(image, input_area);
  if (rect_empty(area)) {
    return 0;
  }
  const unsigned long long count = count_pixels(area);
  if (count == 0ull) {
    return 0;
  }

  const unsigned long long sum = cuda_rect_sum_lightness(image, area);
  return (uint8_t)(0xFFu - (sum / count));
}

static uint8_t cuda_darkness_rect(Image image, Rectangle input_area) {
  Rectangle area = clip_rectangle(image, input_area);
  if (rect_empty(area)) {
    return 0;
  }
  const unsigned long long count = count_pixels(area);
  if (count == 0ull) {
    return 0;
  }

  const unsigned long long sum = cuda_rect_sum_darkness_inverse(image, area);
  return (uint8_t)(0xFFu - (sum / count));
}

static void blackfilter_scan_cuda(Image image, BlackfilterParameters params,
                                  Delta step, RectangleSize stripe_size,
                                  Delta shift, uint64_t stack_dptr,
                                  int stack_cap) {
  if (step.horizontal != 0 && step.vertical != 0) {
    errOutput("blackfilter_scan() called with diagonal steps, impossible! "
              "(%d, %d)",
              step.horizontal, step.vertical);
  }

  const Rectangle image_area = full_image(image);

  Rectangle area = rectangle_from_size(POINT_ORIGIN, stripe_size);
  while (point_in_rectangle(area.vertex[0], image_area)) {
    if (!point_in_rectangle(area.vertex[1], image_area)) {
      Delta d = distance_between(area.vertex[1], image_area.vertex[1]);
      area = shift_rectangle(area, d);
    }

    do {
      const uint8_t blackness = cuda_darkness_rect(image, area);
      if (blackness >= params.abs_threshold) {
        if (!rectangle_overlap_any(area, params.exclusions_count,
                                   params.exclusions)) {
          ensure_kernels_loaded();
          image_ensure_cuda(&image);
          ImageCudaState *st = image_cuda_state(image);
          if (st == NULL || st->dptr == 0) {
            errOutput("CUDA image state missing for blackfilter floodfill.");
          }

          const UnpaperCudaFormat fmt = cuda_format_from_av(image.frame->format);
          if (fmt == UNPAPER_CUDA_FMT_INVALID) {
            errOutput("CUDA blackfilter: unsupported pixel format.");
          }

          const Rectangle clipped = clip_rectangle(image, area);
          if (!rect_empty(clipped)) {
            const int x0 = clipped.vertex[0].x;
            const int y0 = clipped.vertex[0].y;
            const int x1 = clipped.vertex[1].x;
            const int y1 = clipped.vertex[1].y;
            const int img_fmt = (int)fmt;
            const int w = image.frame->width;
            const int h = image.frame->height;
            const uint8_t mask_max = image.abs_black_threshold;
            const unsigned long long intensity =
                params.intensity < 0 ? 0ull
                                     : (unsigned long long)params.intensity;

            void *kparams[] = {
                &st->dptr, &st->linesize, &img_fmt, &w, &h, &x0, &y0, &x1, &y1,
                &mask_max, &intensity, &stack_dptr, &stack_cap,
            };
            unpaper_cuda_launch_kernel(k_blackfilter_floodfill_rect, 1, 1, 1, 1,
                                       1, 1, kparams);

            st->cuda_dirty = true;
            st->cpu_dirty = false;
          }
        }
      }

      area = shift_rectangle(area, step);
    } while (point_in_rectangle(area.vertex[0], image_area));

    area = shift_rectangle(area, shift);
  }
}

static void wipe_rectangle_cuda(Image image, Rectangle input_area, Pixel color) {
  Rectangle area = clip_rectangle(image, input_area);
  RectangleSize sz = size_of_rectangle(area);
  if (sz.width <= 0 || sz.height <= 0) {
    return;
  }

  ensure_kernels_loaded();
  image_ensure_cuda(&image);
  ImageCudaState *st = image_cuda_state(image);
  if (st == NULL || st->dptr == 0) {
    errOutput("CUDA image state missing for wipe_rectangle.");
  }

  const UnpaperCudaFormat fmt = cuda_format_from_av(image.frame->format);
  if (fmt == UNPAPER_CUDA_FMT_INVALID) {
    errOutput("CUDA wipe_rectangle: unsupported pixel format.");
  }

  const int x0 = area.vertex[0].x;
  const int y0 = area.vertex[0].y;
  const int x1 = area.vertex[1].x;
  const int y1 = area.vertex[1].y;

  if (fmt == UNPAPER_CUDA_FMT_MONOWHITE || fmt == UNPAPER_CUDA_FMT_MONOBLACK) {
    const uint8_t gray = pixel_grayscale(color);
    const bool pixel_black = gray < image.abs_black_threshold;
    const uint8_t bit_set =
        (fmt == UNPAPER_CUDA_FMT_MONOWHITE) ? (pixel_black ? 1u : 0u)
                                            : (pixel_black ? 0u : 1u);

    void *params[] = {
        &st->dptr, &st->linesize, &x0, &y0, &x1, &y1, &bit_set,
    };
    const uint32_t block_x = 32;
    const uint32_t block_y = 8;
    const uint32_t bytes_span = (uint32_t)((x1 / 8) - (x0 / 8) + 1);
    const uint32_t grid_x = (bytes_span + block_x - 1) / block_x;
    const uint32_t grid_y = ((uint32_t)sz.height + block_y - 1) / block_y;
    unpaper_cuda_launch_kernel(k_wipe_rect_mono, grid_x, grid_y, 1, block_x,
                               block_y, 1, params);
  } else {
    const int bpp = bytes_per_pixel_from_av(image.frame->format);
    const uint8_t gray = pixel_grayscale(color);
    const uint8_t c0 = (bpp == 3) ? color.r : gray;
    const uint8_t c1 = (bpp == 3) ? color.g : 0xFFu;
    const uint8_t c2 = (bpp == 3) ? color.b : 0u;

    void *params[] = {
        &st->dptr, &st->linesize, &x0, &y0, &x1, &y1, &bpp, &c0, &c1, &c2,
    };
    const uint32_t block_x = 16;
    const uint32_t block_y = 16;
    const uint32_t grid_x = ((uint32_t)sz.width + block_x - 1) / block_x;
    const uint32_t grid_y = ((uint32_t)sz.height + block_y - 1) / block_y;
    unpaper_cuda_launch_kernel(k_wipe_rect_bytes, grid_x, grid_y, 1, block_x,
                               block_y, 1, params);
  }

  st->cuda_dirty = true;
  st->cpu_dirty = false;
}

static bool clip_copy_dims(Image source, Image target, Rectangle source_area,
                           Point target_coords, int *src_x0, int *src_y0,
                           int *dst_x0, int *dst_y0, int *w, int *h) {
  Rectangle area = clip_rectangle(source, source_area);
  RectangleSize area_size = size_of_rectangle(area);

  int sx = area.vertex[0].x;
  int sy = area.vertex[0].y;
  int dx = target_coords.x;
  int dy = target_coords.y;
  int cw = area_size.width;
  int ch = area_size.height;

  const int target_w = target.frame->width;
  const int target_h = target.frame->height;

  if (cw <= 0 || ch <= 0) {
    return false;
  }

  if (dx < 0) {
    const int skip = -dx;
    dx = 0;
    sx += skip;
    cw -= skip;
  }
  if (dy < 0) {
    const int skip = -dy;
    dy = 0;
    sy += skip;
    ch -= skip;
  }

  if (dx + cw > target_w) {
    cw = target_w - dx;
  }
  if (dy + ch > target_h) {
    ch = target_h - dy;
  }

  if (cw <= 0 || ch <= 0) {
    return false;
  }

  *src_x0 = sx;
  *src_y0 = sy;
  *dst_x0 = dx;
  *dst_y0 = dy;
  *w = cw;
  *h = ch;
  return true;
}

static void copy_rectangle_cuda(Image source, Image target,
                                Rectangle source_area, Point target_coords) {
  if (source.frame == NULL || target.frame == NULL) {
    return;
  }

  int src_x0 = 0;
  int src_y0 = 0;
  int dst_x0 = 0;
  int dst_y0 = 0;
  int w = 0;
  int h = 0;
  if (!clip_copy_dims(source, target, source_area, target_coords, &src_x0,
                      &src_y0, &dst_x0, &dst_y0, &w, &h)) {
    return;
  }

  ensure_kernels_loaded();

  image_ensure_cuda(&source);
  image_ensure_cuda(&target);

  ImageCudaState *src_st = image_cuda_state(source);
  ImageCudaState *dst_st = image_cuda_state(target);
  if (src_st == NULL || dst_st == NULL || src_st->dptr == 0 ||
      dst_st->dptr == 0) {
    errOutput("CUDA image state missing for copy_rectangle.");
  }

  const UnpaperCudaFormat src_fmt = cuda_format_from_av(source.frame->format);
  const UnpaperCudaFormat dst_fmt = cuda_format_from_av(target.frame->format);
  if (src_fmt == UNPAPER_CUDA_FMT_INVALID || dst_fmt == UNPAPER_CUDA_FMT_INVALID) {
    errOutput("CUDA copy_rectangle: unsupported pixel format.");
  }

  if (dst_fmt == UNPAPER_CUDA_FMT_MONOWHITE ||
      dst_fmt == UNPAPER_CUDA_FMT_MONOBLACK) {
    const uint8_t threshold = target.abs_black_threshold;
    void *params[] = {
        &src_st->dptr, &src_st->linesize, &src_fmt, &dst_st->dptr,
        &dst_st->linesize, &dst_fmt, &src_x0, &src_y0, &dst_x0,
        &dst_y0, &w, &h, &threshold,
    };

    const uint32_t block_x = 32;
    const uint32_t block_y = 8;
    const uint32_t bytes_span =
        (uint32_t)(((dst_x0 + w - 1) / 8) - (dst_x0 / 8) + 1);
    const uint32_t grid_x = (bytes_span + block_x - 1) / block_x;
    const uint32_t grid_y = ((uint32_t)h + block_y - 1) / block_y;
    unpaper_cuda_launch_kernel(k_copy_rect_to_mono, grid_x, grid_y, 1, block_x,
                               block_y, 1, params);
  } else {
    void *params[] = {
        &src_st->dptr, &src_st->linesize, &src_fmt, &dst_st->dptr,
        &dst_st->linesize, &dst_fmt, &src_x0, &src_y0, &dst_x0,
        &dst_y0, &w, &h,
    };
    const uint32_t block_x = 16;
    const uint32_t block_y = 16;
    const uint32_t grid_x = ((uint32_t)w + block_x - 1) / block_x;
    const uint32_t grid_y = ((uint32_t)h + block_y - 1) / block_y;
    unpaper_cuda_launch_kernel(k_copy_rect_to_bytes, grid_x, grid_y, 1, block_x,
                               block_y, 1, params);
  }

  dst_st->cuda_dirty = true;
  dst_st->cpu_dirty = false;
}

static void center_image_cuda(Image source, Image target, Point target_origin,
                              RectangleSize target_size) {
  Point source_origin = POINT_ORIGIN;
  RectangleSize source_size = size_of_image(source);

  if (source_size.width < target_size.width ||
      source_size.height < target_size.height) {
    wipe_rectangle_cuda(target, rectangle_from_size(target_origin, target_size),
                        target.background);
  }

  if (source_size.width <= target_size.width) {
    target_origin.x += (target_size.width - source_size.width) / 2;
  } else {
    source_origin.x += (source_size.width - target_size.width) / 2;
    source_size.width = target_size.width;
  }
  if (source_size.height <= target_size.height) {
    target_origin.y += (target_size.height - source_size.height) / 2;
  } else {
    source_origin.y += (source_size.height - target_size.height) / 2;
    source_size.height = target_size.height;
  }

  copy_rectangle_cuda(source, target, rectangle_from_size(source_origin, source_size),
                      target_origin);
}

static void flip_rotate_90_cuda(Image *pImage, RotationDirection direction) {
  if (pImage == NULL || pImage->frame == NULL) {
    return;
  }
  if (direction != ROTATE_CLOCKWISE && direction != ROTATE_ANTICLOCKWISE) {
    errOutput("invalid rotation direction.");
  }

  ensure_kernels_loaded();
  image_ensure_cuda(pImage);
  ImageCudaState *src_st = image_cuda_state(*pImage);
  if (src_st == NULL || src_st->dptr == 0) {
    errOutput("CUDA image state missing for flip_rotate_90.");
  }

  RectangleSize src_size = size_of_image(*pImage);
  Image newimage = create_compatible_image(
      *pImage, (RectangleSize){.width = src_size.height, .height = src_size.width},
      false);
  image_ensure_cuda(&newimage);
  ImageCudaState *dst_st = image_cuda_state(newimage);
  if (dst_st == NULL || dst_st->dptr == 0) {
    errOutput("CUDA image state missing for flip_rotate_90 output.");
  }

  const UnpaperCudaFormat fmt = cuda_format_from_av(pImage->frame->format);
  if (fmt == UNPAPER_CUDA_FMT_INVALID) {
    errOutput("CUDA flip_rotate_90: unsupported pixel format.");
  }

  if (fmt == UNPAPER_CUDA_FMT_MONOWHITE || fmt == UNPAPER_CUDA_FMT_MONOBLACK) {
    const int dir = (int)direction;
    void *params[] = {
        &src_st->dptr, &src_st->linesize, &dst_st->dptr, &dst_st->linesize,
        &src_size.width, &src_size.height, &dir,
    };
    const uint32_t block_x = 32;
    const uint32_t block_y = 8;
    const uint32_t dst_w = (uint32_t)src_size.height;
    const uint32_t dst_h = (uint32_t)src_size.width;
    const uint32_t dst_bytes_per_row = (dst_w + 7) / 8;
    const uint32_t grid_x = (dst_bytes_per_row + block_x - 1) / block_x;
    const uint32_t grid_y = (dst_h + block_y - 1) / block_y;
    unpaper_cuda_launch_kernel(k_rotate90_mono, grid_x, grid_y, 1, block_x,
                               block_y, 1, params);
  } else {
    const int dir = (int)direction;
    void *params[] = {
        &src_st->dptr, &src_st->linesize, &dst_st->dptr, &dst_st->linesize,
        &fmt, &src_size.width, &src_size.height, &dir,
    };
    const uint32_t block_x = 16;
    const uint32_t block_y = 16;
    const uint32_t grid_x = ((uint32_t)src_size.width + block_x - 1) / block_x;
    const uint32_t grid_y = ((uint32_t)src_size.height + block_y - 1) / block_y;
    unpaper_cuda_launch_kernel(k_rotate90_bytes, grid_x, grid_y, 1, block_x,
                               block_y, 1, params);
  }

  dst_st->cuda_dirty = true;
  dst_st->cpu_dirty = false;

  replace_image(pImage, &newimage);
}

static void mirror_cuda(Image image, Direction direction) {
  if (image.frame == NULL) {
    return;
  }

  ensure_kernels_loaded();
  image_ensure_cuda(&image);
  ImageCudaState *st = image_cuda_state(image);
  if (st == NULL || st->dptr == 0) {
    errOutput("CUDA image state missing for mirror.");
  }

  const UnpaperCudaFormat fmt = cuda_format_from_av(image.frame->format);
  if (fmt == UNPAPER_CUDA_FMT_INVALID) {
    errOutput("CUDA mirror: unsupported pixel format.");
  }

  const uint64_t new_dptr = unpaper_cuda_malloc(st->bytes);
  unpaper_cuda_memcpy_d2d(new_dptr, st->dptr, st->bytes);

  if (fmt == UNPAPER_CUDA_FMT_MONOWHITE || fmt == UNPAPER_CUDA_FMT_MONOBLACK) {
    const int do_h = direction.horizontal ? 1 : 0;
    const int do_v = direction.vertical ? 1 : 0;
    void *params[] = {
        &st->dptr, &st->linesize, &new_dptr, &st->linesize, &st->width,
        &st->height, &do_h, &do_v,
    };
    const uint32_t block_x = 32;
    const uint32_t block_y = 8;
    const uint32_t bytes_per_row = (uint32_t)((st->width + 7) / 8);
    const uint32_t grid_x = (bytes_per_row + block_x - 1) / block_x;
    const uint32_t grid_y = ((uint32_t)st->height + block_y - 1) / block_y;
    unpaper_cuda_launch_kernel(k_mirror_mono, grid_x, grid_y, 1, block_x,
                               block_y, 1, params);
  } else {
    const int do_h = direction.horizontal ? 1 : 0;
    const int do_v = direction.vertical ? 1 : 0;
    void *params[] = {
        &st->dptr, &st->linesize, &new_dptr, &st->linesize, &fmt, &st->width,
        &st->height, &do_h, &do_v,
    };
    const uint32_t block_x = 16;
    const uint32_t block_y = 16;
    const uint32_t grid_x = ((uint32_t)st->width + block_x - 1) / block_x;
    const uint32_t grid_y = ((uint32_t)st->height + block_y - 1) / block_y;
    unpaper_cuda_launch_kernel(k_mirror_bytes, grid_x, grid_y, 1, block_x,
                               block_y, 1, params);
  }

  unpaper_cuda_free(st->dptr);
  st->dptr = new_dptr;
  st->cuda_dirty = true;
  st->cpu_dirty = false;
}

static void shift_image_cuda(Image *pImage, Delta d) {
  if (pImage == NULL || pImage->frame == NULL) {
    return;
  }
  Image newimage =
      create_compatible_image(*pImage, size_of_image(*pImage), true);
  copy_rectangle_cuda(*pImage, newimage, full_image(*pImage),
                      shift_point(POINT_ORIGIN, d));
  replace_image(pImage, &newimage);
}

static void stretch_and_replace_cuda(Image *pImage, RectangleSize size,
                                     Interpolation interpolate_type) {
  if (pImage == NULL || pImage->frame == NULL) {
    return;
  }
  if (compare_sizes(size_of_image(*pImage), size) == 0) {
    return;
  }

  ensure_kernels_loaded();

  const UnpaperCudaFormat fmt = cuda_format_from_av(pImage->frame->format);
  if (fmt == UNPAPER_CUDA_FMT_INVALID) {
    errOutput("CUDA stretch requested, but pixel format is unsupported.");
  }

  image_ensure_cuda(pImage);
  ImageCudaState *src = image_cuda_state(*pImage);
  if (src == NULL || src->dptr == 0) {
    errOutput("CUDA image state missing for stretch_and_replace (source).");
  }

  Image target = create_compatible_image(*pImage, size, false);
  image_ensure_cuda_alloc(&target);
  ImageCudaState *dst = image_cuda_state(target);
  if (dst == NULL || dst->dptr == 0) {
    errOutput("CUDA image state missing for stretch_and_replace (target).");
  }

  const int interp = (int)interpolate_type;
  if (fmt == UNPAPER_CUDA_FMT_MONOWHITE || fmt == UNPAPER_CUDA_FMT_MONOBLACK) {
    void *params[] = {
        &src->dptr, &src->linesize, &fmt, &dst->dptr, &dst->linesize, &fmt,
        &src->width, &src->height, &dst->width, &dst->height, &interp,
        &target.abs_black_threshold,
    };
    const uint32_t block_x = 32;
    const uint32_t block_y = 8;
    const uint32_t bytes_per_row = ((uint32_t)dst->width + 7u) / 8u;
    const uint32_t grid_x = (bytes_per_row + block_x - 1) / block_x;
    const uint32_t grid_y = ((uint32_t)dst->height + block_y - 1) / block_y;
    unpaper_cuda_launch_kernel(k_stretch_mono, grid_x, grid_y, 1, block_x,
                               block_y, 1, params);
  } else {
    void *params[] = {
        &src->dptr, &src->linesize, &dst->dptr, &dst->linesize, &fmt,
        &src->width, &src->height, &dst->width, &dst->height, &interp,
    };
    const uint32_t block_x = 16;
    const uint32_t block_y = 16;
    const uint32_t grid_x = ((uint32_t)dst->width + block_x - 1) / block_x;
    const uint32_t grid_y = ((uint32_t)dst->height + block_y - 1) / block_y;
    unpaper_cuda_launch_kernel(k_stretch_bytes, grid_x, grid_y, 1, block_x,
                               block_y, 1, params);
  }

  dst->cuda_dirty = true;
  dst->cpu_dirty = false;

  replace_image(pImage, &target);
}

static void resize_and_replace_cuda(Image *pImage, RectangleSize size,
                                    Interpolation interpolate_type) {
  if (pImage == NULL || pImage->frame == NULL) {
    return;
  }
  if (compare_sizes(size_of_image(*pImage), size) == 0) {
    return;
  }

  RectangleSize image_size = size_of_image(*pImage);
  verboseLog(VERBOSE_NORMAL, "resizing %dx%d -> %dx%d\n", image_size.width,
             image_size.height, size.width, size.height);

  const float horizontal_ratio = (float)size.width / (float)image_size.width;
  const float vertical_ratio = (float)size.height / (float)image_size.height;

  RectangleSize stretch_size;
  if (horizontal_ratio < vertical_ratio) {
    stretch_size =
        (RectangleSize){size.width, (int32_t)(image_size.height * horizontal_ratio)};
  } else if (vertical_ratio < horizontal_ratio) {
    stretch_size =
        (RectangleSize){(int32_t)(image_size.width * vertical_ratio), size.height};
  } else {
    stretch_size = size;
  }

  stretch_and_replace_cuda(pImage, stretch_size, interpolate_type);

  if (size.width == stretch_size.width && size.height == stretch_size.height) {
    return;
  }

  Image resized = create_compatible_image(*pImage, size, true);
  center_image_cuda(*pImage, resized, POINT_ORIGIN, size);
  replace_image(pImage, &resized);
}

static void apply_masks_cuda(Image image, const Rectangle masks[],
                             size_t masks_count, Pixel color) {
  (void)image;
  (void)masks;
  (void)masks_count;
  (void)color;
  cuda_unimplemented("apply_masks");
}

static void apply_wipes_cuda(Image image, Wipes wipes, Pixel color) {
  (void)image;
  (void)wipes;
  (void)color;
  cuda_unimplemented("apply_wipes");
}

static void apply_border_cuda(Image image, const Border border, Pixel color) {
  (void)image;
  (void)border;
  (void)color;
  cuda_unimplemented("apply_border");
}

static size_t detect_masks_cuda(Image image, MaskDetectionParameters params,
                                const Point points[], size_t points_count,
                                Rectangle masks[]) {
  (void)image;
  (void)params;
  (void)points;
  (void)points_count;
  (void)masks;
  cuda_unimplemented("detect_masks");
  return 0;
}

static void align_mask_cuda(Image image, const Rectangle inside_area,
                            const Rectangle outside,
                            MaskAlignmentParameters params) {
  (void)image;
  (void)inside_area;
  (void)outside;
  (void)params;
  cuda_unimplemented("align_mask");
}

static Border detect_border_cuda(Image image, BorderScanParameters params,
                                 const Rectangle outside_mask) {
  (void)image;
  (void)params;
  (void)outside_mask;
  cuda_unimplemented("detect_border");
  return (Border){0};
}

static void blackfilter_cuda(Image image, BlackfilterParameters params) {
  if (image.frame == NULL) {
    return;
  }

  ensure_kernels_loaded();
  image_ensure_cuda(&image);
  ImageCudaState *st = image_cuda_state(image);
  if (st == NULL || st->dptr == 0) {
    errOutput("CUDA image state missing for blackfilter.");
  }

  const int w = image.frame->width;
  const int h = image.frame->height;
  if (w <= 0 || h <= 0) {
    return;
  }

  const size_t cap = (size_t)w * (size_t)h;
  if (cap > (size_t)INT32_MAX) {
    errOutput("blackfilter CUDA stack too large.");
  }
  const int stack_cap = (int)cap;
  const size_t stack_bytes = cap * (sizeof(int32_t) * 2);
  uint64_t stack_dptr = unpaper_cuda_malloc(stack_bytes);

  if (params.scan_direction.horizontal) {
    blackfilter_scan_cuda(
        image, params, (Delta){params.scan_step.horizontal, 0},
        (RectangleSize){params.scan_size.width, (int32_t)params.scan_depth.vertical},
        (Delta){0, (int32_t)params.scan_depth.vertical}, stack_dptr, stack_cap);
  }

  if (params.scan_direction.vertical) {
    blackfilter_scan_cuda(
        image, params, (Delta){0, params.scan_step.vertical},
        (RectangleSize){(int32_t)params.scan_depth.horizontal, params.scan_size.height},
        (Delta){(int32_t)params.scan_depth.horizontal, 0}, stack_dptr,
        stack_cap);
  }

  unpaper_cuda_free(stack_dptr);
}

static void blurfilter_cuda(Image image, BlurfilterParameters params,
                            uint8_t abs_white_threshold) {
  if (image.frame == NULL) {
    return;
  }

  verboseLog(VERBOSE_NORMAL, "blur-filter...");

  RectangleSize image_size = size_of_image(image);
  if (params.scan_size.width <= 0 || params.scan_size.height <= 0) {
    verboseLog(VERBOSE_NORMAL, " deleted 0 pixels.\n");
    return;
  }

  const uint32_t blocks_per_row =
      (uint32_t)(image_size.width / params.scan_size.width);
  if (blocks_per_row == 0) {
    verboseLog(VERBOSE_NORMAL, " deleted 0 pixels.\n");
    return;
  }

  const uint64_t total_pixels_in_block =
      (uint64_t)params.scan_size.width * (uint64_t)params.scan_size.height;
  uint64_t count = 0;

  uint64_t *count_buffers =
      (uint64_t *)av_mallocz(sizeof(uint64_t) * 3 * (blocks_per_row + 2));
  if (count_buffers == NULL) {
    errOutput("unable to allocate blurfilter count buffers.");
  }

  // Mirror the CPU implementation's pointer layout (including the +1/+2
  // offsets) for parity.
  uint64_t *prevCounts = &count_buffers[0];
  uint64_t *curCounts = &count_buffers[1];
  uint64_t *nextCounts = &count_buffers[2];

  curCounts[0] = total_pixels_in_block;
  curCounts[blocks_per_row] = total_pixels_in_block;
  nextCounts[0] = total_pixels_in_block;
  nextCounts[blocks_per_row] = total_pixels_in_block;

  const int32_t max_left = image_size.width - params.scan_size.width;
  for (int32_t left = 0, block = 1; left <= max_left;
       left += params.scan_size.width) {
    curCounts[block++] = cuda_rect_count_brightness_range(
        image, rectangle_from_size((Point){left, 0}, params.scan_size), 0,
        abs_white_threshold);
  }

  int32_t max_top = image_size.height - params.scan_size.height;
  for (int32_t top = 0; top <= max_top; top += params.scan_size.height) {
    nextCounts[0] = cuda_rect_count_brightness_range(
        image,
        rectangle_from_size((Point){0, top + params.scan_step.vertical},
                            params.scan_size),
        0, abs_white_threshold);

    for (int32_t left = 0, block = 1; left <= max_left;
         left += params.scan_size.width) {

      nextCounts[block + 1] = cuda_rect_count_brightness_range(
          image,
          rectangle_from_size((Point){left + params.scan_size.width,
                                      top + params.scan_step.vertical},
                              params.scan_size),
          0, abs_white_threshold);

      uint64_t max = max3(
          nextCounts[block - 1], nextCounts[block + 1],
          max3(prevCounts[block - 1], prevCounts[block + 1], curCounts[block]));

      if ((((float)max) / (float)total_pixels_in_block) <= params.intensity) {
        wipe_rectangle(
            image, rectangle_from_size((Point){left, top}, params.scan_size),
            PIXEL_WHITE);
        count += curCounts[block];
        curCounts[block] = total_pixels_in_block;
      }

      block++;
    }

    uint64_t *tmpCounts;
    tmpCounts = prevCounts;
    prevCounts = curCounts;
    curCounts = nextCounts;
    nextCounts = tmpCounts;
  }

  av_free(count_buffers);
  verboseLog(VERBOSE_NORMAL, " deleted %" PRIu64 " pixels.\n", count);
}

static void noisefilter_cuda(Image image, uint64_t intensity,
                             uint8_t min_white_level) {
  if (image.frame == NULL) {
    return;
  }

  ensure_kernels_loaded();
  image_ensure_cuda(&image);
  ImageCudaState *st = image_cuda_state(image);
  if (st == NULL || st->dptr == 0) {
    errOutput("CUDA image state missing for noisefilter.");
  }

  const UnpaperCudaFormat fmt = cuda_format_from_av(image.frame->format);
  if (fmt == UNPAPER_CUDA_FMT_INVALID) {
    errOutput("CUDA noisefilter: unsupported pixel format.");
  }

  const int img_fmt = (int)fmt;
  const int w = image.frame->width;
  const int h = image.frame->height;

  void *params[] = {
      &st->dptr, &st->linesize, &img_fmt, &w, &h, &intensity, &min_white_level,
  };
  unpaper_cuda_launch_kernel(k_noisefilter, 1, 1, 1, 1, 1, 1, params);

  st->cuda_dirty = true;
  st->cpu_dirty = false;
}

static void grayfilter_cuda(Image image, GrayfilterParameters params) {
  if (image.frame == NULL) {
    return;
  }

  RectangleSize image_size = size_of_image(image);
  Point filter_origin = POINT_ORIGIN;

  verboseLog(VERBOSE_NORMAL, "gray-filter...");

  do {
    Rectangle area = rectangle_from_size(filter_origin, params.scan_size);
    unsigned long long local_count = cuda_rect_count_brightness_range(
        image, area, 0, image.abs_black_threshold);

    if (local_count == 0ull) {
      uint8_t lightness = cuda_inverse_lightness_rect(image, area);
      if (lightness < params.abs_threshold) {
        wipe_rectangle(image, area, PIXEL_WHITE);
      }
    }

    if (filter_origin.x < image_size.width) {
      filter_origin.x += params.scan_step.horizontal;
    } else {
      filter_origin.x = 0;
      filter_origin.y += params.scan_step.vertical;
    }
  } while (filter_origin.y <= image_size.height);

  verboseLog(VERBOSE_NORMAL, " deleted 0 pixels.\n");
}

static float detect_rotation_cuda(Image image, Rectangle mask,
                                  const DeskewParameters params) {
  (void)image;
  (void)mask;
  (void)params;
  cuda_unimplemented("detect_rotation");
  return 0.0f;
}

static void deskew_cuda(Image source, Rectangle mask, float radians,
                        Interpolation interpolate_type) {
  (void)source;
  (void)mask;
  (void)radians;
  (void)interpolate_type;
  cuda_unimplemented("deskew");
}

const ImageBackend backend_cuda = {
    .name = "cuda",

    .wipe_rectangle = wipe_rectangle_cuda,
    .copy_rectangle = copy_rectangle_cuda,
    .center_image = center_image_cuda,
    .stretch_and_replace = stretch_and_replace_cuda,
    .resize_and_replace = resize_and_replace_cuda,
    .flip_rotate_90 = flip_rotate_90_cuda,
    .mirror = mirror_cuda,
    .shift_image = shift_image_cuda,

    .apply_masks = apply_masks_cuda,
    .apply_wipes = apply_wipes_cuda,
    .apply_border = apply_border_cuda,
    .detect_masks = detect_masks_cuda,
    .align_mask = align_mask_cuda,
    .detect_border = detect_border_cuda,

    .blackfilter = blackfilter_cuda,
    .blurfilter = blurfilter_cuda,
    .noisefilter = noisefilter_cuda,
    .grayfilter = grayfilter_cuda,

    .detect_rotation = detect_rotation_cuda,
    .deskew = deskew_cuda,
};
