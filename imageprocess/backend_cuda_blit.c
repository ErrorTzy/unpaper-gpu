// SPDX-FileCopyrightText: 2025 The unpaper authors
//
// SPDX-License-Identifier: GPL-2.0-only

// CUDA backend: rectangle operations and geometric transforms

#include "imageprocess/backend_cuda_internal.h"

#include <stdlib.h>

#include <libavutil/frame.h>

#include "imageprocess/cuda_runtime.h"
#include "imageprocess/opencv_bridge.h"
#include "imageprocess/opencv_ops.h"
#include "lib/logging.h"
#include "lib/math_util.h"

void wipe_rectangle_cuda(Image image, Rectangle input_area, Pixel color) {
  Rectangle area = clip_rectangle(image, input_area);
  RectangleSize sz = size_of_rectangle(area);
  if (sz.width <= 0 || sz.height <= 0) {
    return;
  }

  image_ensure_cuda(&image);
  ImageCudaState *st = image_cuda_state(image);
  if (st == NULL || st->dptr == 0) {
    errOutput("CUDA image state missing for wipe_rectangle.");
  }

  const UnpaperCudaFormat fmt = cuda_format_from_av(image.frame->format);
  if (fmt == UNPAPER_CUDA_FMT_INVALID) {
    errOutput("CUDA wipe_rectangle: unsupported pixel format.");
  }

  int x0 = area.vertex[0].x;
  int y0 = area.vertex[0].y;
  int x1 = area.vertex[1].x;
  int y1 = area.vertex[1].y;

  // Try OpenCV path first (for non-mono formats)
  if (unpaper_opencv_wipe_rect(st->dptr, st->width, st->height, st->linesize,
                               (int)fmt, x0, y0, x1, y1, color.r, color.g,
                               color.b, NULL)) {
    st->cuda_dirty = true;
    st->cpu_dirty = false;
    return;
  }

  // Fall back to custom CUDA kernel for mono formats
  ensure_kernels_loaded();

  if (fmt == UNPAPER_CUDA_FMT_MONOWHITE || fmt == UNPAPER_CUDA_FMT_MONOBLACK) {
    uint8_t gray = pixel_grayscale(color);
    bool pixel_black = gray < image.abs_black_threshold;
    uint8_t bit_set = (fmt == UNPAPER_CUDA_FMT_MONOWHITE)
                          ? (pixel_black ? 1u : 0u)
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
    // This should not happen since OpenCV should handle all byte formats
    errOutput("CUDA wipe_rectangle: OpenCV failed for byte format.");
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

void copy_rectangle_cuda(Image source, Image target, Rectangle source_area,
                         Point target_coords) {
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

  image_ensure_cuda(&source);
  image_ensure_cuda(&target);

  ImageCudaState *src_st = image_cuda_state(source);
  ImageCudaState *dst_st = image_cuda_state(target);
  if (src_st == NULL || dst_st == NULL || src_st->dptr == 0 ||
      dst_st->dptr == 0) {
    errOutput("CUDA image state missing for copy_rectangle.");
  }

  UnpaperCudaFormat src_fmt = cuda_format_from_av(source.frame->format);
  UnpaperCudaFormat dst_fmt = cuda_format_from_av(target.frame->format);
  if (src_fmt == UNPAPER_CUDA_FMT_INVALID ||
      dst_fmt == UNPAPER_CUDA_FMT_INVALID) {
    errOutput("CUDA copy_rectangle: unsupported pixel format.");
  }

  // Try OpenCV path first (for same non-mono formats)
  if (unpaper_opencv_copy_rect(src_st->dptr, src_st->width, src_st->height,
                               src_st->linesize, (int)src_fmt, dst_st->dptr,
                               dst_st->width, dst_st->height, dst_st->linesize,
                               (int)dst_fmt, src_x0, src_y0, dst_x0, dst_y0, w,
                               h, NULL)) {
    dst_st->cuda_dirty = true;
    dst_st->cpu_dirty = false;
    return;
  }

  // Fall back to custom CUDA kernel for mono formats or format conversion
  ensure_kernels_loaded();

  if (dst_fmt == UNPAPER_CUDA_FMT_MONOWHITE ||
      dst_fmt == UNPAPER_CUDA_FMT_MONOBLACK) {
    uint8_t threshold = target.abs_black_threshold;
    void *params[] = {
        &src_st->dptr,     &src_st->linesize, &src_fmt, &dst_st->dptr,
        &dst_st->linesize, &dst_fmt,          &src_x0,  &src_y0,
        &dst_x0,           &dst_y0,           &w,       &h,
        &threshold,
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
        &src_st->dptr,     &src_st->linesize, &src_fmt, &dst_st->dptr,
        &dst_st->linesize, &dst_fmt,          &src_x0,  &src_y0,
        &dst_x0,           &dst_y0,           &w,       &h,
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

void center_image_cuda(Image source, Image target, Point target_origin,
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

  copy_rectangle_cuda(source, target,
                      rectangle_from_size(source_origin, source_size),
                      target_origin);
}

void flip_rotate_90_cuda(Image *pImage, RotationDirection direction) {
  if (pImage == NULL || pImage->frame == NULL) {
    return;
  }
  if (direction != ROTATE_CLOCKWISE && direction != ROTATE_ANTICLOCKWISE) {
    errOutput("invalid rotation direction.");
  }

  image_ensure_cuda(pImage);
  ImageCudaState *src_st = image_cuda_state(*pImage);
  if (src_st == NULL || src_st->dptr == 0) {
    errOutput("CUDA image state missing for flip_rotate_90.");
  }

  RectangleSize src_size = size_of_image(*pImage);
  Image newimage = create_compatible_image(
      *pImage,
      (RectangleSize){.width = src_size.height, .height = src_size.width},
      false);
  image_ensure_cuda(&newimage);
  ImageCudaState *dst_st = image_cuda_state(newimage);
  if (dst_st == NULL || dst_st->dptr == 0) {
    errOutput("CUDA image state missing for flip_rotate_90 output.");
  }

  UnpaperCudaFormat fmt = cuda_format_from_av(pImage->frame->format);
  if (fmt == UNPAPER_CUDA_FMT_INVALID) {
    errOutput("CUDA flip_rotate_90: unsupported pixel format.");
  }

  // Try OpenCV path first (for GRAY8 format only, as OpenCV transpose
  // doesn't support 2 or 3 byte element sizes)
  bool clockwise = (direction == ROTATE_CLOCKWISE);
  if (unpaper_opencv_rotate90(src_st->dptr, src_st->width, src_st->height,
                              src_st->linesize, dst_st->dptr, dst_st->linesize,
                              (int)fmt, clockwise, NULL)) {
    dst_st->cuda_dirty = true;
    dst_st->cpu_dirty = false;
    replace_image(pImage, &newimage);
    return;
  }

  // Fall back to custom CUDA kernel for mono, RGB24, and Y400A formats
  ensure_kernels_loaded();

  if (fmt == UNPAPER_CUDA_FMT_MONOWHITE || fmt == UNPAPER_CUDA_FMT_MONOBLACK) {
    int dir = (int)direction;
    void *params[] = {
        &src_st->dptr,   &src_st->linesize, &dst_st->dptr, &dst_st->linesize,
        &src_size.width, &src_size.height,  &dir,
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
    // Use custom kernel for RGB24 and Y400A (OpenCV transpose doesn't support)
    int dir = (int)direction;
    void *params[] = {
        &src_st->dptr, &src_st->linesize, &dst_st->dptr,    &dst_st->linesize,
        &fmt,          &src_size.width,   &src_size.height, &dir,
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

void mirror_cuda(Image image, Direction direction) {
  if (image.frame == NULL) {
    return;
  }

  image_ensure_cuda(&image);
  ImageCudaState *st = image_cuda_state(image);
  if (st == NULL || st->dptr == 0) {
    errOutput("CUDA image state missing for mirror.");
  }

  UnpaperCudaFormat fmt = cuda_format_from_av(image.frame->format);
  if (fmt == UNPAPER_CUDA_FMT_INVALID) {
    errOutput("CUDA mirror: unsupported pixel format.");
  }

  // Use stream-ordered allocation to avoid blocking other streams
  UnpaperCudaStream *stream = unpaper_cuda_get_current_stream();
  uint64_t new_dptr = unpaper_cuda_malloc_async(stream, st->bytes);

  // Try OpenCV path first (for non-mono formats)
  if (unpaper_opencv_mirror(st->dptr, new_dptr, st->width, st->height,
                            st->linesize, (int)fmt, direction.horizontal,
                            direction.vertical, stream)) {
    unpaper_cuda_free_async(stream, st->dptr);
    st->dptr = new_dptr;
    st->cuda_dirty = true;
    st->cpu_dirty = false;
    return;
  }

  // Fall back to custom CUDA kernel for mono formats
  ensure_kernels_loaded();
  unpaper_cuda_memcpy_d2d_async(stream, new_dptr, st->dptr, st->bytes);

  if (fmt == UNPAPER_CUDA_FMT_MONOWHITE || fmt == UNPAPER_CUDA_FMT_MONOBLACK) {
    int do_h = direction.horizontal ? 1 : 0;
    int do_v = direction.vertical ? 1 : 0;
    void *params[] = {
        &st->dptr,  &st->linesize, &new_dptr, &st->linesize,
        &st->width, &st->height,   &do_h,     &do_v,
    };
    const uint32_t block_x = 32;
    const uint32_t block_y = 8;
    const uint32_t bytes_per_row = (uint32_t)((st->width + 7) / 8);
    const uint32_t grid_x = (bytes_per_row + block_x - 1) / block_x;
    const uint32_t grid_y = ((uint32_t)st->height + block_y - 1) / block_y;
    unpaper_cuda_launch_kernel(k_mirror_mono, grid_x, grid_y, 1, block_x,
                               block_y, 1, params);
  } else {
    // This should not happen since OpenCV should handle all byte formats
    unpaper_cuda_free_async(stream, new_dptr);
    errOutput("CUDA mirror: OpenCV failed for byte format.");
  }

  unpaper_cuda_free_async(stream, st->dptr);
  st->dptr = new_dptr;
  st->cuda_dirty = true;
  st->cpu_dirty = false;
}

void shift_image_cuda(Image *pImage, Delta d) {
  if (pImage == NULL || pImage->frame == NULL) {
    return;
  }
  Image newimage =
      create_compatible_image(*pImage, size_of_image(*pImage), true);
  copy_rectangle_cuda(*pImage, newimage, full_image(*pImage),
                      shift_point(POINT_ORIGIN, d));
  replace_image(pImage, &newimage);
}

void stretch_and_replace_cuda(Image *pImage, RectangleSize size,
                              Interpolation interpolate_type) {
  if (pImage == NULL || pImage->frame == NULL) {
    return;
  }
  if (compare_sizes(size_of_image(*pImage), size) == 0) {
    return;
  }

  UnpaperCudaFormat fmt = cuda_format_from_av(pImage->frame->format);
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

  int interp = (int)interpolate_type;

  // Try OpenCV path first (supports GRAY8 and RGB24)
#ifdef UNPAPER_WITH_OPENCV
  if (unpaper_opencv_resize(src->dptr, src->width, src->height,
                            (size_t)src->linesize, dst->dptr, dst->width,
                            dst->height, (size_t)dst->linesize, (int)fmt,
                            interp, NULL)) {
    dst->cuda_dirty = true;
    dst->cpu_dirty = false;
    replace_image(pImage, &target);
    return;
  }
#endif

  // Fall back to custom CUDA kernels
  ensure_kernels_loaded();

  if (fmt == UNPAPER_CUDA_FMT_MONOWHITE || fmt == UNPAPER_CUDA_FMT_MONOBLACK) {
    void *params[] = {
        &src->dptr,   &src->linesize, &fmt,
        &dst->dptr,   &dst->linesize, &fmt,
        &src->width,  &src->height,   &dst->width,
        &dst->height, &interp,        &target.abs_black_threshold,
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
        &src->dptr,  &src->linesize, &dst->dptr,  &dst->linesize, &fmt,
        &src->width, &src->height,   &dst->width, &dst->height,   &interp,
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

void resize_and_replace_cuda(Image *pImage, RectangleSize size,
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
    stretch_size = (RectangleSize){
        size.width, (int32_t)(image_size.height * horizontal_ratio)};
  } else if (vertical_ratio < horizontal_ratio) {
    stretch_size = (RectangleSize){(int32_t)(image_size.width * vertical_ratio),
                                   size.height};
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
