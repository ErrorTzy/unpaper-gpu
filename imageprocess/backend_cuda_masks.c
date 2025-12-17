// SPDX-FileCopyrightText: 2025 The unpaper authors
//
// SPDX-License-Identifier: GPL-2.0-only

// CUDA backend: mask detection and border operations

#include "imageprocess/backend_cuda_internal.h"

#include <inttypes.h>
#include <stdlib.h>
#include <string.h>

#include <libavutil/frame.h>
#include <libavutil/mem.h>

#include "imageprocess/cuda_runtime.h"
#include "lib/logging.h"
#include "lib/math_util.h"

// Internal helper for mask detection using edge scanning
static bool detect_mask_cuda_internal(Image image, MaskDetectionParameters params,
                                      Point origin, Rectangle *mask) {
  const RectangleSize image_size = size_of_image(image);

  if (params.scan_direction.horizontal) {
    const uint32_t left_edge = detect_edge_cuda(
        image, origin, (Delta){-params.scan_step.horizontal, 0},
        params.scan_size.width, params.scan_depth.horizontal,
        params.scan_threshold.horizontal);
    const uint32_t right_edge =
        detect_edge_cuda(image, origin, (Delta){params.scan_step.horizontal, 0},
                         params.scan_size.width, params.scan_depth.horizontal,
                         params.scan_threshold.horizontal);

    mask->vertex[0].x = origin.x -
                        (params.scan_step.horizontal * (int32_t)left_edge) -
                        params.scan_size.width / 2;
    mask->vertex[1].x = origin.x +
                        (params.scan_step.horizontal * (int32_t)right_edge) +
                        params.scan_size.width / 2;
  } else {
    mask->vertex[0].x = 0;
    mask->vertex[1].x = image_size.width - 1;
  }

  if (params.scan_direction.vertical) {
    const uint32_t top_edge =
        detect_edge_cuda(image, origin, (Delta){0, -params.scan_step.vertical},
                         params.scan_size.height, params.scan_depth.vertical,
                         params.scan_threshold.vertical);
    const uint32_t bottom_edge =
        detect_edge_cuda(image, origin, (Delta){0, params.scan_step.vertical},
                         params.scan_size.height, params.scan_depth.vertical,
                         params.scan_threshold.vertical);

    mask->vertex[0].y = origin.y -
                        (params.scan_step.vertical * (int32_t)top_edge) -
                        params.scan_size.height / 2;
    mask->vertex[1].y = origin.y +
                        (params.scan_step.vertical * (int32_t)bottom_edge) +
                        params.scan_size.height / 2;
  } else {
    mask->vertex[0].y = 0;
    mask->vertex[1].y = image_size.height - 1;
  }

  const RectangleSize size = size_of_rectangle(*mask);
  bool success = true;

  if ((params.minimum_width != -1 && size.width < params.minimum_width) ||
      (params.maximum_width != -1 && size.width > params.maximum_width)) {
    verboseLog(VERBOSE_DEBUG, "mask width (%d) not within min/max (%d / %d)\n",
               size.width, params.minimum_width, params.maximum_width);
    mask->vertex[0].x = origin.x - params.maximum_width / 2;
    mask->vertex[1].x = origin.x + params.maximum_width / 2;
    success = false;
  }

  if ((params.minimum_height != -1 && size.height < params.minimum_height) ||
      (params.maximum_height != -1 && size.height > params.maximum_height)) {
    verboseLog(VERBOSE_DEBUG, "mask height (%d) not within min/max (%d / %d)\n",
               size.height, params.minimum_height, params.maximum_height);
    mask->vertex[0].y = origin.y - params.maximum_height / 2;
    mask->vertex[1].y = origin.y + params.maximum_height / 2;
    success = false;
  }

  return success;
}

// Batched version of detect_border_edge_cuda that eliminates per-iteration
// syncs. Instead of syncing after each pixel count, we:
// 1. Pre-compute ALL positions' dark pixel counts in a single kernel launch
// 2. Single D2H transfer for all counts
// 3. CPU iterates through counts to find the border
//
// This reduces syncs from O(n) to O(1) per border edge detection.
static uint32_t detect_border_edge_cuda_internal(Image image,
                                                 const Rectangle outside_mask,
                                                 Delta step, int32_t size,
                                                 int32_t threshold) {
  Rectangle area = outside_mask;
  const RectangleSize mask_size = size_of_rectangle(outside_mask);
  int32_t max_step;

  // Setup initial scan area (modifies area rectangle)
  if (step.vertical == 0) {
    // Horizontal scan
    if (step.horizontal > 0) {
      area.vertex[1].x = outside_mask.vertex[0].x + size;
    } else {
      area.vertex[0].x = outside_mask.vertex[1].x - size;
    }
    max_step = mask_size.width;
  } else {
    // Vertical scan
    if (step.vertical > 0) {
      area.vertex[1].y = outside_mask.vertex[0].y + size;
    } else {
      area.vertex[0].y = outside_mask.vertex[1].y - size;
    }
    max_step = mask_size.height;
  }

  // Compute actual rectangle dimensions from modified area
  const int rect_w = area.vertex[1].x - area.vertex[0].x + 1;
  const int rect_h = area.vertex[1].y - area.vertex[0].y + 1;

  // Compute step magnitude and number of positions
  int32_t step_magnitude = step.horizontal + step.vertical;
  if (step_magnitude < 0) {
    step_magnitude = -step_magnitude;
  }
  if (step_magnitude == 0) {
    return 0;
  }

  int max_positions = (max_step / step_magnitude) + 1;
  if (max_positions > 2000) {
    max_positions = 2000;
  }
  if (max_positions < 1) {
    max_positions = 1;
  }

  ensure_kernels_loaded();
  image_ensure_cuda((Image *)&image);
  ImageCudaState *st = image_cuda_state(image);
  if (st == NULL || st->dptr == 0) {
    errOutput("CUDA image state missing for detect_border_edge_cuda.");
  }

  const UnpaperCudaFormat fmt = cuda_format_from_av(image.frame->format);
  UnpaperCudaStream *stream = unpaper_cuda_get_current_stream();

  // Allocate GPU buffer for counts
  const size_t buffer_size = (size_t)max_positions * sizeof(unsigned long long);
  uint64_t counts_dptr = unpaper_cuda_malloc_async(stream, buffer_size);

  // Launch batched kernel - one block per scan position
  const int src_fmt = (int)fmt;
  const int src_w = image.frame->width;
  const int src_h = image.frame->height;
  const int base_x0 = area.vertex[0].x;
  const int base_y0 = area.vertex[0].y;
  const int step_x = step.horizontal;
  const int step_y = step.vertical;
  const uint8_t min_brightness = 0;
  const uint8_t max_brightness = image.abs_black_threshold;

  void *params[] = {
      &st->dptr, &st->linesize,  &src_fmt,        &src_w,          &src_h,
      &base_x0,  &base_y0,       &rect_w,         &rect_h,         &step_x,
      &step_y,   &max_positions, &min_brightness, &max_brightness, &counts_dptr,
  };

  // One block per position, 256 threads per block for parallel reduction
  unpaper_cuda_launch_kernel_on_stream(stream, k_batch_scan_brightness_count,
                                       max_positions, 1, 1, 256, 1, 1, params);

  // Single D2H transfer for all results
  unsigned long long *counts_host = (unsigned long long *)malloc(buffer_size);

  unpaper_cuda_memcpy_d2h_async(stream, counts_host, counts_dptr, buffer_size);
  unpaper_cuda_stream_synchronize_on(stream);

  // Free GPU buffer
  unpaper_cuda_free_async(stream, counts_dptr);

  // Now iterate on CPU to find border (same logic as original)
  uint32_t result = 0;
  for (int i = 0; i < max_positions && result < (uint32_t)max_step; i++) {
    if (counts_host[i] >= (unsigned long long)threshold) {
      free(counts_host);
      return result;
    }
    result += (uint32_t)step_magnitude;
  }

  free(counts_host);
  return 0;
}

void apply_masks_cuda(Image image, const Rectangle masks[], size_t masks_count,
                      Pixel color) {
  if (masks_count == 0 || image.frame == NULL) {
    return;
  }

  if (masks_count > (size_t)INT32_MAX) {
    errOutput("apply_masks CUDA: too many masks.");
  }

  ensure_kernels_loaded();
  image_ensure_cuda(&image);
  ImageCudaState *st = image_cuda_state(image);
  if (st == NULL || st->dptr == 0) {
    errOutput("CUDA image state missing for apply_masks.");
  }

  const UnpaperCudaFormat fmt = cuda_format_from_av(image.frame->format);
  if (fmt == UNPAPER_CUDA_FMT_INVALID) {
    errOutput("CUDA apply_masks: unsupported pixel format.");
  }

  const int rect_count = (int)masks_count;
  const size_t rect_bytes = (size_t)rect_count * 4u * sizeof(int32_t);
  int32_t *rects = av_malloc_array((size_t)rect_count * 4u, sizeof(int32_t));
  if (rects == NULL) {
    errOutput("apply_masks CUDA: allocation failed.");
  }
  for (int i = 0; i < rect_count; i++) {
    rects[i * 4 + 0] = masks[i].vertex[0].x;
    rects[i * 4 + 1] = masks[i].vertex[0].y;
    rects[i * 4 + 2] = masks[i].vertex[1].x;
    rects[i * 4 + 3] = masks[i].vertex[1].y;
  }

  // Use stream-ordered allocation to avoid blocking other streams
  UnpaperCudaStream *stream = unpaper_cuda_get_current_stream();
  uint64_t rects_dptr = unpaper_cuda_malloc_async(stream, rect_bytes);
  unpaper_cuda_memcpy_h2d_async(stream, rects_dptr, rects, rect_bytes);
  av_free(rects);

  const int img_fmt = (int)fmt;
  const int img_w = image.frame->width;
  const int img_h = image.frame->height;
  const uint8_t r = color.r;
  const uint8_t g = color.g;
  const uint8_t b = color.b;

  if (fmt == UNPAPER_CUDA_FMT_MONOWHITE || fmt == UNPAPER_CUDA_FMT_MONOBLACK) {
    const uint8_t gray = pixel_grayscale(color);
    const bool pixel_black = gray < image.abs_black_threshold;
    const uint8_t bit_value = (fmt == UNPAPER_CUDA_FMT_MONOWHITE)
                                  ? (pixel_black ? 1u : 0u)
                                  : (pixel_black ? 0u : 1u);

    void *params[] = {
        &st->dptr, &st->linesize, &img_fmt,    &img_w,
        &img_h,    &rects_dptr,   &rect_count, &bit_value,
    };
    const uint32_t block_x = 32;
    const uint32_t block_y = 8;
    const uint32_t bytes_span = (uint32_t)((img_w + 7) / 8);
    const uint32_t grid_x = (bytes_span + block_x - 1) / block_x;
    const uint32_t grid_y = ((uint32_t)img_h + block_y - 1) / block_y;
    unpaper_cuda_launch_kernel(k_apply_masks_mono, grid_x, grid_y, 1, block_x,
                               block_y, 1, params);
  } else {
    void *params[] = {
        &st->dptr,   &st->linesize, &img_fmt, &img_w, &img_h,
        &rects_dptr, &rect_count,   &r,       &g,     &b,
    };
    const uint32_t block_x = 16;
    const uint32_t block_y = 16;
    const uint32_t grid_x = ((uint32_t)img_w + block_x - 1) / block_x;
    const uint32_t grid_y = ((uint32_t)img_h + block_y - 1) / block_y;
    unpaper_cuda_launch_kernel(k_apply_masks_bytes, grid_x, grid_y, 1, block_x,
                               block_y, 1, params);
  }

  unpaper_cuda_free_async(stream, rects_dptr);

  st->cuda_dirty = true;
  st->cpu_dirty = false;
}

void apply_wipes_cuda(Image image, Wipes wipes, Pixel color) {
  for (size_t i = 0; i < wipes.count; i++) {
    wipe_rectangle_cuda(image, wipes.areas[i], color);

    verboseLog(VERBOSE_MORE,
               "wipe [%" PRId32 ",%" PRId32 ",%" PRId32 ",%" PRId32 "]\n",
               wipes.areas[i].vertex[0].x, wipes.areas[i].vertex[0].y,
               wipes.areas[i].vertex[1].x, wipes.areas[i].vertex[1].y);
  }
}

void apply_border_cuda(Image image, const Border border, Pixel color) {
  if (memcmp(&border, &BORDER_NULL, sizeof(BORDER_NULL)) == 0) {
    return;
  }

  RectangleSize image_size = size_of_image(image);
  Rectangle mask = {{
      {border.left, border.top},
      {image_size.width - border.right - 1,
       image_size.height - border.bottom - 1},
  }};
  verboseLog(VERBOSE_NORMAL, "applying border (%d,%d,%d,%d) [%d,%d,%d,%d]\n",
             border.left, border.top, border.right, border.bottom,
             mask.vertex[0].x, mask.vertex[0].y, mask.vertex[1].x,
             mask.vertex[1].y);
  apply_masks_cuda(image, &mask, 1, color);
}

size_t detect_masks_cuda(Image image, MaskDetectionParameters params,
                         const Point points[], size_t points_count,
                         Rectangle masks[]) {
  if (image.frame == NULL) {
    return 0;
  }

  if (!params.scan_direction.horizontal && !params.scan_direction.vertical) {
    return 0;
  }

  static const Rectangle invalid_mask = {{{-1, -1}, {-1, -1}}};

  size_t masks_count = 0;
  for (size_t i = 0; i < points_count; i++) {
    const bool valid =
        detect_mask_cuda_internal(image, params, points[i], &masks[i]);

    if (memcmp(&masks[i], &invalid_mask, sizeof(invalid_mask)) != 0) {
      masks_count++;

      verboseLog(VERBOSE_NORMAL, "auto-masking (%d,%d): %d,%d,%d,%d%s\n",
                 points[i].x, points[i].y, masks[i].vertex[0].x,
                 masks[i].vertex[0].y, masks[i].vertex[1].x,
                 masks[i].vertex[1].y,
                 valid ? "" : " (invalid detection, using full page size)");
    } else {
      verboseLog(VERBOSE_NORMAL, "auto-masking (%d,%d): NO MASK FOUND\n",
                 points[i].x, points[i].y);
    }
  }

  return masks_count;
}

void align_mask_cuda(Image image, const Rectangle inside_area,
                     const Rectangle outside, MaskAlignmentParameters params) {
  const RectangleSize inside_size = size_of_rectangle(inside_area);

  Point target;

  if (params.alignment.left) {
    target.x = outside.vertex[0].x + params.margin.horizontal;
  } else if (params.alignment.right) {
    target.x =
        outside.vertex[1].x - inside_size.width - params.margin.horizontal;
  } else {
    target.x =
        (outside.vertex[0].x + outside.vertex[1].x - inside_size.width) / 2;
  }
  if (params.alignment.top) {
    target.y = outside.vertex[0].y + params.margin.vertical;
  } else if (params.alignment.bottom) {
    target.y =
        outside.vertex[1].y - inside_size.height - params.margin.vertical;
  } else {
    target.y =
        (outside.vertex[0].y + outside.vertex[1].y - inside_size.height) / 2;
  }

  verboseLog(VERBOSE_NORMAL, "aligning mask [%d,%d,%d,%d] (%d,%d): %d, %d\n",
             inside_area.vertex[0].x, inside_area.vertex[0].y,
             inside_area.vertex[1].x, inside_area.vertex[1].y, target.x,
             target.y, target.x - inside_area.vertex[0].x,
             target.y - inside_area.vertex[0].y);

  Image newimage = create_compatible_image(image, inside_size, true);
  copy_rectangle_cuda(image, newimage, inside_area, POINT_ORIGIN);
  wipe_rectangle_cuda(image, inside_area, image.background);
  copy_rectangle_cuda(newimage, image, full_image(newimage), target);
  free_image(&newimage);
}

Border detect_border_cuda(Image image, BorderScanParameters params,
                          const Rectangle outside_mask) {
  RectangleSize image_size = size_of_image(image);

  Border border = {
      .left = outside_mask.vertex[0].x,
      .top = outside_mask.vertex[0].y,
      .right = image_size.width - outside_mask.vertex[1].x,
      .bottom = image_size.height - outside_mask.vertex[1].y,
  };

  if (params.scan_direction.horizontal) {
    border.left += detect_border_edge_cuda_internal(
        image, outside_mask, (Delta){params.scan_step.horizontal, 0},
        params.scan_size.width, params.scan_threshold.horizontal);
    border.right += detect_border_edge_cuda_internal(
        image, outside_mask, (Delta){-params.scan_step.horizontal, 0},
        params.scan_size.width, params.scan_threshold.horizontal);
  }
  if (params.scan_direction.vertical) {
    border.top += detect_border_edge_cuda_internal(
        image, outside_mask, (Delta){0, params.scan_step.vertical},
        params.scan_size.height, params.scan_threshold.vertical);
    border.bottom += detect_border_edge_cuda_internal(
        image, outside_mask, (Delta){0, -params.scan_step.vertical},
        params.scan_size.height, params.scan_threshold.vertical);
  }

  verboseLog(VERBOSE_NORMAL,
             "border detected: (%d,%d,%d,%d) in [%d,%d,%d,%d]\n", border.left,
             border.top, border.right, border.bottom, outside_mask.vertex[0].x,
             outside_mask.vertex[0].y, outside_mask.vertex[1].x,
             outside_mask.vertex[1].y);

  return border;
}
