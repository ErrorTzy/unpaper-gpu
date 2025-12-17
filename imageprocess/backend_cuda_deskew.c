// SPDX-FileCopyrightText: 2025 The unpaper authors
//
// SPDX-License-Identifier: GPL-2.0-only

// CUDA backend: rotation detection and deskew operations

#include "imageprocess/backend_cuda_internal.h"

#include <math.h>
#include <stdlib.h>

#include <libavutil/frame.h>
#include <libavutil/mem.h>

#include "imageprocess/cuda_runtime.h"
#include "imageprocess/opencv_bridge.h"
#include "imageprocess/opencv_ops.h"
#include "lib/logging.h"
#include "lib/math_util.h"

#define CUDA_MAX_ROTATION_SCAN_SIZE 10000

static float detect_edge_rotation_cuda(Image image, ImageCudaState *st,
                                       UnpaperCudaFormat fmt, Rectangle nmask,
                                       RectangleSize mask_size,
                                       const DeskewParameters params,
                                       const float *rotations,
                                       int rotations_count, Delta shift,
                                       int max_blackness_abs) {
  int deskew_scan_size = params.deskewScanSize;
  if (shift.vertical == 0) {
    if (deskew_scan_size == -1) {
      deskew_scan_size = mask_size.height;
    }
    deskew_scan_size =
        min3(deskew_scan_size, CUDA_MAX_ROTATION_SCAN_SIZE, mask_size.height);
  } else {
    if (deskew_scan_size == -1) {
      deskew_scan_size = mask_size.width;
    }
    deskew_scan_size =
        min3(deskew_scan_size, CUDA_MAX_ROTATION_SCAN_SIZE, mask_size.width);
  }

  if (deskew_scan_size <= 0 || rotations_count <= 0) {
    return 0.0f;
  }

  const int max_depth =
      (shift.vertical == 0) ? (mask_size.width / 2) : (mask_size.height / 2);
  if (max_depth <= 0) {
    return 0.0f;
  }

  const size_t coord_count = (size_t)rotations_count * (size_t)deskew_scan_size;
  int *base_x_h = av_malloc_array(coord_count, sizeof(int));
  int *base_y_h = av_malloc_array(coord_count, sizeof(int));
  if (base_x_h == NULL || base_y_h == NULL) {
    av_free(base_x_h);
    av_free(base_y_h);
    errOutput("unable to allocate rotation scan buffers.");
  }

  for (int ai = 0; ai < rotations_count; ai++) {
    const float rotation = rotations[ai];
    const float m = tanf(rotation);

    const int half = deskew_scan_size / 2;
    const int outer_offset = (int)(fabsf(m) * (float)half);

    float X = 0.0f;
    float Y = 0.0f;
    float stepX = 0.0f;
    float stepY = 0.0f;

    if (shift.vertical == 0) { // horizontal detection
      const int mid = mask_size.height / 2;
      const int side_offset = shift.horizontal > 0
                                  ? nmask.vertex[0].x - outer_offset
                                  : nmask.vertex[1].x + outer_offset;
      X = (float)side_offset + (float)half * m;
      Y = (float)nmask.vertex[0].y + (float)mid - (float)half;
      stepX = -m;
      stepY = 1.0f;
    } else { // vertical detection
      const int mid = mask_size.width / 2;
      const int side_offset = shift.vertical > 0
                                  ? nmask.vertex[0].x - outer_offset
                                  : nmask.vertex[1].x + outer_offset;
      X = (float)nmask.vertex[0].x + (float)mid - (float)half;
      Y = (float)side_offset - ((float)half * m);
      stepX = 1.0f;
      stepY = -m;
    }

    int *xrow = base_x_h + (size_t)ai * (size_t)deskew_scan_size;
    int *yrow = base_y_h + (size_t)ai * (size_t)deskew_scan_size;
    for (int li = 0; li < deskew_scan_size; li++) {
      xrow[li] = (int)X;
      yrow[li] = (int)Y;
      X += stepX;
      Y += stepY;
    }
  }

  const int src_w = image.frame->width;
  const int src_h = image.frame->height;
  const int shift_x = shift.horizontal;
  const int shift_y = shift.vertical;
  const int mask_x0 = nmask.vertex[0].x;
  const int mask_y0 = nmask.vertex[0].y;
  const int mask_x1 = nmask.vertex[1].x;
  const int mask_y1 = nmask.vertex[1].y;

  int *peaks_h = av_malloc_array((size_t)rotations_count, sizeof(int));
  if (peaks_h == NULL) {
    av_free(base_x_h);
    av_free(base_y_h);
    errOutput("unable to allocate peak buffer.");
  }

  // Get the current stream for allocations
  UnpaperCudaStream *stream = unpaper_cuda_get_current_stream();

  // Try OpenCV path first (downloads image once, processes on CPU)
#ifdef UNPAPER_WITH_OPENCV
  if (unpaper_opencv_detect_edge_rotation_peaks(
          st->dptr, src_w, src_h, (size_t)st->linesize, (int)fmt, base_x_h,
          base_y_h, deskew_scan_size, max_depth, shift_x, shift_y, mask_x0,
          mask_y0, mask_x1, mask_y1, max_blackness_abs, rotations_count, stream,
          peaks_h)) {
    int max_peak = 0;
    float detected_rotation = 0.0f;
    for (int i = 0; i < rotations_count; i++) {
      const int peak = peaks_h[i];
      if (peak > max_peak) {
        max_peak = peak;
        detected_rotation = rotations[i];
      }
    }

    av_free(base_x_h);
    av_free(base_y_h);
    av_free(peaks_h);
    return detected_rotation;
  }
#endif

  // Fallback to custom CUDA kernel
  ensure_kernels_loaded();

  // Use stream-ordered allocation to avoid blocking other streams
  const size_t coord_bytes = coord_count * sizeof(int);
  uint64_t base_x_d = unpaper_cuda_malloc_async(stream, coord_bytes);
  uint64_t base_y_d = unpaper_cuda_malloc_async(stream, coord_bytes);
  uint64_t peaks_d =
      unpaper_cuda_malloc_async(stream, (size_t)rotations_count * sizeof(int));

  unpaper_cuda_memcpy_h2d_async(stream, base_x_d, base_x_h, coord_bytes);
  unpaper_cuda_memcpy_h2d_async(stream, base_y_d, base_y_h, coord_bytes);

  const int src_fmt = (int)fmt;
  const int scan_size = deskew_scan_size;

  void *params_k[] = {
      &st->dptr,          &st->linesize, &src_fmt,   &src_w,     &src_h,
      &base_x_d,          &base_y_d,     &scan_size, &max_depth, &shift_x,
      &shift_y,           &mask_x0,      &mask_y0,   &mask_x1,   &mask_y1,
      &max_blackness_abs, &peaks_d,
  };

  unpaper_cuda_launch_kernel_on_stream(stream, k_detect_edge_rotation_peaks,
                                       (uint32_t)rotations_count, 1, 1, 256, 1,
                                       1, params_k);

  // Use async D2H + stream sync for stream-specific synchronization
  // Note: peaks_h is av_malloc'd, not pinned, so async D2H will be synchronous
  // anyway but using stream_synchronize_on ensures we only wait on this stream,
  // not all streams
  unpaper_cuda_memcpy_d2h_async(stream, peaks_h, peaks_d,
                                (size_t)rotations_count * sizeof(int));
  unpaper_cuda_stream_synchronize_on(stream);

  int max_peak = 0;
  float detected_rotation = 0.0f;
  for (int i = 0; i < rotations_count; i++) {
    const int peak = peaks_h[i];
    if (peak > max_peak) {
      max_peak = peak;
      detected_rotation = rotations[i];
    }
  }

  av_free(base_x_h);
  av_free(base_y_h);
  av_free(peaks_h);
  unpaper_cuda_free_async(stream, base_x_d);
  unpaper_cuda_free_async(stream, base_y_d);
  unpaper_cuda_free_async(stream, peaks_d);

  return detected_rotation;
}

float detect_rotation_cuda(Image image, Rectangle mask,
                           const DeskewParameters params) {
  if (image.frame == NULL) {
    return 0.0f;
  }

  ensure_kernels_loaded();
  image_ensure_cuda(&image);
  ImageCudaState *st = image_cuda_state(image);
  if (st == NULL || st->dptr == 0) {
    errOutput("CUDA image state missing for detect_rotation.");
  }

  const UnpaperCudaFormat fmt = cuda_format_from_av(image.frame->format);
  if (fmt == UNPAPER_CUDA_FMT_INVALID) {
    errOutput("CUDA detect_rotation: unsupported pixel format.");
  }

  Rectangle nmask = normalize_rectangle(mask);
  RectangleSize mask_size = size_of_rectangle(nmask);

  float rotations[8192];
  int rotations_count = 0;
  for (float rotation = 0.0f; rotation <= params.deskewScanRangeRad;
       rotation = (rotation >= 0.0f) ? -(rotation + params.deskewScanStepRad)
                                     : -rotation) {
    if (rotations_count >= (int)(sizeof(rotations) / sizeof(rotations[0]))) {
      errOutput("deskew-scan configuration is too large for CUDA backend.");
    }
    rotations[rotations_count++] = rotation;
  }

  const int max_blackness_abs =
      (int)(255.0f * (float)params.deskewScanSize * params.deskewScanDepth);

  float rotation[4];
  int count = 0;
  if (params.scan_edges.left) {
    rotation[count] = detect_edge_rotation_cuda(
        image, st, fmt, nmask, mask_size, params, rotations, rotations_count,
        DELTA_RIGHTWARD, max_blackness_abs);
    verboseLog(VERBOSE_NORMAL, "detected rotation left: [%d,%d,%d,%d]: %f\n",
               nmask.vertex[0].x, nmask.vertex[0].y, nmask.vertex[1].x,
               nmask.vertex[1].y, rotation[count]);
    count++;
  }
  if (params.scan_edges.top) {
    rotation[count] = -detect_edge_rotation_cuda(
        image, st, fmt, nmask, mask_size, params, rotations, rotations_count,
        DELTA_DOWNWARD, max_blackness_abs);
    verboseLog(VERBOSE_NORMAL, "detected rotation top: [%d,%d,%d,%d]: %f\n",
               nmask.vertex[0].x, nmask.vertex[0].y, nmask.vertex[1].x,
               nmask.vertex[1].y, rotation[count]);
    count++;
  }
  if (params.scan_edges.right) {
    rotation[count] = detect_edge_rotation_cuda(
        image, st, fmt, nmask, mask_size, params, rotations, rotations_count,
        DELTA_LEFTWARD, max_blackness_abs);
    verboseLog(VERBOSE_NORMAL, "detected rotation right: [%d,%d,%d,%d]: %f\n",
               nmask.vertex[0].x, nmask.vertex[0].y, nmask.vertex[1].x,
               nmask.vertex[1].y, rotation[count]);
    count++;
  }
  if (params.scan_edges.bottom) {
    rotation[count] = -detect_edge_rotation_cuda(
        image, st, fmt, nmask, mask_size, params, rotations, rotations_count,
        DELTA_UPWARD, max_blackness_abs);
    verboseLog(VERBOSE_NORMAL, "detected rotation bottom: [%d,%d,%d,%d]: %f\n",
               nmask.vertex[0].x, nmask.vertex[0].y, nmask.vertex[1].x,
               nmask.vertex[1].y, rotation[count]);
    count++;
  }

  if (count == 0) {
    return 0.0f;
  }

  float total = 0.0f;
  for (int i = 0; i < count; i++) {
    total += rotation[i];
  }
  const float average = total / (float)count;

  total = 0.0f;
  for (int i = 0; i < count; i++) {
    const float d = rotation[i] - average;
    total += d * d;
  }
  const float deviation = sqrtf(total);

  verboseLog(VERBOSE_NORMAL,
             "rotation average: %f  deviation: %f  rotation-scan-deviation "
             "(maximum): %f  [%d,%d,%d,%d]\n",
             average, deviation, params.deskewScanDeviationRad,
             nmask.vertex[0].x, nmask.vertex[0].y, nmask.vertex[1].x,
             nmask.vertex[1].y);

  if (deviation <= params.deskewScanDeviationRad) {
    return average;
  }

  verboseLog(VERBOSE_NONE, "out of deviation range - NO ROTATING\n");
  return 0.0f;
}

void deskew_cuda(Image source, Rectangle mask, float radians,
                 Interpolation interpolate_type) {
  if (source.frame == NULL) {
    return;
  }

  Rectangle nmask = normalize_rectangle(mask);
  RectangleSize out_size = size_of_rectangle(nmask);
  Image rotated = create_compatible_image(source, out_size, true);

  image_ensure_cuda(&source);
  image_ensure_cuda_alloc(&rotated);

  ImageCudaState *src_st = image_cuda_state(source);
  ImageCudaState *dst_st = image_cuda_state(rotated);
  if (src_st == NULL || src_st->dptr == 0) {
    errOutput("CUDA image state missing for deskew source.");
  }
  if (dst_st == NULL || dst_st->dptr == 0) {
    errOutput("CUDA image state missing for deskew target.");
  }

  const UnpaperCudaFormat fmt = cuda_format_from_av(source.frame->format);
  if (fmt == UNPAPER_CUDA_FMT_INVALID) {
    errOutput("CUDA deskew: unsupported pixel format.");
  }

  const Rectangle target_area = full_image(rotated);
  const FloatPoint source_center = center_of_rectangle(nmask);
  const FloatPoint target_center = center_of_rectangle(target_area);

  const float use_radians = -radians;
  const float sinval = sinf(use_radians);
  const float cosval = cosf(use_radians);

  const int src_w = source.frame->width;
  const int src_h = source.frame->height;
  const int dst_w = rotated.frame->width;
  const int dst_h = rotated.frame->height;
  const int interp = (int)interpolate_type;

  const float src_center_x = source_center.x;
  const float src_center_y = source_center.y;
  const float dst_center_x = target_center.x;
  const float dst_center_y = target_center.y;

  // Try OpenCV path first (supports GRAY8 and RGB24)
#ifdef UNPAPER_WITH_OPENCV
  if (unpaper_opencv_deskew(src_st->dptr, src_w, src_h,
                            (size_t)src_st->linesize, dst_st->dptr, dst_w,
                            dst_h, (size_t)dst_st->linesize, (int)fmt,
                            src_center_x, src_center_y, dst_center_x,
                            dst_center_y, cosval, sinval, interp, NULL)) {
    dst_st->cuda_dirty = true;
    dst_st->cpu_dirty = false;
    copy_rectangle(rotated, source, full_image(rotated), mask.vertex[0]);
    free_image(&rotated);
    return;
  }
#endif

  // Fall back to custom CUDA kernels
  ensure_kernels_loaded();

  const int bytespp = bytes_per_pixel_from_av(source.frame->format);
  if (bytespp != 0) {
    const int img_fmt = (int)fmt;
    void *params_k[] = {
        &src_st->dptr, &src_st->linesize,
        &dst_st->dptr, &dst_st->linesize,
        &img_fmt,      &src_w,
        &src_h,        &dst_w,
        &dst_h,        &src_center_x,
        &src_center_y, &dst_center_x,
        &dst_center_y, &cosval,
        &sinval,       &interp,
    };

    const uint32_t block_x = 16;
    const uint32_t block_y = 16;
    const uint32_t grid_x =
        (uint32_t)((dst_w + (int)block_x - 1) / (int)block_x);
    const uint32_t grid_y =
        (uint32_t)((dst_h + (int)block_y - 1) / (int)block_y);
    unpaper_cuda_launch_kernel(k_rotate_bytes, grid_x, grid_y, 1, block_x,
                               block_y, 1, params_k);
  } else {
    const int src_fmt = (int)fmt;
    const int dst_fmt = (int)fmt;
    const uint8_t abs_black_threshold = source.abs_black_threshold;

    void *params_k[] = {
        &src_st->dptr,
        &src_st->linesize,
        &src_fmt,
        &dst_st->dptr,
        &dst_st->linesize,
        &dst_fmt,
        &src_w,
        &src_h,
        &dst_w,
        &dst_h,
        &src_center_x,
        &src_center_y,
        &dst_center_x,
        &dst_center_y,
        &cosval,
        &sinval,
        &interp,
        &abs_black_threshold,
    };

    const int bytes_per_row = (dst_w + 7) / 8;
    const uint32_t block_x = 32;
    const uint32_t block_y = 8;
    const uint32_t grid_x =
        (uint32_t)((bytes_per_row + (int)block_x - 1) / (int)block_x);
    const uint32_t grid_y =
        (uint32_t)((dst_h + (int)block_y - 1) / (int)block_y);
    unpaper_cuda_launch_kernel(k_rotate_mono, grid_x, grid_y, 1, block_x,
                               block_y, 1, params_k);
  }

  dst_st->cuda_dirty = true;
  dst_st->cpu_dirty = false;

  copy_rectangle(rotated, source, full_image(rotated), mask.vertex[0]);
  free_image(&rotated);
}
