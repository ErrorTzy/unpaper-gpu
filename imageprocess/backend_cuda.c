// SPDX-FileCopyrightText: 2025 The unpaper authors
//
// SPDX-License-Identifier: GPL-2.0-only

// CUDA backend: infrastructure, shared helpers, and vtable

#include "imageprocess/backend.h"
#include "imageprocess/backend_cuda_internal.h"

#include <inttypes.h>
#include <math.h>
#include <pthread.h>
#include <stdatomic.h>
#include <stdlib.h>
#include <string.h>

#include <libavutil/frame.h>
#include <libavutil/pixfmt.h>

#include "imageprocess/cuda_runtime.h"
#include "lib/logging.h"
#include "lib/math_util.h"

extern const char unpaper_cuda_kernels_ptx[];

// Thread-safe kernel loading with double-checked locking
static _Atomic(void *) cuda_module = NULL;
static pthread_mutex_t cuda_module_mutex = PTHREAD_MUTEX_INITIALIZER;

// Kernel handles (exposed via internal header)
void *k_wipe_rect_bytes;
void *k_wipe_rect_mono;
void *k_copy_rect_to_bytes;
void *k_copy_rect_to_mono;
void *k_mirror_bytes;
void *k_mirror_mono;
void *k_rotate90_bytes;
void *k_rotate90_mono;
void *k_stretch_bytes;
void *k_stretch_mono;
void *k_count_brightness_range;
void *k_sum_lightness_rect;
void *k_sum_grayscale_rect;
void *k_sum_darkness_inverse_rect;
void *k_apply_masks_bytes;
void *k_apply_masks_mono;
void *k_noisefilter_build_labels;
void *k_noisefilter_propagate;
void *k_noisefilter_count;
void *k_noisefilter_apply;
void *k_noisefilter_apply_mask;
void *k_blackfilter_floodfill_rect;
void *k_blackfilter_scan_parallel;
void *k_blackfilter_wipe_regions;
void *k_detect_edge_rotation_peaks;
void *k_rotate_bytes;
void *k_rotate_mono;
void *k_batch_scan_grayscale_sum;
void *k_batch_scan_brightness_count;

void ensure_kernels_loaded(void) {
  // Fast path: already loaded
  if (atomic_load_explicit(&cuda_module, memory_order_acquire) != NULL) {
    return;
  }

  // Slow path: load with mutex protection
  pthread_mutex_lock(&cuda_module_mutex);

  // Double-check after acquiring lock
  if (atomic_load_explicit(&cuda_module, memory_order_relaxed) != NULL) {
    pthread_mutex_unlock(&cuda_module_mutex);
    return;
  }

  void *module = unpaper_cuda_module_load_ptx(unpaper_cuda_kernels_ptx);
  k_wipe_rect_bytes =
      unpaper_cuda_module_get_function(module, "unpaper_wipe_rect_bytes");
  k_wipe_rect_mono =
      unpaper_cuda_module_get_function(module, "unpaper_wipe_rect_mono");
  k_copy_rect_to_bytes =
      unpaper_cuda_module_get_function(module, "unpaper_copy_rect_to_bytes");
  k_copy_rect_to_mono =
      unpaper_cuda_module_get_function(module, "unpaper_copy_rect_to_mono");
  k_mirror_bytes =
      unpaper_cuda_module_get_function(module, "unpaper_mirror_bytes");
  k_mirror_mono =
      unpaper_cuda_module_get_function(module, "unpaper_mirror_mono");
  k_rotate90_bytes =
      unpaper_cuda_module_get_function(module, "unpaper_rotate90_bytes");
  k_rotate90_mono =
      unpaper_cuda_module_get_function(module, "unpaper_rotate90_mono");
  k_stretch_bytes =
      unpaper_cuda_module_get_function(module, "unpaper_stretch_bytes");
  k_stretch_mono =
      unpaper_cuda_module_get_function(module, "unpaper_stretch_mono");
  k_count_brightness_range = unpaper_cuda_module_get_function(
      module, "unpaper_count_brightness_range");
  k_sum_lightness_rect =
      unpaper_cuda_module_get_function(module, "unpaper_sum_lightness_rect");
  k_sum_grayscale_rect =
      unpaper_cuda_module_get_function(module, "unpaper_sum_grayscale_rect");
  k_sum_darkness_inverse_rect = unpaper_cuda_module_get_function(
      module, "unpaper_sum_darkness_inverse_rect");
  k_apply_masks_bytes =
      unpaper_cuda_module_get_function(module, "unpaper_apply_masks_bytes");
  k_apply_masks_mono =
      unpaper_cuda_module_get_function(module, "unpaper_apply_masks_mono");
  k_noisefilter_build_labels = unpaper_cuda_module_get_function(
      module, "unpaper_noisefilter_build_labels");
  k_noisefilter_propagate =
      unpaper_cuda_module_get_function(module, "unpaper_noisefilter_propagate");
  k_noisefilter_count =
      unpaper_cuda_module_get_function(module, "unpaper_noisefilter_count");
  k_noisefilter_apply =
      unpaper_cuda_module_get_function(module, "unpaper_noisefilter_apply");
  k_noisefilter_apply_mask = unpaper_cuda_module_get_function(
      module, "unpaper_noisefilter_apply_mask");
  k_blackfilter_floodfill_rect = unpaper_cuda_module_get_function(
      module, "unpaper_blackfilter_floodfill_rect");
  k_blackfilter_scan_parallel = unpaper_cuda_module_get_function(
      module, "unpaper_blackfilter_scan_parallel");
  k_blackfilter_wipe_regions = unpaper_cuda_module_get_function(
      module, "unpaper_blackfilter_wipe_regions");
  k_detect_edge_rotation_peaks = unpaper_cuda_module_get_function(
      module, "unpaper_detect_edge_rotation_peaks");
  k_rotate_bytes =
      unpaper_cuda_module_get_function(module, "unpaper_rotate_bytes");
  k_rotate_mono =
      unpaper_cuda_module_get_function(module, "unpaper_rotate_mono");
  k_batch_scan_grayscale_sum = unpaper_cuda_module_get_function(
      module, "unpaper_batch_scan_grayscale_sum");
  k_batch_scan_brightness_count = unpaper_cuda_module_get_function(
      module, "unpaper_batch_scan_brightness_count");

  // Publish the module pointer last (release semantics)
  atomic_store_explicit(&cuda_module, module, memory_order_release);
  pthread_mutex_unlock(&cuda_module_mutex);
}

UnpaperCudaFormat cuda_format_from_av(int fmt) {
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

int bytes_per_pixel_from_av(int fmt) {
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

ImageCudaState *image_cuda_state(Image image) {
  if (image.frame == NULL || image.frame->opaque_ref == NULL) {
    return NULL;
  }
  return (ImageCudaState *)image.frame->opaque_ref->data;
}

bool rect_empty(Rectangle area) {
  return (area.vertex[0].x > area.vertex[1].x) ||
         (area.vertex[0].y > area.vertex[1].y);
}

unsigned long long cuda_rect_count_brightness_range(Image image,
                                                    Rectangle input_area,
                                                    uint8_t min_brightness,
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

  // Use stream-ordered allocation to avoid blocking other streams
  UnpaperCudaStream *stream = unpaper_cuda_get_current_stream();
  uint64_t out_dptr =
      unpaper_cuda_malloc_async(stream, sizeof(unsigned long long));
  unpaper_cuda_memset_async(stream, out_dptr, 0, sizeof(unsigned long long));

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
      &st->dptr,
      &st->linesize,
      &src_fmt,
      &src_w,
      &src_h,
      &x0,
      &y0,
      &x1,
      &y1,
      &min_brightness,
      &max_brightness,
      &out_dptr,
  };
  unpaper_cuda_launch_kernel_on_stream(stream, k_count_brightness_range, grid_x,
                                       1, 1, block_x, 1, 1, params);

  // Use per-stream pinned memory for stream-specific D2H sync
  size_t pinned_capacity = 0;
  unsigned long long *out_pinned =
      (unsigned long long *)unpaper_cuda_stream_pinned_reserve(
          stream, sizeof(unsigned long long), &pinned_capacity);
  unsigned long long out_fallback = 0;
  unsigned long long *out_ptr =
      (out_pinned != NULL) ? out_pinned : &out_fallback;

  if (out_pinned != NULL) {
    unpaper_cuda_memcpy_d2h_async(stream, out_ptr, out_dptr, sizeof(*out_ptr));
    unpaper_cuda_stream_synchronize_on(stream);
  } else {
    unpaper_cuda_memcpy_d2h(out_ptr, out_dptr, sizeof(*out_ptr));
  }

  unpaper_cuda_free_async(stream, out_dptr);
  return *out_ptr;
}

unsigned long long cuda_rect_sum_lightness(Image image, Rectangle input_area) {
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

  // Use stream-ordered allocation to avoid blocking other streams
  UnpaperCudaStream *stream = unpaper_cuda_get_current_stream();
  uint64_t out_dptr =
      unpaper_cuda_malloc_async(stream, sizeof(unsigned long long));
  unpaper_cuda_memset_async(stream, out_dptr, 0, sizeof(unsigned long long));

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
      &st->dptr, &st->linesize, &src_fmt, &src_w, &src_h,
      &x0,       &y0,           &x1,      &y1,    &out_dptr,
  };
  unpaper_cuda_launch_kernel_on_stream(stream, k_sum_lightness_rect, grid_x, 1,
                                       1, block_x, 1, 1, params);

  // Use per-stream pinned memory for stream-specific D2H sync
  size_t pinned_capacity = 0;
  unsigned long long *out_pinned =
      (unsigned long long *)unpaper_cuda_stream_pinned_reserve(
          stream, sizeof(unsigned long long), &pinned_capacity);
  unsigned long long out_fallback = 0;
  unsigned long long *out_ptr =
      (out_pinned != NULL) ? out_pinned : &out_fallback;

  if (out_pinned != NULL) {
    unpaper_cuda_memcpy_d2h_async(stream, out_ptr, out_dptr, sizeof(*out_ptr));
    unpaper_cuda_stream_synchronize_on(stream);
  } else {
    unpaper_cuda_memcpy_d2h(out_ptr, out_dptr, sizeof(*out_ptr));
  }

  unpaper_cuda_free_async(stream, out_dptr);
  return *out_ptr;
}

unsigned long long cuda_rect_sum_darkness_inverse(Image image,
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
    errOutput("CUDA image state missing for sum_darkness_inverse.");
  }

  const UnpaperCudaFormat fmt = cuda_format_from_av(image.frame->format);
  if (fmt == UNPAPER_CUDA_FMT_INVALID) {
    errOutput("CUDA sum_darkness_inverse: unsupported pixel format.");
  }

  // Use stream-ordered allocation to avoid blocking other streams
  UnpaperCudaStream *stream = unpaper_cuda_get_current_stream();
  uint64_t out_dptr =
      unpaper_cuda_malloc_async(stream, sizeof(unsigned long long));
  unpaper_cuda_memset_async(stream, out_dptr, 0, sizeof(unsigned long long));

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
      &st->dptr, &st->linesize, &src_fmt, &src_w, &src_h,
      &x0,       &y0,           &x1,      &y1,    &out_dptr,
  };
  unpaper_cuda_launch_kernel_on_stream(stream, k_sum_darkness_inverse_rect,
                                       grid_x, 1, 1, block_x, 1, 1, params);

  // Use per-stream pinned memory for stream-specific D2H sync
  size_t pinned_capacity = 0;
  unsigned long long *out_pinned =
      (unsigned long long *)unpaper_cuda_stream_pinned_reserve(
          stream, sizeof(unsigned long long), &pinned_capacity);
  unsigned long long out_fallback = 0;
  unsigned long long *out_ptr =
      (out_pinned != NULL) ? out_pinned : &out_fallback;

  if (out_pinned != NULL) {
    unpaper_cuda_memcpy_d2h_async(stream, out_ptr, out_dptr, sizeof(*out_ptr));
    unpaper_cuda_stream_synchronize_on(stream);
  } else {
    unpaper_cuda_memcpy_d2h(out_ptr, out_dptr, sizeof(*out_ptr));
  }

  unpaper_cuda_free_async(stream, out_dptr);
  return *out_ptr;
}

uint8_t cuda_inverse_lightness_rect(Image image, Rectangle input_area) {
  Rectangle area = clip_rectangle(image, input_area);
  uint64_t count = count_pixels(area);
  if (count == 0) {
    return 0;
  }
  const unsigned long long sum = cuda_rect_sum_lightness(image, area);
  return (uint8_t)(sum / count);
}

uint8_t cuda_darkness_rect(Image image, Rectangle input_area) {
  Rectangle area = clip_rectangle(image, input_area);
  uint64_t count = count_pixels(area);
  if (count == 0) {
    return 0;
  }
  const unsigned long long sum = cuda_rect_sum_darkness_inverse(image, area);
  return (uint8_t)(0xFFu - (sum / count));
}

// Batched version of detect_edge_cuda that eliminates per-iteration syncs.
// Instead of syncing after each brightness computation, we:
// 1. Pre-compute ALL positions' brightness values in a single kernel launch
// 2. Single D2H transfer for all values
// 3. CPU iterates through values to find the edge
//
// This reduces syncs from O(n) to O(1) per edge detection.
uint32_t detect_edge_cuda(Image image, Point origin, Delta step,
                          int32_t scan_size, int32_t scan_depth,
                          float threshold) {
  const RectangleSize image_size = size_of_image(image);
  Rectangle scan_area;
  int max_positions;
  int rect_w, rect_h;

  // Setup scan area and compute max positions
  if (step.vertical == 0) {
    // Horizontal scanning (vertical border detection)
    if (scan_depth == -1) {
      scan_depth = image_size.height;
    }
    scan_area = rectangle_from_size(
        shift_point(origin, (Delta){-scan_size / 2, -scan_depth / 2}),
        (RectangleSize){scan_size, scan_depth});
    rect_w = scan_size;
    rect_h = scan_depth;
    // Max positions until we go outside image
    if (step.horizontal > 0) {
      max_positions =
          (image_size.width - scan_area.vertex[0].x) / step.horizontal + 1;
    } else {
      max_positions = (scan_area.vertex[1].x + 1) / (-step.horizontal) + 1;
    }
  } else if (step.horizontal == 0) {
    // Vertical scanning (horizontal border detection)
    if (scan_depth == -1) {
      scan_depth = image_size.width;
    }
    scan_area = rectangle_from_size(
        shift_point(origin, (Delta){-scan_depth / 2, -scan_size / 2}),
        (RectangleSize){scan_depth, scan_size});
    rect_w = scan_depth;
    rect_h = scan_size;
    // Max positions until we go outside image
    if (step.vertical > 0) {
      max_positions =
          (image_size.height - scan_area.vertex[0].y) / step.vertical + 1;
    } else {
      max_positions = (scan_area.vertex[1].y + 1) / (-step.vertical) + 1;
    }
  } else {
    errOutput("detect_edge_cuda() called with diagonal steps, impossible! "
              "(%" PRId32 ", %" PRId32 ")",
              step.horizontal, step.vertical);
    return 0;
  }

  // Clamp to reasonable maximum to avoid huge allocations
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
    errOutput("CUDA image state missing for detect_edge_cuda.");
  }

  const UnpaperCudaFormat fmt = cuda_format_from_av(image.frame->format);
  UnpaperCudaStream *stream = unpaper_cuda_get_current_stream();

  // Allocate GPU buffers for sums and counts
  const size_t buffer_size = (size_t)max_positions * sizeof(unsigned long long);
  uint64_t sums_dptr = unpaper_cuda_malloc_async(stream, buffer_size);
  uint64_t counts_dptr = unpaper_cuda_malloc_async(stream, buffer_size);

  // Launch batched kernel - one block per scan position
  const int src_fmt = (int)fmt;
  const int src_w = image.frame->width;
  const int src_h = image.frame->height;
  const int base_x0 = scan_area.vertex[0].x;
  const int base_y0 = scan_area.vertex[0].y;
  const int step_x = step.horizontal;
  const int step_y = step.vertical;

  void *params[] = {
      &st->dptr, &st->linesize,  &src_fmt,   &src_w,       &src_h,
      &base_x0,  &base_y0,       &rect_w,    &rect_h,      &step_x,
      &step_y,   &max_positions, &sums_dptr, &counts_dptr,
  };

  // One block per position, 256 threads per block for parallel reduction
  unpaper_cuda_launch_kernel_on_stream(stream, k_batch_scan_grayscale_sum,
                                       max_positions, 1, 1, 256, 1, 1, params);

  // Single D2H transfer for all results
  unsigned long long *sums_host = (unsigned long long *)malloc(buffer_size);
  unsigned long long *counts_host = (unsigned long long *)malloc(buffer_size);

  // Use async memcpy + single sync
  unpaper_cuda_memcpy_d2h_async(stream, sums_host, sums_dptr, buffer_size);
  unpaper_cuda_memcpy_d2h_async(stream, counts_host, counts_dptr, buffer_size);
  unpaper_cuda_stream_synchronize_on(stream);

  // Free GPU buffers
  unpaper_cuda_free_async(stream, sums_dptr);
  unpaper_cuda_free_async(stream, counts_dptr);

  // Now iterate on CPU to find edge (same logic as original)
  uint32_t total = 0;
  uint32_t count = 0;

  for (int i = 0; i < max_positions; i++) {
    uint8_t blackness;
    if (counts_host[i] == 0) {
      blackness = 0;
    } else {
      const unsigned long long avg = sums_host[i] / counts_host[i];
      blackness = (uint8_t)(0xFFu - (uint8_t)avg);
    }

    total += blackness;
    count++;

    // Check termination condition
    if (!((blackness >= ((threshold * total) / count)) && blackness != 0)) {
      break;
    }
  }

  free(sums_host);
  free(counts_host);

  return count;
}

// CUDA backend vtable - references functions from split modules
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
