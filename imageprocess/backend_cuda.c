// SPDX-FileCopyrightText: 2025 The unpaper authors
//
// SPDX-License-Identifier: GPL-2.0-only

#include "imageprocess/backend.h"

#include <inttypes.h>
#include <math.h>
#include <pthread.h>
#include <stdatomic.h>
#include <string.h>

#include <libavutil/frame.h>
#include <libavutil/mem.h>
#include <libavutil/pixfmt.h>

#include "imageprocess/cuda_kernels_format.h"
#include "imageprocess/cuda_mempool.h"
#include "imageprocess/cuda_runtime.h"
#include "imageprocess/npp_integral.h"
#include "imageprocess/npp_wrapper.h"
#include "imageprocess/opencv_bridge.h"
#include "imageprocess/opencv_ops.h"
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

#define CUDA_MAX_ROTATION_SCAN_SIZE 10000

extern const char unpaper_cuda_kernels_ptx[];

// Thread-safe kernel loading with double-checked locking
static _Atomic(void *) cuda_module = NULL;
static pthread_mutex_t cuda_module_mutex = PTHREAD_MUTEX_INITIALIZER;
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
static void *k_sum_grayscale_rect;
static void *k_sum_darkness_inverse_rect;
static void *k_apply_masks_bytes;
static void *k_apply_masks_mono;
static void *k_noisefilter_build_labels;
static void *k_noisefilter_propagate;
static void *k_noisefilter_count;
static void *k_noisefilter_apply;
static void *k_noisefilter_apply_mask;
static void *k_blackfilter_floodfill_rect;
static void *k_blackfilter_scan_parallel;
static void *k_blackfilter_wipe_regions;
static void *k_detect_edge_rotation_peaks;
static void *k_rotate_bytes;
static void *k_rotate_mono;
static void *k_batch_scan_grayscale_sum;
static void *k_batch_scan_brightness_count;

static void ensure_kernels_loaded(void) {
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

static unsigned long long
cuda_rect_count_brightness_range(Image image, Rectangle input_area,
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
  uint64_t out_dptr = unpaper_cuda_malloc_async(stream, sizeof(unsigned long long));
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
  unsigned long long *out_pinned = (unsigned long long *)unpaper_cuda_stream_pinned_reserve(
      stream, sizeof(unsigned long long), &pinned_capacity);
  unsigned long long out_fallback = 0;
  unsigned long long *out_ptr = (out_pinned != NULL) ? out_pinned : &out_fallback;

  if (out_pinned != NULL) {
    unpaper_cuda_memcpy_d2h_async(stream, out_ptr, out_dptr, sizeof(*out_ptr));
    unpaper_cuda_stream_synchronize_on(stream);
  } else {
    unpaper_cuda_memcpy_d2h(out_ptr, out_dptr, sizeof(*out_ptr));
  }

  unpaper_cuda_free_async(stream, out_dptr);
  return *out_ptr;
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

  // Use stream-ordered allocation to avoid blocking other streams
  UnpaperCudaStream *stream = unpaper_cuda_get_current_stream();
  uint64_t out_dptr = unpaper_cuda_malloc_async(stream, sizeof(unsigned long long));
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
  unsigned long long *out_pinned = (unsigned long long *)unpaper_cuda_stream_pinned_reserve(
      stream, sizeof(unsigned long long), &pinned_capacity);
  unsigned long long out_fallback = 0;
  unsigned long long *out_ptr = (out_pinned != NULL) ? out_pinned : &out_fallback;

  if (out_pinned != NULL) {
    unpaper_cuda_memcpy_d2h_async(stream, out_ptr, out_dptr, sizeof(*out_ptr));
    unpaper_cuda_stream_synchronize_on(stream);
  } else {
    unpaper_cuda_memcpy_d2h(out_ptr, out_dptr, sizeof(*out_ptr));
  }

  unpaper_cuda_free_async(stream, out_dptr);
  return *out_ptr;
}

// Batched version of detect_edge_cuda that eliminates per-iteration syncs.
// Instead of syncing after each brightness computation, we:
// 1. Pre-compute ALL positions' brightness values in a single kernel launch
// 2. Single D2H transfer for all values
// 3. CPU iterates through values to find the edge
//
// This reduces syncs from O(n) to O(1) per edge detection.
static uint32_t detect_edge_cuda(Image image, Point origin, Delta step,
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
      max_positions = (image_size.width - scan_area.vertex[0].x) / step.horizontal + 1;
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
      max_positions = (image_size.height - scan_area.vertex[0].y) / step.vertical + 1;
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
      &st->dptr,      &st->linesize,  &src_fmt,       &src_w,    &src_h,
      &base_x0,       &base_y0,       &rect_w,        &rect_h,   &step_x,
      &step_y,        &max_positions, &sums_dptr,     &counts_dptr,
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

static bool detect_mask_cuda(Image image, MaskDetectionParameters params,
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

// Batched version of detect_border_edge_cuda that eliminates per-iteration syncs.
// Instead of syncing after each pixel count, we:
// 1. Pre-compute ALL positions' dark pixel counts in a single kernel launch
// 2. Single D2H transfer for all counts
// 3. CPU iterates through counts to find the border
//
// This reduces syncs from O(n) to O(1) per border edge detection.
static uint32_t detect_border_edge_cuda(Image image,
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
      &st->dptr,        &st->linesize, &src_fmt,        &src_w,
      &src_h,           &base_x0,      &base_y0,        &rect_w,
      &rect_h,          &step_x,       &step_y,         &max_positions,
      &min_brightness,  &max_brightness, &counts_dptr,
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

  // Use stream-ordered allocation to avoid blocking other streams
  UnpaperCudaStream *stream = unpaper_cuda_get_current_stream();
  uint64_t out_dptr = unpaper_cuda_malloc_async(stream, sizeof(unsigned long long));
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
  unsigned long long *out_pinned = (unsigned long long *)unpaper_cuda_stream_pinned_reserve(
      stream, sizeof(unsigned long long), &pinned_capacity);
  unsigned long long out_fallback = 0;
  unsigned long long *out_ptr = (out_pinned != NULL) ? out_pinned : &out_fallback;

  if (out_pinned != NULL) {
    unpaper_cuda_memcpy_d2h_async(stream, out_ptr, out_dptr, sizeof(*out_ptr));
    unpaper_cuda_stream_synchronize_on(stream);
  } else {
    unpaper_cuda_memcpy_d2h(out_ptr, out_dptr, sizeof(*out_ptr));
  }

  unpaper_cuda_free_async(stream, out_dptr);
  return *out_ptr;
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

          const UnpaperCudaFormat fmt =
              cuda_format_from_av(image.frame->format);
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
                &st->dptr,  &st->linesize, &img_fmt,   &w,  &h,
                &x0,        &y0,           &x1,        &y1, &mask_max,
                &intensity, &stack_dptr,   &stack_cap,
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

static void wipe_rectangle_cuda(Image image, Rectangle input_area,
                                Pixel color) {
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

  const int x0 = area.vertex[0].x;
  const int y0 = area.vertex[0].y;
  const int x1 = area.vertex[1].x;
  const int y1 = area.vertex[1].y;

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
    const uint8_t gray = pixel_grayscale(color);
    const bool pixel_black = gray < image.abs_black_threshold;
    const uint8_t bit_set = (fmt == UNPAPER_CUDA_FMT_MONOWHITE)
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
    const uint8_t threshold = target.abs_black_threshold;
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

  copy_rectangle_cuda(source, target,
                      rectangle_from_size(source_origin, source_size),
                      target_origin);
}

static void flip_rotate_90_cuda(Image *pImage, RotationDirection direction) {
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

  const UnpaperCudaFormat fmt = cuda_format_from_av(pImage->frame->format);
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
    const int dir = (int)direction;
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
    const int dir = (int)direction;
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

static void mirror_cuda(Image image, Direction direction) {
  if (image.frame == NULL) {
    return;
  }

  image_ensure_cuda(&image);
  ImageCudaState *st = image_cuda_state(image);
  if (st == NULL || st->dptr == 0) {
    errOutput("CUDA image state missing for mirror.");
  }

  const UnpaperCudaFormat fmt = cuda_format_from_av(image.frame->format);
  if (fmt == UNPAPER_CUDA_FMT_INVALID) {
    errOutput("CUDA mirror: unsupported pixel format.");
  }

  // Use stream-ordered allocation to avoid blocking other streams
  UnpaperCudaStream *stream = unpaper_cuda_get_current_stream();
  const uint64_t new_dptr = unpaper_cuda_malloc_async(stream, st->bytes);

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
    const int do_h = direction.horizontal ? 1 : 0;
    const int do_v = direction.vertical ? 1 : 0;
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

static void apply_masks_cuda(Image image, const Rectangle masks[],
                             size_t masks_count, Pixel color) {
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

static void apply_wipes_cuda(Image image, Wipes wipes, Pixel color) {
  for (size_t i = 0; i < wipes.count; i++) {
    wipe_rectangle_cuda(image, wipes.areas[i], color);

    verboseLog(VERBOSE_MORE,
               "wipe [%" PRId32 ",%" PRId32 ",%" PRId32 ",%" PRId32 "]\n",
               wipes.areas[i].vertex[0].x, wipes.areas[i].vertex[0].y,
               wipes.areas[i].vertex[1].x, wipes.areas[i].vertex[1].y);
  }
}

static void apply_border_cuda(Image image, const Border border, Pixel color) {
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

static size_t detect_masks_cuda(Image image, MaskDetectionParameters params,
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
    const bool valid = detect_mask_cuda(image, params, points[i], &masks[i]);

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

static void align_mask_cuda(Image image, const Rectangle inside_area,
                            const Rectangle outside,
                            MaskAlignmentParameters params) {
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

static Border detect_border_cuda(Image image, BorderScanParameters params,
                                 const Rectangle outside_mask) {
  RectangleSize image_size = size_of_image(image);

  Border border = {
      .left = outside_mask.vertex[0].x,
      .top = outside_mask.vertex[0].y,
      .right = image_size.width - outside_mask.vertex[1].x,
      .bottom = image_size.height - outside_mask.vertex[1].y,
  };

  if (params.scan_direction.horizontal) {
    border.left += detect_border_edge_cuda(
        image, outside_mask, (Delta){params.scan_step.horizontal, 0},
        params.scan_size.width, params.scan_threshold.horizontal);
    border.right += detect_border_edge_cuda(
        image, outside_mask, (Delta){-params.scan_step.horizontal, 0},
        params.scan_size.width, params.scan_threshold.horizontal);
  }
  if (params.scan_direction.vertical) {
    border.top += detect_border_edge_cuda(
        image, outside_mask, (Delta){0, params.scan_step.vertical},
        params.scan_size.height, params.scan_threshold.vertical);
    border.bottom += detect_border_edge_cuda(
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

// Parallel blackfilter using integral image + GPU parallel scan/wipe
// This replaces the sequential flood-fill approach with fully parallel processing:
// 1. Build grayscale integral image on GPU (NPP)
// 2. Parallel scan all stripe positions to find dark blocks
// 3. Parallel wipe all dark pixels in detected regions
//
// Note: This approach wipes ALL dark pixels in detected regions, not just
// connected ones like the original flood-fill. For document scanning cleanup,
// this produces equivalent results and is much faster.
static bool blackfilter_cuda_parallel(Image image, BlackfilterParameters params,
                                      UnpaperCudaStream *stream) {
  if (image.frame == NULL) {
    return false;
  }

  const int w = image.frame->width;
  const int h = image.frame->height;
  if (w <= 0 || h <= 0) {
    return true; // Nothing to do
  }

  // Check kernel availability
  if (k_blackfilter_scan_parallel == NULL ||
      k_blackfilter_wipe_regions == NULL) {
    return false;
  }

  // Initialize NPP if needed
  if (!unpaper_npp_init()) {
    return false;
  }

  ImageCudaState *st = image_cuda_state(image);
  if (st == NULL || st->dptr == 0) {
    return false;
  }

  const UnpaperCudaFormat fmt = cuda_format_from_av(image.frame->format);
  if (fmt == UNPAPER_CUDA_FMT_INVALID || fmt == UNPAPER_CUDA_FMT_MONOWHITE ||
      fmt == UNPAPER_CUDA_FMT_MONOBLACK) {
    // Mono formats need special handling, fall back to sequential
    return false;
  }

  // For non-GRAY8 formats, convert to grayscale for integral computation
  uint64_t gray_device = 0;
  size_t gray_pitch = 0;
  bool gray_allocated = false;

  if (fmt == UNPAPER_CUDA_FMT_GRAY8) {
    gray_device = st->dptr;
    gray_pitch = (size_t)st->linesize;
  } else if (fmt == UNPAPER_CUDA_FMT_RGB24) {
    // RGB24: Convert to grayscale using OpenCV CUDA
    // Use stream-ordered allocation to avoid blocking other streams
    gray_pitch = (size_t)((w + 255) & ~255); // 256-byte aligned
    size_t gray_bytes = gray_pitch * (size_t)h;
    gray_device = unpaper_cuda_malloc_async(stream, gray_bytes);
    if (gray_device == 0) {
      return false;
    }
    gray_allocated = true;

    // Use OpenCV to convert RGB24 to grayscale
    if (!unpaper_opencv_rgb_to_gray(st->dptr, w, h, (size_t)st->linesize,
                                    gray_device, gray_pitch, stream)) {
      unpaper_cuda_free_async(stream, gray_device);
      return false;
    }
  } else if (fmt == UNPAPER_CUDA_FMT_Y400A) {
    // Y400A: Fall back to sequential for simplicity
    return false;
  } else {
    // Other formats: fall back to sequential
    return false;
  }

  // Compute integral image on GPU using NPP
  UnpaperNppContext *npp_ctx = unpaper_npp_context_create(stream);
  UnpaperNppIntegral integral = {0};
  bool integral_ok = unpaper_npp_integral_8u32s(gray_device, w, h, gray_pitch,
                                                npp_ctx, &integral);
  if (npp_ctx != NULL) {
    unpaper_npp_context_destroy(npp_ctx);
  }

  if (gray_allocated) {
    unpaper_cuda_free_async(stream, gray_device);
  }

  if (!integral_ok) {
    return false;
  }

  // Allocate output buffers for scan results
  // Maximum possible rectangles: one per scan position
  const int max_h_positions =
      params.scan_direction.horizontal
          ? ((w / params.scan_step.horizontal + 1) *
             (h / (int)params.scan_depth.vertical + 1))
          : 0;
  const int max_v_positions =
      params.scan_direction.vertical
          ? ((h / params.scan_step.vertical + 1) *
             (w / (int)params.scan_depth.horizontal + 1))
          : 0;
  const int max_rects = max_h_positions + max_v_positions + 1024;

  // Allocate GPU buffers for rectangle output
  // Use stream-ordered allocation to avoid blocking other streams
  uint64_t rects_device = unpaper_cuda_malloc_async(stream, (size_t)max_rects * 4 * sizeof(int32_t));
  uint64_t count_device = unpaper_cuda_malloc_async(stream, sizeof(int));
  if (rects_device == 0 || count_device == 0) {
    unpaper_npp_integral_free(integral.device_ptr);
    if (rects_device != 0) unpaper_cuda_free_async(stream, rects_device);
    if (count_device != 0) unpaper_cuda_free_async(stream, count_device);
    return false;
  }

  // Initialize count to 0 (use async to avoid blocking)
  unpaper_cuda_memset_async(stream, count_device, 0, sizeof(int));

  const int integral_step = (int)integral.step_bytes;
  const int intensity = params.intensity < 0 ? 0 : params.intensity;

  // Horizontal scan (stripes are vertical bands)
  if (params.scan_direction.horizontal) {
    const int stripe_h = (int)params.scan_depth.vertical;
    const int scan_w = params.scan_size.width;
    const int scan_h = stripe_h;
    const int step_x = params.scan_step.horizontal;
    const int step_y = 0;

    for (int stripe_y = 0; stripe_y < h; stripe_y += stripe_h) {
      const int stripe_size = (stripe_y + stripe_h > h) ? (h - stripe_y) : stripe_h;
      if (stripe_size < scan_h) continue;

      // Calculate number of scan positions in this stripe
      const int num_positions = (w - scan_w) / step_x + 1;
      if (num_positions <= 0) continue;

      const uint32_t threads = 256;
      const uint32_t blocks = (uint32_t)((num_positions + threads - 1) / threads);

      void *scan_args[] = {
          &integral.device_ptr, &integral_step,  &w,           &h,
          &scan_w,              &scan_h,         &step_x,      &step_y,
          &stripe_y,            &stripe_size,    &params.abs_threshold,
          &intensity,           &rects_device,   &count_device, &max_rects,
      };
      unpaper_cuda_launch_kernel_on_stream(stream, k_blackfilter_scan_parallel,
                                           blocks, 1, 1, threads, 1, 1, scan_args);
    }
  }

  // Vertical scan (stripes are horizontal bands)
  if (params.scan_direction.vertical) {
    const int stripe_w = (int)params.scan_depth.horizontal;
    const int scan_w = stripe_w;
    const int scan_h = params.scan_size.height;
    const int step_x = 0;
    const int step_y = params.scan_step.vertical;

    for (int stripe_x = 0; stripe_x < w; stripe_x += stripe_w) {
      const int stripe_size = (stripe_x + stripe_w > w) ? (w - stripe_x) : stripe_w;
      if (stripe_size < scan_w) continue;

      const int num_positions = (h - scan_h) / step_y + 1;
      if (num_positions <= 0) continue;

      const uint32_t threads = 256;
      const uint32_t blocks = (uint32_t)((num_positions + threads - 1) / threads);

      void *scan_args[] = {
          &integral.device_ptr, &integral_step,  &w,           &h,
          &scan_w,              &scan_h,         &step_x,      &step_y,
          &stripe_x,            &stripe_size,    &params.abs_threshold,
          &intensity,           &rects_device,   &count_device, &max_rects,
      };
      unpaper_cuda_launch_kernel_on_stream(stream, k_blackfilter_scan_parallel,
                                           blocks, 1, 1, threads, 1, 1, scan_args);
    }
  }

  // Sync to ensure all scans complete
  if (stream != NULL) {
    unpaper_cuda_stream_synchronize_on(stream);
  } else {
    unpaper_cuda_stream_synchronize();
  }

  // Download rectangle count
  int rect_count = 0;
  unpaper_cuda_memcpy_d2h(&rect_count, count_device, sizeof(int));

  // Free integral (no longer needed)
  unpaper_npp_integral_free(integral.device_ptr);

  if (rect_count > 0) {
    // Filter out rectangles that overlap with exclusions
    // Download rectangles to CPU for exclusion filtering
    int32_t *rects_host = NULL;
    if (params.exclusions_count > 0) {
      rects_host = (int32_t *)av_malloc((size_t)rect_count * 4 * sizeof(int32_t));
      if (rects_host != NULL) {
        unpaper_cuda_memcpy_d2h(rects_host, rects_device,
                                (size_t)rect_count * 4 * sizeof(int32_t));

        // Filter out excluded rectangles
        int new_count = 0;
        for (int i = 0; i < rect_count; i++) {
          int x0 = rects_host[i * 4 + 0];
          int y0 = rects_host[i * 4 + 1];
          int x1 = rects_host[i * 4 + 2];
          int y1 = rects_host[i * 4 + 3];

          Rectangle r = {{{x0, y0}, {x1, y1}}};
          if (!rectangle_overlap_any(r, params.exclusions_count, params.exclusions)) {
            rects_host[new_count * 4 + 0] = x0;
            rects_host[new_count * 4 + 1] = y0;
            rects_host[new_count * 4 + 2] = x1;
            rects_host[new_count * 4 + 3] = y1;
            new_count++;
          }
        }
        rect_count = new_count;

        // Upload filtered rectangles back to GPU
        if (rect_count > 0) {
          unpaper_cuda_memcpy_h2d(rects_device, rects_host,
                                  (size_t)rect_count * 4 * sizeof(int32_t));
        }
        av_free(rects_host);
      }
    }

    if (rect_count > 0) {
      // Launch parallel wipe kernel
      const int img_fmt = (int)fmt;
      const uint8_t black_threshold = image.abs_black_threshold;

      const uint32_t block2d = 16;
      const uint32_t grid_x = (uint32_t)((w + block2d - 1) / block2d);
      const uint32_t grid_y = (uint32_t)((h + block2d - 1) / block2d);

      void *wipe_args[] = {
          &st->dptr,      &st->linesize, &img_fmt,    &w,
          &h,             &rects_device, &rect_count, &black_threshold,
      };
      unpaper_cuda_launch_kernel_on_stream(stream, k_blackfilter_wipe_regions,
                                           grid_x, grid_y, 1, block2d, block2d,
                                           1, wipe_args);

      // Sync after wipe
      if (stream != NULL) {
        unpaper_cuda_stream_synchronize_on(stream);
      } else {
        unpaper_cuda_stream_synchronize();
      }

      st->cuda_dirty = true;
      st->cpu_dirty = false;
    }
  }

  unpaper_cuda_free_async(stream, rects_device);
  unpaper_cuda_free_async(stream, count_device);

  return true;
}

// Sequential blackfilter (fallback for formats not supported by parallel version)
static void blackfilter_cuda_sequential(Image image, BlackfilterParameters params) {
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
  // Use scratch pool to avoid cudaMalloc/cudaFree serialization across streams.
  uint64_t stack_dptr = cuda_mempool_scratch_global_acquire(stack_bytes);
  if (stack_dptr == 0) {
    errOutput("blackfilter CUDA: failed to acquire scratch buffer.");
  }

  if (params.scan_direction.horizontal) {
    blackfilter_scan_cuda(
        image, params, (Delta){params.scan_step.horizontal, 0},
        (RectangleSize){params.scan_size.width,
                        (int32_t)params.scan_depth.vertical},
        (Delta){0, (int32_t)params.scan_depth.vertical}, stack_dptr, stack_cap);
  }

  if (params.scan_direction.vertical) {
    blackfilter_scan_cuda(image, params, (Delta){0, params.scan_step.vertical},
                          (RectangleSize){(int32_t)params.scan_depth.horizontal,
                                          params.scan_size.height},
                          (Delta){(int32_t)params.scan_depth.horizontal, 0},
                          stack_dptr, stack_cap);
  }

  cuda_mempool_scratch_global_release(stack_dptr);
}

static void blackfilter_cuda(Image image, BlackfilterParameters params) {
  if (image.frame == NULL) {
    return;
  }

  ensure_kernels_loaded();
  image_ensure_cuda(&image);

  // NOTE: Parallel version disabled - it uses block-based detection without
  // flood-fill, which misses connected dark regions that span multiple stripes.
  // The sequential version correctly uses flood-fill to cover all connected pixels.
  // TODO: Fix parallel version to also flood-fill from detected regions.
#if 0
  // Try parallel version first (for GRAY8 format)
  UnpaperCudaStream *stream = unpaper_cuda_get_current_stream();
  if (blackfilter_cuda_parallel(image, params, stream)) {
    verboseLog(VERBOSE_MORE, "blackfilter: using parallel GPU path\n");
    return;
  }
#endif

  // Use sequential version with flood-fill
  verboseLog(VERBOSE_MORE, "blackfilter: using sequential GPU path\n");
  blackfilter_cuda_sequential(image, params);
}

static void blurfilter_cuda_fallback(Image image, BlurfilterParameters params,
                                     uint8_t abs_white_threshold) {
  RectangleSize image_size = size_of_image(image);
  if (params.scan_size.width <= 0 || params.scan_size.height <= 0) {
    return;
  }

  const uint32_t blocks_per_row =
      (uint32_t)(image_size.width / params.scan_size.width);
  if (blocks_per_row == 0) {
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

static void blurfilter_cuda(Image image, BlurfilterParameters params,
                            uint8_t abs_white_threshold) {
  if (image.frame == NULL) {
    return;
  }

  verboseLog(VERBOSE_NORMAL, "blur-filter...");

  if (params.scan_size.width <= 0 || params.scan_size.height <= 0) {
    verboseLog(VERBOSE_NORMAL, " deleted 0 pixels.\n");
    return;
  }

  ensure_kernels_loaded();
  image_ensure_cuda(&image);
  ImageCudaState *st = image_cuda_state(image);
  if (st == NULL || st->dptr == 0) {
    errOutput("CUDA image state missing for blurfilter.");
  }

  const UnpaperCudaFormat fmt = cuda_format_from_av(image.frame->format);
  if (fmt == UNPAPER_CUDA_FMT_INVALID) {
    errOutput("CUDA blurfilter: unsupported pixel format.");
  }

  // Try OpenCV path for supported formats
  if (fmt == UNPAPER_CUDA_FMT_GRAY8 || fmt == UNPAPER_CUDA_FMT_Y400A ||
      fmt == UNPAPER_CUDA_FMT_RGB24) {
    UnpaperCudaStream *stream = unpaper_cuda_get_current_stream();
    if (unpaper_opencv_blurfilter(
            st->dptr, image.frame->width, image.frame->height,
            (size_t)st->linesize, (int)fmt, params.scan_size.width,
            params.scan_size.height, params.scan_step.horizontal,
            params.scan_step.vertical, abs_white_threshold, params.intensity,
            stream)) {
      st->cuda_dirty = true;
      st->cpu_dirty = false;
      verboseLog(VERBOSE_NORMAL, " (OpenCV) done.\n");
      return;
    }
  }

  // Fallback to custom CUDA implementation
  blurfilter_cuda_fallback(image, params, abs_white_threshold);
}

// Custom CUDA CCL-based noisefilter (fallback)
static void noisefilter_cuda_custom(Image image, uint64_t intensity,
                                    uint8_t min_white_level) {
  ImageCudaState *st = image_cuda_state(image);
  if (st == NULL || st->dptr == 0) {
    errOutput("CUDA image state missing for noisefilter.");
  }

  const UnpaperCudaFormat fmt = cuda_format_from_av(image.frame->format);
  const int img_fmt = (int)fmt;
  const int w = image.frame->width;
  const int h = image.frame->height;
  const size_t num_pixels = (size_t)w * (size_t)h;

  UnpaperCudaStream *stream = unpaper_cuda_get_current_stream();
  const uint32_t block2d_x = 16u;
  const uint32_t block2d_y = 16u;
  const uint32_t grid2d_x =
      (uint32_t)((w + (int)block2d_x - 1) / (int)block2d_x);
  const uint32_t grid2d_y =
      (uint32_t)((h + (int)block2d_y - 1) / (int)block2d_y);

  const size_t labels_bytes = num_pixels * sizeof(uint32_t);
  const size_t counts_bytes = (num_pixels + 1) * sizeof(uint32_t);
  const size_t total_bytes = labels_bytes * 2 + counts_bytes + sizeof(uint32_t);
  size_t scratch_capacity = 0;
  const uint64_t scratch =
      unpaper_cuda_scratch_reserve(total_bytes, &scratch_capacity);
  if (scratch_capacity < total_bytes) {
    errOutput("CUDA noisefilter: unable to reserve scratch buffer.");
  }

  const uint64_t labels_a = scratch;
  const uint64_t labels_b = labels_a + labels_bytes;
  const uint64_t counts = labels_b + labels_bytes;
  const uint64_t changed = counts + counts_bytes;

  void *params_build[] = {&st->dptr, &st->linesize,    &img_fmt, &w,
                          &h,        &min_white_level, &labels_a};
  unpaper_cuda_launch_kernel_on_stream(stream, k_noisefilter_build_labels,
                                       grid2d_x, grid2d_y, 1, block2d_x,
                                       block2d_y, 1, params_build);

  size_t pinned_capacity = 0;
  int *changed_host = (int *)unpaper_cuda_stream_pinned_reserve(
      stream, sizeof(int), &pinned_capacity);
  int changed_fallback = 0;
  if (changed_host == NULL) {
    changed_host = &changed_fallback;
  }

  uint64_t labels_in = labels_a;
  uint64_t labels_out = labels_b;
  const int max_iters = (w + h > 0) ? (w + h) : 1;

  for (int iter = 0; iter < max_iters; iter++) {
    unpaper_cuda_memset_d8(changed, 0, sizeof(int));
    void *params_prop[] = {&labels_in, &labels_out, &w, &h, &changed};
    unpaper_cuda_launch_kernel_on_stream(stream, k_noisefilter_propagate,
                                         grid2d_x, grid2d_y, 1, block2d_x,
                                         block2d_y, 1, params_prop);

    *changed_host = 0;
    if (changed_host == &changed_fallback) {
      unpaper_cuda_memcpy_d2h(changed_host, changed, sizeof(int));
    } else {
      unpaper_cuda_memcpy_d2h_async(stream, changed_host, changed, sizeof(int));
      unpaper_cuda_stream_synchronize_on(stream);
    }

    if (*changed_host == 0) {
      labels_in = labels_out;
      break;
    }

    uint64_t tmp = labels_in;
    labels_in = labels_out;
    labels_out = tmp;
  }

  const uint64_t final_labels = labels_in;

  unpaper_cuda_memset_d8(counts, 0, counts_bytes);
  const uint32_t block1d = 256u;
  const uint32_t grid1d = (uint32_t)(((num_pixels + block1d - 1) / block1d));
  const int num_pixels_i = (int)num_pixels;
  void *params_count[] = {&final_labels, &num_pixels_i, &counts};
  unpaper_cuda_launch_kernel_on_stream(stream, k_noisefilter_count, grid1d, 1,
                                       1, block1d, 1, 1, params_count);

  void *params_apply[] = {&st->dptr, &st->linesize, &img_fmt, &w,
                          &h,        &final_labels, &counts,  &intensity};
  unpaper_cuda_launch_kernel_on_stream(stream, k_noisefilter_apply, grid2d_x,
                                       grid2d_y, 1, block2d_x, block2d_y, 1,
                                       params_apply);

  unpaper_cuda_stream_synchronize_on(stream);

  st->cuda_dirty = true;
  st->cpu_dirty = false;
}

// OpenCV CUDA CCL-based noisefilter for GRAY8, Y400A, and RGB24 images
static bool noisefilter_cuda_opencv(Image image, uint64_t intensity,
                                    uint8_t min_white_level) {
  if (!unpaper_opencv_ccl_supported()) {
    return false;
  }

  ImageCudaState *st = image_cuda_state(image);
  if (st == NULL || st->dptr == 0) {
    return false;
  }

  const UnpaperCudaFormat fmt = cuda_format_from_av(image.frame->format);
  if (fmt != UNPAPER_CUDA_FMT_GRAY8 && fmt != UNPAPER_CUDA_FMT_Y400A &&
      fmt != UNPAPER_CUDA_FMT_RGB24) {
    // Only GRAY8, Y400A, and RGB24 supported via OpenCV
    return false;
  }

  const int w = image.frame->width;
  const int h = image.frame->height;

  UnpaperCudaStream *stream = unpaper_cuda_get_current_stream();

  // Extract dark mask using OpenCV
  UnpaperOpencvMask mask = {0};
  if (!unpaper_opencv_extract_dark_mask(st->dptr, w, h, (size_t)st->linesize,
                                        (int)fmt, min_white_level, stream,
                                        &mask)) {
    return false;
  }

  // Run CCL and remove small components
  UnpaperOpencvCclStats stats = {0};
  bool ok = unpaper_opencv_cuda_ccl(mask.device_ptr, mask.width, mask.height,
                                    mask.pitch_bytes, 255, (uint32_t)intensity,
                                    stream, &stats);

  if (!ok) {
    unpaper_opencv_mask_free(&mask);
    return false;
  }

  // Apply mask on GPU: where mask is 0 and pixel is dark, set to white
  ensure_kernels_loaded();

  const int img_fmt = (int)fmt;
  const int mask_linesize = (int)mask.pitch_bytes;
  void *params[] = {
      &st->dptr, &st->linesize,    &img_fmt,       &w,
      &h,        &mask.device_ptr, &mask_linesize, &min_white_level,
  };

  const uint32_t block_x = 16;
  const uint32_t block_y = 16;
  const uint32_t grid_x = ((uint32_t)w + block_x - 1) / block_x;
  const uint32_t grid_y = ((uint32_t)h + block_y - 1) / block_y;

  unpaper_cuda_launch_kernel_on_stream(stream, k_noisefilter_apply_mask, grid_x,
                                       grid_y, 1, block_x, block_y, 1, params);

  unpaper_opencv_mask_free(&mask);

  st->cuda_dirty = true;
  st->cpu_dirty = false;

  verboseLog(VERBOSE_MORE, " (OpenCV CCL: %d components, %d removed)",
             stats.label_count, stats.removed_components);

  return true;
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

  const int w = image.frame->width;
  const int h = image.frame->height;
  if (w <= 0 || h <= 0) {
    return;
  }

  const size_t num_pixels = (size_t)w * (size_t)h;
  if (num_pixels == 0) {
    return;
  }

  // Try OpenCV path for GRAY8, Y400A, and RGB24 images
  if ((fmt == UNPAPER_CUDA_FMT_GRAY8 || fmt == UNPAPER_CUDA_FMT_Y400A ||
       fmt == UNPAPER_CUDA_FMT_RGB24) &&
      noisefilter_cuda_opencv(image, intensity, min_white_level)) {
    return;
  }

  // Fallback to custom CUDA CCL
  noisefilter_cuda_custom(image, intensity, min_white_level);
}

static void grayfilter_cuda_fallback(Image image, GrayfilterParameters params) {
  RectangleSize image_size = size_of_image(image);
  Point filter_origin = POINT_ORIGIN;

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
}

static void grayfilter_cuda(Image image, GrayfilterParameters params) {
  if (image.frame == NULL) {
    return;
  }

  verboseLog(VERBOSE_NORMAL, "gray-filter...");

  ensure_kernels_loaded();
  image_ensure_cuda(&image);
  ImageCudaState *st = image_cuda_state(image);
  if (st == NULL || st->dptr == 0) {
    errOutput("CUDA image state missing for grayfilter.");
  }

  const UnpaperCudaFormat fmt = cuda_format_from_av(image.frame->format);
  if (fmt == UNPAPER_CUDA_FMT_INVALID) {
    errOutput("CUDA grayfilter: unsupported pixel format.");
  }

  // Try OpenCV path for supported formats
  if (fmt == UNPAPER_CUDA_FMT_GRAY8 || fmt == UNPAPER_CUDA_FMT_Y400A ||
      fmt == UNPAPER_CUDA_FMT_RGB24) {
    UnpaperCudaStream *stream = unpaper_cuda_get_current_stream();
    if (unpaper_opencv_grayfilter(
            st->dptr, image.frame->width, image.frame->height,
            (size_t)st->linesize, (int)fmt, params.scan_size.width,
            params.scan_size.height, params.scan_step.horizontal,
            params.scan_step.vertical, image.abs_black_threshold,
            params.abs_threshold, stream)) {
      st->cuda_dirty = true;
      st->cpu_dirty = false;
      verboseLog(VERBOSE_NORMAL, " (OpenCV) done.\n");
      return;
    }
  }

  // Fallback to custom CUDA implementation
  grayfilter_cuda_fallback(image, params);
  verboseLog(VERBOSE_NORMAL, " deleted 0 pixels.\n");
}

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
  uint64_t peaks_d = unpaper_cuda_malloc_async(stream, (size_t)rotations_count * sizeof(int));

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
                                       (uint32_t)rotations_count, 1, 1, 256, 1, 1,
                                       params_k);

  // Use async D2H + stream sync for stream-specific synchronization
  // Note: peaks_h is av_malloc'd, not pinned, so async D2H will be synchronous anyway
  // but using stream_synchronize_on ensures we only wait on this stream, not all streams
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

static float detect_rotation_cuda(Image image, Rectangle mask,
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

static void deskew_cuda(Image source, Rectangle mask, float radians,
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
