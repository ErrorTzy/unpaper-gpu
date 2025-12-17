// SPDX-FileCopyrightText: 2025 The unpaper authors
//
// SPDX-License-Identifier: GPL-2.0-only

// CUDA backend: filter operations (blackfilter, blurfilter, noisefilter,
// grayfilter)

#include "imageprocess/backend_cuda_internal.h"

#include <inttypes.h>
#include <stdlib.h>

#include <libavutil/frame.h>
#include <libavutil/mem.h>

#include "imageprocess/cuda_mempool.h"
#include "imageprocess/cuda_runtime.h"
#include "imageprocess/npp_integral.h"
#include "imageprocess/opencv_bridge.h"
#include "lib/logging.h"
#include "lib/math_util.h"

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

// Parallel blackfilter using integral image + GPU parallel scan/wipe
// This replaces the sequential flood-fill approach with fully parallel
// processing:
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
  const int max_h_positions = params.scan_direction.horizontal
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
  uint64_t rects_device = unpaper_cuda_malloc_async(
      stream, (size_t)max_rects * 4 * sizeof(int32_t));
  uint64_t count_device = unpaper_cuda_malloc_async(stream, sizeof(int));
  if (rects_device == 0 || count_device == 0) {
    unpaper_npp_integral_free(integral.device_ptr);
    if (rects_device != 0)
      unpaper_cuda_free_async(stream, rects_device);
    if (count_device != 0)
      unpaper_cuda_free_async(stream, count_device);
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
      const int stripe_size =
          (stripe_y + stripe_h > h) ? (h - stripe_y) : stripe_h;
      if (stripe_size < scan_h)
        continue;

      // Calculate number of scan positions in this stripe
      const int num_positions = (w - scan_w) / step_x + 1;
      if (num_positions <= 0)
        continue;

      const uint32_t threads = 256;
      const uint32_t blocks =
          (uint32_t)((num_positions + threads - 1) / threads);

      void *scan_args[] = {
          &integral.device_ptr,
          &integral_step,
          &w,
          &h,
          &scan_w,
          &scan_h,
          &step_x,
          &step_y,
          &stripe_y,
          &stripe_size,
          &params.abs_threshold,
          &intensity,
          &rects_device,
          &count_device,
          &max_rects,
      };
      unpaper_cuda_launch_kernel_on_stream(stream, k_blackfilter_scan_parallel,
                                           blocks, 1, 1, threads, 1, 1,
                                           scan_args);
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
      const int stripe_size =
          (stripe_x + stripe_w > w) ? (w - stripe_x) : stripe_w;
      if (stripe_size < scan_w)
        continue;

      const int num_positions = (h - scan_h) / step_y + 1;
      if (num_positions <= 0)
        continue;

      const uint32_t threads = 256;
      const uint32_t blocks =
          (uint32_t)((num_positions + threads - 1) / threads);

      void *scan_args[] = {
          &integral.device_ptr,
          &integral_step,
          &w,
          &h,
          &scan_w,
          &scan_h,
          &step_x,
          &step_y,
          &stripe_x,
          &stripe_size,
          &params.abs_threshold,
          &intensity,
          &rects_device,
          &count_device,
          &max_rects,
      };
      unpaper_cuda_launch_kernel_on_stream(stream, k_blackfilter_scan_parallel,
                                           blocks, 1, 1, threads, 1, 1,
                                           scan_args);
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
      rects_host =
          (int32_t *)av_malloc((size_t)rect_count * 4 * sizeof(int32_t));
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
          if (!rectangle_overlap_any(r, params.exclusions_count,
                                     params.exclusions)) {
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
          &st->dptr, &st->linesize, &img_fmt,    &w,
          &h,        &rects_device, &rect_count, &black_threshold,
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

// Sequential blackfilter (fallback for formats not supported by parallel
// version)
static void blackfilter_cuda_sequential(Image image,
                                        BlackfilterParameters params) {
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

void blackfilter_cuda(Image image, BlackfilterParameters params) {
  if (image.frame == NULL) {
    return;
  }

  ensure_kernels_loaded();
  image_ensure_cuda(&image);

  // NOTE: Parallel version disabled - it uses block-based detection without
  // flood-fill, which misses connected dark regions that span multiple stripes.
  // The sequential version correctly uses flood-fill to cover all connected
  // pixels.
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

void blurfilter_cuda(Image image, BlurfilterParameters params,
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

void noisefilter_cuda(Image image, uint64_t intensity,
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

void grayfilter_cuda(Image image, GrayfilterParameters params) {
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
