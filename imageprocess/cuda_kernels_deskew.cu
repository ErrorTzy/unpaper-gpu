// SPDX-FileCopyrightText: 2025 The unpaper authors
//
// SPDX-License-Identifier: GPL-2.0-only

// Rotation detection and deskew/rotation kernels

#include "imageprocess/cuda_kernels_common.cuh"

// ============================================================================
// Edge rotation peak detection kernel
// ============================================================================

extern "C" __global__ void unpaper_detect_edge_rotation_peaks(
    const uint8_t *src, int src_linesize, int fmt, int src_w, int src_h,
    const int *base_x_all, const int *base_y_all, int scan_size, int max_depth,
    int shift_x, int shift_y, int mask_x0, int mask_y0, int mask_x1,
    int mask_y1, int max_blackness_abs, int *out_peaks) {
  const int angle_idx = (int)blockIdx.x;
  const int tid = (int)threadIdx.x;

  const int *base_x = base_x_all + (size_t)angle_idx * (size_t)scan_size;
  const int *base_y = base_y_all + (size_t)angle_idx * (size_t)scan_size;

  __shared__ int sh_sum[256];
  __shared__ int sh_last_blackness;
  __shared__ int sh_max_diff;
  __shared__ int sh_accumulated;
  __shared__ int sh_dep;
  __shared__ int sh_continue;

  const UnpaperCudaFormat f = (UnpaperCudaFormat)fmt;

  if (tid == 0) {
    sh_last_blackness = 0;
    sh_max_diff = 0;
    sh_accumulated = 0;
    sh_dep = 0;
    sh_continue = 1;
  }
  __syncthreads();

  while (true) {
    if (tid == 0) {
      sh_continue =
          (sh_accumulated < max_blackness_abs && sh_dep < max_depth) ? 1 : 0;
    }
    __syncthreads();
    if (sh_continue == 0) {
      break;
    }

    const int dep = sh_dep;

    int local_sum = 0;
    for (int i = tid; i < scan_size; i += (int)blockDim.x) {
      const int x = base_x[i] + dep * shift_x;
      const int y = base_y[i] + dep * shift_y;
      if (x < mask_x0 || x > mask_x1 || y < mask_y0 || y > mask_y1) {
        continue;
      }
      const uint8_t inv =
          read_darkness_inverse(src, src_linesize, f, src_w, src_h, x, y);
      local_sum += (int)(255u - inv);
    }

    sh_sum[tid] = local_sum;
    __syncthreads();

    for (int offset = (int)blockDim.x / 2; offset > 0; offset >>= 1) {
      if (tid < offset) {
        sh_sum[tid] += sh_sum[tid + offset];
      }
      __syncthreads();
    }

    const int blackness = sh_sum[0];
    if (tid == 0) {
      const int diff = blackness - sh_last_blackness;
      sh_last_blackness = blackness;
      if (diff >= sh_max_diff) {
        sh_max_diff = diff;
      }
      sh_accumulated += blackness;
      sh_dep++;
    }
    __syncthreads();
  }

  if (tid == 0) {
    out_peaks[angle_idx] = (sh_dep < max_depth) ? sh_max_diff : 0;
  }
}

// ============================================================================
// Rotation transform kernels
// ============================================================================

extern "C" __global__ void
unpaper_rotate_bytes(const uint8_t *src, int src_linesize, uint8_t *dst,
                     int dst_linesize, int fmt, int src_w, int src_h, int dst_w,
                     int dst_h, float src_center_x, float src_center_y,
                     float dst_center_x, float dst_center_y, float cosval,
                     float sinval, int interp_type) {
  const int x = (int)(blockIdx.x * blockDim.x + threadIdx.x);
  const int y = (int)(blockIdx.y * blockDim.y + threadIdx.y);
  if (x < 0 || y < 0 || x >= dst_w || y >= dst_h) {
    return;
  }

  const float dx = (float)x - dst_center_x;
  const float dy = (float)y - dst_center_y;
  const float sx = src_center_x + dx * cosval + dy * sinval;
  const float sy = src_center_y + dy * cosval - dx * sinval;

  uint8_t r = 255u, g = 255u, b = 255u;
  const UnpaperCudaFormat f = (UnpaperCudaFormat)fmt;
  if (interp_type == 0) {
    interp_nn_round(src, src_linesize, f, src_w, src_h, sx, sy, &r, &g, &b);
  } else if (interp_type == 1) {
    interp_bilinear(src, src_linesize, f, src_w, src_h, sx, sy, &r, &g, &b);
  } else {
    interp_bicubic(src, src_linesize, f, src_w, src_h, sx, sy, &r, &g, &b);
  }

  write_pixel(dst, dst_linesize, f, x, y, r, g, b);
}

extern "C" __global__ void
unpaper_rotate_mono(const uint8_t *src, int src_linesize, int src_fmt,
                    uint8_t *dst, int dst_linesize, int dst_fmt, int src_w,
                    int src_h, int dst_w, int dst_h, float src_center_x,
                    float src_center_y, float dst_center_x, float dst_center_y,
                    float cosval, float sinval, int interp_type,
                    uint8_t abs_black_threshold) {
  const int bytes_per_row = (dst_w + 7) / 8;
  const int byte_x = (int)(blockIdx.x * blockDim.x + threadIdx.x);
  const int y = (int)(blockIdx.y * blockDim.y + threadIdx.y);
  if (byte_x < 0 || byte_x >= bytes_per_row || y < 0 || y >= dst_h) {
    return;
  }

  const int dst_byte_start_x = byte_x * 8;
  uint8_t out = 0;
  for (int bit = 0; bit < 8; bit++) {
    const int x = dst_byte_start_x + bit;
    if (x >= dst_w) {
      continue;
    }

    const float dx = (float)x - dst_center_x;
    const float dy = (float)y - dst_center_y;
    const float sx = src_center_x + dx * cosval + dy * sinval;
    const float sy = src_center_y + dy * cosval - dx * sinval;

    uint8_t r = 255u, g = 255u, b = 255u;
    const UnpaperCudaFormat f = (UnpaperCudaFormat)src_fmt;
    if (interp_type == 0) {
      interp_nn_round(src, src_linesize, f, src_w, src_h, sx, sy, &r, &g, &b);
    } else if (interp_type == 1) {
      interp_bilinear(src, src_linesize, f, src_w, src_h, sx, sy, &r, &g, &b);
    } else {
      interp_bicubic(src, src_linesize, f, src_w, src_h, sx, sy, &r, &g, &b);
    }

    const uint8_t gray = grayscale_u8(r, g, b);
    const bool pixel_black = gray < abs_black_threshold;
    const bool bit_set =
        ((UnpaperCudaFormat)dst_fmt == UNPAPER_CUDA_FMT_MONOWHITE)
            ? pixel_black
            : !pixel_black;
    if (bit_set) {
      out = (uint8_t)(out | (uint8_t)(0x80u >> bit));
    }
  }

  dst[(size_t)y * (size_t)dst_linesize + (size_t)byte_x] = out;
}

// ============================================================================
// Batched edge scan kernels
// ============================================================================
// These kernels eliminate the per-iteration cudaStreamSynchronize overhead
// in detect_edge_cuda() and detect_border_edge_cuda() by computing ALL scan
// positions in a single kernel launch with a single D2H transfer.

/**
 * Batch scan for grayscale sums - used by detect_edge_cuda().
 *
 * Each block processes one scan position. The base rectangle is shifted by
 * (step_x * block_idx, step_y * block_idx) and the sum of grayscale values
 * is computed using parallel reduction within the block.
 *
 * Parameters:
 *   src, src_linesize, src_fmt, src_w, src_h: Source image
 *   base_x0, base_y0: Top-left of base rectangle (at position 0)
 *   rect_w, rect_h: Rectangle dimensions
 *   step_x, step_y: Step direction (one must be 0)
 *   max_positions: Number of positions to compute
 *   out_sums: Output array of uint64_t[max_positions] for grayscale sums
 *   out_counts: Output array of uint64_t[max_positions] for pixel counts
 */
extern "C" __global__ void unpaper_batch_scan_grayscale_sum(
    const uint8_t *src, int src_linesize, int src_fmt, int src_w, int src_h,
    int base_x0, int base_y0, int rect_w, int rect_h, int step_x, int step_y,
    int max_positions, unsigned long long *out_sums,
    unsigned long long *out_counts) {
  const int pos_idx = (int)blockIdx.x;
  if (pos_idx >= max_positions) {
    return;
  }

  const int tid = (int)threadIdx.x;
  const UnpaperCudaFormat fmt = (UnpaperCudaFormat)src_fmt;

  // Compute rectangle for this position
  const int x0 = base_x0 + pos_idx * step_x;
  const int y0 = base_y0 + pos_idx * step_y;

  // Shared memory for parallel reduction
  __shared__ unsigned long long sh_sum[256];
  __shared__ unsigned long long sh_count[256];

  // Compute local sum for this thread's pixels
  unsigned long long local_sum = 0;
  unsigned long long local_count = 0;

  const int total_pixels = rect_w * rect_h;
  for (int i = tid; i < total_pixels; i += (int)blockDim.x) {
    const int rx = i % rect_w;
    const int ry = i / rect_w;
    const int x = x0 + rx;
    const int y = y0 + ry;

    // Only count pixels inside image bounds
    if (x >= 0 && x < src_w && y >= 0 && y < src_h) {
      uint8_t r, g, b;
      read_rgb(src, src_linesize, fmt, x, y, &r, &g, &b);
      local_sum += grayscale_u8(r, g, b);
      local_count++;
    }
  }

  sh_sum[tid] = local_sum;
  sh_count[tid] = local_count;
  __syncthreads();

  // Parallel reduction
  for (int offset = (int)blockDim.x / 2; offset > 0; offset >>= 1) {
    if (tid < offset) {
      sh_sum[tid] += sh_sum[tid + offset];
      sh_count[tid] += sh_count[tid + offset];
    }
    __syncthreads();
  }

  // Write result
  if (tid == 0) {
    out_sums[pos_idx] = sh_sum[0];
    out_counts[pos_idx] = sh_count[0];
  }
}

/**
 * Batch scan for brightness range counts - used by detect_border_edge_cuda().
 *
 * Each block processes one scan position. Counts pixels where grayscale
 * is within [min_brightness, max_brightness].
 *
 * Parameters:
 *   src, src_linesize, src_fmt, src_w, src_h: Source image
 *   base_x0, base_y0: Top-left of base rectangle (at position 0)
 *   rect_w, rect_h: Rectangle dimensions
 *   step_x, step_y: Step direction (one must be 0)
 *   max_positions: Number of positions to compute
 *   min_brightness, max_brightness: Brightness range for counting
 *   out_counts: Output array of uint64_t[max_positions]
 */
extern "C" __global__ void unpaper_batch_scan_brightness_count(
    const uint8_t *src, int src_linesize, int src_fmt, int src_w, int src_h,
    int base_x0, int base_y0, int rect_w, int rect_h, int step_x, int step_y,
    int max_positions, uint8_t min_brightness, uint8_t max_brightness,
    unsigned long long *out_counts) {
  const int pos_idx = (int)blockIdx.x;
  if (pos_idx >= max_positions) {
    return;
  }

  const int tid = (int)threadIdx.x;
  const UnpaperCudaFormat fmt = (UnpaperCudaFormat)src_fmt;

  // Compute rectangle for this position
  const int x0 = base_x0 + pos_idx * step_x;
  const int y0 = base_y0 + pos_idx * step_y;

  // Shared memory for parallel reduction
  __shared__ unsigned long long sh_count[256];

  // Compute local count for this thread's pixels
  unsigned long long local_count = 0;

  const int total_pixels = rect_w * rect_h;
  for (int i = tid; i < total_pixels; i += (int)blockDim.x) {
    const int rx = i % rect_w;
    const int ry = i / rect_w;
    const int x = x0 + rx;
    const int y = y0 + ry;

    // Only count pixels inside image bounds
    if (x >= 0 && x < src_w && y >= 0 && y < src_h) {
      uint8_t r, g, b;
      read_rgb(src, src_linesize, fmt, x, y, &r, &g, &b);
      const uint8_t gray = grayscale_u8(r, g, b);
      if (gray >= min_brightness && gray <= max_brightness) {
        local_count++;
      }
    }
  }

  sh_count[tid] = local_count;
  __syncthreads();

  // Parallel reduction
  for (int offset = (int)blockDim.x / 2; offset > 0; offset >>= 1) {
    if (tid < offset) {
      sh_count[tid] += sh_count[tid + offset];
    }
    __syncthreads();
  }

  // Write result
  if (tid == 0) {
    out_counts[pos_idx] = sh_count[0];
  }
}
