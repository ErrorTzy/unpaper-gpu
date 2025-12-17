// SPDX-FileCopyrightText: 2025 The unpaper authors
//
// SPDX-License-Identifier: GPL-2.0-only

// Mask detection, border operations, and statistics kernels

#include "imageprocess/cuda_kernels_common.cuh"

// ============================================================================
// Statistics kernels
// ============================================================================

extern "C" __global__ void unpaper_count_brightness_range(
    const uint8_t *src, int src_linesize, int src_fmt, int src_w, int src_h,
    int x0, int y0, int x1, int y1, uint8_t min_brightness,
    uint8_t max_brightness, unsigned long long *out_count) {
  const int w = x1 - x0 + 1;
  const int h = y1 - y0 + 1;
  if (w <= 0 || h <= 0) {
    return;
  }

  const unsigned long long total =
      (unsigned long long)w * (unsigned long long)h;
  unsigned long long idx =
      (unsigned long long)blockIdx.x * (unsigned long long)blockDim.x +
      (unsigned long long)threadIdx.x;
  const unsigned long long stride =
      (unsigned long long)blockDim.x * (unsigned long long)gridDim.x;

  const UnpaperCudaFormat fmt = (UnpaperCudaFormat)src_fmt;
  while (idx < total) {
    const int rx = (int)(idx % (unsigned long long)w);
    const int ry = (int)(idx / (unsigned long long)w);
    const int x = x0 + rx;
    const int y = y0 + ry;

    uint8_t r, g, b;
    read_rgb_safe(src, src_linesize, fmt, src_w, src_h, x, y, &r, &g, &b);
    const uint8_t gray = grayscale_u8(r, g, b);
    if (gray >= min_brightness && gray <= max_brightness) {
      atomicAdd(out_count, 1ull);
    }

    idx += stride;
  }
}

extern "C" __global__ void
unpaper_sum_lightness_rect(const uint8_t *src, int src_linesize, int src_fmt,
                           int src_w, int src_h, int x0, int y0, int x1, int y1,
                           unsigned long long *out_sum) {
  const int w = x1 - x0 + 1;
  const int h = y1 - y0 + 1;
  if (w <= 0 || h <= 0) {
    return;
  }

  const unsigned long long total =
      (unsigned long long)w * (unsigned long long)h;
  unsigned long long idx =
      (unsigned long long)blockIdx.x * (unsigned long long)blockDim.x +
      (unsigned long long)threadIdx.x;
  const unsigned long long stride =
      (unsigned long long)blockDim.x * (unsigned long long)gridDim.x;

  const UnpaperCudaFormat fmt = (UnpaperCudaFormat)src_fmt;
  while (idx < total) {
    const int rx = (int)(idx % (unsigned long long)w);
    const int ry = (int)(idx / (unsigned long long)w);
    const int x = x0 + rx;
    const int y = y0 + ry;

    uint8_t r, g, b;
    read_rgb_safe(src, src_linesize, fmt, src_w, src_h, x, y, &r, &g, &b);
    atomicAdd(out_sum, (unsigned long long)lightness_u8(r, g, b));

    idx += stride;
  }
}

extern "C" __global__ void
unpaper_sum_grayscale_rect(const uint8_t *src, int src_linesize, int src_fmt,
                           int src_w, int src_h, int x0, int y0, int x1, int y1,
                           unsigned long long *out_sum) {
  const int w = x1 - x0 + 1;
  const int h = y1 - y0 + 1;
  if (w <= 0 || h <= 0) {
    return;
  }

  const unsigned long long total =
      (unsigned long long)w * (unsigned long long)h;
  unsigned long long idx =
      (unsigned long long)blockIdx.x * (unsigned long long)blockDim.x +
      (unsigned long long)threadIdx.x;
  const unsigned long long stride =
      (unsigned long long)blockDim.x * (unsigned long long)gridDim.x;

  const UnpaperCudaFormat fmt = (UnpaperCudaFormat)src_fmt;
  while (idx < total) {
    const int rx = (int)(idx % (unsigned long long)w);
    const int ry = (int)(idx / (unsigned long long)w);
    const int x = x0 + rx;
    const int y = y0 + ry;

    uint8_t r, g, b;
    read_rgb_safe(src, src_linesize, fmt, src_w, src_h, x, y, &r, &g, &b);
    atomicAdd(out_sum, (unsigned long long)grayscale_u8(r, g, b));

    idx += stride;
  }
}

extern "C" __global__ void unpaper_sum_darkness_inverse_rect(
    const uint8_t *src, int src_linesize, int src_fmt, int src_w, int src_h,
    int x0, int y0, int x1, int y1, unsigned long long *out_sum) {
  const int w = x1 - x0 + 1;
  const int h = y1 - y0 + 1;
  if (w <= 0 || h <= 0) {
    return;
  }

  const unsigned long long total =
      (unsigned long long)w * (unsigned long long)h;
  unsigned long long idx =
      (unsigned long long)blockIdx.x * (unsigned long long)blockDim.x +
      (unsigned long long)threadIdx.x;
  const unsigned long long stride =
      (unsigned long long)blockDim.x * (unsigned long long)gridDim.x;

  const UnpaperCudaFormat fmt = (UnpaperCudaFormat)src_fmt;
  while (idx < total) {
    const int rx = (int)(idx % (unsigned long long)w);
    const int ry = (int)(idx / (unsigned long long)w);
    const int x = x0 + rx;
    const int y = y0 + ry;

    uint8_t r, g, b;
    read_rgb_safe(src, src_linesize, fmt, src_w, src_h, x, y, &r, &g, &b);
    atomicAdd(out_sum, (unsigned long long)darkness_inverse_u8(r, g, b));

    idx += stride;
  }
}

// ============================================================================
// Mask application kernels
// ============================================================================

static __device__ __forceinline__ bool
rect_contains_point(const int32_t *rects, int rect_index, int x, int y) {
  const int32_t x0 = rects[rect_index * 4 + 0];
  const int32_t y0 = rects[rect_index * 4 + 1];
  const int32_t x1 = rects[rect_index * 4 + 2];
  const int32_t y1 = rects[rect_index * 4 + 3];
  return x >= x0 && x <= x1 && y >= y0 && y <= y1;
}

extern "C" __global__ void
unpaper_apply_masks_bytes(uint8_t *img, int img_linesize, int img_fmt,
                          int img_w, int img_h, const int32_t *rects,
                          int rect_count, uint8_t r, uint8_t g, uint8_t b) {
  const int x = (int)(blockIdx.x * blockDim.x + threadIdx.x);
  const int y = (int)(blockIdx.y * blockDim.y + threadIdx.y);
  if (x < 0 || y < 0 || x >= img_w || y >= img_h) {
    return;
  }

  const UnpaperCudaFormat fmt = (UnpaperCudaFormat)img_fmt;
  if (fmt == UNPAPER_CUDA_FMT_MONOWHITE || fmt == UNPAPER_CUDA_FMT_MONOBLACK) {
    return;
  }

  for (int i = 0; i < rect_count; i++) {
    if (rect_contains_point(rects, i, x, y)) {
      return;
    }
  }

  write_pixel(img, img_linesize, fmt, x, y, r, g, b);
}

extern "C" __global__ void
unpaper_apply_masks_mono(uint8_t *img, int img_linesize, int img_fmt, int img_w,
                         int img_h, const int32_t *rects, int rect_count,
                         uint8_t bit_value) {
  const int bytes_span = (img_w + 7) / 8;

  const int byte_x = (int)(blockIdx.x * blockDim.x + threadIdx.x);
  const int y = (int)(blockIdx.y * blockDim.y + threadIdx.y);
  if (byte_x < 0 || byte_x >= bytes_span || y < 0 || y >= img_h) {
    return;
  }

  const UnpaperCudaFormat fmt = (UnpaperCudaFormat)img_fmt;
  if (fmt != UNPAPER_CUDA_FMT_MONOWHITE && fmt != UNPAPER_CUDA_FMT_MONOBLACK) {
    return;
  }

  const int byte_start_x = byte_x * 8;
  uint8_t mask = 0;
  for (int bit = 0; bit < 8; bit++) {
    const int x = byte_start_x + bit;
    if (x < 0 || x >= img_w) {
      continue;
    }

    bool inside = false;
    for (int i = 0; i < rect_count; i++) {
      if (rect_contains_point(rects, i, x, y)) {
        inside = true;
        break;
      }
    }
    if (!inside) {
      mask |= (uint8_t)(0x80u >> bit);
    }
  }

  if (mask == 0) {
    return;
  }

  uint8_t *p = img + (size_t)y * (size_t)img_linesize + (size_t)byte_x;
  const uint8_t orig = *p;
  if (bit_value) {
    *p = (uint8_t)(orig | mask);
  } else {
    *p = (uint8_t)(orig & (uint8_t)(~mask));
  }
}
