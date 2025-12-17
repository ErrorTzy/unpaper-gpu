// SPDX-FileCopyrightText: 2025 The unpaper authors
//
// SPDX-License-Identifier: GPL-2.0-only

#ifndef CUDA_KERNELS_COMMON_CUH
#define CUDA_KERNELS_COMMON_CUH

#include <math.h>
#include <stdint.h>

#include "imageprocess/cuda_kernels_format.h"

// ============================================================================
// Color conversion helpers
// ============================================================================

static __device__ __forceinline__ uint8_t grayscale_u8(uint8_t r, uint8_t g,
                                                       uint8_t b) {
  return (uint8_t)(((uint32_t)r + (uint32_t)g + (uint32_t)b) / 3u);
}

static __device__ __forceinline__ uint8_t min3_u8(uint8_t a, uint8_t b,
                                                  uint8_t c) {
  uint8_t m = a < b ? a : b;
  return m < c ? m : c;
}

static __device__ __forceinline__ uint8_t max3_u8(uint8_t a, uint8_t b,
                                                  uint8_t c) {
  uint8_t m = a > b ? a : b;
  return m > c ? m : c;
}

static __device__ __forceinline__ uint8_t lightness_u8(uint8_t r, uint8_t g,
                                                       uint8_t b) {
  return min3_u8(r, g, b);
}

static __device__ __forceinline__ uint8_t darkness_inverse_u8(uint8_t r,
                                                              uint8_t g,
                                                              uint8_t b) {
  return max3_u8(r, g, b);
}

static __device__ __forceinline__ uint8_t clip_uint8(int v) {
  if (v < 0) {
    return 0u;
  }
  if (v > 255) {
    return 255u;
  }
  return (uint8_t)v;
}

// ============================================================================
// Pixel read/write helpers
// ============================================================================

static __device__ __forceinline__ void
read_rgb(const uint8_t *src, int src_linesize, UnpaperCudaFormat fmt, int x,
         int y, uint8_t *r, uint8_t *g, uint8_t *b) {
  const uint8_t *row = src + (size_t)y * (size_t)src_linesize;
  switch (fmt) {
  case UNPAPER_CUDA_FMT_GRAY8: {
    const uint8_t v = row[x];
    *r = v;
    *g = v;
    *b = v;
  } break;
  case UNPAPER_CUDA_FMT_Y400A: {
    const uint8_t v = row[x * 2];
    *r = v;
    *g = v;
    *b = v;
  } break;
  case UNPAPER_CUDA_FMT_RGB24: {
    const uint8_t *p = row + x * 3;
    *r = p[0];
    *g = p[1];
    *b = p[2];
  } break;
  case UNPAPER_CUDA_FMT_MONOWHITE:
  case UNPAPER_CUDA_FMT_MONOBLACK: {
    const uint8_t byte = row[x / 8];
    const uint8_t mask = (uint8_t)(0x80u >> (x & 7));
    const bool bit_set = (byte & mask) != 0;
    const bool is_white =
        (fmt == UNPAPER_CUDA_FMT_MONOBLACK) ? bit_set : (!bit_set);
    const uint8_t v = is_white ? 255u : 0u;
    *r = v;
    *g = v;
    *b = v;
  } break;
  default:
    *r = 255u;
    *g = 255u;
    *b = 255u;
    break;
  }
}

static __device__ __forceinline__ void
read_rgb_safe(const uint8_t *src, int src_linesize, UnpaperCudaFormat fmt,
              int src_w, int src_h, int x, int y, uint8_t *r, uint8_t *g,
              uint8_t *b) {
  if (x < 0 || y < 0 || x >= src_w || y >= src_h) {
    *r = 255u;
    *g = 255u;
    *b = 255u;
    return;
  }
  read_rgb(src, src_linesize, fmt, x, y, r, g, b);
}

static __device__ __forceinline__ void
write_pixel(uint8_t *dst, int dst_linesize, UnpaperCudaFormat fmt, int x, int y,
            uint8_t r, uint8_t g, uint8_t b) {
  uint8_t *row = dst + (size_t)y * (size_t)dst_linesize;
  switch (fmt) {
  case UNPAPER_CUDA_FMT_GRAY8: {
    row[x] = grayscale_u8(r, g, b);
  } break;
  case UNPAPER_CUDA_FMT_Y400A: {
    row[x * 2] = grayscale_u8(r, g, b);
    row[x * 2 + 1] = 0xFFu;
  } break;
  case UNPAPER_CUDA_FMT_RGB24: {
    uint8_t *p = row + x * 3;
    p[0] = r;
    p[1] = g;
    p[2] = b;
  } break;
  default:
    break;
  }
}

static __device__ __forceinline__ uint8_t read_darkness_inverse(
    const uint8_t *src, int src_linesize, UnpaperCudaFormat fmt, int src_w,
    int src_h, int x, int y) {
  if (x < 0 || y < 0 || x >= src_w || y >= src_h) {
    return 255u;
  }

  const uint8_t *row = src + (size_t)y * (size_t)src_linesize;
  switch (fmt) {
  case UNPAPER_CUDA_FMT_GRAY8:
    return row[x];
  case UNPAPER_CUDA_FMT_Y400A:
    return row[x * 2];
  case UNPAPER_CUDA_FMT_RGB24: {
    const uint8_t *p = row + x * 3;
    return darkness_inverse_u8(p[0], p[1], p[2]);
  }
  case UNPAPER_CUDA_FMT_MONOWHITE:
  case UNPAPER_CUDA_FMT_MONOBLACK: {
    const uint8_t byte = row[x / 8];
    const uint8_t mask = (uint8_t)(0x80u >> (x & 7));
    const bool bit_set = (byte & mask) != 0;
    const bool is_white =
        (fmt == UNPAPER_CUDA_FMT_MONOBLACK) ? bit_set : (!bit_set);
    return is_white ? 255u : 0u;
  }
  default:
    return 255u;
  }
}

static __device__ __forceinline__ uint8_t
get_grayscale_safe(const uint8_t *src, int src_linesize, UnpaperCudaFormat fmt,
                   int src_w, int src_h, int x, int y) {
  uint8_t r, g, b;
  read_rgb_safe(src, src_linesize, fmt, src_w, src_h, x, y, &r, &g, &b);
  return grayscale_u8(r, g, b);
}

static __device__ __forceinline__ uint8_t get_darkness_inverse_safe(
    const uint8_t *src, int src_linesize, UnpaperCudaFormat fmt, int src_w,
    int src_h, int x, int y) {
  uint8_t r, g, b;
  read_rgb_safe(src, src_linesize, fmt, src_w, src_h, x, y, &r, &g, &b);
  return darkness_inverse_u8(r, g, b);
}

static __device__ __forceinline__ void
set_pixel_white_safe(uint8_t *dst, int dst_linesize, UnpaperCudaFormat fmt,
                     int dst_w, int dst_h, int x, int y) {
  if (x < 0 || y < 0 || x >= dst_w || y >= dst_h) {
    return;
  }

  uint8_t *row = dst + (size_t)y * (size_t)dst_linesize;
  switch (fmt) {
  case UNPAPER_CUDA_FMT_GRAY8:
    row[x] = 255u;
    break;
  case UNPAPER_CUDA_FMT_Y400A:
    row[x * 2] = 255u;
    row[x * 2 + 1] = 0xFFu;
    break;
  case UNPAPER_CUDA_FMT_RGB24: {
    uint8_t *p = row + x * 3;
    p[0] = 255u;
    p[1] = 255u;
    p[2] = 255u;
  } break;
  case UNPAPER_CUDA_FMT_MONOWHITE:
  case UNPAPER_CUDA_FMT_MONOBLACK: {
    uint8_t *bytep = row + (x / 8);
    const uint8_t mask = (uint8_t)(0x80u >> (x & 7));
    if (fmt == UNPAPER_CUDA_FMT_MONOBLACK) {
      *bytep = (uint8_t)(*bytep | mask); // bit set = white
    } else {
      *bytep = (uint8_t)(*bytep & (uint8_t)~mask); // bit clear = white
    }
  } break;
  default:
    break;
  }
}

// ============================================================================
// Interpolation helpers
// ============================================================================

static __device__ __forceinline__ uint8_t linear_scale(float x, uint8_t a,
                                                       uint8_t b) {
  const float fa = (1.0f - x) * (float)a;
  const float fb = x * (float)b;
  return (uint8_t)(fa + fb);
}

static __device__ __forceinline__ void
linear_pixel(float x, uint8_t ar, uint8_t ag, uint8_t ab, uint8_t br,
             uint8_t bg, uint8_t bb, uint8_t *or_, uint8_t *og, uint8_t *ob) {
  *or_ = linear_scale(x, ar, br);
  *og = linear_scale(x, ag, bg);
  *ob = linear_scale(x, ab, bb);
}

static __device__ __forceinline__ uint8_t cubic_scale(float factor, uint8_t a,
                                                      uint8_t b, uint8_t c,
                                                      uint8_t d) {
  const float fa = (float)a;
  const float fb = (float)b;
  const float fc = (float)c;
  const float fd = (float)d;
  const float f = factor;

  const float term = (fc - fa +
                      f * (2.0f * fa - 5.0f * fb + 4.0f * fc - fd +
                           f * (3.0f * (fb - fc) + fd - fa)));

  const int result = (int)(fb + 0.5f * f * term);
  return clip_uint8(result);
}

static __device__ __forceinline__ void
cubic_pixel(float factor, uint8_t a0r, uint8_t a0g, uint8_t a0b, uint8_t a1r,
            uint8_t a1g, uint8_t a1b, uint8_t a2r, uint8_t a2g, uint8_t a2b,
            uint8_t a3r, uint8_t a3g, uint8_t a3b, uint8_t *or_, uint8_t *og,
            uint8_t *ob) {
  *or_ = cubic_scale(factor, a0r, a1r, a2r, a3r);
  *og = cubic_scale(factor, a0g, a1g, a2g, a3g);
  *ob = cubic_scale(factor, a0b, a1b, a2b, a3b);
}

static __device__ __forceinline__ void
interp_nn(const uint8_t *src, int src_linesize, UnpaperCudaFormat fmt,
          int src_w, int src_h, float sx, float sy, uint8_t *r, uint8_t *g,
          uint8_t *b) {
  const int ix = (int)floorf(sx + 0.5f);
  const int iy = (int)floorf(sy + 0.5f);
  read_rgb_safe(src, src_linesize, fmt, src_w, src_h, ix, iy, r, g, b);
}

static __device__ __forceinline__ void
interp_nn_round(const uint8_t *src, int src_linesize, UnpaperCudaFormat fmt,
                int src_w, int src_h, float sx, float sy, uint8_t *r,
                uint8_t *g, uint8_t *b) {
  const int ix = (int)roundf(sx);
  const int iy = (int)roundf(sy);
  read_rgb_safe(src, src_linesize, fmt, src_w, src_h, ix, iy, r, g, b);
}

static __device__ __forceinline__ void
interp_bilinear(const uint8_t *src, int src_linesize, UnpaperCudaFormat fmt,
                int src_w, int src_h, float sx, float sy, uint8_t *r,
                uint8_t *g, uint8_t *b) {
  const int p1x = (int)floorf(sx);
  const int p1y = (int)floorf(sy);
  const int p2x = (int)ceilf(sx);
  const int p2y = (int)ceilf(sy);

  if (p2x < 0 || p2y < 0 || p2x >= src_w || p2y >= src_h) {
    read_rgb_safe(src, src_linesize, fmt, src_w, src_h, p1x, p1y, r, g, b);
    return;
  }

  if (p1x == p2x && p1y == p2y) {
    read_rgb_safe(src, src_linesize, fmt, src_w, src_h, p1x, p1y, r, g, b);
    return;
  }

  if (p1x == p2x) {
    uint8_t r1, g1, b1, r2, g2, b2;
    read_rgb_safe(src, src_linesize, fmt, src_w, src_h, p1x, p1y, &r1, &g1,
                  &b1);
    read_rgb_safe(src, src_linesize, fmt, src_w, src_h, p2x, p2y, &r2, &g2,
                  &b2);
    linear_pixel(sx - (float)p1x, r1, g1, b1, r2, g2, b2, r, g, b);
    return;
  }

  if (p1y == p2y) {
    uint8_t r1, g1, b1, r2, g2, b2;
    read_rgb_safe(src, src_linesize, fmt, src_w, src_h, p1x, p1y, &r1, &g1,
                  &b1);
    read_rgb_safe(src, src_linesize, fmt, src_w, src_h, p2x, p2y, &r2, &g2,
                  &b2);
    linear_pixel(sy - (float)p1y, r1, g1, b1, r2, g2, b2, r, g, b);
    return;
  }

  uint8_t r11, g11, b11, r21, g21, b21, r12, g12, b12, r22, g22, b22;
  read_rgb_safe(src, src_linesize, fmt, src_w, src_h, p1x, p1y, &r11, &g11,
                &b11);
  read_rgb_safe(src, src_linesize, fmt, src_w, src_h, p2x, p1y, &r21, &g21,
                &b21);
  read_rgb_safe(src, src_linesize, fmt, src_w, src_h, p1x, p2y, &r12, &g12,
                &b12);
  read_rgb_safe(src, src_linesize, fmt, src_w, src_h, p2x, p2y, &r22, &g22,
                &b22);

  uint8_t rh1, gh1, bh1, rh2, gh2, bh2;
  linear_pixel(sx - (float)p1x, r11, g11, b11, r21, g21, b21, &rh1, &gh1, &bh1);
  linear_pixel(sx - (float)p1x, r12, g12, b12, r22, g22, b22, &rh2, &gh2, &bh2);
  linear_pixel(sy - (float)p1y, rh1, gh1, bh1, rh2, gh2, bh2, r, g, b);
}

static __device__ __forceinline__ void
interp_bicubic(const uint8_t *src, int src_linesize, UnpaperCudaFormat fmt,
               int src_w, int src_h, float sx, float sy, uint8_t *r, uint8_t *g,
               uint8_t *b) {
  const int px = (int)sx;
  const int py = (int)sy;

  uint8_t row_r[4];
  uint8_t row_g[4];
  uint8_t row_b[4];

  const float fx = sx - (float)px;
  const float fy = sy - (float)py;

  for (int i = -1; i < 3; i++) {
    uint8_t q0r, q0g, q0b, q1r, q1g, q1b, q2r, q2g, q2b, q3r, q3g, q3b;
    read_rgb_safe(src, src_linesize, fmt, src_w, src_h, px - 1, py + i, &q0r,
                  &q0g, &q0b);
    read_rgb_safe(src, src_linesize, fmt, src_w, src_h, px, py + i, &q1r, &q1g,
                  &q1b);
    read_rgb_safe(src, src_linesize, fmt, src_w, src_h, px + 1, py + i, &q2r,
                  &q2g, &q2b);
    read_rgb_safe(src, src_linesize, fmt, src_w, src_h, px + 2, py + i, &q3r,
                  &q3g, &q3b);

    cubic_pixel(fx, q0r, q0g, q0b, q1r, q1g, q1b, q2r, q2g, q2b, q3r, q3g, q3b,
                &row_r[i + 1], &row_g[i + 1], &row_b[i + 1]);
  }

  cubic_pixel(fy, row_r[0], row_g[0], row_b[0], row_r[1], row_g[1], row_b[1],
              row_r[2], row_g[2], row_b[2], row_r[3], row_g[3], row_b[3], r, g,
              b);
}

// ============================================================================
// Integral image helper
// ============================================================================

// Helper: compute sum from NPP integral image for rectangle (x0,y0) to (x1,y1)
// NPP format: I[y,x] = sum of pixels from (0,0) to (x-1,y-1) EXCLUSIVE
// First row (y=0) and first column (x=0) are always zero.
// Sum of rect (x0,y0)-(x1,y1) = I[y1+1,x1+1] - I[y0,x1+1] - I[y1+1,x0] +
// I[y0,x0]
//
// IMPORTANT: The integral image is now (img_w+1) x (img_h+1) due to padding.
// This ensures all boundary accesses are valid for tiles within the original
// image.
static __device__ __forceinline__ int64_t
npp_integral_rect_sum(const int32_t *integral, int integral_step_i32, int img_w,
                      int img_h, int x0, int y0, int x1, int y1) {
  // Clamp coordinates to original image bounds
  if (x0 < 0)
    x0 = 0;
  if (y0 < 0)
    y0 = 0;
  if (x1 >= img_w)
    x1 = img_w - 1;
  if (y1 >= img_h)
    y1 = img_h - 1;
  if (x0 > x1 || y0 > y1) {
    return 0;
  }

  // With padded integral (img_w+1 x img_h+1), all accesses are now valid:
  // - y1+1 <= img_h (since y1 <= img_h-1), integral height = img_h+1, valid
  // - x1+1 <= img_w (since x1 <= img_w-1), integral width = img_w+1, valid

  // Bottom-right: I[y1+1, x1+1]
  int64_t br = integral[(y1 + 1) * integral_step_i32 + (x1 + 1)];

  // Top-right: I[y0, x1+1]
  int64_t tr = integral[y0 * integral_step_i32 + (x1 + 1)];

  // Bottom-left: I[y1+1, x0]
  int64_t bl = integral[(y1 + 1) * integral_step_i32 + x0];

  // Top-left: I[y0, x0]
  int64_t tl = integral[y0 * integral_step_i32 + x0];

  return br - tr - bl + tl;
}

// ============================================================================
// Point struct for flood-fill and output blocks
// ============================================================================

typedef struct {
  int x;
  int y;
} UnpaperCudaPoint;

// Output structure for blurfilter blocks
typedef struct {
  int x;
  int y;
} UnpaperBlurfilterBlock;

// Output structure for grayfilter tiles
typedef struct {
  int x;
  int y;
} UnpaperGrayfilterTile;

#endif // CUDA_KERNELS_COMMON_CUH
