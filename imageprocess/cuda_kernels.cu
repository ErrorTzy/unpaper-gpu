// SPDX-FileCopyrightText: 2025 The unpaper authors
//
// SPDX-License-Identifier: GPL-2.0-only

#include <stdint.h>
#include <math.h>

#include "imageprocess/cuda_kernels_format.h"

static __device__ __forceinline__ uint8_t grayscale_u8(uint8_t r, uint8_t g,
                                                       uint8_t b) {
  return (uint8_t)(((uint32_t)r + (uint32_t)g + (uint32_t)b) / 3u);
}

static __device__ __forceinline__ void read_rgb(const uint8_t *src,
                                                int src_linesize,
                                                UnpaperCudaFormat fmt, int x,
                                                int y, uint8_t *r, uint8_t *g,
                                                uint8_t *b) {
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
    const bool is_white = (fmt == UNPAPER_CUDA_FMT_MONOBLACK) ? bit_set
                                                              : (!bit_set);
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

static __device__ __forceinline__ void read_rgb_safe(
    const uint8_t *src, int src_linesize, UnpaperCudaFormat fmt, int src_w,
    int src_h, int x, int y, uint8_t *r, uint8_t *g, uint8_t *b) {
  if (x < 0 || y < 0 || x >= src_w || y >= src_h) {
    *r = 255u;
    *g = 255u;
    *b = 255u;
    return;
  }
  read_rgb(src, src_linesize, fmt, x, y, r, g, b);
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

static __device__ __forceinline__ uint8_t linear_scale(float x, uint8_t a,
                                                       uint8_t b) {
  const float fa = (1.0f - x) * (float)a;
  const float fb = x * (float)b;
  return (uint8_t)(fa + fb);
}

static __device__ __forceinline__ void linear_pixel(float x, uint8_t ar,
                                                    uint8_t ag, uint8_t ab,
                                                    uint8_t br, uint8_t bg,
                                                    uint8_t bb, uint8_t *or_,
                                                    uint8_t *og,
                                                    uint8_t *ob) {
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

  const float term =
      (fc - fa +
       f * (2.0f * fa - 5.0f * fb + 4.0f * fc - fd +
            f * (3.0f * (fb - fc) + fd - fa)));

  const int result = (int)(fb + 0.5f * f * term);
  return clip_uint8(result);
}

static __device__ __forceinline__ void cubic_pixel(float factor, uint8_t a0r,
                                                   uint8_t a0g, uint8_t a0b,
                                                   uint8_t a1r, uint8_t a1g,
                                                   uint8_t a1b, uint8_t a2r,
                                                   uint8_t a2g, uint8_t a2b,
                                                   uint8_t a3r, uint8_t a3g,
                                                   uint8_t a3b, uint8_t *or_,
                                                   uint8_t *og, uint8_t *ob) {
  *or_ = cubic_scale(factor, a0r, a1r, a2r, a3r);
  *og = cubic_scale(factor, a0g, a1g, a2g, a3g);
  *ob = cubic_scale(factor, a0b, a1b, a2b, a3b);
}

static __device__ __forceinline__ void interp_nn(const uint8_t *src,
                                                 int src_linesize,
                                                 UnpaperCudaFormat fmt,
                                                 int src_w, int src_h,
                                                 float sx, float sy,
                                                 uint8_t *r, uint8_t *g,
                                                 uint8_t *b) {
  const int ix = (int)floorf(sx + 0.5f);
  const int iy = (int)floorf(sy + 0.5f);
  read_rgb_safe(src, src_linesize, fmt, src_w, src_h, ix, iy, r, g, b);
}

static __device__ __forceinline__ void interp_bilinear(
    const uint8_t *src, int src_linesize, UnpaperCudaFormat fmt, int src_w,
    int src_h, float sx, float sy, uint8_t *r, uint8_t *g, uint8_t *b) {
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
  linear_pixel(sx - (float)p1x, r11, g11, b11, r21, g21, b21, &rh1, &gh1,
               &bh1);
  linear_pixel(sx - (float)p1x, r12, g12, b12, r22, g22, b22, &rh2, &gh2,
               &bh2);
  linear_pixel(sy - (float)p1y, rh1, gh1, bh1, rh2, gh2, bh2, r, g, b);
}

static __device__ __forceinline__ void interp_bicubic(
    const uint8_t *src, int src_linesize, UnpaperCudaFormat fmt, int src_w,
    int src_h, float sx, float sy, uint8_t *r, uint8_t *g, uint8_t *b) {
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

    cubic_pixel(fx, q0r, q0g, q0b, q1r, q1g, q1b, q2r, q2g, q2b, q3r, q3g,
                q3b, &row_r[i + 1], &row_g[i + 1], &row_b[i + 1]);
  }

  cubic_pixel(fy, row_r[0], row_g[0], row_b[0], row_r[1], row_g[1], row_b[1],
              row_r[2], row_g[2], row_b[2], row_r[3], row_g[3], row_b[3], r, g,
              b);
}

static __device__ __forceinline__ void write_pixel(uint8_t *dst,
                                                   int dst_linesize,
                                                   UnpaperCudaFormat fmt, int x,
                                                   int y, uint8_t r, uint8_t g,
                                                   uint8_t b) {
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

static __device__ __forceinline__ uint8_t darkness_inverse_u8(uint8_t r,
                                                              uint8_t g,
                                                              uint8_t b);

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
    const bool is_white = (fmt == UNPAPER_CUDA_FMT_MONOBLACK) ? bit_set
                                                              : (!bit_set);
    return is_white ? 255u : 0u;
  }
  default:
    return 255u;
  }
}

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

static __device__ __forceinline__ void interp_nn_round(
    const uint8_t *src, int src_linesize, UnpaperCudaFormat fmt, int src_w,
    int src_h, float sx, float sy, uint8_t *r, uint8_t *g, uint8_t *b) {
  const int ix = (int)roundf(sx);
  const int iy = (int)roundf(sy);
  read_rgb_safe(src, src_linesize, fmt, src_w, src_h, ix, iy, r, g, b);
}

extern "C" __global__ void unpaper_rotate_bytes(
    const uint8_t *src, int src_linesize, uint8_t *dst, int dst_linesize,
    int fmt, int src_w, int src_h, int dst_w, int dst_h, float src_center_x,
    float src_center_y, float dst_center_x, float dst_center_y, float cosval,
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

extern "C" __global__ void unpaper_rotate_mono(
    const uint8_t *src, int src_linesize, int src_fmt, uint8_t *dst,
    int dst_linesize, int dst_fmt, int src_w, int src_h, int dst_w, int dst_h,
    float src_center_x, float src_center_y, float dst_center_x,
    float dst_center_y, float cosval, float sinval, int interp_type,
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
    const bool bit_set = ((UnpaperCudaFormat)dst_fmt == UNPAPER_CUDA_FMT_MONOWHITE)
                             ? pixel_black
                             : !pixel_black;
    if (bit_set) {
      out = (uint8_t)(out | (uint8_t)(0x80u >> bit));
    }
  }

  dst[(size_t)y * (size_t)dst_linesize + (size_t)byte_x] = out;
}

extern "C" __global__ void unpaper_wipe_rect_bytes(
    uint8_t *dst, int dst_linesize, int x0, int y0, int x1, int y1,
    int bytes_per_pixel, uint8_t c0, uint8_t c1, uint8_t c2) {
  const int rx = x1 - x0 + 1;
  const int ry = y1 - y0 + 1;
  const int x = x0 + (int)(blockIdx.x * blockDim.x + threadIdx.x);
  const int y = y0 + (int)(blockIdx.y * blockDim.y + threadIdx.y);
  if (x < x0 || y < y0 || x >= x0 + rx || y >= y0 + ry) {
    return;
  }

  uint8_t *p = dst + (size_t)y * (size_t)dst_linesize +
               (size_t)x * (size_t)bytes_per_pixel;
  if (bytes_per_pixel == 1) {
    p[0] = c0;
  } else if (bytes_per_pixel == 2) {
    p[0] = c0;
    p[1] = c1;
  } else if (bytes_per_pixel == 3) {
    p[0] = c0;
    p[1] = c1;
    p[2] = c2;
  }
}

extern "C" __global__ void unpaper_wipe_rect_mono(uint8_t *dst, int dst_linesize,
                                                  int x0, int y0, int x1,
                                                  int y1, uint8_t bit_set) {
  const int first_byte = x0 / 8;
  const int last_byte = x1 / 8;
  const int bytes_span = last_byte - first_byte + 1;

  const int byte_off = (int)(blockIdx.x * blockDim.x + threadIdx.x);
  const int y = y0 + (int)(blockIdx.y * blockDim.y + threadIdx.y);
  if (byte_off < 0 || byte_off >= bytes_span || y < y0 || y > y1) {
    return;
  }

  const int byte_x = first_byte + byte_off;
  uint8_t mask = 0;
  const int byte_start_x = byte_x * 8;
  for (int bit = 0; bit < 8; bit++) {
    const int x = byte_start_x + bit;
    if (x < x0 || x > x1) {
      continue;
    }
    mask |= (uint8_t)(0x80u >> bit);
  }

  uint8_t *p = dst + (size_t)y * (size_t)dst_linesize + (size_t)byte_x;
  const uint8_t orig = *p;
  if (bit_set) {
    *p = (uint8_t)(orig | mask);
  } else {
    *p = (uint8_t)(orig & (uint8_t)(~mask));
  }
}

extern "C" __global__ void unpaper_copy_rect_to_bytes(
    const uint8_t *src, int src_linesize, int src_fmt, uint8_t *dst,
    int dst_linesize, int dst_fmt, int src_x0, int src_y0, int dst_x0,
    int dst_y0, int w, int h) {
  const int x = (int)(blockIdx.x * blockDim.x + threadIdx.x);
  const int y = (int)(blockIdx.y * blockDim.y + threadIdx.y);
  if (x < 0 || y < 0 || x >= w || y >= h) {
    return;
  }

  uint8_t r = 255u, g = 255u, b = 255u;
  read_rgb(src, src_linesize, (UnpaperCudaFormat)src_fmt, src_x0 + x,
           src_y0 + y, &r, &g, &b);
  write_pixel(dst, dst_linesize, (UnpaperCudaFormat)dst_fmt, dst_x0 + x,
              dst_y0 + y, r, g, b);
}

extern "C" __global__ void unpaper_copy_rect_to_mono(
    const uint8_t *src, int src_linesize, int src_fmt, uint8_t *dst,
    int dst_linesize, int dst_fmt, int src_x0, int src_y0, int dst_x0,
    int dst_y0, int w, int h, uint8_t abs_black_threshold) {
  const int first_byte = dst_x0 / 8;
  const int last_byte = (dst_x0 + w - 1) / 8;
  const int bytes_span = last_byte - first_byte + 1;

  const int byte_off = (int)(blockIdx.x * blockDim.x + threadIdx.x);
  const int y = (int)(blockIdx.y * blockDim.y + threadIdx.y);
  if (byte_off < 0 || byte_off >= bytes_span || y < 0 || y >= h) {
    return;
  }

  const int out_y = dst_y0 + y;
  const int byte_x = first_byte + byte_off;
  const int byte_start_x = byte_x * 8;

  uint8_t v = dst[(size_t)out_y * (size_t)dst_linesize + (size_t)byte_x];
  for (int bit = 0; bit < 8; bit++) {
    const int out_x = byte_start_x + bit;
    if (out_x < dst_x0 || out_x >= dst_x0 + w) {
      continue;
    }

    const int rel_x = out_x - dst_x0;
    const int src_x = src_x0 + rel_x;
    const int src_y = src_y0 + y;

    uint8_t r = 255u, g = 255u, b = 255u;
    read_rgb(src, src_linesize, (UnpaperCudaFormat)src_fmt, src_x, src_y, &r,
             &g, &b);

    const uint8_t gray = grayscale_u8(r, g, b);
    const bool pixel_black = gray < abs_black_threshold;

    const bool bit_set =
        ((UnpaperCudaFormat)dst_fmt == UNPAPER_CUDA_FMT_MONOWHITE) ? pixel_black
                                                                   : !pixel_black;
    const uint8_t mask = (uint8_t)(0x80u >> bit);
    if (bit_set) {
      v = (uint8_t)(v | mask);
    } else {
      v = (uint8_t)(v & (uint8_t)(~mask));
    }
  }

  dst[(size_t)out_y * (size_t)dst_linesize + (size_t)byte_x] = v;
}

extern "C" __global__ void unpaper_mirror_bytes(const uint8_t *src,
                                                int src_linesize, uint8_t *dst,
                                                int dst_linesize, int fmt,
                                                int width, int height, int do_h,
                                                int do_v) {
  const int x = (int)(blockIdx.x * blockDim.x + threadIdx.x);
  const int y = (int)(blockIdx.y * blockDim.y + threadIdx.y);
  if (x < 0 || y < 0 || x >= width || y >= height) {
    return;
  }

  const int sx = do_h ? (width - 1 - x) : x;
  const int sy = do_v ? (height - 1 - y) : y;

  uint8_t r = 255u, g = 255u, b = 255u;
  read_rgb(src, src_linesize, (UnpaperCudaFormat)fmt, sx, sy, &r, &g, &b);
  write_pixel(dst, dst_linesize, (UnpaperCudaFormat)fmt, x, y, r, g, b);
}

extern "C" __global__ void unpaper_mirror_mono(const uint8_t *src,
                                               int src_linesize, uint8_t *dst,
                                               int dst_linesize, int width,
                                               int height, int do_h, int do_v) {
  const int bytes_per_row = (width + 7) / 8;
  const int byte_x = (int)(blockIdx.x * blockDim.x + threadIdx.x);
  const int y = (int)(blockIdx.y * blockDim.y + threadIdx.y);
  if (byte_x < 0 || byte_x >= bytes_per_row || y < 0 || y >= height) {
    return;
  }

  uint8_t out = dst[(size_t)y * (size_t)dst_linesize + (size_t)byte_x];
  const int dst_byte_start_x = byte_x * 8;
  for (int bit = 0; bit < 8; bit++) {
    const int x = dst_byte_start_x + bit;
    if (x >= width) {
      continue;
    }
    const int sx = do_h ? (width - 1 - x) : x;
    const int sy = do_v ? (height - 1 - y) : y;

    const uint8_t src_byte =
        src[(size_t)sy * (size_t)src_linesize + (size_t)(sx / 8)];
    const uint8_t src_mask = (uint8_t)(0x80u >> (sx & 7));
    const bool src_set = (src_byte & src_mask) != 0;

    const uint8_t dst_mask = (uint8_t)(0x80u >> bit);
    if (src_set) {
      out = (uint8_t)(out | dst_mask);
    } else {
      out = (uint8_t)(out & (uint8_t)(~dst_mask));
    }
  }
  dst[(size_t)y * (size_t)dst_linesize + (size_t)byte_x] = out;
}

extern "C" __global__ void unpaper_rotate90_bytes(
    const uint8_t *src, int src_linesize, uint8_t *dst, int dst_linesize,
    int fmt, int src_w, int src_h, int direction) {
  const int x = (int)(blockIdx.x * blockDim.x + threadIdx.x);
  const int y = (int)(blockIdx.y * blockDim.y + threadIdx.y);
  if (x < 0 || y < 0 || x >= src_w || y >= src_h) {
    return;
  }

  const int dst_w = src_h;
  const int dst_h = src_w;
  (void)dst_w;
  (void)dst_h;

  int dx = 0;
  int dy = 0;
  if (direction > 0) {
    dx = src_h - 1 - y;
    dy = x;
  } else {
    dx = y;
    dy = src_w - 1 - x;
  }

  uint8_t r = 255u, g = 255u, b = 255u;
  read_rgb(src, src_linesize, (UnpaperCudaFormat)fmt, x, y, &r, &g, &b);
  write_pixel(dst, dst_linesize, (UnpaperCudaFormat)fmt, dx, dy, r, g, b);
}

extern "C" __global__ void unpaper_rotate90_mono(const uint8_t *src,
                                                 int src_linesize, uint8_t *dst,
                                                 int dst_linesize, int src_w,
                                                 int src_h, int direction) {
  const int dst_w = src_h;
  const int dst_h = src_w;
  const int dst_bytes_per_row = (dst_w + 7) / 8;

  const int byte_x = (int)(blockIdx.x * blockDim.x + threadIdx.x);
  const int dy = (int)(blockIdx.y * blockDim.y + threadIdx.y);
  if (byte_x < 0 || byte_x >= dst_bytes_per_row || dy < 0 || dy >= dst_h) {
    return;
  }

  const int dst_byte_start_x = byte_x * 8;
  uint8_t out = 0;
  for (int bit = 0; bit < 8; bit++) {
    const int dx = dst_byte_start_x + bit;
    if (dx >= dst_w) {
      continue;
    }

    int sx = 0;
    int sy = 0;
    if (direction > 0) {
      sx = dy;
      sy = src_h - 1 - dx;
    } else {
      sy = dx;
      sx = src_w - 1 - dy;
    }

    const uint8_t src_byte =
        src[(size_t)sy * (size_t)src_linesize + (size_t)(sx / 8)];
    const uint8_t src_mask = (uint8_t)(0x80u >> (sx & 7));
    const bool src_set = (src_byte & src_mask) != 0;

    if (src_set) {
      out = (uint8_t)(out | (uint8_t)(0x80u >> bit));
    }
  }

  dst[(size_t)dy * (size_t)dst_linesize + (size_t)byte_x] = out;
}

extern "C" __global__ void unpaper_stretch_bytes(
    const uint8_t *src, int src_linesize, uint8_t *dst, int dst_linesize,
    int fmt, int src_w, int src_h, int dst_w, int dst_h, int interp_type) {
  const int x = (int)(blockIdx.x * blockDim.x + threadIdx.x);
  const int y = (int)(blockIdx.y * blockDim.y + threadIdx.y);
  if (x < 0 || y < 0 || x >= dst_w || y >= dst_h) {
    return;
  }

  const float hratio = (float)src_w / (float)dst_w;
  const float vratio = (float)src_h / (float)dst_h;

  const float sx = (float)x * hratio;
  const float sy = (float)y * vratio;

  uint8_t r = 255u, g = 255u, b = 255u;
  const UnpaperCudaFormat f = (UnpaperCudaFormat)fmt;
  if (interp_type == 0) {
    interp_nn(src, src_linesize, f, src_w, src_h, sx, sy, &r, &g, &b);
  } else if (interp_type == 1) {
    interp_bilinear(src, src_linesize, f, src_w, src_h, sx, sy, &r, &g, &b);
  } else {
    interp_bicubic(src, src_linesize, f, src_w, src_h, sx, sy, &r, &g, &b);
  }

  write_pixel(dst, dst_linesize, f, x, y, r, g, b);
}

extern "C" __global__ void unpaper_stretch_mono(
    const uint8_t *src, int src_linesize, int src_fmt, uint8_t *dst,
    int dst_linesize, int dst_fmt, int src_w, int src_h, int dst_w, int dst_h,
    int interp_type, uint8_t abs_black_threshold) {
  const int bytes_per_row = (dst_w + 7) / 8;
  const int byte_x = (int)(blockIdx.x * blockDim.x + threadIdx.x);
  const int y = (int)(blockIdx.y * blockDim.y + threadIdx.y);
  if (byte_x < 0 || byte_x >= bytes_per_row || y < 0 || y >= dst_h) {
    return;
  }

  const float hratio = (float)src_w / (float)dst_w;
  const float vratio = (float)src_h / (float)dst_h;

  const int dst_byte_start_x = byte_x * 8;
  uint8_t out = 0;
  for (int bit = 0; bit < 8; bit++) {
    const int x = dst_byte_start_x + bit;
    if (x >= dst_w) {
      continue;
    }

    const float sx = (float)x * hratio;
    const float sy = (float)y * vratio;

    uint8_t r = 255u, g = 255u, b = 255u;
    const UnpaperCudaFormat f = (UnpaperCudaFormat)src_fmt;
    if (interp_type == 0) {
      interp_nn(src, src_linesize, f, src_w, src_h, sx, sy, &r, &g, &b);
    } else if (interp_type == 1) {
      interp_bilinear(src, src_linesize, f, src_w, src_h, sx, sy, &r, &g, &b);
    } else {
      interp_bicubic(src, src_linesize, f, src_w, src_h, sx, sy, &r, &g, &b);
    }

    const uint8_t gray = grayscale_u8(r, g, b);
    const bool pixel_black = gray < abs_black_threshold;
    const bool bit_set = ((UnpaperCudaFormat)dst_fmt == UNPAPER_CUDA_FMT_MONOWHITE)
                             ? pixel_black
                             : !pixel_black;
    if (bit_set) {
      out = (uint8_t)(out | (uint8_t)(0x80u >> bit));
    }
  }

  dst[(size_t)y * (size_t)dst_linesize + (size_t)byte_x] = out;
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

static __device__ __forceinline__ void set_pixel_white_safe(
    uint8_t *dst, int dst_linesize, UnpaperCudaFormat fmt, int dst_w, int dst_h,
    int x, int y) {
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

extern "C" __global__ void unpaper_count_brightness_range(
    const uint8_t *src, int src_linesize, int src_fmt, int src_w, int src_h,
    int x0, int y0, int x1, int y1, uint8_t min_brightness,
    uint8_t max_brightness, unsigned long long *out_count) {
  const int w = x1 - x0 + 1;
  const int h = y1 - y0 + 1;
  if (w <= 0 || h <= 0) {
    return;
  }

  const unsigned long long total = (unsigned long long)w * (unsigned long long)h;
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

extern "C" __global__ void unpaper_sum_lightness_rect(
    const uint8_t *src, int src_linesize, int src_fmt, int src_w, int src_h,
    int x0, int y0, int x1, int y1, unsigned long long *out_sum) {
  const int w = x1 - x0 + 1;
  const int h = y1 - y0 + 1;
  if (w <= 0 || h <= 0) {
    return;
  }

  const unsigned long long total = (unsigned long long)w * (unsigned long long)h;
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

extern "C" __global__ void unpaper_sum_grayscale_rect(
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

  const unsigned long long total = (unsigned long long)w * (unsigned long long)h;
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

static __device__ __forceinline__ bool rect_contains_point(
    const int32_t *rects, int rect_index, int x, int y) {
  const int32_t x0 = rects[rect_index * 4 + 0];
  const int32_t y0 = rects[rect_index * 4 + 1];
  const int32_t x1 = rects[rect_index * 4 + 2];
  const int32_t y1 = rects[rect_index * 4 + 3];
  return x >= x0 && x <= x1 && y >= y0 && y <= y1;
}

extern "C" __global__ void unpaper_apply_masks_bytes(
    uint8_t *img, int img_linesize, int img_fmt, int img_w, int img_h,
    const int32_t *rects, int rect_count, uint8_t r, uint8_t g, uint8_t b) {
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

extern "C" __global__ void unpaper_apply_masks_mono(
    uint8_t *img, int img_linesize, int img_fmt, int img_w, int img_h,
    const int32_t *rects, int rect_count, uint8_t bit_value) {
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

static __device__ __forceinline__ uint8_t get_grayscale_safe(
    const uint8_t *src, int src_linesize, UnpaperCudaFormat fmt, int src_w,
    int src_h, int x, int y) {
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

static __device__ __forceinline__ bool noisefilter_is_dark(
    const uint8_t *img, int img_linesize, UnpaperCudaFormat fmt, int w, int h,
    int x, int y, uint8_t min_white_level) {
  const uint8_t darkness =
      get_darkness_inverse_safe(img, img_linesize, fmt, w, h, x, y);
  return darkness < min_white_level;
}

extern "C" __global__ void unpaper_noisefilter_build_labels(
    const uint8_t *img, int img_linesize, int img_fmt, int w, int h,
    uint8_t min_white_level, uint32_t *labels) {
  const int x = (int)(blockIdx.x * blockDim.x + threadIdx.x);
  const int y = (int)(blockIdx.y * blockDim.y + threadIdx.y);
  if (x >= w || y >= h) {
    return;
  }

  const int idx = y * w + x;
  const UnpaperCudaFormat fmt = (UnpaperCudaFormat)img_fmt;
  labels[idx] = noisefilter_is_dark(img, img_linesize, fmt, w, h, x, y,
                                    min_white_level)
                    ? (uint32_t)(idx + 1)
                    : 0u;
}

extern "C" __global__ void unpaper_noisefilter_propagate(
    const uint32_t *labels_in, uint32_t *labels_out, int w, int h,
    int *changed) {
  const int x = (int)(blockIdx.x * blockDim.x + threadIdx.x);
  const int y = (int)(blockIdx.y * blockDim.y + threadIdx.y);
  if (x >= w || y >= h) {
    return;
  }

  const int idx = y * w + x;
  const uint32_t self = labels_in[idx];
  if (self == 0u) {
    labels_out[idx] = 0u;
    return;
  }

  uint32_t min_label = self;
  for (int dy = -1; dy <= 1; dy++) {
    const int yy = y + dy;
    if (yy < 0 || yy >= h) {
      continue;
    }
    for (int dx = -1; dx <= 1; dx++) {
      const int xx = x + dx;
      if (xx < 0 || xx >= w) {
        continue;
      }
      const uint32_t neighbor = labels_in[yy * w + xx];
      if (neighbor != 0u && neighbor < min_label) {
        min_label = neighbor;
      }
    }
  }

  labels_out[idx] = min_label;
  if (min_label != self) {
    atomicExch(changed, 1);
  }
}

extern "C" __global__ void unpaper_noisefilter_count(const uint32_t *labels,
                                                     int num_pixels,
                                                     uint32_t *counts) {
  const int idx = (int)(blockIdx.x * blockDim.x + threadIdx.x);
  if (idx >= num_pixels) {
    return;
  }

  const uint32_t label = labels[idx];
  if (label == 0u) {
    return;
  }
  atomicAdd(&counts[label], 1u);
}

extern "C" __global__ void unpaper_noisefilter_apply(
    uint8_t *img, int img_linesize, int img_fmt, int w, int h,
    const uint32_t *labels, const uint32_t *counts,
    unsigned long long intensity) {
  const int x = (int)(blockIdx.x * blockDim.x + threadIdx.x);
  const int y = (int)(blockIdx.y * blockDim.y + threadIdx.y);
  if (x >= w || y >= h) {
    return;
  }

  const int idx = y * w + x;
  const uint32_t label = labels[idx];
  if (label == 0u) {
    return;
  }

  const uint32_t comp_size = counts[label];
  if ((unsigned long long)comp_size <= intensity) {
    set_pixel_white_safe(img, img_linesize, (UnpaperCudaFormat)img_fmt, w, h, x,
                         y);
  }
}

// Apply noisefilter mask directly on GPU:
// If mask is 0 (small component removed) and pixel is dark, set to white.
extern "C" __global__ void unpaper_noisefilter_apply_mask(
    uint8_t *img, int img_linesize, int img_fmt, int w, int h,
    const uint8_t *mask, int mask_linesize, uint8_t min_white_level) {
  const int x = (int)(blockIdx.x * blockDim.x + threadIdx.x);
  const int y = (int)(blockIdx.y * blockDim.y + threadIdx.y);
  if (x >= w || y >= h) {
    return;
  }

  // Check mask: if non-zero, pixel is in a large component - keep as-is
  const uint8_t mask_val = mask[y * mask_linesize + x];
  if (mask_val != 0) {
    return;
  }

  // Mask is 0 - check if pixel is dark
  const UnpaperCudaFormat fmt = (UnpaperCudaFormat)img_fmt;
  uint8_t lightness;

  if (fmt == UNPAPER_CUDA_FMT_GRAY8) {
    lightness = img[y * img_linesize + x];
  } else if (fmt == UNPAPER_CUDA_FMT_Y400A) {
    lightness = img[y * img_linesize + x * 2];
  } else if (fmt == UNPAPER_CUDA_FMT_RGB24) {
    const uint8_t *p = &img[y * img_linesize + x * 3];
    const uint8_t max_rg = (p[0] > p[1]) ? p[0] : p[1];
    const uint8_t min_rg = (p[0] < p[1]) ? p[0] : p[1];
    const uint8_t max_rgb = (max_rg > p[2]) ? max_rg : p[2];
    const uint8_t min_rgb = (min_rg < p[2]) ? min_rg : p[2];
    lightness = (uint8_t)((max_rgb + min_rgb) / 2);
  } else {
    return; // Unsupported format
  }

  // If dark, set to white
  if (lightness < min_white_level) {
    if (fmt == UNPAPER_CUDA_FMT_GRAY8) {
      img[y * img_linesize + x] = 255;
    } else if (fmt == UNPAPER_CUDA_FMT_Y400A) {
      img[y * img_linesize + x * 2] = 255; // Set Y to white, keep alpha
    } else if (fmt == UNPAPER_CUDA_FMT_RGB24) {
      uint8_t *p = &img[y * img_linesize + x * 3];
      p[0] = p[1] = p[2] = 255;
    }
  }
}

typedef struct {
  int x;
  int y;
} UnpaperCudaPoint;

static __device__ __forceinline__ unsigned long long flood_fill_line_cuda(
    uint8_t *img, int img_linesize, UnpaperCudaFormat fmt, int w, int h, int px,
    int py, int step_x, int step_y, uint8_t mask_min, uint8_t mask_max,
    unsigned long long intensity) {
  unsigned long long distance = 0;
  unsigned long long intensityCount = 1ull;

  int x = px;
  int y = py;
  while (true) {
    x += step_x;
    y += step_y;

    if (x < 0 || y < 0 || x >= w || y >= h) {
      return distance;
    }

    const uint8_t pixel =
        get_grayscale_safe(img, img_linesize, fmt, w, h, x, y);
    if (pixel >= mask_min && pixel <= mask_max) {
      intensityCount = intensity;
    } else {
      intensityCount--;
    }

    if (intensityCount == 0ull) {
      return distance;
    }

    set_pixel_white_safe(img, img_linesize, fmt, w, h, x, y);
    distance++;
  }
}

static __device__ __forceinline__ void stack_push(UnpaperCudaPoint *stack,
                                                  int *top, int cap, int x,
                                                  int y) {
  const int t = *top;
  if (t >= cap) {
    return;
  }
  stack[t] = (UnpaperCudaPoint){.x = x, .y = y};
  *top = t + 1;
}

static __device__ __forceinline__ bool stack_pop(UnpaperCudaPoint *stack,
                                                 int *top,
                                                 UnpaperCudaPoint *out) {
  const int t = *top;
  if (t <= 0) {
    return false;
  }
  *top = t - 1;
  *out = stack[t - 1];
  return true;
}

static __device__ __forceinline__ void flood_fill_cuda(
    uint8_t *img, int img_linesize, UnpaperCudaFormat fmt, int w, int h, int sx,
    int sy, uint8_t mask_min, uint8_t mask_max, unsigned long long intensity,
    UnpaperCudaPoint *stack, int stack_cap) {
  const uint8_t seed = get_grayscale_safe(img, img_linesize, fmt, w, h, sx, sy);
  if (seed < mask_min || seed > mask_max) {
    return;
  }

  int top = 0;
  stack_push(stack, &top, stack_cap, sx, sy);

  UnpaperCudaPoint p;
  while (stack_pop(stack, &top, &p)) {
    const uint8_t pixel =
        get_grayscale_safe(img, img_linesize, fmt, w, h, p.x, p.y);
    if (pixel < mask_min || pixel > mask_max) {
      continue;
    }

    set_pixel_white_safe(img, img_linesize, fmt, w, h, p.x, p.y);

    const unsigned long long left = flood_fill_line_cuda(
        img, img_linesize, fmt, w, h, p.x, p.y, -1, 0, mask_min, mask_max,
        intensity);
    const unsigned long long up = flood_fill_line_cuda(
        img, img_linesize, fmt, w, h, p.x, p.y, 0, -1, mask_min, mask_max,
        intensity);
    const unsigned long long right = flood_fill_line_cuda(
        img, img_linesize, fmt, w, h, p.x, p.y, 1, 0, mask_min, mask_max,
        intensity);
    const unsigned long long down = flood_fill_line_cuda(
        img, img_linesize, fmt, w, h, p.x, p.y, 0, 1, mask_min, mask_max,
        intensity);

    int qx = p.x;
    for (unsigned long long d = 0; d < left; d++) {
      qx -= 1;
      stack_push(stack, &top, stack_cap, qx, p.y + 1);
      stack_push(stack, &top, stack_cap, qx, p.y - 1);
    }

    int qy = p.y;
    for (unsigned long long d = 0; d < up; d++) {
      qy -= 1;
      stack_push(stack, &top, stack_cap, p.x + 1, qy);
      stack_push(stack, &top, stack_cap, p.x - 1, qy);
    }

    qx = p.x;
    for (unsigned long long d = 0; d < right; d++) {
      qx += 1;
      stack_push(stack, &top, stack_cap, qx, p.y + 1);
      stack_push(stack, &top, stack_cap, qx, p.y - 1);
    }

    qy = p.y;
    for (unsigned long long d = 0; d < down; d++) {
      qy += 1;
      stack_push(stack, &top, stack_cap, p.x + 1, qy);
      stack_push(stack, &top, stack_cap, p.x - 1, qy);
    }
  }
}

extern "C" __global__ void unpaper_blackfilter_floodfill_rect(
    uint8_t *img, int img_linesize, int img_fmt, int w, int h, int x0, int y0,
    int x1, int y1, uint8_t mask_max, unsigned long long intensity,
    UnpaperCudaPoint *stack, int stack_cap) {
  if (blockIdx.x != 0 || threadIdx.x != 0 || blockIdx.y != 0 || threadIdx.y != 0) {
    return;
  }

  const UnpaperCudaFormat fmt = (UnpaperCudaFormat)img_fmt;

  for (int y = y0; y <= y1; y++) {
    for (int x = x0; x <= x1; x++) {
      flood_fill_cuda(img, img_linesize, fmt, w, h, x, y, 0u, mask_max,
                      intensity, stack, stack_cap);
    }
  }
}

// Blurfilter GPU scan kernel
// Scans blocks in the dark mask integral image and identifies isolated blocks
// to wipe. Uses NPP integral format where I[y,x] = sum from (0,0) to (x-1,y-1).
//
// Each thread processes one potential block position. A block is considered
// isolated if all 4 diagonal neighbors have dark pixel ratio <= intensity.
// Output is a list of block coordinates to wipe.

// Helper: compute sum from NPP integral image for rectangle (x0,y0) to (x1,y1)
// NPP format: I[y,x] = sum of pixels from (0,0) to (x-1,y-1)
// Sum of rect (x0,y0)-(x1,y1) = I[y1+1,x1+1] - I[y0,x1+1] - I[y1+1,x0] + I[y0,x0]
static __device__ __forceinline__ int64_t npp_integral_rect_sum(
    const int32_t *integral, int integral_step_i32, int img_w, int img_h,
    int x0, int y0, int x1, int y1) {
  // Clamp coordinates
  if (x0 < 0) x0 = 0;
  if (y0 < 0) y0 = 0;
  if (x1 >= img_w) x1 = img_w - 1;
  if (y1 >= img_h) y1 = img_h - 1;
  if (x0 > x1 || y0 > y1) {
    return 0;
  }

  // Get corner values, handling boundary cases
  // Bottom-right: I[y1+1, x1+1]
  int64_t br = 0;
  if (y1 + 1 < img_h && x1 + 1 < img_w) {
    br = integral[(y1 + 1) * integral_step_i32 + (x1 + 1)];
  }

  // Top-right: I[y0, x1+1]
  int64_t tr = 0;
  if (x1 + 1 < img_w) {
    tr = integral[y0 * integral_step_i32 + (x1 + 1)];
  }

  // Bottom-left: I[y1+1, x0]
  int64_t bl = 0;
  if (y1 + 1 < img_h) {
    bl = integral[(y1 + 1) * integral_step_i32 + x0];
  }

  // Top-left: I[y0, x0]
  int64_t tl = integral[y0 * integral_step_i32 + x0];

  return br - tr - bl + tl;
}

// Output structure for block coordinates (x, y as pixel coordinates)
typedef struct {
  int x;
  int y;
} UnpaperBlurfilterBlock;

extern "C" __global__ void unpaper_blurfilter_scan(
    const int32_t *integral,     // GPU integral image (NPP format)
    int integral_step,           // Bytes per row of integral image
    int img_w, int img_h,        // Original image dimensions
    int block_w, int block_h,    // Block size
    float intensity,             // Isolation threshold (ratio)
    UnpaperBlurfilterBlock *out_blocks, // Output: block coordinates
    int *out_count,              // Output: number of blocks found (atomic)
    int max_blocks) {            // Maximum blocks to output
  // Calculate which block this thread processes
  const int bx = (int)(blockIdx.x * blockDim.x + threadIdx.x);
  const int by = (int)(blockIdx.y * blockDim.y + threadIdx.y);

  // Calculate number of blocks in each dimension
  const int blocks_per_row = img_w / block_w;
  const int blocks_per_col = img_h / block_h;

  if (bx >= blocks_per_row || by >= blocks_per_col) {
    return;
  }

  const int integral_step_i32 = integral_step / (int)sizeof(int32_t);
  const int64_t total_pixels = (int64_t)block_w * (int64_t)block_h;

  // Get dark pixel count for this block
  const int x0 = bx * block_w;
  const int y0 = by * block_h;
  const int x1 = x0 + block_w - 1;
  const int y1 = y0 + block_h - 1;

  // In the dark mask integral, each dark pixel contributes 255
  // So dark_count = sum / 255
  int64_t dark_sum = npp_integral_rect_sum(integral, integral_step_i32, img_w,
                                           img_h, x0, y0, x1, y1);
  int64_t dark_count = dark_sum / 255;

  // If this block has no dark pixels, skip (nothing to wipe)
  if (dark_count == 0) {
    return;
  }

  // Check all 4 diagonal neighbors for isolation
  // A block is isolated if ALL neighbors AND the block itself have ratio <= intensity
  // Missing boundary neighbors are treated as having max density (100%)
  // to prevent wiping edge blocks where we can't determine isolation
  // Start with current block's count (matches CPU blurfilter logic)
  int64_t max_neighbor = dark_count;

  // Upper-left diagonal (bx-1, by-1)
  if (bx > 0 && by > 0) {
    int nx0 = (bx - 1) * block_w;
    int ny0 = (by - 1) * block_h;
    int64_t n_sum = npp_integral_rect_sum(integral, integral_step_i32, img_w,
                                          img_h, nx0, ny0,
                                          nx0 + block_w - 1, ny0 + block_h - 1);
    int64_t n_count = n_sum / 255;
    if (n_count > max_neighbor) max_neighbor = n_count;
  } else {
    // Boundary: treat as max density
    max_neighbor = total_pixels;
  }

  // Upper-right diagonal (bx+1, by-1)
  if (bx < blocks_per_row - 1 && by > 0) {
    int nx0 = (bx + 1) * block_w;
    int ny0 = (by - 1) * block_h;
    int64_t n_sum = npp_integral_rect_sum(integral, integral_step_i32, img_w,
                                          img_h, nx0, ny0,
                                          nx0 + block_w - 1, ny0 + block_h - 1);
    int64_t n_count = n_sum / 255;
    if (n_count > max_neighbor) max_neighbor = n_count;
  } else {
    // Boundary: treat as max density
    max_neighbor = total_pixels;
  }

  // Lower-left diagonal (bx-1, by+1)
  if (bx > 0 && by < blocks_per_col - 1) {
    int nx0 = (bx - 1) * block_w;
    int ny0 = (by + 1) * block_h;
    int64_t n_sum = npp_integral_rect_sum(integral, integral_step_i32, img_w,
                                          img_h, nx0, ny0,
                                          nx0 + block_w - 1, ny0 + block_h - 1);
    int64_t n_count = n_sum / 255;
    if (n_count > max_neighbor) max_neighbor = n_count;
  } else {
    // Boundary: treat as max density
    max_neighbor = total_pixels;
  }

  // Lower-right diagonal (bx+1, by+1)
  if (bx < blocks_per_row - 1 && by < blocks_per_col - 1) {
    int nx0 = (bx + 1) * block_w;
    int ny0 = (by + 1) * block_h;
    int64_t n_sum = npp_integral_rect_sum(integral, integral_step_i32, img_w,
                                          img_h, nx0, ny0,
                                          nx0 + block_w - 1, ny0 + block_h - 1);
    int64_t n_count = n_sum / 255;
    if (n_count > max_neighbor) max_neighbor = n_count;
  } else {
    // Boundary: treat as max density
    max_neighbor = total_pixels;
  }

  // Check isolation criterion: max neighbor ratio <= intensity
  float ratio = (float)max_neighbor / (float)total_pixels;
  if (ratio <= intensity) {
    // This block should be wiped - add to output list atomically
    int idx = atomicAdd(out_count, 1);
    if (idx < max_blocks) {
      out_blocks[idx].x = x0;
      out_blocks[idx].y = y0;
    }
  }
}
