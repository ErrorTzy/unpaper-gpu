// SPDX-FileCopyrightText: 2025 The unpaper authors
//
// SPDX-License-Identifier: GPL-2.0-only

#include <stdint.h>

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
