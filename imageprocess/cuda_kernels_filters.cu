// SPDX-FileCopyrightText: 2025 The unpaper authors
//
// SPDX-License-Identifier: GPL-2.0-only

// Filter kernels: noisefilter, blackfilter, blurfilter, grayfilter, format
// conversion

#include "imageprocess/cuda_kernels_common.cuh"

// ============================================================================
// Noisefilter kernels
// ============================================================================

static __device__ __forceinline__ bool
noisefilter_is_dark(const uint8_t *img, int img_linesize, UnpaperCudaFormat fmt,
                    int w, int h, int x, int y, uint8_t min_white_level) {
  const uint8_t darkness =
      get_darkness_inverse_safe(img, img_linesize, fmt, w, h, x, y);
  return darkness < min_white_level;
}

extern "C" __global__ void
unpaper_noisefilter_build_labels(const uint8_t *img, int img_linesize,
                                 int img_fmt, int w, int h,
                                 uint8_t min_white_level, uint32_t *labels) {
  const int x = (int)(blockIdx.x * blockDim.x + threadIdx.x);
  const int y = (int)(blockIdx.y * blockDim.y + threadIdx.y);
  if (x >= w || y >= h) {
    return;
  }

  const int idx = y * w + x;
  const UnpaperCudaFormat fmt = (UnpaperCudaFormat)img_fmt;
  labels[idx] =
      noisefilter_is_dark(img, img_linesize, fmt, w, h, x, y, min_white_level)
          ? (uint32_t)(idx + 1)
          : 0u;
}

extern "C" __global__ void
unpaper_noisefilter_propagate(const uint32_t *labels_in, uint32_t *labels_out,
                              int w, int h, int *changed) {
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

extern "C" __global__ void
unpaper_noisefilter_apply(uint8_t *img, int img_linesize, int img_fmt, int w,
                          int h, const uint32_t *labels, const uint32_t *counts,
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
extern "C" __global__ void
unpaper_noisefilter_apply_mask(uint8_t *img, int img_linesize, int img_fmt,
                               int w, int h, const uint8_t *mask,
                               int mask_linesize, uint8_t min_white_level) {
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

// ============================================================================
// Blackfilter flood-fill kernels
// ============================================================================

static __device__ __forceinline__ unsigned long long
flood_fill_line_cuda(uint8_t *img, int img_linesize, UnpaperCudaFormat fmt,
                     int w, int h, int px, int py, int step_x, int step_y,
                     uint8_t mask_min, uint8_t mask_max,
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

static __device__ __forceinline__ void
stack_push(UnpaperCudaPoint *stack, int *top, int cap, int x, int y) {
  const int t = *top;
  if (t >= cap) {
    return;
  }
  stack[t] = (UnpaperCudaPoint){.x = x, .y = y};
  *top = t + 1;
}

static __device__ __forceinline__ bool
stack_pop(UnpaperCudaPoint *stack, int *top, UnpaperCudaPoint *out) {
  const int t = *top;
  if (t <= 0) {
    return false;
  }
  *top = t - 1;
  *out = stack[t - 1];
  return true;
}

static __device__ __forceinline__ void
flood_fill_cuda(uint8_t *img, int img_linesize, UnpaperCudaFormat fmt, int w,
                int h, int sx, int sy, uint8_t mask_min, uint8_t mask_max,
                unsigned long long intensity, UnpaperCudaPoint *stack,
                int stack_cap) {
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

    const unsigned long long left =
        flood_fill_line_cuda(img, img_linesize, fmt, w, h, p.x, p.y, -1, 0,
                             mask_min, mask_max, intensity);
    const unsigned long long up =
        flood_fill_line_cuda(img, img_linesize, fmt, w, h, p.x, p.y, 0, -1,
                             mask_min, mask_max, intensity);
    const unsigned long long right =
        flood_fill_line_cuda(img, img_linesize, fmt, w, h, p.x, p.y, 1, 0,
                             mask_min, mask_max, intensity);
    const unsigned long long down =
        flood_fill_line_cuda(img, img_linesize, fmt, w, h, p.x, p.y, 0, 1,
                             mask_min, mask_max, intensity);

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
  if (blockIdx.x != 0 || threadIdx.x != 0 || blockIdx.y != 0 ||
      threadIdx.y != 0) {
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

// GPU-parallel blackfilter: wipe dark pixels in multiple rectangular regions
// This replaces the sequential flood-fill approach with fully parallel
// processing. Each thread handles one pixel, checks if it's in any wipe region
// and dark, and sets it to white if so.
//
// Parameters:
//   rects: Array of rectangles [x0,y0,x1,y1, x0,y0,x1,y1, ...] (expanded by
//   intensity) rect_count: Number of rectangles black_threshold: Pixels with
//   grayscale <= this are considered "dark"
extern "C" __global__ void
unpaper_blackfilter_wipe_regions(uint8_t *img, int img_linesize, int img_fmt,
                                 int w, int h, const int32_t *rects,
                                 int rect_count, uint8_t black_threshold) {
  const int x = (int)(blockIdx.x * blockDim.x + threadIdx.x);
  const int y = (int)(blockIdx.y * blockDim.y + threadIdx.y);
  if (x >= w || y >= h) {
    return;
  }

  // Check if pixel is in any wipe region
  bool in_region = false;
  for (int i = 0; i < rect_count; i++) {
    const int rx0 = rects[i * 4 + 0];
    const int ry0 = rects[i * 4 + 1];
    const int rx1 = rects[i * 4 + 2];
    const int ry1 = rects[i * 4 + 3];
    if (x >= rx0 && x <= rx1 && y >= ry0 && y <= ry1) {
      in_region = true;
      break;
    }
  }

  if (!in_region) {
    return;
  }

  // Check if pixel is dark enough to wipe
  const UnpaperCudaFormat fmt = (UnpaperCudaFormat)img_fmt;
  const uint8_t gray = get_grayscale_safe(img, img_linesize, fmt, w, h, x, y);
  if (gray <= black_threshold) {
    set_pixel_white_safe(img, img_linesize, fmt, w, h, x, y);
  }
}

// GPU-parallel blackfilter scan: find dark blocks using integral image
// Each thread handles one scan position, checks darkness, outputs block coords.
//
// Parameters:
//   integral: GPU integral image in NPP format (padded to (w+1)x(h+1))
//   integral_step: Bytes per row
//   img_w, img_h: Original image dimensions
//   scan_w, scan_h: Scan block size
//   step_x, step_y: Step between scan positions (one must be 0)
//   threshold: Darkness threshold (0-255, blocks with darkness >= this are
//   dark) intensity: Expansion in pixels (added to block bounds before wiping)
//   out_rects: Output array of rectangles [x0,y0,x1,y1, ...] expanded by
//   intensity out_count: Atomic counter for number of rectangles found
//   max_rects: Maximum rectangles to output
extern "C" __global__ void unpaper_blackfilter_scan_parallel(
    const int32_t *integral, int integral_step, int img_w, int img_h,
    int scan_w, int scan_h, int step_x, int step_y, int stripe_offset,
    int stripe_size, uint8_t threshold, int intensity, int32_t *out_rects,
    int *out_count, int max_rects) {
  // Calculate which scan position this thread handles
  const int tid = (int)(blockIdx.x * blockDim.x + threadIdx.x);

  // Compute scan positions based on step direction
  int pos_x, pos_y;
  if (step_x != 0) {
    // Horizontal scan: stripe is vertical, positions are horizontal
    const int num_positions = (img_w - scan_w) / step_x + 1;
    if (tid >= num_positions)
      return;
    pos_x = tid * step_x;
    pos_y = stripe_offset;
    // Clamp to not exceed stripe bounds
    if (pos_y + scan_h > stripe_offset + stripe_size) {
      return;
    }
  } else {
    // Vertical scan: stripe is horizontal, positions are vertical
    const int num_positions = (img_h - scan_h) / step_y + 1;
    if (tid >= num_positions)
      return;
    pos_x = stripe_offset;
    pos_y = tid * step_y;
    if (pos_x + scan_w > stripe_offset + stripe_size) {
      return;
    }
  }

  // Bounds check
  if (pos_x + scan_w > img_w || pos_y + scan_h > img_h) {
    return;
  }

  // Compute darkness using integral image
  const int integral_step_i32 = integral_step / (int)sizeof(int32_t);
  const int64_t pixel_sum =
      npp_integral_rect_sum(integral, integral_step_i32, img_w, img_h, pos_x,
                            pos_y, pos_x + scan_w - 1, pos_y + scan_h - 1);

  const int64_t pixel_count = (int64_t)scan_w * (int64_t)scan_h;
  const int64_t avg_brightness = pixel_sum / pixel_count;
  const uint8_t darkness = (uint8_t)(255 - (int)avg_brightness);

  // Check if dark enough
  if (darkness < threshold) {
    return;
  }

  // Add rectangle to output (expanded by intensity)
  const int idx = atomicAdd(out_count, 1);
  if (idx >= max_rects) {
    return;
  }

  // Expand rectangle by intensity, clamp to image bounds
  int x0 = pos_x - intensity;
  int y0 = pos_y - intensity;
  int x1 = pos_x + scan_w - 1 + intensity;
  int y1 = pos_y + scan_h - 1 + intensity;
  if (x0 < 0)
    x0 = 0;
  if (y0 < 0)
    y0 = 0;
  if (x1 >= img_w)
    x1 = img_w - 1;
  if (y1 >= img_h)
    y1 = img_h - 1;

  out_rects[idx * 4 + 0] = x0;
  out_rects[idx * 4 + 1] = y0;
  out_rects[idx * 4 + 2] = x1;
  out_rects[idx * 4 + 3] = y1;
}

// ============================================================================
// Blurfilter scan kernel
// ============================================================================
// Scans blocks in the dark mask integral image and identifies isolated blocks
// to wipe. Uses NPP integral format where I[y,x] = sum from (0,0) to (x-1,y-1).
//
// Each thread processes one potential block position. A block is considered
// isolated if all 4 diagonal neighbors have dark pixel ratio <= intensity.
// Output is a list of block coordinates to wipe.

extern "C" __global__ void unpaper_blurfilter_scan(
    const int32_t *integral,            // GPU integral image (NPP format)
    int integral_step,                  // Bytes per row of integral image
    int img_w, int img_h,               // Original image dimensions
    int block_w, int block_h,           // Block size
    float intensity,                    // Isolation threshold (ratio)
    UnpaperBlurfilterBlock *out_blocks, // Output: block coordinates
    int *out_count,   // Output: number of blocks found (atomic)
    int max_blocks) { // Maximum blocks to output
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
  // A block is isolated if ALL neighbors AND the block itself have ratio <=
  // intensity Missing boundary neighbors are treated as having max density
  // (100%) to prevent wiping edge blocks where we can't determine isolation
  // Start with current block's count (matches CPU blurfilter logic)
  int64_t max_neighbor = dark_count;

  // Upper-left diagonal (bx-1, by-1)
  if (bx > 0 && by > 0) {
    int nx0 = (bx - 1) * block_w;
    int ny0 = (by - 1) * block_h;
    int64_t n_sum =
        npp_integral_rect_sum(integral, integral_step_i32, img_w, img_h, nx0,
                              ny0, nx0 + block_w - 1, ny0 + block_h - 1);
    int64_t n_count = n_sum / 255;
    if (n_count > max_neighbor)
      max_neighbor = n_count;
  } else {
    // Boundary: treat as max density
    max_neighbor = total_pixels;
  }

  // Upper-right diagonal (bx+1, by-1)
  if (bx < blocks_per_row - 1 && by > 0) {
    int nx0 = (bx + 1) * block_w;
    int ny0 = (by - 1) * block_h;
    int64_t n_sum =
        npp_integral_rect_sum(integral, integral_step_i32, img_w, img_h, nx0,
                              ny0, nx0 + block_w - 1, ny0 + block_h - 1);
    int64_t n_count = n_sum / 255;
    if (n_count > max_neighbor)
      max_neighbor = n_count;
  } else {
    // Boundary: treat as max density
    max_neighbor = total_pixels;
  }

  // Lower-left diagonal (bx-1, by+1)
  if (bx > 0 && by < blocks_per_col - 1) {
    int nx0 = (bx - 1) * block_w;
    int ny0 = (by + 1) * block_h;
    int64_t n_sum =
        npp_integral_rect_sum(integral, integral_step_i32, img_w, img_h, nx0,
                              ny0, nx0 + block_w - 1, ny0 + block_h - 1);
    int64_t n_count = n_sum / 255;
    if (n_count > max_neighbor)
      max_neighbor = n_count;
  } else {
    // Boundary: treat as max density
    max_neighbor = total_pixels;
  }

  // Lower-right diagonal (bx+1, by+1)
  if (bx < blocks_per_row - 1 && by < blocks_per_col - 1) {
    int nx0 = (bx + 1) * block_w;
    int ny0 = (by + 1) * block_h;
    int64_t n_sum =
        npp_integral_rect_sum(integral, integral_step_i32, img_w, img_h, nx0,
                              ny0, nx0 + block_w - 1, ny0 + block_h - 1);
    int64_t n_count = n_sum / 255;
    if (n_count > max_neighbor)
      max_neighbor = n_count;
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

// ============================================================================
// Grayfilter scan kernel
// ============================================================================
/**
 * GPU kernel for grayfilter tile detection.
 *
 * Grayfilter identifies tiles that have:
 * 1. No dark pixels (dark_count == 0)
 * 2. High average lightness (inverse_lightness < gray_threshold)
 *
 * These tiles are wiped to white to remove light gray artifacts.
 *
 * Unlike blurfilter, grayfilter:
 * - Uses step-based positioning (not block-aligned grid)
 * - Uses two integrals (gray sum + dark pixel count)
 * - Has no neighbor isolation check
 *
 * @param gray_integral   GPU integral of grayscale image (NPP format)
 * @param dark_integral   GPU integral of dark mask (pixels <= black_threshold
 * -> 255)
 * @param gray_step       Bytes per row of gray integral image
 * @param dark_step       Bytes per row of dark integral image
 * @param img_w, img_h    Original image dimensions
 * @param tile_w, tile_h  Tile size
 * @param step_x, step_y  Step size for tile positions
 * @param gray_threshold  Inverse lightness threshold (255 - avg_lightness must
 * be < this)
 * @param out_tiles       Output: tile coordinates to wipe
 * @param out_count       Output: number of tiles found (atomic counter)
 * @param max_tiles       Maximum tiles to output
 */
extern "C" __global__ void unpaper_grayfilter_scan(
    const int32_t *gray_integral, // GPU integral of grayscale image
    const int32_t *dark_integral, // GPU integral of dark mask
    int gray_step,                // Bytes per row of gray integral
    int dark_step,                // Bytes per row of dark integral
    int img_w, int img_h,         // Original image dimensions
    int tile_w, int tile_h,       // Tile size
    int step_x, int step_y,       // Step size
    int gray_threshold, // Inverse lightness threshold (use int for alignment)
    UnpaperGrayfilterTile *out_tiles, // Output: tile coordinates
    int *out_count,                   // Output: number of tiles found (atomic)
    int max_tiles) {                  // Maximum tiles to output
  // Calculate number of tile positions in each dimension
  // Tiles are placed at positions: 0, step_x, 2*step_x, ... while x + tile_w <=
  // img_w
  const int tiles_per_row = (img_w - tile_w) / step_x + 1;
  const int tiles_per_col = (img_h - tile_h) / step_y + 1;

  // Calculate which tile this thread processes
  const int tx = (int)(blockIdx.x * blockDim.x + threadIdx.x);
  const int ty = (int)(blockIdx.y * blockDim.y + threadIdx.y);

  if (tx >= tiles_per_row || ty >= tiles_per_col) {
    return;
  }

  // Calculate tile position in pixels
  const int x0 = tx * step_x;
  const int y0 = ty * step_y;
  const int x1 = x0 + tile_w - 1;
  const int y1 = y0 + tile_h - 1;

  // Verify tile is within image bounds
  if (x1 >= img_w || y1 >= img_h) {
    return;
  }

  const int gray_step_i32 = gray_step / (int)sizeof(int32_t);
  const int dark_step_i32 = dark_step / (int)sizeof(int32_t);
  const int64_t tile_pixels = (int64_t)tile_w * (int64_t)tile_h;

  // Count dark pixels using dark integral
  // In the dark mask, each dark pixel contributes 255 to the sum
  int64_t dark_sum = npp_integral_rect_sum(dark_integral, dark_step_i32, img_w,
                                           img_h, x0, y0, x1, y1);
  int64_t dark_count = dark_sum / 255;

  // If there are any dark pixels, this tile doesn't match grayfilter criteria
  if (dark_count != 0) {
    return;
  }

  // No dark pixels - check average lightness
  int64_t lightness_sum = npp_integral_rect_sum(gray_integral, gray_step_i32,
                                                img_w, img_h, x0, y0, x1, y1);
  uint8_t avg_lightness = (uint8_t)(lightness_sum / tile_pixels);
  uint8_t inverse_lightness = 255 - avg_lightness;

  // Wipe if inverse_lightness is below threshold (tile is very light)
  if (inverse_lightness < gray_threshold) {
    // Add tile to output list atomically
    int idx = atomicAdd(out_count, 1);
    if (idx < max_tiles) {
      out_tiles[idx].x = x0;
      out_tiles[idx].y = y0;
    }
  }
}

// ============================================================================
// 1-bit to 8-bit expansion kernel for JBIG2 images
// ============================================================================
// Expands packed 1-bit data to 8-bit grayscale (0 or 255 per pixel)
// Each thread processes 8 output pixels (1 input byte)
// This is the hot path for JBIG2 PDF processing on GPU
//
// Parameters:
//   src: packed 1-bit data (MSB first, as from JBIG2 decoder)
//   src_stride: bytes per row in source (includes padding)
//   dst: 8-bit grayscale output
//   dst_stride: bytes per row in destination
//   width: image width in pixels
//   height: image height in pixels
//   invert: if true, 1-bit=white(255), 0-bit=black(0)
//           if false, 1-bit=black(0), 0-bit=white(255)
extern "C" __global__ void
unpaper_expand_1bit_to_8bit(const uint8_t *src, int src_stride, uint8_t *dst,
                            int dst_stride, int width, int height, int invert) {
  // Each thread handles one byte of input (8 pixels of output)
  const int byte_x = (int)(blockIdx.x * blockDim.x + threadIdx.x);
  const int y = (int)(blockIdx.y * blockDim.y + threadIdx.y);

  if (y >= height)
    return;

  const int src_bytes_per_row = (width + 7) / 8;
  if (byte_x >= src_bytes_per_row)
    return;

  // Read one byte of packed 1-bit data
  const uint8_t packed = src[(size_t)y * (size_t)src_stride + (size_t)byte_x];

  // Output pixel values based on invert flag.
  // JBIG2 typically: 1=black, 0=white, so invert=false gives black=0, white=255.
  const uint8_t val_bit_set = invert ? 255u : 0u;
  const uint8_t val_bit_clr = invert ? 0u : 255u;

  // Expand 8 bits to 8 bytes
  const int pixel_x = byte_x * 8;
  uint8_t *dst_row = dst + (size_t)y * (size_t)dst_stride + (size_t)pixel_x;

// Unrolled loop for 8 pixels - MSB first
#pragma unroll
  for (int bit = 0; bit < 8; bit++) {
    const int px = pixel_x + bit;
    if (px < width) {
      const bool bit_set = (packed & (0x80u >> bit)) != 0;
      dst_row[bit] = bit_set ? val_bit_set : val_bit_clr;
    }
  }
}
