// SPDX-FileCopyrightText: 2025 The unpaper authors
//
// SPDX-License-Identifier: GPL-2.0-only

// Internal header for CUDA backend modules.
// This file should NOT be included by code outside imageprocess/.

#pragma once

#include <stdbool.h>
#include <stdint.h>

#include "imageprocess/backend.h"
#include "imageprocess/cuda_kernels_format.h"
#include "imageprocess/image.h"

// ImageCudaState tracks GPU memory and dirty flags for an image.
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

// Kernel handles - initialized by ensure_kernels_loaded()
extern void *k_wipe_rect_bytes;
extern void *k_wipe_rect_mono;
extern void *k_copy_rect_to_bytes;
extern void *k_copy_rect_to_mono;
extern void *k_mirror_bytes;
extern void *k_mirror_mono;
extern void *k_rotate90_bytes;
extern void *k_rotate90_mono;
extern void *k_stretch_bytes;
extern void *k_stretch_mono;
extern void *k_count_brightness_range;
extern void *k_sum_lightness_rect;
extern void *k_sum_grayscale_rect;
extern void *k_sum_darkness_inverse_rect;
extern void *k_apply_masks_bytes;
extern void *k_apply_masks_mono;
extern void *k_noisefilter_build_labels;
extern void *k_noisefilter_propagate;
extern void *k_noisefilter_count;
extern void *k_noisefilter_apply;
extern void *k_noisefilter_apply_mask;
extern void *k_blackfilter_floodfill_rect;
extern void *k_blackfilter_scan_parallel;
extern void *k_blackfilter_wipe_regions;
extern void *k_detect_edge_rotation_peaks;
extern void *k_rotate_bytes;
extern void *k_rotate_mono;
extern void *k_batch_scan_grayscale_sum;
extern void *k_batch_scan_brightness_count;
extern void *k_expand_1bit_to_8bit;

// Ensure CUDA kernels are loaded (thread-safe, lazy initialization)
void ensure_kernels_loaded(void);

// Format conversion utilities
UnpaperCudaFormat cuda_format_from_av(int fmt);
int bytes_per_pixel_from_av(int fmt);

// Pixel utility
static inline uint8_t pixel_grayscale(Pixel pixel) {
  return (uint8_t)(((uint32_t)pixel.r + (uint32_t)pixel.g + (uint32_t)pixel.b) /
                   3u);
}

// Image state access
ImageCudaState *image_cuda_state(Image image);

// Rectangle utilities
bool rect_empty(Rectangle area);

// Brightness/scan helper functions (used across modules)
unsigned long long cuda_rect_count_brightness_range(Image image,
                                                    Rectangle input_area,
                                                    uint8_t min_brightness,
                                                    uint8_t max_brightness);
unsigned long long cuda_rect_sum_lightness(Image image, Rectangle input_area);
unsigned long long cuda_rect_sum_darkness_inverse(Image image,
                                                  Rectangle input_area);
uint8_t cuda_inverse_lightness_rect(Image image, Rectangle input_area);
uint8_t cuda_darkness_rect(Image image, Rectangle input_area);

// Edge detection helper (used by masks and border detection)
uint32_t detect_edge_cuda(Image image, Point origin, Delta step,
                          int32_t scan_size, int32_t scan_depth,
                          float threshold);

// Backend vtable function declarations (implemented in separate files)
// blit operations
void wipe_rectangle_cuda(Image image, Rectangle input_area, Pixel color);
void copy_rectangle_cuda(Image source, Image target, Rectangle source_area,
                         Point target_coords);
void center_image_cuda(Image source, Image target, Point target_origin,
                       RectangleSize target_size);
void flip_rotate_90_cuda(Image *pImage, RotationDirection direction);
void mirror_cuda(Image image, Direction direction);
void shift_image_cuda(Image *pImage, Delta d);
void stretch_and_replace_cuda(Image *pImage, RectangleSize size,
                              Interpolation interpolate_type);
void resize_and_replace_cuda(Image *pImage, RectangleSize size,
                             Interpolation interpolate_type);

// mask/border operations
void apply_masks_cuda(Image image, const Rectangle masks[], size_t masks_count,
                      Pixel color);
void apply_wipes_cuda(Image image, Wipes wipes, Pixel color);
void apply_border_cuda(Image image, const Border border, Pixel color);
size_t detect_masks_cuda(Image image, MaskDetectionParameters params,
                         const Point points[], size_t points_count,
                         Rectangle masks[]);
void align_mask_cuda(Image image, const Rectangle inside_area,
                     const Rectangle outside, MaskAlignmentParameters params);
Border detect_border_cuda(Image image, BorderScanParameters params,
                          const Rectangle outside_mask);

// filter operations
void blackfilter_cuda(Image image, BlackfilterParameters params);
void blurfilter_cuda(Image image, BlurfilterParameters params,
                     uint8_t abs_white_threshold);
void noisefilter_cuda(Image image, uint64_t intensity, uint8_t min_white_level);
void grayfilter_cuda(Image image, GrayfilterParameters params);

// deskew operations
float detect_rotation_cuda(Image image, Rectangle mask,
                           const DeskewParameters params);
void deskew_cuda(Image source, Rectangle mask, float radians,
                 Interpolation interpolate_type);
