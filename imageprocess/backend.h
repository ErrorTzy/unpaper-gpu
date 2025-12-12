// SPDX-FileCopyrightText: 2025 The unpaper authors
//
// SPDX-License-Identifier: GPL-2.0-only

#pragma once

#include <stddef.h>
#include <stdint.h>

#include "imageprocess/blit.h"
#include "imageprocess/deskew.h"
#include "imageprocess/filters.h"
#include "imageprocess/image.h"
#include "imageprocess/interpolate.h"
#include "imageprocess/masks.h"
#include "imageprocess/primitives.h"
#include "lib/options.h"

typedef struct {
  const char *name;

  void (*wipe_rectangle)(Image image, Rectangle input_area, Pixel color);
  void (*copy_rectangle)(Image source, Image target, Rectangle source_area,
                         Point target_coords);
  void (*center_image)(Image source, Image target, Point target_origin,
                       RectangleSize target_size);
  void (*stretch_and_replace)(Image *pImage, RectangleSize size,
                              Interpolation interpolate_type);
  void (*resize_and_replace)(Image *pImage, RectangleSize size,
                             Interpolation interpolate_type);
  void (*flip_rotate_90)(Image *pImage, RotationDirection direction);
  void (*mirror)(Image image, Direction direction);
  void (*shift_image)(Image *pImage, Delta d);

  void (*apply_masks)(Image image, const Rectangle masks[], size_t masks_count,
                      Pixel color);
  void (*apply_wipes)(Image image, Wipes wipes, Pixel color);
  void (*apply_border)(Image image, const Border border, Pixel color);
  size_t (*detect_masks)(Image image, MaskDetectionParameters params,
                         const Point points[], size_t points_count,
                         Rectangle masks[]);
  void (*align_mask)(Image image, const Rectangle inside_area,
                     const Rectangle outside, MaskAlignmentParameters params);
  Border (*detect_border)(Image image, BorderScanParameters params,
                          const Rectangle outside_mask);

  void (*blackfilter)(Image image, BlackfilterParameters params);
  void (*blurfilter)(Image image, BlurfilterParameters params,
                     uint8_t abs_white_threshold);
  void (*noisefilter)(Image image, uint64_t intensity, uint8_t min_white_level);
  void (*grayfilter)(Image image, GrayfilterParameters params);

  float (*detect_rotation)(Image image, Rectangle mask,
                           const DeskewParameters params);
  void (*deskew)(Image source, Rectangle mask, float radians,
                 Interpolation interpolate_type);
} ImageBackend;

const ImageBackend *image_backend_get(void);
void image_backend_select(UnpaperDevice device);

