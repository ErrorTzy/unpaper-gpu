// SPDX-FileCopyrightText: 2005 The unpaper authors
//
// SPDX-License-Identifier: GPL-2.0-only

#pragma once

#include "imageprocess/primitives.h"

typedef struct AVFrame AVFrame;

typedef struct {
  AVFrame *frame;
  Pixel background;
  uint8_t abs_black_threshold;
} Image;

#define EMPTY_IMAGE                                                            \
  (Image) {                                                                   \
    .frame = NULL, .background = PIXEL_WHITE, .abs_black_threshold = 0,       \
  }

Image create_image(RectangleSize size, int pixel_format, bool fill,
                   Pixel sheet_background, uint8_t abs_black_threshold);
void replace_image(Image *image, Image *new_image);
void free_image(Image *image);
Image create_compatible_image(Image source, RectangleSize size, bool fill);

RectangleSize size_of_image(Image image);
Rectangle full_image(Image image);
Rectangle clip_rectangle(Image image, Rectangle area);

void image_ensure_cuda(Image *image);
void image_ensure_cpu(Image *image);

void image_mark_cpu_dirty(Image *image);
void image_mark_cuda_dirty(Image *image);
