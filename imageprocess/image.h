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
  (Image) {                                                                    \
    .frame = NULL, .background = PIXEL_WHITE, .abs_black_threshold = 0,        \
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

void image_ensure_cuda_alloc(Image *image);

void image_mark_cpu_dirty(Image *image);
void image_mark_cuda_dirty(Image *image);

// Create Image from existing GPU memory (for nvImageCodec decoded images)
// If owns_memory is true, the GPU memory will be freed when the Image is freed.
// If owns_memory is false, the caller retains ownership (e.g., pool-managed).
// The pitch is the row stride in bytes.
Image create_image_from_gpu(void *gpu_ptr, size_t pitch, int width, int height,
                            int pixel_format, Pixel background,
                            uint8_t abs_black_threshold, bool owns_memory);

// Check if image data is currently GPU-resident (valid on GPU)
bool image_is_gpu_resident(Image *image);

// Set the GPU-resident status of an image
// When true, image_ensure_cuda() will skip upload (GPU already has valid data)
void image_set_gpu_resident(Image *image, bool resident);

// Get GPU device pointer for GPU-resident image (returns NULL if not
// GPU-resident)
void *image_get_gpu_ptr(Image *image);

// Get GPU pitch (row stride) for GPU-resident image (returns 0 if not
// GPU-resident)
size_t image_get_gpu_pitch(Image *image);
