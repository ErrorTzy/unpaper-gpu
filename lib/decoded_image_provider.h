// SPDX-FileCopyrightText: 2025 The unpaper authors
//
// SPDX-License-Identifier: GPL-2.0-only

#pragma once

#include <stdbool.h>
#include <stddef.h>

struct AVFrame;
struct DecodeQueue;
struct BatchDecodeQueue;

typedef enum {
  DECODED_IMAGE_PROVIDER_NONE = 0,
  DECODED_IMAGE_PROVIDER_DECODE_QUEUE,
  DECODED_IMAGE_PROVIDER_BATCH_DECODE_QUEUE,
} DecodedImageProviderType;

typedef struct {
  bool valid;
  struct AVFrame *frame;
  bool on_gpu;
  void *gpu_ptr;
  size_t gpu_pitch;
  int gpu_width;
  int gpu_height;
  int gpu_format;
  bool gpu_owns_memory;
} DecodedImageView;

typedef struct {
  DecodedImageProviderType provider_type;
  void *image;
  DecodedImageView view;
} DecodedImageHandle;

typedef struct {
  DecodedImageProviderType type;
  void *ctx;
  bool (*get)(void *ctx, int job_index, int input_index,
              DecodedImageHandle *out);
  void (*release)(void *ctx, DecodedImageHandle *handle);
} DecodedImageProvider;

void decoded_image_provider_reset(DecodedImageProvider *provider);

bool decoded_image_provider_init_decode_queue(DecodedImageProvider *provider,
                                              struct DecodeQueue *queue);

bool decoded_image_provider_init_batch_decode_queue(
    DecodedImageProvider *provider, struct BatchDecodeQueue *queue);

bool decoded_image_provider_get(DecodedImageProvider *provider, int job_index,
                                int input_index, DecodedImageHandle *out);

void decoded_image_provider_release(DecodedImageProvider *provider,
                                    DecodedImageHandle *handle);

void decoded_image_handle_detach_gpu(DecodedImageHandle *handle);
