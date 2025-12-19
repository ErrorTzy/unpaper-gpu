// SPDX-FileCopyrightText: 2025 The unpaper authors
//
// SPDX-License-Identifier: GPL-2.0-only

#include "lib/decoded_image_provider.h"

#include "lib/batch_decode_queue.h"
#include "lib/decode_queue.h"

#include <string.h>

static bool decode_queue_provider_get(void *ctx, int job_index, int input_index,
                                      DecodedImageHandle *out) {
  DecodeQueue *queue = (DecodeQueue *)ctx;
  DecodedImage *decoded = decode_queue_get(queue, job_index, input_index);
  if (decoded == NULL || !decoded->valid) {
    return false;
  }

#ifdef UNPAPER_WITH_CUDA
  if (decoded->on_gpu && decoded->gpu_ptr != NULL) {
    decoded_image_wait_gpu_complete(decoded);
  }
#endif

  out->provider_type = DECODED_IMAGE_PROVIDER_DECODE_QUEUE;
  out->image = decoded;
  out->view.valid = decoded->valid;
  out->view.frame = decoded->frame;
  out->view.on_gpu = decoded->on_gpu;
  out->view.gpu_ptr = decoded->gpu_ptr;
  out->view.gpu_pitch = decoded->gpu_pitch;
  out->view.gpu_width = decoded->gpu_width;
  out->view.gpu_height = decoded->gpu_height;
  out->view.gpu_format = decoded->gpu_format;
  out->view.gpu_owns_memory = decoded->on_gpu;

  return true;
}

static void decode_queue_provider_release(void *ctx,
                                          DecodedImageHandle *handle) {
  DecodeQueue *queue = (DecodeQueue *)ctx;
  DecodedImage *decoded = (DecodedImage *)handle->image;
  if (queue && decoded) {
    decode_queue_release(queue, decoded);
  }
}

static bool batch_decode_queue_provider_get(void *ctx, int job_index,
                                            int input_index,
                                            DecodedImageHandle *out) {
  BatchDecodeQueue *queue = (BatchDecodeQueue *)ctx;
  BatchDecodedImage *decoded =
      batch_decode_queue_get(queue, job_index, input_index);
  if (decoded == NULL || !decoded->valid) {
    return false;
  }

  out->provider_type = DECODED_IMAGE_PROVIDER_BATCH_DECODE_QUEUE;
  out->image = decoded;
  out->view.valid = decoded->valid;
  out->view.frame = decoded->frame;
  out->view.on_gpu = decoded->on_gpu;
  out->view.gpu_ptr = decoded->gpu_ptr;
  out->view.gpu_pitch = decoded->gpu_pitch;
  out->view.gpu_width = decoded->gpu_width;
  out->view.gpu_height = decoded->gpu_height;
  out->view.gpu_format = decoded->gpu_format;
  out->view.gpu_owns_memory = false;

  return true;
}

static void batch_decode_queue_provider_release(void *ctx,
                                                DecodedImageHandle *handle) {
  BatchDecodeQueue *queue = (BatchDecodeQueue *)ctx;
  BatchDecodedImage *decoded = (BatchDecodedImage *)handle->image;
  if (queue && decoded) {
    batch_decode_queue_release(queue, decoded);
  }
}

void decoded_image_provider_reset(DecodedImageProvider *provider) {
  if (!provider) {
    return;
  }
  memset(provider, 0, sizeof(*provider));
}

bool decoded_image_provider_init_decode_queue(DecodedImageProvider *provider,
                                              DecodeQueue *queue) {
  if (!provider) {
    return false;
  }
  decoded_image_provider_reset(provider);
  if (!queue) {
    return false;
  }
  provider->type = DECODED_IMAGE_PROVIDER_DECODE_QUEUE;
  provider->ctx = queue;
  provider->get = decode_queue_provider_get;
  provider->release = decode_queue_provider_release;
  return true;
}

bool decoded_image_provider_init_batch_decode_queue(
    DecodedImageProvider *provider, BatchDecodeQueue *queue) {
  if (!provider) {
    return false;
  }
  decoded_image_provider_reset(provider);
  if (!queue) {
    return false;
  }
  provider->type = DECODED_IMAGE_PROVIDER_BATCH_DECODE_QUEUE;
  provider->ctx = queue;
  provider->get = batch_decode_queue_provider_get;
  provider->release = batch_decode_queue_provider_release;
  return true;
}

bool decoded_image_provider_get(DecodedImageProvider *provider, int job_index,
                                int input_index, DecodedImageHandle *out) {
  if (!provider || !provider->get || !out) {
    return false;
  }
  memset(out, 0, sizeof(*out));
  return provider->get(provider->ctx, job_index, input_index, out);
}

void decoded_image_provider_release(DecodedImageProvider *provider,
                                    DecodedImageHandle *handle) {
  if (!provider || !provider->release || !handle || !handle->image) {
    return;
  }
  provider->release(provider->ctx, handle);
  memset(handle, 0, sizeof(*handle));
}

void decoded_image_handle_detach_gpu(DecodedImageHandle *handle) {
  if (!handle || !handle->image) {
    return;
  }

  if (handle->provider_type == DECODED_IMAGE_PROVIDER_DECODE_QUEUE) {
    DecodedImage *decoded = (DecodedImage *)handle->image;
    decoded->gpu_ptr = NULL;
    decoded->on_gpu = false;
    handle->view.gpu_ptr = NULL;
    handle->view.on_gpu = false;
  }
}
