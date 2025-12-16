// SPDX-FileCopyrightText: 2025 The unpaper authors
//
// SPDX-License-Identifier: GPL-2.0-only

#include "imageprocess/image.h"

#include <libavutil/buffer.h>
#include <libavutil/frame.h>
#include <libavutil/pixfmt.h>

#include "lib/logging.h"

#if defined(UNPAPER_WITH_CUDA) && (UNPAPER_WITH_CUDA)
#include "imageprocess/cuda_mempool.h"
#include "imageprocess/cuda_runtime.h"
#endif

typedef struct {
  uint64_t dptr;
  size_t bytes;
  int format;
  int width;
  int height;
  int linesize;
  bool cpu_dirty;
  bool cuda_dirty;
  bool from_external; // True if GPU memory was allocated externally (e.g.,
                      // nvJPEG)
  bool owns_memory;   // True if this Image owns the GPU memory and should free
                      // it
} ImageCudaState;

#if defined(UNPAPER_WITH_CUDA) && (UNPAPER_WITH_CUDA)
#include <cuda_runtime_api.h> // for cudaFree

static void image_cuda_state_free(void *opaque, uint8_t *data) {
  (void)opaque;
  if (data == NULL) {
    return;
  }
  ImageCudaState *st = (ImageCudaState *)data;
  if (st->dptr != 0) {
    // Only free GPU memory if we own it (not borrowed from pool, etc.)
    if (st->owns_memory) {
      if (st->from_external) {
        // External memory (e.g., nvJPEG per-image) - use direct cudaFree
        cudaFree((void *)(uintptr_t)st->dptr);
      } else {
        // Pool-allocated memory - use pool release (falls back to cudaFree if
        // no pool active)
        cuda_mempool_global_release(st->dptr);
      }
    }
    // Clear pointer regardless of ownership
    st->dptr = 0;
  }
  av_free(st);
}

static ImageCudaState *image_cuda_state_get(Image *image, bool create) {
  if (image == NULL || image->frame == NULL) {
    return NULL;
  }

  if (image->frame->opaque_ref != NULL) {
    return (ImageCudaState *)image->frame->opaque_ref->data;
  }
  if (!create) {
    return NULL;
  }

  ImageCudaState *st = av_mallocz(sizeof(*st));
  if (st == NULL) {
    errOutput("unable to allocate CUDA image state.");
  }

  image->frame->opaque_ref = av_buffer_create((uint8_t *)st, sizeof(*st),
                                              image_cuda_state_free, NULL, 0);
  if (image->frame->opaque_ref == NULL) {
    av_free(st);
    errOutput("unable to allocate CUDA state buffer.");
  }

  st->dptr = 0;
  st->bytes = 0;
  st->format = -1;
  st->width = 0;
  st->height = 0;
  st->linesize = 0;
  st->cpu_dirty = true;
  st->cuda_dirty = false;
  st->from_external = false;
  st->owns_memory = true; // Default: Image owns its GPU memory
  return st;
}
#endif

void image_cuda_release(Image *image) {
#if defined(UNPAPER_WITH_CUDA) && (UNPAPER_WITH_CUDA)
  if (image == NULL || image->frame == NULL) {
    return;
  }
  av_buffer_unref(&image->frame->opaque_ref);
#else
  (void)image;
#endif
}

void image_mark_cpu_dirty(Image *image) {
#if defined(UNPAPER_WITH_CUDA) && (UNPAPER_WITH_CUDA)
  ImageCudaState *st = image_cuda_state_get(image, false);
  if (st == NULL) {
    return;
  }
  st->cpu_dirty = true;
  st->cuda_dirty = false;
#else
  (void)image;
#endif
}

void image_mark_cuda_dirty(Image *image) {
#if defined(UNPAPER_WITH_CUDA) && (UNPAPER_WITH_CUDA)
  ImageCudaState *st = image_cuda_state_get(image, false);
  if (st == NULL) {
    return;
  }
  st->cuda_dirty = true;
  st->cpu_dirty = false;
#else
  (void)image;
#endif
}

void image_ensure_cuda(Image *image) {
#if defined(UNPAPER_WITH_CUDA) && (UNPAPER_WITH_CUDA)
  ImageCudaState *st = image_cuda_state_get(image, true);
  if (st == NULL || image == NULL || image->frame == NULL) {
    return;
  }

  if (image->frame->width <= 0 || image->frame->height <= 0) {
    errOutput("invalid image size for CUDA upload.");
  }

  switch (image->frame->format) {
  case AV_PIX_FMT_Y400A:
  case AV_PIX_FMT_GRAY8:
  case AV_PIX_FMT_RGB24:
  case AV_PIX_FMT_MONOBLACK:
  case AV_PIX_FMT_MONOWHITE:
    break;
  default:
    errOutput("CUDA upload requested, but pixel format is unsupported.");
  }

  UnpaperCudaInitStatus init_status = unpaper_cuda_try_init();
  if (init_status != UNPAPER_CUDA_INIT_OK) {
    errOutput("%s", unpaper_cuda_init_status_string(init_status));
  }

  const size_t bytes =
      (size_t)image->frame->linesize[0] * (size_t)image->frame->height;
  if (bytes == 0) {
    errOutput("invalid image buffer size for CUDA upload.");
  }

  const bool need_alloc = (st->dptr == 0) || (st->bytes != bytes) ||
                          (st->format != image->frame->format) ||
                          (st->width != image->frame->width) ||
                          (st->height != image->frame->height) ||
                          (st->linesize != image->frame->linesize[0]);
  if (need_alloc) {
    if (st->dptr != 0) {
      // Use pool release (falls back to cudaFree if no pool active)
      cuda_mempool_global_release(st->dptr);
      st->dptr = 0;
    }
    // Use pool acquire (falls back to cudaMalloc if no pool active)
    st->dptr = cuda_mempool_global_acquire(bytes);
    st->bytes = bytes;
    st->format = image->frame->format;
    st->width = image->frame->width;
    st->height = image->frame->height;
    st->linesize = image->frame->linesize[0];
    st->cpu_dirty = true;
    st->cuda_dirty = false;
  }

  if (st->cpu_dirty) {
    // Use async copy on current stream to avoid blocking other workers
    // This is critical for stream parallelism - synchronous cudaMemcpy would
    // serialize all streams because it operates on the default stream!
    UnpaperCudaStream *stream = unpaper_cuda_get_current_stream();
    unpaper_cuda_memcpy_h2d_async(stream, st->dptr, image->frame->data[0],
                                  bytes);
    // Sync only this stream to ensure copy is complete before processing
    unpaper_cuda_stream_synchronize_on(stream);
    st->cpu_dirty = false;
    st->cuda_dirty = false;
  }
#else
  (void)image;
  errOutput("CUDA upload requested, but this build has no CUDA support.");
#endif
}

void image_ensure_cuda_alloc(Image *image) {
#if defined(UNPAPER_WITH_CUDA) && (UNPAPER_WITH_CUDA)
  ImageCudaState *st = image_cuda_state_get(image, true);
  if (st == NULL || image == NULL || image->frame == NULL) {
    return;
  }

  if (image->frame->width <= 0 || image->frame->height <= 0) {
    errOutput("invalid image size for CUDA allocation.");
  }

  switch (image->frame->format) {
  case AV_PIX_FMT_Y400A:
  case AV_PIX_FMT_GRAY8:
  case AV_PIX_FMT_RGB24:
  case AV_PIX_FMT_MONOBLACK:
  case AV_PIX_FMT_MONOWHITE:
    break;
  default:
    errOutput("CUDA allocation requested, but pixel format is unsupported.");
  }

  UnpaperCudaInitStatus init_status = unpaper_cuda_try_init();
  if (init_status != UNPAPER_CUDA_INIT_OK) {
    errOutput("%s", unpaper_cuda_init_status_string(init_status));
  }

  const size_t bytes =
      (size_t)image->frame->linesize[0] * (size_t)image->frame->height;
  if (bytes == 0) {
    errOutput("invalid image buffer size for CUDA allocation.");
  }

  const bool need_alloc = (st->dptr == 0) || (st->bytes != bytes) ||
                          (st->format != image->frame->format) ||
                          (st->width != image->frame->width) ||
                          (st->height != image->frame->height) ||
                          (st->linesize != image->frame->linesize[0]);
  if (need_alloc) {
    if (st->dptr != 0) {
      // Use pool release (falls back to cudaFree if no pool active)
      cuda_mempool_global_release(st->dptr);
      st->dptr = 0;
    }
    // Use pool acquire (falls back to cudaMalloc if no pool active)
    st->dptr = cuda_mempool_global_acquire(bytes);
    st->bytes = bytes;
    st->format = image->frame->format;
    st->width = image->frame->width;
    st->height = image->frame->height;
    st->linesize = image->frame->linesize[0];
  }

  st->cpu_dirty = false;
  st->cuda_dirty = false;
#else
  (void)image;
  errOutput("CUDA allocation requested, but this build has no CUDA support.");
#endif
}

void image_ensure_cpu(Image *image) {
#if defined(UNPAPER_WITH_CUDA) && (UNPAPER_WITH_CUDA)
  ImageCudaState *st = image_cuda_state_get(image, false);
  if (image == NULL || image->frame == NULL || st == NULL) {
    return;
  }
  if (st->dptr == 0) {
    return;
  }

  if (st->cuda_dirty) {
    const size_t bytes =
        (size_t)image->frame->linesize[0] * (size_t)image->frame->height;
    if (bytes == 0 || bytes != st->bytes) {
      errOutput("invalid image buffer size for CUDA download.");
    }

    UnpaperCudaInitStatus init_status = unpaper_cuda_try_init();
    if (init_status != UNPAPER_CUDA_INIT_OK) {
      errOutput("%s", unpaper_cuda_init_status_string(init_status));
    }

    // Use async copy on current stream to avoid blocking other workers
    // This is critical for stream parallelism - blocking cudaMemcpy would
    // serialize all streams!
    UnpaperCudaStream *stream = unpaper_cuda_get_current_stream();
    unpaper_cuda_memcpy_d2h_async(stream, image->frame->data[0], st->dptr,
                                  bytes);
    // Sync only this stream to ensure copy is complete
    unpaper_cuda_stream_synchronize_on(stream);
    st->cuda_dirty = false;
    st->cpu_dirty = false;
  }
#else
  (void)image;
#endif
}

Image create_image_from_gpu(void *gpu_ptr, size_t pitch, int width, int height,
                            int pixel_format, Pixel background,
                            uint8_t abs_black_threshold, bool owns_memory) {
  Image result = EMPTY_IMAGE;

#if defined(UNPAPER_WITH_CUDA) && (UNPAPER_WITH_CUDA)
  if (gpu_ptr == NULL || width <= 0 || height <= 0) {
    return result;
  }

  // Validate pixel format
  switch (pixel_format) {
  case AV_PIX_FMT_GRAY8:
  case AV_PIX_FMT_RGB24:
    break;
  default:
    errOutput("create_image_from_gpu: unsupported pixel format %d",
              pixel_format);
    return result;
  }

  // Create AVFrame for metadata and CPU buffer
  AVFrame *frame = av_frame_alloc();
  if (frame == NULL) {
    return result;
  }

  frame->width = width;
  frame->height = height;
  frame->format = pixel_format;

  // Allocate CPU buffer to match the GPU buffer layout
  // CRITICAL: The CPU buffer linesize MUST match the GPU pitch for correct
  // data transfer in image_ensure_cpu. av_frame_get_buffer uses its own
  // alignment which may not match nvJPEG's pitch.

  // Manually allocate the data buffer with the exact size we need
  size_t buffer_size = pitch * (size_t)height;
  uint8_t *buffer = av_malloc(buffer_size);
  if (buffer == NULL) {
    av_frame_free(&frame);
    return result;
  }

  // Set up the frame data to use our buffer
  frame->data[0] = buffer;
  frame->linesize[0] = (int)pitch;

  // Create a buffer reference so AVFrame will free the memory
  frame->buf[0] = av_buffer_create(buffer, buffer_size, NULL, NULL, 0);
  if (frame->buf[0] == NULL) {
    av_free(buffer);
    av_frame_free(&frame);
    return result;
  }

  // Create CUDA state and attach GPU pointer
  ImageCudaState *st = av_mallocz(sizeof(*st));
  if (st == NULL) {
    av_frame_free(&frame);
    errOutput("create_image_from_gpu: unable to allocate CUDA state");
    return result;
  }

  frame->opaque_ref = av_buffer_create((uint8_t *)st, sizeof(*st),
                                       image_cuda_state_free, NULL, 0);
  if (frame->opaque_ref == NULL) {
    av_free(st);
    av_frame_free(&frame);
    errOutput("create_image_from_gpu: unable to create CUDA state buffer");
    return result;
  }

  // Set GPU state
  st->dptr = (uint64_t)(uintptr_t)gpu_ptr;
  st->bytes = pitch * (size_t)height;
  st->format = pixel_format;
  st->width = width;
  st->height = height;
  st->linesize = (int)pitch;
  // GPU has valid data, CPU buffer is uninitialized
  // cuda_dirty=true means GPU has new data that needs download to CPU
  // cpu_dirty=false means CPU data doesn't need upload to GPU
  st->cpu_dirty = false;
  st->cuda_dirty =
      true; // GPU has valid data that needs to be downloaded when CPU is needed
  st->from_external = true;   // Mark as external memory (not from pool)
  st->owns_memory = owns_memory; // Whether to free GPU memory on release

  result.frame = frame;
  result.background = background;
  result.abs_black_threshold = abs_black_threshold;

#else
  (void)gpu_ptr;
  (void)pitch;
  (void)width;
  (void)height;
  (void)pixel_format;
  (void)background;
  (void)abs_black_threshold;
  (void)owns_memory;
  errOutput("create_image_from_gpu: CUDA support not available");
#endif

  return result;
}

bool image_is_gpu_resident(Image *image) {
#if defined(UNPAPER_WITH_CUDA) && (UNPAPER_WITH_CUDA)
  if (image == NULL || image->frame == NULL) {
    return false;
  }

  ImageCudaState *st = image_cuda_state_get(image, false);
  if (st == NULL) {
    return false;
  }

  // GPU is resident (has valid data) if:
  // 1. We have GPU memory allocated (dptr != 0), AND
  // 2. GPU data is current (cuda_dirty = true means GPU was modified and is
  //    the source of truth, OR cpu_dirty = false means data is synced)
  // After CUDA processing, cuda_dirty=true indicates GPU has valid results.
  return (st->dptr != 0) && (st->cuda_dirty || !st->cpu_dirty);
#else
  (void)image;
  return false;
#endif
}

void image_set_gpu_resident(Image *image, bool resident) {
#if defined(UNPAPER_WITH_CUDA) && (UNPAPER_WITH_CUDA)
  if (image == NULL || image->frame == NULL) {
    return;
  }

  ImageCudaState *st = image_cuda_state_get(image, resident);
  if (st == NULL) {
    return;
  }

  if (resident) {
    // Mark GPU as current - image_ensure_cuda will skip upload
    st->cuda_dirty = false;
    st->cpu_dirty = false; // CPU might be stale but that's okay
  } else {
    // Mark CPU as current - next image_ensure_cuda will upload
    st->cpu_dirty = true;
    st->cuda_dirty = false;
  }
#else
  (void)image;
  (void)resident;
#endif
}

void *image_get_gpu_ptr(Image *image) {
#if defined(UNPAPER_WITH_CUDA) && (UNPAPER_WITH_CUDA)
  if (image == NULL || image->frame == NULL) {
    return NULL;
  }

  ImageCudaState *st = image_cuda_state_get(image, false);
  if (st == NULL || st->dptr == 0) {
    return NULL;
  }

  return (void *)(uintptr_t)st->dptr;
#else
  (void)image;
  return NULL;
#endif
}

size_t image_get_gpu_pitch(Image *image) {
#if defined(UNPAPER_WITH_CUDA) && (UNPAPER_WITH_CUDA)
  if (image == NULL || image->frame == NULL) {
    return 0;
  }

  ImageCudaState *st = image_cuda_state_get(image, false);
  if (st == NULL || st->dptr == 0) {
    return 0;
  }

  return (size_t)st->linesize;
#else
  (void)image;
  return 0;
#endif
}
