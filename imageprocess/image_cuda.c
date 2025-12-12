// SPDX-FileCopyrightText: 2025 The unpaper authors
//
// SPDX-License-Identifier: GPL-2.0-only

#include "imageprocess/image.h"

#include <libavutil/buffer.h>
#include <libavutil/frame.h>
#include <libavutil/pixfmt.h>

#include "lib/logging.h"

#if defined(UNPAPER_WITH_CUDA) && (UNPAPER_WITH_CUDA)
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
} ImageCudaState;

#if defined(UNPAPER_WITH_CUDA) && (UNPAPER_WITH_CUDA)
static void image_cuda_state_free(void *opaque, uint8_t *data) {
  (void)opaque;
  if (data == NULL) {
    return;
  }
  ImageCudaState *st = (ImageCudaState *)data;
  if (st->dptr != 0) {
    unpaper_cuda_free(st->dptr);
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

  image->frame->opaque_ref =
      av_buffer_create((uint8_t *)st, sizeof(*st), image_cuda_state_free, NULL,
                       0);
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

  const size_t bytes = (size_t)image->frame->linesize[0] *
                       (size_t)image->frame->height;
  if (bytes == 0) {
    errOutput("invalid image buffer size for CUDA upload.");
  }

  const bool need_alloc =
      (st->dptr == 0) || (st->bytes != bytes) ||
      (st->format != image->frame->format) || (st->width != image->frame->width) ||
      (st->height != image->frame->height) ||
      (st->linesize != image->frame->linesize[0]);
  if (need_alloc) {
    if (st->dptr != 0) {
      unpaper_cuda_free(st->dptr);
      st->dptr = 0;
    }
    st->dptr = unpaper_cuda_malloc(bytes);
    st->bytes = bytes;
    st->format = image->frame->format;
    st->width = image->frame->width;
    st->height = image->frame->height;
    st->linesize = image->frame->linesize[0];
    st->cpu_dirty = true;
    st->cuda_dirty = false;
  }

  if (st->cpu_dirty) {
    unpaper_cuda_memcpy_h2d(st->dptr, image->frame->data[0], bytes);
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

  const size_t bytes = (size_t)image->frame->linesize[0] *
                       (size_t)image->frame->height;
  if (bytes == 0) {
    errOutput("invalid image buffer size for CUDA allocation.");
  }

  const bool need_alloc =
      (st->dptr == 0) || (st->bytes != bytes) ||
      (st->format != image->frame->format) ||
      (st->width != image->frame->width) || (st->height != image->frame->height) ||
      (st->linesize != image->frame->linesize[0]);
  if (need_alloc) {
    if (st->dptr != 0) {
      unpaper_cuda_free(st->dptr);
      st->dptr = 0;
    }
    st->dptr = unpaper_cuda_malloc(bytes);
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
    const size_t bytes = (size_t)image->frame->linesize[0] *
                         (size_t)image->frame->height;
    if (bytes == 0 || bytes != st->bytes) {
      errOutput("invalid image buffer size for CUDA download.");
    }

    UnpaperCudaInitStatus init_status = unpaper_cuda_try_init();
    if (init_status != UNPAPER_CUDA_INIT_OK) {
      errOutput("%s", unpaper_cuda_init_status_string(init_status));
    }

    unpaper_cuda_memcpy_d2h(image->frame->data[0], st->dptr, bytes);
    st->cuda_dirty = false;
    st->cpu_dirty = false;
  }
#else
  (void)image;
#endif
}
