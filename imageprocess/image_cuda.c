// SPDX-FileCopyrightText: 2025 The unpaper authors
//
// SPDX-License-Identifier: GPL-2.0-only

#include "imageprocess/image.h"

#include <libavutil/frame.h>
#include <libavutil/pixfmt.h>

#include "lib/logging.h"

#if defined(UNPAPER_WITH_CUDA) && (UNPAPER_WITH_CUDA)
#include "imageprocess/cuda_runtime.h"
#endif

void image_cuda_release(Image *image) {
#if defined(UNPAPER_WITH_CUDA) && (UNPAPER_WITH_CUDA)
  if (image == NULL) {
    return;
  }
  if (image->cuda.dptr != 0) {
    unpaper_cuda_free(image->cuda.dptr);
    image->cuda.dptr = 0;
    image->cuda.bytes = 0;
    image->cuda.format = -1;
    image->cuda.width = 0;
    image->cuda.height = 0;
    image->cuda.linesize = 0;
  }
  image->cpu_dirty = false;
  image->cuda_dirty = false;
#else
  (void)image;
#endif
}

void image_mark_cpu_dirty(Image *image) {
#if defined(UNPAPER_WITH_CUDA) && (UNPAPER_WITH_CUDA)
  if (image == NULL) {
    return;
  }
  image->cpu_dirty = true;
  image->cuda_dirty = false;
#else
  (void)image;
#endif
}

void image_mark_cuda_dirty(Image *image) {
#if defined(UNPAPER_WITH_CUDA) && (UNPAPER_WITH_CUDA)
  if (image == NULL) {
    return;
  }
  image->cuda_dirty = true;
  image->cpu_dirty = false;
#else
  (void)image;
#endif
}

void image_ensure_cuda(Image *image) {
#if defined(UNPAPER_WITH_CUDA) && (UNPAPER_WITH_CUDA)
  if (image == NULL || image->frame == NULL) {
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

  UnpaperCudaInitStatus st = unpaper_cuda_try_init();
  if (st != UNPAPER_CUDA_INIT_OK) {
    errOutput("%s", unpaper_cuda_init_status_string(st));
  }

  const size_t bytes = (size_t)image->frame->linesize[0] *
                       (size_t)image->frame->height;
  if (bytes == 0) {
    errOutput("invalid image buffer size for CUDA upload.");
  }

  const bool need_alloc =
      (image->cuda.dptr == 0) || (image->cuda.bytes != bytes) ||
      (image->cuda.format != image->frame->format) ||
      (image->cuda.width != image->frame->width) ||
      (image->cuda.height != image->frame->height) ||
      (image->cuda.linesize != image->frame->linesize[0]);
  if (need_alloc) {
    image_cuda_release(image);
    image->cuda.dptr = unpaper_cuda_malloc(bytes);
    image->cuda.bytes = bytes;
    image->cuda.format = image->frame->format;
    image->cuda.width = image->frame->width;
    image->cuda.height = image->frame->height;
    image->cuda.linesize = image->frame->linesize[0];
    image->cpu_dirty = true;
    image->cuda_dirty = false;
  }

  if (image->cpu_dirty) {
    unpaper_cuda_memcpy_h2d(image->cuda.dptr, image->frame->data[0], bytes);
    image->cpu_dirty = false;
    image->cuda_dirty = false;
  }
#else
  (void)image;
  errOutput("CUDA upload requested, but this build has no CUDA support.");
#endif
}

void image_ensure_cpu(Image *image) {
#if defined(UNPAPER_WITH_CUDA) && (UNPAPER_WITH_CUDA)
  if (image == NULL || image->frame == NULL) {
    return;
  }
  if (image->cuda.dptr == 0) {
    return;
  }

  if (image->cuda_dirty) {
    const size_t bytes = (size_t)image->frame->linesize[0] *
                         (size_t)image->frame->height;
    if (bytes == 0 || bytes != image->cuda.bytes) {
      errOutput("invalid image buffer size for CUDA download.");
    }

    UnpaperCudaInitStatus st = unpaper_cuda_try_init();
    if (st != UNPAPER_CUDA_INIT_OK) {
      errOutput("%s", unpaper_cuda_init_status_string(st));
    }

    unpaper_cuda_memcpy_d2h(image->frame->data[0], image->cuda.dptr, bytes);
    image->cuda_dirty = false;
    image->cpu_dirty = false;
  }
#else
  (void)image;
#endif
}

