// SPDX-FileCopyrightText: 2025 The unpaper authors
//
// SPDX-License-Identifier: GPL-2.0-only

#include <libavutil/frame.h>
#include <libavutil/pixfmt.h>

#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "imageprocess/cuda_runtime.h"
#include "imageprocess/image.h"

static void fill_pattern(uint8_t *buf, size_t bytes, uint8_t seed) {
  for (size_t i = 0; i < bytes; i++) {
    buf[i] = (uint8_t)(seed + (uint8_t)(i * 131u));
  }
}

static void assert_buf_eq(const uint8_t *a, const uint8_t *b, size_t bytes,
                          const char *label) {
  if (memcmp(a, b, bytes) == 0) {
    return;
  }

  fprintf(stderr, "buffer mismatch: %s\n", label);
  for (size_t i = 0; i < bytes; i++) {
    if (a[i] != b[i]) {
      fprintf(stderr, "first mismatch at %zu: got=%u expected=%u\n", i,
              (unsigned)a[i], (unsigned)b[i]);
      break;
    }
  }
  exit(1);
}

static void run_roundtrip_for_format(enum AVPixelFormat fmt) {
  Image image = EMPTY_IMAGE;

  image.frame = av_frame_alloc();
  if (image.frame == NULL) {
    fprintf(stderr, "failed to allocate AVFrame\n");
    exit(1);
  }

  image.frame->width = 37;
  image.frame->height = 19;
  image.frame->format = fmt;

  int ret = av_frame_get_buffer(image.frame, 8);
  if (ret < 0) {
    fprintf(stderr, "av_frame_get_buffer failed (%d)\n", ret);
    exit(1);
  }

  const size_t bytes = (size_t)image.frame->linesize[0] *
                       (size_t)image.frame->height;
  if (bytes == 0) {
    fprintf(stderr, "invalid buffer size\n");
    exit(1);
  }

  uint8_t *expected_a = malloc(bytes);
  uint8_t *expected_c = malloc(bytes);
  if (expected_a == NULL || expected_c == NULL) {
    fprintf(stderr, "malloc failed\n");
    exit(1);
  }

  fill_pattern(image.frame->data[0], bytes, 0x10);
  memcpy(expected_a, image.frame->data[0], bytes);
  image_mark_cpu_dirty(&image);
  image_ensure_cuda(&image);

  fill_pattern(image.frame->data[0], bytes, 0xA0);
  image_mark_cuda_dirty(&image);
  image_ensure_cpu(&image);
  assert_buf_eq(image.frame->data[0], expected_a, bytes, "download after upload");

  fill_pattern(image.frame->data[0], bytes, 0x33);
  memcpy(expected_c, image.frame->data[0], bytes);
  image_mark_cpu_dirty(&image);
  image_ensure_cuda(&image);

  fill_pattern(image.frame->data[0], bytes, 0xCC);
  image_mark_cuda_dirty(&image);
  image_ensure_cpu(&image);
  assert_buf_eq(image.frame->data[0], expected_c, bytes, "upload then download");

  free(expected_a);
  free(expected_c);

  unpaper_cuda_free(image.cuda.dptr);
  image.cuda.dptr = 0;

  av_frame_free(&image.frame);
}

int main(void) {
  UnpaperCudaInitStatus st = unpaper_cuda_try_init();
  if (st != UNPAPER_CUDA_INIT_OK) {
    fprintf(stderr, "skipping: %s\n", unpaper_cuda_init_status_string(st));
    return 77;
  }

  run_roundtrip_for_format(AV_PIX_FMT_GRAY8);
  run_roundtrip_for_format(AV_PIX_FMT_RGB24);
  run_roundtrip_for_format(AV_PIX_FMT_MONOWHITE);
  run_roundtrip_for_format(AV_PIX_FMT_MONOBLACK);
  run_roundtrip_for_format(AV_PIX_FMT_Y400A);

  return 0;
}

