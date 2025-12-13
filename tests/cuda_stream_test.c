// SPDX-FileCopyrightText: 2025 The unpaper authors
//
// SPDX-License-Identifier: GPL-2.0-only

#include <assert.h>
#include <stdbool.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "imageprocess/cuda_runtime.h"
#include "lib/logging.h"

static void fill_pattern(uint8_t *buf, size_t bytes) {
  for (size_t i = 0; i < bytes; i++) {
    buf[i] = (uint8_t)(i * 37u + 11u);
  }
}

int main(void) {
  UnpaperCudaInitStatus st = unpaper_cuda_try_init();
  if (st != UNPAPER_CUDA_INIT_OK) {
    errOutput("%s", unpaper_cuda_init_status_string(st));
    return 1;
  }

  const size_t bytes = 4096;
  UnpaperCudaStream *stream = unpaper_cuda_stream_create();
  if (stream == NULL) {
    stream = unpaper_cuda_stream_get_default();
  }
  assert(stream != NULL);

  size_t pinned_capacity = 0;
  uint8_t *pinned = (uint8_t *)unpaper_cuda_stream_pinned_reserve(
      stream, bytes, &pinned_capacity);
  assert(pinned != NULL);
  assert(pinned_capacity >= bytes);
  fill_pattern(pinned, bytes);

  uint64_t dptr = unpaper_cuda_malloc(bytes);
  assert(dptr != 0);
  unpaper_cuda_memset_d8(dptr, 0, bytes);

  uint8_t *dst = malloc(bytes);
  uint8_t *first_run = malloc(bytes);
  assert(dst != NULL && first_run != NULL);

  size_t scratch_capacity = 0;
  uint64_t scratch1 = unpaper_cuda_scratch_reserve(bytes, &scratch_capacity);
  assert(scratch1 != 0);
  assert(scratch_capacity >= bytes);
  size_t scratch_capacity2 = 0;
  uint64_t scratch2 =
      unpaper_cuda_scratch_reserve(bytes / 2, &scratch_capacity2);
  assert(scratch2 == scratch1);
  assert(scratch_capacity2 == scratch_capacity);

  for (int run = 0; run < 2; run++) {
    memset(dst, 0, bytes);
    unpaper_cuda_memcpy_h2d_async(stream, dptr, pinned, bytes);
    unpaper_cuda_memcpy_d2h_async(stream, dst, dptr, bytes);
    unpaper_cuda_stream_synchronize_on(stream);
    assert(memcmp(dst, pinned, bytes) == 0);
    if (run == 0) {
      memcpy(first_run, dst, bytes);
    } else {
      assert(memcmp(dst, first_run, bytes) == 0);
    }
  }

  free(dst);
  free(first_run);
  unpaper_cuda_free(dptr);
  unpaper_cuda_stream_destroy(stream);
  unpaper_cuda_scratch_release_all();
  return 0;
}
