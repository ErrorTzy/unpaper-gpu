// SPDX-FileCopyrightText: 2025 The unpaper authors
//
// SPDX-License-Identifier: GPL-2.0-only

#pragma once

#include <stddef.h>
#include <stdint.h>

typedef enum {
  UNPAPER_CUDA_INIT_OK = 0,
  UNPAPER_CUDA_INIT_NO_RUNTIME = 1,
  UNPAPER_CUDA_INIT_NO_DEVICE = 2,
  UNPAPER_CUDA_INIT_ERROR = 3,
} UnpaperCudaInitStatus;

UnpaperCudaInitStatus unpaper_cuda_try_init(void);
const char *unpaper_cuda_init_status_string(UnpaperCudaInitStatus st);

uint64_t unpaper_cuda_malloc(size_t bytes);
void unpaper_cuda_free(uint64_t dptr);

void unpaper_cuda_memcpy_h2d(uint64_t dst, const void *src, size_t bytes);
void unpaper_cuda_memcpy_d2h(void *dst, uint64_t src, size_t bytes);
