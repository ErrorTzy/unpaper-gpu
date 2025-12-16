// SPDX-FileCopyrightText: 2025 The unpaper authors
//
// SPDX-License-Identifier: GPL-2.0-only

#pragma once

typedef enum {
  UNPAPER_CUDA_FMT_INVALID = 0,
  UNPAPER_CUDA_FMT_GRAY8 = 1,
  UNPAPER_CUDA_FMT_Y400A = 2,
  UNPAPER_CUDA_FMT_RGB24 = 3,
  UNPAPER_CUDA_FMT_MONOWHITE = 4,
  UNPAPER_CUDA_FMT_MONOBLACK = 5,
} UnpaperCudaFormat;
