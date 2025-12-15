// SPDX-FileCopyrightText: 2025 The unpaper authors
//
// SPDX-License-Identifier: GPL-2.0-only

#pragma once

#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>

#include "imageprocess/cuda_runtime.h"

#ifdef __cplusplus
extern "C" {
#endif

// Opaque NPP stream context handle
typedef struct UnpaperNppContext UnpaperNppContext;

// Initialize NPP subsystem (call once after CUDA init)
bool unpaper_npp_init(void);

// Check if NPP is available
bool unpaper_npp_available(void);

// Create an NPP context for a CUDA stream
// The context caches device properties for efficient NPP calls
UnpaperNppContext *unpaper_npp_context_create(UnpaperCudaStream *stream);

// Destroy an NPP context
void unpaper_npp_context_destroy(UnpaperNppContext *ctx);

// Get the raw NppStreamContext pointer (for NPP function calls)
// Returns NULL if ctx is NULL
void *unpaper_npp_context_get_raw(UnpaperNppContext *ctx);

// Get device properties (cached after first call)
typedef struct {
  int device_id;
  int multiprocessor_count;
  int max_threads_per_multiprocessor;
  int max_threads_per_block;
  size_t shared_mem_per_block;
  int compute_capability_major;
  int compute_capability_minor;
} UnpaperNppDeviceProps;

bool unpaper_npp_get_device_props(UnpaperNppDeviceProps *props);

// Error code to string conversion
const char *unpaper_npp_status_string(int status);

#ifdef __cplusplus
}
#endif
