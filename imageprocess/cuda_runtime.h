// SPDX-FileCopyrightText: 2025 The unpaper authors
//
// SPDX-License-Identifier: GPL-2.0-only

#pragma once

#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>

typedef struct UnpaperCudaStream UnpaperCudaStream;

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
void unpaper_cuda_memcpy_d2d(uint64_t dst, uint64_t src, size_t bytes);

void unpaper_cuda_memcpy_h2d_async(UnpaperCudaStream *stream, uint64_t dst,
                                   const void *src, size_t bytes);
void unpaper_cuda_memcpy_d2h_async(UnpaperCudaStream *stream, void *dst,
                                   uint64_t src, size_t bytes);
void unpaper_cuda_memcpy_d2d_async(UnpaperCudaStream *stream, uint64_t dst,
                                   uint64_t src, size_t bytes);

typedef struct {
  void *ptr;
  size_t bytes;
  bool is_pinned;
} UnpaperCudaPinnedBuffer;

bool unpaper_cuda_pinned_alloc(UnpaperCudaPinnedBuffer *buf, size_t bytes);
void unpaper_cuda_pinned_free(UnpaperCudaPinnedBuffer *buf);
void *unpaper_cuda_stream_pinned_reserve(UnpaperCudaStream *stream,
                                         size_t bytes, size_t *capacity_out);

uint64_t unpaper_cuda_scratch_reserve(size_t bytes, size_t *capacity_out);
void unpaper_cuda_scratch_release_all(void);

UnpaperCudaStream *unpaper_cuda_stream_create(void);
UnpaperCudaStream *unpaper_cuda_stream_get_default(void);
void unpaper_cuda_stream_destroy(UnpaperCudaStream *stream);
void unpaper_cuda_set_current_stream(UnpaperCudaStream *stream);
UnpaperCudaStream *unpaper_cuda_get_current_stream(void);
void unpaper_cuda_stream_synchronize_on(UnpaperCudaStream *stream);

void unpaper_cuda_memset_d8(uint64_t dst, uint8_t value, size_t bytes);

void *unpaper_cuda_module_load_ptx(const char *ptx);
void unpaper_cuda_module_unload(void *module);
void *unpaper_cuda_module_get_function(void *module, const char *name);
void unpaper_cuda_launch_kernel(void *func, uint32_t grid_x, uint32_t grid_y,
                                uint32_t grid_z, uint32_t block_x,
                                uint32_t block_y, uint32_t block_z,
                                void **kernel_params);
void unpaper_cuda_launch_kernel_on_stream(UnpaperCudaStream *stream,
                                          void *func, uint32_t grid_x,
                                          uint32_t grid_y, uint32_t grid_z,
                                          uint32_t block_x, uint32_t block_y,
                                          uint32_t block_z,
                                          void **kernel_params);

bool unpaper_cuda_events_supported(void);
bool unpaper_cuda_event_pair_start(void **start, void **stop);
double unpaper_cuda_event_pair_stop_ms(void **start, void **stop);
void unpaper_cuda_stream_synchronize(void);
bool unpaper_cuda_events_supported_on(UnpaperCudaStream *stream);
bool unpaper_cuda_event_pair_start_on(UnpaperCudaStream *stream, void **start,
                                      void **stop);
double unpaper_cuda_event_pair_stop_ms_on(UnpaperCudaStream *stream,
                                          void **start, void **stop);
