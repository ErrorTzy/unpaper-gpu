// SPDX-FileCopyrightText: 2025 The unpaper authors
//
// SPDX-License-Identifier: GPL-2.0-only

#pragma once

#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

// ============================================================================
// PDF Performance Optimizations (PR 8)
// ============================================================================
//
// This module provides performance optimizations for PDF processing:
// - Pinned memory buffer pool for zero-copy GPU transfers
// - Memory pool for encoded output buffers
// - Page-level stream assignment for GPU batching
//
// Thread-safety: All functions are thread-safe unless noted otherwise.

// ============================================================================
// Pinned Memory Buffer Pool
// ============================================================================
//
// Provides pre-allocated pinned memory buffers for PDF image data.
// Pinned memory enables zero-copy transfers between CPU and GPU.

typedef struct PdfPinnedPool PdfPinnedPool;

// Buffer handle from the pool
typedef struct {
  void *ptr;      // Buffer pointer (pinned memory on GPU builds)
  size_t size;    // Actual data size
  size_t capacity;// Buffer capacity
  int slot_index; // Internal: slot index for release
  bool is_pinned; // True if this is pinned memory
} PdfPinnedBuffer;

// Create a pinned memory pool.
// num_buffers: Number of pre-allocated buffers
// buffer_size: Size of each buffer (should accommodate largest expected image)
//
// Returns NULL on failure.
PdfPinnedPool *pdf_pinned_pool_create(int num_buffers, size_t buffer_size);

// Destroy a pinned memory pool.
void pdf_pinned_pool_destroy(PdfPinnedPool *pool);

// Acquire a buffer from the pool.
// min_size: Minimum required buffer size
//
// Returns a buffer with capacity >= min_size, or ptr=NULL if none available.
// The buffer MUST be released with pdf_pinned_pool_release() after use.
PdfPinnedBuffer pdf_pinned_pool_acquire(PdfPinnedPool *pool, size_t min_size);

// Release a buffer back to the pool.
void pdf_pinned_pool_release(PdfPinnedPool *pool, PdfPinnedBuffer *buffer);

// Get pool statistics.
typedef struct {
  int total_buffers;
  int buffers_in_use;
  size_t buffer_capacity;
  size_t total_acquired;  // Total acquisitions
  size_t pool_hits;       // Acquisitions satisfied from pool
  size_t pool_misses;     // Acquisitions that fell back to malloc
  size_t oversized_allocs;// Allocations larger than pool buffer size
} PdfPinnedPoolStats;

PdfPinnedPoolStats pdf_pinned_pool_get_stats(const PdfPinnedPool *pool);

// ============================================================================
// Encoded Buffer Pool
// ============================================================================
//
// Provides pooled buffers for encoded JPEG/JP2 output.
// Reduces malloc/free overhead in the encode path.

typedef struct PdfEncodePool PdfEncodePool;

// Buffer handle from the encode pool
typedef struct {
  uint8_t *data;    // Buffer pointer
  size_t size;      // Actual data size
  size_t capacity;  // Buffer capacity
  int slot_index;   // Internal: slot index for release
  bool from_pool;   // True if from pool, false if malloc'd
} PdfEncodeBuffer;

// Create an encode buffer pool.
// num_buffers: Number of pre-allocated buffers
// initial_size: Initial buffer size (will grow if needed)
//
// Returns NULL on failure.
PdfEncodePool *pdf_encode_pool_create(int num_buffers, size_t initial_size);

// Destroy an encode buffer pool.
void pdf_encode_pool_destroy(PdfEncodePool *pool);

// Acquire a buffer from the pool.
// min_size: Minimum required buffer size
//
// Returns a buffer with capacity >= min_size.
// Buffer may be from pool (from_pool=true) or malloc'd (from_pool=false).
PdfEncodeBuffer pdf_encode_pool_acquire(PdfEncodePool *pool, size_t min_size);

// Resize a buffer (may reallocate).
// This is used when encoding and the actual size exceeds initial estimate.
// Returns true on success.
bool pdf_encode_pool_resize(PdfEncodePool *pool, PdfEncodeBuffer *buffer,
                            size_t new_size);

// Release a buffer back to the pool.
void pdf_encode_pool_release(PdfEncodePool *pool, PdfEncodeBuffer *buffer);

// Detach buffer from pool (caller takes ownership of memory).
// Use this when passing the buffer to the page accumulator.
// After detach, the buffer's memory is NOT returned to the pool.
uint8_t *pdf_encode_pool_detach(PdfEncodePool *pool, PdfEncodeBuffer *buffer);

// Get encode pool statistics.
typedef struct {
  int total_buffers;
  int buffers_in_use;
  size_t initial_capacity;
  size_t total_acquired;
  size_t pool_hits;
  size_t pool_misses;
  size_t total_resizes;
  size_t bytes_allocated;
} PdfEncodePoolStats;

PdfEncodePoolStats pdf_encode_pool_get_stats(const PdfEncodePool *pool);

// ============================================================================
// Stream Assignment (GPU only)
// ============================================================================
//
// Manages CUDA streams for page-level parallelism in GPU processing.

#ifdef UNPAPER_WITH_CUDA

struct UnpaperCudaStream;

// Get a CUDA stream for a page index.
// This distributes pages across available streams in round-robin fashion.
struct UnpaperCudaStream *pdf_get_stream_for_page(int page_index,
                                                   int num_streams);

#endif // UNPAPER_WITH_CUDA

// ============================================================================
// Initialization/Cleanup
// ============================================================================

// Initialize PDF performance module.
// Call once at startup before using any other functions.
// num_pinned_buffers: Number of pinned buffers to pre-allocate
// pinned_buffer_size: Size of each pinned buffer
// num_encode_buffers: Number of encode buffers to pre-allocate
// encode_buffer_size: Initial size of each encode buffer
//
// Returns true on success.
bool pdf_perf_init(int num_pinned_buffers, size_t pinned_buffer_size,
                   int num_encode_buffers, size_t encode_buffer_size);

// Cleanup PDF performance module.
// Call once at shutdown.
void pdf_perf_cleanup(void);

// Get global pinned memory pool (may be NULL if not initialized).
PdfPinnedPool *pdf_perf_get_pinned_pool(void);

// Get global encode buffer pool (may be NULL if not initialized).
PdfEncodePool *pdf_perf_get_encode_pool(void);

// Print performance statistics.
void pdf_perf_print_stats(void);

#ifdef __cplusplus
}
#endif
