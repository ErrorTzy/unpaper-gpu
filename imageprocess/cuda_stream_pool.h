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

// Forward declaration
typedef struct UnpaperCudaStream UnpaperCudaStream;

// CUDA stream pool for batch processing.
// Pre-allocates multiple CUDA streams to enable concurrent GPU operations.
// Thread-safe for concurrent batch processing with multiple workers.

typedef struct CudaStreamPool CudaStreamPool;

// Pool statistics for monitoring
typedef struct {
  size_t stream_count;       // Number of streams in pool
  size_t total_acquisitions; // Number of acquire calls
  size_t waits;              // Number of times we had to wait for a free stream
  size_t peak_in_use;        // High water mark of concurrent usage
  size_t current_in_use;     // Currently acquired streams
} CudaStreamPoolStats;

// Create a stream pool with N pre-allocated CUDA streams.
// stream_count: number of streams to create (e.g., 4 for quad-buffered)
// Returns NULL on failure.
CudaStreamPool *cuda_stream_pool_create(size_t stream_count);

// Destroy the pool and all streams.
void cuda_stream_pool_destroy(CudaStreamPool *pool);

// Acquire a stream from the pool.
// If all streams are in use, waits for one to become available.
// Returns NULL on error (pool destroyed or CUDA error).
UnpaperCudaStream *cuda_stream_pool_acquire(CudaStreamPool *pool);

// Release a stream back to the pool.
// NOTE: Does NOT synchronize the stream (for parallelism).
// Caller must ensure any needed CPU-visible results are obtained before release.
void cuda_stream_pool_release(CudaStreamPool *pool, UnpaperCudaStream *stream);

// Get current pool statistics.
CudaStreamPoolStats cuda_stream_pool_get_stats(const CudaStreamPool *pool);

// Print pool statistics to stderr (for --perf output).
void cuda_stream_pool_print_stats(const CudaStreamPool *pool);

// Global stream pool management for batch processing.
// These functions manage a singleton pool used during batch mode.

// Initialize global stream pool with specified number of streams.
// Call once before batch processing starts.
bool cuda_stream_pool_global_init(size_t stream_count);

// Destroy global stream pool.
// Call after batch processing completes.
void cuda_stream_pool_global_cleanup(void);

// Check if global stream pool is active.
bool cuda_stream_pool_global_active(void);

// Acquire a stream from global pool.
// Returns NULL if no pool is active (falls back to default stream handling).
UnpaperCudaStream *cuda_stream_pool_global_acquire(void);

// Release a stream to global pool.
// No-op if no pool is active.
void cuda_stream_pool_global_release(UnpaperCudaStream *stream);

// Get global pool statistics.
CudaStreamPoolStats cuda_stream_pool_global_get_stats(void);

// Print global pool statistics.
void cuda_stream_pool_global_print_stats(void);

#ifdef __cplusplus
}
#endif
