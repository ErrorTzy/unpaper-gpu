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

// GPU memory pool for efficient batch processing.
// Pre-allocates GPU buffers to eliminate per-image cudaMalloc overhead.
// Thread-safe for concurrent batch processing.

typedef struct CudaMemPool CudaMemPool;

// Pool statistics for monitoring
typedef struct {
  size_t total_allocations;    // Number of acquire calls
  size_t pool_hits;            // Reuses from pool (no cudaMalloc)
  size_t pool_misses;          // Required new allocation
  size_t size_mismatches;      // Misses due to image size > pool buffer (mixed sizes)
  size_t pool_exhaustion;      // Misses due to all pool slots in use
  size_t current_in_use;       // Currently checked-out buffers
  size_t peak_in_use;          // High water mark
  size_t total_bytes_pooled;   // Total GPU memory in pool
  size_t buffer_count;         // Number of buffers in pool
  size_t buffer_size;          // Size of each buffer
} CudaMemPoolStats;

// Create a memory pool with pre-allocated GPU buffers.
// buffer_count: number of buffers to pre-allocate (e.g., 8 for triple-buffered 4-stream)
// buffer_size: size of each buffer in bytes (e.g., 4MB for A1 images)
// Returns NULL on failure.
CudaMemPool *cuda_mempool_create(size_t buffer_count, size_t buffer_size);

// Destroy the pool and free all GPU memory.
void cuda_mempool_destroy(CudaMemPool *pool);

// Acquire a buffer from the pool.
// If the requested size matches the pool's buffer size, returns a pooled buffer.
// If size differs or pool is exhausted, falls back to direct cudaMalloc.
// Returns 0 on failure.
uint64_t cuda_mempool_acquire(CudaMemPool *pool, size_t bytes);

// Release a buffer back to the pool.
// If the buffer came from the pool, returns it for reuse.
// If it was a fallback allocation, frees it with cudaFree.
void cuda_mempool_release(CudaMemPool *pool, uint64_t dptr);

// Get current pool statistics.
CudaMemPoolStats cuda_mempool_get_stats(const CudaMemPool *pool);

// Print pool statistics to stderr (for --perf output).
void cuda_mempool_print_stats(const CudaMemPool *pool);

// Global pool management for batch processing.
// These functions manage a singleton pool used by image_cuda.c.

// Initialize global pool with specified parameters.
// Call once before batch processing starts.
bool cuda_mempool_global_init(size_t buffer_count, size_t buffer_size);

// Destroy global pool.
// Call after batch processing completes.
void cuda_mempool_global_cleanup(void);

// Check if global pool is active.
bool cuda_mempool_global_active(void);

// Acquire from global pool (falls back to direct alloc if no pool).
uint64_t cuda_mempool_global_acquire(size_t bytes);

// Release to global pool (falls back to cudaFree if no pool).
void cuda_mempool_global_release(uint64_t dptr);

// Get global pool statistics.
CudaMemPoolStats cuda_mempool_global_get_stats(void);

// Print global pool statistics.
void cuda_mempool_global_print_stats(void);

// Integral buffer pool - separate from image pool due to different buffer sizes.
// Integral buffers are width*height*4 bytes (int32), typically ~35MB for A1 images.
// For batch processing with N streams, allocate 2*N buffers for double-buffering.

// Initialize global integral pool with specified parameters.
// buffer_count: typically 2 * stream_count
// buffer_size: typically width * height * 4, aligned to 512 bytes
bool cuda_mempool_integral_global_init(size_t buffer_count, size_t buffer_size);

// Destroy global integral pool.
void cuda_mempool_integral_global_cleanup(void);

// Check if global integral pool is active.
bool cuda_mempool_integral_global_active(void);

// Acquire from global integral pool (falls back to direct alloc if no pool).
uint64_t cuda_mempool_integral_global_acquire(size_t bytes);

// Release to global integral pool (falls back to cudaFree if no pool).
void cuda_mempool_integral_global_release(uint64_t dptr);

// Get global integral pool statistics.
CudaMemPoolStats cuda_mempool_integral_global_get_stats(void);

// Print global integral pool statistics.
void cuda_mempool_integral_global_print_stats(void);

#ifdef __cplusplus
}
#endif
