// SPDX-FileCopyrightText: 2025 The unpaper authors
//
// SPDX-License-Identifier: GPL-2.0-only

#include "imageprocess/cuda_mempool.h"

#if defined(UNPAPER_WITH_CUDA) && (UNPAPER_WITH_CUDA)

#include <pthread.h>
#include <stdatomic.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "imageprocess/cuda_runtime.h"
#include "lib/logging.h"

// Buffer slot states
#define SLOT_FREE 0
#define SLOT_IN_USE 1

typedef struct {
  uint64_t dptr;       // GPU device pointer
  atomic_int in_use;   // SLOT_FREE or SLOT_IN_USE
} PoolSlot;

// Fallback allocation tracking (for non-pooled allocations)
typedef struct FallbackAlloc {
  uint64_t dptr;
  size_t bytes;
  struct FallbackAlloc *next;
} FallbackAlloc;

struct CudaMemPool {
  PoolSlot *slots;              // Pre-allocated buffer slots
  size_t slot_count;            // Number of slots
  size_t buffer_size;           // Size of each buffer

  // Fallback allocations for size mismatches
  FallbackAlloc *fallback_head;
  pthread_mutex_t fallback_mutex;

  // Statistics (atomic for thread safety)
  atomic_size_t total_allocations;
  atomic_size_t pool_hits;
  atomic_size_t pool_misses;
  atomic_size_t size_mismatches;   // Misses due to image size > pool buffer
  atomic_size_t pool_exhaustion;   // Misses due to all slots in use
  atomic_size_t current_in_use;
  atomic_size_t peak_in_use;
};

// Global singleton pool
static CudaMemPool *global_pool = NULL;
static pthread_mutex_t global_pool_mutex = PTHREAD_MUTEX_INITIALIZER;

CudaMemPool *cuda_mempool_create(size_t buffer_count, size_t buffer_size) {
  if (buffer_count == 0 || buffer_size == 0) {
    return NULL;
  }

  CudaMemPool *pool = calloc(1, sizeof(CudaMemPool));
  if (pool == NULL) {
    return NULL;
  }

  pool->slots = calloc(buffer_count, sizeof(PoolSlot));
  if (pool->slots == NULL) {
    free(pool);
    return NULL;
  }

  pool->slot_count = buffer_count;
  pool->buffer_size = buffer_size;
  pool->fallback_head = NULL;
  pthread_mutex_init(&pool->fallback_mutex, NULL);

  atomic_init(&pool->total_allocations, 0);
  atomic_init(&pool->pool_hits, 0);
  atomic_init(&pool->pool_misses, 0);
  atomic_init(&pool->size_mismatches, 0);
  atomic_init(&pool->pool_exhaustion, 0);
  atomic_init(&pool->current_in_use, 0);
  atomic_init(&pool->peak_in_use, 0);

  // Pre-allocate all GPU buffers
  for (size_t i = 0; i < buffer_count; i++) {
    pool->slots[i].dptr = unpaper_cuda_malloc(buffer_size);
    if (pool->slots[i].dptr == 0) {
      // Allocation failed - clean up and return NULL
      for (size_t j = 0; j < i; j++) {
        unpaper_cuda_free(pool->slots[j].dptr);
      }
      pthread_mutex_destroy(&pool->fallback_mutex);
      free(pool->slots);
      free(pool);
      return NULL;
    }
    atomic_init(&pool->slots[i].in_use, SLOT_FREE);
  }

  return pool;
}

void cuda_mempool_destroy(CudaMemPool *pool) {
  if (pool == NULL) {
    return;
  }

  // Free pooled buffers
  for (size_t i = 0; i < pool->slot_count; i++) {
    if (pool->slots[i].dptr != 0) {
      unpaper_cuda_free(pool->slots[i].dptr);
    }
  }
  free(pool->slots);

  // Free fallback allocations
  pthread_mutex_lock(&pool->fallback_mutex);
  FallbackAlloc *fb = pool->fallback_head;
  while (fb != NULL) {
    FallbackAlloc *next = fb->next;
    if (fb->dptr != 0) {
      unpaper_cuda_free(fb->dptr);
    }
    free(fb);
    fb = next;
  }
  pthread_mutex_unlock(&pool->fallback_mutex);

  pthread_mutex_destroy(&pool->fallback_mutex);
  free(pool);
}

uint64_t cuda_mempool_acquire(CudaMemPool *pool, size_t bytes) {
  if (pool == NULL || bytes == 0) {
    return 0;
  }

  atomic_fetch_add(&pool->total_allocations, 1);

  // Check if size matches pool buffer size
  bool size_mismatch = (bytes > pool->buffer_size);
  if (!size_mismatch) {
    // Try to acquire a free slot (lock-free)
    for (size_t i = 0; i < pool->slot_count; i++) {
      int expected = SLOT_FREE;
      if (atomic_compare_exchange_strong(&pool->slots[i].in_use,
                                         &expected, SLOT_IN_USE)) {
        // Got a slot
        atomic_fetch_add(&pool->pool_hits, 1);
        size_t in_use = atomic_fetch_add(&pool->current_in_use, 1) + 1;

        // Update peak atomically
        size_t peak = atomic_load(&pool->peak_in_use);
        while (in_use > peak) {
          if (atomic_compare_exchange_weak(&pool->peak_in_use, &peak, in_use)) {
            break;
          }
        }

        return pool->slots[i].dptr;
      }
    }
  }

  // Pool exhausted or size mismatch - fallback to direct allocation
  atomic_fetch_add(&pool->pool_misses, 1);
  if (size_mismatch) {
    atomic_fetch_add(&pool->size_mismatches, 1);
  } else {
    atomic_fetch_add(&pool->pool_exhaustion, 1);
  }

  uint64_t dptr = unpaper_cuda_malloc(bytes);
  if (dptr == 0) {
    return 0;
  }

  // Track fallback allocation
  FallbackAlloc *fb = malloc(sizeof(FallbackAlloc));
  if (fb != NULL) {
    fb->dptr = dptr;
    fb->bytes = bytes;

    pthread_mutex_lock(&pool->fallback_mutex);
    fb->next = pool->fallback_head;
    pool->fallback_head = fb;
    pthread_mutex_unlock(&pool->fallback_mutex);
  }

  size_t in_use = atomic_fetch_add(&pool->current_in_use, 1) + 1;
  size_t peak = atomic_load(&pool->peak_in_use);
  while (in_use > peak) {
    if (atomic_compare_exchange_weak(&pool->peak_in_use, &peak, in_use)) {
      break;
    }
  }

  return dptr;
}

void cuda_mempool_release(CudaMemPool *pool, uint64_t dptr) {
  if (pool == NULL || dptr == 0) {
    return;
  }

  // Check if this is a pooled buffer
  for (size_t i = 0; i < pool->slot_count; i++) {
    if (pool->slots[i].dptr == dptr) {
      // Return to pool
      atomic_store(&pool->slots[i].in_use, SLOT_FREE);
      atomic_fetch_sub(&pool->current_in_use, 1);
      return;
    }
  }

  // Not a pooled buffer - check fallback list
  pthread_mutex_lock(&pool->fallback_mutex);
  FallbackAlloc **pp = &pool->fallback_head;
  while (*pp != NULL) {
    if ((*pp)->dptr == dptr) {
      FallbackAlloc *fb = *pp;
      *pp = fb->next;
      pthread_mutex_unlock(&pool->fallback_mutex);

      unpaper_cuda_free(fb->dptr);
      free(fb);
      atomic_fetch_sub(&pool->current_in_use, 1);
      return;
    }
    pp = &(*pp)->next;
  }
  pthread_mutex_unlock(&pool->fallback_mutex);

  // Unknown buffer - just free it
  unpaper_cuda_free(dptr);
}

CudaMemPoolStats cuda_mempool_get_stats(const CudaMemPool *pool) {
  CudaMemPoolStats stats = {0};
  if (pool == NULL) {
    return stats;
  }

  stats.total_allocations = atomic_load(&pool->total_allocations);
  stats.pool_hits = atomic_load(&pool->pool_hits);
  stats.pool_misses = atomic_load(&pool->pool_misses);
  stats.size_mismatches = atomic_load(&pool->size_mismatches);
  stats.pool_exhaustion = atomic_load(&pool->pool_exhaustion);
  stats.current_in_use = atomic_load(&pool->current_in_use);
  stats.peak_in_use = atomic_load(&pool->peak_in_use);
  stats.buffer_count = pool->slot_count;
  stats.buffer_size = pool->buffer_size;
  stats.total_bytes_pooled = pool->slot_count * pool->buffer_size;

  return stats;
}

void cuda_mempool_print_stats(const CudaMemPool *pool) {
  if (pool == NULL) {
    return;
  }

  CudaMemPoolStats stats = cuda_mempool_get_stats(pool);

  double hit_rate = 0.0;
  if (stats.total_allocations > 0) {
    hit_rate = 100.0 * (double)stats.pool_hits / (double)stats.total_allocations;
  }

  fprintf(stderr,
          "GPU Memory Pool Statistics:\n"
          "  Pool size: %zu buffers x %zu bytes = %.2f MB\n"
          "  Total acquisitions: %zu\n"
          "  Pool hits: %zu (%.1f%%)\n"
          "  Pool misses: %zu\n"
          "  Peak concurrent usage: %zu\n",
          stats.buffer_count, stats.buffer_size,
          (double)stats.total_bytes_pooled / (1024.0 * 1024.0),
          stats.total_allocations, stats.pool_hits, hit_rate,
          stats.pool_misses, stats.peak_in_use);

  // Show breakdown of miss reasons
  if (stats.pool_misses > 0) {
    if (stats.size_mismatches > 0) {
      fprintf(stderr,
              "  WARNING: %zu allocations required larger buffers (mixed image sizes)\n"
              "           Consider increasing pool buffer size for optimal performance\n",
              stats.size_mismatches);
    }
    if (stats.pool_exhaustion > 0) {
      fprintf(stderr,
              "  Pool exhaustion: %zu (all buffers in use, needed direct allocation)\n",
              stats.pool_exhaustion);
    }
  }
}

// Global pool implementation

bool cuda_mempool_global_init(size_t buffer_count, size_t buffer_size) {
  pthread_mutex_lock(&global_pool_mutex);

  if (global_pool != NULL) {
    // Already initialized
    pthread_mutex_unlock(&global_pool_mutex);
    return true;
  }

  global_pool = cuda_mempool_create(buffer_count, buffer_size);

  pthread_mutex_unlock(&global_pool_mutex);
  return global_pool != NULL;
}

void cuda_mempool_global_cleanup(void) {
  pthread_mutex_lock(&global_pool_mutex);

  if (global_pool != NULL) {
    cuda_mempool_destroy(global_pool);
    global_pool = NULL;
  }

  pthread_mutex_unlock(&global_pool_mutex);
}

bool cuda_mempool_global_active(void) {
  pthread_mutex_lock(&global_pool_mutex);
  bool active = (global_pool != NULL);
  pthread_mutex_unlock(&global_pool_mutex);
  return active;
}

uint64_t cuda_mempool_global_acquire(size_t bytes) {
  // Fast path: check without lock
  CudaMemPool *pool = global_pool;
  if (pool != NULL) {
    return cuda_mempool_acquire(pool, bytes);
  }

  // No pool - direct allocation
  return unpaper_cuda_malloc(bytes);
}

void cuda_mempool_global_release(uint64_t dptr) {
  if (dptr == 0) {
    return;
  }

  // Fast path: check without lock
  CudaMemPool *pool = global_pool;
  if (pool != NULL) {
    cuda_mempool_release(pool, dptr);
    return;
  }

  // No pool - direct free
  unpaper_cuda_free(dptr);
}

CudaMemPoolStats cuda_mempool_global_get_stats(void) {
  CudaMemPoolStats stats = {0};

  pthread_mutex_lock(&global_pool_mutex);
  if (global_pool != NULL) {
    stats = cuda_mempool_get_stats(global_pool);
  }
  pthread_mutex_unlock(&global_pool_mutex);

  return stats;
}

void cuda_mempool_global_print_stats(void) {
  pthread_mutex_lock(&global_pool_mutex);
  if (global_pool != NULL) {
    cuda_mempool_print_stats(global_pool);
  }
  pthread_mutex_unlock(&global_pool_mutex);
}

#else // !UNPAPER_WITH_CUDA

// Stub implementations for non-CUDA builds

CudaMemPool *cuda_mempool_create(size_t buffer_count, size_t buffer_size) {
  (void)buffer_count;
  (void)buffer_size;
  return NULL;
}

void cuda_mempool_destroy(CudaMemPool *pool) { (void)pool; }

uint64_t cuda_mempool_acquire(CudaMemPool *pool, size_t bytes) {
  (void)pool;
  (void)bytes;
  return 0;
}

void cuda_mempool_release(CudaMemPool *pool, uint64_t dptr) {
  (void)pool;
  (void)dptr;
}

CudaMemPoolStats cuda_mempool_get_stats(const CudaMemPool *pool) {
  (void)pool;
  CudaMemPoolStats stats = {0};
  return stats;
}

void cuda_mempool_print_stats(const CudaMemPool *pool) { (void)pool; }

bool cuda_mempool_global_init(size_t buffer_count, size_t buffer_size) {
  (void)buffer_count;
  (void)buffer_size;
  return false;
}

void cuda_mempool_global_cleanup(void) {}

bool cuda_mempool_global_active(void) { return false; }

uint64_t cuda_mempool_global_acquire(size_t bytes) {
  (void)bytes;
  return 0;
}

void cuda_mempool_global_release(uint64_t dptr) { (void)dptr; }

CudaMemPoolStats cuda_mempool_global_get_stats(void) {
  CudaMemPoolStats stats = {0};
  return stats;
}

void cuda_mempool_global_print_stats(void) {}

#endif // UNPAPER_WITH_CUDA
