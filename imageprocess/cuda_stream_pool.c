// SPDX-FileCopyrightText: 2025 The unpaper authors
//
// SPDX-License-Identifier: GPL-2.0-only

#include "imageprocess/cuda_stream_pool.h"

#if defined(UNPAPER_WITH_CUDA) && (UNPAPER_WITH_CUDA)

#include <pthread.h>
#include <stdatomic.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "imageprocess/cuda_runtime.h"
#include "lib/logging.h"

// Stream slot states
#define SLOT_FREE 0
#define SLOT_IN_USE 1

typedef struct {
  UnpaperCudaStream *stream;
  atomic_int in_use; // SLOT_FREE or SLOT_IN_USE
} StreamSlot;

struct CudaStreamPool {
  StreamSlot *slots;
  size_t stream_count;

  // Condition variable for waiting when all streams are busy
  pthread_mutex_t mutex;
  pthread_cond_t available;

  // Statistics (atomic for thread safety)
  atomic_size_t total_acquisitions;
  atomic_size_t waits;
  atomic_size_t current_in_use;
  atomic_size_t peak_in_use;
};

// Global singleton pool
static CudaStreamPool *global_stream_pool = NULL;
static pthread_mutex_t global_stream_pool_mutex = PTHREAD_MUTEX_INITIALIZER;

CudaStreamPool *cuda_stream_pool_create(size_t stream_count) {
  if (stream_count == 0) {
    return NULL;
  }

  // Ensure CUDA is initialized first
  UnpaperCudaInitStatus init_status = unpaper_cuda_try_init();
  if (init_status != UNPAPER_CUDA_INIT_OK) {
    return NULL;
  }

  CudaStreamPool *pool = calloc(1, sizeof(CudaStreamPool));
  if (pool == NULL) {
    return NULL;
  }

  pool->slots = calloc(stream_count, sizeof(StreamSlot));
  if (pool->slots == NULL) {
    free(pool);
    return NULL;
  }

  pool->stream_count = stream_count;
  pthread_mutex_init(&pool->mutex, NULL);
  pthread_cond_init(&pool->available, NULL);

  atomic_init(&pool->total_acquisitions, 0);
  atomic_init(&pool->waits, 0);
  atomic_init(&pool->current_in_use, 0);
  atomic_init(&pool->peak_in_use, 0);

  // Create all streams
  for (size_t i = 0; i < stream_count; i++) {
    pool->slots[i].stream = unpaper_cuda_stream_create();
    if (pool->slots[i].stream == NULL) {
      // Creation failed - clean up and return NULL
      for (size_t j = 0; j < i; j++) {
        unpaper_cuda_stream_destroy(pool->slots[j].stream);
      }
      pthread_mutex_destroy(&pool->mutex);
      pthread_cond_destroy(&pool->available);
      free(pool->slots);
      free(pool);
      return NULL;
    }
    atomic_init(&pool->slots[i].in_use, SLOT_FREE);
  }

  return pool;
}

void cuda_stream_pool_destroy(CudaStreamPool *pool) {
  if (pool == NULL) {
    return;
  }

  // Synchronize and destroy all streams
  for (size_t i = 0; i < pool->stream_count; i++) {
    if (pool->slots[i].stream != NULL) {
      // Ensure stream is idle before destroying
      unpaper_cuda_stream_synchronize_on(pool->slots[i].stream);
      unpaper_cuda_stream_destroy(pool->slots[i].stream);
    }
  }

  pthread_mutex_destroy(&pool->mutex);
  pthread_cond_destroy(&pool->available);
  free(pool->slots);
  free(pool);
}

UnpaperCudaStream *cuda_stream_pool_acquire(CudaStreamPool *pool) {
  if (pool == NULL) {
    return NULL;
  }

  atomic_fetch_add(&pool->total_acquisitions, 1);

  // Fast path: try lock-free acquire
  for (size_t i = 0; i < pool->stream_count; i++) {
    int expected = SLOT_FREE;
    if (atomic_compare_exchange_strong(&pool->slots[i].in_use, &expected,
                                       SLOT_IN_USE)) {
      // Got a slot - update statistics
      size_t in_use = atomic_fetch_add(&pool->current_in_use, 1) + 1;

      // Update peak atomically
      size_t peak = atomic_load(&pool->peak_in_use);
      while (in_use > peak) {
        if (atomic_compare_exchange_weak(&pool->peak_in_use, &peak, in_use)) {
          break;
        }
      }

      return pool->slots[i].stream;
    }
  }

  // Slow path: all streams busy, wait for one to become available
  atomic_fetch_add(&pool->waits, 1);

  pthread_mutex_lock(&pool->mutex);

  while (1) {
    // Check again for a free slot
    for (size_t i = 0; i < pool->stream_count; i++) {
      int expected = SLOT_FREE;
      if (atomic_compare_exchange_strong(&pool->slots[i].in_use, &expected,
                                         SLOT_IN_USE)) {
        pthread_mutex_unlock(&pool->mutex);

        size_t in_use = atomic_fetch_add(&pool->current_in_use, 1) + 1;
        size_t peak = atomic_load(&pool->peak_in_use);
        while (in_use > peak) {
          if (atomic_compare_exchange_weak(&pool->peak_in_use, &peak, in_use)) {
            break;
          }
        }

        return pool->slots[i].stream;
      }
    }

    // Wait for a signal that a stream was released
    pthread_cond_wait(&pool->available, &pool->mutex);
  }
}

void cuda_stream_pool_release(CudaStreamPool *pool, UnpaperCudaStream *stream) {
  if (pool == NULL || stream == NULL) {
    return;
  }

  // Find the slot for this stream
  for (size_t i = 0; i < pool->stream_count; i++) {
    if (pool->slots[i].stream == stream) {
      // NOTE: We do NOT synchronize the stream here to allow parallelism.
      // Stream-ordered operations (malloc_async, memcpy_async, etc.) will
      // queue correctly after previous work. If a consumer needs results,
      // they must sync themselves (e.g., D2H copy is implicit sync).

      // Mark slot as free
      atomic_store(&pool->slots[i].in_use, SLOT_FREE);
      atomic_fetch_sub(&pool->current_in_use, 1);

      // Signal that a stream is available
      pthread_mutex_lock(&pool->mutex);
      pthread_cond_signal(&pool->available);
      pthread_mutex_unlock(&pool->mutex);
      return;
    }
  }

  // Stream not from this pool - ignore
}

CudaStreamPoolStats cuda_stream_pool_get_stats(const CudaStreamPool *pool) {
  CudaStreamPoolStats stats = {0};
  if (pool == NULL) {
    return stats;
  }

  stats.stream_count = pool->stream_count;
  stats.total_acquisitions = atomic_load(&pool->total_acquisitions);
  stats.waits = atomic_load(&pool->waits);
  stats.current_in_use = atomic_load(&pool->current_in_use);
  stats.peak_in_use = atomic_load(&pool->peak_in_use);

  return stats;
}

void cuda_stream_pool_print_stats(const CudaStreamPool *pool) {
  if (pool == NULL) {
    return;
  }

  CudaStreamPoolStats stats = cuda_stream_pool_get_stats(pool);

  double contention_rate = 0.0;
  if (stats.total_acquisitions > 0) {
    contention_rate =
        100.0 * (double)stats.waits / (double)stats.total_acquisitions;
  }

  fprintf(stderr,
          "CUDA Stream Pool Statistics:\n"
          "  Pool size: %zu streams\n"
          "  Total acquisitions: %zu\n"
          "  Wait events (contention): %zu (%.1f%%)\n"
          "  Peak concurrent usage: %zu\n",
          stats.stream_count, stats.total_acquisitions, stats.waits,
          contention_rate, stats.peak_in_use);
}

// Global pool implementation

bool cuda_stream_pool_global_init(size_t stream_count) {
  pthread_mutex_lock(&global_stream_pool_mutex);

  if (global_stream_pool != NULL) {
    // Already initialized
    pthread_mutex_unlock(&global_stream_pool_mutex);
    return true;
  }

  global_stream_pool = cuda_stream_pool_create(stream_count);

  pthread_mutex_unlock(&global_stream_pool_mutex);
  return global_stream_pool != NULL;
}

void cuda_stream_pool_global_cleanup(void) {
  pthread_mutex_lock(&global_stream_pool_mutex);

  if (global_stream_pool != NULL) {
    cuda_stream_pool_destroy(global_stream_pool);
    global_stream_pool = NULL;
  }

  pthread_mutex_unlock(&global_stream_pool_mutex);
}

bool cuda_stream_pool_global_active(void) {
  pthread_mutex_lock(&global_stream_pool_mutex);
  bool active = (global_stream_pool != NULL);
  pthread_mutex_unlock(&global_stream_pool_mutex);
  return active;
}

UnpaperCudaStream *cuda_stream_pool_global_acquire(void) {
  // Fast path: check without lock
  CudaStreamPool *pool = global_stream_pool;
  if (pool != NULL) {
    return cuda_stream_pool_acquire(pool);
  }

  // No pool - return NULL (caller should use default stream)
  return NULL;
}

void cuda_stream_pool_global_release(UnpaperCudaStream *stream) {
  if (stream == NULL) {
    return;
  }

  // Fast path: check without lock
  CudaStreamPool *pool = global_stream_pool;
  if (pool != NULL) {
    cuda_stream_pool_release(pool, stream);
    return;
  }

  // No pool - nothing to do
}

CudaStreamPoolStats cuda_stream_pool_global_get_stats(void) {
  CudaStreamPoolStats stats = {0};

  pthread_mutex_lock(&global_stream_pool_mutex);
  if (global_stream_pool != NULL) {
    stats = cuda_stream_pool_get_stats(global_stream_pool);
  }
  pthread_mutex_unlock(&global_stream_pool_mutex);

  return stats;
}

void cuda_stream_pool_global_print_stats(void) {
  pthread_mutex_lock(&global_stream_pool_mutex);
  if (global_stream_pool != NULL) {
    cuda_stream_pool_print_stats(global_stream_pool);
  }
  pthread_mutex_unlock(&global_stream_pool_mutex);
}

#else // !UNPAPER_WITH_CUDA

// Stub implementations for non-CUDA builds

CudaStreamPool *cuda_stream_pool_create(size_t stream_count) {
  (void)stream_count;
  return NULL;
}

void cuda_stream_pool_destroy(CudaStreamPool *pool) { (void)pool; }

UnpaperCudaStream *cuda_stream_pool_acquire(CudaStreamPool *pool) {
  (void)pool;
  return NULL;
}

void cuda_stream_pool_release(CudaStreamPool *pool, UnpaperCudaStream *stream) {
  (void)pool;
  (void)stream;
}

CudaStreamPoolStats cuda_stream_pool_get_stats(const CudaStreamPool *pool) {
  (void)pool;
  CudaStreamPoolStats stats = {0};
  return stats;
}

void cuda_stream_pool_print_stats(const CudaStreamPool *pool) { (void)pool; }

bool cuda_stream_pool_global_init(size_t stream_count) {
  (void)stream_count;
  return false;
}

void cuda_stream_pool_global_cleanup(void) {}

bool cuda_stream_pool_global_active(void) { return false; }

UnpaperCudaStream *cuda_stream_pool_global_acquire(void) { return NULL; }

void cuda_stream_pool_global_release(UnpaperCudaStream *stream) {
  (void)stream;
}

CudaStreamPoolStats cuda_stream_pool_global_get_stats(void) {
  CudaStreamPoolStats stats = {0};
  return stats;
}

void cuda_stream_pool_global_print_stats(void) {}

#endif // UNPAPER_WITH_CUDA
