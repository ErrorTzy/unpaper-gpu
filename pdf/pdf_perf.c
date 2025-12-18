// SPDX-FileCopyrightText: 2025 The unpaper authors
//
// SPDX-License-Identifier: GPL-2.0-only

#include "pdf_perf.h"
#include "lib/logging.h"

#include <pthread.h>
#include <stdatomic.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#ifdef UNPAPER_WITH_CUDA
#include "imageprocess/cuda_runtime.h"
#include "imageprocess/cuda_stream_pool.h"
#endif

// ============================================================================
// Pinned Memory Buffer Pool Implementation
// ============================================================================

// Slot states
#define SLOT_FREE 0
#define SLOT_IN_USE 1

typedef struct {
  void *ptr;
  size_t capacity;
  atomic_int state;
  bool is_pinned;
} PinnedSlot;

struct PdfPinnedPool {
  PinnedSlot *slots;
  int num_slots;
  size_t buffer_size;

  // Statistics
  atomic_size_t total_acquired;
  atomic_size_t pool_hits;
  atomic_size_t pool_misses;
  atomic_size_t oversized_allocs;
};

PdfPinnedPool *pdf_pinned_pool_create(int num_buffers, size_t buffer_size) {
  if (num_buffers <= 0 || buffer_size == 0) {
    return NULL;
  }

  PdfPinnedPool *pool = calloc(1, sizeof(PdfPinnedPool));
  if (!pool) {
    return NULL;
  }

  pool->slots = calloc((size_t)num_buffers, sizeof(PinnedSlot));
  if (!pool->slots) {
    free(pool);
    return NULL;
  }

  pool->num_slots = num_buffers;
  pool->buffer_size = buffer_size;

  atomic_init(&pool->total_acquired, 0);
  atomic_init(&pool->pool_hits, 0);
  atomic_init(&pool->pool_misses, 0);
  atomic_init(&pool->oversized_allocs, 0);

  // Pre-allocate pinned buffers
  for (int i = 0; i < num_buffers; i++) {
    PinnedSlot *slot = &pool->slots[i];
    atomic_init(&slot->state, SLOT_FREE);
    slot->capacity = buffer_size;

#ifdef UNPAPER_WITH_CUDA
    UnpaperCudaPinnedBuffer buf;
    if (unpaper_cuda_pinned_alloc(&buf, buffer_size)) {
      slot->ptr = buf.ptr;
      slot->is_pinned = true;
    } else {
      // Fall back to regular malloc
      slot->ptr = malloc(buffer_size);
      slot->is_pinned = false;
    }
#else
    slot->ptr = malloc(buffer_size);
    slot->is_pinned = false;
#endif

    if (!slot->ptr) {
      // Cleanup already allocated slots
      for (int j = 0; j < i; j++) {
#ifdef UNPAPER_WITH_CUDA
        if (pool->slots[j].is_pinned) {
          UnpaperCudaPinnedBuffer buf = {
              .ptr = pool->slots[j].ptr,
              .bytes = pool->slots[j].capacity,
              .is_pinned = true};
          unpaper_cuda_pinned_free(&buf);
        } else {
          free(pool->slots[j].ptr);
        }
#else
        free(pool->slots[j].ptr);
#endif
      }
      free(pool->slots);
      free(pool);
      return NULL;
    }
  }

  verboseLog(VERBOSE_DEBUG,
             "PDF pinned pool: created %d buffers of %zu bytes%s\n",
             num_buffers, buffer_size,
             pool->slots[0].is_pinned ? " (pinned)" : " (regular)");

  return pool;
}

void pdf_pinned_pool_destroy(PdfPinnedPool *pool) {
  if (!pool) {
    return;
  }

  for (int i = 0; i < pool->num_slots; i++) {
    PinnedSlot *slot = &pool->slots[i];
    if (slot->ptr) {
#ifdef UNPAPER_WITH_CUDA
      if (slot->is_pinned) {
        UnpaperCudaPinnedBuffer buf = {
            .ptr = slot->ptr, .bytes = slot->capacity, .is_pinned = true};
        unpaper_cuda_pinned_free(&buf);
      } else {
        free(slot->ptr);
      }
#else
      free(slot->ptr);
#endif
    }
  }

  free(pool->slots);
  free(pool);
}

PdfPinnedBuffer pdf_pinned_pool_acquire(PdfPinnedPool *pool, size_t min_size) {
  PdfPinnedBuffer result = {0};

  if (!pool) {
    // No pool - fall back to malloc
    result.ptr = malloc(min_size);
    result.capacity = min_size;
    result.is_pinned = false;
    result.slot_index = -1;
    return result;
  }

  atomic_fetch_add(&pool->total_acquired, 1);

  // Check if size exceeds pool buffer size
  if (min_size > pool->buffer_size) {
    atomic_fetch_add(&pool->oversized_allocs, 1);
    // Fall back to malloc for oversized requests
    result.ptr = malloc(min_size);
    result.capacity = min_size;
    result.is_pinned = false;
    result.slot_index = -1;
    return result;
  }

  // Try to find a free slot
  for (int i = 0; i < pool->num_slots; i++) {
    int expected = SLOT_FREE;
    if (atomic_compare_exchange_strong(&pool->slots[i].state, &expected,
                                       SLOT_IN_USE)) {
      atomic_fetch_add(&pool->pool_hits, 1);
      result.ptr = pool->slots[i].ptr;
      result.capacity = pool->slots[i].capacity;
      result.is_pinned = pool->slots[i].is_pinned;
      result.slot_index = i;
      return result;
    }
  }

  // No free slot - fall back to malloc
  atomic_fetch_add(&pool->pool_misses, 1);
  result.ptr = malloc(min_size);
  result.capacity = min_size;
  result.is_pinned = false;
  result.slot_index = -1;
  return result;
}

void pdf_pinned_pool_release(PdfPinnedPool *pool, PdfPinnedBuffer *buffer) {
  if (!buffer || !buffer->ptr) {
    return;
  }

  if (buffer->slot_index >= 0 && pool &&
      buffer->slot_index < pool->num_slots) {
    // Return to pool
    atomic_store(&pool->slots[buffer->slot_index].state, SLOT_FREE);
  } else {
    // Not from pool - free directly
    free(buffer->ptr);
  }

  buffer->ptr = NULL;
  buffer->size = 0;
  buffer->capacity = 0;
  buffer->slot_index = -1;
}

PdfPinnedPoolStats pdf_pinned_pool_get_stats(const PdfPinnedPool *pool) {
  PdfPinnedPoolStats stats = {0};
  if (!pool) {
    return stats;
  }

  stats.total_buffers = pool->num_slots;
  stats.buffer_capacity = pool->buffer_size;
  stats.total_acquired = atomic_load(&pool->total_acquired);
  stats.pool_hits = atomic_load(&pool->pool_hits);
  stats.pool_misses = atomic_load(&pool->pool_misses);
  stats.oversized_allocs = atomic_load(&pool->oversized_allocs);

  // Count in-use buffers
  for (int i = 0; i < pool->num_slots; i++) {
    if (atomic_load(&pool->slots[i].state) == SLOT_IN_USE) {
      stats.buffers_in_use++;
    }
  }

  return stats;
}

// ============================================================================
// Encoded Buffer Pool Implementation
// ============================================================================

typedef struct {
  uint8_t *data;
  size_t capacity;
  atomic_int state;
} EncodeSlot;

struct PdfEncodePool {
  EncodeSlot *slots;
  int num_slots;
  size_t initial_size;

  // Statistics
  atomic_size_t total_acquired;
  atomic_size_t pool_hits;
  atomic_size_t pool_misses;
  atomic_size_t total_resizes;
  atomic_size_t bytes_allocated;
};

PdfEncodePool *pdf_encode_pool_create(int num_buffers, size_t initial_size) {
  if (num_buffers <= 0 || initial_size == 0) {
    return NULL;
  }

  PdfEncodePool *pool = calloc(1, sizeof(PdfEncodePool));
  if (!pool) {
    return NULL;
  }

  pool->slots = calloc((size_t)num_buffers, sizeof(EncodeSlot));
  if (!pool->slots) {
    free(pool);
    return NULL;
  }

  pool->num_slots = num_buffers;
  pool->initial_size = initial_size;

  atomic_init(&pool->total_acquired, 0);
  atomic_init(&pool->pool_hits, 0);
  atomic_init(&pool->pool_misses, 0);
  atomic_init(&pool->total_resizes, 0);
  atomic_init(&pool->bytes_allocated, 0);

  // Pre-allocate buffers
  for (int i = 0; i < num_buffers; i++) {
    pool->slots[i].data = malloc(initial_size);
    pool->slots[i].capacity = initial_size;
    atomic_init(&pool->slots[i].state, SLOT_FREE);

    if (!pool->slots[i].data) {
      // Cleanup
      for (int j = 0; j < i; j++) {
        free(pool->slots[j].data);
      }
      free(pool->slots);
      free(pool);
      return NULL;
    }

    atomic_fetch_add(&pool->bytes_allocated, initial_size);
  }

  verboseLog(VERBOSE_DEBUG,
             "PDF encode pool: created %d buffers of %zu bytes\n", num_buffers,
             initial_size);

  return pool;
}

void pdf_encode_pool_destroy(PdfEncodePool *pool) {
  if (!pool) {
    return;
  }

  for (int i = 0; i < pool->num_slots; i++) {
    if (pool->slots[i].data) {
      free(pool->slots[i].data);
    }
  }

  free(pool->slots);
  free(pool);
}

PdfEncodeBuffer pdf_encode_pool_acquire(PdfEncodePool *pool, size_t min_size) {
  PdfEncodeBuffer result = {0};

  if (!pool) {
    // No pool - fall back to malloc
    result.data = malloc(min_size);
    result.capacity = min_size;
    result.from_pool = false;
    result.slot_index = -1;
    return result;
  }

  atomic_fetch_add(&pool->total_acquired, 1);

  // Try to find a free slot
  for (int i = 0; i < pool->num_slots; i++) {
    int expected = SLOT_FREE;
    if (atomic_compare_exchange_strong(&pool->slots[i].state, &expected,
                                       SLOT_IN_USE)) {
      atomic_fetch_add(&pool->pool_hits, 1);

      // Grow buffer if needed
      if (pool->slots[i].capacity < min_size) {
        uint8_t *new_data = realloc(pool->slots[i].data, min_size);
        if (new_data) {
          size_t old_cap = pool->slots[i].capacity;
          pool->slots[i].data = new_data;
          pool->slots[i].capacity = min_size;
          atomic_fetch_add(&pool->bytes_allocated, min_size - old_cap);
          atomic_fetch_add(&pool->total_resizes, 1);
        }
      }

      result.data = pool->slots[i].data;
      result.capacity = pool->slots[i].capacity;
      result.from_pool = true;
      result.slot_index = i;
      return result;
    }
  }

  // No free slot - fall back to malloc
  atomic_fetch_add(&pool->pool_misses, 1);
  result.data = malloc(min_size);
  result.capacity = min_size;
  result.from_pool = false;
  result.slot_index = -1;
  return result;
}

bool pdf_encode_pool_resize(PdfEncodePool *pool, PdfEncodeBuffer *buffer,
                            size_t new_size) {
  if (!buffer || !buffer->data) {
    return false;
  }

  if (new_size <= buffer->capacity) {
    buffer->size = new_size;
    return true;
  }

  if (buffer->from_pool && pool && buffer->slot_index >= 0 &&
      buffer->slot_index < pool->num_slots) {
    // Pool buffer - grow in place
    uint8_t *new_data = realloc(buffer->data, new_size);
    if (!new_data) {
      return false;
    }
    size_t old_cap = pool->slots[buffer->slot_index].capacity;
    pool->slots[buffer->slot_index].data = new_data;
    pool->slots[buffer->slot_index].capacity = new_size;
    buffer->data = new_data;
    buffer->capacity = new_size;
    buffer->size = new_size;
    atomic_fetch_add(&pool->bytes_allocated, new_size - old_cap);
    atomic_fetch_add(&pool->total_resizes, 1);
    return true;
  } else {
    // Not from pool - simple realloc
    uint8_t *new_data = realloc(buffer->data, new_size);
    if (!new_data) {
      return false;
    }
    buffer->data = new_data;
    buffer->capacity = new_size;
    buffer->size = new_size;
    return true;
  }
}

void pdf_encode_pool_release(PdfEncodePool *pool, PdfEncodeBuffer *buffer) {
  if (!buffer || !buffer->data) {
    return;
  }

  if (buffer->from_pool && pool && buffer->slot_index >= 0 &&
      buffer->slot_index < pool->num_slots) {
    // Return to pool (don't free the data, just mark as free)
    atomic_store(&pool->slots[buffer->slot_index].state, SLOT_FREE);
  } else {
    // Not from pool - free directly
    free(buffer->data);
  }

  buffer->data = NULL;
  buffer->size = 0;
  buffer->capacity = 0;
  buffer->slot_index = -1;
  buffer->from_pool = false;
}

uint8_t *pdf_encode_pool_detach(PdfEncodePool *pool, PdfEncodeBuffer *buffer) {
  if (!buffer || !buffer->data) {
    return NULL;
  }

  uint8_t *data = buffer->data;

  if (buffer->from_pool && pool && buffer->slot_index >= 0 &&
      buffer->slot_index < pool->num_slots) {
    // Detaching from pool - allocate new buffer for the slot
    EncodeSlot *slot = &pool->slots[buffer->slot_index];
    slot->data = malloc(pool->initial_size);
    slot->capacity = pool->initial_size;
    if (slot->data) {
      atomic_fetch_add(&pool->bytes_allocated, pool->initial_size);
    }
    atomic_store(&slot->state, SLOT_FREE);
  }

  buffer->data = NULL;
  buffer->size = 0;
  buffer->capacity = 0;
  buffer->slot_index = -1;
  buffer->from_pool = false;

  return data;
}

PdfEncodePoolStats pdf_encode_pool_get_stats(const PdfEncodePool *pool) {
  PdfEncodePoolStats stats = {0};
  if (!pool) {
    return stats;
  }

  stats.total_buffers = pool->num_slots;
  stats.initial_capacity = pool->initial_size;
  stats.total_acquired = atomic_load(&pool->total_acquired);
  stats.pool_hits = atomic_load(&pool->pool_hits);
  stats.pool_misses = atomic_load(&pool->pool_misses);
  stats.total_resizes = atomic_load(&pool->total_resizes);
  stats.bytes_allocated = atomic_load(&pool->bytes_allocated);

  // Count in-use buffers
  for (int i = 0; i < pool->num_slots; i++) {
    if (atomic_load(&pool->slots[i].state) == SLOT_IN_USE) {
      stats.buffers_in_use++;
    }
  }

  return stats;
}

// ============================================================================
// Stream Assignment (GPU only)
// ============================================================================

#ifdef UNPAPER_WITH_CUDA

struct UnpaperCudaStream *pdf_get_stream_for_page(int page_index,
                                                   int num_streams) {
  // Round-robin stream assignment via global stream pool
  (void)page_index;
  (void)num_streams;

  // Get stream from global pool
  return cuda_stream_pool_global_acquire();
}

#endif // UNPAPER_WITH_CUDA

// ============================================================================
// Global State
// ============================================================================

static PdfPinnedPool *g_pinned_pool = NULL;
static PdfEncodePool *g_encode_pool = NULL;
static pthread_mutex_t g_init_mutex = PTHREAD_MUTEX_INITIALIZER;
static bool g_initialized = false;

bool pdf_perf_init(int num_pinned_buffers, size_t pinned_buffer_size,
                   int num_encode_buffers, size_t encode_buffer_size) {
  pthread_mutex_lock(&g_init_mutex);

  if (g_initialized) {
    pthread_mutex_unlock(&g_init_mutex);
    return true;
  }

  // Create pinned pool
  if (num_pinned_buffers > 0 && pinned_buffer_size > 0) {
    g_pinned_pool =
        pdf_pinned_pool_create(num_pinned_buffers, pinned_buffer_size);
    if (!g_pinned_pool) {
      verboseLog(VERBOSE_MORE, "PDF perf: failed to create pinned pool\n");
    }
  }

  // Create encode pool
  if (num_encode_buffers > 0 && encode_buffer_size > 0) {
    g_encode_pool =
        pdf_encode_pool_create(num_encode_buffers, encode_buffer_size);
    if (!g_encode_pool) {
      verboseLog(VERBOSE_MORE, "PDF perf: failed to create encode pool\n");
    }
  }

  g_initialized = true;

  pthread_mutex_unlock(&g_init_mutex);
  return true;
}

void pdf_perf_cleanup(void) {
  pthread_mutex_lock(&g_init_mutex);

  if (g_pinned_pool) {
    pdf_pinned_pool_destroy(g_pinned_pool);
    g_pinned_pool = NULL;
  }

  if (g_encode_pool) {
    pdf_encode_pool_destroy(g_encode_pool);
    g_encode_pool = NULL;
  }

  g_initialized = false;

  pthread_mutex_unlock(&g_init_mutex);
}

PdfPinnedPool *pdf_perf_get_pinned_pool(void) { return g_pinned_pool; }

PdfEncodePool *pdf_perf_get_encode_pool(void) { return g_encode_pool; }

void pdf_perf_print_stats(void) {
  if (g_pinned_pool) {
    PdfPinnedPoolStats stats = pdf_pinned_pool_get_stats(g_pinned_pool);
    fprintf(stderr,
            "PDF Pinned Memory Pool:\n"
            "  Buffers: %d (in use: %d)\n"
            "  Buffer capacity: %zu bytes\n"
            "  Total acquired: %zu\n"
            "  Pool hits: %zu (%.1f%%)\n"
            "  Pool misses: %zu\n"
            "  Oversized allocs: %zu\n",
            stats.total_buffers, stats.buffers_in_use, stats.buffer_capacity,
            stats.total_acquired,
            stats.pool_hits,
            stats.total_acquired > 0
                ? (100.0 * (double)stats.pool_hits / (double)stats.total_acquired)
                : 0.0,
            stats.pool_misses, stats.oversized_allocs);
  }

  if (g_encode_pool) {
    PdfEncodePoolStats stats = pdf_encode_pool_get_stats(g_encode_pool);
    fprintf(stderr,
            "PDF Encode Buffer Pool:\n"
            "  Buffers: %d (in use: %d)\n"
            "  Initial capacity: %zu bytes\n"
            "  Total acquired: %zu\n"
            "  Pool hits: %zu (%.1f%%)\n"
            "  Pool misses: %zu\n"
            "  Total resizes: %zu\n"
            "  Total bytes allocated: %zu\n",
            stats.total_buffers, stats.buffers_in_use, stats.initial_capacity,
            stats.total_acquired,
            stats.pool_hits,
            stats.total_acquired > 0
                ? (100.0 * (double)stats.pool_hits / (double)stats.total_acquired)
                : 0.0,
            stats.pool_misses, stats.total_resizes, stats.bytes_allocated);
  }
}
