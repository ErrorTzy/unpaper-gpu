// SPDX-FileCopyrightText: 2025 The unpaper authors
//
// SPDX-License-Identifier: GPL-2.0-only

#include "lib/threadpool.h"

#include <pthread.h>
#include <stdatomic.h>
#include <stdlib.h>
#include <unistd.h>

// Work queue capacity - power of 2 for efficient modulo
#define QUEUE_CAPACITY 4096

typedef struct {
  ThreadPoolWorkFn fn;
  void *arg;
} WorkItem;

struct ThreadPool {
  pthread_t *threads;
  int num_threads;

  // Work queue (ring buffer)
  WorkItem queue[QUEUE_CAPACITY];
  atomic_size_t head;    // Next position to read
  atomic_size_t tail;    // Next position to write
  atomic_size_t pending; // Number of items being processed

  // Synchronization
  pthread_mutex_t mutex;
  pthread_cond_t work_available;
  pthread_cond_t work_done;

  // Shutdown flag
  atomic_bool shutdown;
};

static void *worker_thread(void *arg);

ThreadPool *threadpool_create(int num_threads) {
  if (num_threads <= 0) {
    long nprocs = sysconf(_SC_NPROCESSORS_ONLN);
    num_threads = (nprocs > 0) ? (int)nprocs : 1;
    if (num_threads > 64)
      num_threads = 64;
  }

  ThreadPool *pool = calloc(1, sizeof(ThreadPool));
  if (!pool)
    return NULL;

  pool->num_threads = num_threads;
  atomic_init(&pool->head, 0);
  atomic_init(&pool->tail, 0);
  atomic_init(&pool->pending, 0);
  atomic_init(&pool->shutdown, false);

  if (pthread_mutex_init(&pool->mutex, NULL) != 0) {
    free(pool);
    return NULL;
  }

  if (pthread_cond_init(&pool->work_available, NULL) != 0) {
    pthread_mutex_destroy(&pool->mutex);
    free(pool);
    return NULL;
  }

  if (pthread_cond_init(&pool->work_done, NULL) != 0) {
    pthread_cond_destroy(&pool->work_available);
    pthread_mutex_destroy(&pool->mutex);
    free(pool);
    return NULL;
  }

  pool->threads = calloc((size_t)num_threads, sizeof(pthread_t));
  if (!pool->threads) {
    pthread_cond_destroy(&pool->work_done);
    pthread_cond_destroy(&pool->work_available);
    pthread_mutex_destroy(&pool->mutex);
    free(pool);
    return NULL;
  }

  // Create worker threads
  for (int i = 0; i < num_threads; i++) {
    // Pass thread ID via pointer arithmetic trick
    if (pthread_create(&pool->threads[i], NULL, worker_thread, pool) != 0) {
      // Failed to create thread - shutdown already created ones
      atomic_store(&pool->shutdown, true);
      pthread_cond_broadcast(&pool->work_available);
      for (int j = 0; j < i; j++) {
        pthread_join(pool->threads[j], NULL);
      }
      free(pool->threads);
      pthread_cond_destroy(&pool->work_done);
      pthread_cond_destroy(&pool->work_available);
      pthread_mutex_destroy(&pool->mutex);
      free(pool);
      return NULL;
    }
  }

  return pool;
}

void threadpool_destroy(ThreadPool *pool) {
  if (!pool)
    return;

  // Wait for pending work
  threadpool_wait(pool);

  // Signal shutdown
  atomic_store(&pool->shutdown, true);
  pthread_cond_broadcast(&pool->work_available);

  // Join all threads
  for (int i = 0; i < pool->num_threads; i++) {
    pthread_join(pool->threads[i], NULL);
  }

  free(pool->threads);
  pthread_cond_destroy(&pool->work_done);
  pthread_cond_destroy(&pool->work_available);
  pthread_mutex_destroy(&pool->mutex);
  free(pool);
}

bool threadpool_submit(ThreadPool *pool, ThreadPoolWorkFn fn, void *arg) {
  if (!pool || !fn)
    return false;

  pthread_mutex_lock(&pool->mutex);

  // Wait if queue is full
  size_t tail = atomic_load(&pool->tail);
  size_t head = atomic_load(&pool->head);
  while ((tail - head) >= QUEUE_CAPACITY) {
    pthread_cond_wait(&pool->work_done, &pool->mutex);
    if (atomic_load(&pool->shutdown)) {
      pthread_mutex_unlock(&pool->mutex);
      return false;
    }
    head = atomic_load(&pool->head);
    tail = atomic_load(&pool->tail);
  }

  // Add work to queue
  size_t idx = tail & (QUEUE_CAPACITY - 1);
  pool->queue[idx].fn = fn;
  pool->queue[idx].arg = arg;
  atomic_store(&pool->tail, tail + 1);
  atomic_fetch_add(&pool->pending, 1);

  pthread_cond_signal(&pool->work_available);
  pthread_mutex_unlock(&pool->mutex);

  return true;
}

void threadpool_wait(ThreadPool *pool) {
  if (!pool)
    return;

  pthread_mutex_lock(&pool->mutex);
  while (atomic_load(&pool->pending) > 0 ||
         atomic_load(&pool->head) < atomic_load(&pool->tail)) {
    pthread_cond_wait(&pool->work_done, &pool->mutex);
  }
  pthread_mutex_unlock(&pool->mutex);
}

int threadpool_get_num_threads(const ThreadPool *pool) {
  return pool ? pool->num_threads : 0;
}

static void *worker_thread(void *arg) {
  ThreadPool *pool = (ThreadPool *)arg;

  // Determine thread ID by finding our thread in the array
  pthread_t self = pthread_self();
  int thread_id = 0;
  for (int i = 0; i < pool->num_threads; i++) {
    if (pthread_equal(pool->threads[i], self)) {
      thread_id = i;
      break;
    }
  }

  while (true) {
    pthread_mutex_lock(&pool->mutex);

    // Wait for work
    while (atomic_load(&pool->head) >= atomic_load(&pool->tail) &&
           !atomic_load(&pool->shutdown)) {
      pthread_cond_wait(&pool->work_available, &pool->mutex);
    }

    if (atomic_load(&pool->shutdown) &&
        atomic_load(&pool->head) >= atomic_load(&pool->tail)) {
      pthread_mutex_unlock(&pool->mutex);
      break;
    }

    // Get work from queue
    size_t head = atomic_load(&pool->head);
    size_t idx = head & (QUEUE_CAPACITY - 1);
    WorkItem item = pool->queue[idx];
    atomic_store(&pool->head, head + 1);

    pthread_mutex_unlock(&pool->mutex);

    // Execute work
    item.fn(item.arg, thread_id);

    // Signal completion
    atomic_fetch_sub(&pool->pending, 1);
    pthread_cond_broadcast(&pool->work_done);
  }

  return NULL;
}
