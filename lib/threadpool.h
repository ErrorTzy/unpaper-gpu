// SPDX-FileCopyrightText: 2025 The unpaper authors
//
// SPDX-License-Identifier: GPL-2.0-only

#pragma once

#include <stdbool.h>
#include <stddef.h>

// Opaque thread pool handle
typedef struct ThreadPool ThreadPool;

// Work function signature: void work_fn(void *arg, int thread_id)
typedef void (*ThreadPoolWorkFn)(void *arg, int thread_id);

// Create a thread pool with the specified number of worker threads.
// If num_threads is 0, uses the number of available CPU cores.
// Returns NULL on error.
ThreadPool *threadpool_create(int num_threads);

// Destroy the thread pool, waiting for all queued work to complete.
void threadpool_destroy(ThreadPool *pool);

// Submit work to the pool. The work function will be called with the given
// argument and the worker thread's ID (0-based). This function may block
// if the work queue is full.
// Returns true on success, false on error.
bool threadpool_submit(ThreadPool *pool, ThreadPoolWorkFn fn, void *arg);

// Wait for all submitted work to complete.
void threadpool_wait(ThreadPool *pool);

// Get the number of worker threads in the pool.
int threadpool_get_num_threads(const ThreadPool *pool);
