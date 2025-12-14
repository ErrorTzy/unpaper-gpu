// SPDX-FileCopyrightText: 2025 The unpaper authors
//
// SPDX-License-Identifier: GPL-2.0-only

#pragma once

#include "lib/batch.h"
#include "lib/options.h"
#include "sheet_process.h"

#include <pthread.h>
#include <stdbool.h>

// Forward declarations
struct ThreadPool;

// Batch worker context - shared state for all workers
typedef struct {
  const Options *options;
  BatchQueue *queue;
  const SheetProcessConfig *config; // Sheet processing configuration
  pthread_mutex_t progress_mutex;   // Protects progress updates
  bool perf_enabled;                // Enable per-job performance output
} BatchWorkerContext;

// Per-job context passed to worker function
typedef struct {
  BatchWorkerContext *ctx;
  size_t job_index;
} BatchJobContext;

// Initialize batch worker context
void batch_worker_init(BatchWorkerContext *ctx, const Options *options,
                       BatchQueue *queue);

// Cleanup batch worker context
void batch_worker_cleanup(BatchWorkerContext *ctx);

// Set the sheet processing configuration
void batch_worker_set_config(BatchWorkerContext *ctx,
                             const SheetProcessConfig *config);

// Process all jobs in the queue using the thread pool
// Returns number of failed jobs (0 = all succeeded)
int batch_process_parallel(BatchWorkerContext *ctx, struct ThreadPool *pool);

// Process a single job (for sequential mode or direct calls)
// Returns true on success, false on failure
bool batch_process_job(BatchWorkerContext *ctx, size_t job_index);
