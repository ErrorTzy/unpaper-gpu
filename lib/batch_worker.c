// SPDX-FileCopyrightText: 2025 The unpaper authors
//
// SPDX-License-Identifier: GPL-2.0-only

#include "lib/batch_worker.h"
#include "lib/threadpool.h"
#include "sheet_process.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

void batch_worker_init(BatchWorkerContext *ctx, const Options *options,
                       BatchQueue *queue) {
  ctx->options = options;
  ctx->queue = queue;
  ctx->perf_enabled = options->perf;
  ctx->config = NULL;
  pthread_mutex_init(&ctx->progress_mutex, NULL);
}

void batch_worker_cleanup(BatchWorkerContext *ctx) {
  pthread_mutex_destroy(&ctx->progress_mutex);
}

void batch_worker_set_config(BatchWorkerContext *ctx,
                             const SheetProcessConfig *config) {
  ctx->config = config;
}

bool batch_process_job(BatchWorkerContext *ctx, size_t job_index) {
  if (!ctx->config) {
    fprintf(stderr, "Batch worker config not set\n");
    return false;
  }

  BatchJob *job = batch_queue_get(ctx->queue, job_index);
  if (!job) {
    return false;
  }

  SheetProcessState state;
  sheet_process_state_init(&state, ctx->config, job);

  bool success = process_sheet(&state, ctx->config);

  sheet_process_state_cleanup(&state);

  return success;
}

// Worker function called by thread pool
static void batch_worker_fn(void *arg, int thread_id) {
  BatchJobContext *job_ctx = (BatchJobContext *)arg;
  BatchWorkerContext *ctx = job_ctx->ctx;

  // Process the job
  bool success = batch_process_job(ctx, job_ctx->job_index);

  // Update progress with thread safety
  pthread_mutex_lock(&ctx->progress_mutex);
  batch_progress_update(ctx->queue, (int)job_ctx->job_index,
                        success ? BATCH_JOB_COMPLETED : BATCH_JOB_FAILED);
  pthread_mutex_unlock(&ctx->progress_mutex);

  // Free the job context
  free(job_ctx);

  (void)thread_id; // May be used for thread-local state in future
}

int batch_process_parallel(BatchWorkerContext *ctx, ThreadPool *pool) {
  size_t job_count = batch_queue_count(ctx->queue);

  // Submit all jobs to the thread pool
  for (size_t i = 0; i < job_count; i++) {
    BatchJobContext *job_ctx = malloc(sizeof(BatchJobContext));
    if (!job_ctx) {
      fprintf(stderr, "Failed to allocate job context\n");
      continue;
    }
    job_ctx->ctx = ctx;
    job_ctx->job_index = i;

    if (!threadpool_submit(pool, batch_worker_fn, job_ctx)) {
      fprintf(stderr, "Failed to submit job %zu to thread pool\n", i);
      free(job_ctx);
    }
  }

  // Wait for all jobs to complete
  threadpool_wait(pool);

  return (int)ctx->queue->failed;
}
