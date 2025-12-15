// SPDX-FileCopyrightText: 2025 The unpaper authors
//
// SPDX-License-Identifier: GPL-2.0-only

#include "lib/batch_worker.h"
#include "lib/decode_queue.h"
#include "lib/threadpool.h"
#include "sheet_process.h"

#include <libavutil/frame.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#ifdef UNPAPER_WITH_CUDA
#include "imageprocess/cuda_runtime.h"
#include "imageprocess/cuda_stream_pool.h"
#endif

void batch_worker_init(BatchWorkerContext *ctx, const Options *options,
                       BatchQueue *queue) {
  ctx->options = options;
  ctx->queue = queue;
  ctx->perf_enabled = options->perf;
  ctx->config = NULL;
  ctx->use_stream_pool = false;
  ctx->decode_queue = NULL;
  pthread_mutex_init(&ctx->progress_mutex, NULL);
}

void batch_worker_cleanup(BatchWorkerContext *ctx) {
  pthread_mutex_destroy(&ctx->progress_mutex);
}

void batch_worker_set_config(BatchWorkerContext *ctx,
                             const SheetProcessConfig *config) {
  ctx->config = config;
}

void batch_worker_enable_stream_pool(BatchWorkerContext *ctx, bool enable) {
  ctx->use_stream_pool = enable;
}

void batch_worker_set_decode_queue(BatchWorkerContext *ctx,
                                   DecodeQueue *decode_queue) {
  ctx->decode_queue = decode_queue;
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

  // Get pre-decoded images from queue if available
  DecodedImage *decoded_images[BATCH_MAX_FILES_PER_SHEET] = {NULL};
  if (ctx->decode_queue != NULL) {
    for (int i = 0; i < job->input_count; i++) {
      if (job->input_files[i] != NULL) {
        DecodedImage *decoded = decode_queue_get(ctx->decode_queue,
                                                 (int)job_index, i);
        if (decoded != NULL && decoded->valid && decoded->frame != NULL) {
          decoded_images[i] = decoded;
          // Transfer frame to state (clone to avoid double-free)
          AVFrame *frame_copy = av_frame_clone(decoded->frame);
          if (frame_copy) {
            sheet_process_state_set_decoded(&state, frame_copy, i);
          }
        }
      }
    }
  }

  bool success = process_sheet(&state, ctx->config);

  // Release decoded images back to queue
  if (ctx->decode_queue != NULL) {
    for (int i = 0; i < BATCH_MAX_FILES_PER_SHEET; i++) {
      if (decoded_images[i] != NULL) {
        decode_queue_release(ctx->decode_queue, decoded_images[i]);
      }
    }
  }

  sheet_process_state_cleanup(&state);

  return success;
}

// Worker function called by thread pool
static void batch_worker_fn(void *arg, int thread_id) {
  BatchJobContext *job_ctx = (BatchJobContext *)arg;
  BatchWorkerContext *ctx = job_ctx->ctx;

#ifdef UNPAPER_WITH_CUDA
  UnpaperCudaStream *stream = NULL;

  // Acquire a stream from the pool for this job
  if (ctx->use_stream_pool && cuda_stream_pool_global_active()) {
    stream = cuda_stream_pool_global_acquire();
    if (stream != NULL) {
      // Set this stream as current for all CUDA operations in this job
      unpaper_cuda_set_current_stream(stream);
    }
  }
#endif

  // Process the job
  bool success = batch_process_job(ctx, job_ctx->job_index);

#ifdef UNPAPER_WITH_CUDA
  // Release the stream back to the pool
  if (stream != NULL) {
    // Synchronize and release - the pool will sync internally
    cuda_stream_pool_global_release(stream);
    // Reset to default stream
    unpaper_cuda_set_current_stream(NULL);
  }
#endif

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
