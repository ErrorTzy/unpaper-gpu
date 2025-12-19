// SPDX-FileCopyrightText: 2025 The unpaper authors
//
// SPDX-License-Identifier: GPL-2.0-only

#include "lib/batch_worker.h"
#include "lib/batch_decode_queue.h"
#include "lib/decoded_image_provider.h"
#include "lib/decode_queue.h"
#include "lib/encode_queue.h"
#include "lib/gpu_monitor.h"
#include "lib/logging.h"
#include "lib/threadpool.h"
#include "src/core/sheet_pipeline.h"

#include <libavutil/frame.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#ifdef UNPAPER_WITH_CUDA
#include "imageprocess/cuda_runtime.h"
#include "imageprocess/cuda_stream_pool.h"
#include "imageprocess/image.h"
#endif

void batch_worker_init(BatchWorkerContext *ctx, const Options *options,
                       BatchQueue *queue) {
  ctx->options = options;
  ctx->queue = queue;
  ctx->perf_enabled = options->perf;
  ctx->config = NULL;
  ctx->use_stream_pool = false;
  ctx->decode_queue = NULL;
  ctx->batch_decode_queue = NULL;
  ctx->encode_queue = NULL;
  ctx->post_process_fn = NULL;
  ctx->post_process_user_ctx = NULL;
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

void batch_worker_set_batch_decode_queue(BatchWorkerContext *ctx,
                                         BatchDecodeQueue *batch_decode_queue) {
  ctx->batch_decode_queue = batch_decode_queue;
}

void batch_worker_set_encode_queue(BatchWorkerContext *ctx,
                                   EncodeQueue *encode_queue) {
  ctx->encode_queue = encode_queue;
}

void batch_worker_set_post_process_callback(BatchWorkerContext *ctx,
                                            BatchWorkerPostProcessFn fn,
                                            void *user_ctx) {
  if (!ctx) {
    return;
  }
  ctx->post_process_fn = fn;
  ctx->post_process_user_ctx = user_ctx;
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

  // Set encode queue for async encoding if available
  if (ctx->encode_queue != NULL) {
    sheet_process_state_set_encode_queue(&state, ctx->encode_queue,
                                         (int)job_index);
  }

  // Get pre-decoded images from queue if available
  // Batch decode queue (PR36B) takes precedence over per-image decode queue
  DecodedImageProvider provider;
  DecodedImageHandle decoded_images[BATCH_MAX_FILES_PER_SHEET] = {0};

  decoded_image_provider_reset(&provider);
  if (ctx->batch_decode_queue != NULL) {
    decoded_image_provider_init_batch_decode_queue(&provider,
                                                   ctx->batch_decode_queue);
  } else if (ctx->decode_queue != NULL) {
    decoded_image_provider_init_decode_queue(&provider, ctx->decode_queue);
  }

  if (provider.get != NULL) {
    for (int i = 0; i < job->input_count; i++) {
      if (job->input_files[i] == NULL) {
        continue;
      }
      if (!decoded_image_provider_get(&provider, (int)job_index, i,
                                      &decoded_images[i])) {
        continue;
      }

#ifdef UNPAPER_WITH_CUDA
      if (decoded_images[i].view.on_gpu &&
          decoded_images[i].view.gpu_ptr != NULL) {
        Image gpu_image = create_image_from_gpu(
            decoded_images[i].view.gpu_ptr, decoded_images[i].view.gpu_pitch,
            decoded_images[i].view.gpu_width, decoded_images[i].view.gpu_height,
            decoded_images[i].view.gpu_format, ctx->options->sheet_background,
            ctx->options->abs_black_threshold,
            decoded_images[i].view.gpu_owns_memory);

        if (gpu_image.frame != NULL) {
          if (decoded_images[i].view.gpu_owns_memory) {
            decoded_image_handle_detach_gpu(&decoded_images[i]);
          }
          sheet_process_state_set_gpu_decoded_image(&state, gpu_image, i);
        }
      } else
#endif
          if (decoded_images[i].view.frame != NULL) {
        // CPU-decoded frame - clone and transfer
        AVFrame *frame_copy = av_frame_clone(decoded_images[i].view.frame);
        if (frame_copy) {
          sheet_process_state_set_decoded(&state, frame_copy, i);
        }
      }
    }
  }

  bool success = sheet_pipeline_run(&state, ctx->config);

  if (success && ctx->post_process_fn != NULL) {
    if (!ctx->post_process_fn(ctx, job_index, &state, ctx->post_process_user_ctx)) {
      success = false;
    }
  }

  // Release decoded images back to queue
  if (provider.release != NULL) {
    for (int i = 0; i < BATCH_MAX_FILES_PER_SHEET; i++) {
      if (decoded_images[i].image != NULL) {
        decoded_image_provider_release(&provider, &decoded_images[i]);
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
  void *ev_start = NULL;
  void *ev_stop = NULL;
  size_t gpu_job_id = 0;

  // Acquire a stream from the pool for this job
  if (ctx->use_stream_pool && cuda_stream_pool_global_active()) {
    stream = cuda_stream_pool_global_acquire();
    if (stream != NULL) {
      // Set this stream as current for all CUDA operations in this job
      unpaper_cuda_set_current_stream(stream);

      // Record GPU job start for occupancy monitoring
      if (gpu_monitor_global_active()) {
        gpu_job_id = gpu_monitor_global_job_start();

        // Start GPU timing events for this job
        unpaper_cuda_event_pair_start_on(stream, &ev_start, &ev_stop);
      }
    }
  }
#endif

  // Process the job
  bool success = batch_process_job(ctx, job_ctx->job_index);

  // Log error details if job failed
  if (!success) {
    BatchJob *job = batch_queue_get(ctx->queue, job_ctx->job_index);
    if (job) {
      fprintf(stderr, "ERROR: Job %zu (sheet %d) failed", job_ctx->job_index,
              job->sheet_nr);
      if (job->input_files[0]) {
        fprintf(stderr, " - input: %s", job->input_files[0]);
      }
      if (job->output_files[0]) {
        fprintf(stderr, " - output: %s", job->output_files[0]);
      }
      fprintf(stderr, "\n");
    }
  }

#ifdef UNPAPER_WITH_CUDA
  // Release the stream back to the pool
  if (stream != NULL) {
    double gpu_time_ms = 0.0;

    // Stop GPU timing and get elapsed time
    if (ev_start != NULL && ev_stop != NULL) {
      gpu_time_ms =
          unpaper_cuda_event_pair_stop_ms_on(stream, &ev_start, &ev_stop);
    }

    // Record GPU job end with timing
    if (gpu_monitor_global_active() && gpu_job_id > 0) {
      gpu_monitor_global_job_end(gpu_job_id, gpu_time_ms);
    }

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
