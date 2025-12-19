// SPDX-FileCopyrightText: 2025 The unpaper authors
//
// SPDX-License-Identifier: GPL-2.0-only

#include "lib/batch.h"
#include "lib/logging.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>

#define INITIAL_CAPACITY 64

void batch_queue_init(BatchQueue *queue) {
  queue->jobs = NULL;
  queue->count = 0;
  queue->capacity = 0;
  queue->completed = 0;
  queue->failed = 0;
  queue->parallelism = 0;
  queue->progress = false;
}

void batch_job_free(BatchJob *job) {
  for (int i = 0; i < job->input_count; i++) {
    free(job->input_files[i]);
    job->input_files[i] = NULL;
  }
  for (int i = 0; i < job->output_count; i++) {
    free(job->output_files[i]);
    job->output_files[i] = NULL;
  }
}

void batch_queue_free(BatchQueue *queue) {
  if (queue->jobs) {
    for (size_t i = 0; i < queue->count; i++) {
      batch_job_free(&queue->jobs[i]);
    }
    free(queue->jobs);
  }
  queue->jobs = NULL;
  queue->count = 0;
  queue->capacity = 0;
}

BatchJob *batch_queue_add(BatchQueue *queue) {
  // Grow if needed
  if (queue->count >= queue->capacity) {
    size_t new_capacity =
        queue->capacity == 0 ? INITIAL_CAPACITY : queue->capacity * 2;
    BatchJob *new_jobs = realloc(queue->jobs, new_capacity * sizeof(BatchJob));
    if (!new_jobs) {
      return NULL;
    }
    queue->jobs = new_jobs;
    queue->capacity = new_capacity;
  }

  BatchJob *job = &queue->jobs[queue->count++];
  memset(job, 0, sizeof(*job));
  job->status = BATCH_JOB_PENDING;
  job->layout_override = -1;
  return job;
}

int batch_detect_parallelism(void) {
  // Use sysconf for POSIX systems
  long nprocs = sysconf(_SC_NPROCESSORS_ONLN);
  if (nprocs < 1) {
    nprocs = 1;
  }
  // Cap at a reasonable maximum
  if (nprocs > 64) {
    nprocs = 64;
  }
  return (int)nprocs;
}

int batch_detect_cuda_parallelism(void) {
  // For CUDA batch processing with PNG/non-JPEG files:
  // - Decode uses FFmpeg (CPU-bound)
  // - Processing uses CUDA (GPU-bound)
  // - Encode uses FFmpeg (CPU-bound)
  //
  // Benchmarking shows CPU cores / 3 (rounded) is optimal because:
  // - Too many workers saturate CPU decode/encode threads
  // - GPU can handle the processing with 8 streams efficiently
  // - This balances CPU I/O with GPU compute
  long nprocs = sysconf(_SC_NPROCESSORS_ONLN);
  if (nprocs < 1) {
    nprocs = 1;
  }

  // CPU cores / 3, with minimum of 2 and maximum of 16
  int parallelism = (int)((nprocs + 1) / 3); // Round to nearest
  if (parallelism < 2) {
    parallelism = 2;
  }
  if (parallelism > 16) {
    parallelism = 16;
  }
  return parallelism;
}

void batch_progress_start(BatchQueue *queue) {
  if (!queue->progress)
    return;

  fprintf(stderr, "Batch processing: %zu jobs queued\n", queue->count);
  if (queue->parallelism > 0) {
    fprintf(stderr, "Parallelism: %d workers\n", queue->parallelism);
  }
}

void batch_progress_update(BatchQueue *queue, int job_index,
                           BatchJobStatus status) {
  if (job_index < 0 || (size_t)job_index >= queue->count)
    return;

  BatchJob *job = &queue->jobs[job_index];
  BatchJobStatus old_status = job->status;
  job->status = status;

  // Update counters
  if (status == BATCH_JOB_COMPLETED && old_status != BATCH_JOB_COMPLETED) {
    queue->completed++;
  } else if (status == BATCH_JOB_FAILED && old_status != BATCH_JOB_FAILED) {
    queue->failed++;
  }

  if (!queue->progress)
    return;

  // Print progress
  size_t done = queue->completed + queue->failed;
  const char *status_str = "";
  switch (status) {
  case BATCH_JOB_IN_PROGRESS:
    status_str = "processing";
    break;
  case BATCH_JOB_COMPLETED:
    status_str = "done";
    break;
  case BATCH_JOB_FAILED:
    status_str = "FAILED";
    break;
  default:
    return;
  }

  // Format: [42/100] processing sheet 42...
  fprintf(stderr, "[%zu/%zu] %s sheet %d", done, queue->count, status_str,
          job->sheet_nr);

  if (job->input_files[0]) {
    fprintf(stderr, " (%s)", job->input_files[0]);
  }
  fprintf(stderr, "\n");
}

void batch_progress_finish(BatchQueue *queue) {
  if (!queue->progress)
    return;

  fprintf(stderr, "\nBatch complete: %zu succeeded, %zu failed\n",
          queue->completed, queue->failed);
}
