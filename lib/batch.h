// SPDX-FileCopyrightText: 2025 The unpaper authors
//
// SPDX-License-Identifier: GPL-2.0-only

#pragma once

#include <limits.h>
#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>

// Maximum input/output files per sheet (matches existing code)
#define BATCH_MAX_FILES_PER_SHEET 2

// Job status
typedef enum {
  BATCH_JOB_PENDING = 0,
  BATCH_JOB_IN_PROGRESS,
  BATCH_JOB_COMPLETED,
  BATCH_JOB_FAILED,
} BatchJobStatus;

// A single batch job representing one sheet to process
typedef struct {
  int sheet_nr;  // Sheet number (1-based)
  int input_nr;  // Input file counter start
  int output_nr; // Output file counter start

  // Input file paths (NULL for blank pages)
  char *input_files[BATCH_MAX_FILES_PER_SHEET];
  int input_count;

  // Output file paths
  char *output_files[BATCH_MAX_FILES_PER_SHEET];
  int output_count;
  int output_page_base; // Base output index for non-file pipelines (e.g., PDF)
  int layout_override;  // -1 = use options->layout, otherwise Layout enum value

  BatchJobStatus status;
} BatchJob;

// Job queue for batch processing
typedef struct {
  BatchJob *jobs;
  size_t count;
  size_t capacity;

  // Progress tracking
  size_t completed;
  size_t failed;

  // Configuration
  int parallelism; // Number of parallel workers (0 = auto)
  bool progress;   // Show progress output
} BatchQueue;

// Initialize a batch queue
void batch_queue_init(BatchQueue *queue);

// Free batch queue resources
void batch_queue_free(BatchQueue *queue);

// Add a job to the queue
// Returns pointer to the new job, or NULL on error
BatchJob *batch_queue_add(BatchQueue *queue);

// Get total job count
static inline size_t batch_queue_count(const BatchQueue *queue) {
  return queue->count;
}

// Get a job by index
static inline BatchJob *batch_queue_get(BatchQueue *queue, size_t index) {
  if (index >= queue->count)
    return NULL;
  return &queue->jobs[index];
}

// Progress reporting
void batch_progress_start(BatchQueue *queue);
void batch_progress_update(BatchQueue *queue, int job_index,
                           BatchJobStatus status);
void batch_progress_finish(BatchQueue *queue);

// Auto-detect optimal parallelism based on CPU cores
int batch_detect_parallelism(void);

// Auto-detect optimal parallelism for CUDA batch processing (PNG/non-JPEG)
// Returns CPU cores / 3 (rounded), which balances GPU compute with CPU
// decode/encode
int batch_detect_cuda_parallelism(void);

// Free a single job's resources (strings)
void batch_job_free(BatchJob *job);
