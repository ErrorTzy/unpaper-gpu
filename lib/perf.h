// SPDX-FileCopyrightText: 2025 The unpaper authors
//
// SPDX-License-Identifier: GPL-2.0-only

#pragma once

#include <stdbool.h>
#include <time.h>

typedef enum {
  PERF_STAGE_DECODE = 0,
  PERF_STAGE_UPLOAD,
  PERF_STAGE_FILTERS,
  PERF_STAGE_MASKS,
  PERF_STAGE_DESKEW,
  PERF_STAGE_DOWNLOAD,
  PERF_STAGE_ENCODE,
  PERF_STAGE_COUNT,
} PerfStage;

typedef struct {
  struct timespec wall_start;
  bool wall_running;
  double wall_ms;

#if defined(UNPAPER_WITH_CUDA) && (UNPAPER_WITH_CUDA)
  void *cuda_start;
  void *cuda_stop;
  bool cuda_running;
  double cuda_ms;
#endif
} PerfStageTimer;

typedef struct {
  bool enabled;
  bool use_cuda;
  bool cuda_events_available;
  PerfStageTimer timers[PERF_STAGE_COUNT];
} PerfRecorder;

void perf_recorder_init(PerfRecorder *recorder, bool enabled, bool use_cuda);
void perf_stage_begin(PerfRecorder *recorder, PerfStage stage);
void perf_stage_end(PerfRecorder *recorder, PerfStage stage);
void perf_recorder_print(const PerfRecorder *recorder, int sheet_nr,
                         const char *device_name);
const char *perf_stage_name(PerfStage stage);

// Batch-level timing for overall performance metrics
typedef struct {
  bool enabled;
  struct timespec start;
  struct timespec end;
  size_t total_jobs;
  size_t completed_jobs;
  size_t failed_jobs;
} BatchPerfRecorder;

void batch_perf_init(BatchPerfRecorder *recorder, bool enabled);
void batch_perf_start(BatchPerfRecorder *recorder);
void batch_perf_end(BatchPerfRecorder *recorder, size_t completed, size_t failed);
void batch_perf_print(const BatchPerfRecorder *recorder, const char *device_name);

