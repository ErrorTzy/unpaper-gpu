// SPDX-FileCopyrightText: 2025 The unpaper authors
//
// SPDX-License-Identifier: GPL-2.0-only

#include "lib/perf.h"

#include <stdio.h>
#include <string.h>

#if defined(UNPAPER_WITH_CUDA) && (UNPAPER_WITH_CUDA)
#include "imageprocess/cuda_runtime.h"
#endif

static double timespec_diff_ms(struct timespec start, struct timespec end) {
  const double sec = (double)(end.tv_sec - start.tv_sec);
  const double nsec = (double)(end.tv_nsec - start.tv_nsec);
  return sec * 1000.0 + nsec / 1e6;
}

static void stage_timer_reset(PerfStageTimer *t) {
  t->wall_running = false;
  t->wall_ms = 0.0;
#if defined(UNPAPER_WITH_CUDA) && (UNPAPER_WITH_CUDA)
  t->cuda_running = false;
  t->cuda_ms = 0.0;
  t->cuda_start = NULL;
  t->cuda_stop = NULL;
#endif
}

void perf_recorder_init(PerfRecorder *recorder, bool enabled, bool use_cuda) {
  if (recorder == NULL) {
    return;
  }

  recorder->enabled = enabled;
  recorder->use_cuda = use_cuda;
#if defined(UNPAPER_WITH_CUDA) && (UNPAPER_WITH_CUDA)
  recorder->cuda_events_available =
      use_cuda && unpaper_cuda_events_supported();
#else
  (void)use_cuda;
  recorder->cuda_events_available = false;
#endif

  for (int i = 0; i < PERF_STAGE_COUNT; i++) {
    stage_timer_reset(&recorder->timers[i]);
  }
}

void perf_stage_begin(PerfRecorder *recorder, PerfStage stage) {
  if (recorder == NULL || !recorder->enabled) {
    return;
  }
  PerfStageTimer *t = &recorder->timers[stage];

  if (!t->wall_running) {
    clock_gettime(CLOCK_MONOTONIC, &t->wall_start);
    t->wall_running = true;
  }

#if defined(UNPAPER_WITH_CUDA) && (UNPAPER_WITH_CUDA)
  if (recorder->cuda_events_available && !t->cuda_running) {
    if (unpaper_cuda_event_pair_start(&t->cuda_start, &t->cuda_stop)) {
      t->cuda_running = true;
    }
  }
#endif
}

void perf_stage_end(PerfRecorder *recorder, PerfStage stage) {
  if (recorder == NULL || !recorder->enabled) {
    return;
  }
  PerfStageTimer *t = &recorder->timers[stage];

  if (t->wall_running) {
    struct timespec end;
    clock_gettime(CLOCK_MONOTONIC, &end);
    t->wall_ms += timespec_diff_ms(t->wall_start, end);
    t->wall_running = false;
  }

#if defined(UNPAPER_WITH_CUDA) && (UNPAPER_WITH_CUDA)
  if (recorder->cuda_events_available && t->cuda_running) {
    t->cuda_ms += unpaper_cuda_event_pair_stop_ms(&t->cuda_start, &t->cuda_stop);
    t->cuda_running = false;
  }
#endif
}

const char *perf_stage_name(PerfStage stage) {
  switch (stage) {
  case PERF_STAGE_DECODE:
    return "decode";
  case PERF_STAGE_UPLOAD:
    return "upload";
  case PERF_STAGE_FILTERS:
    return "filters";
  case PERF_STAGE_MASKS:
    return "masks/borders";
  case PERF_STAGE_DESKEW:
    return "deskew";
  case PERF_STAGE_DOWNLOAD:
    return "download";
  case PERF_STAGE_ENCODE:
    return "encode";
  case PERF_STAGE_COUNT:
    break;
  }
  return "unknown";
}

void perf_recorder_print(const PerfRecorder *recorder, int sheet_nr,
                         const char *device_name) {
  if (recorder == NULL || !recorder->enabled) {
    return;
  }

  printf("perf sheet %d [%s]:", sheet_nr,
         device_name == NULL ? "unknown" : device_name);
  for (int i = 0; i < PERF_STAGE_COUNT; i++) {
    const PerfStageTimer *t = &recorder->timers[i];
    printf(" %s=%.2fms", perf_stage_name((PerfStage)i), t->wall_ms);
#if defined(UNPAPER_WITH_CUDA) && (UNPAPER_WITH_CUDA)
    if (recorder->cuda_events_available) {
      printf(" (gpu=%.2fms)", t->cuda_ms);
    }
#endif
    if (i + 1 != PERF_STAGE_COUNT) {
      printf(",");
    }
  }
  printf("\n");
}

// Batch-level performance tracking

void batch_perf_init(BatchPerfRecorder *recorder, bool enabled) {
  if (recorder == NULL) {
    return;
  }
  recorder->enabled = enabled;
  recorder->total_jobs = 0;
  recorder->completed_jobs = 0;
  recorder->failed_jobs = 0;
  recorder->start.tv_sec = 0;
  recorder->start.tv_nsec = 0;
  recorder->end.tv_sec = 0;
  recorder->end.tv_nsec = 0;
}

void batch_perf_start(BatchPerfRecorder *recorder) {
  if (recorder == NULL || !recorder->enabled) {
    return;
  }
  clock_gettime(CLOCK_MONOTONIC, &recorder->start);
}

void batch_perf_end(BatchPerfRecorder *recorder, size_t completed,
                    size_t failed) {
  if (recorder == NULL || !recorder->enabled) {
    return;
  }
  clock_gettime(CLOCK_MONOTONIC, &recorder->end);
  recorder->completed_jobs = completed;
  recorder->failed_jobs = failed;
  recorder->total_jobs = completed + failed;
}

void batch_perf_print(const BatchPerfRecorder *recorder,
                      const char *device_name) {
  if (recorder == NULL || !recorder->enabled) {
    return;
  }

  double elapsed_ms = timespec_diff_ms(recorder->start, recorder->end);
  double elapsed_sec = elapsed_ms / 1000.0;
  double images_per_sec =
      elapsed_sec > 0.0 ? (double)recorder->total_jobs / elapsed_sec : 0.0;
  double ms_per_image =
      recorder->total_jobs > 0 ? elapsed_ms / (double)recorder->total_jobs : 0.0;

  printf("\n");
  printf("Batch Performance Summary [%s]:\n",
         device_name == NULL ? "unknown" : device_name);
  printf("  Total time:     %.2f ms (%.2f sec)\n", elapsed_ms, elapsed_sec);
  printf("  Images:         %zu total, %zu completed, %zu failed\n",
         recorder->total_jobs, recorder->completed_jobs, recorder->failed_jobs);
  printf("  Throughput:     %.2f images/sec\n", images_per_sec);
  printf("  Avg per image:  %.2f ms\n", ms_per_image);
}

