// SPDX-FileCopyrightText: 2025 The unpaper authors
//
// SPDX-License-Identifier: GPL-2.0-only

#include "lib/gpu_monitor.h"

#if defined(UNPAPER_WITH_CUDA) && (UNPAPER_WITH_CUDA)

#include <cuda_runtime.h>
#include <pthread.h>
#include <stdatomic.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "imageprocess/cuda_runtime.h"

// Maximum concurrent jobs to track (should match stream pool size)
#define MAX_CONCURRENT_JOBS 64

struct GpuMonitor {
  // Active job tracking
  atomic_size_t active_jobs;   // Currently running GPU jobs
  pthread_mutex_t job_mutex;

  // Job ID generation
  atomic_size_t next_job_id;

  // Statistics (protected by stats_mutex for compound updates)
  pthread_mutex_t stats_mutex;
  size_t total_jobs;
  size_t peak_concurrent;
  size_t concurrent_samples;
  size_t concurrent_sum;  // Sum for average calculation

  // Timing statistics
  double total_gpu_time_ms;
  double min_gpu_time_ms;
  double max_gpu_time_ms;

  // Memory tracking
  size_t peak_memory_used;
  size_t initial_memory_free;
  size_t final_memory_free;
  bool batch_started;
};

// Global singleton monitor
static GpuMonitor *global_gpu_monitor = NULL;
static pthread_mutex_t global_monitor_mutex = PTHREAD_MUTEX_INITIALIZER;

bool gpu_monitor_get_memory_info(GpuMemoryInfo *info) {
  if (info == NULL) {
    return false;
  }

  memset(info, 0, sizeof(*info));

  // Try to get GPU memory info
  size_t free_bytes = 0;
  size_t total_bytes = 0;
  cudaError_t err = cudaMemGetInfo(&free_bytes, &total_bytes);
  if (err != cudaSuccess) {
    return false;
  }

  info->free_bytes = free_bytes;
  info->total_bytes = total_bytes;
  info->used_bytes = total_bytes - free_bytes;
  if (total_bytes > 0) {
    info->usage_percent = 100.0 * (double)info->used_bytes / (double)total_bytes;
  }

  return true;
}

GpuMonitor *gpu_monitor_create(void) {
  // Ensure CUDA is initialized
  UnpaperCudaInitStatus init_status = unpaper_cuda_try_init();
  if (init_status != UNPAPER_CUDA_INIT_OK) {
    return NULL;
  }

  GpuMonitor *monitor = calloc(1, sizeof(GpuMonitor));
  if (monitor == NULL) {
    return NULL;
  }

  atomic_init(&monitor->active_jobs, 0);
  atomic_init(&monitor->next_job_id, 1);

  pthread_mutex_init(&monitor->job_mutex, NULL);
  pthread_mutex_init(&monitor->stats_mutex, NULL);

  monitor->total_jobs = 0;
  monitor->peak_concurrent = 0;
  monitor->concurrent_samples = 0;
  monitor->concurrent_sum = 0;

  monitor->total_gpu_time_ms = 0.0;
  monitor->min_gpu_time_ms = 1e9;  // Very large initial value
  monitor->max_gpu_time_ms = 0.0;

  monitor->peak_memory_used = 0;
  monitor->initial_memory_free = 0;
  monitor->final_memory_free = 0;
  monitor->batch_started = false;

  return monitor;
}

void gpu_monitor_destroy(GpuMonitor *monitor) {
  if (monitor == NULL) {
    return;
  }

  pthread_mutex_destroy(&monitor->job_mutex);
  pthread_mutex_destroy(&monitor->stats_mutex);
  free(monitor);
}

size_t gpu_monitor_job_start(GpuMonitor *monitor) {
  if (monitor == NULL) {
    return 0;
  }

  size_t job_id = atomic_fetch_add(&monitor->next_job_id, 1);

  // Increment active jobs
  size_t active = atomic_fetch_add(&monitor->active_jobs, 1) + 1;

  // Update peak concurrent (lock-free update)
  pthread_mutex_lock(&monitor->stats_mutex);
  if (active > monitor->peak_concurrent) {
    monitor->peak_concurrent = active;
  }

  // Sample concurrency at job start
  monitor->concurrent_sum += active;
  monitor->concurrent_samples++;
  pthread_mutex_unlock(&monitor->stats_mutex);

  return job_id;
}

void gpu_monitor_job_end(GpuMonitor *monitor, size_t job_id,
                         double gpu_time_ms) {
  if (monitor == NULL) {
    return;
  }
  (void)job_id;  // Job ID for potential per-job tracking in future

  // Decrement active jobs
  atomic_fetch_sub(&monitor->active_jobs, 1);

  // Update statistics
  pthread_mutex_lock(&monitor->stats_mutex);
  monitor->total_jobs++;
  monitor->total_gpu_time_ms += gpu_time_ms;

  if (gpu_time_ms > 0) {
    if (gpu_time_ms < monitor->min_gpu_time_ms) {
      monitor->min_gpu_time_ms = gpu_time_ms;
    }
    if (gpu_time_ms > monitor->max_gpu_time_ms) {
      monitor->max_gpu_time_ms = gpu_time_ms;
    }
  }
  pthread_mutex_unlock(&monitor->stats_mutex);
}

void gpu_monitor_sample(GpuMonitor *monitor) {
  if (monitor == NULL) {
    return;
  }

  // Sample current concurrency
  size_t active = atomic_load(&monitor->active_jobs);

  pthread_mutex_lock(&monitor->stats_mutex);
  monitor->concurrent_sum += active;
  monitor->concurrent_samples++;

  // Sample memory usage
  GpuMemoryInfo mem_info;
  if (gpu_monitor_get_memory_info(&mem_info)) {
    if (mem_info.used_bytes > monitor->peak_memory_used) {
      monitor->peak_memory_used = mem_info.used_bytes;
    }
  }
  pthread_mutex_unlock(&monitor->stats_mutex);
}

void gpu_monitor_batch_start(GpuMonitor *monitor) {
  if (monitor == NULL) {
    return;
  }

  GpuMemoryInfo mem_info;
  if (gpu_monitor_get_memory_info(&mem_info)) {
    monitor->initial_memory_free = mem_info.free_bytes;
    monitor->peak_memory_used = mem_info.used_bytes;
  }
  monitor->batch_started = true;
}

void gpu_monitor_batch_end(GpuMonitor *monitor) {
  if (monitor == NULL) {
    return;
  }

  GpuMemoryInfo mem_info;
  if (gpu_monitor_get_memory_info(&mem_info)) {
    monitor->final_memory_free = mem_info.free_bytes;
  }
}

GpuOccupancyStats gpu_monitor_get_stats(const GpuMonitor *monitor) {
  GpuOccupancyStats stats = {0};
  if (monitor == NULL) {
    return stats;
  }

  stats.total_gpu_jobs = monitor->total_jobs;
  stats.peak_concurrent_jobs = monitor->peak_concurrent;
  stats.concurrent_samples = monitor->concurrent_samples;

  if (monitor->concurrent_samples > 0) {
    stats.avg_concurrent_jobs =
        (double)monitor->concurrent_sum / (double)monitor->concurrent_samples;
  }

  stats.total_gpu_time_ms = monitor->total_gpu_time_ms;
  if (monitor->total_jobs > 0) {
    stats.avg_gpu_time_ms = monitor->total_gpu_time_ms / (double)monitor->total_jobs;
  }
  stats.min_gpu_time_ms = (monitor->total_jobs > 0) ? monitor->min_gpu_time_ms : 0.0;
  stats.max_gpu_time_ms = monitor->max_gpu_time_ms;

  stats.peak_memory_used = monitor->peak_memory_used;
  stats.initial_memory_free = monitor->initial_memory_free;
  stats.final_memory_free = monitor->final_memory_free;

  return stats;
}

void gpu_monitor_print_stats(const GpuMonitor *monitor) {
  if (monitor == NULL) {
    return;
  }

  GpuOccupancyStats stats = gpu_monitor_get_stats(monitor);

  fprintf(stderr,
          "GPU Occupancy Statistics:\n"
          "  Total GPU jobs: %zu\n"
          "  Peak concurrent jobs: %zu\n"
          "  Average concurrent jobs: %.2f\n"
          "  Concurrency samples: %zu\n",
          stats.total_gpu_jobs, stats.peak_concurrent_jobs,
          stats.avg_concurrent_jobs, stats.concurrent_samples);

  if (stats.total_gpu_jobs > 0) {
    fprintf(stderr,
            "  GPU time (total): %.2f ms\n"
            "  GPU time (avg): %.2f ms\n"
            "  GPU time (min): %.2f ms\n"
            "  GPU time (max): %.2f ms\n",
            stats.total_gpu_time_ms, stats.avg_gpu_time_ms,
            stats.min_gpu_time_ms, stats.max_gpu_time_ms);
  }

  // Memory statistics
  if (stats.peak_memory_used > 0) {
    fprintf(stderr,
            "  Peak GPU memory used: %.1f MB\n",
            (double)stats.peak_memory_used / (1024.0 * 1024.0));
  }
  if (stats.initial_memory_free > 0) {
    fprintf(stderr,
            "  Initial GPU memory free: %.1f MB\n"
            "  Final GPU memory free: %.1f MB\n",
            (double)stats.initial_memory_free / (1024.0 * 1024.0),
            (double)stats.final_memory_free / (1024.0 * 1024.0));
  }
}

// Global monitor implementation

bool gpu_monitor_global_init(void) {
  pthread_mutex_lock(&global_monitor_mutex);

  if (global_gpu_monitor != NULL) {
    pthread_mutex_unlock(&global_monitor_mutex);
    return true;
  }

  global_gpu_monitor = gpu_monitor_create();

  pthread_mutex_unlock(&global_monitor_mutex);
  return global_gpu_monitor != NULL;
}

void gpu_monitor_global_cleanup(void) {
  pthread_mutex_lock(&global_monitor_mutex);

  if (global_gpu_monitor != NULL) {
    gpu_monitor_destroy(global_gpu_monitor);
    global_gpu_monitor = NULL;
  }

  pthread_mutex_unlock(&global_monitor_mutex);
}

bool gpu_monitor_global_active(void) {
  pthread_mutex_lock(&global_monitor_mutex);
  bool active = (global_gpu_monitor != NULL);
  pthread_mutex_unlock(&global_monitor_mutex);
  return active;
}

size_t gpu_monitor_global_job_start(void) {
  GpuMonitor *monitor = global_gpu_monitor;
  if (monitor != NULL) {
    return gpu_monitor_job_start(monitor);
  }
  return 0;
}

void gpu_monitor_global_job_end(size_t job_id, double gpu_time_ms) {
  GpuMonitor *monitor = global_gpu_monitor;
  if (monitor != NULL) {
    gpu_monitor_job_end(monitor, job_id, gpu_time_ms);
  }
}

void gpu_monitor_global_sample(void) {
  GpuMonitor *monitor = global_gpu_monitor;
  if (monitor != NULL) {
    gpu_monitor_sample(monitor);
  }
}

void gpu_monitor_global_batch_start(void) {
  GpuMonitor *monitor = global_gpu_monitor;
  if (monitor != NULL) {
    gpu_monitor_batch_start(monitor);
  }
}

void gpu_monitor_global_batch_end(void) {
  GpuMonitor *monitor = global_gpu_monitor;
  if (monitor != NULL) {
    gpu_monitor_batch_end(monitor);
  }
}

GpuOccupancyStats gpu_monitor_global_get_stats(void) {
  GpuOccupancyStats stats = {0};
  pthread_mutex_lock(&global_monitor_mutex);
  if (global_gpu_monitor != NULL) {
    stats = gpu_monitor_get_stats(global_gpu_monitor);
  }
  pthread_mutex_unlock(&global_monitor_mutex);
  return stats;
}

void gpu_monitor_global_print_stats(void) {
  pthread_mutex_lock(&global_monitor_mutex);
  if (global_gpu_monitor != NULL) {
    gpu_monitor_print_stats(global_gpu_monitor);
  }
  pthread_mutex_unlock(&global_monitor_mutex);
}

#else // !UNPAPER_WITH_CUDA

// Stub implementations for non-CUDA builds

bool gpu_monitor_get_memory_info(GpuMemoryInfo *info) {
  (void)info;
  return false;
}

GpuMonitor *gpu_monitor_create(void) { return NULL; }
void gpu_monitor_destroy(GpuMonitor *monitor) { (void)monitor; }

size_t gpu_monitor_job_start(GpuMonitor *monitor) {
  (void)monitor;
  return 0;
}

void gpu_monitor_job_end(GpuMonitor *monitor, size_t job_id,
                         double gpu_time_ms) {
  (void)monitor;
  (void)job_id;
  (void)gpu_time_ms;
}

void gpu_monitor_sample(GpuMonitor *monitor) { (void)monitor; }
void gpu_monitor_batch_start(GpuMonitor *monitor) { (void)monitor; }
void gpu_monitor_batch_end(GpuMonitor *monitor) { (void)monitor; }

GpuOccupancyStats gpu_monitor_get_stats(const GpuMonitor *monitor) {
  (void)monitor;
  GpuOccupancyStats stats = {0};
  return stats;
}

void gpu_monitor_print_stats(const GpuMonitor *monitor) { (void)monitor; }

bool gpu_monitor_global_init(void) { return false; }
void gpu_monitor_global_cleanup(void) {}
bool gpu_monitor_global_active(void) { return false; }
size_t gpu_monitor_global_job_start(void) { return 0; }
void gpu_monitor_global_job_end(size_t job_id, double gpu_time_ms) {
  (void)job_id;
  (void)gpu_time_ms;
}
void gpu_monitor_global_sample(void) {}
void gpu_monitor_global_batch_start(void) {}
void gpu_monitor_global_batch_end(void) {}

GpuOccupancyStats gpu_monitor_global_get_stats(void) {
  GpuOccupancyStats stats = {0};
  return stats;
}

void gpu_monitor_global_print_stats(void) {}

#endif // UNPAPER_WITH_CUDA
