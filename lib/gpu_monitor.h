// SPDX-FileCopyrightText: 2025 The unpaper authors
//
// SPDX-License-Identifier: GPL-2.0-only

#pragma once

#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

// GPU occupancy and utilization monitoring for batch processing.
// Tracks concurrent GPU operations to verify concurrent kernel execution.

typedef struct GpuMonitor GpuMonitor;

// GPU memory information
typedef struct {
  size_t free_bytes;    // Currently free GPU memory
  size_t total_bytes;   // Total GPU memory
  size_t used_bytes;    // Currently used GPU memory
  double usage_percent; // Usage as percentage (0-100)
} GpuMemoryInfo;

// GPU occupancy statistics
typedef struct {
  // Concurrent execution tracking
  size_t total_gpu_jobs;       // Total GPU jobs processed
  size_t peak_concurrent_jobs; // Maximum concurrent GPU jobs observed
  size_t concurrent_samples;   // Number of times concurrency was sampled
  double avg_concurrent_jobs;  // Average concurrent GPU jobs

  // Timing statistics
  double total_gpu_time_ms; // Total GPU processing time
  double avg_gpu_time_ms;   // Average per-job GPU time
  double min_gpu_time_ms;   // Minimum GPU job time
  double max_gpu_time_ms;   // Maximum GPU job time

  // Memory statistics
  size_t peak_memory_used;    // Peak GPU memory usage observed
  size_t initial_memory_free; // Memory free at start of batch
  size_t final_memory_free;   // Memory free at end of batch
} GpuOccupancyStats;

// Create GPU monitor for tracking batch GPU operations.
// Returns NULL on failure or if CUDA is not available.
GpuMonitor *gpu_monitor_create(void);

// Destroy GPU monitor.
void gpu_monitor_destroy(GpuMonitor *monitor);

// Record the start of a GPU job.
// Returns a job ID for use with gpu_monitor_job_end.
size_t gpu_monitor_job_start(GpuMonitor *monitor);

// Record the end of a GPU job.
// job_id: The ID returned by gpu_monitor_job_start.
// gpu_time_ms: The GPU execution time for this job (from CUDA events).
void gpu_monitor_job_end(GpuMonitor *monitor, size_t job_id,
                         double gpu_time_ms);

// Get current GPU memory information.
// Returns true on success, false if CUDA is unavailable.
bool gpu_monitor_get_memory_info(GpuMemoryInfo *info);

// Sample current GPU state (call periodically during batch processing).
// This updates concurrent job counts and memory statistics.
void gpu_monitor_sample(GpuMonitor *monitor);

// Mark the start of batch processing (records initial memory state).
void gpu_monitor_batch_start(GpuMonitor *monitor);

// Mark the end of batch processing (records final memory state).
void gpu_monitor_batch_end(GpuMonitor *monitor);

// Get current occupancy statistics.
GpuOccupancyStats gpu_monitor_get_stats(const GpuMonitor *monitor);

// Print GPU occupancy statistics to stderr (for --perf output).
void gpu_monitor_print_stats(const GpuMonitor *monitor);

// Global GPU monitor management (singleton for batch mode).

// Initialize global GPU monitor.
bool gpu_monitor_global_init(void);

// Destroy global GPU monitor.
void gpu_monitor_global_cleanup(void);

// Check if global GPU monitor is active.
bool gpu_monitor_global_active(void);

// Record job start on global monitor.
size_t gpu_monitor_global_job_start(void);

// Record job end on global monitor.
void gpu_monitor_global_job_end(size_t job_id, double gpu_time_ms);

// Sample global monitor state.
void gpu_monitor_global_sample(void);

// Mark batch start on global monitor.
void gpu_monitor_global_batch_start(void);

// Mark batch end on global monitor.
void gpu_monitor_global_batch_end(void);

// Get global monitor statistics.
GpuOccupancyStats gpu_monitor_global_get_stats(void);

// Print global monitor statistics.
void gpu_monitor_global_print_stats(void);

#ifdef __cplusplus
}
#endif
