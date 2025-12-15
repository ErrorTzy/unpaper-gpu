// SPDX-FileCopyrightText: 2025 The unpaper authors
//
// SPDX-License-Identifier: GPL-2.0-only

#pragma once

#include <pthread.h>
#include <stdbool.h>
#include <stddef.h>

// Forward declarations
struct AVFrame;

// Maximum output files per encode job (matches BATCH_MAX_FILES_PER_SHEET)
#define ENCODE_MAX_OUTPUTS 2

// An encode job - represents one processed sheet to be encoded
typedef struct {
  struct AVFrame *frame;                    // Frame to encode (owned by queue)
  char *output_files[ENCODE_MAX_OUTPUTS];   // Output file paths (borrowed)
  int output_count;                         // Number of output files
  int output_pixel_format;                  // AVPixelFormat for output
  int job_index;                            // Original job index for ordering
  bool valid;                               // True if job is valid
  bool uses_pinned_memory;                  // True if frame uses CUDA pinned memory
} EncodeJob;

// Encode queue - bounded queue for async encoding
typedef struct EncodeQueue EncodeQueue;

// Statistics for the encode queue
typedef struct {
  size_t images_queued;         // Total images queued by producers
  size_t images_encoded;        // Total images encoded by consumers
  size_t producer_waits;        // Times producer waited (queue full)
  size_t consumer_waits;        // Times consumer waited (queue empty)
  size_t peak_queue_depth;      // Maximum queue occupancy observed
  double total_encode_time_ms;  // Total time spent encoding
  double avg_encode_time_ms;    // Average encode time per image
} EncodeQueueStats;

// Create an encode queue
// - queue_depth: Maximum pending encode jobs to buffer (e.g., 8-16)
// - num_encoder_threads: Number of parallel encoder threads (e.g., 2-4)
EncodeQueue *encode_queue_create(size_t queue_depth, int num_encoder_threads);

// Destroy the encode queue and wait for all pending encodes
void encode_queue_destroy(EncodeQueue *queue);

// Start the encoder consumer thread(s)
// Returns true on success
bool encode_queue_start(EncodeQueue *queue);

// Signal that no more jobs will be submitted
// Encoder threads will finish remaining work and exit
void encode_queue_signal_done(EncodeQueue *queue);

// Wait for all encoder threads to complete
void encode_queue_wait(EncodeQueue *queue);

// Submit an image for encoding (producer interface)
// - frame: AVFrame to encode (ownership transferred to queue)
// - output_files: Array of output file paths (borrowed, must remain valid)
// - output_count: Number of output files
// - output_pixel_format: AVPixelFormat for output encoding
// - job_index: Job index for statistics/ordering
// - uses_pinned_memory: Whether frame uses CUDA pinned memory
// Returns true on success, false if queue is shutting down
bool encode_queue_submit(EncodeQueue *queue, struct AVFrame *frame,
                         char **output_files, int output_count,
                         int output_pixel_format, int job_index,
                         bool uses_pinned_memory);

// Check if queue is accepting submissions
bool encode_queue_active(const EncodeQueue *queue);

// Get queue statistics
EncodeQueueStats encode_queue_get_stats(const EncodeQueue *queue);

// Print queue statistics to stderr
void encode_queue_print_stats(const EncodeQueue *queue);
