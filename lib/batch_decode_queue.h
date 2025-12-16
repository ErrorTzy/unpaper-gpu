// SPDX-FileCopyrightText: 2025 The unpaper authors
//
// SPDX-License-Identifier: GPL-2.0-only

#pragma once

#include "lib/batch.h"
#include "lib/options.h"

#include <pthread.h>
#include <stdbool.h>
#include <stddef.h>

// Forward declarations
struct AVFrame;
struct Image;

// ============================================================================
// Batch Decode Queue (PR36B)
// ============================================================================
// This module implements the batch-collect-decode-distribute architecture
// for maximum GPU decode throughput. Unlike the per-image decode_queue,
// this queue:
//
// 1. Collects JPEG file data in parallel (multiple I/O threads)
// 2. Batch decodes using nvjpegDecodeBatched() (single sync per chunk)
// 3. Distributes decoded images to worker threads
//
// Key performance advantage: nvjpegDecodeBatched() achieves ~3x+ scaling
// over per-image nvjpegDecode() because it uses a single sync point and
// allows nvJPEG to optimize GPU parallelism internally.

// Maximum images to process in a single decode chunk.
// This bounds GPU memory usage while maintaining batch efficiency.
// 8 images at 4000x6000x3 = ~576MB GPU RAM
#define BATCH_DECODE_CHUNK_SIZE 8

// A decoded image ready for processing (same as DecodedImage for compatibility)
typedef struct {
  struct AVFrame *frame;   // Decoded CPU frame (NULL if on_gpu)
  int job_index;           // Which job this belongs to
  int input_index;         // Which input within the job (0 or 1)
  bool valid;              // True if decoding succeeded
  bool uses_pinned_memory; // True if frame data is in pinned memory
  // GPU decode fields
  bool on_gpu;         // True if decoded directly to GPU via nvJPEG
  void *gpu_ptr;       // GPU memory pointer (if on_gpu)
  size_t gpu_pitch;    // Row pitch in bytes
  int gpu_width;       // Image width in pixels
  int gpu_height;      // Image height in pixels
  int gpu_channels;    // Number of channels (1 for gray, 3 for RGB)
  int gpu_format;      // AVPixelFormat equivalent
  bool gpu_pool_owned; // True if gpu_ptr is from batch pool (don't free)
} BatchDecodedImage;

// Batch decode queue - opaque structure
typedef struct BatchDecodeQueue BatchDecodeQueue;

// Statistics for the batch decode queue
typedef struct {
  size_t total_images;         // Total images processed
  size_t gpu_batched_decodes;  // Images decoded via nvjpegDecodeBatched
  size_t gpu_single_decodes;   // Images decoded via nvjpegDecode (fallback)
  size_t cpu_decodes;          // Images decoded via FFmpeg (non-JPEG)
  size_t decode_failures;      // Failed decodes
  size_t batch_calls;          // Number of nvjpeg_decode_batch calls
  size_t io_threads_used;      // Number of I/O threads used
  size_t chunks_processed;     // Number of decode chunks processed
  size_t peak_queue_depth;     // Maximum queue occupancy
  double total_io_time_ms;     // Total time spent on file I/O
  double total_decode_time_ms; // Total time spent on GPU decode
} BatchDecodeQueueStats;

// ============================================================================
// Queue Creation and Destruction
// ============================================================================

// Create a batch decode queue.
// Parameters:
//   queue_depth: Number of decoded image slots to buffer (e.g., 16-32)
//   num_io_threads: Number of parallel file I/O threads (e.g., 4-8)
//   max_width, max_height: Maximum image dimensions to support
// Returns: Queue pointer or NULL on failure
BatchDecodeQueue *batch_decode_queue_create(size_t queue_depth,
                                            int num_io_threads, int max_width,
                                            int max_height);

// Destroy the batch decode queue and free all resources.
void batch_decode_queue_destroy(BatchDecodeQueue *queue);

// ============================================================================
// Configuration
// ============================================================================

// Enable GPU decode using nvJPEG batched API.
// Must be called before start_producer.
// When enabled, JPEG files are batch-decoded directly to GPU memory.
void batch_decode_queue_enable_gpu(BatchDecodeQueue *queue, bool enable);

// Set the decode chunk size (number of images per batch decode call).
// Must be called before start_producer.
// Parameters:
//   queue: The batch decode queue
//   chunk_size: Number of images per batch (1-64, 0 uses
//   BATCH_DECODE_CHUNK_SIZE)
void batch_decode_queue_set_chunk_size(BatchDecodeQueue *queue, int chunk_size);

// ============================================================================
// Producer Control
// ============================================================================

// Start the producer threads (I/O + decode orchestration).
// Parameters:
//   queue: The batch decode queue
//   batch_queue: Source of jobs to process
//   options: Processing options
// Returns: true on success
bool batch_decode_queue_start(BatchDecodeQueue *queue, BatchQueue *batch_queue,
                              const Options *options);

// Stop all producer threads and wait for completion.
void batch_decode_queue_stop(BatchDecodeQueue *queue);

// Check if producers have finished (non-blocking).
bool batch_decode_queue_done(BatchDecodeQueue *queue);

// ============================================================================
// Consumer Interface
// ============================================================================

// Get a decoded image for a specific job/input.
// Blocks until the image is available or producers signal completion.
// Returns: Pointer to decoded image or NULL if not available
BatchDecodedImage *batch_decode_queue_get(BatchDecodeQueue *queue,
                                          int job_index, int input_index);

// Release a consumed image back to the queue.
// The image resources will be freed.
void batch_decode_queue_release(BatchDecodeQueue *queue,
                                BatchDecodedImage *image);

// ============================================================================
// Statistics
// ============================================================================

// Get queue statistics.
BatchDecodeQueueStats
batch_decode_queue_get_stats(const BatchDecodeQueue *queue);

// Print queue statistics to stderr.
void batch_decode_queue_print_stats(const BatchDecodeQueue *queue);
