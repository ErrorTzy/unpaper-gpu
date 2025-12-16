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

// A pre-decoded image ready for processing
typedef struct {
  struct AVFrame *frame;   // Decoded frame (may use pinned memory)
  int job_index;           // Which job this belongs to
  int input_index;         // Which input within the job (0 or 1)
  bool valid;              // True if decoding succeeded
  bool uses_pinned_memory; // True if frame data is in pinned memory
  // GPU decode fields (PR36+)
  bool on_gpu;      // True if decoded directly to GPU via nvJPEG
  void *gpu_ptr;    // GPU memory pointer (if on_gpu)
  size_t gpu_pitch; // Row pitch in bytes
  int gpu_width;    // Image width in pixels
  int gpu_height;   // Image height in pixels
  int gpu_channels; // Number of channels (1 for gray, 3 for RGB)
  int gpu_format;   // AVPixelFormat equivalent (AV_PIX_FMT_GRAY8 or
                    // AV_PIX_FMT_RGB24)
  void *gpu_completion_event;    // CUDA event for async decode completion
  bool gpu_event_from_pool;      // True if event came from global event pool
} DecodedImage;

// Wait for GPU decode to complete (if image was decoded to GPU).
// Must call before accessing gpu_ptr from a different CUDA stream.
// Safe to call even if image was not GPU-decoded.
void decoded_image_wait_gpu_complete(DecodedImage *image);

// Decode queue - bounded queue of pre-decoded images
typedef struct DecodeQueue DecodeQueue;

// Statistics for the decode queue
typedef struct {
  size_t images_decoded;     // Total images decoded by producer
  size_t images_consumed;    // Total images consumed by workers
  size_t producer_waits;     // Times producer waited (queue full)
  size_t consumer_waits;     // Times consumer waited (queue empty)
  size_t pinned_allocations; // Pinned memory allocations
  size_t peak_queue_depth;   // Maximum queue occupancy observed
  // GPU decode stats (PR36+)
  size_t gpu_decodes;         // Images decoded via nvJPEG to GPU
  size_t cpu_decodes;         // Images decoded via FFmpeg to CPU
  size_t gpu_decode_failures; // nvJPEG failures (fell back to CPU)
} DecodeQueueStats;

// Create a decode queue
// - queue_depth: Maximum pre-decoded images to buffer (e.g., 4-8)
// - use_pinned_memory: Allocate decoded frames in CUDA-pinned memory
// - num_producers: Number of producer threads (1 = sequential, >1 = parallel)
DecodeQueue *decode_queue_create(size_t queue_depth, bool use_pinned_memory);

// Create a decode queue with multiple producer threads
// - queue_depth: Maximum pre-decoded images to buffer
// - use_pinned_memory: Allocate decoded frames in CUDA-pinned memory
// - num_producers: Number of producer threads for parallel decoding
DecodeQueue *decode_queue_create_parallel(size_t queue_depth,
                                          bool use_pinned_memory,
                                          int num_producers);

// Destroy the decode queue and free all resources
void decode_queue_destroy(DecodeQueue *queue);

// Enable GPU decode using nvJPEG for JPEG files
// Must be called before start_producer
// When enabled, JPEG files will be decoded directly to GPU memory,
// eliminating the H2D transfer for JPEG inputs.
void decode_queue_enable_gpu_decode(DecodeQueue *queue, bool enable);

// Start the producer thread
// - queue: The decode queue
// - batch_queue: Source of jobs to decode
// - options: Processing options (for background color, threshold)
// Returns true on success
bool decode_queue_start_producer(DecodeQueue *queue, BatchQueue *batch_queue,
                                 const Options *options);

// Stop the producer thread (waits for it to finish)
void decode_queue_stop_producer(DecodeQueue *queue);

// Consumer interface - get a pre-decoded image for a specific job/input
// Blocks until the image is available or producer signals completion
// Returns NULL if the producer has finished and no more images
DecodedImage *decode_queue_get(DecodeQueue *queue, int job_index,
                               int input_index);

// Release a consumed image back to the queue
// The image's frame will be freed
void decode_queue_release(DecodeQueue *queue, DecodedImage *image);

// Check if producer has finished (non-blocking)
bool decode_queue_producer_done(DecodeQueue *queue);

// Get queue statistics
DecodeQueueStats decode_queue_get_stats(const DecodeQueue *queue);

// Print queue statistics to stderr
void decode_queue_print_stats(const DecodeQueue *queue);
