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

// Forward declarations
typedef struct UnpaperCudaStream UnpaperCudaStream;

// nvJPEG decode output format
typedef enum {
  NVJPEG_FMT_GRAY8 = 0,    // Single-channel grayscale (8-bit)
  NVJPEG_FMT_RGB = 1,      // Interleaved RGB (24-bit)
  NVJPEG_FMT_BGR = 2,      // Interleaved BGR (24-bit)
} NvJpegOutputFormat;

// Per-stream state for nvJPEG decode operations.
// Each CUDA stream requires dedicated state for concurrent decode.
// Thread-safety model:
// - nvjpegHandle_t: Thread-safe, ONE per process (shared)
// - nvjpegJpegState_t: NOT thread-safe, ONE per stream
// - nvjpegBufferDevice_t: NOT thread-safe, ONE per stream
// - nvjpegBufferPinned_t: NOT thread-safe, TWO per stream (double-buffer)
typedef struct NvJpegStreamState NvJpegStreamState;

// Decoded image information returned by nvjpeg_decode_to_gpu
typedef struct {
  void *gpu_ptr;           // GPU device pointer to decoded image
  size_t pitch;            // Row pitch in bytes (may include padding)
  int width;               // Image width in pixels
  int height;              // Image height in pixels
  int channels;            // Number of channels (1 for gray, 3 for RGB)
  NvJpegOutputFormat fmt;  // Output format
} NvJpegDecodedImage;

// Pool statistics for monitoring
typedef struct {
  size_t total_decodes;            // Number of decode calls
  size_t successful_decodes;       // Successful decodes
  size_t fallback_decodes;         // Fell back to CPU (FFmpeg)
  size_t concurrent_peak;          // Peak concurrent decodes
  size_t current_in_use;           // Currently active stream states
  size_t stream_state_count;       // Total stream states in pool
} NvJpegStats;

// ============================================================================
// Global Context Management
// ============================================================================

// Initialize the global nvJPEG context.
// num_streams: Number of stream states to pre-allocate (typically matches
//              the number of CUDA streams used for batch processing)
// Returns true on success, false on failure.
// Thread-safe: Can be called from any thread, but only initializes once.
//
// CRITICAL: This uses custom stream-ordered allocators (cudaMallocAsync)
// to avoid serialization across streams. Without this, nvJPEG's internal
// cudaMalloc calls would serialize all concurrent decodes.
bool nvjpeg_context_init(int num_streams);

// Clean up global nvJPEG context and free all resources.
// Thread-safe: Can be called from any thread.
void nvjpeg_context_cleanup(void);

// Check if nvJPEG is initialized and available.
bool nvjpeg_is_available(void);

// Get current statistics.
NvJpegStats nvjpeg_get_stats(void);

// Print statistics to stderr.
void nvjpeg_print_stats(void);

// ============================================================================
// Stream State Management
// ============================================================================

// Acquire a stream state for decode operations.
// Returns NULL if pool is exhausted (all states in use).
// Thread-safe: Uses lock-free atomic acquisition.
NvJpegStreamState *nvjpeg_acquire_stream_state(void);

// Release a stream state back to the pool.
// Thread-safe: Uses lock-free atomic release.
void nvjpeg_release_stream_state(NvJpegStreamState *state);

// ============================================================================
// JPEG Decode Operations
// ============================================================================

// Get information about a JPEG image without decoding.
// jpeg_data: Pointer to JPEG data in host memory
// jpeg_size: Size of JPEG data in bytes
// width, height: Output parameters for image dimensions
// channels: Output parameter for number of channels (1 or 3)
// Returns true on success.
bool nvjpeg_get_image_info(const uint8_t *jpeg_data, size_t jpeg_size,
                           int *width, int *height, int *channels);

// Decode a JPEG image directly to GPU memory.
// This is the primary decode function for batch processing.
//
// jpeg_data: Pointer to JPEG data in host memory
// jpeg_size: Size of JPEG data in bytes
// state: Per-stream state (acquired via nvjpeg_acquire_stream_state)
// stream: CUDA stream for async operations (can be NULL for default stream)
// output_fmt: Desired output format (NVJPEG_FMT_GRAY8 or NVJPEG_FMT_RGB)
// out: Output structure filled with decoded image info
//
// Returns true on success, false on failure (invalid JPEG, out of memory, etc.)
//
// Memory ownership:
// - GPU memory (out->gpu_ptr) is owned by the stream state's device buffer
// - Memory remains valid until the next decode on the same stream state
// - Caller must copy data if needed beyond the stream state's lifetime
//
// Threading model:
// - Each decode must use a different stream state for concurrent execution
// - The CUDA stream parameter determines execution ordering
bool nvjpeg_decode_to_gpu(const uint8_t *jpeg_data, size_t jpeg_size,
                          NvJpegStreamState *state,
                          UnpaperCudaStream *stream,
                          NvJpegOutputFormat output_fmt,
                          NvJpegDecodedImage *out);

// Synchronous decode from file (convenience wrapper).
// Opens file, reads to memory, decodes, then closes file.
// Allocates new GPU memory for output (caller must free with cudaFree).
//
// filename: Path to JPEG file
// stream: CUDA stream (can be NULL)
// output_fmt: Desired output format
// out: Output structure (gpu_ptr will be newly allocated)
//
// Returns true on success.
bool nvjpeg_decode_file_to_gpu(const char *filename,
                               UnpaperCudaStream *stream,
                               NvJpegOutputFormat output_fmt,
                               NvJpegDecodedImage *out);

#ifdef __cplusplus
}
#endif
