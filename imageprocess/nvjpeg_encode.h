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

// ============================================================================
// nvJPEG GPU Encode API (PR37)
// ============================================================================
// This module provides GPU-resident JPEG encoding using nvJPEG.
// Key performance benefits:
// - No D2H transfer needed for GPU-resident images
// - Async encoding with CUDA streams
// - Pre-allocated output buffers for consistent performance
//
// Memory model:
// - Input: GPU device pointer (interleaved RGB or grayscale)
// - Output: Host memory buffer containing JPEG bitstream
//
// Threading model:
// - nvjpegEncoderState is NOT thread-safe, one per stream
// - Use nvjpeg_encode_acquire_state() / nvjpeg_encode_release_state()

// nvJPEG encode input format
typedef enum {
  NVJPEG_ENC_FMT_GRAY8 = 0, // Single-channel grayscale (8-bit)
  NVJPEG_ENC_FMT_RGB = 1,   // Interleaved RGB (24-bit)
  NVJPEG_ENC_FMT_BGR = 2,   // Interleaved BGR (24-bit)
} NvJpegEncodeFormat;

// nvJPEG chroma subsampling for encoding
typedef enum {
  NVJPEG_ENC_SUBSAMPLING_444 = 0, // No subsampling (highest quality)
  NVJPEG_ENC_SUBSAMPLING_422 = 1, // Horizontal 2:1 subsampling
  NVJPEG_ENC_SUBSAMPLING_420 = 2, // Both 2:1 subsampling (most common)
  NVJPEG_ENC_SUBSAMPLING_GRAY = 3 // Grayscale (no chroma)
} NvJpegEncodeSubsampling;

// Per-stream encoder state
typedef struct NvJpegEncoderState NvJpegEncoderState;

// Encoded image output
typedef struct {
  uint8_t *jpeg_data; // JPEG bitstream data (host memory, caller must free)
  size_t jpeg_size;   // Size of JPEG data in bytes
  int width;          // Original image width
  int height;         // Original image height
} NvJpegEncodedImage;

// Encode statistics
typedef struct {
  size_t total_encodes;       // Number of encode calls
  size_t successful_encodes;  // Successful encodes
  size_t failed_encodes;      // Failed encodes
  size_t total_bytes_out;     // Total JPEG bytes produced
  size_t concurrent_peak;     // Peak concurrent encodes
  size_t current_in_use;      // Currently active encoder states
  size_t encoder_state_count; // Total encoder states in pool
} NvJpegEncodeStats;

// ============================================================================
// Global Context Management
// ============================================================================

// Initialize the nvJPEG encoder context.
// Requires nvjpeg_context_init() to be called first (for shared handle).
//
// num_encoders: Number of encoder states to pre-allocate (typically matches
//               the number of CUDA streams used for batch processing)
// quality: JPEG quality (1-100, default 85)
// subsampling: Chroma subsampling mode
//
// Returns true on success.
// Thread-safe: Can be called from any thread, but only initializes once.
bool nvjpeg_encode_init(int num_encoders, int quality,
                        NvJpegEncodeSubsampling subsampling);

// Clean up nvJPEG encoder context.
void nvjpeg_encode_cleanup(void);

// Check if nvJPEG encoder is initialized and available.
bool nvjpeg_encode_is_available(void);

// Get encode statistics.
NvJpegEncodeStats nvjpeg_encode_get_stats(void);

// Print encode statistics to stderr.
void nvjpeg_encode_print_stats(void);

// ============================================================================
// Encoder State Management
// ============================================================================

// Acquire an encoder state for encode operations.
// Returns NULL if pool is exhausted.
// Thread-safe: Uses lock-free atomic acquisition.
NvJpegEncoderState *nvjpeg_encode_acquire_state(void);

// Release an encoder state back to the pool.
// Thread-safe: Uses lock-free atomic release.
void nvjpeg_encode_release_state(NvJpegEncoderState *state);

// ============================================================================
// Single Image Encode
// ============================================================================

// Encode a single image from GPU memory to JPEG.
// This is the primary encode function for GPU-resident images.
//
// gpu_ptr: GPU device pointer to image data (interleaved format)
// pitch: Row pitch in bytes (may include padding)
// width: Image width in pixels
// height: Image height in pixels
// format: Input pixel format (NVJPEG_ENC_FMT_RGB or NVJPEG_ENC_FMT_GRAY8)
// state: Per-stream encoder state (acquired via nvjpeg_encode_acquire_state)
// stream: CUDA stream for async operations (can be NULL for default stream)
// out: Output structure filled with JPEG data
//
// Returns true on success.
//
// Memory ownership:
// - out->jpeg_data is allocated by this function (malloc)
// - Caller must free(out->jpeg_data) when done
//
// Threading model:
// - Each encode must use a different encoder state for concurrent execution
bool nvjpeg_encode_from_gpu(const void *gpu_ptr, size_t pitch, int width,
                            int height, NvJpegEncodeFormat format,
                            NvJpegEncoderState *state,
                            UnpaperCudaStream *stream, NvJpegEncodedImage *out);

// Convenience wrapper: encode from GPU and write directly to file.
// Returns true on success.
bool nvjpeg_encode_gpu_to_file(const void *gpu_ptr, size_t pitch, int width,
                               int height, NvJpegEncodeFormat format,
                               UnpaperCudaStream *stream, const char *filename);

// ============================================================================
// Batched Encode API
// ============================================================================
// For maximum throughput when encoding multiple images.

// Maximum images to encode in a single batch call.
#define NVJPEG_MAX_ENCODE_BATCH_SIZE 16

// Initialize batched encoder with pre-allocated resources.
// Must call nvjpeg_encode_init() first.
//
// max_batch_size: Maximum images per batch (capped at
// NVJPEG_MAX_ENCODE_BATCH_SIZE) max_width, max_height: Maximum image dimensions
//
// Returns true on success.
bool nvjpeg_encode_batch_init(int max_batch_size, int max_width,
                              int max_height);

// Encode a batch of images from GPU memory.
//
// gpu_ptrs: Array of GPU device pointers
// pitches: Array of row pitches
// widths: Array of image widths
// heights: Array of image heights
// formats: Array of input formats (or single format if all same)
// batch_size: Number of images to encode
// outputs: Pre-allocated array of NvJpegEncodedImage structs
//
// Returns number of successfully encoded images.
// Each outputs[i].jpeg_data must be freed by caller.
int nvjpeg_encode_batch(const void *const *gpu_ptrs, const size_t *pitches,
                        const int *widths, const int *heights,
                        NvJpegEncodeFormat format, int batch_size,
                        NvJpegEncodedImage *outputs);

// Check if batched encoder is initialized.
bool nvjpeg_encode_batch_is_ready(void);

// Clean up batched encoder resources.
void nvjpeg_encode_batch_cleanup(void);

// ============================================================================
// Quality Control
// ============================================================================

// Set JPEG quality for subsequent encodes (1-100).
// Can be called after init to change quality dynamically.
// Thread-safe: Applies to all encoder states.
void nvjpeg_encode_set_quality(int quality);

// Get current quality setting.
int nvjpeg_encode_get_quality(void);

// Set chroma subsampling for subsequent encodes.
void nvjpeg_encode_set_subsampling(NvJpegEncodeSubsampling subsampling);

// Get current subsampling setting.
NvJpegEncodeSubsampling nvjpeg_encode_get_subsampling(void);

#ifdef __cplusplus
}
#endif
