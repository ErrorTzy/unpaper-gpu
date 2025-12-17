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
// nvImageCodec Unified GPU Codec API (PR4)
// ============================================================================
// This module provides a unified interface for GPU-accelerated image decode
// and encode, supporting both JPEG and JPEG2000 formats.
//
// Implementation priority:
// 1. nvImageCodec (if available) - unified API for JPEG/JP2/TIFF/PNG
// 2. nvJPEG (fallback) - JPEG only, always available with CUDA
//
// Key performance benefits:
// - JPEG2000 GPU decode/encode for archival PDF workflows
// - Automatic format detection from image headers
// - Same API for both formats, simplifying queue integration
// - Zero-copy paths when used with compatible GPU buffers

// ============================================================================
// Image Format Detection
// ============================================================================

// Supported image formats
typedef enum {
  NVIMGCODEC_FORMAT_UNKNOWN = 0,
  NVIMGCODEC_FORMAT_JPEG = 1,     // JPEG (JFIF/EXIF)
  NVIMGCODEC_FORMAT_JPEG2000 = 2, // JPEG2000 (JP2/J2K)
  NVIMGCODEC_FORMAT_PNG = 3,      // PNG (for future extension)
  NVIMGCODEC_FORMAT_TIFF = 4,     // TIFF (for future extension)
} NvImgCodecFormat;

// Detect image format from data header
// Returns the format detected from the magic bytes
NvImgCodecFormat nvimgcodec_detect_format(const uint8_t *data, size_t size);

// Check if format is supported for GPU decode
bool nvimgcodec_format_decode_supported(NvImgCodecFormat format);

// Check if format is supported for GPU encode
bool nvimgcodec_format_encode_supported(NvImgCodecFormat format);

// ============================================================================
// Decode Output Format
// ============================================================================

typedef enum {
  NVIMGCODEC_OUT_GRAY8 = 0, // Single-channel grayscale (8-bit)
  NVIMGCODEC_OUT_RGB = 1,   // Interleaved RGB (24-bit)
  NVIMGCODEC_OUT_BGR = 2,   // Interleaved BGR (24-bit)
} NvImgCodecOutputFormat;

// ============================================================================
// Decoded Image Result
// ============================================================================

typedef struct {
  void *gpu_ptr;               // GPU device pointer to decoded image
  size_t pitch;                // Row pitch in bytes (may include padding)
  int width;                   // Image width in pixels
  int height;                  // Image height in pixels
  int channels;                // Number of channels (1 for gray, 3 for RGB)
  NvImgCodecOutputFormat fmt;  // Output format
  NvImgCodecFormat source_fmt; // Original image format (JPEG/JP2)
  void *completion_event;      // CUDA event signaled when decode completes
  bool event_from_pool;        // True if event came from global event pool
} NvImgCodecDecodedImage;

// Wait for decode to complete (sync on completion event).
// Call this before accessing gpu_ptr data from a different stream.
void nvimgcodec_wait_decode_complete(NvImgCodecDecodedImage *image);

// Release completion event back to pool (or destroy if not from pool).
void nvimgcodec_release_completion_event(void *event, bool from_pool);

// ============================================================================
// Encode Output Result
// ============================================================================

typedef struct {
  uint8_t *data; // Encoded bitstream data (host memory, caller must free)
  size_t size;   // Size of encoded data in bytes
  int width;     // Original image width
  int height;    // Original image height
  NvImgCodecFormat fmt; // Output format (JPEG or JP2)
} NvImgCodecEncodedImage;

// ============================================================================
// Decode Stream State
// ============================================================================
// Per-stream state for decode operations.
// Each CUDA stream requires dedicated state for concurrent decode.
typedef struct NvImgCodecDecodeState NvImgCodecDecodeState;

// ============================================================================
// Encode Stream State
// ============================================================================
// Per-stream state for encode operations.
typedef struct NvImgCodecEncodeState NvImgCodecEncodeState;

// ============================================================================
// Global Context Management
// ============================================================================

// Initialize the nvImageCodec context.
// This initializes nvImageCodec if available, otherwise falls back to nvJPEG.
//
// num_streams: Number of stream states to pre-allocate
// Returns true on success.
// Thread-safe: Can be called from any thread, but only initializes once.
bool nvimgcodec_init(int num_streams);

// Clean up nvImageCodec context and free all resources.
void nvimgcodec_cleanup(void);

// Check if nvImageCodec (unified) is available and initialized.
// Returns true if nvImageCodec library is loaded, false if using nvJPEG
// fallback.
bool nvimgcodec_is_available(void);

// Check if any GPU codec (nvImageCodec or nvJPEG fallback) is available.
bool nvimgcodec_any_available(void);

// Check if JPEG2000 is supported (only with nvImageCodec, not nvJPEG fallback).
bool nvimgcodec_jp2_supported(void);

// ============================================================================
// Statistics
// ============================================================================

typedef struct {
  size_t total_decodes;      // Number of decode calls
  size_t successful_decodes; // Successful decodes
  size_t jpeg_decodes;       // JPEG decodes
  size_t jp2_decodes;        // JPEG2000 decodes
  size_t fallback_decodes;   // Fell back to CPU (FFmpeg)
  size_t total_encodes;      // Number of encode calls
  size_t successful_encodes; // Successful encodes
  size_t jpeg_encodes;       // JPEG encodes
  size_t jp2_encodes;        // JP2 encodes
  bool using_nvimgcodec; // True if using nvImageCodec, false if nvJPEG fallback
} NvImgCodecStats;

// Get current statistics.
NvImgCodecStats nvimgcodec_get_stats(void);

// Print statistics to stderr.
void nvimgcodec_print_stats(void);

// ============================================================================
// Decode Stream State Management
// ============================================================================

// Acquire a decode state for decode operations.
// Returns NULL if pool is exhausted.
NvImgCodecDecodeState *nvimgcodec_acquire_decode_state(void);

// Release a decode state back to the pool.
void nvimgcodec_release_decode_state(NvImgCodecDecodeState *state);

// ============================================================================
// Encode Stream State Management
// ============================================================================

// Acquire an encode state for encode operations.
// Returns NULL if pool is exhausted.
NvImgCodecEncodeState *nvimgcodec_acquire_encode_state(void);

// Release an encode state back to the pool.
void nvimgcodec_release_encode_state(NvImgCodecEncodeState *state);

// ============================================================================
// Image Decode Operations
// ============================================================================

// Get information about an image without decoding.
// Detects format automatically from header bytes.
// Returns true on success.
bool nvimgcodec_get_image_info(const uint8_t *data, size_t size,
                               NvImgCodecFormat *format, int *width,
                               int *height, int *channels);

// Decode an image directly to GPU memory.
// Automatically detects format and uses appropriate codec.
//
// data: Pointer to image data in host memory
// size: Size of image data in bytes
// state: Per-stream decode state (acquired via nvimgcodec_acquire_decode_state)
// stream: CUDA stream for async operations (can be NULL for default stream)
// output_fmt: Desired output format (NVIMGCODEC_OUT_GRAY8 or
// NVIMGCODEC_OUT_RGB) out: Output structure filled with decoded image info
//
// Returns true on success, false on failure.
//
// Memory ownership:
// - GPU memory (out->gpu_ptr) is allocated by this function
// - Caller must free with cudaFree when done
bool nvimgcodec_decode(const uint8_t *data, size_t size,
                       NvImgCodecDecodeState *state, UnpaperCudaStream *stream,
                       NvImgCodecOutputFormat output_fmt,
                       NvImgCodecDecodedImage *out);

// Decode from file directly to GPU memory.
// Convenience wrapper that reads file and decodes.
bool nvimgcodec_decode_file(const char *filename, UnpaperCudaStream *stream,
                            NvImgCodecOutputFormat output_fmt,
                            NvImgCodecDecodedImage *out);

// ============================================================================
// Image Encode Operations
// ============================================================================

// Encode input format (matches NvJpegEncodeFormat for compatibility)
typedef enum {
  NVIMGCODEC_ENC_FMT_GRAY8 = 0, // Single-channel grayscale (8-bit)
  NVIMGCODEC_ENC_FMT_RGB = 1,   // Interleaved RGB (24-bit)
  NVIMGCODEC_ENC_FMT_BGR = 2,   // Interleaved BGR (24-bit)
} NvImgCodecEncodeInputFormat;

// Encode parameters
typedef struct {
  NvImgCodecFormat output_format; // JPEG or JP2
  int quality;                    // Quality (1-100, default 85)
  bool lossless;                  // Lossless encoding (JP2 only)
} NvImgCodecEncodeParams;

// Default encode parameters for JPEG output
static inline NvImgCodecEncodeParams nvimgcodec_default_jpeg_params(void) {
  NvImgCodecEncodeParams params = {
      .output_format = NVIMGCODEC_FORMAT_JPEG,
      .quality = 85,
      .lossless = false,
  };
  return params;
}

// Default encode parameters for JP2 lossless output
static inline NvImgCodecEncodeParams
nvimgcodec_default_jp2_lossless_params(void) {
  NvImgCodecEncodeParams params = {
      .output_format = NVIMGCODEC_FORMAT_JPEG2000,
      .quality = 100,
      .lossless = true,
  };
  return params;
}

// Encode a GPU-resident image.
//
// gpu_ptr: GPU device pointer to image data (interleaved format)
// pitch: Row pitch in bytes
// width: Image width in pixels
// height: Image height in pixels
// input_fmt: Input pixel format
// state: Per-stream encode state
// stream: CUDA stream for async operations
// params: Encode parameters (format, quality)
// out: Output structure filled with encoded data
//
// Returns true on success.
//
// Memory ownership:
// - out->data is allocated by this function (malloc)
// - Caller must free(out->data) when done
bool nvimgcodec_encode(const void *gpu_ptr, size_t pitch, int width, int height,
                       NvImgCodecEncodeInputFormat input_fmt,
                       NvImgCodecEncodeState *state, UnpaperCudaStream *stream,
                       const NvImgCodecEncodeParams *params,
                       NvImgCodecEncodedImage *out);

// Convenience: Encode GPU image to JPEG.
bool nvimgcodec_encode_jpeg(const void *gpu_ptr, size_t pitch, int width,
                            int height, NvImgCodecEncodeInputFormat input_fmt,
                            int quality, NvImgCodecEncodeState *state,
                            UnpaperCudaStream *stream,
                            NvImgCodecEncodedImage *out);

// Convenience: Encode GPU image to JP2 (lossless by default).
bool nvimgcodec_encode_jp2(const void *gpu_ptr, size_t pitch, int width,
                           int height, NvImgCodecEncodeInputFormat input_fmt,
                           bool lossless, NvImgCodecEncodeState *state,
                           UnpaperCudaStream *stream,
                           NvImgCodecEncodedImage *out);

// Convenience: Encode GPU image and write to file.
bool nvimgcodec_encode_to_file(const void *gpu_ptr, size_t pitch, int width,
                               int height,
                               NvImgCodecEncodeInputFormat input_fmt,
                               UnpaperCudaStream *stream,
                               const NvImgCodecEncodeParams *params,
                               const char *filename);

// ============================================================================
// Quality Control
// ============================================================================

// Set default JPEG quality for subsequent encodes (1-100).
void nvimgcodec_set_jpeg_quality(int quality);

// Get current default JPEG quality.
int nvimgcodec_get_jpeg_quality(void);

#ifdef __cplusplus
}
#endif
