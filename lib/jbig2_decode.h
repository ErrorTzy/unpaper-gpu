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

// Decoded JBIG2 image (1-bit packed bitmap)
typedef struct {
  uint8_t *data;   // 1-bit packed pixels (MSB first, rows padded to byte)
  uint32_t width;  // Image width in pixels
  uint32_t height; // Image height in pixels
  uint32_t stride; // Bytes per row (includes padding)
} Jbig2DecodedImage;

// Decode JBIG2 data to a 1-bit bitmap.
//
// Parameters:
//   data: Raw JBIG2 stream data (from PDF image stream)
//   size: Size of data in bytes
//   globals: Optional global dictionary data (from PDF JBIG2Globals, may be
//   NULL) globals_size: Size of globals in bytes (0 if no globals) out: Output
//   structure filled with decoded bitmap
//
// Returns true on success, false on failure.
// Caller must free out->data with jbig2_free_image().
//
// Performance note: JBIG2 decode is CPU-only. For batch processing,
// decode can run in parallel across multiple pages.
bool jbig2_decode(const uint8_t *data, size_t size, const uint8_t *globals,
                  size_t globals_size, Jbig2DecodedImage *out);

// Free a decoded JBIG2 image.
// Safe to call with NULL or zero-initialized struct.
void jbig2_free_image(Jbig2DecodedImage *image);

// Expand a 1-bit JBIG2 image to 8-bit grayscale.
// This is useful for feeding into the image processing pipeline.
//
// Parameters:
//   jbig2: Input 1-bit packed bitmap
//   gray_out: Output buffer (must be at least width * height bytes)
//   gray_stride: Output row stride in bytes
//   invert: If true, invert values (JBIG2 typically uses 0=white, 1=black)
//
// Returns true on success, false on failure.
bool jbig2_expand_to_gray8(const Jbig2DecodedImage *jbig2, uint8_t *gray_out,
                           size_t gray_stride, bool invert);

// Check if JBIG2 support is available at runtime.
// Returns true if jbig2dec was compiled in.
bool jbig2_is_available(void);

// Get the last error message (thread-local).
const char *jbig2_get_last_error(void);

#ifdef __cplusplus
}
#endif
