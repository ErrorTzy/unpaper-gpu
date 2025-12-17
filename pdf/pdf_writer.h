// SPDX-FileCopyrightText: 2025 The unpaper authors
//
// SPDX-License-Identifier: GPL-2.0-only

#pragma once

#include "pdf_reader.h"

#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

// Opaque handle to a PDF writer
typedef struct PdfWriter PdfWriter;

// Pixel format for pdf_writer_add_page_pixels
typedef enum {
  PDF_PIXEL_GRAY8 = 0, // 8-bit grayscale
  PDF_PIXEL_RGB24,     // 24-bit RGB (R, G, B, R, G, B, ...)
} PdfPixelFormat;

// ============================================================================
// Writer Creation and Management
// ============================================================================

// Create a new PDF writer.
// path: Output file path
// meta: Optional metadata to copy into the new PDF (may be NULL)
// dpi: Default DPI for pages (used when computing page size from pixels)
//
// Returns NULL on failure. Caller must close with pdf_writer_close().
// Thread-safety: NOT thread-safe. Create one PdfWriter per thread.
PdfWriter *pdf_writer_create(const char *path, const PdfMetadata *meta,
                             int dpi);

// Close and finalize the PDF.
// This writes the PDF to disk and frees all resources.
// After calling this, the PdfWriter handle is invalid.
// Safe to call with NULL.
// Returns true on success, false if the save failed.
bool pdf_writer_close(PdfWriter *writer);

// Abort and discard the PDF without saving.
// Use this when an error occurs and you don't want to write a partial file.
void pdf_writer_abort(PdfWriter *writer);

// ============================================================================
// Page Addition - Zero-Copy Paths
// ============================================================================

// Add a page with JPEG image data.
// This is the zero-copy fast path for JPEG images - the raw JPEG bytes are
// embedded directly into the PDF without re-encoding.
//
// data: Raw JPEG data (including headers)
// len: Size of JPEG data in bytes
// width, height: Image dimensions in pixels
// dpi: Optional DPI override (0 to use writer's default DPI)
//
// Returns true on success.
//
// Performance: O(1) - just copies bytes, no decode/encode.
bool pdf_writer_add_page_jpeg(PdfWriter *writer, const uint8_t *data,
                              size_t len, int width, int height, int dpi);

// Add a page with JPEG2000 image data.
// This is the zero-copy fast path for JP2 images.
//
// data: Raw JP2 data
// len: Size of JP2 data in bytes
// width, height: Image dimensions in pixels
// dpi: Optional DPI override (0 to use writer's default DPI)
//
// Returns true on success.
bool pdf_writer_add_page_jp2(PdfWriter *writer, const uint8_t *data, size_t len,
                             int width, int height, int dpi);

// ============================================================================
// Page Addition - Pixel Path
// ============================================================================

// Add a page from raw pixel data.
// This path compresses pixels into the PDF using Flate compression.
// Use pdf_writer_add_page_jpeg when possible for better compression.
//
// pixels: Raw pixel data
// width, height: Image dimensions in pixels
// stride: Row stride in bytes (must be >= width * bytes_per_pixel)
// format: Pixel format (PDF_PIXEL_GRAY8 or PDF_PIXEL_RGB24)
// dpi: Optional DPI override (0 to use writer's default DPI)
//
// Returns true on success.
bool pdf_writer_add_page_pixels(PdfWriter *writer, const uint8_t *pixels,
                                int width, int height, int stride,
                                PdfPixelFormat format, int dpi);

// ============================================================================
// Utility Functions
// ============================================================================

// Get the number of pages added so far.
int pdf_writer_page_count(const PdfWriter *writer);

// Get the last error message (thread-local).
// Shares error buffer with pdf_reader functions.
const char *pdf_writer_get_last_error(void);

#ifdef __cplusplus
}
#endif
