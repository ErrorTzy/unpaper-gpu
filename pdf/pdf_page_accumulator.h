// SPDX-FileCopyrightText: 2025 The unpaper authors
//
// SPDX-License-Identifier: GPL-2.0-only

#pragma once

#include "pdf_writer.h"

#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

// ============================================================================
// PDF Page Accumulator
// ============================================================================
//
// Collects encoded pages from parallel workers and writes them to the PDF
// in sequential order. Pages can arrive out-of-order from workers, but they
// are written to the PDF in order (0, 1, 2, ...).
//
// Thread-safety: Multiple threads can submit pages concurrently.
// The accumulator uses internal locking for thread safety.

// Opaque handle to a page accumulator
typedef struct PdfPageAccumulator PdfPageAccumulator;

// Page data types
typedef enum {
  PDF_PAGE_DATA_JPEG,   // JPEG bytes (zero-copy path)
  PDF_PAGE_DATA_JP2,    // JP2 bytes (zero-copy path)
  PDF_PAGE_DATA_PIXELS, // Raw pixel data (Flate compression)
} PdfPageDataType;

// Encoded page data
typedef struct {
  int page_index;       // 0-based page index
  PdfPageDataType type; // Data type
  uint8_t *data;        // Image/pixel data (owned by accumulator after submit)
  size_t data_size;     // Size of data in bytes
  int width;            // Image width in pixels
  int height;           // Image height in pixels
  int stride;           // Row stride (for pixels only)
  PdfPixelFormat pixel_format; // Pixel format (for pixels only)
  int dpi;                     // Page DPI
} PdfEncodedPage;

// ============================================================================
// API
// ============================================================================

// Create a page accumulator.
// writer: PDF writer to write pages to (must remain valid until destroy)
// total_pages: Total number of pages expected
//
// Returns NULL on failure.
PdfPageAccumulator *pdf_page_accumulator_create(PdfWriter *writer,
                                                int total_pages);

// Destroy a page accumulator and free resources.
// This does NOT close the PDF writer - caller must do that separately.
void pdf_page_accumulator_destroy(PdfPageAccumulator *acc);

// Submit an encoded page for writing.
// Pages can be submitted in any order; they are written in sequential order.
// The accumulator takes ownership of page->data - caller must not free it.
//
// Returns true on success.
// Thread-safe: Can be called from multiple threads concurrently.
bool pdf_page_accumulator_submit(PdfPageAccumulator *acc,
                                 const PdfEncodedPage *page);

// Mark a page as failed (will not be submitted).
// This prevents the accumulator from waiting forever for a failed page.
// Thread-safe.
void pdf_page_accumulator_mark_failed(PdfPageAccumulator *acc, int page_index);

// Wait for all pages to be written.
// This blocks until all submitted pages have been written to the PDF.
// Returns true if all pages were written successfully.
bool pdf_page_accumulator_wait(PdfPageAccumulator *acc);

// Get the number of pages written so far.
int pdf_page_accumulator_pages_written(const PdfPageAccumulator *acc);

// Get the number of failed pages.
int pdf_page_accumulator_pages_failed(const PdfPageAccumulator *acc);

#ifdef __cplusplus
}
#endif
