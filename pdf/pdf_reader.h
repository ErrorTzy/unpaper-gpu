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

// Opaque handle to an opened PDF document
typedef struct PdfDocument PdfDocument;

// Image format detected from PDF stream
typedef enum {
  PDF_IMAGE_UNKNOWN = 0,
  PDF_IMAGE_JPEG,  // DCT-encoded (JPEG)
  PDF_IMAGE_JP2,   // JPX-encoded (JPEG2000)
  PDF_IMAGE_JBIG2, // JBIG2-encoded (B&W)
  PDF_IMAGE_CCITT, // CCITT Fax encoded (B&W)
  PDF_IMAGE_PNG,   // Flate with PNG predictor
  PDF_IMAGE_RAW,   // Raw uncompressed samples
  PDF_IMAGE_FLATE, // Flate compressed
} PdfImageFormat;

// Information about an extracted image
typedef struct {
  uint8_t *data;          // Raw image bytes (owned by caller after extraction)
  size_t size;            // Size of data in bytes
  int width;              // Image width in pixels
  int height;             // Image height in pixels
  int components;         // Number of color components (1=gray, 3=RGB, 4=CMYK)
  int bits_per_component; // Bits per component (1, 2, 4, 8, 16)
  PdfImageFormat format;  // Detected format of the raw bytes
  bool is_mask;           // True if this is a mask/stencil image

  // JBIG2-specific: global dictionary data (NULL if not present)
  uint8_t *jbig2_globals;    // JBIG2 global dictionary bytes (may be NULL)
  size_t jbig2_globals_size; // Size of globals in bytes (0 if no globals)
} PdfImage;

// PDF metadata
typedef struct {
  char *title;
  char *author;
  char *subject;
  char *keywords;
  char *creator;
  char *producer;
  char *creation_date;
  char *modification_date;
} PdfMetadata;

// Page information
typedef struct {
  float width;  // Page width in points (1/72 inch)
  float height; // Page height in points
  int rotation; // Rotation in degrees (0, 90, 180, 270)
} PdfPageInfo;

// ============================================================================
// Document Management
// ============================================================================

// Open a PDF document from file.
// Returns NULL on failure. Caller must close with pdf_close().
// Thread-safety: NOT thread-safe. Create one PdfDocument per thread.
PdfDocument *pdf_open(const char *path);

// Open a PDF document from memory buffer.
// The buffer must remain valid for the lifetime of the PdfDocument.
// Returns NULL on failure.
PdfDocument *pdf_open_memory(const uint8_t *data, size_t size);

// Close a PDF document and free all resources.
// Safe to call with NULL.
void pdf_close(PdfDocument *doc);

// Get the number of pages in the document.
// Returns -1 on error.
int pdf_page_count(PdfDocument *doc);

// Check if the document requires a password.
bool pdf_doc_needs_password(PdfDocument *doc);

// Authenticate with a password.
// Returns true if authentication succeeded.
bool pdf_doc_authenticate(PdfDocument *doc, const char *password);

// ============================================================================
// Page Information
// ============================================================================

// Get information about a page.
// page: 0-indexed page number.
// Returns false on error.
bool pdf_get_page_info(PdfDocument *doc, int page, PdfPageInfo *info);

// ============================================================================
// Image Extraction (Zero-Copy Path)
// ============================================================================

// Extract the raw image bytes from a page.
// This extracts the primary image content from the page in its native format,
// avoiding re-encoding. For scanned documents, each page typically contains
// a single full-page image.
//
// page: 0-indexed page number.
// image: Output structure filled with image data. Caller must free image->data.
//
// Returns true if an image was extracted, false if no suitable image found
// or the page uses vector graphics (use pdf_render_page() as fallback).
//
// Performance note: This is the zero-copy path. For JPEG/JP2/JBIG2 images,
// the raw compressed bytes are returned without decompression, enabling
// direct GPU decode via nvJPEG/nvImageCodec.
bool pdf_extract_page_image(PdfDocument *doc, int page, PdfImage *image);

// Free an extracted image's data.
// Safe to call with NULL or zero-initialized PdfImage.
void pdf_free_image(PdfImage *image);

// ============================================================================
// Page Rendering (Fallback Path)
// ============================================================================

// Render a page to RGB pixels at the specified DPI.
// This is the fallback path for pages that don't contain extractable images
// (vector graphics, text, multiple images, etc.).
//
// page: 0-indexed page number.
// dpi: Resolution in dots per inch (typical: 150-300).
// width, height: Output parameters for rendered dimensions.
// stride: Output parameter for row stride in bytes.
//
// Returns pointer to RGB24 pixel data (caller must free with free()).
// Returns NULL on error.
//
// Performance note: This path requires CPU rendering and is slower than
// image extraction, but handles all PDF content types.
uint8_t *pdf_render_page(PdfDocument *doc, int page, int dpi, int *width,
                         int *height, int *stride);

// Render a page to grayscale pixels at the specified DPI.
// Same as pdf_render_page but returns 8-bit grayscale.
uint8_t *pdf_render_page_gray(PdfDocument *doc, int page, int dpi, int *width,
                              int *height, int *stride);

// ============================================================================
// Metadata
// ============================================================================

// Get document metadata.
// Returns a PdfMetadata structure with string fields (may be NULL).
// Caller must free with pdf_free_metadata().
PdfMetadata pdf_get_metadata(PdfDocument *doc);

// Free metadata strings.
void pdf_free_metadata(PdfMetadata *meta);

// ============================================================================
// Utility Functions
// ============================================================================

// Get a human-readable name for an image format.
const char *pdf_image_format_name(PdfImageFormat format);

// Check if a file appears to be a PDF based on extension.
bool pdf_is_pdf_file(const char *filename);

// Get the last error message (thread-local).
const char *pdf_get_last_error(void);

#ifdef __cplusplus
}
#endif
