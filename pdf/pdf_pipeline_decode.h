// SPDX-FileCopyrightText: 2025 The unpaper authors
//
// SPDX-License-Identifier: GPL-2.0-only

#pragma once

#include "pdf_reader.h"

#include <stdbool.h>

#ifdef __cplusplus
extern "C" {
#endif

struct AVFrame;

// Decode a single PDF page to an AVFrame.
// Strategy:
// 1) Try to extract embedded image and decode it (JPEG/PNG/Flate/JBIG2)
// 2) Fallback: render the page to RGB pixels at dpi
//
// Returns an allocated AVFrame on success, NULL on failure.
struct AVFrame *pdf_pipeline_decode_page_to_frame(PdfDocument *doc,
                                                  int page_idx, int dpi);

// Compute the expected pixel size for a page at the given DPI, accounting for
// page rotation. Returns true on success.
bool pdf_pipeline_page_expected_size(PdfDocument *doc, int page_idx, int dpi,
                                     int *out_width, int *out_height);

// Decode an extracted PdfImage into an AVFrame.
// Returns an allocated AVFrame on success, NULL if format unsupported/fails.
struct AVFrame *pdf_pipeline_decode_image_to_frame(const PdfImage *pdf_img);

// Render a PDF page to an RGB24 AVFrame at dpi.
// Returns an allocated AVFrame on success, NULL on failure.
struct AVFrame *pdf_pipeline_render_page_to_frame(PdfDocument *doc,
                                                  int page_idx, int dpi);

#ifdef __cplusplus
}
#endif
