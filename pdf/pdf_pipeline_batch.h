// SPDX-FileCopyrightText: 2025 The unpaper authors
//
// SPDX-License-Identifier: GPL-2.0-only

#pragma once

#include "lib/options.h"
#include "sheet_process.h"

#include <stdbool.h>

#ifdef __cplusplus
extern "C" {
#endif

// ============================================================================
// Batch PDF Pipeline
// ============================================================================
//
// Processes multi-page PDFs with parallel page processing:
// - Pre-fetch N pages into decode queue (hide I/O latency)
// - Worker pool processes pages in parallel
// - Page accumulator ensures sequential PDF write (pages in order)
// - Progress reporting: "Processing page 5/100..."
//
// Architecture:
//   PDF pages 0..N -> Decode Queue -> Worker Pool -> Page Accumulator -> PDF
//
// The key constraint is that PDF pages must be written in order, while
// processing can happen in parallel. The page accumulator collects encoded
// pages and writes them sequentially.

// ============================================================================
// Configuration
// ============================================================================

// Batch PDF pipeline configuration
typedef struct {
  int parallelism;        // Number of worker threads (0 = auto)
  int decode_queue_depth; // Decode queue depth (0 = auto: parallelism * 2)
  bool progress;          // Show progress output
  bool use_gpu;           // Use GPU pipeline (requires CUDA)
} PdfBatchConfig;

// Initialize batch config with defaults
void pdf_batch_config_init(PdfBatchConfig *config);

// ============================================================================
// Pipeline API
// ============================================================================

// Process a PDF file using the batch pipeline.
// This is the main entry point for batch PDF processing.
//
// input_path: Input PDF file path
// output_path: Output PDF file path
// options: Processing options
// config: Sheet processing configuration
// batch_config: Batch pipeline configuration (NULL for defaults)
//
// Returns 0 on success, or number of failed pages.
int pdf_pipeline_batch_process(const char *input_path, const char *output_path,
                               const Options *options,
                               const SheetProcessConfig *sheet_config,
                               const PdfBatchConfig *batch_config);

// Check if batch PDF pipeline is available.
// Returns true if the pipeline can be used.
bool pdf_pipeline_batch_available(void);

#ifdef __cplusplus
}
#endif
