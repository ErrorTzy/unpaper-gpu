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

// Process a PDF file through the CPU pipeline.
// Reads input PDF, processes each page, and writes to output PDF.
//
// input_path: Path to input PDF file
// output_path: Path to output PDF file
// options: Processing options
// config: Sheet processing configuration
//
// Returns the number of failed pages (0 = success).
int pdf_pipeline_cpu_process(const char *input_path, const char *output_path,
                             const Options *options,
                             const SheetProcessConfig *config);

// Check if a filename has a PDF extension.
// Convenience wrapper around pdf_is_pdf_file() for use in main.
bool pdf_pipeline_is_pdf(const char *filename);

#ifdef __cplusplus
}
#endif
