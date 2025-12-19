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

// Process a PDF file through the GPU pipeline.
// This is a thin wrapper over the batch PDF pipeline with CUDA enabled.
// It reuses the generic batch worker infrastructure (decode_queue +
// batch_process_parallel) with a PDF custom decoder that can return
// GPU-resident images and a PDF post-process that GPU-encodes into the page
// accumulator.
//
// input_path: Path to input PDF file
// output_path: Path to output PDF file
// options: Processing options (includes pdf_quality_mode, pdf_render_dpi)
// config: Sheet processing configuration
//
// Returns the number of failed pages (0 = success).
int pdf_pipeline_gpu_process(const char *input_path, const char *output_path,
                             const Options *options,
                             const SheetProcessConfig *config);

// Check if GPU PDF pipeline is available.
// Returns true if CUDA and nvImageCodec are initialized.
bool pdf_pipeline_gpu_available(void);

#ifdef __cplusplus
}
#endif
