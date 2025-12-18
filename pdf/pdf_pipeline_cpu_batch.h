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

// Internal helper: CPU PDF processing implemented as a wrapper around the
// generic batch pipeline (decode_queue + batch_process_parallel).
//
// parallelism/decode_queue_depth:
// - 0 means "auto" (parallelism) or "auto: parallelism * 2" (depth)
// progress:
// - controls batch-style progress output
//
// Returns number of failed pages (0 = success).
int pdf_pipeline_cpu_process_batch(const char *input_path,
                                   const char *output_path,
                                   const Options *options,
                                   const SheetProcessConfig *config,
                                   int parallelism, int decode_queue_depth,
                                   bool progress);

#ifdef __cplusplus
}
#endif

