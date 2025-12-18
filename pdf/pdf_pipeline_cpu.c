// SPDX-FileCopyrightText: 2025 The unpaper authors
//
// SPDX-License-Identifier: GPL-2.0-only

#include "pdf_pipeline_cpu.h"

#include "pdf_pipeline_cpu_batch.h"
#include "pdf_reader.h"

bool pdf_pipeline_is_pdf(const char *filename) {
  return pdf_is_pdf_file(filename);
}

int pdf_pipeline_cpu_process(const char *input_path, const char *output_path,
                             const Options *options,
                             const SheetProcessConfig *config) {
  // Implement the CPU PDF pipeline via the generic batch infrastructure:
  // decode_queue (producer) + batch_process_parallel (workers).
  return pdf_pipeline_cpu_process_batch(input_path, output_path, options, config,
                                        1, 2, false);
}

