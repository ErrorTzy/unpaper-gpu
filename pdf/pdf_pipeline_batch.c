// SPDX-FileCopyrightText: 2025 The unpaper authors
//
// SPDX-License-Identifier: GPL-2.0-only

#include "pdf_pipeline_batch.h"

#include "lib/logging.h"
#include "pdf_pipeline_cpu_batch.h"

void pdf_batch_config_init(PdfBatchConfig *config) {
  if (!config) {
    return;
  }
  config->parallelism = 0;
  config->decode_queue_depth = 0;
  config->progress = true;
  config->use_gpu = false;
}

bool pdf_pipeline_batch_available(void) { return true; }

int pdf_pipeline_batch_process(const char *input_path, const char *output_path,
                               const Options *options,
                               const SheetProcessConfig *sheet_config,
                               const PdfBatchConfig *batch_config) {
  if (!input_path || !output_path || !options || !sheet_config) {
    verboseLog(VERBOSE_NORMAL, "Batch PDF pipeline: invalid arguments\n");
    return -1;
  }

  PdfBatchConfig default_config;
  if (!batch_config) {
    pdf_batch_config_init(&default_config);
    batch_config = &default_config;
  }

  Options local_options = *options;
  local_options.device =
      batch_config->use_gpu ? UNPAPER_DEVICE_CUDA : UNPAPER_DEVICE_CPU;

  verboseLog(VERBOSE_NORMAL, "Using batch PDF pipeline (%s)\n",
             batch_config->use_gpu ? "GPU" : "CPU");

  return pdf_pipeline_cpu_process_batch(
      input_path, output_path, &local_options, sheet_config,
      batch_config->parallelism, batch_config->decode_queue_depth,
      batch_config->progress);
}
