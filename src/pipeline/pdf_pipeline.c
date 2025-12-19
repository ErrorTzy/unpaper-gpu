// SPDX-FileCopyrightText: 2025 The unpaper authors
//
// SPDX-License-Identifier: GPL-2.0-only

#include <stdbool.h>
#include <stdint.h>
#include <sys/stat.h>

#include "src/pipeline/pdf_pipeline.h"

#include "lib/batch.h"
#include "sheet_process.h"
#include "unpaper.h"

#include "pdf/pdf_pipeline_batch.h"
#include "pdf/pdf_pipeline_cpu.h"
#ifdef UNPAPER_WITH_CUDA
#include "pdf/pdf_pipeline_gpu.h"
#endif

bool pdf_pipeline_requested(const char *input_file, const char *output_file) {
  return input_file && output_file && pdf_pipeline_is_pdf(input_file) &&
         pdf_pipeline_is_pdf(output_file);
}

int pdf_pipeline_run(int argc, char **argv, int optind,
                     const OptionsResolved *resolved) {
  Options options = resolved->options;
  const char *input_file = argv[optind];
  const char *output_file = (optind + 1 < argc) ? argv[optind + 1] : NULL;

  // PDF mode supports only one input PDF and one output PDF filename.
  // Additional positional args would be ambiguous with the image-mode
  // "N input files / M output files" convention.
  if (argc != optind + 2) {
    errOutput("PDF mode requires exactly one input PDF and one output PDF.");
  }

  // Enforce limits implied by the batch worker structures.
  if (options.input_count < 1)
    options.input_count = 1;
  if (options.output_count < 1)
    options.output_count = 1;
  if (options.input_count > BATCH_MAX_FILES_PER_SHEET ||
      options.output_count > BATCH_MAX_FILES_PER_SHEET) {
    errOutput("PDF mode supports --input-pages/--output-pages up to %d.",
              BATCH_MAX_FILES_PER_SHEET);
  }

  // Check if input file exists
  struct stat statBuf;
  if (stat(input_file, &statBuf) != 0) {
    errOutput("unable to open PDF file %s.", input_file);
  }

  // Check for existing output file
  if (!options.overwrite_output) {
    if (stat(output_file, &statBuf) == 0) {
      errOutput("output file '%s' already present.\n", output_file);
    }
  }

  verboseLog(VERBOSE_NORMAL, "PDF mode: %s -> %s\n", input_file, output_file);

  // Set up sheet processing configuration
  Rectangle preMasks[MAX_MASKS];
  size_t preMaskCount = 0;
  Point points[MAX_POINTS];
  size_t pointCount = 0;
  int32_t middleWipe[2] = {0, 0};
  Rectangle blackfilterExclude[MAX_MASKS];

  SheetProcessConfig config;
  sheet_process_config_init(&config, &options, preMasks, preMaskCount, points,
                            pointCount, middleWipe, blackfilterExclude, 0);

  // Run PDF pipeline (batch, GPU, or CPU based on options)
  int failed;

  // Use batch pipeline if batch mode is enabled or parallelism is requested
  if (options.batch_mode || options.batch_jobs > 1) {
    PdfBatchConfig batch_config;
    pdf_batch_config_init(&batch_config);
    batch_config.parallelism = options.batch_jobs;
    batch_config.progress = options.batch_progress;
#ifdef UNPAPER_WITH_CUDA
    batch_config.use_gpu = (options.device == UNPAPER_DEVICE_CUDA) &&
                           pdf_pipeline_gpu_available();
#endif
    verboseLog(VERBOSE_NORMAL, "Using batch PDF pipeline (%s)\n",
               batch_config.use_gpu ? "GPU" : "CPU");
    failed = pdf_pipeline_batch_process(input_file, output_file, &options,
                                        &config, &batch_config);
  }
#ifdef UNPAPER_WITH_CUDA
  else if (options.device == UNPAPER_DEVICE_CUDA &&
           pdf_pipeline_gpu_available()) {
    verboseLog(VERBOSE_NORMAL, "Using GPU PDF pipeline\n");
    failed = pdf_pipeline_gpu_process(input_file, output_file, &options,
                                      &config);
  }
#endif
  else {
    verboseLog(VERBOSE_NORMAL, "Using CPU PDF pipeline\n");
    failed = pdf_pipeline_cpu_process(input_file, output_file, &options,
                                      &config);
  }

  return (failed > 0) ? 1 : 0;
}
