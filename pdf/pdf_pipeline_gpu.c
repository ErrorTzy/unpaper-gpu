// SPDX-FileCopyrightText: 2025 The unpaper authors
//
// SPDX-License-Identifier: GPL-2.0-only

#include "pdf_pipeline_gpu.h"

#include "lib/logging.h"
#include "pdf_pipeline_cpu_batch.h"

#ifdef UNPAPER_WITH_CUDA
#include "imageprocess/cuda_runtime.h"
#endif

bool pdf_pipeline_gpu_available(void) {
#ifdef UNPAPER_WITH_CUDA
  return unpaper_cuda_try_init() == UNPAPER_CUDA_INIT_OK;
#else
  return false;
#endif
}

int pdf_pipeline_gpu_process(const char *input_path, const char *output_path,
                             const Options *options,
                             const SheetProcessConfig *config) {
#ifdef UNPAPER_WITH_CUDA
  if (!input_path || !output_path || !options || !config) {
    verboseLog(VERBOSE_NORMAL, "GPU PDF pipeline: invalid arguments\n");
    return -1;
  }

  Options local_options = *options;
  local_options.device = UNPAPER_DEVICE_CUDA;

  return pdf_pipeline_cpu_process_batch(input_path, output_path, &local_options,
                                        config, 0, 0, options->batch_progress);
#else
  (void)input_path;
  (void)output_path;
  (void)options;
  (void)config;
  verboseLog(VERBOSE_NORMAL, "GPU PDF pipeline: CUDA not compiled in\n");
  return -1;
#endif
}
