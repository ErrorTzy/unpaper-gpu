// SPDX-FileCopyrightText: 2025 The unpaper authors
//
// SPDX-License-Identifier: GPL-2.0-only

#pragma once

#include "src/cli/cli_options.h"

#ifdef __cplusplus
extern "C" {
#endif

#if defined(UNPAPER_WITH_PDF) && (UNPAPER_WITH_PDF)

bool pdf_pipeline_requested(const char *input_file, const char *output_file);
int pdf_pipeline_run(int argc, char **argv, int optind,
                     const OptionsResolved *resolved);

#endif

#ifdef __cplusplus
}
#endif
