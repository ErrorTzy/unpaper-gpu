// SPDX-FileCopyrightText: 2025 The unpaper authors
//
// SPDX-License-Identifier: GPL-2.0-only

#pragma once

#include "src/cli/cli_options.h"

#ifdef __cplusplus
extern "C" {
#endif

int image_pipeline_run(int argc, char **argv, OptionsResolved *resolved);

#ifdef __cplusplus
}
#endif
