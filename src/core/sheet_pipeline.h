// SPDX-FileCopyrightText: 2025 The unpaper authors
//
// SPDX-License-Identifier: GPL-2.0-only

#pragma once

#include <stdbool.h>

#include "sheet_process.h"

#ifdef __cplusplus
extern "C" {
#endif

// Core pipeline entry point.
// Currently delegates to process_sheet() without behavior changes.
bool sheet_pipeline_run(SheetProcessState *state,
                        const SheetProcessConfig *config);

#ifdef __cplusplus
}
#endif
