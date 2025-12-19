// SPDX-FileCopyrightText: 2025 The unpaper authors
//
// SPDX-License-Identifier: GPL-2.0-only

#include "src/core/sheet_pipeline.h"

bool sheet_pipeline_run(SheetProcessState *state,
                        const SheetProcessConfig *config) {
  return process_sheet(state, config);
}
