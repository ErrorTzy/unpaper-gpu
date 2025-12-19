// SPDX-FileCopyrightText: 2025 The unpaper authors
//
// SPDX-License-Identifier: GPL-2.0-only

#pragma once

#include "sheet_process.h"

#include <libavutil/pixfmt.h>

// Per-sheet stage context shared across the pipeline stages.
typedef struct {
  Layout layout;
  enum AVPixelFormat output_pixel_format;
  MaskDetectionParameters mask_params;
  BlackfilterParameters blackfilter_params;
} SheetStageContext;

void sheet_stage_context_init(SheetStageContext *ctx, SheetProcessState *state,
                              const SheetProcessConfig *config);

bool sheet_stages_run(SheetProcessState *state,
                      const SheetProcessConfig *config,
                      SheetStageContext *ctx);
