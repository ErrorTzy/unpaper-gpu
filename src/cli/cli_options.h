// SPDX-FileCopyrightText: 2025 The unpaper authors
//
// SPDX-License-Identifier: GPL-2.0-only

#pragma once

#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>

#include "constants.h"
#include "imageprocess/primitives.h"
#include "lib/options.h"

#ifdef __cplusplus
extern "C" {
#endif

typedef enum {
  CLI_PARSE_OK = 0,
  CLI_PARSE_EXIT = 1,
} CliParseStatus;

typedef struct {
  CliParseStatus status;
  int exit_code;
} CliParseResult;

typedef struct {
  Options options;
  bool device_explicit;
  int optind;

  size_t point_count;
  Point points[MAX_POINTS];

  size_t mask_count;
  Rectangle masks[MAX_MASKS];

  size_t pre_mask_count;
  Rectangle pre_masks[MAX_MASKS];

  int32_t middle_wipe[2];

  Rectangle blackfilter_exclude[MAX_MASKS];
  size_t blackfilter_exclude_count;
} OptionsResolved;

CliParseResult cli_options_parse(int argc, char **argv, OptionsResolved *out);

const char *cli_welcome_message(void);

bool cli_is_jpeg_filename(const char *filename);

#ifdef __cplusplus
}
#endif
