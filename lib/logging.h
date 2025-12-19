// SPDX-FileCopyrightText: 2005 The unpaper authors
//
// SPDX-License-Identifier: GPL-2.0-only

#pragma once

#include <stdbool.h>
#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef enum {
  VERBOSE_QUIET = -1,
  VERBOSE_NONE = 0,
  VERBOSE_NORMAL = 1,
  VERBOSE_MORE = 2,
  VERBOSE_DEBUG = 3,
  VERBOSE_DEBUG_SAVE = 4
} VerboseLevel;

// TODO: stop exposing the global variable.
extern VerboseLevel verbose;

typedef struct {
  bool has_job;
  size_t job_index;
  int sheet_nr;
  const char *device;
} LogContext;

// Set per-thread log context (job/sheet/device).
void log_context_set(const LogContext *ctx);
void log_context_clear(void);

void verboseLog(VerboseLevel level, const char *fmt, ...)
    __attribute__((format(printf, 2, 3)));
void logOutput(const char *fmt, ...) __attribute__((format(printf, 1, 2)));
void errOutput(const char *fmt, ...) __attribute__((format(printf, 1, 2)))
__attribute__((noreturn));

#ifdef __cplusplus
}
#endif
