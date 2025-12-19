// SPDX-FileCopyrightText: 2005 The unpaper authors
//
// SPDX-License-Identifier: GPL-2.0-only

#include <stdarg.h>
#include <stdbool.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "logging.h"

VerboseLevel verbose = VERBOSE_NONE;

static _Thread_local LogContext log_context = {0};
static _Thread_local bool log_line_start = true;

static bool log_context_active(void) {
  return log_context.has_job || log_context.sheet_nr > 0 ||
         (log_context.device && log_context.device[0] != '\0');
}

static void log_emit_prefix(FILE *stream) {
  if (!log_context_active()) {
    return;
  }

  fputc('[', stream);
  bool needs_space = false;

  if (log_context.has_job) {
    fprintf(stream, "job=%zu", log_context.job_index);
    needs_space = true;
  }

  if (log_context.sheet_nr > 0) {
    fprintf(stream, "%ssheet=%d", needs_space ? " " : "",
            log_context.sheet_nr);
    needs_space = true;
  }

  if (log_context.device && log_context.device[0] != '\0') {
    fprintf(stream, "%sdevice=%s", needs_space ? " " : "", log_context.device);
  }

  fputs("] ", stream);
}

static void log_write(FILE *stream, const char *message) {
  if (!message) {
    return;
  }

  for (const char *p = message; *p != '\0'; ++p) {
    if (log_line_start && log_context_active() && *p != '\n') {
      log_emit_prefix(stream);
    }

    fputc(*p, stream);

    if (*p == '\n') {
      log_line_start = true;
    } else {
      log_line_start = false;
    }
  }
}

static void log_vprint(FILE *stream, const char *fmt, va_list args) {
  char stack_buf[1024];
  va_list args_copy;
  va_copy(args_copy, args);
  int needed = vsnprintf(stack_buf, sizeof(stack_buf), fmt, args_copy);
  va_end(args_copy);

  if (needed < 0) {
    return;
  }

  if ((size_t)needed < sizeof(stack_buf)) {
    log_write(stream, stack_buf);
    return;
  }

  size_t buf_size = (size_t)needed + 1;
  char *heap_buf = malloc(buf_size);
  if (!heap_buf) {
    log_write(stream, stack_buf);
    return;
  }

  vsnprintf(heap_buf, buf_size, fmt, args);
  log_write(stream, heap_buf);
  free(heap_buf);
}

void log_context_set(const LogContext *ctx) {
  if (ctx) {
    log_context = *ctx;
  } else {
    memset(&log_context, 0, sizeof(log_context));
  }
}

void log_context_clear(void) {
  memset(&log_context, 0, sizeof(log_context));
}

void verboseLog(VerboseLevel level, const char *fmt, ...) {
  if (verbose < level)
    return;

  va_list vl;
  va_start(vl, fmt);
  log_vprint(stderr, fmt, vl);
  va_end(vl);
}

void logOutput(const char *fmt, ...) {
  va_list vl;
  va_start(vl, fmt);
  log_vprint(stderr, fmt, vl);
  va_end(vl);
}

/**
 * Print an error and exit process
 */
void errOutput(const char *fmt, ...) {
  va_list vl;

  fprintf(stderr, "unpaper: error: ");

  va_start(vl, fmt);
  vfprintf(stderr, fmt, vl);
  va_end(vl);

  fprintf(stderr, "\nTry 'man unpaper' for more information.\n");

  exit(1);
}
