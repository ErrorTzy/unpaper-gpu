// SPDX-FileCopyrightText: 2025 The unpaper authors
//
// SPDX-License-Identifier: GPL-2.0-only

#include "jbig2_decode.h"

#ifdef UNPAPER_WITH_JBIG2

#include <jbig2.h>
#include <stdarg.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

// Thread-local error message buffer
static __thread char last_error[512] = {0};

static void set_error(const char *fmt, ...) {
  va_list args;
  va_start(args, fmt);
  vsnprintf(last_error, sizeof(last_error), fmt, args);
  va_end(args);
}

const char *jbig2_get_last_error(void) { return last_error; }

bool jbig2_is_available(void) { return true; }

// Error callback for jbig2dec
static void jbig2_error_callback(void *data, const char *msg,
                                 Jbig2Severity severity, uint32_t seg_idx) {
  (void)data;
  if (severity >= JBIG2_SEVERITY_WARNING) {
    if (seg_idx != JBIG2_UNKNOWN_SEGMENT_NUMBER) {
      set_error("JBIG2 segment %u: %s", seg_idx, msg);
    } else {
      set_error("JBIG2: %s", msg);
    }
  }
}

bool jbig2_decode(const uint8_t *data, size_t size, const uint8_t *globals,
                  size_t globals_size, Jbig2DecodedImage *out) {
  if (data == NULL || size == 0 || out == NULL) {
    set_error("Invalid arguments");
    return false;
  }

  memset(out, 0, sizeof(*out));
  last_error[0] = '\0';

  Jbig2GlobalCtx *global_ctx = NULL;
  Jbig2Ctx *ctx = NULL;
  bool success = false;

  // If we have globals, create a global context first
  if (globals != NULL && globals_size > 0) {
    Jbig2Ctx *globals_ctx = jbig2_ctx_new(NULL, JBIG2_OPTIONS_EMBEDDED, NULL,
                                          jbig2_error_callback, NULL);
    if (globals_ctx == NULL) {
      set_error("Failed to create JBIG2 globals context");
      return false;
    }

    int ret = jbig2_data_in(globals_ctx, globals, globals_size);
    if (ret < 0) {
      set_error("Failed to parse JBIG2 globals");
      jbig2_ctx_free(globals_ctx);
      return false;
    }

    global_ctx = jbig2_make_global_ctx(globals_ctx);
    // Note: globals_ctx is consumed by jbig2_make_global_ctx
  }

  // Create main decoding context
  ctx = jbig2_ctx_new(NULL, JBIG2_OPTIONS_EMBEDDED, global_ctx,
                      jbig2_error_callback, NULL);
  if (ctx == NULL) {
    set_error("Failed to create JBIG2 context");
    if (global_ctx != NULL) {
      jbig2_global_ctx_free(global_ctx);
    }
    return false;
  }

  // Feed the data
  int ret = jbig2_data_in(ctx, data, size);
  if (ret < 0) {
    set_error("Failed to decode JBIG2 data");
    goto cleanup;
  }

  // Signal end of data and get the page
  jbig2_complete_page(ctx);

  Jbig2Image *image = jbig2_page_out(ctx);
  if (image == NULL) {
    set_error("No page decoded from JBIG2 data");
    goto cleanup;
  }

  // Copy the decoded image data
  size_t data_size = (size_t)image->stride * image->height;
  out->data = malloc(data_size);
  if (out->data == NULL) {
    set_error("Out of memory");
    jbig2_release_page(ctx, image);
    goto cleanup;
  }

  memcpy(out->data, image->data, data_size);
  out->width = image->width;
  out->height = image->height;
  out->stride = image->stride;

  jbig2_release_page(ctx, image);
  success = true;

cleanup:
  jbig2_ctx_free(ctx);
  if (global_ctx != NULL) {
    jbig2_global_ctx_free(global_ctx);
  }

  return success;
}

void jbig2_free_image(Jbig2DecodedImage *image) {
  if (image == NULL)
    return;
  free(image->data);
  memset(image, 0, sizeof(*image));
}

bool jbig2_expand_to_gray8(const Jbig2DecodedImage *jbig2, uint8_t *gray_out,
                           size_t gray_stride, bool invert) {
  if (jbig2 == NULL || jbig2->data == NULL || gray_out == NULL) {
    set_error("Invalid arguments");
    return false;
  }

  if (gray_stride < jbig2->width) {
    set_error("Output stride too small");
    return false;
  }

  // JBIG2 uses 1-bit packed pixels, MSB first
  // Value 0 typically means white (background), 1 means black (foreground)
  // For grayscale output: white=255, black=0 (unless inverted)
  const uint8_t white_val = invert ? 0 : 255;
  const uint8_t black_val = invert ? 255 : 0;

  for (uint32_t y = 0; y < jbig2->height; y++) {
    const uint8_t *src_row = jbig2->data + y * jbig2->stride;
    uint8_t *dst_row = gray_out + y * gray_stride;

    for (uint32_t x = 0; x < jbig2->width; x++) {
      uint32_t byte_idx = x >> 3;
      uint32_t bit_idx = 7 - (x & 7); // MSB first
      uint8_t bit = (src_row[byte_idx] >> bit_idx) & 1;
      dst_row[x] = bit ? black_val : white_val;
    }
  }

  return true;
}

#else // !UNPAPER_WITH_JBIG2

// Stub implementations when JBIG2 support is not compiled in

static __thread char last_error[512] = "JBIG2 support not compiled in";

const char *jbig2_get_last_error(void) { return last_error; }

bool jbig2_is_available(void) { return false; }

bool jbig2_decode(const uint8_t *data, size_t size, const uint8_t *globals,
                  size_t globals_size, Jbig2DecodedImage *out) {
  (void)data;
  (void)size;
  (void)globals;
  (void)globals_size;
  (void)out;
  return false;
}

void jbig2_free_image(Jbig2DecodedImage *image) {
  if (image == NULL)
    return;
  free(image->data);
  memset(image, 0, sizeof(*image));
}

bool jbig2_expand_to_gray8(const Jbig2DecodedImage *jbig2, uint8_t *gray_out,
                           size_t gray_stride, bool invert) {
  (void)jbig2;
  (void)gray_out;
  (void)gray_stride;
  (void)invert;
  return false;
}

#endif // UNPAPER_WITH_JBIG2
