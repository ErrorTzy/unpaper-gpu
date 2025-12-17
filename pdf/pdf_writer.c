// SPDX-FileCopyrightText: 2025 The unpaper authors
//
// SPDX-License-Identifier: GPL-2.0-only

#include "pdf_writer.h"

#include <mupdf/fitz.h>
#include <mupdf/pdf.h>
#include <stdarg.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

// Thread-local error message buffer (shared with pdf_reader.c via extern)
static __thread char writer_last_error[512] = {0};

// Internal structure for PdfWriter
struct PdfWriter {
  fz_context *ctx;
  pdf_document *doc;
  char *path;
  int default_dpi;
  int page_count;
  bool aborted;
};

// Set last error message
static void set_error(const char *fmt, ...) {
  va_list args;
  va_start(args, fmt);
  vsnprintf(writer_last_error, sizeof(writer_last_error), fmt, args);
  va_end(args);
}

const char *pdf_writer_get_last_error(void) { return writer_last_error; }

// Set metadata in the PDF Info dictionary
static void set_metadata(fz_context *ctx, pdf_document *doc,
                         const PdfMetadata *meta) {
  if (meta == NULL)
    return;

  fz_try(ctx) {
    pdf_obj *info = pdf_dict_get(ctx, pdf_trailer(ctx, doc), PDF_NAME(Info));
    if (info == NULL) {
      info = pdf_new_dict(ctx, doc, 8);
      pdf_dict_put_drop(ctx, pdf_trailer(ctx, doc), PDF_NAME(Info), info);
    }

    if (meta->title) {
      pdf_dict_put_text_string(ctx, info, PDF_NAME(Title), meta->title);
    }
    if (meta->author) {
      pdf_dict_put_text_string(ctx, info, PDF_NAME(Author), meta->author);
    }
    if (meta->subject) {
      pdf_dict_put_text_string(ctx, info, PDF_NAME(Subject), meta->subject);
    }
    if (meta->keywords) {
      pdf_dict_put_text_string(ctx, info, PDF_NAME(Keywords), meta->keywords);
    }
    if (meta->creator) {
      pdf_dict_put_text_string(ctx, info, PDF_NAME(Creator), meta->creator);
    }

    // Set producer to indicate this was created by unpaper
    pdf_dict_put_text_string(ctx, info, PDF_NAME(Producer), "unpaper");

    // Preserve original dates if provided
    if (meta->creation_date) {
      pdf_dict_put_text_string(ctx, info, PDF_NAME(CreationDate),
                               meta->creation_date);
    }
    if (meta->modification_date) {
      pdf_dict_put_text_string(ctx, info, PDF_NAME(ModDate),
                               meta->modification_date);
    }
  }
  fz_catch(ctx) {
    // Ignore metadata errors - they're not critical
  }
}

PdfWriter *pdf_writer_create(const char *path, const PdfMetadata *meta,
                             int dpi) {
  if (path == NULL) {
    set_error("NULL path");
    return NULL;
  }

  if (dpi <= 0) {
    dpi = 72; // Default to 72 DPI (1 point = 1 pixel)
  }

  // Create context
  fz_context *ctx = fz_new_context(NULL, NULL, FZ_STORE_DEFAULT);
  if (ctx == NULL) {
    set_error("Failed to create MuPDF context");
    return NULL;
  }

  // Create a new empty PDF document
  pdf_document *doc = NULL;
  fz_try(ctx) { doc = pdf_create_document(ctx); }
  fz_catch(ctx) {
    set_error("Failed to create PDF document: %s", fz_caught_message(ctx));
    fz_drop_context(ctx);
    return NULL;
  }

  // Allocate our wrapper
  PdfWriter *writer = calloc(1, sizeof(PdfWriter));
  if (writer == NULL) {
    set_error("Out of memory");
    pdf_drop_document(ctx, doc);
    fz_drop_context(ctx);
    return NULL;
  }

  writer->ctx = ctx;
  writer->doc = doc;
  writer->path = strdup(path);
  writer->default_dpi = dpi;
  writer->page_count = 0;
  writer->aborted = false;

  if (writer->path == NULL) {
    set_error("Out of memory");
    pdf_drop_document(ctx, doc);
    fz_drop_context(ctx);
    free(writer);
    return NULL;
  }

  // Set metadata
  set_metadata(ctx, doc, meta);

  return writer;
}

// Create an image XObject from raw compressed data (JPEG or JP2)
// This is the zero-copy path - data is embedded directly without re-encoding
static pdf_obj *create_image_xobject(fz_context *ctx, pdf_document *doc,
                                     const uint8_t *data, size_t len, int width,
                                     int height, bool is_jpeg, int components) {
  pdf_obj *image_obj = NULL;
  fz_buffer *buf = NULL;

  fz_var(image_obj);
  fz_var(buf);

  fz_try(ctx) {
    // Create buffer from data (MuPDF takes ownership when we add the stream)
    buf = fz_new_buffer_from_copied_data(ctx, data, len);

    // Create the image dictionary
    image_obj = pdf_new_dict(ctx, doc, 10);

    pdf_dict_put(ctx, image_obj, PDF_NAME(Type), PDF_NAME(XObject));
    pdf_dict_put(ctx, image_obj, PDF_NAME(Subtype), PDF_NAME(Image));
    pdf_dict_put_int(ctx, image_obj, PDF_NAME(Width), width);
    pdf_dict_put_int(ctx, image_obj, PDF_NAME(Height), height);
    pdf_dict_put_int(ctx, image_obj, PDF_NAME(BitsPerComponent), 8);

    // Set color space based on components
    if (components == 1) {
      pdf_dict_put(ctx, image_obj, PDF_NAME(ColorSpace), PDF_NAME(DeviceGray));
    } else if (components == 3) {
      pdf_dict_put(ctx, image_obj, PDF_NAME(ColorSpace), PDF_NAME(DeviceRGB));
    } else if (components == 4) {
      pdf_dict_put(ctx, image_obj, PDF_NAME(ColorSpace), PDF_NAME(DeviceCMYK));
    }

    // Set the appropriate filter
    if (is_jpeg) {
      pdf_dict_put(ctx, image_obj, PDF_NAME(Filter), PDF_NAME(DCTDecode));
    } else {
      // JP2 uses JPXDecode
      pdf_dict_put(ctx, image_obj, PDF_NAME(Filter), PDF_NAME(JPXDecode));
    }

    // Add the image as an indirect object with its stream data
    image_obj = pdf_add_object(ctx, doc, image_obj);
    pdf_update_stream(ctx, doc, image_obj, buf, 1); // 1 = raw (already encoded)
  }
  fz_always(ctx) { fz_drop_buffer(ctx, buf); }
  fz_catch(ctx) {
    pdf_drop_obj(ctx, image_obj);
    fz_rethrow(ctx);
  }

  return image_obj;
}

// Detect number of components from JPEG header
static int detect_jpeg_components(const uint8_t *data, size_t len) {
  // Look for SOF0/SOF1/SOF2 markers (Start Of Frame)
  // Format: FF C0/C1/C2 length(2) precision(1) height(2) width(2) components(1)
  for (size_t i = 0; i + 10 < len; i++) {
    if (data[i] == 0xFF) {
      uint8_t marker = data[i + 1];
      // SOF0 = C0, SOF1 = C1, SOF2 = C2 (progressive)
      if (marker == 0xC0 || marker == 0xC1 || marker == 0xC2) {
        // Components byte is at offset 9 from marker
        return data[i + 9];
      }
      // Skip variable-length markers
      if (marker >= 0xC0 && marker <= 0xFE && marker != 0xD0 &&
          marker != 0xD1 && marker != 0xD2 && marker != 0xD3 &&
          marker != 0xD4 && marker != 0xD5 && marker != 0xD6 &&
          marker != 0xD7 && marker != 0xD8 && marker != 0xD9) {
        if (i + 3 < len) {
          size_t seg_len = ((size_t)data[i + 2] << 8) | data[i + 3];
          i += seg_len + 1; // +1 because loop will ++i
        }
      }
    }
  }
  return 3; // Default to RGB
}

// Add a page with an image XObject
static bool add_page_with_image(PdfWriter *writer, pdf_obj *image_obj,
                                int width, int height, int dpi) {
  fz_context *ctx = writer->ctx;
  pdf_document *doc = writer->doc;
  int effective_dpi = (dpi > 0) ? dpi : writer->default_dpi;

  // Calculate page size in points (72 points = 1 inch)
  float page_width = (float)width * 72.0f / (float)effective_dpi;
  float page_height = (float)height * 72.0f / (float)effective_dpi;
  fz_rect mediabox = fz_make_rect(0, 0, page_width, page_height);

  fz_buffer *contents_buf = NULL;
  pdf_obj *resources = NULL;
  pdf_obj *xobjects = NULL;
  pdf_obj *page_obj = NULL;

  fz_var(contents_buf);
  fz_var(resources);
  fz_var(xobjects);
  fz_var(page_obj);

  fz_try(ctx) {
    // Create page content stream that draws the image
    // Format: q (save state) w 0 0 h 0 0 cm (transform) /Im0 Do (draw) Q
    // (restore) The cm matrix scales the 1x1 unit square to page_width x
    // page_height
    contents_buf = fz_new_buffer(ctx, 64);
    fz_append_printf(ctx, contents_buf, "q %g 0 0 %g 0 0 cm /Im0 Do Q",
                     page_width, page_height);

    // Create resources with the image
    resources = pdf_new_dict(ctx, doc, 2);
    xobjects = pdf_new_dict(ctx, doc, 1);
    // Use pdf_dict_puts to add the XObject reference with a string key
    // This ensures the indirect reference is properly serialized
    pdf_dict_puts(ctx, xobjects, "Im0", image_obj);
    pdf_dict_put(ctx, resources, PDF_NAME(XObject), xobjects);

    // Create the page
    page_obj = pdf_add_page(ctx, doc, mediabox, 0, resources, contents_buf);
    pdf_insert_page(ctx, doc, -1, page_obj); // -1 = append at end

    writer->page_count++;
  }
  fz_always(ctx) {
    fz_drop_buffer(ctx, contents_buf);
    pdf_drop_obj(ctx, resources);
    pdf_drop_obj(ctx, page_obj);
  }
  fz_catch(ctx) {
    set_error("Failed to add page: %s", fz_caught_message(ctx));
    return false;
  }

  return true;
}

bool pdf_writer_add_page_jpeg(PdfWriter *writer, const uint8_t *data,
                              size_t len, int width, int height, int dpi) {
  if (writer == NULL || data == NULL || len == 0) {
    set_error("Invalid arguments");
    return false;
  }

  if (width <= 0 || height <= 0) {
    set_error("Invalid dimensions: %dx%d", width, height);
    return false;
  }

  if (writer->aborted) {
    set_error("Writer has been aborted");
    return false;
  }

  fz_context *ctx = writer->ctx;
  pdf_obj *image_obj = NULL;
  bool result = false;

  // Detect components from JPEG header
  int components = detect_jpeg_components(data, len);

  fz_try(ctx) {
    image_obj = create_image_xobject(ctx, writer->doc, data, len, width, height,
                                     true, components);
    result = add_page_with_image(writer, image_obj, width, height, dpi);
  }
  fz_always(ctx) { pdf_drop_obj(ctx, image_obj); }
  fz_catch(ctx) {
    set_error("Failed to add JPEG page: %s", fz_caught_message(ctx));
    return false;
  }

  return result;
}

bool pdf_writer_add_page_jp2(PdfWriter *writer, const uint8_t *data, size_t len,
                             int width, int height, int dpi) {
  if (writer == NULL || data == NULL || len == 0) {
    set_error("Invalid arguments");
    return false;
  }

  if (width <= 0 || height <= 0) {
    set_error("Invalid dimensions: %dx%d", width, height);
    return false;
  }

  if (writer->aborted) {
    set_error("Writer has been aborted");
    return false;
  }

  fz_context *ctx = writer->ctx;
  pdf_obj *image_obj = NULL;
  bool result = false;

  // JP2 typically uses 3 components (RGB), but we could detect from header
  // For now, assume RGB
  int components = 3;

  fz_try(ctx) {
    image_obj = create_image_xobject(ctx, writer->doc, data, len, width, height,
                                     false, components);
    result = add_page_with_image(writer, image_obj, width, height, dpi);
  }
  fz_always(ctx) { pdf_drop_obj(ctx, image_obj); }
  fz_catch(ctx) {
    set_error("Failed to add JP2 page: %s", fz_caught_message(ctx));
    return false;
  }

  return result;
}

bool pdf_writer_add_page_pixels(PdfWriter *writer, const uint8_t *pixels,
                                int width, int height, int stride,
                                PdfPixelFormat format, int dpi) {
  if (writer == NULL || pixels == NULL) {
    set_error("Invalid arguments");
    return false;
  }

  if (width <= 0 || height <= 0) {
    set_error("Invalid dimensions: %dx%d", width, height);
    return false;
  }

  if (writer->aborted) {
    set_error("Writer has been aborted");
    return false;
  }

  int components = (format == PDF_PIXEL_GRAY8) ? 1 : 3;
  int min_stride = width * components;
  if (stride < min_stride) {
    set_error("Invalid stride: %d (minimum %d)", stride, min_stride);
    return false;
  }

  fz_context *ctx = writer->ctx;
  pdf_obj *image_obj = NULL;
  fz_buffer *raw_buf = NULL;
  bool result = false;

  fz_var(image_obj);
  fz_var(raw_buf);

  fz_try(ctx) {
    // Create buffer for pixel data (packed, no stride padding)
    size_t row_size = (size_t)width * components;
    size_t total_size = row_size * (size_t)height;
    raw_buf = fz_new_buffer(ctx, total_size);

    // Copy pixels, removing any stride padding
    for (int y = 0; y < height; y++) {
      fz_append_data(ctx, raw_buf, pixels + (size_t)y * stride, row_size);
    }

    // Create image dictionary with Flate compression
    image_obj = pdf_new_dict(ctx, writer->doc, 10);

    pdf_dict_put(ctx, image_obj, PDF_NAME(Type), PDF_NAME(XObject));
    pdf_dict_put(ctx, image_obj, PDF_NAME(Subtype), PDF_NAME(Image));
    pdf_dict_put_int(ctx, image_obj, PDF_NAME(Width), width);
    pdf_dict_put_int(ctx, image_obj, PDF_NAME(Height), height);
    pdf_dict_put_int(ctx, image_obj, PDF_NAME(BitsPerComponent), 8);

    if (components == 1) {
      pdf_dict_put(ctx, image_obj, PDF_NAME(ColorSpace), PDF_NAME(DeviceGray));
    } else {
      pdf_dict_put(ctx, image_obj, PDF_NAME(ColorSpace), PDF_NAME(DeviceRGB));
    }

    // Add as object and attach stream (MuPDF will compress with Flate)
    image_obj = pdf_add_object(ctx, writer->doc, image_obj);
    pdf_update_stream(ctx, writer->doc, image_obj, raw_buf,
                      0); // 0 = compress with Flate

    result = add_page_with_image(writer, image_obj, width, height, dpi);
  }
  fz_always(ctx) {
    fz_drop_buffer(ctx, raw_buf);
    pdf_drop_obj(ctx, image_obj);
  }
  fz_catch(ctx) {
    set_error("Failed to add pixel page: %s", fz_caught_message(ctx));
    return false;
  }

  return result;
}

int pdf_writer_page_count(const PdfWriter *writer) {
  if (writer == NULL)
    return 0;
  return writer->page_count;
}

bool pdf_writer_close(PdfWriter *writer) {
  if (writer == NULL)
    return true;

  if (writer->aborted) {
    // Already aborted, just clean up
    pdf_drop_document(writer->ctx, writer->doc);
    fz_drop_context(writer->ctx);
    free(writer->path);
    free(writer);
    return false;
  }

  fz_context *ctx = writer->ctx;
  bool success = false;

  fz_try(ctx) {
    // Save the document
    pdf_save_document(ctx, writer->doc, writer->path, NULL);
    success = true;
  }
  fz_catch(ctx) { set_error("Failed to save PDF: %s", fz_caught_message(ctx)); }

  // Clean up
  pdf_drop_document(ctx, writer->doc);
  fz_drop_context(ctx);
  free(writer->path);
  free(writer);

  return success;
}

void pdf_writer_abort(PdfWriter *writer) {
  if (writer == NULL)
    return;

  writer->aborted = true;

  // Clean up without saving
  pdf_drop_document(writer->ctx, writer->doc);
  fz_drop_context(writer->ctx);
  free(writer->path);
  free(writer);
}
