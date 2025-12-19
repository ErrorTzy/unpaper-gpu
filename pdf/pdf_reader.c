// SPDX-FileCopyrightText: 2025 The unpaper authors
//
// SPDX-License-Identifier: GPL-2.0-only

#include "pdf_reader.h"

#include <mupdf/fitz.h>
#include <mupdf/pdf.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <strings.h>

// Thread-local error message buffer
static __thread char last_error[512] = {0};

// Internal structure for PdfDocument
struct PdfDocument {
  fz_context *ctx;
  fz_document *doc;
  pdf_document *pdf; // NULL if not a PDF (could be other doc type)
  char *path;
  int page_count;
};

// Set last error message
static void set_error(const char *fmt, ...) {
  va_list args;
  va_start(args, fmt);
  vsnprintf(last_error, sizeof(last_error), fmt, args);
  va_end(args);
}

const char *pdf_get_last_error(void) { return last_error; }

const char *pdf_image_format_name(PdfImageFormat format) {
  switch (format) {
  case PDF_IMAGE_JPEG:
    return "JPEG";
  case PDF_IMAGE_JP2:
    return "JPEG2000";
  case PDF_IMAGE_JBIG2:
    return "JBIG2";
  case PDF_IMAGE_CCITT:
    return "CCITT";
  case PDF_IMAGE_PNG:
    return "PNG";
  case PDF_IMAGE_RAW:
    return "RAW";
  case PDF_IMAGE_FLATE:
    return "FLATE";
  case PDF_IMAGE_UNKNOWN:
  default:
    return "UNKNOWN";
  }
}

bool pdf_is_pdf_file(const char *filename) {
  if (filename == NULL)
    return false;

  size_t len = strlen(filename);
  if (len < 4)
    return false;

  const char *ext = filename + len - 4;
  return strcasecmp(ext, ".pdf") == 0;
}

PdfDocument *pdf_open(const char *path) {
  if (path == NULL) {
    set_error("NULL path");
    return NULL;
  }

  // Create context
  fz_context *ctx = fz_new_context(NULL, NULL, FZ_STORE_DEFAULT);
  if (ctx == NULL) {
    set_error("Failed to create MuPDF context");
    return NULL;
  }

  // Register document handlers
  fz_try(ctx) { fz_register_document_handlers(ctx); }
  fz_catch(ctx) {
    set_error("Failed to register document handlers: %s",
              fz_caught_message(ctx));
    fz_drop_context(ctx);
    return NULL;
  }

  // Open document
  fz_document *doc = NULL;
  fz_try(ctx) { doc = fz_open_document(ctx, path); }
  fz_catch(ctx) {
    set_error("Failed to open document: %s", fz_caught_message(ctx));
    fz_drop_context(ctx);
    return NULL;
  }

  // Count pages
  int page_count = 0;
  fz_try(ctx) { page_count = fz_count_pages(ctx, doc); }
  fz_catch(ctx) {
    set_error("Failed to count pages: %s", fz_caught_message(ctx));
    fz_drop_document(ctx, doc);
    fz_drop_context(ctx);
    return NULL;
  }

  // Allocate our wrapper
  PdfDocument *pdfdoc = calloc(1, sizeof(PdfDocument));
  if (pdfdoc == NULL) {
    set_error("Out of memory");
    fz_drop_document(ctx, doc);
    fz_drop_context(ctx);
    return NULL;
  }

  pdfdoc->ctx = ctx;
  pdfdoc->doc = doc;
  pdfdoc->page_count = page_count;
  pdfdoc->path = strdup(path);

  // Check if this is a PDF document (for PDF-specific operations)
  pdfdoc->pdf = pdf_document_from_fz_document(ctx, doc);

  return pdfdoc;
}

PdfDocument *pdf_open_memory(const uint8_t *data, size_t size) {
  if (data == NULL || size == 0) {
    set_error("Invalid data");
    return NULL;
  }

  fz_context *ctx = fz_new_context(NULL, NULL, FZ_STORE_DEFAULT);
  if (ctx == NULL) {
    set_error("Failed to create MuPDF context");
    return NULL;
  }

  fz_try(ctx) { fz_register_document_handlers(ctx); }
  fz_catch(ctx) {
    set_error("Failed to register document handlers: %s",
              fz_caught_message(ctx));
    fz_drop_context(ctx);
    return NULL;
  }

  // Create stream from memory
  fz_stream *stream = NULL;
  fz_document *doc = NULL;

  fz_try(ctx) {
    stream = fz_open_memory(ctx, data, size);
    doc = fz_open_document_with_stream(ctx, "application/pdf", stream);
  }
  fz_always(ctx) { fz_drop_stream(ctx, stream); }
  fz_catch(ctx) {
    set_error("Failed to open document from memory: %s",
              fz_caught_message(ctx));
    fz_drop_context(ctx);
    return NULL;
  }

  int page_count = 0;
  fz_try(ctx) { page_count = fz_count_pages(ctx, doc); }
  fz_catch(ctx) {
    set_error("Failed to count pages: %s", fz_caught_message(ctx));
    fz_drop_document(ctx, doc);
    fz_drop_context(ctx);
    return NULL;
  }

  PdfDocument *pdfdoc = calloc(1, sizeof(PdfDocument));
  if (pdfdoc == NULL) {
    set_error("Out of memory");
    fz_drop_document(ctx, doc);
    fz_drop_context(ctx);
    return NULL;
  }

  pdfdoc->ctx = ctx;
  pdfdoc->doc = doc;
  pdfdoc->page_count = page_count;
  pdfdoc->path = NULL;
  pdfdoc->pdf = pdf_document_from_fz_document(ctx, doc);

  return pdfdoc;
}

void pdf_close(PdfDocument *doc) {
  if (doc == NULL)
    return;

  if (doc->doc != NULL) {
    fz_drop_document(doc->ctx, doc->doc);
  }

  if (doc->ctx != NULL) {
    fz_drop_context(doc->ctx);
  }

  free(doc->path);
  free(doc);
}

int pdf_page_count(PdfDocument *doc) {
  if (doc == NULL)
    return -1;
  return doc->page_count;
}

bool pdf_doc_needs_password(PdfDocument *doc) {
  if (doc == NULL)
    return false;
  return fz_needs_password(doc->ctx, doc->doc);
}

bool pdf_doc_authenticate(PdfDocument *doc, const char *password) {
  if (doc == NULL)
    return false;
  return fz_authenticate_password(doc->ctx, doc->doc, password);
}

bool pdf_get_page_info(PdfDocument *doc, int page, PdfPageInfo *info) {
  if (doc == NULL || info == NULL || page < 0 || page >= doc->page_count) {
    set_error("Invalid arguments");
    return false;
  }

  memset(info, 0, sizeof(*info));

  fz_page *fzpage = NULL;
  fz_try(doc->ctx) {
    fzpage = fz_load_page(doc->ctx, doc->doc, page);
    fz_rect bounds = fz_bound_page(doc->ctx, fzpage);
    info->width = bounds.x1 - bounds.x0;
    info->height = bounds.y1 - bounds.y0;

    // Get rotation from PDF page object if available
    if (doc->pdf != NULL) {
      pdf_page *pdfpage = (pdf_page *)fzpage;
      pdf_obj *pageobj = pdfpage->obj;
      info->rotation =
          pdf_to_int(doc->ctx, pdf_dict_gets(doc->ctx, pageobj, "Rotate"));
    }
  }
  fz_always(doc->ctx) { fz_drop_page(doc->ctx, fzpage); }
  fz_catch(doc->ctx) {
    set_error("Failed to get page info: %s", fz_caught_message(doc->ctx));
    return false;
  }

  return true;
}

// Convert MuPDF image type to our PdfImageFormat
static PdfImageFormat convert_image_type(int type) {
  switch (type) {
  case FZ_IMAGE_JPEG:
    return PDF_IMAGE_JPEG;
  case FZ_IMAGE_JPX:
    return PDF_IMAGE_JP2;
  case FZ_IMAGE_JBIG2:
    return PDF_IMAGE_JBIG2;
  case FZ_IMAGE_FAX:
    return PDF_IMAGE_CCITT;
  case FZ_IMAGE_PNG:
    return PDF_IMAGE_PNG;
  case FZ_IMAGE_FLATE:
    return PDF_IMAGE_FLATE;
  case FZ_IMAGE_RAW:
    return PDF_IMAGE_RAW;
  default:
    return PDF_IMAGE_UNKNOWN;
  }
}

// Extract the first/largest image from page resources
// This walks the PDF XObject resources to find embedded images
static bool extract_image_from_resources(PdfDocument *doc, pdf_page *page,
                                         PdfImage *image) {
  fz_context *ctx = doc->ctx;
  pdf_obj *resources = NULL;
  pdf_obj *xobject = NULL;
  fz_image *best_image = NULL;
  int best_area = 0;

  fz_var(best_image);

  fz_try(ctx) {
    resources = pdf_page_resources(ctx, page);
    if (resources == NULL) {
      fz_throw(ctx, FZ_ERROR_GENERIC, "No resources on page");
    }

    xobject = pdf_dict_gets(ctx, resources, "XObject");
    if (xobject == NULL) {
      fz_throw(ctx, FZ_ERROR_GENERIC, "No XObject resources");
    }

    int n = pdf_dict_len(ctx, xobject);
    for (int i = 0; i < n; i++) {
      pdf_obj *ref = pdf_dict_get_val(ctx, xobject, i);
      pdf_obj *subtype = pdf_dict_gets(ctx, ref, "Subtype");

      // Check if this is an image
      if (pdf_name_eq(ctx, subtype, PDF_NAME(Image))) {
        // Load the image
        fz_image *img = pdf_load_image(ctx, doc->pdf, ref);
        int area = img->w * img->h;

        // Keep the largest image
        if (area > best_area) {
          if (best_image != NULL) {
            fz_drop_image(ctx, best_image);
          }
          best_image = img;
          best_area = area;
        } else {
          fz_drop_image(ctx, img);
        }
      }
    }

    if (best_image == NULL) {
      fz_throw(ctx, FZ_ERROR_GENERIC, "No images found in resources");
    }

    // Get the compressed buffer from the image
    fz_compressed_buffer *cbuf = fz_compressed_image_buffer(ctx, best_image);
    if (cbuf == NULL || cbuf->buffer == NULL) {
      fz_throw(ctx, FZ_ERROR_GENERIC, "Image not compressed or not accessible");
    }

    // Extract the raw bytes
    unsigned char *data = NULL;
    size_t size = fz_buffer_storage(ctx, cbuf->buffer, &data);
    if (size == 0 || data == NULL) {
      fz_throw(ctx, FZ_ERROR_GENERIC, "Empty buffer");
    }

    image->data = malloc(size);
    if (image->data == NULL) {
      fz_throw(ctx, FZ_ERROR_GENERIC, "Out of memory");
    }

    memcpy(image->data, data, size);
    image->size = size;
    image->width = best_image->w;
    image->height = best_image->h;
    image->components = best_image->n;
    image->bits_per_component = best_image->bpc;
    image->format = convert_image_type(cbuf->params.type);
    image->is_mask = best_image->imagemask;
    image->jbig2_globals = NULL;
    image->jbig2_globals_size = 0;

    // Extract JBIG2 globals if present
    if (cbuf->params.type == FZ_IMAGE_JBIG2 &&
        cbuf->params.u.jbig2.globals != NULL) {
      fz_buffer *globals_buf =
          fz_jbig2_globals_data(ctx, cbuf->params.u.jbig2.globals);
      if (globals_buf != NULL) {
        unsigned char *globals_data = NULL;
        size_t globals_size =
            fz_buffer_storage(ctx, globals_buf, &globals_data);
        if (globals_size > 0 && globals_data != NULL) {
          image->jbig2_globals = malloc(globals_size);
          if (image->jbig2_globals != NULL) {
            memcpy(image->jbig2_globals, globals_data, globals_size);
            image->jbig2_globals_size = globals_size;
          }
        }
      }
    }
  }
  fz_always(ctx) { fz_drop_image(ctx, best_image); }
  fz_catch(ctx) {
    set_error("Failed to extract image: %s", fz_caught_message(ctx));
    free(image->data);
    memset(image, 0, sizeof(*image));
    return false;
  }

  return true;
}

bool pdf_extract_page_image(PdfDocument *doc, int page, PdfImage *image) {
  if (doc == NULL || image == NULL || page < 0 || page >= doc->page_count) {
    set_error("Invalid arguments");
    return false;
  }

  memset(image, 0, sizeof(*image));

  // We need PDF-specific access for efficient image extraction
  if (doc->pdf == NULL) {
    set_error("Not a PDF document");
    return false;
  }

  fz_context *ctx = doc->ctx;
  pdf_page *pdfpage = NULL;
  bool success = false;

  fz_var(pdfpage);

  fz_try(ctx) {
    pdfpage = pdf_load_page(ctx, doc->pdf, page);
    success = extract_image_from_resources(doc, pdfpage, image);
  }
  fz_always(ctx) {
    if (pdfpage != NULL) {
      fz_drop_page(ctx, &pdfpage->super);
    }
  }
  fz_catch(ctx) {
    set_error("Failed to extract image: %s", fz_caught_message(ctx));
    return false;
  }

  return success;
}

void pdf_free_image(PdfImage *image) {
  if (image == NULL)
    return;
  free(image->data);
  free(image->jbig2_globals);
  memset(image, 0, sizeof(*image));
}

uint8_t *pdf_render_page(PdfDocument *doc, int page, int dpi, int *width,
                         int *height, int *stride) {
  if (doc == NULL || page < 0 || page >= doc->page_count) {
    set_error("Invalid arguments");
    return NULL;
  }

  fz_context *ctx = doc->ctx;
  fz_page *fzpage = NULL;
  fz_pixmap *pix = NULL;
  uint8_t *result = NULL;

  fz_var(fzpage);
  fz_var(pix);

  fz_try(ctx) {
    fzpage = fz_load_page(ctx, doc->doc, page);

    // Calculate transform for desired DPI
    float zoom = dpi / 72.0f;
    fz_matrix ctm = fz_scale(zoom, zoom);

    // Get page bounds and apply transform
    fz_rect bounds = fz_bound_page(ctx, fzpage);
    fz_irect bbox = fz_round_rect(fz_transform_rect(bounds, ctm));

    // Create RGB pixmap
    fz_colorspace *cs = fz_device_rgb(ctx);
    pix = fz_new_pixmap_with_bbox(ctx, cs, bbox, NULL, 0);
    fz_clear_pixmap_with_value(ctx, pix, 255); // White background

    // Render page
    fz_device *dev = fz_new_draw_device(ctx, ctm, pix);
    fz_run_page(ctx, fzpage, dev, fz_identity, NULL);
    fz_close_device(ctx, dev);
    fz_drop_device(ctx, dev);

    // Extract pixels
    int w = fz_pixmap_width(ctx, pix);
    int h = fz_pixmap_height(ctx, pix);
    int s = fz_pixmap_stride(ctx, pix);
    int n = fz_pixmap_components(ctx, pix);

    // Allocate output buffer (RGB24, no alpha)
    size_t out_stride = (size_t)w * 3;
    result = malloc(out_stride * (size_t)h);
    if (result == NULL) {
      fz_throw(ctx, FZ_ERROR_GENERIC, "Out of memory");
    }

    // Copy pixels, removing alpha if present
    unsigned char *src = fz_pixmap_samples(ctx, pix);
    for (int y = 0; y < h; y++) {
      unsigned char *src_row = src + (size_t)y * s;
      unsigned char *dst_row = result + y * out_stride;
      for (int x = 0; x < w; x++) {
        dst_row[x * 3 + 0] = src_row[x * n + 0];
        dst_row[x * 3 + 1] = src_row[x * n + 1];
        dst_row[x * 3 + 2] = src_row[x * n + 2];
      }
    }

    if (width)
      *width = w;
    if (height)
      *height = h;
    if (stride)
      *stride = (int)out_stride;
  }
  fz_always(ctx) {
    fz_drop_pixmap(ctx, pix);
    fz_drop_page(ctx, fzpage);
  }
  fz_catch(ctx) {
    set_error("Failed to render page: %s", fz_caught_message(ctx));
    free(result);
    return NULL;
  }

  return result;
}

uint8_t *pdf_render_page_gray(PdfDocument *doc, int page, int dpi, int *width,
                              int *height, int *stride) {
  if (doc == NULL || page < 0 || page >= doc->page_count) {
    set_error("Invalid arguments");
    return NULL;
  }

  fz_context *ctx = doc->ctx;
  fz_page *fzpage = NULL;
  fz_pixmap *pix = NULL;
  uint8_t *result = NULL;

  fz_var(fzpage);
  fz_var(pix);

  fz_try(ctx) {
    fzpage = fz_load_page(ctx, doc->doc, page);

    float zoom = dpi / 72.0f;
    fz_matrix ctm = fz_scale(zoom, zoom);

    fz_rect bounds = fz_bound_page(ctx, fzpage);
    fz_irect bbox = fz_round_rect(fz_transform_rect(bounds, ctm));

    // Create grayscale pixmap
    fz_colorspace *cs = fz_device_gray(ctx);
    pix = fz_new_pixmap_with_bbox(ctx, cs, bbox, NULL, 0);
    fz_clear_pixmap_with_value(ctx, pix, 255);

    fz_device *dev = fz_new_draw_device(ctx, ctm, pix);
    fz_run_page(ctx, fzpage, dev, fz_identity, NULL);
    fz_close_device(ctx, dev);
    fz_drop_device(ctx, dev);

    int w = fz_pixmap_width(ctx, pix);
    int h = fz_pixmap_height(ctx, pix);
    int s = fz_pixmap_stride(ctx, pix);

    result = malloc((size_t)w * (size_t)h);
    if (result == NULL) {
      fz_throw(ctx, FZ_ERROR_GENERIC, "Out of memory");
    }

    // Copy pixels
    unsigned char *src = fz_pixmap_samples(ctx, pix);
    for (int y = 0; y < h; y++) {
      memcpy(result + (size_t)y * w, src + (size_t)y * s, (size_t)w);
    }

    if (width)
      *width = w;
    if (height)
      *height = h;
    if (stride)
      *stride = w;
  }
  fz_always(ctx) {
    fz_drop_pixmap(ctx, pix);
    fz_drop_page(ctx, fzpage);
  }
  fz_catch(ctx) {
    set_error("Failed to render page: %s", fz_caught_message(ctx));
    free(result);
    return NULL;
  }

  return result;
}

static void compute_ctm_bbox_for_target(fz_context *ctx, fz_page *fzpage,
                                        int target_width, int target_height,
                                        fz_matrix *ctm_out, fz_irect *bbox_out) {
  fz_rect bounds = fz_bound_page(ctx, fzpage);
  float bw = bounds.x1 - bounds.x0;
  float bh = bounds.y1 - bounds.y0;

  // Guard against degenerate pages.
  if (bw <= 0.0f)
    bw = 1.0f;
  if (bh <= 0.0f)
    bh = 1.0f;

  float sx = (float)target_width / bw;
  float sy = (float)target_height / bh;

  fz_matrix ctm = fz_scale(sx, sy);
  fz_irect bbox = fz_round_rect(fz_transform_rect(bounds, ctm));

  // One adjustment pass to compensate for rounding.
  int w = bbox.x1 - bbox.x0;
  int h = bbox.y1 - bbox.y0;
  if (w > 0 && w != target_width) {
    sx *= (float)target_width / (float)w;
  }
  if (h > 0 && h != target_height) {
    sy *= (float)target_height / (float)h;
  }

  ctm = fz_scale(sx, sy);
  bbox = fz_round_rect(fz_transform_rect(bounds, ctm));

  *ctm_out = ctm;
  *bbox_out = bbox;
}

uint8_t *pdf_render_page_to_size(PdfDocument *doc, int page, int target_width,
                                 int target_height, int *width, int *height,
                                 int *stride) {
  if (doc == NULL || page < 0 || page >= doc->page_count || target_width <= 0 ||
      target_height <= 0) {
    set_error("Invalid arguments");
    return NULL;
  }

  fz_context *ctx = doc->ctx;
  fz_page *fzpage = NULL;
  fz_pixmap *pix = NULL;
  uint8_t *result = NULL;

  fz_var(fzpage);
  fz_var(pix);

  fz_try(ctx) {
    fzpage = fz_load_page(ctx, doc->doc, page);

    fz_matrix ctm;
    fz_irect bbox;
    compute_ctm_bbox_for_target(ctx, fzpage, target_width, target_height, &ctm,
                                &bbox);

    fz_colorspace *cs = fz_device_rgb(ctx);
    pix = fz_new_pixmap_with_bbox(ctx, cs, bbox, NULL, 0);
    fz_clear_pixmap_with_value(ctx, pix, 255);

    fz_device *dev = fz_new_draw_device(ctx, ctm, pix);
    fz_run_page(ctx, fzpage, dev, fz_identity, NULL);
    fz_close_device(ctx, dev);
    fz_drop_device(ctx, dev);

    int w = fz_pixmap_width(ctx, pix);
    int h = fz_pixmap_height(ctx, pix);
    int s = fz_pixmap_stride(ctx, pix);
    int n = fz_pixmap_components(ctx, pix);

    size_t out_stride = (size_t)w * 3;
    result = malloc(out_stride * (size_t)h);
    if (result == NULL) {
      fz_throw(ctx, FZ_ERROR_GENERIC, "Out of memory");
    }

    unsigned char *src = fz_pixmap_samples(ctx, pix);
    for (int y = 0; y < h; y++) {
      unsigned char *src_row = src + (size_t)y * s;
      unsigned char *dst_row = result + (size_t)y * out_stride;
      for (int x = 0; x < w; x++) {
        dst_row[x * 3 + 0] = src_row[x * n + 0];
        dst_row[x * 3 + 1] = src_row[x * n + 1];
        dst_row[x * 3 + 2] = src_row[x * n + 2];
      }
    }

    if (width)
      *width = w;
    if (height)
      *height = h;
    if (stride)
      *stride = (int)out_stride;
  }
  fz_always(ctx) {
    fz_drop_pixmap(ctx, pix);
    fz_drop_page(ctx, fzpage);
  }
  fz_catch(ctx) {
    set_error("Failed to render page: %s", fz_caught_message(ctx));
    free(result);
    return NULL;
  }

  return result;
}

uint8_t *pdf_render_page_gray_to_size(PdfDocument *doc, int page,
                                      int target_width, int target_height,
                                      int *width, int *height, int *stride) {
  if (doc == NULL || page < 0 || page >= doc->page_count || target_width <= 0 ||
      target_height <= 0) {
    set_error("Invalid arguments");
    return NULL;
  }

  fz_context *ctx = doc->ctx;
  fz_page *fzpage = NULL;
  fz_pixmap *pix = NULL;
  uint8_t *result = NULL;

  fz_var(fzpage);
  fz_var(pix);

  fz_try(ctx) {
    fzpage = fz_load_page(ctx, doc->doc, page);

    fz_matrix ctm;
    fz_irect bbox;
    compute_ctm_bbox_for_target(ctx, fzpage, target_width, target_height, &ctm,
                                &bbox);

    fz_colorspace *cs = fz_device_gray(ctx);
    pix = fz_new_pixmap_with_bbox(ctx, cs, bbox, NULL, 0);
    fz_clear_pixmap_with_value(ctx, pix, 255);

    fz_device *dev = fz_new_draw_device(ctx, ctm, pix);
    fz_run_page(ctx, fzpage, dev, fz_identity, NULL);
    fz_close_device(ctx, dev);
    fz_drop_device(ctx, dev);

    int w = fz_pixmap_width(ctx, pix);
    int h = fz_pixmap_height(ctx, pix);
    int s = fz_pixmap_stride(ctx, pix);

    result = malloc((size_t)w * (size_t)h);
    if (result == NULL) {
      fz_throw(ctx, FZ_ERROR_GENERIC, "Out of memory");
    }

    unsigned char *src = fz_pixmap_samples(ctx, pix);
    for (int y = 0; y < h; y++) {
      memcpy(result + (size_t)y * (size_t)w, src + (size_t)y * (size_t)s,
             (size_t)w);
    }

    if (width)
      *width = w;
    if (height)
      *height = h;
    if (stride)
      *stride = w;
  }
  fz_always(ctx) {
    fz_drop_pixmap(ctx, pix);
    fz_drop_page(ctx, fzpage);
  }
  fz_catch(ctx) {
    set_error("Failed to render page: %s", fz_caught_message(ctx));
    free(result);
    return NULL;
  }

  return result;
}

// Helper to extract a string from PDF info dict
static char *extract_info_string(fz_context *ctx, pdf_obj *info,
                                 const char *key) {
  pdf_obj *obj = pdf_dict_gets(ctx, info, key);
  if (obj == NULL)
    return NULL;

  const char *str = pdf_to_text_string(ctx, obj);
  if (str == NULL || str[0] == '\0')
    return NULL;

  return strdup(str);
}

PdfMetadata pdf_get_metadata(PdfDocument *doc) {
  PdfMetadata meta = {0};

  if (doc == NULL || doc->pdf == NULL)
    return meta;

  fz_context *ctx = doc->ctx;

  fz_try(ctx) {
    pdf_obj *info = pdf_dict_gets(ctx, pdf_trailer(ctx, doc->pdf), "Info");
    if (info != NULL) {
      meta.title = extract_info_string(ctx, info, "Title");
      meta.author = extract_info_string(ctx, info, "Author");
      meta.subject = extract_info_string(ctx, info, "Subject");
      meta.keywords = extract_info_string(ctx, info, "Keywords");
      meta.creator = extract_info_string(ctx, info, "Creator");
      meta.producer = extract_info_string(ctx, info, "Producer");
      meta.creation_date = extract_info_string(ctx, info, "CreationDate");
      meta.modification_date = extract_info_string(ctx, info, "ModDate");
    }
  }
  fz_catch(ctx) {
    // Ignore metadata errors
  }

  return meta;
}

void pdf_free_metadata(PdfMetadata *meta) {
  if (meta == NULL)
    return;

  free(meta->title);
  free(meta->author);
  free(meta->subject);
  free(meta->keywords);
  free(meta->creator);
  free(meta->producer);
  free(meta->creation_date);
  free(meta->modification_date);

  memset(meta, 0, sizeof(*meta));
}
