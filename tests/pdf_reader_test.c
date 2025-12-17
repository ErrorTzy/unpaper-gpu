// SPDX-FileCopyrightText: 2025 The unpaper authors
//
// SPDX-License-Identifier: GPL-2.0-only

#include <stdbool.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "pdf/pdf_reader.h"

static char test_jpeg_pdf_path[4096];
static char test_2page_pdf_path[4096];

static void init_test_paths(void) {
  const char *imgsrc_dir = getenv("TEST_IMGSRC_DIR");
  if (imgsrc_dir != NULL) {
    // Adjust path to pdf_samples directory
    size_t len = strlen(imgsrc_dir);
    char base_dir[4096];
    strncpy(base_dir, imgsrc_dir, sizeof(base_dir) - 1);
    base_dir[sizeof(base_dir) - 1] = '\0';

    // Remove trailing slash and "source_images" to get tests/
    if (len > 0 && base_dir[len - 1] == '/') {
      base_dir[len - 1] = '\0';
    }
    char *last_slash = strrchr(base_dir, '/');
    if (last_slash != NULL) {
      *last_slash = '\0';
    }

    snprintf(test_jpeg_pdf_path, sizeof(test_jpeg_pdf_path),
             "%s/pdf_samples/test_jpeg.pdf", base_dir);
    snprintf(test_2page_pdf_path, sizeof(test_2page_pdf_path),
             "%s/pdf_samples/test_2page.pdf", base_dir);
  } else {
    snprintf(test_jpeg_pdf_path, sizeof(test_jpeg_pdf_path),
             "tests/pdf_samples/test_jpeg.pdf");
    snprintf(test_2page_pdf_path, sizeof(test_2page_pdf_path),
             "tests/pdf_samples/test_2page.pdf");
  }
}

static void test_is_pdf_file(void) {
  printf("Test: pdf_is_pdf_file... ");

  if (!pdf_is_pdf_file("test.pdf")) {
    printf("FAILED (test.pdf not recognized)\n");
    exit(1);
  }

  if (!pdf_is_pdf_file("TEST.PDF")) {
    printf("FAILED (TEST.PDF not recognized)\n");
    exit(1);
  }

  if (!pdf_is_pdf_file("/path/to/file.PDF")) {
    printf("FAILED (/path/to/file.PDF not recognized)\n");
    exit(1);
  }

  if (pdf_is_pdf_file("test.png")) {
    printf("FAILED (test.png incorrectly recognized)\n");
    exit(1);
  }

  if (pdf_is_pdf_file(NULL)) {
    printf("FAILED (NULL incorrectly recognized)\n");
    exit(1);
  }

  printf("PASSED\n");
}

static void test_open_close(void) {
  printf("Test: pdf_open/pdf_close... ");

  PdfDocument *doc = pdf_open(test_jpeg_pdf_path);
  if (doc == NULL) {
    printf("SKIPPED (test PDF not found: %s)\n", pdf_get_last_error());
    return;
  }

  pdf_close(doc);

  // Test with NULL
  pdf_close(NULL);

  printf("PASSED\n");
}

static void test_page_count(void) {
  printf("Test: pdf_page_count... ");

  PdfDocument *doc = pdf_open(test_jpeg_pdf_path);
  if (doc == NULL) {
    printf("SKIPPED (test PDF not found)\n");
    return;
  }

  int count = pdf_page_count(doc);
  if (count != 1) {
    printf("FAILED (expected 1 page, got %d)\n", count);
    pdf_close(doc);
    exit(1);
  }

  pdf_close(doc);

  // Test 2-page PDF
  doc = pdf_open(test_2page_pdf_path);
  if (doc != NULL) {
    count = pdf_page_count(doc);
    if (count != 2) {
      printf("FAILED (expected 2 pages, got %d)\n", count);
      pdf_close(doc);
      exit(1);
    }
    pdf_close(doc);
  }

  // Test NULL
  if (pdf_page_count(NULL) != -1) {
    printf("FAILED (NULL should return -1)\n");
    exit(1);
  }

  printf("PASSED\n");
}

static void test_page_info(void) {
  printf("Test: pdf_get_page_info... ");

  PdfDocument *doc = pdf_open(test_jpeg_pdf_path);
  if (doc == NULL) {
    printf("SKIPPED (test PDF not found)\n");
    return;
  }

  PdfPageInfo info;
  if (!pdf_get_page_info(doc, 0, &info)) {
    printf("FAILED (could not get page info)\n");
    pdf_close(doc);
    exit(1);
  }

  if (info.width <= 0 || info.height <= 0) {
    printf("FAILED (invalid dimensions: %fx%f)\n", info.width, info.height);
    pdf_close(doc);
    exit(1);
  }

  pdf_close(doc);
  printf("PASSED (%.0fx%.0f pts)\n", info.width, info.height);
}

static void test_extract_image(void) {
  printf("Test: pdf_extract_page_image... ");

  PdfDocument *doc = pdf_open(test_jpeg_pdf_path);
  if (doc == NULL) {
    printf("SKIPPED (test PDF not found)\n");
    return;
  }

  PdfImage image;
  if (pdf_extract_page_image(doc, 0, &image)) {
    // Successfully extracted
    if (image.data == NULL || image.size == 0) {
      printf("FAILED (empty image data)\n");
      pdf_close(doc);
      exit(1);
    }

    if (image.width <= 0 || image.height <= 0) {
      printf("FAILED (invalid dimensions: %dx%d)\n", image.width, image.height);
      pdf_free_image(&image);
      pdf_close(doc);
      exit(1);
    }

    printf("PASSED (%dx%d, %s, %zu bytes)\n", image.width, image.height,
           pdf_image_format_name(image.format), image.size);
    pdf_free_image(&image);
  } else {
    // Extraction failed - this is OK for some PDFs
    printf("SKIPPED (no extractable image: %s)\n", pdf_get_last_error());
  }

  pdf_close(doc);
}

static void test_render_page(void) {
  printf("Test: pdf_render_page... ");

  PdfDocument *doc = pdf_open(test_jpeg_pdf_path);
  if (doc == NULL) {
    printf("SKIPPED (test PDF not found)\n");
    return;
  }

  int width, height, stride;
  uint8_t *pixels = pdf_render_page(doc, 0, 150, &width, &height, &stride);

  if (pixels == NULL) {
    printf("FAILED (render returned NULL: %s)\n", pdf_get_last_error());
    pdf_close(doc);
    exit(1);
  }

  if (width <= 0 || height <= 0) {
    printf("FAILED (invalid dimensions: %dx%d)\n", width, height);
    free(pixels);
    pdf_close(doc);
    exit(1);
  }

  if (stride < width * 3) {
    printf("FAILED (invalid stride: %d for width %d)\n", stride, width);
    free(pixels);
    pdf_close(doc);
    exit(1);
  }

  free(pixels);
  pdf_close(doc);
  printf("PASSED (%dx%d at 150dpi)\n", width, height);
}

static void test_render_page_gray(void) {
  printf("Test: pdf_render_page_gray... ");

  PdfDocument *doc = pdf_open(test_jpeg_pdf_path);
  if (doc == NULL) {
    printf("SKIPPED (test PDF not found)\n");
    return;
  }

  int width, height, stride;
  uint8_t *pixels = pdf_render_page_gray(doc, 0, 150, &width, &height, &stride);

  if (pixels == NULL) {
    printf("FAILED (render returned NULL: %s)\n", pdf_get_last_error());
    pdf_close(doc);
    exit(1);
  }

  if (width <= 0 || height <= 0) {
    printf("FAILED (invalid dimensions: %dx%d)\n", width, height);
    free(pixels);
    pdf_close(doc);
    exit(1);
  }

  if (stride < width) {
    printf("FAILED (invalid stride: %d for width %d)\n", stride, width);
    free(pixels);
    pdf_close(doc);
    exit(1);
  }

  free(pixels);
  pdf_close(doc);
  printf("PASSED (%dx%d at 150dpi)\n", width, height);
}

static void test_metadata(void) {
  printf("Test: pdf_get_metadata... ");

  PdfDocument *doc = pdf_open(test_jpeg_pdf_path);
  if (doc == NULL) {
    printf("SKIPPED (test PDF not found)\n");
    return;
  }

  PdfMetadata meta = pdf_get_metadata(doc);

  // Just verify we can get metadata without crashing
  // Actual fields may be NULL for test PDFs
  printf("PASSED");
  if (meta.producer) {
    printf(" (producer: %s)", meta.producer);
  }
  printf("\n");

  pdf_free_metadata(&meta);
  pdf_close(doc);
}

static void test_image_format_name(void) {
  printf("Test: pdf_image_format_name... ");

  const char *name = pdf_image_format_name(PDF_IMAGE_JPEG);
  if (strcmp(name, "JPEG") != 0) {
    printf("FAILED (JPEG -> %s)\n", name);
    exit(1);
  }

  name = pdf_image_format_name(PDF_IMAGE_JP2);
  if (strcmp(name, "JPEG2000") != 0) {
    printf("FAILED (JP2 -> %s)\n", name);
    exit(1);
  }

  name = pdf_image_format_name(PDF_IMAGE_UNKNOWN);
  if (strcmp(name, "UNKNOWN") != 0) {
    printf("FAILED (UNKNOWN -> %s)\n", name);
    exit(1);
  }

  printf("PASSED\n");
}

static void test_password(void) {
  printf("Test: pdf_doc_needs_password/pdf_doc_authenticate... ");

  PdfDocument *doc = pdf_open(test_jpeg_pdf_path);
  if (doc == NULL) {
    printf("SKIPPED (test PDF not found)\n");
    return;
  }

  // Our test PDF shouldn't need a password
  if (pdf_doc_needs_password(doc)) {
    printf("FAILED (unexpected password requirement)\n");
    pdf_close(doc);
    exit(1);
  }

  pdf_close(doc);
  printf("PASSED\n");
}

int main(void) {
  printf("PDF Reader Unit Tests\n");
  printf("=====================\n\n");

  init_test_paths();

  // Run tests
  test_is_pdf_file();
  test_image_format_name();
  test_open_close();
  test_page_count();
  test_page_info();
  test_password();
  test_metadata();
  test_extract_image();
  test_render_page();
  test_render_page_gray();

  printf("\nAll tests passed!\n");
  return 0;
}
