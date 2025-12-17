// SPDX-FileCopyrightText: 2025 The unpaper authors
//
// SPDX-License-Identifier: GPL-2.0-only

#include <stdbool.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "lib/jbig2_decode.h"

#ifdef UNPAPER_WITH_PDF
#include "pdf/pdf_reader.h"
#endif

static char test_jbig2_pdf_path[4096];

static void init_test_paths(void) {
  const char *imgsrc_dir = getenv("TEST_IMGSRC_DIR");
  if (imgsrc_dir != NULL) {
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

    snprintf(test_jbig2_pdf_path, sizeof(test_jbig2_pdf_path),
             "%s/pdf_samples/test_jbig2.pdf", base_dir);
  } else {
    snprintf(test_jbig2_pdf_path, sizeof(test_jbig2_pdf_path),
             "tests/pdf_samples/test_jbig2.pdf");
  }
}

static void test_is_available(void) {
  printf("Test: jbig2_is_available... ");

#ifdef UNPAPER_WITH_JBIG2
  if (!jbig2_is_available()) {
    printf("FAILED (should be available)\n");
    exit(1);
  }
  printf("PASSED (available)\n");
#else
  if (jbig2_is_available()) {
    printf("FAILED (should not be available)\n");
    exit(1);
  }
  printf("PASSED (not available, as expected)\n");
#endif
}

static void test_null_safety(void) {
  printf("Test: null safety... ");

  // Test decode with NULL
  Jbig2DecodedImage img = {0};
  bool result = jbig2_decode(NULL, 0, NULL, 0, &img);
  if (result) {
    printf("FAILED (decode with NULL data should fail)\n");
    exit(1);
  }

  // Test expand with NULL
  result = jbig2_expand_to_gray8(NULL, NULL, 0, false);
  if (result) {
    printf("FAILED (expand with NULL should fail)\n");
    exit(1);
  }

  // Test free with NULL (should not crash)
  jbig2_free_image(NULL);
  jbig2_free_image(&img);

  printf("PASSED\n");
}

#if defined(UNPAPER_WITH_JBIG2) && defined(UNPAPER_WITH_PDF)
static void test_decode_from_pdf(void) {
  printf("Test: decode JBIG2 from PDF... ");

  // Open the test PDF
  PdfDocument *doc = pdf_open(test_jbig2_pdf_path);
  if (doc == NULL) {
    printf("SKIPPED (test PDF not found: %s)\n", pdf_get_last_error());
    return;
  }

  // Extract the JBIG2 image
  PdfImage pdf_img = {0};
  if (!pdf_extract_page_image(doc, 0, &pdf_img)) {
    printf("SKIPPED (could not extract image: %s)\n", pdf_get_last_error());
    pdf_close(doc);
    return;
  }

  if (pdf_img.format != PDF_IMAGE_JBIG2) {
    printf("SKIPPED (image is not JBIG2: %s)\n",
           pdf_image_format_name(pdf_img.format));
    pdf_free_image(&pdf_img);
    pdf_close(doc);
    return;
  }

  // Decode the JBIG2 data
  Jbig2DecodedImage jbig2_img = {0};
  if (!jbig2_decode(pdf_img.data, pdf_img.size, pdf_img.jbig2_globals,
                    pdf_img.jbig2_globals_size, &jbig2_img)) {
    printf("FAILED (decode failed: %s)\n", jbig2_get_last_error());
    pdf_free_image(&pdf_img);
    pdf_close(doc);
    exit(1);
  }

  // Verify dimensions (our test image is 200x100)
  if (jbig2_img.width != 200 || jbig2_img.height != 100) {
    printf("FAILED (unexpected dimensions: %ux%u, expected 200x100)\n",
           jbig2_img.width, jbig2_img.height);
    jbig2_free_image(&jbig2_img);
    pdf_free_image(&pdf_img);
    pdf_close(doc);
    exit(1);
  }

  // Verify data is not NULL
  if (jbig2_img.data == NULL || jbig2_img.stride == 0) {
    printf("FAILED (invalid output data)\n");
    jbig2_free_image(&jbig2_img);
    pdf_free_image(&pdf_img);
    pdf_close(doc);
    exit(1);
  }

  printf("PASSED (%ux%u, stride=%u)\n", jbig2_img.width, jbig2_img.height,
         jbig2_img.stride);

  jbig2_free_image(&jbig2_img);
  pdf_free_image(&pdf_img);
  pdf_close(doc);
}

static void test_expand_to_gray8(void) {
  printf("Test: expand JBIG2 to grayscale... ");

  // Open the test PDF
  PdfDocument *doc = pdf_open(test_jbig2_pdf_path);
  if (doc == NULL) {
    printf("SKIPPED (test PDF not found)\n");
    return;
  }

  // Extract the JBIG2 image
  PdfImage pdf_img = {0};
  if (!pdf_extract_page_image(doc, 0, &pdf_img) ||
      pdf_img.format != PDF_IMAGE_JBIG2) {
    printf("SKIPPED (could not extract JBIG2 image)\n");
    pdf_free_image(&pdf_img);
    pdf_close(doc);
    return;
  }

  // Decode
  Jbig2DecodedImage jbig2_img = {0};
  if (!jbig2_decode(pdf_img.data, pdf_img.size, pdf_img.jbig2_globals,
                    pdf_img.jbig2_globals_size, &jbig2_img)) {
    printf("FAILED (decode failed)\n");
    pdf_free_image(&pdf_img);
    pdf_close(doc);
    exit(1);
  }

  // Allocate grayscale buffer
  size_t gray_stride = jbig2_img.width;
  uint8_t *gray = malloc(gray_stride * jbig2_img.height);
  if (gray == NULL) {
    printf("FAILED (out of memory)\n");
    jbig2_free_image(&jbig2_img);
    pdf_free_image(&pdf_img);
    pdf_close(doc);
    exit(1);
  }

  // Expand to grayscale
  if (!jbig2_expand_to_gray8(&jbig2_img, gray, gray_stride, true)) {
    printf("FAILED (expand failed: %s)\n", jbig2_get_last_error());
    free(gray);
    jbig2_free_image(&jbig2_img);
    pdf_free_image(&pdf_img);
    pdf_close(doc);
    exit(1);
  }

  // Verify grayscale values are either 0 or 255 (B&W)
  int white_count = 0, black_count = 0;
  for (size_t i = 0; i < gray_stride * jbig2_img.height; i++) {
    if (gray[i] == 255) {
      white_count++;
    } else if (gray[i] == 0) {
      black_count++;
    }
  }

  // Our test image has a black rectangle on white background
  if (white_count == 0 || black_count == 0) {
    printf("FAILED (unexpected pixel values: white=%d, black=%d)\n",
           white_count, black_count);
    free(gray);
    jbig2_free_image(&jbig2_img);
    pdf_free_image(&pdf_img);
    pdf_close(doc);
    exit(1);
  }

  printf("PASSED (white=%d, black=%d pixels)\n", white_count, black_count);

  free(gray);
  jbig2_free_image(&jbig2_img);
  pdf_free_image(&pdf_img);
  pdf_close(doc);
}
#endif // UNPAPER_WITH_JBIG2 && UNPAPER_WITH_PDF

int main(void) {
  printf("JBIG2 Decode Unit Tests\n");
  printf("=======================\n\n");

  init_test_paths();

  // Basic tests that work with or without JBIG2 support
  test_is_available();
  test_null_safety();

#if defined(UNPAPER_WITH_JBIG2) && defined(UNPAPER_WITH_PDF)
  // Tests that require both JBIG2 and PDF support
  test_decode_from_pdf();
  test_expand_to_gray8();
#else
  printf("\nNote: JBIG2+PDF integration tests skipped (requires both "
         "-Djbig2=enabled and -Dpdf=enabled)\n");
#endif

  printf("\nAll tests passed!\n");
  return 0;
}
