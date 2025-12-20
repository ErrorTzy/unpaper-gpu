// SPDX-FileCopyrightText: 2025 The unpaper authors
//
// SPDX-License-Identifier: GPL-2.0-only

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>

#include "lib/options.h"
#include "pdf/pdf_pipeline_batch.h"
#include "pdf/pdf_pipeline_cpu.h"
#include "pdf/pdf_reader.h"
#include "sheet_process.h"

static char test_jpeg_pdf_path[4096];
static char test_2page_pdf_path[4096];
static char test_output_path[4096];

static void init_test_paths(void) {
  const char *imgsrc_dir = getenv("TEST_IMGSRC_DIR");
  if (imgsrc_dir != NULL) {
    size_t len = strlen(imgsrc_dir);
    char base_dir[4096];
    strncpy(base_dir, imgsrc_dir, sizeof(base_dir) - 1);
    base_dir[sizeof(base_dir) - 1] = '\0';

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
    snprintf(test_output_path, sizeof(test_output_path),
             "%s/test_output_cpu.pdf", base_dir);
  } else {
    snprintf(test_jpeg_pdf_path, sizeof(test_jpeg_pdf_path),
             "tests/pdf_samples/test_jpeg.pdf");
    snprintf(test_2page_pdf_path, sizeof(test_2page_pdf_path),
             "tests/pdf_samples/test_2page.pdf");
    snprintf(test_output_path, sizeof(test_output_path),
             "/tmp/test_output_cpu.pdf");
  }
}

static void cleanup_output(void) { unlink(test_output_path); }

static void test_is_pdf(void) {
  printf("Test: pdf_pipeline_is_pdf... ");

  if (!pdf_pipeline_is_pdf("test.pdf")) {
    printf("FAILED (test.pdf not recognized)\n");
    exit(1);
  }

  if (!pdf_pipeline_is_pdf("/path/to/file.PDF")) {
    printf("FAILED (/path/to/file.PDF not recognized)\n");
    exit(1);
  }

  if (pdf_pipeline_is_pdf("test.png")) {
    printf("FAILED (test.png incorrectly recognized)\n");
    exit(1);
  }

  if (pdf_pipeline_is_pdf(NULL)) {
    printf("FAILED (NULL incorrectly recognized)\n");
    exit(1);
  }

  printf("PASSED\n");
}

static void test_single_page_pdf(void) {
  printf("Test: single page PDF processing... ");
  fflush(stdout);

  // Check if test file exists
  PdfDocument *doc = pdf_open(test_jpeg_pdf_path);
  if (doc == NULL) {
    printf("SKIPPED (test PDF not found: %s)\n", test_jpeg_pdf_path);
    return;
  }
  int input_pages = pdf_page_count(doc);
  PdfPageInfo input_info;
  if (!pdf_get_page_info(doc, 0, &input_info)) {
    printf("SKIPPED (could not get input page info)\n");
    pdf_close(doc);
    return;
  }
  pdf_close(doc);

  // Clean up any previous output
  cleanup_output();

  // Set up options with defaults
  Options options;
  options_init(&options);
  options.device = UNPAPER_DEVICE_CPU;
  options.write_output = true;
  options.perf = false;

  // Set up sheet processing config
  Rectangle preMasks[MAX_MASKS];
  Point points[MAX_POINTS];
  int32_t middleWipe[2] = {0, 0};
  Rectangle blackfilterExclude[MAX_MASKS];

  // Initialize filter parameters with defaults
  options_init_filter_defaults(&options, blackfilterExclude);

  SheetProcessConfig config;
  sheet_process_config_init(&config, &options, preMasks, 0, points, 0,
                            middleWipe, blackfilterExclude, 0);

  // Process the PDF
  int failed = pdf_pipeline_cpu_process(test_jpeg_pdf_path, test_output_path,
                                        &options, &config);

  if (failed != 0) {
    printf("FAILED (%d pages failed)\n", failed);
    cleanup_output();
    exit(1);
  }

  // Verify output exists and has correct page count
  doc = pdf_open(test_output_path);
  if (doc == NULL) {
    printf("FAILED (output PDF not created: %s)\n", pdf_get_last_error());
    cleanup_output();
    exit(1);
  }

  int output_pages = pdf_page_count(doc);
  PdfPageInfo output_info;
  if (!pdf_get_page_info(doc, 0, &output_info)) {
    printf("FAILED (could not get output page info)\n");
    pdf_close(doc);
    cleanup_output();
    exit(1);
  }
  pdf_close(doc);

  if (output_pages != input_pages) {
    printf("FAILED (page count mismatch: %d -> %d)\n", input_pages,
           output_pages);
    cleanup_output();
    exit(1);
  }

  cleanup_output();
  printf("PASSED (%d page(s), %.0fx%.0f -> %.0fx%.0f pts)\n", output_pages,
         input_info.width, input_info.height, output_info.width,
         output_info.height);
}

static void test_multi_page_pdf_batch_pipeline(void) {
  printf("Test: multi-page PDF batch pipeline (CPU wrapper)... ");

  PdfDocument *doc = pdf_open(test_2page_pdf_path);
  if (doc == NULL) {
    printf("SKIPPED (test PDF not found: %s)\n", test_2page_pdf_path);
    return;
  }
  int input_pages = pdf_page_count(doc);
  pdf_close(doc);

  if (input_pages < 2) {
    printf("SKIPPED (test PDF has only %d page)\n", input_pages);
    return;
  }

  cleanup_output();

  Options options;
  options_init(&options);
  options.device = UNPAPER_DEVICE_CPU;
  options.write_output = true;
  options.perf = false;

  Rectangle preMasks[MAX_MASKS];
  Point points[MAX_POINTS];
  int32_t middleWipe[2] = {0, 0};
  Rectangle blackfilterExclude[MAX_MASKS];

  options_init_filter_defaults(&options, blackfilterExclude);

  SheetProcessConfig config;
  sheet_process_config_init(&config, &options, preMasks, 0, points, 0,
                            middleWipe, blackfilterExclude, 0);

  PdfBatchConfig batch_config;
  pdf_batch_config_init(&batch_config);
  batch_config.parallelism = 2;
  batch_config.progress = false;
  batch_config.use_gpu = false;

  int failed = pdf_pipeline_batch_process(test_2page_pdf_path, test_output_path,
                                          &options, &config, &batch_config);

  if (failed != 0) {
    printf("FAILED (%d pages failed)\n", failed);
    cleanup_output();
    exit(1);
  }

  doc = pdf_open(test_output_path);
  if (doc == NULL) {
    printf("FAILED (output PDF not created: %s)\n", pdf_get_last_error());
    cleanup_output();
    exit(1);
  }

  int output_pages = pdf_page_count(doc);
  pdf_close(doc);

  if (output_pages != input_pages) {
    printf("FAILED (page count mismatch: %d -> %d)\n", input_pages,
           output_pages);
    cleanup_output();
    exit(1);
  }

  cleanup_output();
  printf("PASSED (%d pages)\n", output_pages);
}

static void test_invalid_input(void) {
  printf("Test: invalid input handling... ");

  Options options;
  options_init(&options);

  Rectangle preMasks[MAX_MASKS];
  Point points[MAX_POINTS];
  int32_t middleWipe[2] = {0, 0};
  Rectangle blackfilterExclude[MAX_MASKS];

  options_init_filter_defaults(&options, blackfilterExclude);

  SheetProcessConfig config;
  sheet_process_config_init(&config, &options, preMasks, 0, points, 0,
                            middleWipe, blackfilterExclude, 0);

  // Test NULL input
  int result =
      pdf_pipeline_cpu_process(NULL, test_output_path, &options, &config);
  if (result >= 0) {
    printf("FAILED (NULL input should fail)\n");
    exit(1);
  }

  // Test NULL output
  result =
      pdf_pipeline_cpu_process(test_jpeg_pdf_path, NULL, &options, &config);
  if (result >= 0) {
    printf("FAILED (NULL output should fail)\n");
    exit(1);
  }

  // Test non-existent input
  result = pdf_pipeline_cpu_process("/nonexistent/file.pdf", test_output_path,
                                    &options, &config);
  if (result >= 0) {
    printf("FAILED (non-existent input should fail)\n");
    exit(1);
  }

  printf("PASSED\n");
}

int main(void) {
  printf("PDF Pipeline CPU Unit Tests\n");
  printf("===========================\n\n");

  init_test_paths();

  test_is_pdf();
  test_invalid_input();
  test_single_page_pdf();
  test_multi_page_pdf_batch_pipeline();

  printf("\nAll tests passed!\n");
  return 0;
}
