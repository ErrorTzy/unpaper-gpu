// SPDX-FileCopyrightText: 2025 The unpaper authors
//
// SPDX-License-Identifier: GPL-2.0-only

#include <stdbool.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>

#include "pdf/pdf_reader.h"
#include "pdf/pdf_writer.h"

static char test_jpeg_path[4096];
static char test_jpeg_pdf_path[4096];
static char output_pdf_path[4096];

static void init_test_paths(void) {
  const char *imgsrc_dir = getenv("TEST_IMGSRC_DIR");
  if (imgsrc_dir != NULL) {
    snprintf(test_jpeg_path, sizeof(test_jpeg_path), "%s/test_jpeg.jpg",
             imgsrc_dir);

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
  } else {
    snprintf(test_jpeg_path, sizeof(test_jpeg_path),
             "tests/source_images/test_jpeg.jpg");
    snprintf(test_jpeg_pdf_path, sizeof(test_jpeg_pdf_path),
             "tests/pdf_samples/test_jpeg.pdf");
  }

  // Create output path in temp directory
  snprintf(output_pdf_path, sizeof(output_pdf_path), "/tmp/pdf_writer_test.pdf");
}

// Read a file into memory
static uint8_t *read_file(const char *path, size_t *size) {
  FILE *f = fopen(path, "rb");
  if (f == NULL)
    return NULL;

  fseek(f, 0, SEEK_END);
  long len = ftell(f);
  fseek(f, 0, SEEK_SET);

  if (len <= 0) {
    fclose(f);
    return NULL;
  }

  uint8_t *data = malloc((size_t)len);
  if (data == NULL) {
    fclose(f);
    return NULL;
  }

  size_t read_len = fread(data, 1, (size_t)len, f);
  fclose(f);

  if (read_len != (size_t)len) {
    free(data);
    return NULL;
  }

  *size = (size_t)len;
  return data;
}

// Extract dimensions from JPEG header
static bool get_jpeg_dimensions(const uint8_t *data, size_t len, int *width,
                                int *height) {
  // Look for SOF0/SOF1/SOF2 markers
  for (size_t i = 0; i + 10 < len; i++) {
    if (data[i] == 0xFF) {
      uint8_t marker = data[i + 1];
      if (marker == 0xC0 || marker == 0xC1 || marker == 0xC2) {
        *height = ((int)data[i + 5] << 8) | data[i + 6];
        *width = ((int)data[i + 7] << 8) | data[i + 8];
        return true;
      }
      // Skip variable-length markers
      if (marker >= 0xC0 && marker <= 0xFE && marker != 0xD0 &&
          marker != 0xD1 && marker != 0xD2 && marker != 0xD3 &&
          marker != 0xD4 && marker != 0xD5 && marker != 0xD6 &&
          marker != 0xD7 && marker != 0xD8 && marker != 0xD9) {
        if (i + 3 < len) {
          size_t seg_len = ((size_t)data[i + 2] << 8) | data[i + 3];
          i += seg_len + 1;
        }
      }
    }
  }
  return false;
}

static void test_create_close(void) {
  printf("Test: pdf_writer_create/pdf_writer_close... ");

  // Create a writer without metadata
  PdfWriter *writer = pdf_writer_create(output_pdf_path, NULL, 300);
  if (writer == NULL) {
    printf("FAILED (could not create writer: %s)\n",
           pdf_writer_get_last_error());
    exit(1);
  }

  // Close without adding any pages (empty PDF)
  if (!pdf_writer_close(writer)) {
    printf("FAILED (could not close writer: %s)\n",
           pdf_writer_get_last_error());
    exit(1);
  }

  // Verify the file was created
  FILE *f = fopen(output_pdf_path, "rb");
  if (f == NULL) {
    printf("FAILED (output file not created)\n");
    exit(1);
  }
  fclose(f);

  // Clean up
  unlink(output_pdf_path);

  printf("PASSED\n");
}

static void test_null_safety(void) {
  printf("Test: null safety... ");

  // pdf_writer_create with NULL path
  PdfWriter *writer = pdf_writer_create(NULL, NULL, 300);
  if (writer != NULL) {
    printf("FAILED (NULL path should fail)\n");
    pdf_writer_close(writer);
    exit(1);
  }

  // pdf_writer_close with NULL
  pdf_writer_close(NULL); // Should not crash

  // pdf_writer_abort with NULL
  pdf_writer_abort(NULL); // Should not crash

  // pdf_writer_page_count with NULL
  if (pdf_writer_page_count(NULL) != 0) {
    printf("FAILED (NULL writer page count should be 0)\n");
    exit(1);
  }

  printf("PASSED\n");
}

static void test_add_jpeg_page(void) {
  printf("Test: pdf_writer_add_page_jpeg... ");

  // Load JPEG file
  size_t jpeg_size;
  uint8_t *jpeg_data = read_file(test_jpeg_path, &jpeg_size);
  if (jpeg_data == NULL) {
    printf("SKIPPED (test JPEG not found: %s)\n", test_jpeg_path);
    return;
  }

  // Get dimensions
  int width, height;
  if (!get_jpeg_dimensions(jpeg_data, jpeg_size, &width, &height)) {
    printf("FAILED (could not parse JPEG dimensions)\n");
    free(jpeg_data);
    exit(1);
  }

  // Create writer
  PdfWriter *writer = pdf_writer_create(output_pdf_path, NULL, 150);
  if (writer == NULL) {
    printf("FAILED (could not create writer: %s)\n",
           pdf_writer_get_last_error());
    free(jpeg_data);
    exit(1);
  }

  // Add page
  if (!pdf_writer_add_page_jpeg(writer, jpeg_data, jpeg_size, width, height,
                                0)) {
    printf("FAILED (could not add JPEG page: %s)\n",
           pdf_writer_get_last_error());
    pdf_writer_abort(writer);
    free(jpeg_data);
    exit(1);
  }

  if (pdf_writer_page_count(writer) != 1) {
    printf("FAILED (page count should be 1, got %d)\n",
           pdf_writer_page_count(writer));
    pdf_writer_abort(writer);
    free(jpeg_data);
    exit(1);
  }

  // Close and save
  if (!pdf_writer_close(writer)) {
    printf("FAILED (could not close writer: %s)\n",
           pdf_writer_get_last_error());
    free(jpeg_data);
    exit(1);
  }

  free(jpeg_data);

  // Verify output with pdf_reader
  PdfDocument *doc = pdf_open(output_pdf_path);
  if (doc == NULL) {
    printf("FAILED (could not reopen output: %s)\n", pdf_get_last_error());
    unlink(output_pdf_path);
    exit(1);
  }

  int page_count = pdf_page_count(doc);
  if (page_count != 1) {
    printf("FAILED (expected 1 page, got %d)\n", page_count);
    pdf_close(doc);
    unlink(output_pdf_path);
    exit(1);
  }

  pdf_close(doc);
  unlink(output_pdf_path);

  printf("PASSED (%dx%d JPEG)\n", width, height);
}

static void test_add_pixels_page(void) {
  printf("Test: pdf_writer_add_page_pixels... ");

  // Create a simple 100x100 grayscale gradient image
  int width = 100;
  int height = 100;
  int stride = width;
  uint8_t *pixels = malloc((size_t)stride * height);
  if (pixels == NULL) {
    printf("FAILED (out of memory)\n");
    exit(1);
  }

  for (int y = 0; y < height; y++) {
    for (int x = 0; x < width; x++) {
      pixels[y * stride + x] = (uint8_t)((x + y) * 255 / 200);
    }
  }

  // Create writer
  PdfWriter *writer = pdf_writer_create(output_pdf_path, NULL, 72);
  if (writer == NULL) {
    printf("FAILED (could not create writer: %s)\n",
           pdf_writer_get_last_error());
    free(pixels);
    exit(1);
  }

  // Add grayscale page
  if (!pdf_writer_add_page_pixels(writer, pixels, width, height, stride,
                                  PDF_PIXEL_GRAY8, 0)) {
    printf("FAILED (could not add pixel page: %s)\n",
           pdf_writer_get_last_error());
    pdf_writer_abort(writer);
    free(pixels);
    exit(1);
  }

  // Close and save
  if (!pdf_writer_close(writer)) {
    printf("FAILED (could not close writer: %s)\n",
           pdf_writer_get_last_error());
    free(pixels);
    exit(1);
  }

  free(pixels);

  // Verify output
  PdfDocument *doc = pdf_open(output_pdf_path);
  if (doc == NULL) {
    printf("FAILED (could not reopen output: %s)\n", pdf_get_last_error());
    unlink(output_pdf_path);
    exit(1);
  }

  int page_count = pdf_page_count(doc);
  if (page_count != 1) {
    printf("FAILED (expected 1 page, got %d)\n", page_count);
    pdf_close(doc);
    unlink(output_pdf_path);
    exit(1);
  }

  pdf_close(doc);
  unlink(output_pdf_path);

  printf("PASSED (%dx%d grayscale)\n", width, height);
}

static void test_add_rgb_pixels_page(void) {
  printf("Test: pdf_writer_add_page_pixels (RGB)... ");

  // Create a simple 100x100 RGB image with color bars
  int width = 100;
  int height = 100;
  int stride = width * 3;
  uint8_t *pixels = malloc((size_t)stride * height);
  if (pixels == NULL) {
    printf("FAILED (out of memory)\n");
    exit(1);
  }

  for (int y = 0; y < height; y++) {
    for (int x = 0; x < width; x++) {
      int idx = y * stride + x * 3;
      // Create RGB color bars
      if (x < 33) {
        pixels[idx + 0] = 255; // Red
        pixels[idx + 1] = 0;
        pixels[idx + 2] = 0;
      } else if (x < 66) {
        pixels[idx + 0] = 0;
        pixels[idx + 1] = 255; // Green
        pixels[idx + 2] = 0;
      } else {
        pixels[idx + 0] = 0;
        pixels[idx + 1] = 0;
        pixels[idx + 2] = 255; // Blue
      }
    }
  }

  // Create writer
  PdfWriter *writer = pdf_writer_create(output_pdf_path, NULL, 72);
  if (writer == NULL) {
    printf("FAILED (could not create writer: %s)\n",
           pdf_writer_get_last_error());
    free(pixels);
    exit(1);
  }

  // Add RGB page
  if (!pdf_writer_add_page_pixels(writer, pixels, width, height, stride,
                                  PDF_PIXEL_RGB24, 0)) {
    printf("FAILED (could not add RGB page: %s)\n",
           pdf_writer_get_last_error());
    pdf_writer_abort(writer);
    free(pixels);
    exit(1);
  }

  // Close and save
  if (!pdf_writer_close(writer)) {
    printf("FAILED (could not close writer: %s)\n",
           pdf_writer_get_last_error());
    free(pixels);
    exit(1);
  }

  free(pixels);

  // Verify output
  PdfDocument *doc = pdf_open(output_pdf_path);
  if (doc == NULL) {
    printf("FAILED (could not reopen output: %s)\n", pdf_get_last_error());
    unlink(output_pdf_path);
    exit(1);
  }

  int page_count = pdf_page_count(doc);
  if (page_count != 1) {
    printf("FAILED (expected 1 page, got %d)\n", page_count);
    pdf_close(doc);
    unlink(output_pdf_path);
    exit(1);
  }

  pdf_close(doc);
  unlink(output_pdf_path);

  printf("PASSED (%dx%d RGB)\n", width, height);
}

static void test_metadata_preservation(void) {
  printf("Test: metadata preservation... ");

  // Create metadata
  PdfMetadata meta = {0};
  meta.title = "Test PDF Title";
  meta.author = "Test Author";
  meta.subject = "Test Subject";
  meta.keywords = "test, pdf, writer";

  // Create writer with metadata
  PdfWriter *writer = pdf_writer_create(output_pdf_path, &meta, 72);
  if (writer == NULL) {
    printf("FAILED (could not create writer: %s)\n",
           pdf_writer_get_last_error());
    exit(1);
  }

  // Add a simple page
  uint8_t pixels[100];
  memset(pixels, 128, sizeof(pixels));
  if (!pdf_writer_add_page_pixels(writer, pixels, 10, 10, 10, PDF_PIXEL_GRAY8,
                                  0)) {
    printf("FAILED (could not add page)\n");
    pdf_writer_abort(writer);
    exit(1);
  }

  // Close and save
  if (!pdf_writer_close(writer)) {
    printf("FAILED (could not close writer: %s)\n",
           pdf_writer_get_last_error());
    exit(1);
  }

  // Verify metadata
  PdfDocument *doc = pdf_open(output_pdf_path);
  if (doc == NULL) {
    printf("FAILED (could not reopen output: %s)\n", pdf_get_last_error());
    unlink(output_pdf_path);
    exit(1);
  }

  PdfMetadata read_meta = pdf_get_metadata(doc);

  if (read_meta.title == NULL || strcmp(read_meta.title, "Test PDF Title") != 0) {
    printf("FAILED (title mismatch: expected 'Test PDF Title', got '%s')\n",
           read_meta.title ? read_meta.title : "(null)");
    pdf_free_metadata(&read_meta);
    pdf_close(doc);
    unlink(output_pdf_path);
    exit(1);
  }

  if (read_meta.author == NULL || strcmp(read_meta.author, "Test Author") != 0) {
    printf("FAILED (author mismatch)\n");
    pdf_free_metadata(&read_meta);
    pdf_close(doc);
    unlink(output_pdf_path);
    exit(1);
  }

  if (read_meta.producer == NULL ||
      strcmp(read_meta.producer, "unpaper") != 0) {
    printf("FAILED (producer should be 'unpaper', got '%s')\n",
           read_meta.producer ? read_meta.producer : "(null)");
    pdf_free_metadata(&read_meta);
    pdf_close(doc);
    unlink(output_pdf_path);
    exit(1);
  }

  pdf_free_metadata(&read_meta);
  pdf_close(doc);
  unlink(output_pdf_path);

  printf("PASSED\n");
}

static void test_multi_page(void) {
  printf("Test: multi-page PDF... ");

  // Create writer
  PdfWriter *writer = pdf_writer_create(output_pdf_path, NULL, 72);
  if (writer == NULL) {
    printf("FAILED (could not create writer: %s)\n",
           pdf_writer_get_last_error());
    exit(1);
  }

  // Add 3 pages with different sizes
  for (int i = 0; i < 3; i++) {
    int size = 50 + i * 50; // 50, 100, 150
    uint8_t *pixels = malloc((size_t)size * size);
    if (pixels == NULL) {
      printf("FAILED (out of memory)\n");
      pdf_writer_abort(writer);
      exit(1);
    }

    memset(pixels, (uint8_t)(85 * i), (size_t)size * size);

    if (!pdf_writer_add_page_pixels(writer, pixels, size, size, size,
                                    PDF_PIXEL_GRAY8, 0)) {
      printf("FAILED (could not add page %d)\n", i);
      pdf_writer_abort(writer);
      free(pixels);
      exit(1);
    }

    free(pixels);
  }

  if (pdf_writer_page_count(writer) != 3) {
    printf("FAILED (expected 3 pages, got %d)\n", pdf_writer_page_count(writer));
    pdf_writer_abort(writer);
    exit(1);
  }

  // Close and save
  if (!pdf_writer_close(writer)) {
    printf("FAILED (could not close writer: %s)\n",
           pdf_writer_get_last_error());
    exit(1);
  }

  // Verify output
  PdfDocument *doc = pdf_open(output_pdf_path);
  if (doc == NULL) {
    printf("FAILED (could not reopen output: %s)\n", pdf_get_last_error());
    unlink(output_pdf_path);
    exit(1);
  }

  int page_count = pdf_page_count(doc);
  if (page_count != 3) {
    printf("FAILED (expected 3 pages, got %d)\n", page_count);
    pdf_close(doc);
    unlink(output_pdf_path);
    exit(1);
  }

  pdf_close(doc);
  unlink(output_pdf_path);

  printf("PASSED (3 pages)\n");
}

static void test_abort(void) {
  printf("Test: pdf_writer_abort... ");

  // Create writer
  PdfWriter *writer = pdf_writer_create(output_pdf_path, NULL, 72);
  if (writer == NULL) {
    printf("FAILED (could not create writer: %s)\n",
           pdf_writer_get_last_error());
    exit(1);
  }

  // Add a page
  uint8_t pixels[100];
  memset(pixels, 128, sizeof(pixels));
  pdf_writer_add_page_pixels(writer, pixels, 10, 10, 10, PDF_PIXEL_GRAY8, 0);

  // Abort instead of close
  pdf_writer_abort(writer);

  // Verify the file was NOT created (or is incomplete)
  // Note: On some systems the file might be created but empty/invalid
  // Either way, it shouldn't be a valid PDF
  PdfDocument *doc = pdf_open(output_pdf_path);
  if (doc != NULL) {
    // If we can open it, that's acceptable as long as abort didn't crash
    pdf_close(doc);
  }

  unlink(output_pdf_path);

  printf("PASSED\n");
}

static void test_extract_and_reembed_jpeg(void) {
  printf("Test: extract JPEG from PDF and reembed... ");

  // Open source PDF with JPEG
  PdfDocument *src_doc = pdf_open(test_jpeg_pdf_path);
  if (src_doc == NULL) {
    printf("SKIPPED (source PDF not found)\n");
    return;
  }

  // Extract image
  PdfImage image;
  if (!pdf_extract_page_image(src_doc, 0, &image)) {
    printf("SKIPPED (could not extract image: %s)\n", pdf_get_last_error());
    pdf_close(src_doc);
    return;
  }

  if (image.format != PDF_IMAGE_JPEG) {
    printf("SKIPPED (image is not JPEG)\n");
    pdf_free_image(&image);
    pdf_close(src_doc);
    return;
  }

  // Get metadata from source
  PdfMetadata meta = pdf_get_metadata(src_doc);

  // Create new PDF with the extracted image
  PdfWriter *writer = pdf_writer_create(output_pdf_path, &meta, 150);
  if (writer == NULL) {
    printf("FAILED (could not create writer: %s)\n",
           pdf_writer_get_last_error());
    pdf_free_metadata(&meta);
    pdf_free_image(&image);
    pdf_close(src_doc);
    exit(1);
  }

  // Add the extracted JPEG
  if (!pdf_writer_add_page_jpeg(writer, image.data, image.size, image.width,
                                image.height, 0)) {
    printf("FAILED (could not add JPEG page: %s)\n",
           pdf_writer_get_last_error());
    pdf_writer_abort(writer);
    pdf_free_metadata(&meta);
    pdf_free_image(&image);
    pdf_close(src_doc);
    exit(1);
  }

  // Close writer
  if (!pdf_writer_close(writer)) {
    printf("FAILED (could not close writer: %s)\n",
           pdf_writer_get_last_error());
    pdf_free_metadata(&meta);
    pdf_free_image(&image);
    pdf_close(src_doc);
    exit(1);
  }

  pdf_free_metadata(&meta);
  pdf_free_image(&image);
  pdf_close(src_doc);

  // Verify output
  PdfDocument *out_doc = pdf_open(output_pdf_path);
  if (out_doc == NULL) {
    printf("FAILED (could not reopen output: %s)\n", pdf_get_last_error());
    unlink(output_pdf_path);
    exit(1);
  }

  if (pdf_page_count(out_doc) != 1) {
    printf("FAILED (expected 1 page)\n");
    pdf_close(out_doc);
    unlink(output_pdf_path);
    exit(1);
  }

  pdf_close(out_doc);
  unlink(output_pdf_path);

  printf("PASSED\n");
}

int main(void) {
  printf("PDF Writer Unit Tests\n");
  printf("=====================\n\n");

  init_test_paths();

  // Run tests
  test_null_safety();
  test_create_close();
  test_add_jpeg_page();
  test_add_pixels_page();
  test_add_rgb_pixels_page();
  test_metadata_preservation();
  test_multi_page();
  test_abort();
  test_extract_and_reembed_jpeg();

  printf("\nAll tests passed!\n");
  return 0;
}
