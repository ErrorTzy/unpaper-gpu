// SPDX-FileCopyrightText: 2025 The unpaper authors
//
// SPDX-License-Identifier: GPL-2.0-only

#include "pdf_pipeline_gpu.h"

#include "constants.h"
#include "imageprocess/backend.h"
#include "imageprocess/backend_cuda_internal.h"
#include "imageprocess/blit.h"
#include "imageprocess/cuda_runtime.h"
#include "imageprocess/deskew.h"
#include "imageprocess/filters.h"
#include "imageprocess/image.h"
#include "imageprocess/masks.h"
#include "imageprocess/nvimgcodec.h"
#include "imageprocess/pixel.h"
#include "lib/logging.h"
#include "lib/perf.h"
#include "pdf_reader.h"
#include "pdf_writer.h"
#include "sheet_process.h"

#ifdef UNPAPER_WITH_JBIG2
#include "lib/jbig2_decode.h"
#endif

#include <libavutil/frame.h>
#include <libavutil/pixfmt.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

// Default JPEG quality when not specified
#define DEFAULT_JPEG_QUALITY 85

bool pdf_pipeline_gpu_available(void) {
#if defined(UNPAPER_WITH_CUDA) && (UNPAPER_WITH_CUDA)
  // Try to initialize nvimgcodec if not already initialized
  if (!nvimgcodec_any_available()) {
    // Try to initialize with 8 streams
    if (!nvimgcodec_init(8)) {
      return false;
    }
  }
  return nvimgcodec_any_available();
#else
  return false;
#endif
}

#if defined(UNPAPER_WITH_CUDA) && (UNPAPER_WITH_CUDA)

// Expand 1-bit packed data to 8-bit grayscale on GPU
// Uses the unpaper_expand_1bit_to_8bit kernel
static bool gpu_expand_1bit_to_8bit(UnpaperCudaStream *stream,
                                    const uint8_t *src_1bit_host,
                                    int src_stride_1bit, int width, int height,
                                    uint64_t *dst_8bit_gpu,
                                    int *dst_stride_8bit, bool invert) {
  ensure_kernels_loaded();

  // Calculate sizes
  int src_bytes_per_row = (width + 7) / 8;
  size_t src_size = (size_t)src_bytes_per_row * (size_t)height;

  // Allocate GPU memory for 1-bit source
  uint64_t src_gpu = unpaper_cuda_malloc_async(stream, src_size);
  if (src_gpu == 0) {
    verboseLog(VERBOSE_MORE, "GPU expand: failed to allocate source memory\n");
    return false;
  }

  // Upload 1-bit data to GPU (H2D)
  // Copy row by row if stride differs
  if (src_stride_1bit == src_bytes_per_row) {
    unpaper_cuda_memcpy_h2d_async(stream, src_gpu, src_1bit_host, src_size);
  } else {
    // Copy row by row
    for (int y = 0; y < height; y++) {
      unpaper_cuda_memcpy_h2d_async(
          stream, src_gpu + (size_t)y * (size_t)src_bytes_per_row,
          src_1bit_host + (size_t)y * (size_t)src_stride_1bit,
          (size_t)src_bytes_per_row);
    }
  }

  // Allocate GPU memory for 8-bit destination
  *dst_stride_8bit = width; // No padding for simplicity
  size_t dst_size = (size_t)width * (size_t)height;
  *dst_8bit_gpu = unpaper_cuda_malloc_async(stream, dst_size);
  if (*dst_8bit_gpu == 0) {
    unpaper_cuda_free_async(stream, src_gpu);
    verboseLog(VERBOSE_MORE,
               "GPU expand: failed to allocate destination memory\n");
    return false;
  }

  // Launch kernel
  int invert_int = invert ? 1 : 0;
  void *params[] = {
      (void *)&src_gpu,     (void *)&src_bytes_per_row,
      (void *)dst_8bit_gpu, (void *)dst_stride_8bit,
      (void *)&width,       (void *)&height,
      (void *)&invert_int,
  };

  // Each thread handles 8 pixels (one byte of packed input)
  uint32_t threads_x = 16;
  uint32_t threads_y = 16;
  uint32_t blocks_x = ((uint32_t)src_bytes_per_row + threads_x - 1) / threads_x;
  uint32_t blocks_y = ((uint32_t)height + threads_y - 1) / threads_y;

  unpaper_cuda_launch_kernel_on_stream(stream, k_expand_1bit_to_8bit, blocks_x,
                                       blocks_y, 1, threads_x, threads_y, 1,
                                       params);

  // Free source 1-bit GPU memory (we only need the 8-bit result)
  unpaper_cuda_free_async(stream, src_gpu);

  return true;
}

// Decode JBIG2 to GPU-resident 8-bit grayscale
#ifdef UNPAPER_WITH_JBIG2
static bool decode_jbig2_to_gpu(const PdfImage *pdf_img,
                                UnpaperCudaStream *stream, uint64_t *gpu_ptr,
                                int *gpu_pitch, int *width, int *height) {
  if (pdf_img == NULL || pdf_img->data == NULL || pdf_img->size == 0) {
    return false;
  }

  // CPU decode to 1-bit
  Jbig2DecodedImage jbig2_img = {0};
  if (!jbig2_decode(pdf_img->data, pdf_img->size, pdf_img->jbig2_globals,
                    pdf_img->jbig2_globals_size, &jbig2_img)) {
    verboseLog(VERBOSE_MORE, "JBIG2 decode failed: %s\n",
               jbig2_get_last_error());
    return false;
  }

  *width = (int)jbig2_img.width;
  *height = (int)jbig2_img.height;

  // GPU expand 1-bit to 8-bit
  // JBIG2 uses 1=black, 0=white, so invert to get 0=black, 255=white
  bool ok =
      gpu_expand_1bit_to_8bit(stream, jbig2_img.data, (int)jbig2_img.stride,
                              *width, *height, gpu_ptr, gpu_pitch, true);

  jbig2_free_image(&jbig2_img);
  return ok;
}
#endif

// Decode image bytes to GPU using nvImageCodec (JPEG/JP2)
static bool decode_bytes_to_gpu(const uint8_t *data, size_t size,
                                PdfImageFormat format,
                                NvImgCodecDecodeState *dec_state,
                                UnpaperCudaStream *stream,
                                NvImgCodecDecodedImage *gpu_img) {
  // Only JPEG and JP2 supported via nvImageCodec
  if (format != PDF_IMAGE_JPEG && format != PDF_IMAGE_JP2) {
    return false;
  }

  // Check format compatibility
  NvImgCodecFormat nvfmt = nvimgcodec_detect_format(data, size);
  if (nvfmt == NVIMGCODEC_FORMAT_UNKNOWN) {
    verboseLog(VERBOSE_MORE, "GPU decode: unrecognized format\n");
    return false;
  }

  // Decode to grayscale (most documents are B&W/grayscale)
  bool ok = nvimgcodec_decode(data, size, dec_state, stream,
                              NVIMGCODEC_OUT_GRAY8, gpu_img);
  if (!ok) {
    verboseLog(VERBOSE_MORE, "GPU decode: nvimgcodec_decode failed\n");
  }

  return ok;
}

// Render page on CPU and upload to GPU
static bool render_page_to_gpu(PdfDocument *doc, int page_idx, int dpi,
                               UnpaperCudaStream *stream, uint64_t *gpu_ptr,
                               int *gpu_pitch, int *width, int *height,
                               bool grayscale) {
  int stride = 0;
  uint8_t *pixels;
  int components;

  if (grayscale) {
    pixels = pdf_render_page_gray(doc, page_idx, dpi, width, height, &stride);
    components = 1;
  } else {
    pixels = pdf_render_page(doc, page_idx, dpi, width, height, &stride);
    components = 3;
  }

  if (pixels == NULL) {
    verboseLog(VERBOSE_MORE, "GPU pipeline: render failed for page %d\n",
               page_idx + 1);
    return false;
  }

  // Allocate GPU memory
  *gpu_pitch = (*width) * components;
  size_t gpu_size = (size_t)(*gpu_pitch) * (size_t)(*height);
  *gpu_ptr = unpaper_cuda_malloc_async(stream, gpu_size);
  if (*gpu_ptr == 0) {
    free(pixels);
    return false;
  }

  // Upload to GPU (H2D)
  if (stride == *gpu_pitch) {
    unpaper_cuda_memcpy_h2d_async(stream, *gpu_ptr, pixels, gpu_size);
  } else {
    // Copy row by row
    for (int y = 0; y < *height; y++) {
      unpaper_cuda_memcpy_h2d_async(
          stream, *gpu_ptr + (size_t)y * (size_t)(*gpu_pitch),
          pixels + (size_t)y * (size_t)stride, (size_t)(*gpu_pitch));
    }
  }

  free(pixels);
  return true;
}

#endif // UNPAPER_WITH_CUDA

int pdf_pipeline_gpu_process(const char *input_path, const char *output_path,
                             const Options *options,
                             const SheetProcessConfig *config) {
#if defined(UNPAPER_WITH_CUDA) && (UNPAPER_WITH_CUDA)
  if (input_path == NULL || output_path == NULL || options == NULL ||
      config == NULL) {
    verboseLog(VERBOSE_NORMAL, "GPU PDF pipeline: invalid arguments\n");
    return -1;
  }

  if (!pdf_pipeline_gpu_available()) {
    verboseLog(VERBOSE_NORMAL,
               "GPU PDF pipeline: CUDA/nvImageCodec not available\n");
    return -1;
  }

  verboseLog(VERBOSE_NORMAL, "GPU PDF pipeline: %s -> %s\n", input_path,
             output_path);

  // Initialize nvImageCodec if needed
  if (!nvimgcodec_init(8)) {
    verboseLog(VERBOSE_NORMAL, "GPU PDF pipeline: nvimgcodec init failed\n");
    return -1;
  }

  // Select CUDA backend
  image_backend_select(UNPAPER_DEVICE_CUDA);

  // Open input PDF
  PdfDocument *doc = pdf_open(input_path);
  if (doc == NULL) {
    verboseLog(VERBOSE_NORMAL, "GPU PDF pipeline: failed to open input: %s\n",
               pdf_get_last_error());
    return -1;
  }

  int page_count = pdf_page_count(doc);
  if (page_count <= 0) {
    verboseLog(VERBOSE_NORMAL, "GPU PDF pipeline: no pages in document\n");
    pdf_close(doc);
    return -1;
  }

  verboseLog(VERBOSE_NORMAL, "GPU PDF pipeline: %d pages to process\n",
             page_count);

  // Get metadata for output
  PdfMetadata meta = pdf_get_metadata(doc);

  // Create output PDF writer
  PdfWriter *writer =
      pdf_writer_create(output_path, &meta, options->pdf_render_dpi);
  if (writer == NULL) {
    verboseLog(VERBOSE_NORMAL,
               "GPU PDF pipeline: failed to create output: %s\n",
               pdf_writer_get_last_error());
    pdf_free_metadata(&meta);
    pdf_close(doc);
    return -1;
  }

  pdf_free_metadata(&meta);

  // Create modified options with write_output = false
  // so process_sheet() doesn't try to save files
  Options pdf_options = *options;
  pdf_options.write_output = false;

  // Create config with modified options
  SheetProcessConfig pdf_config = *config;
  pdf_config.options = &pdf_options;

  // Get CUDA stream
  UnpaperCudaStream *stream = unpaper_cuda_stream_get_default();

  int failed_pages = 0;
  PerfRecorder perf;

  // JPEG quality to use
  int jpeg_quality = (options->jpeg_quality > 0) ? options->jpeg_quality
                                                 : DEFAULT_JPEG_QUALITY;

  // Process each page
  for (int page_idx = 0; page_idx < page_count; page_idx++) {
    perf_recorder_init(&perf, options->perf, true);

    verboseLog(VERBOSE_NORMAL, "GPU PDF pipeline: processing page %d/%d\n",
               page_idx + 1, page_count);

    Image page_image = EMPTY_IMAGE;
    int width = 0, height = 0;
    bool decoded = false;

    // Stage 1: Decode to GPU
    perf_stage_begin(&perf, PERF_STAGE_DECODE);

    // Try to extract embedded image
    PdfImage pdf_img = {0};
    if (pdf_extract_page_image(doc, page_idx, &pdf_img)) {
      verboseLog(VERBOSE_MORE, "GPU PDF pipeline: extracted %s image %dx%d\n",
                 pdf_image_format_name(pdf_img.format), pdf_img.width,
                 pdf_img.height);

#ifdef UNPAPER_WITH_JBIG2
      // Handle JBIG2 with GPU expansion
      if (pdf_img.format == PDF_IMAGE_JBIG2) {
        uint64_t gpu_ptr = 0;
        int gpu_pitch = 0;
        if (decode_jbig2_to_gpu(&pdf_img, stream, &gpu_ptr, &gpu_pitch, &width,
                                &height)) {
          // Sync stream before creating Image
          unpaper_cuda_stream_synchronize_on(stream);

          page_image = create_image_from_gpu(
              (void *)(uintptr_t)gpu_ptr, (size_t)gpu_pitch, width, height,
              AV_PIX_FMT_GRAY8, options->sheet_background,
              options->abs_black_threshold, true);
          decoded = (page_image.frame != NULL);
          verboseLog(VERBOSE_MORE,
                     "GPU PDF pipeline: JBIG2 decoded to GPU %dx%d\n", width,
                     height);
        }
      } else
#endif
        // Handle JPEG/JP2 with nvImageCodec
        if (pdf_img.format == PDF_IMAGE_JPEG ||
            pdf_img.format == PDF_IMAGE_JP2) {
          NvImgCodecDecodeState *dec_state = nvimgcodec_acquire_decode_state();
          if (dec_state != NULL) {
            NvImgCodecDecodedImage gpu_img = {0};
            if (decode_bytes_to_gpu(pdf_img.data, pdf_img.size, pdf_img.format,
                                    dec_state, stream, &gpu_img)) {
              width = gpu_img.width;
              height = gpu_img.height;

              // Wait for decode completion
              nvimgcodec_wait_decode_complete(&gpu_img);

              page_image = create_image_from_gpu(
                  gpu_img.gpu_ptr, gpu_img.pitch, width, height,
                  AV_PIX_FMT_GRAY8, options->sheet_background,
                  options->abs_black_threshold, true);
              decoded = (page_image.frame != NULL);

              // Release completion event
              nvimgcodec_release_completion_event(gpu_img.completion_event,
                                                  gpu_img.event_from_pool);

              verboseLog(VERBOSE_MORE,
                         "GPU PDF pipeline: %s decoded to GPU %dx%d\n",
                         pdf_image_format_name(pdf_img.format), width, height);
            }
            nvimgcodec_release_decode_state(dec_state);
          }
        }

      pdf_free_image(&pdf_img);
    }

    // Fallback: render page on CPU and upload to GPU
    if (!decoded) {
      uint64_t gpu_ptr = 0;
      int gpu_pitch = 0;
      if (render_page_to_gpu(doc, page_idx, options->pdf_render_dpi, stream,
                             &gpu_ptr, &gpu_pitch, &width, &height, true)) {
        // Sync stream
        unpaper_cuda_stream_synchronize_on(stream);

        page_image = create_image_from_gpu(
            (void *)(uintptr_t)gpu_ptr, (size_t)gpu_pitch, width, height,
            AV_PIX_FMT_GRAY8, options->sheet_background,
            options->abs_black_threshold, true);
        decoded = (page_image.frame != NULL);
        verboseLog(VERBOSE_MORE,
                   "GPU PDF pipeline: rendered and uploaded to GPU %dx%d\n",
                   width, height);
      }
    }

    perf_stage_end(&perf, PERF_STAGE_DECODE);

    if (!decoded || page_image.frame == NULL) {
      verboseLog(VERBOSE_NORMAL, "GPU PDF pipeline: failed to decode page %d\n",
                 page_idx + 1);
      failed_pages++;
      continue;
    }

    // Stage 2: Process using process_sheet()
    BatchJob job = {0};
    job.sheet_nr = page_idx + 1;
    job.input_count = 1;
    job.output_count = 0; // No file output
    job.input_files[0] = NULL;

    SheetProcessState state;
    sheet_process_state_init(&state, &pdf_config, &job);

    // Set the GPU decoded image - process_sheet will use it
    sheet_process_state_set_gpu_decoded_image(&state, page_image, 0);

    // Process the sheet (all filters, deskew, etc.)
    bool process_ok = process_sheet(&state, &pdf_config);

    if (!process_ok || state.sheet.frame == NULL) {
      verboseLog(VERBOSE_NORMAL,
                 "GPU PDF pipeline: processing failed for page %d\n",
                 page_idx + 1);
      sheet_process_state_cleanup(&state);
      failed_pages++;
      continue;
    }

    // Stage 3: Encode and add to output PDF
    perf_stage_begin(&perf, PERF_STAGE_ENCODE);

    // Ensure image is on GPU for encoding
    image_ensure_cuda(&state.sheet);

    int out_width = state.sheet.frame->width;
    int out_height = state.sheet.frame->height;
    bool page_success = false;

    // Get GPU pointer and pitch
    void *gpu_ptr = image_get_gpu_ptr(&state.sheet);
    size_t gpu_pitch = image_get_gpu_pitch(&state.sheet);

    verboseLog(VERBOSE_MORE, "GPU PDF pipeline: encode gpu_ptr=%p pitch=%zu\n",
               gpu_ptr, gpu_pitch);

    if (gpu_ptr != NULL) {
      NvImgCodecEncodeState *enc_state = nvimgcodec_acquire_encode_state();
      if (enc_state != NULL) {
        NvImgCodecEncodedImage encoded = {0};
        bool encode_ok = false;

        // Determine input format based on pixel format
        NvImgCodecEncodeInputFormat input_fmt = NVIMGCODEC_ENC_FMT_RGB;
        if (state.sheet.frame->format == AV_PIX_FMT_GRAY8) {
          input_fmt = NVIMGCODEC_ENC_FMT_GRAY8;
        }

        // Encode based on quality mode
        if (options->pdf_quality_mode == PDF_QUALITY_HIGH &&
            nvimgcodec_jp2_supported()) {
          // High quality: JP2 lossless
          encode_ok = nvimgcodec_encode_jp2(gpu_ptr, gpu_pitch, out_width,
                                            out_height, input_fmt, true,
                                            enc_state, stream, &encoded);
          if (encode_ok && encoded.data != NULL) {
            page_success = pdf_writer_add_page_jp2(
                writer, encoded.data, encoded.size, out_width, out_height,
                options->pdf_render_dpi);
            free(encoded.data);
          }
        } else {
          // Fast mode: JPEG
          encode_ok = nvimgcodec_encode_jpeg(
              gpu_ptr, gpu_pitch, out_width, out_height, input_fmt,
              jpeg_quality, enc_state, stream, &encoded);
          verboseLog(VERBOSE_MORE,
                     "GPU PDF pipeline: JPEG encode_ok=%d data=%p size=%zu\n",
                     encode_ok, encoded.data, encoded.size);
          if (encode_ok && encoded.data != NULL) {
            page_success = pdf_writer_add_page_jpeg(
                writer, encoded.data, encoded.size, out_width, out_height,
                options->pdf_render_dpi);
            verboseLog(VERBOSE_MORE,
                       "GPU PDF pipeline: add_page_jpeg success=%d\n",
                       page_success);
            free(encoded.data);
          }
        }

        nvimgcodec_release_encode_state(enc_state);
      }
    }

    // Fallback to CPU encode if GPU encode failed
    if (!page_success) {
      verboseLog(VERBOSE_MORE,
                 "GPU PDF pipeline: GPU encode failed, falling back to CPU\n");
      image_ensure_cpu(&state.sheet);

      // Use pixels path
      int stride = state.sheet.frame->linesize[0];
      PdfPixelFormat fmt = (state.sheet.frame->format == AV_PIX_FMT_GRAY8)
                               ? PDF_PIXEL_GRAY8
                               : PDF_PIXEL_RGB24;
      page_success = pdf_writer_add_page_pixels(
          writer, state.sheet.frame->data[0], out_width, out_height, stride,
          fmt, options->pdf_render_dpi);
    }

    perf_stage_end(&perf, PERF_STAGE_ENCODE);

    if (!page_success) {
      verboseLog(VERBOSE_NORMAL,
                 "GPU PDF pipeline: failed to add page %d to output\n",
                 page_idx + 1);
      failed_pages++;
    }

    // Cleanup
    sheet_process_state_cleanup(&state);

    // Print per-page perf stats
    if (options->perf) {
      perf_recorder_print(&perf, page_idx + 1, "cuda");
    }
  }

  // Close output PDF
  bool write_success = pdf_writer_close(writer);
  if (!write_success) {
    verboseLog(VERBOSE_NORMAL, "GPU PDF pipeline: failed to save output: %s\n",
               pdf_writer_get_last_error());
    failed_pages = page_count;
  }

  // Close input PDF
  pdf_close(doc);

  verboseLog(VERBOSE_NORMAL,
             "GPU PDF pipeline: complete. %d pages processed, %d failed\n",
             page_count, failed_pages);

  return failed_pages;

#else
  // Non-CUDA build
  (void)input_path;
  (void)output_path;
  (void)options;
  (void)config;
  verboseLog(VERBOSE_NORMAL, "GPU PDF pipeline: CUDA not compiled in\n");
  return -1;
#endif
}
