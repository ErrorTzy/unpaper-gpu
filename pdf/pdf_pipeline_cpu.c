// SPDX-FileCopyrightText: 2025 The unpaper authors
//
// SPDX-License-Identifier: GPL-2.0-only

#include "pdf_pipeline_cpu.h"

#include "imageprocess/blit.h"
#include "imageprocess/deskew.h"
#include "imageprocess/filters.h"
#include "imageprocess/image.h"
#include "imageprocess/masks.h"
#include "imageprocess/pixel.h"
#include "lib/logging.h"
#include "lib/perf.h"
#include "pdf_reader.h"
#include "pdf_writer.h"

#include <libavcodec/avcodec.h>
#include <libavutil/opt.h>
#include <libavformat/avformat.h>
#include <libavutil/avutil.h>
#include <libavutil/frame.h>
#include <libavutil/imgutils.h>
#include <libavutil/pixfmt.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

// Default DPI for rendering pages when image extraction fails
#define PDF_RENDER_DPI 300

// JPEG quality for output (when not using direct embedding)
#define PDF_OUTPUT_JPEG_QUALITY 85

bool pdf_pipeline_is_pdf(const char *filename) { return pdf_is_pdf_file(filename); }

// Decode raw image bytes (JPEG/PNG/etc) using FFmpeg
// Returns an allocated AVFrame on success, NULL on failure
static AVFrame *decode_image_bytes(const uint8_t *data, size_t size,
                                   PdfImageFormat format) {
  if (data == NULL || size == 0) {
    return NULL;
  }

  // Only decode JPEG and PNG with FFmpeg
  // Other formats (JBIG2, CCITT, JP2) would need specialized decoders
  if (format != PDF_IMAGE_JPEG && format != PDF_IMAGE_PNG &&
      format != PDF_IMAGE_FLATE) {
    return NULL;
  }

  AVFormatContext *fmt_ctx = NULL;
  AVCodecContext *codec_ctx = NULL;
  const AVCodec *codec = NULL;
  AVFrame *frame = NULL;
  AVPacket *pkt = NULL;
  int ret;

  // Create input buffer for probing
  uint8_t *avio_ctx_buffer = av_malloc(size);
  if (!avio_ctx_buffer) {
    return NULL;
  }
  memcpy(avio_ctx_buffer, data, size);

  // Create custom AVIOContext from memory
  AVIOContext *avio_ctx = avio_alloc_context(avio_ctx_buffer, (int)size, 0,
                                              NULL, NULL, NULL, NULL);
  if (!avio_ctx) {
    av_free(avio_ctx_buffer);
    return NULL;
  }

  fmt_ctx = avformat_alloc_context();
  if (!fmt_ctx) {
    avio_context_free(&avio_ctx);
    return NULL;
  }
  fmt_ctx->pb = avio_ctx;

  // Open input
  ret = avformat_open_input(&fmt_ctx, NULL, NULL, NULL);
  if (ret < 0) {
    goto cleanup;
  }

  ret = avformat_find_stream_info(fmt_ctx, NULL);
  if (ret < 0) {
    goto cleanup;
  }

  if (fmt_ctx->nb_streams < 1) {
    goto cleanup;
  }

  codec = avcodec_find_decoder(fmt_ctx->streams[0]->codecpar->codec_id);
  if (!codec) {
    goto cleanup;
  }

  codec_ctx = avcodec_alloc_context3(codec);
  if (!codec_ctx) {
    goto cleanup;
  }

  ret = avcodec_parameters_to_context(codec_ctx, fmt_ctx->streams[0]->codecpar);
  if (ret < 0) {
    goto cleanup;
  }

  ret = avcodec_open2(codec_ctx, codec, NULL);
  if (ret < 0) {
    goto cleanup;
  }

  pkt = av_packet_alloc();
  if (!pkt) {
    goto cleanup;
  }

  ret = av_read_frame(fmt_ctx, pkt);
  if (ret < 0) {
    goto cleanup;
  }

  ret = avcodec_send_packet(codec_ctx, pkt);
  if (ret < 0) {
    goto cleanup;
  }

  frame = av_frame_alloc();
  if (!frame) {
    goto cleanup;
  }

  ret = avcodec_receive_frame(codec_ctx, frame);
  if (ret < 0) {
    av_frame_free(&frame);
    frame = NULL;
    goto cleanup;
  }

cleanup:
  av_packet_free(&pkt);
  avcodec_free_context(&codec_ctx);
  if (fmt_ctx) {
    // Note: avformat_close_input will free avio_ctx_buffer via avio_ctx
    avformat_close_input(&fmt_ctx);
  }
  if (avio_ctx) {
    av_freep(&avio_ctx->buffer);
    avio_context_free(&avio_ctx);
  }

  return frame;
}

// Create an Image from an AVFrame
static Image image_from_frame(AVFrame *frame, Pixel background,
                              uint8_t abs_black_threshold) {
  if (frame == NULL) {
    return EMPTY_IMAGE;
  }

  RectangleSize size = {.width = frame->width, .height = frame->height};
  Image img;

  switch (frame->format) {
  case AV_PIX_FMT_Y400A:
  case AV_PIX_FMT_GRAY8:
  case AV_PIX_FMT_RGB24:
  case AV_PIX_FMT_MONOBLACK:
  case AV_PIX_FMT_MONOWHITE:
    img = create_image(size, frame->format, false, background,
                       abs_black_threshold);
    av_frame_free(&img.frame);
    img.frame = av_frame_clone(frame);
    break;

  case AV_PIX_FMT_PAL8: {
    // Convert palette to RGB24
    img = create_image(size, AV_PIX_FMT_RGB24, false, background,
                       abs_black_threshold);
    const uint32_t *palette = (const uint32_t *)frame->data[1];
    for (int y = 0; y < frame->height; y++) {
      for (int x = 0; x < frame->width; x++) {
        const uint8_t idx = frame->data[0][y * frame->linesize[0] + x];
        set_pixel(img, (Point){x, y}, pixel_from_value(palette[idx]));
      }
    }
  } break;

  case AV_PIX_FMT_YUV420P:
  case AV_PIX_FMT_YUVJ420P:
  case AV_PIX_FMT_YUV422P:
  case AV_PIX_FMT_YUVJ422P:
  case AV_PIX_FMT_YUV444P:
  case AV_PIX_FMT_YUVJ444P: {
    // Convert YUV to RGB24 manually
    img = create_image(size, AV_PIX_FMT_RGB24, false, background,
                       abs_black_threshold);

    // Determine subsampling
    int ss_h = 1, ss_v = 1;
    if (frame->format == AV_PIX_FMT_YUV420P ||
        frame->format == AV_PIX_FMT_YUVJ420P) {
      ss_h = 2;
      ss_v = 2;
    } else if (frame->format == AV_PIX_FMT_YUV422P ||
               frame->format == AV_PIX_FMT_YUVJ422P) {
      ss_h = 2;
      ss_v = 1;
    }

    for (int y = 0; y < frame->height; y++) {
      for (int x = 0; x < frame->width; x++) {
        int Y = frame->data[0][y * frame->linesize[0] + x];
        int U = frame->data[1][(y / ss_v) * frame->linesize[1] + (x / ss_h)];
        int V = frame->data[2][(y / ss_v) * frame->linesize[2] + (x / ss_h)];

        // YUV to RGB conversion (BT.601)
        int C = Y - 16;
        int D = U - 128;
        int E = V - 128;

        int R = (298 * C + 409 * E + 128) >> 8;
        int G = (298 * C - 100 * D - 208 * E + 128) >> 8;
        int B = (298 * C + 516 * D + 128) >> 8;

        // Clamp
        if (R < 0)
          R = 0;
        if (R > 255)
          R = 255;
        if (G < 0)
          G = 0;
        if (G > 255)
          G = 255;
        if (B < 0)
          B = 0;
        if (B > 255)
          B = 255;

        uint8_t *dst =
            img.frame->data[0] + y * img.frame->linesize[0] + x * 3;
        dst[0] = (uint8_t)R;
        dst[1] = (uint8_t)G;
        dst[2] = (uint8_t)B;
      }
    }
  } break;

  default:
    // Unsupported format
    return EMPTY_IMAGE;
  }

  return img;
}

// Create an Image from raw RGB/gray pixels
static Image image_from_pixels(const uint8_t *pixels, int width, int height,
                               int stride, int components, Pixel background,
                               uint8_t abs_black_threshold) {
  if (pixels == NULL || width <= 0 || height <= 0) {
    return EMPTY_IMAGE;
  }

  enum AVPixelFormat fmt =
      (components == 1) ? AV_PIX_FMT_GRAY8 : AV_PIX_FMT_RGB24;
  RectangleSize size = {.width = width, .height = height};

  Image img = create_image(size, fmt, false, background, abs_black_threshold);

  // Copy pixels
  int bytes_per_row = width * components;
  for (int y = 0; y < height; y++) {
    memcpy(img.frame->data[0] + y * img.frame->linesize[0],
           pixels + y * stride, bytes_per_row);
  }

  return img;
}

// Encode an image to JPEG using FFmpeg
// Returns allocated buffer and size, caller must free
static uint8_t *encode_image_jpeg(const Image *img, int quality, size_t *out_len) {
  if (img == NULL || img->frame == NULL || out_len == NULL) {
    return NULL;
  }

  *out_len = 0;

  const AVCodec *codec = avcodec_find_encoder(AV_CODEC_ID_MJPEG);
  if (!codec) {
    return NULL;
  }

  AVCodecContext *ctx = avcodec_alloc_context3(codec);
  if (!ctx) {
    return NULL;
  }

  // Determine input format and set encoder parameters
  ctx->width = img->frame->width;
  ctx->height = img->frame->height;
  ctx->time_base = (AVRational){1, 1};

  // MJPEG encoder requires YUVJ420P or YUVJ444P
  // We'll convert from RGB24 or GRAY8
  if (img->frame->format == AV_PIX_FMT_GRAY8) {
    ctx->pix_fmt = AV_PIX_FMT_YUVJ420P;
  } else {
    ctx->pix_fmt = AV_PIX_FMT_YUVJ420P;
  }

  // Set quality (1-100 -> qmin/qmax mapping)
  // FFmpeg MJPEG uses 2-31 qscale, lower = better
  int qscale = 31 - (quality * 29 / 100);
  if (qscale < 2)
    qscale = 2;
  if (qscale > 31)
    qscale = 31;

  av_opt_set_int(ctx->priv_data, "qscale", qscale, 0);
  ctx->flags |= AV_CODEC_FLAG_QSCALE;
  ctx->global_quality = FF_QP2LAMBDA * qscale;

  int ret = avcodec_open2(ctx, codec, NULL);
  if (ret < 0) {
    avcodec_free_context(&ctx);
    return NULL;
  }

  // Create frame for encoding (convert format if needed)
  AVFrame *enc_frame = av_frame_alloc();
  if (!enc_frame) {
    avcodec_free_context(&ctx);
    return NULL;
  }

  enc_frame->width = ctx->width;
  enc_frame->height = ctx->height;
  enc_frame->format = ctx->pix_fmt;

  ret = av_frame_get_buffer(enc_frame, 0);
  if (ret < 0) {
    av_frame_free(&enc_frame);
    avcodec_free_context(&ctx);
    return NULL;
  }

  // Convert input to YUVJ420P
  if (img->frame->format == AV_PIX_FMT_GRAY8) {
    // Grayscale -> YUVJ420P: Y = gray, U = V = 128
    for (int y = 0; y < ctx->height; y++) {
      memcpy(enc_frame->data[0] + y * enc_frame->linesize[0],
             img->frame->data[0] + y * img->frame->linesize[0], ctx->width);
    }
    // Fill UV with 128 (neutral gray)
    int uv_height = (ctx->height + 1) / 2;
    int uv_width = (ctx->width + 1) / 2;
    for (int y = 0; y < uv_height; y++) {
      memset(enc_frame->data[1] + y * enc_frame->linesize[1], 128, uv_width);
      memset(enc_frame->data[2] + y * enc_frame->linesize[2], 128, uv_width);
    }
  } else if (img->frame->format == AV_PIX_FMT_RGB24) {
    // RGB24 -> YUVJ420P conversion
    for (int y = 0; y < ctx->height; y++) {
      const uint8_t *src =
          img->frame->data[0] + y * img->frame->linesize[0];
      uint8_t *dst_y = enc_frame->data[0] + y * enc_frame->linesize[0];
      for (int x = 0; x < ctx->width; x++) {
        int R = src[x * 3 + 0];
        int G = src[x * 3 + 1];
        int B = src[x * 3 + 2];
        // BT.601 full-range
        dst_y[x] = (uint8_t)((66 * R + 129 * G + 25 * B + 128) / 256 + 16);
      }
    }
    // UV planes (subsampled)
    int uv_height = (ctx->height + 1) / 2;
    int uv_width = (ctx->width + 1) / 2;
    for (int y = 0; y < uv_height; y++) {
      const uint8_t *src0 =
          img->frame->data[0] + (y * 2) * img->frame->linesize[0];
      const uint8_t *src1 =
          (y * 2 + 1 < ctx->height)
              ? img->frame->data[0] + (y * 2 + 1) * img->frame->linesize[0]
              : src0;
      uint8_t *dst_u = enc_frame->data[1] + y * enc_frame->linesize[1];
      uint8_t *dst_v = enc_frame->data[2] + y * enc_frame->linesize[2];
      for (int x = 0; x < uv_width; x++) {
        // Average 2x2 block
        int x0 = x * 2;
        int x1 = (x * 2 + 1 < ctx->width) ? x * 2 + 1 : x0;
        int R = (src0[x0 * 3 + 0] + src0[x1 * 3 + 0] + src1[x0 * 3 + 0] +
                 src1[x1 * 3 + 0] + 2) /
                4;
        int G = (src0[x0 * 3 + 1] + src0[x1 * 3 + 1] + src1[x0 * 3 + 1] +
                 src1[x1 * 3 + 1] + 2) /
                4;
        int B = (src0[x0 * 3 + 2] + src0[x1 * 3 + 2] + src1[x0 * 3 + 2] +
                 src1[x1 * 3 + 2] + 2) /
                4;
        dst_u[x] = (uint8_t)((-38 * R - 74 * G + 112 * B + 128) / 256 + 128);
        dst_v[x] = (uint8_t)((112 * R - 94 * G - 18 * B + 128) / 256 + 128);
      }
    }
  } else {
    // Unsupported input format
    av_frame_free(&enc_frame);
    avcodec_free_context(&ctx);
    return NULL;
  }

  // Encode
  AVPacket *pkt = av_packet_alloc();
  if (!pkt) {
    av_frame_free(&enc_frame);
    avcodec_free_context(&ctx);
    return NULL;
  }

  ret = avcodec_send_frame(ctx, enc_frame);
  if (ret < 0) {
    av_packet_free(&pkt);
    av_frame_free(&enc_frame);
    avcodec_free_context(&ctx);
    return NULL;
  }

  ret = avcodec_receive_packet(ctx, pkt);
  if (ret < 0) {
    av_packet_free(&pkt);
    av_frame_free(&enc_frame);
    avcodec_free_context(&ctx);
    return NULL;
  }

  // Copy output
  uint8_t *out = malloc(pkt->size);
  if (out) {
    memcpy(out, pkt->data, pkt->size);
    *out_len = pkt->size;
  }

  av_packet_free(&pkt);
  av_frame_free(&enc_frame);
  avcodec_free_context(&ctx);

  return out;
}

int pdf_pipeline_cpu_process(const char *input_path, const char *output_path,
                             const Options *options,
                             const SheetProcessConfig *config) {
  if (input_path == NULL || output_path == NULL || options == NULL ||
      config == NULL) {
    verboseLog(VERBOSE_NORMAL, "PDF pipeline: invalid arguments\n");
    return -1;
  }

  verboseLog(VERBOSE_NORMAL, "PDF pipeline: %s -> %s\n", input_path,
             output_path);

  // Open input PDF
  PdfDocument *doc = pdf_open(input_path);
  if (doc == NULL) {
    verboseLog(VERBOSE_NORMAL, "PDF pipeline: failed to open input: %s\n",
               pdf_get_last_error());
    return -1;
  }

  int page_count = pdf_page_count(doc);
  if (page_count <= 0) {
    verboseLog(VERBOSE_NORMAL, "PDF pipeline: no pages in document\n");
    pdf_close(doc);
    return -1;
  }

  verboseLog(VERBOSE_NORMAL, "PDF pipeline: %d pages to process\n", page_count);

  // Get metadata for output
  PdfMetadata meta = pdf_get_metadata(doc);

  // Create output PDF writer
  PdfWriter *writer = pdf_writer_create(output_path, &meta, PDF_RENDER_DPI);
  if (writer == NULL) {
    verboseLog(VERBOSE_NORMAL, "PDF pipeline: failed to create output: %s\n",
               pdf_writer_get_last_error());
    pdf_free_metadata(&meta);
    pdf_close(doc);
    return -1;
  }

  pdf_free_metadata(&meta);

  int failed_pages = 0;
  PerfRecorder perf;

  // Process each page
  for (int page_idx = 0; page_idx < page_count; page_idx++) {
    perf_recorder_init(&perf, options->perf,
                       options->device == UNPAPER_DEVICE_CUDA);

    verboseLog(VERBOSE_NORMAL, "PDF pipeline: processing page %d/%d\n",
               page_idx + 1, page_count);

    Image page_image = EMPTY_IMAGE;
    int width = 0, height = 0;
    bool extracted = false;

    // Stage 1: Extract or render page
    perf_stage_begin(&perf, PERF_STAGE_DECODE);

    // Try to extract embedded image first (zero-copy path)
    PdfImage pdf_img = {0};
    if (pdf_extract_page_image(doc, page_idx, &pdf_img)) {
      verboseLog(VERBOSE_MORE, "PDF pipeline: extracted %s image %dx%d\n",
                 pdf_image_format_name(pdf_img.format), pdf_img.width,
                 pdf_img.height);

      // Try to decode with FFmpeg
      AVFrame *frame = decode_image_bytes(pdf_img.data, pdf_img.size,
                                          pdf_img.format);
      if (frame) {
        page_image = image_from_frame(frame, options->sheet_background,
                                      options->abs_black_threshold);
        av_frame_free(&frame);
        if (page_image.frame != NULL) {
          width = page_image.frame->width;
          height = page_image.frame->height;
          extracted = true;
        }
      }
      pdf_free_image(&pdf_img);
    }

    // Fall back to rendering if extraction failed
    if (!extracted) {
      int stride = 0;
      uint8_t *pixels =
          pdf_render_page(doc, page_idx, PDF_RENDER_DPI, &width, &height, &stride);
      if (pixels) {
        verboseLog(VERBOSE_MORE,
                   "PDF pipeline: rendered page at %d DPI (%dx%d)\n",
                   PDF_RENDER_DPI, width, height);
        page_image = image_from_pixels(pixels, width, height, stride, 3,
                                       options->sheet_background,
                                       options->abs_black_threshold);
        free(pixels);
      }
    }

    perf_stage_end(&perf, PERF_STAGE_DECODE);

    if (page_image.frame == NULL) {
      verboseLog(VERBOSE_NORMAL,
                 "PDF pipeline: failed to get image for page %d\n",
                 page_idx + 1);
      failed_pages++;
      continue;
    }

    // Stage 2: Process the image using existing sheet processing
    // We create a BatchJob-like structure for process_sheet
    BatchJob job = {0};
    job.sheet_nr = page_idx + 1;
    job.input_count = 1;
    job.output_count = 1;
    job.input_files[0] = NULL;  // We're passing the image directly
    job.output_files[0] = NULL; // We'll encode after processing

    SheetProcessState state;
    sheet_process_state_init(&state, config, &job);

    // Transfer ownership of the page image to the state
    // The sheet processing expects images in state.page or state.sheet
    state.page = page_image;
    state.sheet = EMPTY_IMAGE;
    state.input_size = size_of_image(page_image);

    // Create sheet from page (single-page layout)
    state.sheet = create_image(state.input_size, AV_PIX_FMT_RGB24, true,
                               options->sheet_background,
                               options->abs_black_threshold);

    // Copy page to sheet
    center_image(state.page, state.sheet, POINT_ORIGIN, state.input_size);
    free_image(&state.page);
    state.page = EMPTY_IMAGE;

    // Run the processing pipeline (filters, deskew, etc.)
    perf_stage_begin(&state.perf, PERF_STAGE_FILTERS);

    // Apply processing operations that don't require parameter initialization
    // Note: Filters (blackfilter, blurfilter, grayfilter) are disabled by
    // default in PDF mode because their parameters aren't initialized through
    // command-line parsing. Use the full unpaper CLI for filter support.

    // Pre-mirroring
    if (options->pre_mirror.horizontal || options->pre_mirror.vertical) {
      mirror(state.sheet, options->pre_mirror);
    }

    // Pre-shifting
    if (options->pre_shift.horizontal != 0 ||
        options->pre_shift.vertical != 0) {
      shift_image(&state.sheet, options->pre_shift);
    }

    // Noisefilter - only requires intensity, which is initialized by
    // options_init
    if (!isExcluded(state.sheet_nr, options->no_noisefilter_multi_index,
                    options->ignore_multi_index)) {
      noisefilter(state.sheet, options->noisefilter_intensity,
                  options->abs_white_threshold);
    }

    // Post-mirroring
    if (options->post_mirror.horizontal || options->post_mirror.vertical) {
      mirror(state.sheet, options->post_mirror);
    }

    // Post-shifting
    if (options->post_shift.horizontal != 0 ||
        options->post_shift.vertical != 0) {
      shift_image(&state.sheet, options->post_shift);
    }

    // Post-rotating
    if (options->post_rotate != 0) {
      flip_rotate_90(&state.sheet, options->post_rotate / 90);
    }

    perf_stage_end(&state.perf, PERF_STAGE_FILTERS);

    // Stage 3: Encode processed image and add to output PDF
    perf_stage_begin(&perf, PERF_STAGE_ENCODE);

    // Ensure image is on CPU
    image_ensure_cpu(&state.sheet);

    int out_width = state.sheet.frame->width;
    int out_height = state.sheet.frame->height;

    // Encode to JPEG
    size_t jpeg_len = 0;
    int quality = (options->jpeg_quality > 0) ? options->jpeg_quality
                                               : PDF_OUTPUT_JPEG_QUALITY;
    uint8_t *jpeg_data = encode_image_jpeg(&state.sheet, quality, &jpeg_len);

    bool page_success = false;
    if (jpeg_data && jpeg_len > 0) {
      // Add JPEG page to output PDF
      page_success = pdf_writer_add_page_jpeg(writer, jpeg_data, jpeg_len,
                                               out_width, out_height,
                                               PDF_RENDER_DPI);
      free(jpeg_data);
    }

    if (!page_success) {
      // Fall back to pixel embedding
      verboseLog(VERBOSE_MORE, "PDF pipeline: JPEG encode failed, using pixels\n");
      int stride = state.sheet.frame->linesize[0];
      PdfPixelFormat fmt = (state.sheet.frame->format == AV_PIX_FMT_GRAY8)
                               ? PDF_PIXEL_GRAY8
                               : PDF_PIXEL_RGB24;
      page_success = pdf_writer_add_page_pixels(
          writer, state.sheet.frame->data[0], out_width, out_height, stride,
          fmt, PDF_RENDER_DPI);
    }

    perf_stage_end(&perf, PERF_STAGE_ENCODE);

    if (!page_success) {
      verboseLog(VERBOSE_NORMAL, "PDF pipeline: failed to add page %d to output\n",
                 page_idx + 1);
      failed_pages++;
    }

    // Cleanup
    sheet_process_state_cleanup(&state);

    // Print per-page perf stats
    if (options->perf) {
      perf_recorder_print(&perf, page_idx + 1, "cpu");
    }
  }

  // Close output PDF
  bool write_success = pdf_writer_close(writer);
  if (!write_success) {
    verboseLog(VERBOSE_NORMAL, "PDF pipeline: failed to save output: %s\n",
               pdf_writer_get_last_error());
    failed_pages = page_count; // Mark all as failed
  }

  // Close input PDF
  pdf_close(doc);

  verboseLog(VERBOSE_NORMAL,
             "PDF pipeline: complete. %d pages processed, %d failed\n",
             page_count, failed_pages);

  return failed_pages;
}
