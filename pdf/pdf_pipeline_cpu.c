// SPDX-FileCopyrightText: 2025 The unpaper authors
//
// SPDX-License-Identifier: GPL-2.0-only

#include "pdf_pipeline_cpu.h"

#include "imageprocess/backend.h"
#include "imageprocess/image.h"
#include "lib/logging.h"
#include "lib/perf.h"
#include "pdf_reader.h"
#include "pdf_writer.h"
#include "sheet_process.h"

#ifdef UNPAPER_WITH_JBIG2
#include "lib/jbig2_decode.h"
#endif

#include <libavcodec/avcodec.h>
#include <libavformat/avformat.h>
#include <libavutil/avutil.h>
#include <libavutil/frame.h>
#include <libavutil/imgutils.h>
#include <libavutil/opt.h>
#include <libavutil/pixfmt.h>
#include <libswscale/swscale.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

// Default DPI for rendering pages when image extraction fails
#define PDF_RENDER_DPI 300

// JPEG quality for output (when not using direct embedding)
#define PDF_OUTPUT_JPEG_QUALITY 85

bool pdf_pipeline_is_pdf(const char *filename) {
  return pdf_is_pdf_file(filename);
}

// Decode JBIG2 image to AVFrame
#ifdef UNPAPER_WITH_JBIG2
static AVFrame *decode_jbig2_to_frame(const PdfImage *pdf_img) {
  if (pdf_img == NULL || pdf_img->data == NULL || pdf_img->size == 0) {
    return NULL;
  }

  Jbig2DecodedImage jbig2_img = {0};
  if (!jbig2_decode(pdf_img->data, pdf_img->size, pdf_img->jbig2_globals,
                    pdf_img->jbig2_globals_size, &jbig2_img)) {
    verboseLog(VERBOSE_MORE, "JBIG2 decode failed: %s\n",
               jbig2_get_last_error());
    return NULL;
  }

  // Create grayscale AVFrame
  AVFrame *frame = av_frame_alloc();
  if (frame == NULL) {
    jbig2_free_image(&jbig2_img);
    return NULL;
  }

  frame->width = (int)jbig2_img.width;
  frame->height = (int)jbig2_img.height;
  frame->format = AV_PIX_FMT_GRAY8;

  if (av_frame_get_buffer(frame, 0) < 0) {
    av_frame_free(&frame);
    jbig2_free_image(&jbig2_img);
    return NULL;
  }

  // Expand 1-bit to 8-bit grayscale
  // JBIG2 typically uses 1=black, 0=white (inverted from typical grayscale)
  if (!jbig2_expand_to_gray8(&jbig2_img, frame->data[0],
                             (size_t)frame->linesize[0], true)) {
    verboseLog(VERBOSE_MORE, "JBIG2 expand failed: %s\n",
               jbig2_get_last_error());
    av_frame_free(&frame);
    jbig2_free_image(&jbig2_img);
    return NULL;
  }

  jbig2_free_image(&jbig2_img);
  return frame;
}
#endif // UNPAPER_WITH_JBIG2

// Decode raw image bytes (JPEG/PNG/etc) using FFmpeg
// Returns an allocated AVFrame on success, NULL on failure
static AVFrame *decode_image_bytes(const uint8_t *data, size_t size,
                                   PdfImageFormat format) {
  if (data == NULL || size == 0) {
    return NULL;
  }

  // Only decode JPEG and PNG with FFmpeg
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
  AVIOContext *avio_ctx =
      avio_alloc_context(avio_ctx_buffer, (int)size, 0, NULL, NULL, NULL, NULL);
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
  }

cleanup:
  if (pkt) {
    av_packet_free(&pkt);
  }
  if (codec_ctx) {
    avcodec_free_context(&codec_ctx);
  }
  if (fmt_ctx) {
    avformat_close_input(&fmt_ctx);
  }
  if (avio_ctx) {
    avio_context_free(&avio_ctx);
  }

  return frame;
}

// Render page to AVFrame
static AVFrame *render_page_to_frame(PdfDocument *doc, int page_idx, int dpi) {
  int width = 0, height = 0, stride = 0;
  uint8_t *pixels =
      pdf_render_page(doc, page_idx, dpi, &width, &height, &stride);
  if (!pixels) {
    return NULL;
  }

  AVFrame *frame = av_frame_alloc();
  if (!frame) {
    free(pixels);
    return NULL;
  }

  frame->width = width;
  frame->height = height;
  frame->format = AV_PIX_FMT_RGB24;

  if (av_frame_get_buffer(frame, 0) < 0) {
    av_frame_free(&frame);
    free(pixels);
    return NULL;
  }

  // Copy pixels to frame
  for (int y = 0; y < height; y++) {
    memcpy(frame->data[0] + y * frame->linesize[0], pixels + y * stride,
           width * 3);
  }

  free(pixels);
  return frame;
}

// Encode Image to JPEG bytes
static uint8_t *encode_image_jpeg(Image *image, int quality, size_t *out_len) {
  if (image == NULL || image->frame == NULL || out_len == NULL) {
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

  ctx->width = image->frame->width;
  ctx->height = image->frame->height;
  ctx->time_base = (AVRational){1, 25};
  ctx->pix_fmt = AV_PIX_FMT_YUVJ420P;

  // Set quality
  ctx->qmin = 1;
  ctx->qmax = 31;
  int q = 31 - (quality * 30 / 100);
  if (q < 1)
    q = 1;
  if (q > 31)
    q = 31;
  ctx->global_quality = q * FF_QP2LAMBDA;
  ctx->flags |= AV_CODEC_FLAG_QSCALE;

  if (avcodec_open2(ctx, codec, NULL) < 0) {
    avcodec_free_context(&ctx);
    return NULL;
  }

  // Convert to YUV if needed
  AVFrame *yuv_frame = NULL;
  struct SwsContext *sws_ctx = NULL;

  if (image->frame->format != AV_PIX_FMT_YUVJ420P) {
    yuv_frame = av_frame_alloc();
    if (!yuv_frame) {
      avcodec_free_context(&ctx);
      return NULL;
    }
    yuv_frame->width = ctx->width;
    yuv_frame->height = ctx->height;
    yuv_frame->format = AV_PIX_FMT_YUVJ420P;

    if (av_frame_get_buffer(yuv_frame, 0) < 0) {
      av_frame_free(&yuv_frame);
      avcodec_free_context(&ctx);
      return NULL;
    }

    sws_ctx =
        sws_getContext(image->frame->width, image->frame->height,
                       image->frame->format, ctx->width, ctx->height,
                       AV_PIX_FMT_YUVJ420P, SWS_BILINEAR, NULL, NULL, NULL);

    if (!sws_ctx) {
      av_frame_free(&yuv_frame);
      avcodec_free_context(&ctx);
      return NULL;
    }

    sws_scale(sws_ctx, (const uint8_t *const *)image->frame->data,
              image->frame->linesize, 0, image->frame->height, yuv_frame->data,
              yuv_frame->linesize);

    sws_freeContext(sws_ctx);
  }

  AVFrame *enc_frame = yuv_frame ? yuv_frame : image->frame;

  AVPacket *pkt = av_packet_alloc();
  if (!pkt) {
    if (yuv_frame)
      av_frame_free(&yuv_frame);
    avcodec_free_context(&ctx);
    return NULL;
  }

  int ret = avcodec_send_frame(ctx, enc_frame);
  if (ret < 0) {
    av_packet_free(&pkt);
    if (yuv_frame)
      av_frame_free(&yuv_frame);
    avcodec_free_context(&ctx);
    return NULL;
  }

  ret = avcodec_receive_packet(ctx, pkt);
  if (ret < 0) {
    av_packet_free(&pkt);
    if (yuv_frame)
      av_frame_free(&yuv_frame);
    avcodec_free_context(&ctx);
    return NULL;
  }

  // Copy packet data
  uint8_t *jpeg_data = malloc(pkt->size);
  if (jpeg_data) {
    memcpy(jpeg_data, pkt->data, pkt->size);
    *out_len = pkt->size;
  }

  av_packet_free(&pkt);
  if (yuv_frame)
    av_frame_free(&yuv_frame);
  avcodec_free_context(&ctx);

  return jpeg_data;
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

  // Select CPU backend
  image_backend_select(UNPAPER_DEVICE_CPU);

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

  // Create modified options with write_output = false
  // so process_sheet() doesn't try to save files
  Options pdf_options = *options;
  pdf_options.write_output = false;

  // Create config with modified options
  SheetProcessConfig pdf_config = *config;
  pdf_config.options = &pdf_options;

  int failed_pages = 0;
  PerfRecorder perf;

  int quality = (options->jpeg_quality > 0) ? options->jpeg_quality
                                            : PDF_OUTPUT_JPEG_QUALITY;

  // Process each page
  for (int page_idx = 0; page_idx < page_count; page_idx++) {
    perf_recorder_init(&perf, options->perf, false);

    verboseLog(VERBOSE_NORMAL, "PDF pipeline: processing page %d/%d\n",
               page_idx + 1, page_count);

    AVFrame *page_frame = NULL;

    // Stage 1: Decode page to AVFrame
    perf_stage_begin(&perf, PERF_STAGE_DECODE);

    // Try to extract embedded image
    PdfImage pdf_img = {0};
    if (pdf_extract_page_image(doc, page_idx, &pdf_img)) {
      verboseLog(VERBOSE_MORE, "PDF pipeline: extracted %s image %dx%d\n",
                 pdf_image_format_name(pdf_img.format), pdf_img.width,
                 pdf_img.height);

#ifdef UNPAPER_WITH_JBIG2
      if (pdf_img.format == PDF_IMAGE_JBIG2) {
        page_frame = decode_jbig2_to_frame(&pdf_img);
        if (page_frame) {
          verboseLog(VERBOSE_MORE, "PDF pipeline: JBIG2 decoded %dx%d\n",
                     page_frame->width, page_frame->height);
        }
      } else
#endif
          if (pdf_img.format == PDF_IMAGE_JPEG ||
              pdf_img.format == PDF_IMAGE_PNG ||
              pdf_img.format == PDF_IMAGE_FLATE) {
        page_frame =
            decode_image_bytes(pdf_img.data, pdf_img.size, pdf_img.format);
        if (page_frame) {
          verboseLog(VERBOSE_MORE, "PDF pipeline: decoded %s %dx%d\n",
                     pdf_image_format_name(pdf_img.format), page_frame->width,
                     page_frame->height);
        }
      }

      pdf_free_image(&pdf_img);
    }

    // Fall back to rendering if extraction failed
    if (!page_frame) {
      page_frame = render_page_to_frame(doc, page_idx, PDF_RENDER_DPI);
      if (page_frame) {
        verboseLog(VERBOSE_MORE,
                   "PDF pipeline: rendered page at %d DPI (%dx%d)\n",
                   PDF_RENDER_DPI, page_frame->width, page_frame->height);
      }
    }

    perf_stage_end(&perf, PERF_STAGE_DECODE);

    if (!page_frame) {
      verboseLog(VERBOSE_NORMAL,
                 "PDF pipeline: failed to get image for page %d\n",
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

    // Set the decoded frame - process_sheet will use it
    sheet_process_state_set_decoded(&state, page_frame, 0);

    // Process the sheet (all filters, deskew, etc.)
    bool process_ok = process_sheet(&state, &pdf_config);

    if (!process_ok || state.sheet.frame == NULL) {
      verboseLog(VERBOSE_NORMAL,
                 "PDF pipeline: processing failed for page %d\n", page_idx + 1);
      sheet_process_state_cleanup(&state);
      failed_pages++;
      continue;
    }

    // Stage 3: Encode and add to output PDF
    perf_stage_begin(&perf, PERF_STAGE_ENCODE);

    image_ensure_cpu(&state.sheet);

    int out_width = state.sheet.frame->width;
    int out_height = state.sheet.frame->height;

    size_t jpeg_len = 0;
    uint8_t *jpeg_data = encode_image_jpeg(&state.sheet, quality, &jpeg_len);

    bool page_success = false;
    if (jpeg_data && jpeg_len > 0) {
      page_success = pdf_writer_add_page_jpeg(
          writer, jpeg_data, jpeg_len, out_width, out_height, PDF_RENDER_DPI);
      free(jpeg_data);
    }

    if (!page_success) {
      // Fall back to pixel embedding
      verboseLog(VERBOSE_MORE,
                 "PDF pipeline: JPEG encode failed, using pixels\n");
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
      verboseLog(VERBOSE_NORMAL,
                 "PDF pipeline: failed to add page %d to output\n",
                 page_idx + 1);
      failed_pages++;
    }

    sheet_process_state_cleanup(&state);

    if (options->perf) {
      perf_recorder_print(&perf, page_idx + 1, "cpu");
    }
  }

  // Close output PDF
  bool write_success = pdf_writer_close(writer);
  if (!write_success) {
    verboseLog(VERBOSE_NORMAL, "PDF pipeline: failed to save output: %s\n",
               pdf_writer_get_last_error());
    failed_pages = page_count;
  }

  pdf_close(doc);

  verboseLog(VERBOSE_NORMAL,
             "PDF pipeline: complete. %d pages processed, %d failed\n",
             page_count, failed_pages);

  return failed_pages;
}
