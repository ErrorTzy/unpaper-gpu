// SPDX-FileCopyrightText: 2025 The unpaper authors
//
// SPDX-License-Identifier: GPL-2.0-only

#include "pdf_pipeline_decode.h"

#include "lib/logging.h"

#ifdef UNPAPER_WITH_JBIG2
#include "lib/jbig2_decode.h"
#endif

#include <libavcodec/avcodec.h>
#include <libavformat/avformat.h>
#include <libavutil/avutil.h>
#include <libavutil/frame.h>
#include <libavutil/imgutils.h>
#include <libavutil/pixfmt.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

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
#endif

static AVFrame *decode_image_bytes(const uint8_t *data, size_t size,
                                   PdfImageFormat format) {
  if (data == NULL || size == 0) {
    return NULL;
  }

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

  uint8_t *avio_ctx_buffer = av_malloc(size);
  if (!avio_ctx_buffer) {
    return NULL;
  }
  memcpy(avio_ctx_buffer, data, size);

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

  ret = avformat_open_input(&fmt_ctx, NULL, NULL, NULL);
  if (ret < 0) {
    goto cleanup;
  }

  ret = avformat_find_stream_info(fmt_ctx, NULL);
  if (ret < 0 || fmt_ctx->nb_streams < 1) {
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

AVFrame *pdf_pipeline_render_page_to_frame(PdfDocument *doc, int page_idx,
                                           int dpi) {
  int width = 0, height = 0, stride = 0;
  uint8_t *pixels = pdf_render_page(doc, page_idx, dpi, &width, &height, &stride);
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

  for (int y = 0; y < height; y++) {
    memcpy(frame->data[0] + y * frame->linesize[0],
           pixels + y * stride, (size_t)width * 3);
  }

  free(pixels);
  return frame;
}

AVFrame *pdf_pipeline_decode_image_to_frame(const PdfImage *pdf_img) {
  if (pdf_img == NULL) {
    return NULL;
  }

#ifdef UNPAPER_WITH_JBIG2
  if (pdf_img->format == PDF_IMAGE_JBIG2) {
    return decode_jbig2_to_frame(pdf_img);
  }
#endif

  if (pdf_img->format == PDF_IMAGE_JPEG || pdf_img->format == PDF_IMAGE_PNG ||
      pdf_img->format == PDF_IMAGE_FLATE) {
    return decode_image_bytes(pdf_img->data, pdf_img->size, pdf_img->format);
  }

  return NULL;
}

AVFrame *pdf_pipeline_decode_page_to_frame(PdfDocument *doc, int page_idx,
                                           int dpi) {
  if (!doc || page_idx < 0 || dpi <= 0) {
    return NULL;
  }

  AVFrame *frame = NULL;

  PdfImage pdf_img = {0};
  if (pdf_extract_page_image(doc, page_idx, &pdf_img)) {
    frame = pdf_pipeline_decode_image_to_frame(&pdf_img);
    pdf_free_image(&pdf_img);
  }

  if (!frame) {
    frame = pdf_pipeline_render_page_to_frame(doc, page_idx, dpi);
  }

  return frame;
}

