// SPDX-FileCopyrightText: 2025 The unpaper authors
//
// SPDX-License-Identifier: GPL-2.0-only

#include "pdf_pipeline_decode.h"

#include "lib/logging.h"

#ifdef UNPAPER_WITH_JBIG2
#include "lib/jbig2_decode.h"
#endif

#include <libavcodec/avcodec.h>
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

  // JBIG2 bitmaps are 1=black, 0=white. unpaper expects grayscale where
  // black=0 and white=255, so do not invert here.
  if (!jbig2_expand_to_gray8(&jbig2_img, frame->data[0],
                             (size_t)frame->linesize[0], false)) {
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

static AVFrame *decode_codec_bytes(const uint8_t *data, size_t size,
                                   enum AVCodecID codec_id) {
  if (data == NULL || size == 0) {
    return NULL;
  }

  const AVCodec *codec = avcodec_find_decoder(codec_id);
  if (codec == NULL) {
    return NULL;
  }

  AVCodecContext *codec_ctx = avcodec_alloc_context3(codec);
  if (codec_ctx == NULL) {
    return NULL;
  }

  if (avcodec_open2(codec_ctx, codec, NULL) < 0) {
    avcodec_free_context(&codec_ctx);
    return NULL;
  }

  AVPacket *pkt = av_packet_alloc();
  if (pkt == NULL) {
    avcodec_free_context(&codec_ctx);
    return NULL;
  }

  int ret = av_new_packet(pkt, (int)size);
  if (ret < 0) {
    av_packet_free(&pkt);
    avcodec_free_context(&codec_ctx);
    return NULL;
  }
  memcpy(pkt->data, data, size);

  ret = avcodec_send_packet(codec_ctx, pkt);
  av_packet_free(&pkt);
  if (ret < 0) {
    avcodec_free_context(&codec_ctx);
    return NULL;
  }

  AVFrame *frame = av_frame_alloc();
  if (frame == NULL) {
    avcodec_free_context(&codec_ctx);
    return NULL;
  }

  ret = avcodec_receive_frame(codec_ctx, frame);
  avcodec_free_context(&codec_ctx);
  if (ret < 0) {
    av_frame_free(&frame);
    return NULL;
  }

  return frame;
}

static AVFrame *decode_image_bytes(const uint8_t *data, size_t size,
                                   PdfImageFormat format) {
  if (data == NULL || size == 0) {
    return NULL;
  }

  if (format == PDF_IMAGE_JPEG) {
    return decode_codec_bytes(data, size, AV_CODEC_ID_MJPEG);
  }
  if (format == PDF_IMAGE_PNG) {
    return decode_codec_bytes(data, size, AV_CODEC_ID_PNG);
  }

  // Flate-compressed PDF image streams are not necessarily standalone PNGs;
  // fall back to MuPDF rendering when we can't decode directly.
  return NULL;
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

static AVFrame *pdf_pipeline_render_page_to_frame_size(PdfDocument *doc,
                                                       int page_idx,
                                                       int target_width,
                                                       int target_height) {
  int width = 0, height = 0, stride = 0;
  uint8_t *pixels = pdf_render_page_to_size(doc, page_idx, target_width,
                                            target_height, &width, &height,
                                            &stride);
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
           pixels + (size_t)y * (size_t)stride, (size_t)width * 3);
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
  int extracted_w = 0;
  int extracted_h = 0;

  PdfImage pdf_img = {0};
  if (pdf_extract_page_image(doc, page_idx, &pdf_img)) {
    extracted_w = pdf_img.width;
    extracted_h = pdf_img.height;
    frame = pdf_pipeline_decode_image_to_frame(&pdf_img);
    pdf_free_image(&pdf_img);
  }

  if (!frame) {
    if (extracted_w > 0 && extracted_h > 0) {
      frame = pdf_pipeline_render_page_to_frame_size(doc, page_idx, extracted_w,
                                                     extracted_h);
    } else {
      frame = pdf_pipeline_render_page_to_frame(doc, page_idx, dpi);
    }
  }

  return frame;
}
