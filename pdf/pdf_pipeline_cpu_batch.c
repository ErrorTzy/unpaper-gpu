// SPDX-FileCopyrightText: 2025 The unpaper authors
//
// SPDX-License-Identifier: GPL-2.0-only

#include "pdf_pipeline_cpu_batch.h"

#include "imageprocess/backend.h"
#include "imageprocess/image.h"
#include "lib/batch.h"
#include "lib/batch_worker.h"
#include "lib/decode_queue.h"
#include "lib/logging.h"
#include "lib/threadpool.h"
#include "pdf_pipeline_decode.h"
#include "pdf_page_accumulator.h"
#include "pdf_reader.h"
#include "pdf_writer.h"

#include <libavcodec/avcodec.h>
#include <libavutil/avutil.h>
#include <libavutil/frame.h>
#include <libavutil/pixfmt.h>
#include <libswscale/swscale.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define PDF_RENDER_DPI 300
#define PDF_OUTPUT_JPEG_QUALITY 85
#define MAX_DECODE_QUEUE_DEPTH 32

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

    sws_ctx = sws_getContext(image->frame->width, image->frame->height,
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

  uint8_t *jpeg_data = malloc((size_t)pkt->size);
  if (jpeg_data) {
    memcpy(jpeg_data, pkt->data, (size_t)pkt->size);
    *out_len = (size_t)pkt->size;
  }

  av_packet_free(&pkt);
  if (yuv_frame)
    av_frame_free(&yuv_frame);
  avcodec_free_context(&ctx);

  return jpeg_data;
}

typedef struct {
  PdfDocument *doc;
  int render_dpi;
} PdfDecodeQueueContext;

static bool pdf_decode_queue_decoder(void *user_ctx, const BatchJob *job,
                                     int job_index, int input_index,
                                     DecodedImage *out) {
  (void)job;
  (void)input_index;

  PdfDecodeQueueContext *ctx = (PdfDecodeQueueContext *)user_ctx;
  if (!ctx || !ctx->doc || !out) {
    return false;
  }

  AVFrame *frame =
      pdf_pipeline_decode_page_to_frame(ctx->doc, job_index, ctx->render_dpi);
  out->frame = frame;
  out->valid = (frame != NULL);
  out->uses_pinned_memory = false;
  out->on_gpu = false;
  out->gpu_ptr = NULL;
  out->gpu_completion_event = NULL;
  out->gpu_event_from_pool = false;
  return out->valid;
}

typedef struct {
  PdfPageAccumulator *accumulator;
  const Options *options;
} PdfBatchWriteContext;

static bool pdf_batch_post_process(BatchWorkerContext *worker_ctx,
                                   size_t job_index, SheetProcessState *state,
                                   void *user_ctx) {
  (void)worker_ctx;

  PdfBatchWriteContext *ctx = (PdfBatchWriteContext *)user_ctx;
  if (!ctx || !ctx->accumulator || !ctx->options || !state) {
    return false;
  }

  if (state->sheet.frame == NULL) {
    pdf_page_accumulator_mark_failed(ctx->accumulator, (int)job_index);
    return false;
  }

  image_ensure_cpu(&state->sheet);

  int out_width = state->sheet.frame->width;
  int out_height = state->sheet.frame->height;
  int stride = state->sheet.frame->linesize[0];
  int dpi = ctx->options->pdf_render_dpi > 0 ? ctx->options->pdf_render_dpi
                                             : PDF_RENDER_DPI;

  int quality = (ctx->options->jpeg_quality > 0) ? ctx->options->jpeg_quality
                                                 : PDF_OUTPUT_JPEG_QUALITY;

  size_t jpeg_len = 0;
  uint8_t *jpeg_data = encode_image_jpeg(&state->sheet, quality, &jpeg_len);

  if (jpeg_data && jpeg_len > 0) {
    PdfEncodedPage page = {0};
    page.page_index = (int)job_index;
    page.type = PDF_PAGE_DATA_JPEG;
    page.data = jpeg_data; // Ownership transfers to accumulator
    page.data_size = jpeg_len;
    page.width = out_width;
    page.height = out_height;
    page.dpi = dpi;

    if (!pdf_page_accumulator_submit(ctx->accumulator, &page)) {
      free(jpeg_data);
      pdf_page_accumulator_mark_failed(ctx->accumulator, (int)job_index);
      return false;
    }

    return true;
  }

  // Fallback to raw pixels
  PdfPixelFormat fmt = (state->sheet.frame->format == AV_PIX_FMT_GRAY8)
                           ? PDF_PIXEL_GRAY8
                           : PDF_PIXEL_RGB24;

  size_t pixel_size = (size_t)stride * (size_t)out_height;
  uint8_t *pixels = malloc(pixel_size);
  if (!pixels) {
    pdf_page_accumulator_mark_failed(ctx->accumulator, (int)job_index);
    return false;
  }
  memcpy(pixels, state->sheet.frame->data[0], pixel_size);

  PdfEncodedPage page = {0};
  page.page_index = (int)job_index;
  page.type = PDF_PAGE_DATA_PIXELS;
  page.data = pixels; // Ownership transfers to accumulator
  page.data_size = pixel_size;
  page.width = out_width;
  page.height = out_height;
  page.stride = stride;
  page.pixel_format = fmt;
  page.dpi = dpi;

  if (!pdf_page_accumulator_submit(ctx->accumulator, &page)) {
    free(pixels);
    pdf_page_accumulator_mark_failed(ctx->accumulator, (int)job_index);
    return false;
  }

  return true;
}

int pdf_pipeline_cpu_process_batch(const char *input_path,
                                   const char *output_path,
                                   const Options *options,
                                   const SheetProcessConfig *config,
                                   int parallelism, int decode_queue_depth,
                                   bool progress) {
  if (input_path == NULL || output_path == NULL || options == NULL ||
      config == NULL) {
    verboseLog(VERBOSE_NORMAL, "PDF pipeline (CPU batch): invalid arguments\n");
    return -1;
  }

  verboseLog(VERBOSE_NORMAL, "PDF pipeline (CPU batch): %s -> %s\n", input_path,
             output_path);

  image_backend_select(UNPAPER_DEVICE_CPU);

  PdfDocument *doc = pdf_open(input_path);
  if (doc == NULL) {
    verboseLog(VERBOSE_NORMAL, "PDF pipeline (CPU batch): open failed: %s\n",
               pdf_get_last_error());
    return -1;
  }

  int page_count = pdf_page_count(doc);
  if (page_count <= 0) {
    verboseLog(VERBOSE_NORMAL,
               "PDF pipeline (CPU batch): no pages in document\n");
    pdf_close(doc);
    return -1;
  }

  int dpi = options->pdf_render_dpi > 0 ? options->pdf_render_dpi : PDF_RENDER_DPI;

  PdfMetadata meta = pdf_get_metadata(doc);
  PdfWriter *writer = pdf_writer_create(output_path, &meta, dpi);
  pdf_free_metadata(&meta);

  if (writer == NULL) {
    verboseLog(VERBOSE_NORMAL,
               "PDF pipeline (CPU batch): failed to create output: %s\n",
               pdf_writer_get_last_error());
    pdf_close(doc);
    return -1;
  }

  PdfPageAccumulator *accumulator =
      pdf_page_accumulator_create(writer, page_count);
  if (accumulator == NULL) {
    verboseLog(VERBOSE_NORMAL,
               "PDF pipeline (CPU batch): failed to create accumulator\n");
    pdf_writer_abort(writer);
    pdf_close(doc);
    return -1;
  }

  // Compute worker parallelism
  int workers = parallelism;
  if (workers <= 0) {
    workers = batch_detect_parallelism();
  }
  if (workers > page_count) {
    workers = page_count;
  }
  if (workers < 1) {
    workers = 1;
  }

  // Compute decode queue depth
  int depth = decode_queue_depth;
  if (depth <= 0) {
    depth = workers * 2;
  }
  if (depth > MAX_DECODE_QUEUE_DEPTH) {
    depth = MAX_DECODE_QUEUE_DEPTH;
  }
  if (depth > page_count) {
    depth = page_count;
  }
  if (depth < 1) {
    depth = 1;
  }

  // Build a synthetic batch queue: one job per page.
  BatchQueue batch_queue;
  batch_queue_init(&batch_queue);
  batch_queue.parallelism = workers;
  batch_queue.progress = progress;

  bool queue_ok = true;
  for (int i = 0; i < page_count; i++) {
    BatchJob *job = batch_queue_add(&batch_queue);
    if (!job) {
      queue_ok = false;
      break;
    }
    job->sheet_nr = i + 1;
    job->input_count = 1;
    job->output_count = 0;
    // Provide a stable "input" string so the generic worker requests decode.
    // (The actual decode comes from the custom decode_queue decoder.)
    char buf[128];
    snprintf(buf, sizeof(buf), "PDF page %d", i + 1);
    job->input_files[0] = strdup(buf);
    if (job->input_files[0] == NULL) {
      queue_ok = false;
      break;
    }
  }

  if (!queue_ok || (int)batch_queue.count != page_count) {
    batch_queue_free(&batch_queue);
    pdf_page_accumulator_destroy(accumulator);
    pdf_writer_abort(writer);
    pdf_close(doc);
    return -1;
  }

  if (batch_queue.progress) {
    batch_progress_start(&batch_queue);
  }

  // Create modified options/config so process_sheet does not write files.
  Options pdf_options = *options;
  pdf_options.write_output = false;
  pdf_options.device = UNPAPER_DEVICE_CPU;

  SheetProcessConfig pdf_config = *config;
  pdf_config.options = &pdf_options;

  // Decode queue with a PDF-page custom decoder.
  DecodeQueue *decode_queue = decode_queue_create((size_t)depth, false);
  if (decode_queue == NULL) {
    batch_queue_free(&batch_queue);
    pdf_page_accumulator_destroy(accumulator);
    pdf_writer_abort(writer);
    pdf_close(doc);
    return -1;
  }

  PdfDecodeQueueContext decode_ctx = {.doc = doc, .render_dpi = dpi};
  decode_queue_set_custom_decoder(decode_queue, pdf_decode_queue_decoder,
                                  &decode_ctx);

  if (!decode_queue_start_producer(decode_queue, &batch_queue, &pdf_options)) {
    decode_queue_destroy(decode_queue);
    batch_queue_free(&batch_queue);
    pdf_page_accumulator_destroy(accumulator);
    pdf_writer_abort(writer);
    pdf_close(doc);
    return -1;
  }

  // Worker context (generic batch worker).
  BatchWorkerContext worker_ctx;
  batch_worker_init(&worker_ctx, &pdf_options, &batch_queue);
  batch_worker_set_config(&worker_ctx, &pdf_config);
  batch_worker_set_decode_queue(&worker_ctx, decode_queue);

  PdfBatchWriteContext write_ctx = {.accumulator = accumulator,
                                   .options = &pdf_options};
  batch_worker_set_post_process_callback(&worker_ctx, pdf_batch_post_process,
                                         &write_ctx);

  ThreadPool *pool = threadpool_create(workers);
  if (pool == NULL) {
    decode_queue_stop_producer(decode_queue);
    decode_queue_destroy(decode_queue);
    batch_worker_cleanup(&worker_ctx);
    batch_queue_free(&batch_queue);
    pdf_page_accumulator_destroy(accumulator);
    pdf_writer_abort(writer);
    pdf_close(doc);
    return -1;
  }

  int failed_jobs = batch_process_parallel(&worker_ctx, pool);

  threadpool_destroy(pool);

  decode_queue_stop_producer(decode_queue);
  decode_queue_destroy(decode_queue);

  // Mark any failed pages so the accumulator doesn't wait forever.
  for (size_t i = 0; i < batch_queue.count; i++) {
    BatchJob *job = batch_queue_get(&batch_queue, i);
    if (job && job->status != BATCH_JOB_COMPLETED) {
      pdf_page_accumulator_mark_failed(accumulator, (int)i);
    }
  }

  pdf_page_accumulator_wait(accumulator);

  if (batch_queue.progress) {
    batch_progress_finish(&batch_queue);
  }

  batch_worker_cleanup(&worker_ctx);
  batch_queue_free(&batch_queue);

  int pages_failed = pdf_page_accumulator_pages_failed(accumulator);
  pdf_page_accumulator_destroy(accumulator);

  if (!pdf_writer_close(writer)) {
    verboseLog(VERBOSE_NORMAL,
               "PDF pipeline (CPU batch): failed to save output: %s\n",
               pdf_writer_get_last_error());
    pages_failed = page_count;
  }

  pdf_close(doc);

  // failed_jobs should match pages_failed, but pages_failed is the authoritative
  // "couldn't be written" count.
  (void)failed_jobs;

  return pages_failed;
}
