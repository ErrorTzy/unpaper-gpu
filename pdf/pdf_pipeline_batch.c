// SPDX-FileCopyrightText: 2025 The unpaper authors
//
// SPDX-License-Identifier: GPL-2.0-only

#include "pdf_pipeline_batch.h"
#include "pdf_page_accumulator.h"
#include "pdf_perf.h"
#include "pdf_reader.h"
#include "pdf_writer.h"

#include "imageprocess/backend.h"
#include "imageprocess/image.h"
#include "lib/batch.h"
#include "lib/logging.h"
#include "lib/perf.h"
#include "lib/threadpool.h"
#include "sheet_process.h"

#ifdef UNPAPER_WITH_JBIG2
#include "lib/jbig2_decode.h"
#endif

#ifdef UNPAPER_WITH_CUDA
#include "imageprocess/cuda_runtime.h"
#include "imageprocess/cuda_stream_pool.h"
#include "imageprocess/nvimgcodec.h"
#include <cuda_runtime.h>
#endif

#include <libavcodec/avcodec.h>
#include <libavformat/avformat.h>
#include <libavutil/frame.h>
#include <libavutil/imgutils.h>
#include <libavutil/pixfmt.h>
#include <libswscale/swscale.h>
#include <pthread.h>
#include <stdatomic.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

// ============================================================================
// Constants
// ============================================================================

#define PDF_RENDER_DPI 300
#define PDF_OUTPUT_JPEG_QUALITY 85
#define DEFAULT_DECODE_QUEUE_DEPTH 8
#define MAX_DECODE_QUEUE_DEPTH 32

// Performance pool sizes (PR 8)
#define PDF_PINNED_BUFFER_COUNT 16
#define PDF_PINNED_BUFFER_SIZE (16 * 1024 * 1024) // 16 MB per buffer
#define PDF_ENCODE_BUFFER_COUNT 16
#define PDF_ENCODE_BUFFER_SIZE (4 * 1024 * 1024) // 4 MB per buffer

// ============================================================================
// Decoded Page Slot
// ============================================================================

typedef enum {
  DECODE_SLOT_EMPTY = 0,
  DECODE_SLOT_DECODING,
  DECODE_SLOT_READY,
  DECODE_SLOT_IN_USE,
  DECODE_SLOT_FAILED,
} DecodeSlotState;

typedef struct {
  atomic_int state;
  int page_index;
  AVFrame *frame;         // CPU decoded frame
  bool on_gpu;            // True if decoded to GPU
  void *gpu_ptr;          // GPU memory pointer
  size_t gpu_pitch;       // GPU memory pitch
  int width, height;      // Image dimensions
  int pixel_format;       // AV_PIX_FMT_*
  PdfPinnedBuffer pinned; // Pinned memory buffer (PR 8)
} PdfDecodedPage;

// ============================================================================
// Batch Pipeline Context
// ============================================================================

typedef struct {
  // Input
  PdfDocument *doc;
  int page_count;

  // Output
  PdfWriter *writer;
  PdfPageAccumulator *accumulator;

  // Configuration
  const Options *options;
  const SheetProcessConfig *sheet_config;
  const PdfBatchConfig *batch_config;

  // Decode queue
  PdfDecodedPage *decode_slots;
  int decode_queue_depth;
  pthread_mutex_t decode_mutex;
  pthread_cond_t decode_not_full;
  pthread_cond_t decode_not_empty;

  // Decode producer state
  pthread_t decode_thread;
  atomic_bool decode_running;
  atomic_int next_decode_page;
  atomic_int decode_done_count;

  // Worker state
  atomic_int next_process_page;
  atomic_int pages_processed;
  atomic_int pages_failed;

  // Progress
  pthread_mutex_t progress_mutex;

  // Options copy with write_output = false
  Options pdf_options;
  SheetProcessConfig pdf_config;
} PdfBatchContext;

// ============================================================================
// Decode Functions (from pdf_pipeline_cpu.c patterns)
// ============================================================================

#ifdef UNPAPER_WITH_JBIG2
static AVFrame *decode_jbig2_to_frame(const PdfImage *pdf_img) {
  if (pdf_img == NULL || pdf_img->data == NULL || pdf_img->size == 0) {
    return NULL;
  }

  Jbig2DecodedImage jbig2_img = {0};
  if (!jbig2_decode(pdf_img->data, pdf_img->size, pdf_img->jbig2_globals,
                    pdf_img->jbig2_globals_size, &jbig2_img)) {
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
  if (pkt)
    av_packet_free(&pkt);
  if (codec_ctx)
    avcodec_free_context(&codec_ctx);
  if (fmt_ctx)
    avformat_close_input(&fmt_ctx);
  if (avio_ctx)
    avio_context_free(&avio_ctx);

  return frame;
}

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

  for (int y = 0; y < height; y++) {
    memcpy(frame->data[0] + y * frame->linesize[0], pixels + y * stride,
           width * 3);
  }

  free(pixels);
  return frame;
}

// ============================================================================
// JPEG Encoding (from pdf_pipeline_cpu.c)
// ============================================================================

// PR 8: Encode image to JPEG, optionally using encode buffer pool
static uint8_t *encode_image_jpeg_pooled(Image *image, int quality,
                                         size_t *out_len,
                                         PdfEncodeBuffer *out_buffer) {
  if (image == NULL || image->frame == NULL || out_len == NULL) {
    return NULL;
  }

  *out_len = 0;
  if (out_buffer) {
    out_buffer->data = NULL;
  }

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

  // PR 8: Use encode buffer pool if available
  uint8_t *jpeg_data = NULL;
  PdfEncodePool *pool = pdf_perf_get_encode_pool();
  if (pool && out_buffer) {
    PdfEncodeBuffer buf = pdf_encode_pool_acquire(pool, (size_t)pkt->size);
    if (buf.data) {
      memcpy(buf.data, pkt->data, pkt->size);
      buf.size = (size_t)pkt->size;
      *out_buffer = buf;
      jpeg_data = buf.data;
      *out_len = (size_t)pkt->size;
    }
  }

  // Fallback to malloc if pool not available or acquisition failed
  if (!jpeg_data) {
    jpeg_data = malloc((size_t)pkt->size);
    if (jpeg_data) {
      memcpy(jpeg_data, pkt->data, (size_t)pkt->size);
      *out_len = (size_t)pkt->size;
    }
  }

  av_packet_free(&pkt);
  if (yuv_frame)
    av_frame_free(&yuv_frame);
  avcodec_free_context(&ctx);

  return jpeg_data;
}

// Legacy encode function (for compatibility)
static uint8_t *encode_image_jpeg(Image *image, int quality, size_t *out_len) {
  return encode_image_jpeg_pooled(image, quality, out_len, NULL);
}

// ============================================================================
// Decode Producer Thread
// ============================================================================

static int find_empty_decode_slot(PdfBatchContext *ctx) {
  for (int i = 0; i < ctx->decode_queue_depth; i++) {
    int expected = DECODE_SLOT_EMPTY;
    if (atomic_compare_exchange_strong(&ctx->decode_slots[i].state, &expected,
                                       DECODE_SLOT_DECODING)) {
      return i;
    }
  }
  return -1;
}

static void *decode_producer_thread(void *arg) {
  PdfBatchContext *ctx = (PdfBatchContext *)arg;
  int render_dpi = ctx->options->pdf_render_dpi > 0
                       ? ctx->options->pdf_render_dpi
                       : PDF_RENDER_DPI;

  while (atomic_load(&ctx->decode_running)) {
    int page_idx = atomic_fetch_add(&ctx->next_decode_page, 1);
    if (page_idx >= ctx->page_count) {
      break;
    }

    // Wait for an empty slot
    int slot_idx = -1;
    while (atomic_load(&ctx->decode_running)) {
      slot_idx = find_empty_decode_slot(ctx);
      if (slot_idx >= 0) {
        break;
      }

      pthread_mutex_lock(&ctx->decode_mutex);
      pthread_cond_wait(&ctx->decode_not_full, &ctx->decode_mutex);
      pthread_mutex_unlock(&ctx->decode_mutex);
    }

    if (slot_idx < 0 || !atomic_load(&ctx->decode_running)) {
      break;
    }

    // Decode the page
    PdfDecodedPage *slot = &ctx->decode_slots[slot_idx];
    slot->page_index = page_idx;
    slot->frame = NULL;
    slot->on_gpu = false;
    slot->gpu_ptr = NULL;
    slot->pinned = (PdfPinnedBuffer){0};

    AVFrame *page_frame = NULL;

    // Try to extract embedded image
    PdfImage pdf_img = {0};
    if (pdf_extract_page_image(ctx->doc, page_idx, &pdf_img)) {
      // PR 8: Use pinned memory pool for image data (zero-copy GPU transfer)
      PdfPinnedPool *pinned_pool = pdf_perf_get_pinned_pool();
      if (pinned_pool && pdf_img.size > 0) {
        PdfPinnedBuffer buf =
            pdf_pinned_pool_acquire(pinned_pool, pdf_img.size);
        if (buf.ptr) {
          memcpy(buf.ptr, pdf_img.data, pdf_img.size);
          buf.size = pdf_img.size;
          slot->pinned = buf;
        }
      }

#ifdef UNPAPER_WITH_JBIG2
      if (pdf_img.format == PDF_IMAGE_JBIG2) {
        page_frame = decode_jbig2_to_frame(&pdf_img);
      } else
#endif
          if (pdf_img.format == PDF_IMAGE_JPEG ||
              pdf_img.format == PDF_IMAGE_PNG ||
              pdf_img.format == PDF_IMAGE_FLATE) {
        page_frame =
            decode_image_bytes(pdf_img.data, pdf_img.size, pdf_img.format);
      }
      pdf_free_image(&pdf_img);
    }

    // Fallback: render page
    if (!page_frame) {
      page_frame = render_page_to_frame(ctx->doc, page_idx, render_dpi);
    }

    if (page_frame) {
      slot->frame = page_frame;
      slot->width = page_frame->width;
      slot->height = page_frame->height;
      slot->pixel_format = page_frame->format;
      atomic_store(&slot->state, DECODE_SLOT_READY);
    } else {
      atomic_store(&slot->state, DECODE_SLOT_FAILED);
    }

    atomic_fetch_add(&ctx->decode_done_count, 1);

    // Signal consumers
    pthread_mutex_lock(&ctx->decode_mutex);
    pthread_cond_broadcast(&ctx->decode_not_empty);
    pthread_mutex_unlock(&ctx->decode_mutex);
  }

  // Signal that decode producer is done
  pthread_mutex_lock(&ctx->decode_mutex);
  pthread_cond_broadcast(&ctx->decode_not_empty);
  pthread_mutex_unlock(&ctx->decode_mutex);

  return NULL;
}

// ============================================================================
// Find Decoded Page by Index
// ============================================================================

static PdfDecodedPage *find_decoded_page(PdfBatchContext *ctx, int page_index) {
  for (int i = 0; i < ctx->decode_queue_depth; i++) {
    if (atomic_load(&ctx->decode_slots[i].state) == DECODE_SLOT_READY ||
        atomic_load(&ctx->decode_slots[i].state) == DECODE_SLOT_FAILED) {
      if (ctx->decode_slots[i].page_index == page_index) {
        int expected = DECODE_SLOT_READY;
        if (atomic_compare_exchange_strong(&ctx->decode_slots[i].state,
                                           &expected, DECODE_SLOT_IN_USE)) {
          return &ctx->decode_slots[i];
        }
        expected = DECODE_SLOT_FAILED;
        if (atomic_compare_exchange_strong(&ctx->decode_slots[i].state,
                                           &expected, DECODE_SLOT_IN_USE)) {
          return &ctx->decode_slots[i];
        }
      }
    }
  }
  return NULL;
}

static void release_decode_slot(PdfBatchContext *ctx, PdfDecodedPage *slot) {
  if (slot->frame) {
    av_frame_free(&slot->frame);
    slot->frame = NULL;
  }
#ifdef UNPAPER_WITH_CUDA
  if (slot->on_gpu && slot->gpu_ptr) {
    cudaFree(slot->gpu_ptr);
    slot->gpu_ptr = NULL;
    slot->on_gpu = false;
  }
#endif
  // PR 8: Release pinned memory buffer back to pool
  if (slot->pinned.ptr) {
    PdfPinnedPool *pool = pdf_perf_get_pinned_pool();
    pdf_pinned_pool_release(pool, &slot->pinned);
  }

  atomic_store(&slot->state, DECODE_SLOT_EMPTY);

  pthread_mutex_lock(&ctx->decode_mutex);
  pthread_cond_signal(&ctx->decode_not_full);
  pthread_mutex_unlock(&ctx->decode_mutex);
}

// ============================================================================
// Worker Thread Function
// ============================================================================

typedef struct {
  PdfBatchContext *ctx;
  int page_index;
} WorkerJobContext;

static void worker_process_page(void *arg, int thread_id) {
  WorkerJobContext *job = (WorkerJobContext *)arg;
  PdfBatchContext *ctx = job->ctx;
  int page_idx = job->page_index;

  (void)thread_id;

  // Wait for decoded page
  PdfDecodedPage *decoded = NULL;
  while (1) {
    decoded = find_decoded_page(ctx, page_idx);
    if (decoded) {
      break;
    }

    // Check if decode is done and page not found
    if (atomic_load(&ctx->decode_done_count) >= ctx->page_count) {
      break;
    }

    pthread_mutex_lock(&ctx->decode_mutex);
    pthread_cond_wait(&ctx->decode_not_empty, &ctx->decode_mutex);
    pthread_mutex_unlock(&ctx->decode_mutex);
  }

  if (!decoded || !decoded->frame) {
    // Mark as failed
    pdf_page_accumulator_mark_failed(ctx->accumulator, page_idx);
    atomic_fetch_add(&ctx->pages_failed, 1);

    pthread_mutex_lock(&ctx->progress_mutex);
    if (ctx->batch_config->progress) {
      fprintf(stderr, "\rProcessing page %d/%d (failed)...", page_idx + 1,
              ctx->page_count);
    }
    pthread_mutex_unlock(&ctx->progress_mutex);

    if (decoded) {
      release_decode_slot(ctx, decoded);
    }
    free(job);
    return;
  }

  // Progress output
  pthread_mutex_lock(&ctx->progress_mutex);
  if (ctx->batch_config->progress) {
    fprintf(stderr, "\rProcessing page %d/%d...", page_idx + 1,
            ctx->page_count);
    fflush(stderr);
  }
  pthread_mutex_unlock(&ctx->progress_mutex);

  // Set up sheet processing
  BatchJob batch_job = {0};
  batch_job.sheet_nr = page_idx + 1;
  batch_job.input_count = 1;
  batch_job.output_count = 0;
  batch_job.input_files[0] = NULL;

  SheetProcessState state;
  sheet_process_state_init(&state, &ctx->pdf_config, &batch_job);

  // Set the decoded frame
  AVFrame *frame_copy = av_frame_clone(decoded->frame);
  if (frame_copy) {
    sheet_process_state_set_decoded(&state, frame_copy, 0);
  }

  // Release decode slot early
  release_decode_slot(ctx, decoded);

  // Process the sheet
  bool process_ok = process_sheet(&state, &ctx->pdf_config);

  if (!process_ok || state.sheet.frame == NULL) {
    pdf_page_accumulator_mark_failed(ctx->accumulator, page_idx);
    atomic_fetch_add(&ctx->pages_failed, 1);
    sheet_process_state_cleanup(&state);
    free(job);
    return;
  }

  // Encode the result
  image_ensure_cpu(&state.sheet);

  int out_width = state.sheet.frame->width;
  int out_height = state.sheet.frame->height;
  int quality = (ctx->options->jpeg_quality > 0) ? ctx->options->jpeg_quality
                                                 : PDF_OUTPUT_JPEG_QUALITY;
  int dpi = ctx->options->pdf_render_dpi > 0 ? ctx->options->pdf_render_dpi
                                             : PDF_RENDER_DPI;

  // PR 8: Use encode buffer pool for JPEG encoding
  size_t jpeg_len = 0;
  PdfEncodeBuffer enc_buf = {0};
  uint8_t *jpeg_data =
      encode_image_jpeg_pooled(&state.sheet, quality, &jpeg_len, &enc_buf);

  if (jpeg_data && jpeg_len > 0) {
    PdfEncodedPage page = {0};
    page.page_index = page_idx;
    page.type = PDF_PAGE_DATA_JPEG;
    page.width = out_width;
    page.height = out_height;
    page.dpi = dpi;
    page.data_size = jpeg_len;

    // PR 8: Detach buffer from pool for accumulator ownership
    if (enc_buf.from_pool) {
      PdfEncodePool *pool = pdf_perf_get_encode_pool();
      page.data = pdf_encode_pool_detach(pool, &enc_buf);
    } else {
      page.data = jpeg_data;
    }

    if (!pdf_page_accumulator_submit(ctx->accumulator, &page)) {
      free(page.data);
      pdf_page_accumulator_mark_failed(ctx->accumulator, page_idx);
      atomic_fetch_add(&ctx->pages_failed, 1);
    } else {
      atomic_fetch_add(&ctx->pages_processed, 1);
    }
  } else {
    // Fallback to pixels
    int stride = state.sheet.frame->linesize[0];
    PdfPixelFormat fmt = (state.sheet.frame->format == AV_PIX_FMT_GRAY8)
                             ? PDF_PIXEL_GRAY8
                             : PDF_PIXEL_RGB24;

    // Copy pixel data (accumulator takes ownership)
    size_t pixel_size = (size_t)stride * (size_t)out_height;
    uint8_t *pixels = malloc(pixel_size);
    if (pixels) {
      memcpy(pixels, state.sheet.frame->data[0], pixel_size);

      PdfEncodedPage page = {0};
      page.page_index = page_idx;
      page.type = PDF_PAGE_DATA_PIXELS;
      page.data = pixels;
      page.data_size = pixel_size;
      page.width = out_width;
      page.height = out_height;
      page.stride = stride;
      page.pixel_format = fmt;
      page.dpi = dpi;

      if (!pdf_page_accumulator_submit(ctx->accumulator, &page)) {
        free(pixels);
        pdf_page_accumulator_mark_failed(ctx->accumulator, page_idx);
        atomic_fetch_add(&ctx->pages_failed, 1);
      } else {
        atomic_fetch_add(&ctx->pages_processed, 1);
      }
    } else {
      pdf_page_accumulator_mark_failed(ctx->accumulator, page_idx);
      atomic_fetch_add(&ctx->pages_failed, 1);
    }
  }

  sheet_process_state_cleanup(&state);
  free(job);
}

// ============================================================================
// Public API Implementation
// ============================================================================

void pdf_batch_config_init(PdfBatchConfig *config) {
  if (!config) {
    return;
  }
  config->parallelism = 0;        // Auto-detect
  config->decode_queue_depth = 0; // Auto
  config->progress = true;
  config->use_gpu = false;
}

bool pdf_pipeline_batch_available(void) {
  return true; // Always available with PDF support
}

int pdf_pipeline_batch_process(const char *input_path, const char *output_path,
                               const Options *options,
                               const SheetProcessConfig *sheet_config,
                               const PdfBatchConfig *batch_config) {
  if (!input_path || !output_path || !options || !sheet_config) {
    verboseLog(VERBOSE_NORMAL, "Batch PDF pipeline: invalid arguments\n");
    return -1;
  }

  // Use default config if not provided
  PdfBatchConfig default_config;
  if (!batch_config) {
    pdf_batch_config_init(&default_config);
    batch_config = &default_config;
  }

  verboseLog(VERBOSE_NORMAL, "Batch PDF pipeline: %s -> %s\n", input_path,
             output_path);

  // Select backend
  if (batch_config->use_gpu) {
#ifdef UNPAPER_WITH_CUDA
    image_backend_select(UNPAPER_DEVICE_CUDA);
#else
    verboseLog(VERBOSE_NORMAL,
               "Batch PDF pipeline: GPU requested but CUDA not available\n");
    image_backend_select(UNPAPER_DEVICE_CPU);
#endif
  } else {
    image_backend_select(UNPAPER_DEVICE_CPU);
  }

  // Open input PDF
  PdfDocument *doc = pdf_open(input_path);
  if (!doc) {
    verboseLog(VERBOSE_NORMAL, "Batch PDF pipeline: failed to open: %s\n",
               pdf_get_last_error());
    return -1;
  }

  int page_count = pdf_page_count(doc);
  if (page_count <= 0) {
    verboseLog(VERBOSE_NORMAL, "Batch PDF pipeline: no pages in document\n");
    pdf_close(doc);
    return -1;
  }

  verboseLog(VERBOSE_NORMAL, "Batch PDF pipeline: %d pages to process\n",
             page_count);

  // Get metadata
  PdfMetadata meta = pdf_get_metadata(doc);
  int dpi =
      options->pdf_render_dpi > 0 ? options->pdf_render_dpi : PDF_RENDER_DPI;

  // Create output PDF
  PdfWriter *writer = pdf_writer_create(output_path, &meta, dpi);
  if (!writer) {
    verboseLog(VERBOSE_NORMAL, "Batch PDF pipeline: failed to create output\n");
    pdf_free_metadata(&meta);
    pdf_close(doc);
    return -1;
  }
  pdf_free_metadata(&meta);

  // Create page accumulator
  PdfPageAccumulator *accumulator =
      pdf_page_accumulator_create(writer, page_count);
  if (!accumulator) {
    verboseLog(VERBOSE_NORMAL,
               "Batch PDF pipeline: failed to create accumulator\n");
    pdf_writer_abort(writer);
    pdf_close(doc);
    return -1;
  }

  // Determine parallelism
  int parallelism = batch_config->parallelism;
  if (parallelism <= 0) {
    parallelism = batch_detect_parallelism();
    if (parallelism > page_count) {
      parallelism = page_count;
    }
  }

  // Determine decode queue depth
  int decode_depth = batch_config->decode_queue_depth;
  if (decode_depth <= 0) {
    decode_depth = parallelism * 2;
  }
  if (decode_depth > MAX_DECODE_QUEUE_DEPTH) {
    decode_depth = MAX_DECODE_QUEUE_DEPTH;
  }
  if (decode_depth > page_count) {
    decode_depth = page_count;
  }

  verboseLog(VERBOSE_NORMAL,
             "Batch PDF pipeline: %d workers, %d decode slots\n", parallelism,
             decode_depth);

  // PR 8: Initialize performance pools
  // Calculate pool sizes based on parallelism
  int pinned_count = parallelism * 2;
  if (pinned_count > PDF_PINNED_BUFFER_COUNT) {
    pinned_count = PDF_PINNED_BUFFER_COUNT;
  }
  int encode_count = parallelism * 2;
  if (encode_count > PDF_ENCODE_BUFFER_COUNT) {
    encode_count = PDF_ENCODE_BUFFER_COUNT;
  }

  pdf_perf_init(pinned_count, PDF_PINNED_BUFFER_SIZE, encode_count,
                PDF_ENCODE_BUFFER_SIZE);

  verboseLog(
      VERBOSE_DEBUG,
      "Batch PDF pipeline: perf pools initialized (pinned=%d, encode=%d)\n",
      pinned_count, encode_count);

  // Initialize context
  PdfBatchContext ctx = {0};
  ctx.doc = doc;
  ctx.page_count = page_count;
  ctx.writer = writer;
  ctx.accumulator = accumulator;
  ctx.options = options;
  ctx.sheet_config = sheet_config;
  ctx.batch_config = batch_config;

  // Create modified options
  ctx.pdf_options = *options;
  ctx.pdf_options.write_output = false;

  ctx.pdf_config = *sheet_config;
  ctx.pdf_config.options = &ctx.pdf_options;

  // Allocate decode slots
  ctx.decode_queue_depth = decode_depth;
  ctx.decode_slots = calloc((size_t)decode_depth, sizeof(PdfDecodedPage));
  if (!ctx.decode_slots) {
    pdf_page_accumulator_destroy(accumulator);
    pdf_writer_abort(writer);
    pdf_close(doc);
    return -1;
  }

  for (int i = 0; i < decode_depth; i++) {
    atomic_init(&ctx.decode_slots[i].state, DECODE_SLOT_EMPTY);
  }

  pthread_mutex_init(&ctx.decode_mutex, NULL);
  pthread_cond_init(&ctx.decode_not_full, NULL);
  pthread_cond_init(&ctx.decode_not_empty, NULL);
  pthread_mutex_init(&ctx.progress_mutex, NULL);

  atomic_init(&ctx.decode_running, true);
  atomic_init(&ctx.next_decode_page, 0);
  atomic_init(&ctx.decode_done_count, 0);
  atomic_init(&ctx.next_process_page, 0);
  atomic_init(&ctx.pages_processed, 0);
  atomic_init(&ctx.pages_failed, 0);

  // Start decode producer thread
  if (pthread_create(&ctx.decode_thread, NULL, decode_producer_thread, &ctx) !=
      0) {
    free(ctx.decode_slots);
    pthread_mutex_destroy(&ctx.decode_mutex);
    pthread_cond_destroy(&ctx.decode_not_full);
    pthread_cond_destroy(&ctx.decode_not_empty);
    pthread_mutex_destroy(&ctx.progress_mutex);
    pdf_page_accumulator_destroy(accumulator);
    pdf_writer_abort(writer);
    pdf_close(doc);
    return -1;
  }

  // Create thread pool
  ThreadPool *pool = threadpool_create(parallelism);
  if (!pool) {
    atomic_store(&ctx.decode_running, false);
    pthread_join(ctx.decode_thread, NULL);
    free(ctx.decode_slots);
    pthread_mutex_destroy(&ctx.decode_mutex);
    pthread_cond_destroy(&ctx.decode_not_full);
    pthread_cond_destroy(&ctx.decode_not_empty);
    pthread_mutex_destroy(&ctx.progress_mutex);
    pdf_page_accumulator_destroy(accumulator);
    pdf_writer_abort(writer);
    pdf_close(doc);
    return -1;
  }

  // Submit all pages as jobs
  for (int i = 0; i < page_count; i++) {
    WorkerJobContext *job = malloc(sizeof(WorkerJobContext));
    if (job) {
      job->ctx = &ctx;
      job->page_index = i;
      if (!threadpool_submit(pool, worker_process_page, job)) {
        free(job);
        pdf_page_accumulator_mark_failed(accumulator, i);
      }
    } else {
      pdf_page_accumulator_mark_failed(accumulator, i);
    }
  }

  // Wait for all workers
  threadpool_wait(pool);
  threadpool_destroy(pool);

  // Stop decode thread
  atomic_store(&ctx.decode_running, false);
  pthread_mutex_lock(&ctx.decode_mutex);
  pthread_cond_broadcast(&ctx.decode_not_full);
  pthread_mutex_unlock(&ctx.decode_mutex);
  pthread_join(ctx.decode_thread, NULL);

  // Wait for all pages to be written
  pdf_page_accumulator_wait(accumulator);

  // Clear progress line
  if (batch_config->progress) {
    fprintf(stderr, "\r                                        \r");
  }

  // Get results
  int pages_failed = pdf_page_accumulator_pages_failed(accumulator);

  // Cleanup
  pdf_page_accumulator_destroy(accumulator);

  // Close output PDF
  if (!pdf_writer_close(writer)) {
    verboseLog(VERBOSE_NORMAL, "Batch PDF pipeline: failed to save output\n");
    pages_failed = page_count;
  }

  // Cleanup decode slots
  for (int i = 0; i < decode_depth; i++) {
    if (ctx.decode_slots[i].frame) {
      av_frame_free(&ctx.decode_slots[i].frame);
    }
    // PR 8: Release any remaining pinned buffers
    if (ctx.decode_slots[i].pinned.ptr) {
      PdfPinnedPool *pool = pdf_perf_get_pinned_pool();
      pdf_pinned_pool_release(pool, &ctx.decode_slots[i].pinned);
    }
  }
  free(ctx.decode_slots);

  pthread_mutex_destroy(&ctx.decode_mutex);
  pthread_cond_destroy(&ctx.decode_not_full);
  pthread_cond_destroy(&ctx.decode_not_empty);
  pthread_mutex_destroy(&ctx.progress_mutex);

  pdf_close(doc);

  // PR 8: Print perf stats in debug mode and cleanup
  if (verbose >= VERBOSE_DEBUG) {
    pdf_perf_print_stats();
  }
  pdf_perf_cleanup();

  verboseLog(VERBOSE_NORMAL,
             "Batch PDF pipeline: complete. %d pages, %d failed\n", page_count,
             pages_failed);

  return pages_failed;
}
