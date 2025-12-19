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

#ifdef UNPAPER_WITH_CUDA
#include "imageprocess/cuda_runtime.h"
#include "imageprocess/cuda_stream_pool.h"
#include "imageprocess/nvimgcodec.h"
#include "lib/gpu_monitor.h"
#include <cuda_runtime.h>
#endif

#ifdef UNPAPER_WITH_JBIG2
#include "lib/jbig2_decode.h"
#endif

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
  int input_pages_per_sheet;
  int total_pages;
  bool use_gpu;
  bool nvimgcodec_ok;
} PdfDecodeQueueContext;

#ifdef UNPAPER_WITH_CUDA
static bool pdf_upload_frame_to_gpu(AVFrame *frame, DecodedImage *out) {
  if (!frame || !out) {
    return false;
  }

  int channels = 0;
  if (frame->format == AV_PIX_FMT_GRAY8) {
    channels = 1;
  } else if (frame->format == AV_PIX_FMT_RGB24) {
    channels = 3;
  } else {
    return false;
  }

  if (frame->linesize[0] <= 0 || frame->width <= 0 || frame->height <= 0) {
    return false;
  }
  if (frame->linesize[0] < frame->width * channels) {
    return false;
  }

  size_t pitch = (size_t)frame->linesize[0];
  size_t bytes = pitch * (size_t)frame->height;

  void *gpu_ptr = NULL;
  cudaError_t err = cudaMalloc(&gpu_ptr, bytes);
  if (err != cudaSuccess || gpu_ptr == NULL) {
    return false;
  }

  err = cudaMemcpy(gpu_ptr, frame->data[0], bytes, cudaMemcpyHostToDevice);
  if (err != cudaSuccess) {
    cudaFree(gpu_ptr);
    return false;
  }

  out->on_gpu = true;
  out->gpu_ptr = gpu_ptr;
  out->gpu_pitch = pitch;
  out->gpu_width = frame->width;
  out->gpu_height = frame->height;
  out->gpu_channels = channels;
  out->gpu_format = frame->format;
  out->gpu_completion_event = NULL;
  out->gpu_event_from_pool = false;
  out->frame = NULL;
  out->valid = true;
  return true;
}

static bool pdf_decode_pdfimage_to_gpu(const PdfImage *pdf_img,
                                       DecodedImage *out) {
  if (!pdf_img || !pdf_img->data || pdf_img->size == 0 || !out) {
    return false;
  }

  NvImgCodecDecodeState *dec_state = nvimgcodec_acquire_decode_state();
  if (dec_state == NULL) {
    return false;
  }

  NvImgCodecDecodedImage gpu_img = {0};
  bool ok =
      nvimgcodec_decode(pdf_img->data, pdf_img->size, dec_state, NULL,
                        NVIMGCODEC_OUT_GRAY8, &gpu_img);
  nvimgcodec_release_decode_state(dec_state);

  if (!ok || gpu_img.gpu_ptr == NULL) {
    return false;
  }

  out->on_gpu = true;
  out->gpu_ptr = gpu_img.gpu_ptr;
  out->gpu_pitch = gpu_img.pitch;
  out->gpu_width = gpu_img.width;
  out->gpu_height = gpu_img.height;
  out->gpu_channels = gpu_img.channels;
  out->gpu_format =
      (gpu_img.channels == 1) ? AV_PIX_FMT_GRAY8 : AV_PIX_FMT_RGB24;
  out->gpu_completion_event = gpu_img.completion_event;
  out->gpu_event_from_pool = gpu_img.event_from_pool;
  out->frame = NULL;
  out->valid = true;
  return true;
}
#endif

#if defined(UNPAPER_WITH_JBIG2) && defined(UNPAPER_WITH_CUDA)
static bool pdf_decode_jbig2_to_gpu(const PdfImage *pdf_img, DecodedImage *out) {
  if (!pdf_img || !pdf_img->data || pdf_img->size == 0 || !out) {
    return false;
  }

  Jbig2DecodedImage jbig2_img = {0};
  if (!jbig2_decode(pdf_img->data, pdf_img->size, pdf_img->jbig2_globals,
                    pdf_img->jbig2_globals_size, &jbig2_img)) {
    return false;
  }

  const int width = (int)jbig2_img.width;
  const int height = (int)jbig2_img.height;
  size_t stride = (size_t)width;
  size_t bytes = stride * (size_t)height;
  uint8_t *gray = malloc(bytes);
  if (gray == NULL) {
    jbig2_free_image(&jbig2_img);
    return false;
  }

  if (!jbig2_expand_to_gray8(&jbig2_img, gray, stride, false)) {
    free(gray);
    jbig2_free_image(&jbig2_img);
    return false;
  }

  void *gpu_ptr = NULL;
  cudaError_t err = cudaMalloc(&gpu_ptr, bytes);
  if (err != cudaSuccess || gpu_ptr == NULL) {
    free(gray);
    jbig2_free_image(&jbig2_img);
    return false;
  }

  err = cudaMemcpy(gpu_ptr, gray, bytes, cudaMemcpyHostToDevice);
  free(gray);
  jbig2_free_image(&jbig2_img);
  if (err != cudaSuccess) {
    cudaFree(gpu_ptr);
    return false;
  }

  out->on_gpu = true;
  out->gpu_ptr = gpu_ptr;
  out->gpu_pitch = stride;
  out->gpu_width = width;
  out->gpu_height = height;
  out->gpu_channels = 1;
  out->gpu_format = AV_PIX_FMT_GRAY8;
  out->gpu_completion_event = NULL;
  out->gpu_event_from_pool = false;
  out->frame = NULL;
  out->valid = true;
  return true;
}
#endif

static bool pdf_decode_queue_decoder(void *user_ctx, const BatchJob *job,
                                     int job_index, int input_index,
                                     DecodedImage *out) {
  (void)job;

  PdfDecodeQueueContext *ctx = (PdfDecodeQueueContext *)user_ctx;
  if (!ctx || !ctx->doc || !out) {
    return false;
  }

  if (ctx->input_pages_per_sheet < 1) {
    return false;
  }

  int page_idx = job_index * ctx->input_pages_per_sheet + input_index;
  if (page_idx < 0 || page_idx >= ctx->total_pages) {
    return false;
  }

  if (ctx->use_gpu) {
#ifdef UNPAPER_WITH_CUDA
    PdfImage pdf_img = {0};
    bool extracted = pdf_extract_page_image(ctx->doc, page_idx, &pdf_img);
    if (extracted) {
      verboseLog(VERBOSE_MORE, "GPU PDF pipeline: extracted %s image %dx%d\n",
                 pdf_image_format_name(pdf_img.format), pdf_img.width,
                 pdf_img.height);
    }

    if (extracted && ctx->nvimgcodec_ok &&
        (pdf_img.format == PDF_IMAGE_JPEG ||
         pdf_img.format == PDF_IMAGE_JP2)) {
      if (pdf_decode_pdfimage_to_gpu(&pdf_img, out)) {
        verboseLog(VERBOSE_MORE,
                   "GPU PDF pipeline: %s decoded to GPU %dx%d\n",
                   pdf_image_format_name(pdf_img.format), out->gpu_width,
                   out->gpu_height);
        pdf_free_image(&pdf_img);
        return true;
      }
    }

#if defined(UNPAPER_WITH_JBIG2)
    if (extracted && pdf_img.format == PDF_IMAGE_JBIG2) {
      if (pdf_decode_jbig2_to_gpu(&pdf_img, out)) {
        verboseLog(VERBOSE_MORE,
                   "GPU PDF pipeline: JBIG2 decoded to GPU %dx%d\n",
                   out->gpu_width, out->gpu_height);
        pdf_free_image(&pdf_img);
        return true;
      }
    }
#endif

    if (extracted) {
      pdf_free_image(&pdf_img);
    }

    AVFrame *frame =
        pdf_pipeline_decode_page_to_frame(ctx->doc, page_idx, ctx->render_dpi);
    if (frame) {
      if (pdf_upload_frame_to_gpu(frame, out)) {
        verboseLog(VERBOSE_MORE,
                   "GPU PDF pipeline: rendered and uploaded to GPU %dx%d\n",
                   out->gpu_width, out->gpu_height);
        av_frame_free(&frame);
        return true;
      }

      out->frame = frame;
      out->valid = true;
      out->uses_pinned_memory = false;
      out->on_gpu = false;
      return true;
    }
    return false;
#else
    return false;
#endif
  }

  AVFrame *frame =
      pdf_pipeline_decode_page_to_frame(ctx->doc, page_idx, ctx->render_dpi);
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
  bool use_gpu_encode;
} PdfBatchWriteContext;

static bool pdf_submit_image_as_page(PdfPageAccumulator *accumulator,
                                     const Options *options, Image *image,
                                     int page_index, bool use_gpu_encode) {
  if (accumulator == NULL || options == NULL || image == NULL ||
      image->frame == NULL || page_index < 0) {
    return false;
  }

  int out_width = image->frame->width;
  int out_height = image->frame->height;
  int dpi =
      options->pdf_render_dpi > 0 ? options->pdf_render_dpi : PDF_RENDER_DPI;

  int quality = (options->jpeg_quality > 0) ? options->jpeg_quality
                                            : PDF_OUTPUT_JPEG_QUALITY;

#ifdef UNPAPER_WITH_CUDA
  if (use_gpu_encode && nvimgcodec_any_available()) {
    image_ensure_cuda(image);
    void *gpu_ptr = image_get_gpu_ptr(image);
    size_t gpu_pitch = image_get_gpu_pitch(image);
    if (gpu_ptr != NULL && gpu_pitch > 0) {
      NvImgCodecEncodeState *enc_state = nvimgcodec_acquire_encode_state();
      if (enc_state != NULL) {
        NvImgCodecEncodedImage encoded = {0};
        NvImgCodecEncodeInputFormat input_fmt =
            (image->frame->format == AV_PIX_FMT_GRAY8)
                ? NVIMGCODEC_ENC_FMT_GRAY8
                : NVIMGCODEC_ENC_FMT_RGB;
        bool ok = false;
        PdfPageDataType page_type = PDF_PAGE_DATA_JPEG;

        if (options->pdf_quality_mode == PDF_QUALITY_HIGH &&
            nvimgcodec_jp2_supported()) {
          ok = nvimgcodec_encode_jp2(gpu_ptr, gpu_pitch, out_width, out_height,
                                     input_fmt, true, enc_state,
                                     unpaper_cuda_get_current_stream(),
                                     &encoded);
          page_type = PDF_PAGE_DATA_JP2;
        } else {
          ok = nvimgcodec_encode_jpeg(gpu_ptr, gpu_pitch, out_width, out_height,
                                      input_fmt, quality, enc_state,
                                      unpaper_cuda_get_current_stream(),
                                      &encoded);
          page_type = PDF_PAGE_DATA_JPEG;
        }

        nvimgcodec_release_encode_state(enc_state);

        if (ok && encoded.data != NULL && encoded.size > 0) {
          PdfEncodedPage page = {0};
          page.page_index = page_index;
          page.type = page_type;
          page.data = encoded.data; // Ownership transfers to accumulator
          page.data_size = encoded.size;
          page.width = out_width;
          page.height = out_height;
          page.dpi = dpi;

          if (!pdf_page_accumulator_submit(accumulator, &page)) {
            free(encoded.data);
            return false;
          }

          return true;
        }
      }
    }
  }
#endif

  image_ensure_cpu(image);

  int stride = image->frame->linesize[0];
  size_t jpeg_len = 0;
  uint8_t *jpeg_data = encode_image_jpeg(image, quality, &jpeg_len);

  if (jpeg_data && jpeg_len > 0) {
    PdfEncodedPage page = {0};
    page.page_index = page_index;
    page.type = PDF_PAGE_DATA_JPEG;
    page.data = jpeg_data; // Ownership transfers to accumulator
    page.data_size = jpeg_len;
    page.width = out_width;
    page.height = out_height;
    page.dpi = dpi;

    if (!pdf_page_accumulator_submit(accumulator, &page)) {
      free(jpeg_data);
      return false;
    }

    return true;
  }

  // Fallback to raw pixels
  PdfPixelFormat fmt =
      (image->frame->format == AV_PIX_FMT_GRAY8) ? PDF_PIXEL_GRAY8
                                                 : PDF_PIXEL_RGB24;

  size_t pixel_size = (size_t)stride * (size_t)out_height;
  uint8_t *pixels = malloc(pixel_size);
  if (!pixels) {
    return false;
  }
  memcpy(pixels, image->frame->data[0], pixel_size);

  PdfEncodedPage page = {0};
  page.page_index = page_index;
  page.type = PDF_PAGE_DATA_PIXELS;
  page.data = pixels; // Ownership transfers to accumulator
  page.data_size = pixel_size;
  page.width = out_width;
  page.height = out_height;
  page.stride = stride;
  page.pixel_format = fmt;
  page.dpi = dpi;

  if (!pdf_page_accumulator_submit(accumulator, &page)) {
    free(pixels);
    return false;
  }

  return true;
}

static bool pdf_batch_post_process(BatchWorkerContext *worker_ctx,
                                   size_t job_index, SheetProcessState *state,
                                   void *user_ctx) {
  (void)worker_ctx;

  PdfBatchWriteContext *ctx = (PdfBatchWriteContext *)user_ctx;
  if (!ctx || !ctx->accumulator || !ctx->options || !state) {
    return false;
  }

  int output_pages = ctx->options->output_count;
  if (output_pages < 1) {
    output_pages = 1;
  }
  if (output_pages > BATCH_MAX_FILES_PER_SHEET) {
    output_pages = BATCH_MAX_FILES_PER_SHEET;
  }

  int base_out_page = (int)job_index * output_pages;

  if (state->sheet.frame == NULL) {
    for (int o = 0; o < output_pages; o++) {
      pdf_page_accumulator_mark_failed(ctx->accumulator, base_out_page + o);
    }
    return false;
  }

  bool ok = true;

  if (output_pages == 1) {
    if (!pdf_submit_image_as_page(ctx->accumulator, ctx->options, &state->sheet,
                                  base_out_page, ctx->use_gpu_encode)) {
      pdf_page_accumulator_mark_failed(ctx->accumulator, base_out_page);
      ok = false;
    }
    return ok;
  }

  int sheet_width = state->sheet.frame->width;
  int sheet_height = state->sheet.frame->height;
  int page_width = sheet_width / output_pages;

  if (ctx->use_gpu_encode) {
    image_ensure_cuda(&state->sheet);
  }

  for (int o = 0; o < output_pages; o++) {
    Image page_img = create_compatible_image(
        state->sheet, (RectangleSize){.width = page_width, .height = sheet_height},
        false);

    copy_rectangle(state->sheet, page_img,
                   (Rectangle){{{page_width * o, 0},
                                {page_width * o + page_width, sheet_height}}},
                   POINT_ORIGIN);

    if (!pdf_submit_image_as_page(ctx->accumulator, ctx->options, &page_img,
                                  base_out_page + o, ctx->use_gpu_encode)) {
      pdf_page_accumulator_mark_failed(ctx->accumulator, base_out_page + o);
      ok = false;
    }

    free_image(&page_img);
  }

  return ok;
}

int pdf_pipeline_cpu_process_batch(const char *input_path,
                                   const char *output_path,
                                   const Options *options,
                                   const SheetProcessConfig *config,
                                   int parallelism, int decode_queue_depth,
                                   bool progress) {
  if (input_path == NULL || output_path == NULL || options == NULL ||
      config == NULL) {
    verboseLog(VERBOSE_NORMAL, "PDF pipeline (batch): invalid arguments\n");
    return -1;
  }

  bool use_gpu = false;
  bool nvimgcodec_ok = false;
#ifdef UNPAPER_WITH_CUDA
  if (options->device == UNPAPER_DEVICE_CUDA) {
    UnpaperCudaInitStatus cuda_status = unpaper_cuda_try_init();
    if (cuda_status == UNPAPER_CUDA_INIT_OK) {
      use_gpu = true;
    } else {
      verboseLog(VERBOSE_NORMAL, "PDF pipeline: CUDA unavailable (%s)\n",
                 unpaper_cuda_init_status_string(cuda_status));
    }
  }
#endif

  verboseLog(VERBOSE_NORMAL, "PDF pipeline (batch %s): %s -> %s\n",
             use_gpu ? "GPU" : "CPU", input_path, output_path);

  image_backend_select(use_gpu ? UNPAPER_DEVICE_CUDA : UNPAPER_DEVICE_CPU);

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

  int input_pages = options->input_count;
  if (input_pages < 1) {
    input_pages = 1;
  }
  int output_pages = options->output_count;
  if (output_pages < 1) {
    output_pages = 1;
  }
  if (input_pages > BATCH_MAX_FILES_PER_SHEET ||
      output_pages > BATCH_MAX_FILES_PER_SHEET) {
    verboseLog(VERBOSE_NORMAL,
               "PDF pipeline (CPU batch): --input-pages/--output-pages > %d "
               "not supported\n",
               BATCH_MAX_FILES_PER_SHEET);
    pdf_close(doc);
    return -1;
  }

  int sheet_count = (page_count + input_pages - 1) / input_pages;
  int output_page_count = sheet_count * output_pages;

  int dpi =
      options->pdf_render_dpi > 0 ? options->pdf_render_dpi : PDF_RENDER_DPI;

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
      pdf_page_accumulator_create(writer, output_page_count);
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
  if (workers > sheet_count) {
    workers = sheet_count;
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
  for (int sheet_idx = 0; sheet_idx < sheet_count; sheet_idx++) {
    BatchJob *job = batch_queue_add(&batch_queue);
    if (!job) {
      queue_ok = false;
      break;
    }

    job->sheet_nr = sheet_idx + 1;
    job->output_count = output_pages;

    int first_page = sheet_idx * input_pages;
    int remaining = page_count - first_page;
    int job_inputs = remaining < input_pages ? remaining : input_pages;
    job->input_count = job_inputs;

    for (int i = 0; i < job_inputs; i++) {
      // Provide a stable "input" string so the generic worker requests decode.
      // (The actual decode comes from the custom decode_queue decoder.)
      char buf[128];
      snprintf(buf, sizeof(buf), "PDF page %d", first_page + i + 1);
      job->input_files[i] = strdup(buf);
      if (job->input_files[i] == NULL) {
        queue_ok = false;
        break;
      }
    }
    if (!queue_ok) {
      break;
    }
  }

  if (!queue_ok || (int)batch_queue.count != sheet_count) {
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
  pdf_options.device = use_gpu ? UNPAPER_DEVICE_CUDA : UNPAPER_DEVICE_CPU;

  SheetProcessConfig pdf_config = *config;
  pdf_config.options = &pdf_options;

  // Decode queue with a PDF-page custom decoder.
  bool use_pinned_memory = use_gpu;
  int num_decode_threads = 1;

  // PDF decoding uses MuPDF which is not thread-safe for concurrent decode on
  // a shared document. Keep a single producer thread.
  DecodeQueue *decode_queue = decode_queue_create((size_t)depth, use_pinned_memory);
  if (decode_queue == NULL) {
    batch_queue_free(&batch_queue);
    pdf_page_accumulator_destroy(accumulator);
    pdf_writer_abort(writer);
    pdf_close(doc);
    return -1;
  }

  PdfDecodeQueueContext decode_ctx = {.doc = doc,
                                      .render_dpi = dpi,
                                      .input_pages_per_sheet = input_pages,
                                      .total_pages = page_count,
                                      .use_gpu = use_gpu,
                                      .nvimgcodec_ok = false};

#ifdef UNPAPER_WITH_CUDA
  bool stream_pool_active = false;
  bool gpu_monitor_active = false;
  bool nvimgcodec_initialized = false;

  if (use_gpu) {
    int nvimgcodec_streams =
        (options->cuda_streams > 0) ? options->cuda_streams : num_decode_threads;
    if (nvimgcodec_streams < 1) {
      nvimgcodec_streams = 1;
    }
    if (nvimgcodec_init(nvimgcodec_streams)) {
      nvimgcodec_ok = true;
      decode_ctx.nvimgcodec_ok = true;
      nvimgcodec_initialized = true;
      verboseLog(VERBOSE_NORMAL,
                 "nvImageCodec GPU decode: enabled (%d streams)\n",
                 nvimgcodec_streams);
    } else {
      verboseLog(VERBOSE_NORMAL,
                 "nvImageCodec GPU decode: unavailable, using CPU decode\n");
    }

    size_t stream_count =
        (options->cuda_streams > 0) ? (size_t)options->cuda_streams
                                    : (size_t)workers;
    if (stream_count < 1) {
      stream_count = 1;
    }
    if (cuda_stream_pool_global_init(stream_count)) {
      stream_pool_active = true;
      verboseLog(VERBOSE_NORMAL, "GPU stream pool: %zu streams\n",
                 stream_count);
    } else {
      verboseLog(VERBOSE_NORMAL,
                 "GPU stream pool initialization failed, using default stream\n");
    }

    if (options->perf && gpu_monitor_global_init()) {
      gpu_monitor_active = true;
      gpu_monitor_global_batch_start();
    }
  }
#endif

  decode_queue_set_custom_decoder(decode_queue, pdf_decode_queue_decoder,
                                  &decode_ctx);

  if (!decode_queue_start_producer(decode_queue, &batch_queue, &pdf_options)) {
    decode_queue_destroy(decode_queue);
    batch_queue_free(&batch_queue);
    pdf_page_accumulator_destroy(accumulator);
    pdf_writer_abort(writer);
    pdf_close(doc);
#ifdef UNPAPER_WITH_CUDA
    if (use_gpu) {
      if (gpu_monitor_active) {
        gpu_monitor_global_cleanup();
      }
      if (stream_pool_active) {
        cuda_stream_pool_global_cleanup();
      }
      if (nvimgcodec_initialized) {
        nvimgcodec_cleanup();
      }
    }
#endif
    return -1;
  }

  // Worker context (generic batch worker).
  BatchWorkerContext worker_ctx;
  batch_worker_init(&worker_ctx, &pdf_options, &batch_queue);
  batch_worker_set_config(&worker_ctx, &pdf_config);
  batch_worker_set_decode_queue(&worker_ctx, decode_queue);
#ifdef UNPAPER_WITH_CUDA
  if (use_gpu && cuda_stream_pool_global_active()) {
    batch_worker_enable_stream_pool(&worker_ctx, true);
  }
#endif

  PdfBatchWriteContext write_ctx = {.accumulator = accumulator,
                                   .options = &pdf_options,
                                   .use_gpu_encode = (use_gpu && nvimgcodec_ok)};
  batch_worker_set_post_process_callback(&worker_ctx, pdf_batch_post_process,
                                         &write_ctx);

  ThreadPool *pool = threadpool_create(workers);
  if (pool == NULL) {
    decode_queue_stop_producer(decode_queue);
    if (options->perf) {
      decode_queue_print_stats(decode_queue);
    }
    decode_queue_destroy(decode_queue);
    batch_worker_cleanup(&worker_ctx);
    batch_queue_free(&batch_queue);
    pdf_page_accumulator_destroy(accumulator);
    pdf_writer_abort(writer);
    pdf_close(doc);
#ifdef UNPAPER_WITH_CUDA
    if (use_gpu) {
      if (gpu_monitor_active) {
        gpu_monitor_global_cleanup();
      }
      if (stream_pool_active) {
        cuda_stream_pool_global_cleanup();
      }
      if (nvimgcodec_initialized) {
        nvimgcodec_cleanup();
      }
    }
#endif
    return -1;
  }

  int failed_jobs = batch_process_parallel(&worker_ctx, pool);

  threadpool_destroy(pool);

  decode_queue_stop_producer(decode_queue);
  if (options->perf) {
    decode_queue_print_stats(decode_queue);
  }
  decode_queue_destroy(decode_queue);

  // Mark any failed pages so the accumulator doesn't wait forever.
  for (size_t i = 0; i < batch_queue.count; i++) {
    BatchJob *job = batch_queue_get(&batch_queue, i);
    if (job && job->status != BATCH_JOB_COMPLETED) {
      for (int o = 0; o < output_pages; o++) {
        pdf_page_accumulator_mark_failed(accumulator, (int)i * output_pages + o);
      }
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

#ifdef UNPAPER_WITH_CUDA
  if (use_gpu) {
    if (options->perf) {
      nvimgcodec_print_stats();
      if (stream_pool_active) {
        cuda_stream_pool_global_print_stats();
      }
      if (gpu_monitor_active) {
        gpu_monitor_global_batch_end();
        gpu_monitor_global_print_stats();
      }
      unpaper_cuda_print_async_stats();
    }

    if (gpu_monitor_active) {
      gpu_monitor_global_cleanup();
    }
    if (stream_pool_active) {
      cuda_stream_pool_global_cleanup();
    }
    if (nvimgcodec_initialized) {
      nvimgcodec_cleanup();
    }
  }
#endif

  return pages_failed;
}
