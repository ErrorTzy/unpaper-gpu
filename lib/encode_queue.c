// SPDX-FileCopyrightText: 2025 The unpaper authors
//
// SPDX-License-Identifier: GPL-2.0-only

#include "lib/encode_queue.h"

#include <libavcodec/avcodec.h>
#include <libavformat/avformat.h>
#include <libavutil/avutil.h>
#include <libavutil/frame.h>
#include <libavutil/imgutils.h>
#include <libavutil/opt.h>
#include <pthread.h>
#include <stdatomic.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <strings.h>
#include <sys/time.h>

#ifdef UNPAPER_WITH_CUDA
#include "imageprocess/cuda_runtime.h"
#include "imageprocess/nvjpeg_encode.h"
#include <cuda_runtime.h>
#endif

// Slot states
typedef enum {
  SLOT_EMPTY = 0, // Available for producer
  SLOT_FILLING,   // Producer is filling this slot
  SLOT_READY,     // Ready for encoder
  SLOT_ENCODING,  // Encoder is processing
} SlotState;

// A slot in the encode queue
typedef struct {
  EncodeJob job;
  atomic_int state;
} EncodeSlot;

struct EncodeQueue {
  // Slot array (circular buffer)
  EncodeSlot *slots;
  size_t queue_depth;

  // Encoder threads
  pthread_t *encoder_threads;
  int num_encoder_threads;
  bool threads_started;

  // Control flags
  atomic_bool accepting_jobs; // True while accepting new submissions
  atomic_bool shutdown;       // True when shutting down
  atomic_int active_encoders; // Count of encoders currently processing

  // GPU encoding support (PR37)
  bool gpu_encode_enabled; // True if GPU encoding is enabled
  int gpu_encode_quality;  // JPEG quality for GPU encoding

  // Synchronization
  pthread_mutex_t mutex;
  pthread_cond_t not_full;  // Signal when slot becomes empty
  pthread_cond_t not_empty; // Signal when slot becomes ready

  // Statistics (atomic for thread safety)
  atomic_size_t images_queued;
  atomic_size_t images_encoded;
  atomic_size_t producer_waits;
  atomic_size_t consumer_waits;
  atomic_size_t current_depth;
  atomic_size_t peak_depth;
  atomic_uint_fast64_t total_encode_time_us; // Microseconds for precision
};

// Get current time in microseconds
static uint64_t get_time_us(void) {
  struct timeval tv;
  gettimeofday(&tv, NULL);
  return (uint64_t)tv.tv_sec * 1000000ULL + (uint64_t)tv.tv_usec;
}

// Fast direct PNM writer - matches file.c saveImageDirect()
static bool encode_direct(const char *filename, AVFrame *frame,
                          int output_format) {
  // Handle format conversion if needed
  int format = frame->format;

  // Map Y400A to GRAY8 for output
  if (output_format == AV_PIX_FMT_Y400A) {
    output_format = AV_PIX_FMT_GRAY8;
  }
  // Map MONOBLACK to MONOWHITE for output
  if (output_format == AV_PIX_FMT_MONOBLACK) {
    output_format = AV_PIX_FMT_MONOWHITE;
  }

  // Only direct encode if format matches
  if (format != output_format) {
    return false;
  }

  FILE *f = fopen(filename, "wb");
  if (!f) {
    return false;
  }

  int width = frame->width;
  int height = frame->height;

  switch (format) {
  case AV_PIX_FMT_GRAY8: {
    fprintf(f, "P5\n%d %d\n255\n", width, height);
    for (int y = 0; y < height; y++) {
      fwrite(frame->data[0] + y * frame->linesize[0], 1, width, f);
    }
    break;
  }
  case AV_PIX_FMT_RGB24: {
    fprintf(f, "P6\n%d %d\n255\n", width, height);
    for (int y = 0; y < height; y++) {
      fwrite(frame->data[0] + y * frame->linesize[0], 1, width * 3, f);
    }
    break;
  }
  case AV_PIX_FMT_MONOWHITE: {
    fprintf(f, "P4\n%d %d\n", width, height);
    int row_bytes = (width + 7) / 8;
    for (int y = 0; y < height; y++) {
      const uint8_t *src = frame->data[0] + y * frame->linesize[0];
      for (int x = 0; x < row_bytes; x++) {
        uint8_t inverted = src[x] ^ 0xFF;
        fputc(inverted, f);
      }
    }
    break;
  }
  default:
    fclose(f);
    return false;
  }

  fclose(f);
  return true;
}

// Full encode via FFmpeg - based on file.c saveImage()
static bool encode_ffmpeg(const char *filename, AVFrame *input_frame,
                          int output_format) {
  // Handle format mappings
  if (output_format == AV_PIX_FMT_Y400A) {
    output_format = AV_PIX_FMT_GRAY8;
  }
  if (output_format == AV_PIX_FMT_MONOBLACK) {
    output_format = AV_PIX_FMT_MONOWHITE;
  }

  AVFrame *output_frame = input_frame;
  bool allocated_output = false;

  // Convert format if needed
  if (input_frame->format != output_format) {
    output_frame = av_frame_alloc();
    if (!output_frame) {
      return false;
    }
    output_frame->width = input_frame->width;
    output_frame->height = input_frame->height;
    output_frame->format = output_format;

    if (av_frame_get_buffer(output_frame, 0) < 0) {
      av_frame_free(&output_frame);
      return false;
    }
    allocated_output = true;

    int width = input_frame->width;
    int height = input_frame->height;
    bool converted = false;

    // Fast paths for common conversions
    if (input_frame->format == AV_PIX_FMT_RGB24 &&
        output_format == AV_PIX_FMT_MONOWHITE) {
      for (int y = 0; y < height; y++) {
        const uint8_t *src =
            input_frame->data[0] + y * input_frame->linesize[0];
        uint8_t *dst = output_frame->data[0] + y * output_frame->linesize[0];
        for (int x = 0; x < width; x++) {
          int gray = (src[x * 3] + src[x * 3 + 1] + src[x * 3 + 2]) / 3;
          int bit_index = x % 8;
          if (bit_index == 0)
            dst[x / 8] = 0;
          if (gray > 127)
            dst[x / 8] |= (0x80 >> bit_index);
        }
      }
      converted = true;
    } else if (input_frame->format == AV_PIX_FMT_RGB24 &&
               output_format == AV_PIX_FMT_GRAY8) {
      for (int y = 0; y < height; y++) {
        const uint8_t *src =
            input_frame->data[0] + y * input_frame->linesize[0];
        uint8_t *dst = output_frame->data[0] + y * output_frame->linesize[0];
        for (int x = 0; x < width; x++) {
          // Simple average for grayscale conversion
          dst[x] = (src[x * 3] + src[x * 3 + 1] + src[x * 3 + 2]) / 3;
        }
      }
      converted = true;
    } else if (input_frame->format == AV_PIX_FMT_GRAY8 &&
               output_format == AV_PIX_FMT_MONOWHITE) {
      for (int y = 0; y < height; y++) {
        const uint8_t *src =
            input_frame->data[0] + y * input_frame->linesize[0];
        uint8_t *dst = output_frame->data[0] + y * output_frame->linesize[0];
        for (int x = 0; x < width; x++) {
          int bit_index = x % 8;
          if (bit_index == 0)
            dst[x / 8] = 0;
          if (src[x] > 127)
            dst[x / 8] |= (0x80 >> bit_index);
        }
      }
      converted = true;
    }

    if (!converted) {
      // No fast path available - fall back to FFmpeg encoder
      // Note: av_image_copy doesn't do format conversion, only copies bytes
      av_frame_free(&output_frame);
      output_frame = input_frame;
      allocated_output = false;
    }
  }

  // Try direct encode first
  if (encode_direct(filename, output_frame, output_format)) {
    if (allocated_output) {
      av_frame_free(&output_frame);
    }
    return true;
  }

  // Fall back to FFmpeg encoder
  enum AVCodecID output_codec = -1;
  switch (output_format) {
  case AV_PIX_FMT_RGB24:
    output_codec = AV_CODEC_ID_PPM;
    break;
  case AV_PIX_FMT_GRAY8:
    output_codec = AV_CODEC_ID_PGM;
    break;
  case AV_PIX_FMT_MONOWHITE:
    output_codec = AV_CODEC_ID_PBM;
    break;
  default:
    if (allocated_output) {
      av_frame_free(&output_frame);
    }
    return false;
  }

  const AVCodec *codec = avcodec_find_encoder(output_codec);
  if (!codec) {
    if (allocated_output) {
      av_frame_free(&output_frame);
    }
    return false;
  }

  AVFormatContext *out_ctx = NULL;
  if (avformat_alloc_output_context2(&out_ctx, NULL, "image2", filename) < 0 ||
      out_ctx == NULL) {
    if (allocated_output) {
      av_frame_free(&output_frame);
    }
    return false;
  }

  av_opt_set(out_ctx->priv_data, "update", "true", 0);

  AVStream *video_st = avformat_new_stream(out_ctx, codec);
  if (!video_st) {
    avformat_free_context(out_ctx);
    if (allocated_output) {
      av_frame_free(&output_frame);
    }
    return false;
  }

  AVCodecContext *codec_ctx = avcodec_alloc_context3(codec);
  if (!codec_ctx) {
    avformat_free_context(out_ctx);
    if (allocated_output) {
      av_frame_free(&output_frame);
    }
    return false;
  }

  codec_ctx->width = output_frame->width;
  codec_ctx->height = output_frame->height;
  codec_ctx->pix_fmt = output_frame->format;
  video_st->codecpar->width = output_frame->width;
  video_st->codecpar->height = output_frame->height;
  video_st->codecpar->format = output_frame->format;
  video_st->time_base.den = codec_ctx->time_base.den = 1;
  video_st->time_base.num = codec_ctx->time_base.num = 1;

  if (avcodec_open2(codec_ctx, codec, NULL) < 0) {
    avcodec_free_context(&codec_ctx);
    avformat_free_context(out_ctx);
    if (allocated_output) {
      av_frame_free(&output_frame);
    }
    return false;
  }

  if (avio_open(&out_ctx->pb, filename, AVIO_FLAG_WRITE) < 0) {
    avcodec_free_context(&codec_ctx);
    avformat_free_context(out_ctx);
    if (allocated_output) {
      av_frame_free(&output_frame);
    }
    return false;
  }

  if (avformat_write_header(out_ctx, NULL) < 0) {
    avio_closep(&out_ctx->pb);
    avcodec_free_context(&codec_ctx);
    avformat_free_context(out_ctx);
    if (allocated_output) {
      av_frame_free(&output_frame);
    }
    return false;
  }

  AVPacket *pkt = av_packet_alloc();
  if (!pkt) {
    av_write_trailer(out_ctx);
    avio_closep(&out_ctx->pb);
    avcodec_free_context(&codec_ctx);
    avformat_free_context(out_ctx);
    if (allocated_output) {
      av_frame_free(&output_frame);
    }
    return false;
  }

  bool success = false;
  if (avcodec_send_frame(codec_ctx, output_frame) >= 0) {
    if (avcodec_receive_packet(codec_ctx, pkt) >= 0) {
      av_write_frame(out_ctx, pkt);
      success = true;
    }
  }

  av_packet_free(&pkt);
  av_write_trailer(out_ctx);
  avio_closep(&out_ctx->pb);
  avcodec_free_context(&codec_ctx);
  avformat_free_context(out_ctx);
  if (allocated_output) {
    av_frame_free(&output_frame);
  }

  return success;
}

// Find an empty slot (lock-free fast path)
static int find_empty_slot(EncodeQueue *queue) {
  for (size_t i = 0; i < queue->queue_depth; i++) {
    int expected = SLOT_EMPTY;
    if (atomic_compare_exchange_strong(&queue->slots[i].state, &expected,
                                       SLOT_FILLING)) {
      return (int)i;
    }
  }
  return -1;
}

// Find a ready slot for encoding (lock-free fast path)
static int find_ready_slot(EncodeQueue *queue) {
  for (size_t i = 0; i < queue->queue_depth; i++) {
    int expected = SLOT_READY;
    if (atomic_compare_exchange_strong(&queue->slots[i].state, &expected,
                                       SLOT_ENCODING)) {
      return (int)i;
    }
  }
  return -1;
}

// Free a job's resources
static void free_job_frame(EncodeJob *job) {
  if (!job || !job->frame) {
    return;
  }

#ifdef UNPAPER_WITH_CUDA
  if (job->uses_pinned_memory && job->frame->data[0]) {
    UnpaperCudaPinnedBuffer buf = {
        .ptr = job->frame->data[0], .bytes = 0, .is_pinned = true};
    job->frame->data[0] = NULL;
    unpaper_cuda_pinned_free(&buf);
  }
#endif

  av_frame_free(&job->frame);
  job->frame = NULL;
}

// Encoder thread function
static void *encoder_thread_fn(void *arg) {
  EncodeQueue *queue = (EncodeQueue *)arg;

  while (1) {
    // Try to find a ready slot
    int slot_idx = find_ready_slot(queue);

    if (slot_idx < 0) {
      // No ready slot - check if we should exit
      if (atomic_load(&queue->shutdown)) {
        // One more check for any remaining work
        slot_idx = find_ready_slot(queue);
        if (slot_idx < 0) {
          break; // No work and shutting down
        }
      } else {
        // Wait for work
        atomic_fetch_add(&queue->consumer_waits, 1);
        pthread_mutex_lock(&queue->mutex);
        // Double-check under lock
        if (!atomic_load(&queue->shutdown)) {
          pthread_cond_wait(&queue->not_empty, &queue->mutex);
        }
        pthread_mutex_unlock(&queue->mutex);
        continue;
      }
    }

    // Process the job
    atomic_fetch_add(&queue->active_encoders, 1);
    EncodeSlot *slot = &queue->slots[slot_idx];
    EncodeJob *job = &slot->job;

    uint64_t start_time = get_time_us();

    // Encode each output file
    if (job->valid && job->frame) {
      if (job->output_count == 1) {
        // Single output - encode directly
        encode_ffmpeg(job->output_files[0], job->frame,
                      job->output_pixel_format);
      } else {
        // Multiple outputs - split the frame
        int output_width = job->frame->width / job->output_count;
        for (int j = 0; j < job->output_count; j++) {
          // Create a sub-frame for this output
          AVFrame *sub_frame = av_frame_alloc();
          if (sub_frame) {
            sub_frame->width = output_width;
            sub_frame->height = job->frame->height;
            sub_frame->format = job->frame->format;
            if (av_frame_get_buffer(sub_frame, 0) >= 0) {
              // Copy the relevant portion
              int bytes_per_pixel = 1;
              if (job->frame->format == AV_PIX_FMT_RGB24) {
                bytes_per_pixel = 3;
              } else if (job->frame->format == AV_PIX_FMT_MONOWHITE ||
                         job->frame->format == AV_PIX_FMT_MONOBLACK) {
                // For mono formats, we need to handle bit-packed data
                int src_offset_bits = output_width * j;
                for (int y = 0; y < job->frame->height; y++) {
                  const uint8_t *src_row =
                      job->frame->data[0] + y * job->frame->linesize[0];
                  uint8_t *dst_row =
                      sub_frame->data[0] + y * sub_frame->linesize[0];
                  for (int x = 0; x < output_width; x++) {
                    int src_bit = src_offset_bits + x;
                    int src_byte = src_bit / 8;
                    int src_bit_idx = 7 - (src_bit % 8);
                    int dst_byte = x / 8;
                    int dst_bit_idx = 7 - (x % 8);
                    if (dst_bit_idx == 7) {
                      dst_row[dst_byte] = 0;
                    }
                    if ((src_row[src_byte] >> src_bit_idx) & 1) {
                      dst_row[dst_byte] |= (1 << dst_bit_idx);
                    }
                  }
                }
                encode_ffmpeg(job->output_files[j], sub_frame,
                              job->output_pixel_format);
                av_frame_free(&sub_frame);
                continue;
              }

              for (int y = 0; y < job->frame->height; y++) {
                const uint8_t *src = job->frame->data[0] +
                                     y * job->frame->linesize[0] +
                                     output_width * j * bytes_per_pixel;
                uint8_t *dst = sub_frame->data[0] + y * sub_frame->linesize[0];
                memcpy(dst, src, output_width * bytes_per_pixel);
              }
              encode_ffmpeg(job->output_files[j], sub_frame,
                            job->output_pixel_format);
            }
            av_frame_free(&sub_frame);
          }
        }
      }
    }

    uint64_t end_time = get_time_us();
    atomic_fetch_add(&queue->total_encode_time_us, end_time - start_time);

    // Free resources
    free_job_frame(job);
    job->valid = false;

    // Update stats
    atomic_fetch_add(&queue->images_encoded, 1);
    atomic_fetch_sub(&queue->current_depth, 1);
    atomic_fetch_sub(&queue->active_encoders, 1);

    // Mark slot as empty
    atomic_store(&slot->state, SLOT_EMPTY);

    // Signal producers
    pthread_mutex_lock(&queue->mutex);
    pthread_cond_signal(&queue->not_full);
    pthread_mutex_unlock(&queue->mutex);
  }

  return NULL;
}

EncodeQueue *encode_queue_create(size_t queue_depth, int num_encoder_threads) {
  if (queue_depth == 0 || num_encoder_threads <= 0) {
    return NULL;
  }

  EncodeQueue *queue = calloc(1, sizeof(EncodeQueue));
  if (!queue) {
    return NULL;
  }

  queue->slots = calloc(queue_depth, sizeof(EncodeSlot));
  if (!queue->slots) {
    free(queue);
    return NULL;
  }

  queue->encoder_threads = calloc(num_encoder_threads, sizeof(pthread_t));
  if (!queue->encoder_threads) {
    free(queue->slots);
    free(queue);
    return NULL;
  }

  queue->queue_depth = queue_depth;
  queue->num_encoder_threads = num_encoder_threads;
  queue->threads_started = false;

  atomic_init(&queue->accepting_jobs, true);
  atomic_init(&queue->shutdown, false);
  atomic_init(&queue->active_encoders, 0);

  pthread_mutex_init(&queue->mutex, NULL);
  pthread_cond_init(&queue->not_full, NULL);
  pthread_cond_init(&queue->not_empty, NULL);

  atomic_init(&queue->images_queued, 0);
  atomic_init(&queue->images_encoded, 0);
  atomic_init(&queue->producer_waits, 0);
  atomic_init(&queue->consumer_waits, 0);
  atomic_init(&queue->current_depth, 0);
  atomic_init(&queue->peak_depth, 0);
  atomic_init(&queue->total_encode_time_us, 0);

  // GPU encoding support (PR37)
  queue->gpu_encode_enabled = false;
  queue->gpu_encode_quality = 85;

  // Initialize slots
  for (size_t i = 0; i < queue_depth; i++) {
    atomic_init(&queue->slots[i].state, SLOT_EMPTY);
    queue->slots[i].job.frame = NULL;
    queue->slots[i].job.valid = false;
  }

  return queue;
}

void encode_queue_destroy(EncodeQueue *queue) {
  if (!queue) {
    return;
  }

  // Signal shutdown and wait for threads
  encode_queue_signal_done(queue);
  encode_queue_wait(queue);

  // Free any remaining frames
  for (size_t i = 0; i < queue->queue_depth; i++) {
    free_job_frame(&queue->slots[i].job);
  }

  pthread_mutex_destroy(&queue->mutex);
  pthread_cond_destroy(&queue->not_full);
  pthread_cond_destroy(&queue->not_empty);

  free(queue->encoder_threads);
  free(queue->slots);
  free(queue);
}

bool encode_queue_start(EncodeQueue *queue) {
  if (!queue || queue->threads_started) {
    return false;
  }

  for (int i = 0; i < queue->num_encoder_threads; i++) {
    if (pthread_create(&queue->encoder_threads[i], NULL, encoder_thread_fn,
                       queue) != 0) {
      // Failed - stop already-started threads
      atomic_store(&queue->shutdown, true);
      pthread_mutex_lock(&queue->mutex);
      pthread_cond_broadcast(&queue->not_empty);
      pthread_mutex_unlock(&queue->mutex);
      for (int j = 0; j < i; j++) {
        pthread_join(queue->encoder_threads[j], NULL);
      }
      return false;
    }
  }

  queue->threads_started = true;
  return true;
}

void encode_queue_signal_done(EncodeQueue *queue) {
  if (!queue) {
    return;
  }

  atomic_store(&queue->accepting_jobs, false);
  atomic_store(&queue->shutdown, true);

  // Wake up all encoder threads
  pthread_mutex_lock(&queue->mutex);
  pthread_cond_broadcast(&queue->not_empty);
  pthread_mutex_unlock(&queue->mutex);
}

void encode_queue_wait(EncodeQueue *queue) {
  if (!queue || !queue->threads_started) {
    return;
  }

  for (int i = 0; i < queue->num_encoder_threads; i++) {
    pthread_join(queue->encoder_threads[i], NULL);
  }

  queue->threads_started = false;
}

bool encode_queue_submit(EncodeQueue *queue, struct AVFrame *frame,
                         char **output_files, int output_count,
                         int output_pixel_format, int job_index,
                         bool uses_pinned_memory) {
  if (!queue || !frame || !output_files || output_count <= 0) {
    return false;
  }

  if (!atomic_load(&queue->accepting_jobs)) {
    return false;
  }

  // Find an empty slot
  int slot_idx = -1;
  while (atomic_load(&queue->accepting_jobs)) {
    slot_idx = find_empty_slot(queue);
    if (slot_idx >= 0) {
      break;
    }

    // No empty slot - wait
    atomic_fetch_add(&queue->producer_waits, 1);
    pthread_mutex_lock(&queue->mutex);
    if (atomic_load(&queue->accepting_jobs)) {
      pthread_cond_wait(&queue->not_full, &queue->mutex);
    }
    pthread_mutex_unlock(&queue->mutex);
  }

  if (slot_idx < 0) {
    return false;
  }

  // Fill the slot
  EncodeSlot *slot = &queue->slots[slot_idx];
  slot->job.frame = frame;
  slot->job.output_count =
      output_count > ENCODE_MAX_OUTPUTS ? ENCODE_MAX_OUTPUTS : output_count;
  for (int i = 0; i < slot->job.output_count; i++) {
    slot->job.output_files[i] = output_files[i];
  }
  slot->job.output_pixel_format = output_pixel_format;
  slot->job.job_index = job_index;
  slot->job.valid = true;
  slot->job.uses_pinned_memory = uses_pinned_memory;

  // Update stats
  atomic_fetch_add(&queue->images_queued, 1);
  size_t depth = atomic_fetch_add(&queue->current_depth, 1) + 1;
  size_t peak = atomic_load(&queue->peak_depth);
  while (depth > peak) {
    if (atomic_compare_exchange_weak(&queue->peak_depth, &peak, depth)) {
      break;
    }
  }

  // Mark ready and signal encoders
  atomic_store(&slot->state, SLOT_READY);

  pthread_mutex_lock(&queue->mutex);
  pthread_cond_signal(&queue->not_empty);
  pthread_mutex_unlock(&queue->mutex);

  return true;
}

bool encode_queue_active(const EncodeQueue *queue) {
  if (!queue) {
    return false;
  }
  return atomic_load(&queue->accepting_jobs);
}

EncodeQueueStats encode_queue_get_stats(const EncodeQueue *queue) {
  EncodeQueueStats stats = {0};
  if (!queue) {
    return stats;
  }

  stats.images_queued = atomic_load(&queue->images_queued);
  stats.images_encoded = atomic_load(&queue->images_encoded);
  stats.producer_waits = atomic_load(&queue->producer_waits);
  stats.consumer_waits = atomic_load(&queue->consumer_waits);
  stats.peak_queue_depth = atomic_load(&queue->peak_depth);

  uint64_t total_us = atomic_load(&queue->total_encode_time_us);
  stats.total_encode_time_ms = (double)total_us / 1000.0;
  if (stats.images_encoded > 0) {
    stats.avg_encode_time_ms =
        stats.total_encode_time_ms / (double)stats.images_encoded;
  }

  return stats;
}

void encode_queue_print_stats(const EncodeQueue *queue) {
  if (!queue) {
    return;
  }

  EncodeQueueStats stats = encode_queue_get_stats(queue);

  double producer_wait_rate = 0.0;
  double consumer_wait_rate = 0.0;
  if (stats.images_queued > 0) {
    producer_wait_rate =
        100.0 * (double)stats.producer_waits / (double)stats.images_queued;
  }
  if (stats.images_encoded > 0) {
    consumer_wait_rate =
        100.0 * (double)stats.consumer_waits / (double)stats.images_encoded;
  }

  fprintf(stderr,
          "Encode Queue Statistics:\n"
          "  Queue depth: %zu slots\n"
          "  Encoder threads: %d\n"
          "  Images queued: %zu\n"
          "  Images encoded: %zu\n"
          "  Producer waits (queue full): %zu (%.1f%%)\n"
          "  Consumer waits (queue empty): %zu (%.1f%%)\n"
          "  Peak queue occupancy: %zu\n"
          "  Total encode time: %.2f ms\n"
          "  Average encode time: %.2f ms/image\n",
          queue->queue_depth, queue->num_encoder_threads, stats.images_queued,
          stats.images_encoded, stats.producer_waits, producer_wait_rate,
          stats.consumer_waits, consumer_wait_rate, stats.peak_queue_depth,
          stats.total_encode_time_ms, stats.avg_encode_time_ms);
}

// ============================================================================
// GPU Encode Support (PR37)
// ============================================================================

void encode_queue_enable_gpu(EncodeQueue *queue, bool enable, int quality) {
  if (!queue) {
    return;
  }

  queue->gpu_encode_enabled = enable;
  if (quality > 0) {
    queue->gpu_encode_quality = (quality > 100) ? 100 : quality;
  }

#ifdef UNPAPER_WITH_CUDA
  if (enable && nvjpeg_encode_is_available()) {
    nvjpeg_encode_set_quality(queue->gpu_encode_quality);
  }
#endif
}

bool encode_queue_gpu_enabled(const EncodeQueue *queue) {
  if (!queue) {
    return false;
  }
  return queue->gpu_encode_enabled;
}

// Helper: Check if filename has JPEG extension
static bool is_jpeg_output(const char *filename) {
  if (filename == NULL) {
    return false;
  }
  size_t len = strlen(filename);
  if (len < 4) {
    return false;
  }
  const char *ext = filename + len - 4;
  if (strcasecmp(ext, ".jpg") == 0) {
    return true;
  }
  if (len >= 5) {
    ext = filename + len - 5;
    if (strcasecmp(ext, ".jpeg") == 0) {
      return true;
    }
  }
  return false;
}

bool encode_queue_submit_gpu(EncodeQueue *queue, void *gpu_ptr, size_t pitch,
                             int width, int height, int channels,
                             char **output_files, int output_count,
                             int job_index) {
#ifdef UNPAPER_WITH_CUDA
  if (!queue || !gpu_ptr || !output_files || output_count <= 0) {
    return false;
  }

  if (!atomic_load(&queue->accepting_jobs)) {
    return false;
  }

  // Check if all outputs are JPEG and GPU encoding is available
  bool all_jpeg = true;
  for (int i = 0; i < output_count; i++) {
    if (!is_jpeg_output(output_files[i])) {
      all_jpeg = false;
      break;
    }
  }

  bool use_gpu_encode =
      queue->gpu_encode_enabled && all_jpeg && nvjpeg_encode_is_available();

  if (use_gpu_encode) {
    // Direct GPU encoding path - no queue needed, encode immediately
    // This avoids the D2H transfer entirely
    NvJpegEncodeFormat fmt =
        (channels == 1) ? NVJPEG_ENC_FMT_GRAY8 : NVJPEG_ENC_FMT_RGB;

    for (int i = 0; i < output_count; i++) {
      bool result = nvjpeg_encode_gpu_to_file(gpu_ptr, pitch, width, height,
                                              fmt, NULL, output_files[i]);
      if (!result) {
        // Fall back to CPU encoding for this file
        goto fallback_cpu;
      }
    }

    // Success - update stats
    atomic_fetch_add(&queue->images_queued, 1);
    atomic_fetch_add(&queue->images_encoded, 1);
    return true;
  }

fallback_cpu:
  // Fall back to D2H transfer + CPU encoding
  {
    // Allocate CPU frame
    AVFrame *frame = av_frame_alloc();
    if (!frame) {
      return false;
    }

    frame->width = width;
    frame->height = height;
    frame->format = (channels == 1) ? AV_PIX_FMT_GRAY8 : AV_PIX_FMT_RGB24;

    if (av_frame_get_buffer(frame, 0) < 0) {
      av_frame_free(&frame);
      return false;
    }

    // D2H transfer
    size_t row_bytes = (size_t)width * (size_t)channels;
    for (int y = 0; y < height; y++) {
      cudaError_t err =
          cudaMemcpy(frame->data[0] + y * frame->linesize[0],
                     (uint8_t *)gpu_ptr + y * pitch, row_bytes,
                     cudaMemcpyDeviceToHost);
      if (err != cudaSuccess) {
        av_frame_free(&frame);
        return false;
      }
    }

    // Submit to regular encode queue
    int output_pixel_format = frame->format;
    bool result = encode_queue_submit(queue, frame, output_files, output_count,
                                      output_pixel_format, job_index, false);
    if (!result) {
      av_frame_free(&frame);
    }
    return result;
  }
#else
  // No CUDA support - can't handle GPU pointer
  (void)queue;
  (void)gpu_ptr;
  (void)pitch;
  (void)width;
  (void)height;
  (void)channels;
  (void)output_files;
  (void)output_count;
  (void)job_index;
  return false;
#endif
}
