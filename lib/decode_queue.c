// SPDX-FileCopyrightText: 2025 The unpaper authors
//
// SPDX-License-Identifier: GPL-2.0-only

#include "lib/decode_queue.h"
#include "lib/logging.h"

#include <libavcodec/avcodec.h>
#include <libavformat/avformat.h>
#include <libavutil/avutil.h>
#include <libavutil/frame.h>
#include <libavutil/imgutils.h>
#include <pthread.h>
#include <stdatomic.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#ifdef UNPAPER_WITH_CUDA
#include "imageprocess/cuda_runtime.h"
#endif

// Slot states
typedef enum {
  SLOT_EMPTY = 0,     // Available for producer
  SLOT_DECODING,      // Producer is decoding into this slot
  SLOT_READY,         // Decoded and ready for consumer
  SLOT_IN_USE,        // Consumer is using this slot
} SlotState;

// A slot in the decode queue
typedef struct {
  DecodedImage image;
  atomic_int state;
} DecodeSlot;

// Maximum number of parallel producer threads
#define MAX_PRODUCER_THREADS 16

struct DecodeQueue {
  // Slot array (circular buffer)
  DecodeSlot *slots;
  size_t queue_depth;

  // Producer state - supports multiple threads
  pthread_t producer_threads[MAX_PRODUCER_THREADS];
  int num_producers;
  bool producer_started;
  atomic_bool producer_running;
  atomic_int producers_done;  // Counter of completed producers

  // Shared job counter for work distribution among producers
  atomic_size_t next_job_index;

  // Source data for producer
  BatchQueue *batch_queue;
  const Options *options;

  // Configuration
  bool use_pinned_memory;

  // Synchronization
  pthread_mutex_t mutex;
  pthread_cond_t not_full;   // Signal when slot becomes empty
  pthread_cond_t not_empty;  // Signal when slot becomes ready

  // Statistics (atomic for thread safety)
  atomic_size_t images_decoded;
  atomic_size_t images_consumed;
  atomic_size_t producer_waits;
  atomic_size_t consumer_waits;
  atomic_size_t pinned_allocations;
  atomic_size_t current_depth;
  atomic_size_t peak_depth;
};

// Allocate a frame, optionally with pinned memory
static AVFrame *alloc_frame_pinned(int width, int height, int format,
                                   bool use_pinned, bool *is_pinned) {
  AVFrame *frame = av_frame_alloc();
  if (!frame) {
    return NULL;
  }

  frame->width = width;
  frame->height = height;
  frame->format = format;

  if (is_pinned) {
    *is_pinned = false;
  }

#ifdef UNPAPER_WITH_CUDA
  if (use_pinned) {
    // Calculate required buffer size
    int size = av_image_get_buffer_size(format, width, height, 1);
    if (size > 0) {
      UnpaperCudaPinnedBuffer buf;
      if (unpaper_cuda_pinned_alloc(&buf, size)) {
        // Fill frame with our pinned buffer
        int ret = av_image_fill_arrays(frame->data, frame->linesize, buf.ptr,
                                       format, width, height, 1);
        if (ret >= 0) {
          // Store buffer info in frame's opaque for later cleanup
          // We use buf field directly since we allocated it
          frame->buf[0] = NULL; // Mark as externally managed
          if (is_pinned) {
            *is_pinned = buf.is_pinned;
          }
          return frame;
        }
        // Failed to fill arrays - free pinned buffer
        unpaper_cuda_pinned_free(&buf);
      }
    }
  }
#else
  (void)use_pinned;
#endif

  // Fallback to regular allocation
  int ret = av_frame_get_buffer(frame, 0);
  if (ret < 0) {
    av_frame_free(&frame);
    return NULL;
  }

  return frame;
}

// Decode a single image file
// Returns decoded frame or NULL on failure
static AVFrame *decode_image_file(const char *filename) {
  int ret;
  AVFormatContext *s = NULL;
  AVCodecContext *avctx = NULL;
  const AVCodec *codec;
  AVPacket pkt;
  AVFrame *frame = av_frame_alloc();

  if (!frame) {
    return NULL;
  }

  ret = avformat_open_input(&s, filename, NULL, NULL);
  if (ret < 0) {
    av_frame_free(&frame);
    return NULL;
  }

  avformat_find_stream_info(s, NULL);

  if (s->nb_streams < 1) {
    avformat_close_input(&s);
    av_frame_free(&frame);
    return NULL;
  }

  codec = avcodec_find_decoder(s->streams[0]->codecpar->codec_id);
  if (!codec) {
    avformat_close_input(&s);
    av_frame_free(&frame);
    return NULL;
  }

  avctx = avcodec_alloc_context3(codec);
  if (!avctx) {
    avformat_close_input(&s);
    av_frame_free(&frame);
    return NULL;
  }

  ret = avcodec_parameters_to_context(avctx, s->streams[0]->codecpar);
  if (ret < 0) {
    avcodec_free_context(&avctx);
    avformat_close_input(&s);
    av_frame_free(&frame);
    return NULL;
  }

  ret = avcodec_open2(avctx, codec, NULL);
  if (ret < 0) {
    avcodec_free_context(&avctx);
    avformat_close_input(&s);
    av_frame_free(&frame);
    return NULL;
  }

  ret = av_read_frame(s, &pkt);
  if (ret < 0 || pkt.stream_index != 0) {
    avcodec_free_context(&avctx);
    avformat_close_input(&s);
    av_frame_free(&frame);
    return NULL;
  }

  ret = avcodec_send_packet(avctx, &pkt);
  if (ret < 0) {
    av_packet_unref(&pkt);
    avcodec_free_context(&avctx);
    avformat_close_input(&s);
    av_frame_free(&frame);
    return NULL;
  }

  ret = avcodec_receive_frame(avctx, frame);
  av_packet_unref(&pkt);
  avcodec_free_context(&avctx);
  avformat_close_input(&s);

  if (ret < 0) {
    av_frame_free(&frame);
    return NULL;
  }

  return frame;
}

// Copy frame data to a pinned buffer frame
static AVFrame *copy_to_pinned_frame(AVFrame *src, bool *is_pinned) {
  if (!src) {
    return NULL;
  }

  AVFrame *dst = alloc_frame_pinned(src->width, src->height, src->format,
                                    true, is_pinned);
  if (!dst) {
    return NULL;
  }

  // Copy image data
  av_image_copy(dst->data, dst->linesize,
                (const uint8_t **)src->data, src->linesize,
                src->format, src->width, src->height);

  return dst;
}

// Find an empty slot for producer
static int find_empty_slot(DecodeQueue *queue) {
  for (size_t i = 0; i < queue->queue_depth; i++) {
    int expected = SLOT_EMPTY;
    if (atomic_compare_exchange_strong(&queue->slots[i].state, &expected,
                                       SLOT_DECODING)) {
      return (int)i;
    }
  }
  return -1;
}

// Find a ready slot matching job_index and input_index
static int find_ready_slot(DecodeQueue *queue, int job_index, int input_index) {
  for (size_t i = 0; i < queue->queue_depth; i++) {
    if (atomic_load(&queue->slots[i].state) == SLOT_READY) {
      DecodedImage *img = &queue->slots[i].image;
      if (img->job_index == job_index && img->input_index == input_index) {
        // Try to claim it
        int expected = SLOT_READY;
        if (atomic_compare_exchange_strong(&queue->slots[i].state, &expected,
                                           SLOT_IN_USE)) {
          return (int)i;
        }
      }
    }
  }
  return -1;
}

// Producer thread function - uses work stealing via shared job counter
static void *producer_thread_fn(void *arg) {
  DecodeQueue *queue = (DecodeQueue *)arg;
  BatchQueue *batch = queue->batch_queue;

  // Work stealing loop - each thread grabs jobs atomically
  while (atomic_load(&queue->producer_running)) {
    // Atomically get next job index
    size_t job_idx = atomic_fetch_add(&queue->next_job_index, 1);
    if (job_idx >= batch->count) {
      break;  // No more jobs
    }

    BatchJob *job = batch_queue_get(batch, job_idx);
    if (!job) {
      continue;
    }

    // Decode each input file for this job
    for (int input_idx = 0; input_idx < job->input_count; input_idx++) {
      if (!atomic_load(&queue->producer_running)) {
        break;
      }

      const char *filename = job->input_files[input_idx];
      if (!filename) {
        continue;  // Skip blank pages
      }

      // Wait for an empty slot
      int slot_idx = -1;
      while (atomic_load(&queue->producer_running)) {
        slot_idx = find_empty_slot(queue);
        if (slot_idx >= 0) {
          break;
        }

        // No empty slot - wait
        atomic_fetch_add(&queue->producer_waits, 1);
        pthread_mutex_lock(&queue->mutex);
        pthread_cond_wait(&queue->not_full, &queue->mutex);
        pthread_mutex_unlock(&queue->mutex);
      }

      if (slot_idx < 0 || !atomic_load(&queue->producer_running)) {
        break;
      }

      // Decode the image
      DecodeSlot *slot = &queue->slots[slot_idx];
      AVFrame *decoded = decode_image_file(filename);

      if (decoded) {
        bool is_pinned = false;
        AVFrame *final_frame = decoded;

        // Optionally copy to pinned memory
        if (queue->use_pinned_memory) {
          AVFrame *pinned_frame = copy_to_pinned_frame(decoded, &is_pinned);
          if (pinned_frame) {
            av_frame_free(&decoded);
            final_frame = pinned_frame;
            if (is_pinned) {
              atomic_fetch_add(&queue->pinned_allocations, 1);
            }
          }
        }

        slot->image.frame = final_frame;
        slot->image.job_index = (int)job_idx;
        slot->image.input_index = input_idx;
        slot->image.valid = true;
        slot->image.uses_pinned_memory = is_pinned;
      } else {
        // Decode failed - mark as invalid
        slot->image.frame = NULL;
        slot->image.job_index = (int)job_idx;
        slot->image.input_index = input_idx;
        slot->image.valid = false;
        slot->image.uses_pinned_memory = false;
      }

      // Mark slot as ready
      atomic_store(&slot->state, SLOT_READY);
      atomic_fetch_add(&queue->images_decoded, 1);

      // Update depth tracking
      size_t depth = atomic_fetch_add(&queue->current_depth, 1) + 1;
      size_t peak = atomic_load(&queue->peak_depth);
      while (depth > peak) {
        if (atomic_compare_exchange_weak(&queue->peak_depth, &peak, depth)) {
          break;
        }
      }

      // Signal consumers
      pthread_mutex_lock(&queue->mutex);
      pthread_cond_broadcast(&queue->not_empty);
      pthread_mutex_unlock(&queue->mutex);
    }
  }

  // Mark this producer as done
  int done_count = atomic_fetch_add(&queue->producers_done, 1) + 1;

  // Last producer signals completion
  if (done_count >= queue->num_producers) {
    // Final signal to wake up any waiting consumers
    pthread_mutex_lock(&queue->mutex);
    pthread_cond_broadcast(&queue->not_empty);
    pthread_mutex_unlock(&queue->mutex);
  }

  return NULL;
}

// Internal create function with producer count
static DecodeQueue *decode_queue_create_internal(size_t queue_depth,
                                                  bool use_pinned_memory,
                                                  int num_producers) {
  if (queue_depth == 0 || num_producers < 1) {
    return NULL;
  }
  if (num_producers > MAX_PRODUCER_THREADS) {
    num_producers = MAX_PRODUCER_THREADS;
  }

  DecodeQueue *queue = calloc(1, sizeof(DecodeQueue));
  if (!queue) {
    return NULL;
  }

  queue->slots = calloc(queue_depth, sizeof(DecodeSlot));
  if (!queue->slots) {
    free(queue);
    return NULL;
  }

  queue->queue_depth = queue_depth;
  queue->use_pinned_memory = use_pinned_memory;
  queue->num_producers = num_producers;
  queue->producer_started = false;
  atomic_init(&queue->producer_running, false);
  atomic_init(&queue->producers_done, 0);
  atomic_init(&queue->next_job_index, 0);

  pthread_mutex_init(&queue->mutex, NULL);
  pthread_cond_init(&queue->not_full, NULL);
  pthread_cond_init(&queue->not_empty, NULL);

  atomic_init(&queue->images_decoded, 0);
  atomic_init(&queue->images_consumed, 0);
  atomic_init(&queue->producer_waits, 0);
  atomic_init(&queue->consumer_waits, 0);
  atomic_init(&queue->pinned_allocations, 0);
  atomic_init(&queue->current_depth, 0);
  atomic_init(&queue->peak_depth, 0);

  // Initialize slots
  for (size_t i = 0; i < queue_depth; i++) {
    atomic_init(&queue->slots[i].state, SLOT_EMPTY);
    queue->slots[i].image.frame = NULL;
    queue->slots[i].image.valid = false;
  }

  return queue;
}

DecodeQueue *decode_queue_create(size_t queue_depth, bool use_pinned_memory) {
  return decode_queue_create_internal(queue_depth, use_pinned_memory, 1);
}

DecodeQueue *decode_queue_create_parallel(size_t queue_depth, bool use_pinned_memory,
                                          int num_producers) {
  return decode_queue_create_internal(queue_depth, use_pinned_memory, num_producers);
}

void decode_queue_destroy(DecodeQueue *queue) {
  if (!queue) {
    return;
  }

  // Stop producer if running
  decode_queue_stop_producer(queue);

  // Free any remaining frames in slots
  for (size_t i = 0; i < queue->queue_depth; i++) {
    DecodeSlot *slot = &queue->slots[i];
    if (slot->image.frame) {
#ifdef UNPAPER_WITH_CUDA
      if (slot->image.uses_pinned_memory && slot->image.frame->data[0]) {
        // Free pinned buffer
        UnpaperCudaPinnedBuffer buf = {
          .ptr = slot->image.frame->data[0],
          .bytes = 0,  // Size not needed for free
          .is_pinned = true
        };
        // Clear frame data pointers before free
        slot->image.frame->data[0] = NULL;
        unpaper_cuda_pinned_free(&buf);
      }
#endif
      av_frame_free(&slot->image.frame);
    }
  }

  pthread_mutex_destroy(&queue->mutex);
  pthread_cond_destroy(&queue->not_full);
  pthread_cond_destroy(&queue->not_empty);

  free(queue->slots);
  free(queue);
}

bool decode_queue_start_producer(DecodeQueue *queue, BatchQueue *batch_queue,
                                 const Options *options) {
  if (!queue || !batch_queue || queue->producer_started) {
    return false;
  }

  queue->batch_queue = batch_queue;
  queue->options = options;
  atomic_store(&queue->producer_running, true);
  atomic_store(&queue->producers_done, 0);
  atomic_store(&queue->next_job_index, 0);

  // Start all producer threads
  int started = 0;
  for (int i = 0; i < queue->num_producers; i++) {
    if (pthread_create(&queue->producer_threads[i], NULL, producer_thread_fn, queue) != 0) {
      // Failed to create thread - continue with fewer
      break;
    }
    started++;
  }

  if (started == 0) {
    atomic_store(&queue->producer_running, false);
    return false;
  }

  // Update actual producer count if some failed to start
  queue->num_producers = started;
  queue->producer_started = true;
  return true;
}

void decode_queue_stop_producer(DecodeQueue *queue) {
  if (!queue || !queue->producer_started) {
    return;
  }

  // Signal all producers to stop
  atomic_store(&queue->producer_running, false);

  // Wake up any waiting producers
  pthread_mutex_lock(&queue->mutex);
  pthread_cond_broadcast(&queue->not_full);
  pthread_mutex_unlock(&queue->mutex);

  // Wait for all producer threads to finish
  for (int i = 0; i < queue->num_producers; i++) {
    pthread_join(queue->producer_threads[i], NULL);
  }
  queue->producer_started = false;
}

DecodedImage *decode_queue_get(DecodeQueue *queue, int job_index,
                               int input_index) {
  if (!queue) {
    return NULL;
  }

  while (1) {
    // Try to find a ready slot
    int slot_idx = find_ready_slot(queue, job_index, input_index);
    if (slot_idx >= 0) {
      return &queue->slots[slot_idx].image;
    }

    // Check if all producers are done (no more images coming)
    if (atomic_load(&queue->producers_done) >= queue->num_producers) {
      // One more check for the slot
      slot_idx = find_ready_slot(queue, job_index, input_index);
      if (slot_idx >= 0) {
        return &queue->slots[slot_idx].image;
      }
      return NULL;  // Image not available and won't be
    }

    // Wait for producer
    atomic_fetch_add(&queue->consumer_waits, 1);
    pthread_mutex_lock(&queue->mutex);
    pthread_cond_wait(&queue->not_empty, &queue->mutex);
    pthread_mutex_unlock(&queue->mutex);
  }
}

void decode_queue_release(DecodeQueue *queue, DecodedImage *image) {
  if (!queue || !image) {
    return;
  }

  // Find the slot for this image
  for (size_t i = 0; i < queue->queue_depth; i++) {
    if (&queue->slots[i].image == image) {
      DecodeSlot *slot = &queue->slots[i];

      // Free the frame
      if (slot->image.frame) {
#ifdef UNPAPER_WITH_CUDA
        if (slot->image.uses_pinned_memory && slot->image.frame->data[0]) {
          UnpaperCudaPinnedBuffer buf = {
            .ptr = slot->image.frame->data[0],
            .bytes = 0,
            .is_pinned = true
          };
          slot->image.frame->data[0] = NULL;
          unpaper_cuda_pinned_free(&buf);
        }
#endif
        av_frame_free(&slot->image.frame);
      }

      slot->image.frame = NULL;
      slot->image.valid = false;
      slot->image.uses_pinned_memory = false;

      // Update stats
      atomic_fetch_add(&queue->images_consumed, 1);
      atomic_fetch_sub(&queue->current_depth, 1);

      // Mark slot as empty
      atomic_store(&slot->state, SLOT_EMPTY);

      // Signal producer
      pthread_mutex_lock(&queue->mutex);
      pthread_cond_signal(&queue->not_full);
      pthread_mutex_unlock(&queue->mutex);

      return;
    }
  }
}

bool decode_queue_producer_done(DecodeQueue *queue) {
  if (!queue) {
    return true;
  }
  // All producers are done when the done counter reaches num_producers
  return atomic_load(&queue->producers_done) >= queue->num_producers;
}

DecodeQueueStats decode_queue_get_stats(const DecodeQueue *queue) {
  DecodeQueueStats stats = {0};
  if (!queue) {
    return stats;
  }

  stats.images_decoded = atomic_load(&queue->images_decoded);
  stats.images_consumed = atomic_load(&queue->images_consumed);
  stats.producer_waits = atomic_load(&queue->producer_waits);
  stats.consumer_waits = atomic_load(&queue->consumer_waits);
  stats.pinned_allocations = atomic_load(&queue->pinned_allocations);
  stats.peak_queue_depth = atomic_load(&queue->peak_depth);

  return stats;
}

void decode_queue_print_stats(const DecodeQueue *queue) {
  if (!queue) {
    return;
  }

  DecodeQueueStats stats = decode_queue_get_stats(queue);

  double producer_wait_rate = 0.0;
  double consumer_wait_rate = 0.0;
  if (stats.images_decoded > 0) {
    producer_wait_rate = 100.0 * (double)stats.producer_waits / (double)stats.images_decoded;
  }
  if (stats.images_consumed > 0) {
    consumer_wait_rate = 100.0 * (double)stats.consumer_waits / (double)stats.images_consumed;
  }

  fprintf(stderr,
          "Decode Queue Statistics:\n"
          "  Queue depth: %zu slots\n"
          "  Images decoded: %zu\n"
          "  Images consumed: %zu\n"
          "  Producer waits (queue full): %zu (%.1f%%)\n"
          "  Consumer waits (queue empty): %zu (%.1f%%)\n"
          "  Peak queue occupancy: %zu\n"
          "  Pinned memory allocations: %zu\n",
          queue->queue_depth, stats.images_decoded, stats.images_consumed,
          stats.producer_waits, producer_wait_rate,
          stats.consumer_waits, consumer_wait_rate,
          stats.peak_queue_depth, stats.pinned_allocations);
}
