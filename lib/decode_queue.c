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
#include <libavutil/pixfmt.h>
#include <pthread.h>
#include <stdatomic.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <strings.h> // For strcasecmp

#ifdef UNPAPER_WITH_CUDA
#include "imageprocess/cuda_runtime.h"
#include "imageprocess/nvimgcodec.h"
#include <cuda_runtime_api.h>
#endif

// Slot states
typedef enum {
  SLOT_EMPTY = 0, // Available for producer
  SLOT_DECODING,  // Producer is decoding into this slot
  SLOT_READY,     // Decoded and ready for consumer
  SLOT_IN_USE,    // Consumer is using this slot
} SlotState;

// Check if filename has JPEG extension
static bool is_jpeg_file(const char *filename) {
  if (filename == NULL) {
    return false;
  }
  const char *ext = strrchr(filename, '.');
  if (ext == NULL) {
    return false;
  }
  // Case-insensitive comparison
  if (strcasecmp(ext, ".jpg") == 0 || strcasecmp(ext, ".jpeg") == 0) {
    return true;
  }
  return false;
}

// Check if filename has JPEG2000 extension
static bool is_jp2_file(const char *filename) {
  if (filename == NULL) {
    return false;
  }
  const char *ext = strrchr(filename, '.');
  if (ext == NULL) {
    return false;
  }
  // Case-insensitive comparison for JP2/J2K/J2C extensions
  if (strcasecmp(ext, ".jp2") == 0 || strcasecmp(ext, ".j2k") == 0 ||
      strcasecmp(ext, ".j2c") == 0 || strcasecmp(ext, ".jpx") == 0) {
    return true;
  }
  return false;
}

// Check if file is GPU-decodable (JPEG or JP2)
static bool is_gpu_decodable_file(const char *filename) {
  return is_jpeg_file(filename) || is_jp2_file(filename);
}

#ifdef UNPAPER_WITH_CUDA
// Decode a JPEG or JP2 file directly to GPU memory using nvImageCodec
// Returns true if successful, fills in DecodedImage GPU fields
static bool decode_image_to_gpu(const char *filename, DecodedImage *out) {
  // Check if nvimgcodec is available
  if (!nvimgcodec_any_available()) {
    return false;
  }

  // Read the file into memory
  FILE *f = fopen(filename, "rb");
  if (f == NULL) {
    return false;
  }

  fseek(f, 0, SEEK_END);
  long file_size = ftell(f);
  fseek(f, 0, SEEK_SET);

  if (file_size <= 0 || file_size > (long)(100 * 1024 * 1024)) { // Max 100MB
    fclose(f);
    return false;
  }

  uint8_t *data = malloc((size_t)file_size);
  if (data == NULL) {
    fclose(f);
    return false;
  }

  size_t bytes_read = fread(data, 1, (size_t)file_size, f);
  fclose(f);

  if (bytes_read != (size_t)file_size) {
    free(data);
    return false;
  }

  // Get image info to determine output format
  NvImgCodecFormat format = NVIMGCODEC_FORMAT_UNKNOWN;
  int channels = 1;
  nvimgcodec_get_image_info(data, (size_t)file_size, &format, NULL, NULL,
                            &channels);

  // Check if format is supported (JP2 requires full nvImageCodec)
  if (format == NVIMGCODEC_FORMAT_JPEG2000 && !nvimgcodec_jp2_supported()) {
    free(data);
    return false; // JP2 not supported without nvImageCodec
  }

  // Acquire decode state
  NvImgCodecDecodeState *state = nvimgcodec_acquire_decode_state();
  if (state == NULL) {
    free(data);
    return false;
  }

  // Use grayscale for 1-channel, RGB for multi-channel
  NvImgCodecOutputFormat out_fmt =
      (channels == 1) ? NVIMGCODEC_OUT_GRAY8 : NVIMGCODEC_OUT_RGB;

  NvImgCodecDecodedImage nvout = {0};
  bool result =
      nvimgcodec_decode(data, (size_t)file_size, state, NULL, out_fmt, &nvout);

  nvimgcodec_release_decode_state(state);
  free(data);

  if (!result) {
    return false;
  }

  // Fill DecodedImage GPU fields
  out->on_gpu = true;
  out->gpu_ptr = nvout.gpu_ptr;
  out->gpu_pitch = nvout.pitch;
  out->gpu_width = nvout.width;
  out->gpu_height = nvout.height;
  out->gpu_channels = nvout.channels;
  // Map format to AVPixelFormat
  out->gpu_format = (nvout.channels == 1) ? AV_PIX_FMT_GRAY8 : AV_PIX_FMT_RGB24;
  out->frame = NULL; // No CPU frame needed
  // Store completion event for async decode (caller must sync before use)
  out->gpu_completion_event = nvout.completion_event;
  out->gpu_event_from_pool = nvout.event_from_pool;

  return true;
}
#endif // UNPAPER_WITH_CUDA

// A slot in the decode queue
typedef struct {
  DecodedImage image;
  atomic_int state;
} DecodeSlot;

// Maximum number of parallel producer threads
// Increased to support high core count CPUs (e.g., 24 cores)
#define MAX_PRODUCER_THREADS 24

struct DecodeQueue {
  // Slot array (circular buffer)
  DecodeSlot *slots;
  size_t queue_depth;

  // Producer state - supports multiple threads
  pthread_t producer_threads[MAX_PRODUCER_THREADS];
  int num_producers;
  bool producer_started;
  atomic_bool producer_running;
  atomic_int producers_done; // Counter of completed producers

  // Shared job counter for work distribution among producers
  atomic_size_t next_job_index;

  // Source data for producer
  BatchQueue *batch_queue;
  const Options *options;

  // Configuration
  bool use_pinned_memory;
  bool use_gpu_decode; // Enable nvJPEG GPU decode for JPEG files

  // Synchronization
  pthread_mutex_t mutex;
  pthread_cond_t not_full;  // Signal when slot becomes empty
  pthread_cond_t not_empty; // Signal when slot becomes ready

  // Statistics (atomic for thread safety)
  atomic_size_t images_decoded;
  atomic_size_t images_consumed;
  atomic_size_t producer_waits;
  atomic_size_t consumer_waits;
  atomic_size_t pinned_allocations;
  atomic_size_t current_depth;
  atomic_size_t peak_depth;
  // GPU decode stats (PR36+)
  atomic_size_t gpu_decodes;
  atomic_size_t cpu_decodes;
  atomic_size_t gpu_decode_failures;
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

  AVFrame *dst =
      alloc_frame_pinned(src->width, src->height, src->format, true, is_pinned);
  if (!dst) {
    return NULL;
  }

  // Copy image data
  av_image_copy(dst->data, dst->linesize, (const uint8_t **)src->data,
                src->linesize, src->format, src->width, src->height);

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

#ifdef UNPAPER_WITH_CUDA
  // Initialize CUDA in this thread if GPU decode is enabled
  if (queue->use_gpu_decode) {
    // Ensure CUDA context is initialized in this thread
    unpaper_cuda_try_init();
  }
#endif

  // Work stealing loop - each thread grabs jobs atomically
  while (atomic_load(&queue->producer_running)) {
    // Atomically get next job index
    size_t job_idx = atomic_fetch_add(&queue->next_job_index, 1);
    if (job_idx >= batch->count) {
      break; // No more jobs
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
        continue; // Skip blank pages
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
      bool decode_success = false;

      // Initialize GPU fields
      slot->image.on_gpu = false;
      slot->image.gpu_ptr = NULL;
      slot->image.gpu_pitch = 0;
      slot->image.gpu_width = 0;
      slot->image.gpu_height = 0;
      slot->image.gpu_channels = 0;
      slot->image.gpu_format = 0;
      slot->image.gpu_completion_event = NULL;
      slot->image.gpu_event_from_pool = false;

#ifdef UNPAPER_WITH_CUDA
      // Try GPU decode for JPEG/JP2 files using nvImageCodec
      // Each decode state has its own dedicated CUDA stream for true
      // parallelism. This avoids the threading issues with shared streams.
      if (queue->use_gpu_decode && is_gpu_decodable_file(filename)) {
        decode_success = decode_image_to_gpu(filename, &slot->image);
        if (decode_success) {
          slot->image.job_index = (int)job_idx;
          slot->image.input_index = input_idx;
          slot->image.valid = true;
          slot->image.uses_pinned_memory = false;
          atomic_fetch_add(&queue->gpu_decodes, 1);
        } else {
          // GPU decode failed - fall back to CPU
          atomic_fetch_add(&queue->gpu_decode_failures, 1);
        }
      }
#endif

      // CPU decode path (fallback or non-JPEG)
      if (!decode_success) {
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
          slot->image.on_gpu = false;
          atomic_fetch_add(&queue->cpu_decodes, 1);
          decode_success = true;
        } else {
          // Decode failed - mark as invalid
          slot->image.frame = NULL;
          slot->image.job_index = (int)job_idx;
          slot->image.input_index = input_idx;
          slot->image.valid = false;
          slot->image.uses_pinned_memory = false;
          slot->image.on_gpu = false;
        }
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
  queue->use_gpu_decode = false; // Disabled by default
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
  // GPU decode stats
  atomic_init(&queue->gpu_decodes, 0);
  atomic_init(&queue->cpu_decodes, 0);
  atomic_init(&queue->gpu_decode_failures, 0);

  // Initialize slots
  for (size_t i = 0; i < queue_depth; i++) {
    atomic_init(&queue->slots[i].state, SLOT_EMPTY);
    queue->slots[i].image.frame = NULL;
    queue->slots[i].image.valid = false;
    queue->slots[i].image.on_gpu = false;
    queue->slots[i].image.gpu_ptr = NULL;
  }

  return queue;
}

DecodeQueue *decode_queue_create(size_t queue_depth, bool use_pinned_memory) {
  return decode_queue_create_internal(queue_depth, use_pinned_memory, 1);
}

DecodeQueue *decode_queue_create_parallel(size_t queue_depth,
                                          bool use_pinned_memory,
                                          int num_producers) {
  return decode_queue_create_internal(queue_depth, use_pinned_memory,
                                      num_producers);
}

void decode_queue_enable_gpu_decode(DecodeQueue *queue, bool enable) {
  if (!queue) {
    return;
  }
#ifdef UNPAPER_WITH_CUDA
  queue->use_gpu_decode = enable;
#else
  (void)enable;
#endif
}

void decode_queue_destroy(DecodeQueue *queue) {
  if (!queue) {
    return;
  }

  // Stop producer if running
  decode_queue_stop_producer(queue);

  // Free any remaining frames/GPU memory in slots
  for (size_t i = 0; i < queue->queue_depth; i++) {
    DecodeSlot *slot = &queue->slots[i];
#ifdef UNPAPER_WITH_CUDA
    // Clean up completion event if not already synced
    if (slot->image.gpu_completion_event != NULL) {
      nvimgcodec_release_completion_event(slot->image.gpu_completion_event,
                                      slot->image.gpu_event_from_pool);
      slot->image.gpu_completion_event = NULL;
      slot->image.gpu_event_from_pool = false;
    }
    // Free GPU memory if image was decoded to GPU
    if (slot->image.on_gpu && slot->image.gpu_ptr != NULL) {
      cudaFree(slot->image.gpu_ptr);
      slot->image.gpu_ptr = NULL;
      slot->image.on_gpu = false;
    }
#endif
    if (slot->image.frame) {
#ifdef UNPAPER_WITH_CUDA
      if (slot->image.uses_pinned_memory && slot->image.frame->data[0]) {
        // Free pinned buffer
        UnpaperCudaPinnedBuffer buf = {.ptr = slot->image.frame->data[0],
                                       .bytes = 0, // Size not needed for free
                                       .is_pinned = true};
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
    if (pthread_create(&queue->producer_threads[i], NULL, producer_thread_fn,
                       queue) != 0) {
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
      return NULL; // Image not available and won't be
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

#ifdef UNPAPER_WITH_CUDA
      // Clean up completion event if not already synced
      if (slot->image.gpu_completion_event != NULL) {
        nvimgcodec_release_completion_event(slot->image.gpu_completion_event,
                                        slot->image.gpu_event_from_pool);
        slot->image.gpu_completion_event = NULL;
        slot->image.gpu_event_from_pool = false;
      }
      // Free GPU memory if image was decoded to GPU
      if (slot->image.on_gpu && slot->image.gpu_ptr != NULL) {
        cudaFree(slot->image.gpu_ptr);
        slot->image.gpu_ptr = NULL;
        slot->image.on_gpu = false;
      }
#endif

      // Free the CPU frame
      if (slot->image.frame) {
#ifdef UNPAPER_WITH_CUDA
        if (slot->image.uses_pinned_memory && slot->image.frame->data[0]) {
          UnpaperCudaPinnedBuffer buf = {
              .ptr = slot->image.frame->data[0], .bytes = 0, .is_pinned = true};
          slot->image.frame->data[0] = NULL;
          unpaper_cuda_pinned_free(&buf);
        }
#endif
        av_frame_free(&slot->image.frame);
      }

      slot->image.frame = NULL;
      slot->image.valid = false;
      slot->image.uses_pinned_memory = false;
      slot->image.gpu_ptr = NULL;
      slot->image.on_gpu = false;

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

// Wait for GPU decode to complete (synchronize on completion event).
// Must be called before accessing gpu_ptr from a different CUDA stream.
// Safe to call even if image was not GPU-decoded.
void decoded_image_wait_gpu_complete(DecodedImage *image) {
  if (image == NULL) {
    return;
  }

#ifdef UNPAPER_WITH_CUDA
  if (image->gpu_completion_event != NULL) {
    cudaEvent_t event = (cudaEvent_t)image->gpu_completion_event;
    cudaError_t err = cudaEventSynchronize(event);
    if (err != cudaSuccess) {
      verboseLog(VERBOSE_DEBUG, "decoded_image: event sync failed: %s\n",
                 cudaGetErrorString(err));
    }
    // Release event back to pool or destroy
    nvimgcodec_release_completion_event(image->gpu_completion_event,
                                    image->gpu_event_from_pool);
    image->gpu_completion_event = NULL;
    image->gpu_event_from_pool = false;
  }
#endif
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
  // GPU decode stats
  stats.gpu_decodes = atomic_load(&queue->gpu_decodes);
  stats.cpu_decodes = atomic_load(&queue->cpu_decodes);
  stats.gpu_decode_failures = atomic_load(&queue->gpu_decode_failures);

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
    producer_wait_rate =
        100.0 * (double)stats.producer_waits / (double)stats.images_decoded;
  }
  if (stats.images_consumed > 0) {
    consumer_wait_rate =
        100.0 * (double)stats.consumer_waits / (double)stats.images_consumed;
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
          stats.producer_waits, producer_wait_rate, stats.consumer_waits,
          consumer_wait_rate, stats.peak_queue_depth, stats.pinned_allocations);

  // Print GPU decode stats if any GPU decodes occurred
  if (stats.gpu_decodes > 0 || stats.gpu_decode_failures > 0) {
    double gpu_decode_rate = 0.0;
    if (stats.gpu_decodes + stats.cpu_decodes > 0) {
      gpu_decode_rate = 100.0 * (double)stats.gpu_decodes /
                        (double)(stats.gpu_decodes + stats.cpu_decodes);
    }
    fprintf(stderr,
            "  GPU decode stats:\n"
            "    GPU decodes (nvJPEG): %zu (%.1f%%)\n"
            "    CPU decodes (FFmpeg): %zu\n"
            "    GPU decode failures: %zu\n",
            stats.gpu_decodes, gpu_decode_rate, stats.cpu_decodes,
            stats.gpu_decode_failures);
  }
}
