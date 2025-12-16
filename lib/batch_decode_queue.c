// SPDX-FileCopyrightText: 2025 The unpaper authors
//
// SPDX-License-Identifier: GPL-2.0-only

#include "lib/batch_decode_queue.h"
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
#include <strings.h>
#include <sys/time.h>

#ifdef UNPAPER_WITH_CUDA
#include "imageprocess/cuda_runtime.h"
#include "imageprocess/nvjpeg_decode.h"
#include <cuda_runtime_api.h>
#endif

// ============================================================================
// Internal Types
// ============================================================================

// Slot states for lock-free queue
typedef enum {
  SLOT_EMPTY = 0, // Available for producer
  SLOT_DECODING,  // Producer is processing
  SLOT_READY,     // Ready for consumer
  SLOT_IN_USE,    // Consumer is using
} SlotState;

// A slot in the output queue
typedef struct {
  BatchDecodedImage image;
  atomic_int state;
} DecodeSlot;

// File data collected for batch decode
typedef struct {
  uint8_t *data;     // JPEG file data (malloc'd)
  size_t size;       // Data size in bytes
  int job_index;     // Job index
  int input_index;   // Input index within job
  const char *path;  // File path (borrowed)
  bool is_jpeg;      // True if JPEG file
  bool read_success; // True if file read succeeded
} CollectedFile;

// Maximum I/O threads
#define MAX_IO_THREADS 16

// Maximum batch size for decode - processes all images at once
// to avoid pool buffer reuse issues with chunking
#define MAX_DECODE_CHUNK_SIZE 256

// Maximum JPEG file size to buffer (100MB)
#define MAX_JPEG_FILE_SIZE (100 * 1024 * 1024)

struct BatchDecodeQueue {
  // Output slot array
  DecodeSlot *slots;
  size_t queue_depth;

  // I/O thread pool
  pthread_t io_threads[MAX_IO_THREADS];
  int num_io_threads;

  // Orchestration thread
  pthread_t orchestrator_thread;
  bool orchestrator_started;

  // Shutdown control
  atomic_bool running;
  atomic_int io_threads_done;

  // Source data
  BatchQueue *batch_queue;
  const Options *options;

  // Configuration
  bool use_gpu_decode;
  int max_width;
  int max_height;
  int chunk_size; // Number of images per batch decode call (0 = default)

  // I/O work distribution (lock-free)
  // Each entry represents a (job_index, input_index) pair to read
  atomic_size_t io_work_next;
  size_t io_work_total;

  // Collected file data for current chunk
  CollectedFile *collected_files;
  size_t collected_capacity;
  pthread_mutex_t collected_mutex;
  pthread_cond_t collected_cond;
  atomic_size_t collected_count;
  atomic_size_t collected_target;

  // Synchronization for output slots
  pthread_mutex_t slot_mutex;
  pthread_cond_t slot_not_full;
  pthread_cond_t slot_not_empty;

  // Statistics
  atomic_size_t total_images;
  atomic_size_t gpu_batched_decodes;
  atomic_size_t gpu_single_decodes;
  atomic_size_t cpu_decodes;
  atomic_size_t decode_failures;
  atomic_size_t batch_calls;
  atomic_size_t chunks_processed;
  atomic_size_t current_depth;
  atomic_size_t peak_depth;
  atomic_size_t io_time_us;
  atomic_size_t decode_time_us;
};

// ============================================================================
// Utility Functions
// ============================================================================

static inline double get_time_ms(void) {
  struct timeval tv;
  gettimeofday(&tv, NULL);
  return (double)tv.tv_sec * 1000.0 + (double)tv.tv_usec / 1000.0;
}

// Check if filename has JPEG extension
static bool is_jpeg_file(const char *filename) {
  if (filename == NULL) {
    return false;
  }
  const char *ext = strrchr(filename, '.');
  if (ext == NULL) {
    return false;
  }
  return (strcasecmp(ext, ".jpg") == 0 || strcasecmp(ext, ".jpeg") == 0);
}

// Get effective chunk size (configured or default)
static inline int get_effective_chunk_size(const BatchDecodeQueue *queue) {
  if (queue->chunk_size > 0) {
    return queue->chunk_size;
  }
  return BATCH_DECODE_CHUNK_SIZE;
}

// Update peak depth atomically
static void update_peak_depth(BatchDecodeQueue *queue) {
  size_t depth = atomic_load(&queue->current_depth);
  size_t peak = atomic_load(&queue->peak_depth);
  while (depth > peak) {
    if (atomic_compare_exchange_weak(&queue->peak_depth, &peak, depth)) {
      break;
    }
  }
}

// Find an empty slot
static int find_empty_slot(BatchDecodeQueue *queue) {
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
static int find_ready_slot(BatchDecodeQueue *queue, int job_index,
                           int input_index) {
  for (size_t i = 0; i < queue->queue_depth; i++) {
    if (atomic_load(&queue->slots[i].state) == SLOT_READY) {
      BatchDecodedImage *img = &queue->slots[i].image;
      if (img->job_index == job_index && img->input_index == input_index) {
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

// ============================================================================
// CPU Decode (FFmpeg) for non-JPEG files
// ============================================================================

static AVFrame *decode_image_file_ffmpeg(const char *filename) {
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

// ============================================================================
// I/O Thread - Parallel File Reading
// ============================================================================

// Work item for I/O threads: read a single file
typedef struct {
  int job_index;
  int input_index;
  const char *path;
} IOWorkItem;

// Get the total number of input files across all jobs
static size_t count_total_inputs(BatchQueue *batch) {
  size_t total = 0;
  for (size_t i = 0; i < batch->count; i++) {
    BatchJob *job = batch_queue_get(batch, i);
    for (int j = 0; j < job->input_count; j++) {
      if (job->input_files[j] != NULL) {
        total++;
      }
    }
  }
  return total;
}

// Convert linear index to (job_index, input_index) for work distribution
static bool get_work_item(BatchQueue *batch, size_t linear_idx,
                          IOWorkItem *out) {
  size_t current = 0;
  for (size_t i = 0; i < batch->count; i++) {
    BatchJob *job = batch_queue_get(batch, i);
    for (int j = 0; j < job->input_count; j++) {
      if (job->input_files[j] != NULL) {
        if (current == linear_idx) {
          out->job_index = (int)i;
          out->input_index = j;
          out->path = job->input_files[j];
          return true;
        }
        current++;
      }
    }
  }
  return false;
}

// I/O thread function: reads files and stores data in collected_files array
static void *io_thread_fn(void *arg) {
  BatchDecodeQueue *queue = (BatchDecodeQueue *)arg;

#ifdef UNPAPER_WITH_CUDA
  // Initialize CUDA in this thread (for pinned memory if needed later)
  if (queue->use_gpu_decode) {
    unpaper_cuda_try_init();
  }
#endif

  while (atomic_load(&queue->running)) {
    // Get next work item atomically
    size_t idx = atomic_fetch_add(&queue->io_work_next, 1);
    if (idx >= queue->io_work_total) {
      break; // No more work
    }

    IOWorkItem item;
    if (!get_work_item(queue->batch_queue, idx, &item)) {
      continue;
    }

    double io_start = get_time_ms();

    // Read the file
    FILE *f = fopen(item.path, "rb");
    if (f == NULL) {
      // Record failure
      pthread_mutex_lock(&queue->collected_mutex);
      size_t slot = atomic_fetch_add(&queue->collected_count, 1);
      if (slot < queue->collected_capacity) {
        queue->collected_files[slot].data = NULL;
        queue->collected_files[slot].size = 0;
        queue->collected_files[slot].job_index = item.job_index;
        queue->collected_files[slot].input_index = item.input_index;
        queue->collected_files[slot].path = item.path;
        queue->collected_files[slot].is_jpeg = is_jpeg_file(item.path);
        queue->collected_files[slot].read_success = false;
      }
      if (atomic_load(&queue->collected_count) >=
          atomic_load(&queue->collected_target)) {
        pthread_cond_signal(&queue->collected_cond);
      }
      pthread_mutex_unlock(&queue->collected_mutex);
      continue;
    }

    // Get file size
    fseek(f, 0, SEEK_END);
    long file_size = ftell(f);
    fseek(f, 0, SEEK_SET);

    if (file_size <= 0 || file_size > MAX_JPEG_FILE_SIZE) {
      fclose(f);
      pthread_mutex_lock(&queue->collected_mutex);
      size_t slot = atomic_fetch_add(&queue->collected_count, 1);
      if (slot < queue->collected_capacity) {
        queue->collected_files[slot].data = NULL;
        queue->collected_files[slot].size = 0;
        queue->collected_files[slot].job_index = item.job_index;
        queue->collected_files[slot].input_index = item.input_index;
        queue->collected_files[slot].path = item.path;
        queue->collected_files[slot].is_jpeg = is_jpeg_file(item.path);
        queue->collected_files[slot].read_success = false;
      }
      if (atomic_load(&queue->collected_count) >=
          atomic_load(&queue->collected_target)) {
        pthread_cond_signal(&queue->collected_cond);
      }
      pthread_mutex_unlock(&queue->collected_mutex);
      continue;
    }

    // Allocate buffer and read file
    uint8_t *data = malloc((size_t)file_size);
    if (data == NULL) {
      fclose(f);
      pthread_mutex_lock(&queue->collected_mutex);
      size_t slot = atomic_fetch_add(&queue->collected_count, 1);
      if (slot < queue->collected_capacity) {
        queue->collected_files[slot].data = NULL;
        queue->collected_files[slot].size = 0;
        queue->collected_files[slot].job_index = item.job_index;
        queue->collected_files[slot].input_index = item.input_index;
        queue->collected_files[slot].path = item.path;
        queue->collected_files[slot].is_jpeg = is_jpeg_file(item.path);
        queue->collected_files[slot].read_success = false;
      }
      if (atomic_load(&queue->collected_count) >=
          atomic_load(&queue->collected_target)) {
        pthread_cond_signal(&queue->collected_cond);
      }
      pthread_mutex_unlock(&queue->collected_mutex);
      continue;
    }

    size_t bytes_read = fread(data, 1, (size_t)file_size, f);
    fclose(f);

    double io_end = get_time_ms();
    atomic_fetch_add(&queue->io_time_us, (size_t)((io_end - io_start) * 1000));

    // Store collected file data
    pthread_mutex_lock(&queue->collected_mutex);
    size_t slot = atomic_fetch_add(&queue->collected_count, 1);
    if (slot < queue->collected_capacity) {
      if (bytes_read == (size_t)file_size) {
        queue->collected_files[slot].data = data;
        queue->collected_files[slot].size = (size_t)file_size;
        queue->collected_files[slot].read_success = true;
      } else {
        free(data);
        queue->collected_files[slot].data = NULL;
        queue->collected_files[slot].size = 0;
        queue->collected_files[slot].read_success = false;
      }
      queue->collected_files[slot].job_index = item.job_index;
      queue->collected_files[slot].input_index = item.input_index;
      queue->collected_files[slot].path = item.path;
      queue->collected_files[slot].is_jpeg = is_jpeg_file(item.path);
    } else {
      free(data);
    }

    if (atomic_load(&queue->collected_count) >=
        atomic_load(&queue->collected_target)) {
      pthread_cond_signal(&queue->collected_cond);
    }
    pthread_mutex_unlock(&queue->collected_mutex);
  }

  atomic_fetch_add(&queue->io_threads_done, 1);
  return NULL;
}

// ============================================================================
// Orchestrator Thread - Batch Decode Coordination
// ============================================================================

// Place a decoded image into an output slot
static bool place_decoded_image(BatchDecodeQueue *queue, int job_index,
                                int input_index, BatchDecodedImage *img) {
  while (atomic_load(&queue->running)) {
    int slot_idx = find_empty_slot(queue);
    if (slot_idx >= 0) {
      DecodeSlot *slot = &queue->slots[slot_idx];
      slot->image = *img;
      slot->image.job_index = job_index;
      slot->image.input_index = input_index;
      atomic_store(&slot->state, SLOT_READY);

      size_t depth = atomic_fetch_add(&queue->current_depth, 1) + 1;
      (void)depth;
      update_peak_depth(queue);

      // Signal consumers
      pthread_mutex_lock(&queue->slot_mutex);
      pthread_cond_broadcast(&queue->slot_not_empty);
      pthread_mutex_unlock(&queue->slot_mutex);

      return true;
    }

    // No empty slot - wait
    pthread_mutex_lock(&queue->slot_mutex);
    pthread_cond_wait(&queue->slot_not_full, &queue->slot_mutex);
    pthread_mutex_unlock(&queue->slot_mutex);
  }
  return false;
}

// Process a chunk of collected files
static void process_chunk(BatchDecodeQueue *queue, CollectedFile *files,
                          size_t count) {
  if (count == 0) {
    return;
  }

#ifdef UNPAPER_WITH_CUDA
  // Separate JPEG files (for GPU batch decode) from non-JPEG files (CPU decode)
  CollectedFile *jpeg_files[MAX_DECODE_CHUNK_SIZE];
  size_t jpeg_count = 0;
  CollectedFile *other_files[MAX_DECODE_CHUNK_SIZE];
  size_t other_count = 0;

  for (size_t i = 0; i < count; i++) {
    if (!files[i].read_success) {
      // Failed to read - mark as decode failure
      BatchDecodedImage img = {0};
      img.valid = false;
      place_decoded_image(queue, files[i].job_index, files[i].input_index,
                          &img);
      atomic_fetch_add(&queue->decode_failures, 1);
      atomic_fetch_add(&queue->total_images, 1);
      continue;
    }

    if (queue->use_gpu_decode && files[i].is_jpeg && files[i].data != NULL) {
      jpeg_files[jpeg_count++] = &files[i];
    } else {
      other_files[other_count++] = &files[i];
    }
  }

  // Batch decode JPEG files using nvJPEG
  if (jpeg_count > 0 && nvjpeg_batched_is_ready()) {
    double decode_start = get_time_ms();

    // Prepare arrays for batch decode
    const uint8_t *jpeg_data[MAX_DECODE_CHUNK_SIZE];
    size_t jpeg_sizes[MAX_DECODE_CHUNK_SIZE];
    NvJpegDecodedImage outputs[MAX_DECODE_CHUNK_SIZE];

    for (size_t i = 0; i < jpeg_count; i++) {
      jpeg_data[i] = jpeg_files[i]->data;
      jpeg_sizes[i] = jpeg_files[i]->size;
    }

    // Single batch decode call - key performance optimization!
    int decoded =
        nvjpeg_decode_batch(jpeg_data, jpeg_sizes, (int)jpeg_count, outputs);

    double decode_end = get_time_ms();
    atomic_fetch_add(&queue->decode_time_us,
                     (size_t)((decode_end - decode_start) * 1000));
    atomic_fetch_add(&queue->batch_calls, 1);

    // Place decoded images into output slots
    for (size_t i = 0; i < jpeg_count; i++) {
      BatchDecodedImage img = {0};

      if (outputs[i].gpu_ptr != NULL) {
        // Successfully decoded to GPU (pool buffer - don't free)
        img.valid = true;
        img.on_gpu = true;
        img.gpu_ptr = outputs[i].gpu_ptr;
        img.gpu_pitch = outputs[i].pitch;
        img.gpu_width = outputs[i].width;
        img.gpu_height = outputs[i].height;
        img.gpu_channels = outputs[i].channels;
        img.gpu_format =
            (outputs[i].channels == 1) ? AV_PIX_FMT_GRAY8 : AV_PIX_FMT_RGB24;
        img.frame = NULL;
        img.gpu_pool_owned = true; // Pool buffer - managed by nvjpeg_batched
        atomic_fetch_add(&queue->gpu_batched_decodes, 1);
      } else {
        // Batch decode failed for this image - try single decode fallback
        NvJpegStreamState *state = nvjpeg_acquire_stream_state();
        if (state != NULL) {
          NvJpegDecodedImage nvout = {0};
          int channels = 1;
          nvjpeg_get_image_info(jpeg_data[i], jpeg_sizes[i], NULL, NULL,
                                &channels);
          NvJpegOutputFormat fmt =
              (channels == 1) ? NVJPEG_FMT_GRAY8 : NVJPEG_FMT_RGB;

          if (nvjpeg_decode_to_gpu(jpeg_data[i], jpeg_sizes[i], state, NULL,
                                   fmt, &nvout)) {
            img.valid = true;
            img.on_gpu = true;
            img.gpu_ptr = nvout.gpu_ptr;
            img.gpu_pitch = nvout.pitch;
            img.gpu_width = nvout.width;
            img.gpu_height = nvout.height;
            img.gpu_channels = nvout.channels;
            img.gpu_format =
                (nvout.channels == 1) ? AV_PIX_FMT_GRAY8 : AV_PIX_FMT_RGB24;
            img.frame = NULL;
            img.gpu_pool_owned = false; // Individual allocation - must free
            atomic_fetch_add(&queue->gpu_single_decodes, 1);
          } else {
            // GPU decode completely failed - fall back to CPU
            AVFrame *frame = decode_image_file_ffmpeg(jpeg_files[i]->path);
            if (frame) {
              img.valid = true;
              img.on_gpu = false;
              img.frame = frame;
              atomic_fetch_add(&queue->cpu_decodes, 1);
            } else {
              img.valid = false;
              atomic_fetch_add(&queue->decode_failures, 1);
            }
          }
          nvjpeg_release_stream_state(state);
        } else {
          // No stream state available - CPU fallback
          AVFrame *frame = decode_image_file_ffmpeg(jpeg_files[i]->path);
          if (frame) {
            img.valid = true;
            img.on_gpu = false;
            img.frame = frame;
            atomic_fetch_add(&queue->cpu_decodes, 1);
          } else {
            img.valid = false;
            atomic_fetch_add(&queue->decode_failures, 1);
          }
        }
      }

      place_decoded_image(queue, jpeg_files[i]->job_index,
                          jpeg_files[i]->input_index, &img);
      atomic_fetch_add(&queue->total_images, 1);

      // Free the JPEG data buffer
      free(jpeg_files[i]->data);
      jpeg_files[i]->data = NULL;
    }
  } else if (jpeg_count > 0) {
    // GPU decode not available - decode via CPU
    for (size_t i = 0; i < jpeg_count; i++) {
      other_files[other_count++] = jpeg_files[i];
    }
    jpeg_count = 0;
  }

  // Process non-JPEG files via CPU decode
  for (size_t i = 0; i < other_count; i++) {
    BatchDecodedImage img = {0};

    AVFrame *frame = decode_image_file_ffmpeg(other_files[i]->path);
    if (frame) {
      img.valid = true;
      img.on_gpu = false;
      img.frame = frame;
      atomic_fetch_add(&queue->cpu_decodes, 1);
    } else {
      img.valid = false;
      atomic_fetch_add(&queue->decode_failures, 1);
    }

    place_decoded_image(queue, other_files[i]->job_index,
                        other_files[i]->input_index, &img);
    atomic_fetch_add(&queue->total_images, 1);

    // Free the file data buffer
    if (other_files[i]->data != NULL) {
      free(other_files[i]->data);
      other_files[i]->data = NULL;
    }
  }

#else
  // Non-CUDA build: all files decoded via CPU
  for (size_t i = 0; i < count; i++) {
    BatchDecodedImage img = {0};

    if (!files[i].read_success) {
      img.valid = false;
      atomic_fetch_add(&queue->decode_failures, 1);
    } else {
      AVFrame *frame = decode_image_file_ffmpeg(files[i].path);
      if (frame) {
        img.valid = true;
        img.on_gpu = false;
        img.frame = frame;
        atomic_fetch_add(&queue->cpu_decodes, 1);
      } else {
        img.valid = false;
        atomic_fetch_add(&queue->decode_failures, 1);
      }
      // Free the file data buffer
      if (files[i].data != NULL) {
        free(files[i].data);
        files[i].data = NULL;
      }
    }

    place_decoded_image(queue, files[i].job_index, files[i].input_index, &img);
    atomic_fetch_add(&queue->total_images, 1);
  }
#endif

  atomic_fetch_add(&queue->chunks_processed, 1);
}

// Orchestrator thread: coordinates I/O and batch decode phases
static void *orchestrator_thread_fn(void *arg) {
  BatchDecodeQueue *queue = (BatchDecodeQueue *)arg;

#ifdef UNPAPER_WITH_CUDA
  // Initialize CUDA in this thread (but NOT batched decoder yet -
  // we need to know the total image count first)
  if (queue->use_gpu_decode) {
    unpaper_cuda_try_init();
  }
#endif

  // Count total work items
  queue->io_work_total = count_total_inputs(queue->batch_queue);
  if (queue->io_work_total == 0) {
    return NULL;
  }

#ifdef UNPAPER_WITH_CUDA
  // Now initialize batched decoder with the ACTUAL image count.
  // This ensures we can decode ALL images in one batch call, avoiding
  // pool buffer reuse issues that occur with chunking.
  if (queue->use_gpu_decode) {
    int batch_size = (int)queue->io_work_total;
    // Cap at reasonable limit to avoid excessive GPU memory usage
    // (each buffer is max_width * max_height * 3 bytes)
    if (batch_size > 256) {
      batch_size = 256;
    }
    if (!nvjpeg_batched_init(batch_size, queue->max_width, queue->max_height,
                             NVJPEG_FMT_RGB)) {
      verboseLog(VERBOSE_DEBUG, "batch_decode: nvjpeg_batched_init failed, "
                                "using single-image decode\n");
    }
  }
#endif

  // Allocate collected files buffer (sized for chunk processing)
  queue->collected_capacity = queue->io_work_total;
  queue->collected_files =
      calloc(queue->collected_capacity, sizeof(CollectedFile));
  if (queue->collected_files == NULL) {
    return NULL;
  }

  // Initialize work distribution
  atomic_store(&queue->io_work_next, 0);
  atomic_store(&queue->collected_count, 0);
  atomic_store(&queue->collected_target, queue->io_work_total);
  atomic_store(&queue->io_threads_done, 0);

  // Start I/O threads
  int started = 0;
  for (int i = 0; i < queue->num_io_threads; i++) {
    if (pthread_create(&queue->io_threads[i], NULL, io_thread_fn, queue) == 0) {
      started++;
    }
  }

  if (started == 0) {
    free(queue->collected_files);
    queue->collected_files = NULL;
    return NULL;
  }
  queue->num_io_threads = started;

  // Wait for all files to be collected
  pthread_mutex_lock(&queue->collected_mutex);
  while (atomic_load(&queue->collected_count) <
             atomic_load(&queue->collected_target) &&
         atomic_load(&queue->running)) {
    pthread_cond_wait(&queue->collected_cond, &queue->collected_mutex);
  }
  pthread_mutex_unlock(&queue->collected_mutex);

  // Wait for I/O threads to finish
  for (int i = 0; i < queue->num_io_threads; i++) {
    pthread_join(queue->io_threads[i], NULL);
  }

  // Process all collected files in one batch (or chunks if >256 images)
  // With batched decode initialized to match image count, we can decode
  // all images at once - no pool buffer reuse issues.
  size_t total_collected = atomic_load(&queue->collected_count);
  size_t max_chunk = 256; // Match the cap in nvjpeg_batched_init
  for (size_t offset = 0;
       offset < total_collected && atomic_load(&queue->running);
       offset += max_chunk) {
    size_t chunk_size = total_collected - offset;
    if (chunk_size > max_chunk) {
      chunk_size = max_chunk;
    }
    process_chunk(queue, &queue->collected_files[offset], chunk_size);
  }

  // Cleanup
  free(queue->collected_files);
  queue->collected_files = NULL;

  // Signal completion to any waiting consumers
  pthread_mutex_lock(&queue->slot_mutex);
  pthread_cond_broadcast(&queue->slot_not_empty);
  pthread_mutex_unlock(&queue->slot_mutex);

  return NULL;
}

// ============================================================================
// Public API Implementation
// ============================================================================

BatchDecodeQueue *batch_decode_queue_create(size_t queue_depth,
                                            int num_io_threads, int max_width,
                                            int max_height) {
  if (queue_depth == 0 || num_io_threads < 1) {
    return NULL;
  }
  if (num_io_threads > MAX_IO_THREADS) {
    num_io_threads = MAX_IO_THREADS;
  }

  BatchDecodeQueue *queue = calloc(1, sizeof(BatchDecodeQueue));
  if (!queue) {
    return NULL;
  }

  queue->slots = calloc(queue_depth, sizeof(DecodeSlot));
  if (!queue->slots) {
    free(queue);
    return NULL;
  }

  queue->queue_depth = queue_depth;
  queue->num_io_threads = num_io_threads;
  queue->max_width = max_width;
  queue->max_height = max_height;
  queue->use_gpu_decode = false;
  queue->chunk_size = 0; // 0 = use BATCH_DECODE_CHUNK_SIZE default
  queue->orchestrator_started = false;

  atomic_init(&queue->running, false);
  atomic_init(&queue->io_threads_done, 0);
  atomic_init(&queue->io_work_next, 0);
  atomic_init(&queue->collected_count, 0);
  atomic_init(&queue->collected_target, 0);

  pthread_mutex_init(&queue->collected_mutex, NULL);
  pthread_cond_init(&queue->collected_cond, NULL);
  pthread_mutex_init(&queue->slot_mutex, NULL);
  pthread_cond_init(&queue->slot_not_full, NULL);
  pthread_cond_init(&queue->slot_not_empty, NULL);

  // Initialize statistics
  atomic_init(&queue->total_images, 0);
  atomic_init(&queue->gpu_batched_decodes, 0);
  atomic_init(&queue->gpu_single_decodes, 0);
  atomic_init(&queue->cpu_decodes, 0);
  atomic_init(&queue->decode_failures, 0);
  atomic_init(&queue->batch_calls, 0);
  atomic_init(&queue->chunks_processed, 0);
  atomic_init(&queue->current_depth, 0);
  atomic_init(&queue->peak_depth, 0);
  atomic_init(&queue->io_time_us, 0);
  atomic_init(&queue->decode_time_us, 0);

  // Initialize slots
  for (size_t i = 0; i < queue_depth; i++) {
    atomic_init(&queue->slots[i].state, SLOT_EMPTY);
    memset(&queue->slots[i].image, 0, sizeof(BatchDecodedImage));
  }

  return queue;
}

void batch_decode_queue_destroy(BatchDecodeQueue *queue) {
  if (!queue) {
    return;
  }

  // Stop if still running
  batch_decode_queue_stop(queue);

  // Free any remaining resources in slots
  // Note: pool-managed GPU buffers are freed by nvjpeg_batched_cleanup
  for (size_t i = 0; i < queue->queue_depth; i++) {
    DecodeSlot *slot = &queue->slots[i];
#ifdef UNPAPER_WITH_CUDA
    if (slot->image.on_gpu && slot->image.gpu_ptr != NULL &&
        !slot->image.gpu_pool_owned) {
      cudaFree(slot->image.gpu_ptr);
      slot->image.gpu_ptr = NULL;
    }
#endif
    if (slot->image.frame) {
      av_frame_free(&slot->image.frame);
    }
  }

  pthread_mutex_destroy(&queue->collected_mutex);
  pthread_cond_destroy(&queue->collected_cond);
  pthread_mutex_destroy(&queue->slot_mutex);
  pthread_cond_destroy(&queue->slot_not_full);
  pthread_cond_destroy(&queue->slot_not_empty);

  free(queue->slots);
  free(queue);
}

void batch_decode_queue_enable_gpu(BatchDecodeQueue *queue, bool enable) {
  if (!queue) {
    return;
  }
#ifdef UNPAPER_WITH_CUDA
  queue->use_gpu_decode = enable;
#else
  (void)enable;
#endif
}

void batch_decode_queue_set_chunk_size(BatchDecodeQueue *queue,
                                       int chunk_size) {
  if (!queue) {
    return;
  }
  // Clamp to valid range (1-64, or 0 for default)
  if (chunk_size < 0) {
    chunk_size = 0;
  }
  if (chunk_size > 64) {
    chunk_size = 64;
  }
  queue->chunk_size = chunk_size;
}

bool batch_decode_queue_start(BatchDecodeQueue *queue, BatchQueue *batch_queue,
                              const Options *options) {
  if (!queue || !batch_queue || queue->orchestrator_started) {
    return false;
  }

  queue->batch_queue = batch_queue;
  queue->options = options;
  atomic_store(&queue->running, true);

  // Start orchestrator thread
  if (pthread_create(&queue->orchestrator_thread, NULL, orchestrator_thread_fn,
                     queue) != 0) {
    atomic_store(&queue->running, false);
    return false;
  }

  queue->orchestrator_started = true;
  return true;
}

void batch_decode_queue_stop(BatchDecodeQueue *queue) {
  if (!queue || !queue->orchestrator_started) {
    return;
  }

  // Signal shutdown
  atomic_store(&queue->running, false);

  // Wake up any waiting threads
  pthread_mutex_lock(&queue->collected_mutex);
  pthread_cond_broadcast(&queue->collected_cond);
  pthread_mutex_unlock(&queue->collected_mutex);

  pthread_mutex_lock(&queue->slot_mutex);
  pthread_cond_broadcast(&queue->slot_not_full);
  pthread_cond_broadcast(&queue->slot_not_empty);
  pthread_mutex_unlock(&queue->slot_mutex);

  // Wait for orchestrator to finish
  pthread_join(queue->orchestrator_thread, NULL);
  queue->orchestrator_started = false;

#ifdef UNPAPER_WITH_CUDA
  // Cleanup batched decoder
  nvjpeg_batched_cleanup();
#endif
}

bool batch_decode_queue_done(BatchDecodeQueue *queue) {
  if (!queue) {
    return true;
  }
  // Done when orchestrator has finished (not started or thread completed)
  if (!queue->orchestrator_started) {
    return true;
  }
  // Check if running flag is false (orchestrator completed)
  return !atomic_load(&queue->running) ||
         atomic_load(&queue->total_images) >= queue->io_work_total;
}

BatchDecodedImage *batch_decode_queue_get(BatchDecodeQueue *queue,
                                          int job_index, int input_index) {
  if (!queue) {
    return NULL;
  }

  while (1) {
    // Try to find a ready slot
    int slot_idx = find_ready_slot(queue, job_index, input_index);
    if (slot_idx >= 0) {
      return &queue->slots[slot_idx].image;
    }

    // Check if done (no more images coming)
    if (batch_decode_queue_done(queue)) {
      // One more check
      slot_idx = find_ready_slot(queue, job_index, input_index);
      if (slot_idx >= 0) {
        return &queue->slots[slot_idx].image;
      }
      return NULL;
    }

    // Wait for producer
    pthread_mutex_lock(&queue->slot_mutex);
    pthread_cond_wait(&queue->slot_not_empty, &queue->slot_mutex);
    pthread_mutex_unlock(&queue->slot_mutex);
  }
}

void batch_decode_queue_release(BatchDecodeQueue *queue,
                                BatchDecodedImage *image) {
  if (!queue || !image) {
    return;
  }

  // Find the slot for this image
  for (size_t i = 0; i < queue->queue_depth; i++) {
    if (&queue->slots[i].image == image) {
      DecodeSlot *slot = &queue->slots[i];

#ifdef UNPAPER_WITH_CUDA
      // Only free GPU memory if it's NOT from the batch pool.
      // Pool buffers are managed by nvjpeg_batched and freed in cleanup.
      // CRITICAL: cudaFree is synchronous and would add ~10-50ms overhead
      // per image if called on pool buffers (50 images = 500-2500ms).
      if (slot->image.on_gpu && slot->image.gpu_ptr != NULL &&
          !slot->image.gpu_pool_owned) {
        cudaFree(slot->image.gpu_ptr);
        slot->image.gpu_ptr = NULL;
        slot->image.on_gpu = false;
      }
#endif

      if (slot->image.frame) {
        av_frame_free(&slot->image.frame);
      }

      memset(&slot->image, 0, sizeof(BatchDecodedImage));
      atomic_fetch_sub(&queue->current_depth, 1);
      atomic_store(&slot->state, SLOT_EMPTY);

      // Signal producer
      pthread_mutex_lock(&queue->slot_mutex);
      pthread_cond_signal(&queue->slot_not_full);
      pthread_mutex_unlock(&queue->slot_mutex);

      return;
    }
  }
}

BatchDecodeQueueStats
batch_decode_queue_get_stats(const BatchDecodeQueue *queue) {
  BatchDecodeQueueStats stats = {0};
  if (!queue) {
    return stats;
  }

  stats.total_images = atomic_load(&queue->total_images);
  stats.gpu_batched_decodes = atomic_load(&queue->gpu_batched_decodes);
  stats.gpu_single_decodes = atomic_load(&queue->gpu_single_decodes);
  stats.cpu_decodes = atomic_load(&queue->cpu_decodes);
  stats.decode_failures = atomic_load(&queue->decode_failures);
  stats.batch_calls = atomic_load(&queue->batch_calls);
  stats.io_threads_used = (size_t)queue->num_io_threads;
  stats.chunks_processed = atomic_load(&queue->chunks_processed);
  stats.peak_queue_depth = atomic_load(&queue->peak_depth);
  stats.total_io_time_ms = (double)atomic_load(&queue->io_time_us) / 1000.0;
  stats.total_decode_time_ms =
      (double)atomic_load(&queue->decode_time_us) / 1000.0;

  return stats;
}

void batch_decode_queue_print_stats(const BatchDecodeQueue *queue) {
  if (!queue) {
    return;
  }

  BatchDecodeQueueStats stats = batch_decode_queue_get_stats(queue);

  double gpu_batch_rate = 0.0;
  if (stats.total_images > 0) {
    gpu_batch_rate =
        100.0 * (double)stats.gpu_batched_decodes / (double)stats.total_images;
  }

  fprintf(stderr,
          "Batch Decode Queue Statistics:\n"
          "  Total images: %zu\n"
          "  GPU batched decodes: %zu (%.1f%%)\n"
          "  GPU single decodes: %zu\n"
          "  CPU decodes: %zu\n"
          "  Decode failures: %zu\n"
          "  Batch API calls: %zu\n"
          "  Chunks processed: %zu\n"
          "  I/O threads: %zu\n"
          "  Peak queue depth: %zu\n"
          "  Total I/O time: %.1f ms\n"
          "  Total decode time: %.1f ms\n",
          stats.total_images, stats.gpu_batched_decodes, gpu_batch_rate,
          stats.gpu_single_decodes, stats.cpu_decodes, stats.decode_failures,
          stats.batch_calls, stats.chunks_processed, stats.io_threads_used,
          stats.peak_queue_depth, stats.total_io_time_ms,
          stats.total_decode_time_ms);
}
