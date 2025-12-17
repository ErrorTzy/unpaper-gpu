// SPDX-FileCopyrightText: 2025 The unpaper authors
//
// SPDX-License-Identifier: GPL-2.0-only

#include "imageprocess/nvimgcodec.h"

#if defined(UNPAPER_WITH_NVIMGCODEC) && (UNPAPER_WITH_NVIMGCODEC)

#include <cuda_runtime.h>
#include <nvimgcodec.h>
#include <pthread.h>
#include <stdatomic.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "imageprocess/cuda_runtime.h"
#include "lib/logging.h"

// Slot states for lock-free pool
#define SLOT_FREE 0
#define SLOT_IN_USE 1

// Maximum pool sizes
#define MAX_DECODE_STATES 32
#define MAX_ENCODE_STATES 16

// ============================================================================
// Custom Async Allocators for nvImageCodec
// ============================================================================
// These allocators use cudaMallocAsync/cudaFreeAsync to avoid synchronous
// cudaMalloc calls that would block ALL streams until complete.
// This is CRITICAL for performance - without these, nvImageCodec falls back
// to sync allocators which cause the ~30% performance regression.

static int nvimgcodec_device_malloc(void *ctx, void **ptr, size_t size,
                                    cudaStream_t stream) {
  (void)ctx;
  cudaError_t err = cudaMallocAsync(ptr, size, stream);
  if (err != cudaSuccess) {
    // Fallback to sync malloc if async not available
    err = cudaMalloc(ptr, size);
  }
  return (err == cudaSuccess) ? 0 : -1;
}

static int nvimgcodec_device_free(void *ctx, void *ptr, size_t size,
                                  cudaStream_t stream) {
  (void)ctx;
  (void)size;
  cudaError_t err = cudaFreeAsync(ptr, stream);
  if (err != cudaSuccess) {
    // Fallback to sync free if async not available
    err = cudaFree(ptr);
  }
  return (err == cudaSuccess) ? 0 : -1;
}

static int nvimgcodec_pinned_malloc(void *ctx, void **ptr, size_t size,
                                    cudaStream_t stream) {
  (void)ctx;
  (void)stream;
  cudaError_t err = cudaMallocHost(ptr, size);
  return (err == cudaSuccess) ? 0 : -1;
}

static int nvimgcodec_pinned_free(void *ctx, void *ptr, size_t size,
                                  cudaStream_t stream) {
  (void)ctx;
  (void)size;
  (void)stream;
  cudaError_t err = cudaFreeHost(ptr);
  return (err == cudaSuccess) ? 0 : -1;
}

// Static allocator structures - passed to nvImageCodec via exec_params
static nvimgcodecDeviceAllocator_t g_device_allocator = {
    NVIMGCODEC_STRUCTURE_TYPE_DEVICE_ALLOCATOR,
    sizeof(nvimgcodecDeviceAllocator_t),
    NULL,
    nvimgcodec_device_malloc,
    nvimgcodec_device_free,
    NULL, // device_ctx
    0     // device_mem_padding
};

static nvimgcodecPinnedAllocator_t g_pinned_allocator = {
    NVIMGCODEC_STRUCTURE_TYPE_PINNED_ALLOCATOR,
    sizeof(nvimgcodecPinnedAllocator_t),
    NULL,
    nvimgcodec_pinned_malloc,
    nvimgcodec_pinned_free,
    NULL, // pinned_ctx
    0     // pinned_mem_padding
};

// ============================================================================
// Decode State Structure
// ============================================================================

struct NvImgCodecDecodeState {
  nvimgcodecDecoder_t decoder;
  cudaStream_t stream;          // Dedicated CUDA stream
  cudaEvent_t completion_event; // Pre-allocated event
  atomic_int in_use;

  // Persistent objects for reuse (avoid per-operation allocation overhead)
  nvimgcodecImage_t cached_image;            // Reused across decode calls
  nvimgcodecCodeStream_t cached_code_stream; // Reused across decode calls
};

// ============================================================================
// Encode State Structure
// ============================================================================

struct NvImgCodecEncodeState {
  nvimgcodecEncoder_t encoder;
  cudaStream_t stream;
  atomic_int in_use;

  // Persistent objects for reuse (avoid per-operation allocation overhead)
  nvimgcodecImage_t cached_image;            // Reused across encode calls
  nvimgcodecCodeStream_t cached_code_stream; // Reused across encode calls
};

// ============================================================================
// Global Context
// ============================================================================

typedef struct {
  nvimgcodecInstance_t instance;
  NvImgCodecDecodeState *decode_states;
  NvImgCodecEncodeState *encode_states;
  int num_decode_states;
  int num_encode_states;
  int jpeg_quality;
  bool initialized;

  // Statistics
  atomic_size_t total_decodes;
  atomic_size_t successful_decodes;
  atomic_size_t jpeg_decodes;
  atomic_size_t jp2_decodes;
  atomic_size_t fallback_decodes;
  atomic_size_t total_encodes;
  atomic_size_t successful_encodes;
  atomic_size_t jpeg_encodes;
  atomic_size_t jp2_encodes;
} NvImgCodecContext;

static NvImgCodecContext g_ctx = {0};
static pthread_mutex_t g_init_mutex = PTHREAD_MUTEX_INITIALIZER;

// ============================================================================
// Format Detection
// ============================================================================

NvImgCodecFormat nvimgcodec_detect_format(const uint8_t *data, size_t size) {
  if (data == NULL || size < 4) {
    return NVIMGCODEC_FORMAT_UNKNOWN;
  }

  // JPEG: FFD8FF
  if (size >= 3 && data[0] == 0xFF && data[1] == 0xD8 && data[2] == 0xFF) {
    return NVIMGCODEC_FORMAT_JPEG;
  }

  // JPEG2000 codestream: FF4FFF51
  if (size >= 4 && data[0] == 0xFF && data[1] == 0x4F && data[2] == 0xFF &&
      data[3] == 0x51) {
    return NVIMGCODEC_FORMAT_JPEG2000;
  }

  // JPEG2000 JP2 file format: 0000000C6A5020200D0A870A
  if (size >= 12) {
    static const uint8_t jp2_sig[] = {0x00, 0x00, 0x00, 0x0C, 0x6A, 0x50,
                                      0x20, 0x20, 0x0D, 0x0A, 0x87, 0x0A};
    if (memcmp(data, jp2_sig, 12) == 0) {
      return NVIMGCODEC_FORMAT_JPEG2000;
    }
  }

  // PNG: 89504E47
  if (size >= 4 && data[0] == 0x89 && data[1] == 0x50 && data[2] == 0x4E &&
      data[3] == 0x47) {
    return NVIMGCODEC_FORMAT_PNG;
  }

  // TIFF: 49492A00 or 4D4D002A
  if (size >= 4) {
    if ((data[0] == 0x49 && data[1] == 0x49 && data[2] == 0x2A &&
         data[3] == 0x00) ||
        (data[0] == 0x4D && data[1] == 0x4D && data[2] == 0x00 &&
         data[3] == 0x2A)) {
      return NVIMGCODEC_FORMAT_TIFF;
    }
  }

  return NVIMGCODEC_FORMAT_UNKNOWN;
}

bool nvimgcodec_format_decode_supported(NvImgCodecFormat format) {
  switch (format) {
  case NVIMGCODEC_FORMAT_JPEG:
  case NVIMGCODEC_FORMAT_JPEG2000:
    return true;
  case NVIMGCODEC_FORMAT_PNG:
  case NVIMGCODEC_FORMAT_TIFF:
    return g_ctx.initialized; // Only with full nvImageCodec
  default:
    return false;
  }
}

bool nvimgcodec_format_encode_supported(NvImgCodecFormat format) {
  switch (format) {
  case NVIMGCODEC_FORMAT_JPEG:
  case NVIMGCODEC_FORMAT_JPEG2000:
    return true;
  default:
    return false;
  }
}

// ============================================================================
// Decode State Management
// ============================================================================

static bool init_decode_state(NvImgCodecDecodeState *state) {
  nvimgcodecStatus_t status;

  // Create dedicated CUDA stream
  cudaError_t cuda_err = cudaStreamCreate(&state->stream);
  if (cuda_err != cudaSuccess) {
    verboseLog(VERBOSE_DEBUG, "nvimgcodec: failed to create CUDA stream: %s\n",
               cudaGetErrorString(cuda_err));
    return false;
  }

  // Create completion event
  cuda_err = cudaEventCreateWithFlags(&state->completion_event,
                                      cudaEventDisableTiming);
  if (cuda_err != cudaSuccess) {
    cudaStreamDestroy(state->stream);
    state->stream = NULL;
    return false;
  }

  // Create decoder with GPU backend and async allocators
  nvimgcodecExecutionParams_t exec_params = {
      NVIMGCODEC_STRUCTURE_TYPE_EXECUTION_PARAMS, sizeof(exec_params), NULL};
  exec_params.device_id = 0;
  exec_params.max_num_cpu_threads = 1;
  exec_params.num_backends = 0; // Use all available backends
  // Use custom async allocators to avoid synchronous cudaMalloc/cudaFree
  // which block all streams. This improves nvJPEG internal allocation
  // performance.
  exec_params.device_allocator = &g_device_allocator;
  exec_params.pinned_allocator = &g_pinned_allocator;

  status = nvimgcodecDecoderCreate(g_ctx.instance, &state->decoder,
                                   &exec_params, NULL);
  if (status != NVIMGCODEC_STATUS_SUCCESS) {
    verboseLog(VERBOSE_DEBUG, "nvimgcodec: failed to create decoder\n");
    cudaEventDestroy(state->completion_event);
    cudaStreamDestroy(state->stream);
    state->stream = NULL;
    return false;
  }

  atomic_init(&state->in_use, SLOT_FREE);
  return true;
}

static void cleanup_decode_state(NvImgCodecDecodeState *state) {
  // Destroy cached nvImageCodec objects (reused across decode calls)
  if (state->cached_image != NULL) {
    nvimgcodecImageDestroy(state->cached_image);
    state->cached_image = NULL;
  }
  if (state->cached_code_stream != NULL) {
    nvimgcodecCodeStreamDestroy(state->cached_code_stream);
    state->cached_code_stream = NULL;
  }
  if (state->decoder != NULL) {
    nvimgcodecDecoderDestroy(state->decoder);
    state->decoder = NULL;
  }
  if (state->completion_event != NULL) {
    cudaEventDestroy(state->completion_event);
    state->completion_event = NULL;
  }
  if (state->stream != NULL) {
    cudaStreamDestroy(state->stream);
    state->stream = NULL;
  }
}

// ============================================================================
// Encode State Management
// ============================================================================

static bool init_encode_state(NvImgCodecEncodeState *state) {
  nvimgcodecStatus_t status;

  // Create dedicated CUDA stream
  cudaError_t cuda_err = cudaStreamCreate(&state->stream);
  if (cuda_err != cudaSuccess) {
    return false;
  }

  // Create encoder with GPU backend and async allocators
  nvimgcodecExecutionParams_t exec_params = {
      NVIMGCODEC_STRUCTURE_TYPE_EXECUTION_PARAMS, sizeof(exec_params), NULL};
  exec_params.device_id = 0;
  exec_params.max_num_cpu_threads = 1;
  // Use custom async allocators to avoid synchronous cudaMalloc/cudaFree
  // which block all streams. This improves nvJPEG internal allocation
  // performance.
  exec_params.device_allocator = &g_device_allocator;
  exec_params.pinned_allocator = &g_pinned_allocator;

  status = nvimgcodecEncoderCreate(g_ctx.instance, &state->encoder,
                                   &exec_params, NULL);
  if (status != NVIMGCODEC_STATUS_SUCCESS) {
    cudaStreamDestroy(state->stream);
    state->stream = NULL;
    return false;
  }

  atomic_init(&state->in_use, SLOT_FREE);
  return true;
}

static void cleanup_encode_state(NvImgCodecEncodeState *state) {
  // Destroy cached nvImageCodec objects (reused across encode calls)
  if (state->cached_image != NULL) {
    nvimgcodecImageDestroy(state->cached_image);
    state->cached_image = NULL;
  }
  if (state->cached_code_stream != NULL) {
    nvimgcodecCodeStreamDestroy(state->cached_code_stream);
    state->cached_code_stream = NULL;
  }
  if (state->encoder != NULL) {
    nvimgcodecEncoderDestroy(state->encoder);
    state->encoder = NULL;
  }
  if (state->stream != NULL) {
    cudaStreamDestroy(state->stream);
    state->stream = NULL;
  }
}

// ============================================================================
// Global Context
// ============================================================================

bool nvimgcodec_init(int num_streams) {
  pthread_mutex_lock(&g_init_mutex);

  if (g_ctx.initialized) {
    pthread_mutex_unlock(&g_init_mutex);
    return true;
  }

  // Ensure CUDA is initialized
  UnpaperCudaInitStatus cuda_status = unpaper_cuda_try_init();
  if (cuda_status != UNPAPER_CUDA_INIT_OK) {
    verboseLog(VERBOSE_DEBUG, "nvimgcodec: CUDA not available\n");
    pthread_mutex_unlock(&g_init_mutex);
    return false;
  }

  // Create nvImageCodec instance
  nvimgcodecInstanceCreateInfo_t create_info = {
      NVIMGCODEC_STRUCTURE_TYPE_INSTANCE_CREATE_INFO, sizeof(create_info),
      NULL};
  create_info.load_builtin_modules = 1;
  create_info.load_extension_modules = 1;

  nvimgcodecStatus_t status =
      nvimgcodecInstanceCreate(&g_ctx.instance, &create_info);
  if (status != NVIMGCODEC_STATUS_SUCCESS) {
    verboseLog(VERBOSE_DEBUG, "nvimgcodec: failed to create instance\n");
    pthread_mutex_unlock(&g_init_mutex);
    return false;
  }

  // Allocate decode states
  int num_decode =
      (num_streams > MAX_DECODE_STATES) ? MAX_DECODE_STATES : num_streams;
  g_ctx.decode_states =
      calloc((size_t)num_decode, sizeof(NvImgCodecDecodeState));
  if (g_ctx.decode_states == NULL) {
    nvimgcodecInstanceDestroy(g_ctx.instance);
    pthread_mutex_unlock(&g_init_mutex);
    return false;
  }

  for (int i = 0; i < num_decode; i++) {
    if (!init_decode_state(&g_ctx.decode_states[i])) {
      for (int j = 0; j < i; j++) {
        cleanup_decode_state(&g_ctx.decode_states[j]);
      }
      free(g_ctx.decode_states);
      nvimgcodecInstanceDestroy(g_ctx.instance);
      pthread_mutex_unlock(&g_init_mutex);
      return false;
    }
  }
  g_ctx.num_decode_states = num_decode;

  // Allocate encode states
  int num_encode =
      (num_streams > MAX_ENCODE_STATES) ? MAX_ENCODE_STATES : num_streams;
  g_ctx.encode_states =
      calloc((size_t)num_encode, sizeof(NvImgCodecEncodeState));
  if (g_ctx.encode_states == NULL) {
    for (int i = 0; i < num_decode; i++) {
      cleanup_decode_state(&g_ctx.decode_states[i]);
    }
    free(g_ctx.decode_states);
    nvimgcodecInstanceDestroy(g_ctx.instance);
    pthread_mutex_unlock(&g_init_mutex);
    return false;
  }

  for (int i = 0; i < num_encode; i++) {
    if (!init_encode_state(&g_ctx.encode_states[i])) {
      for (int j = 0; j < i; j++) {
        cleanup_encode_state(&g_ctx.encode_states[j]);
      }
      free(g_ctx.encode_states);
      for (int j = 0; j < num_decode; j++) {
        cleanup_decode_state(&g_ctx.decode_states[j]);
      }
      free(g_ctx.decode_states);
      nvimgcodecInstanceDestroy(g_ctx.instance);
      pthread_mutex_unlock(&g_init_mutex);
      return false;
    }
  }
  g_ctx.num_encode_states = num_encode;

  // Initialize statistics
  g_ctx.jpeg_quality = 85;
  atomic_init(&g_ctx.total_decodes, 0);
  atomic_init(&g_ctx.successful_decodes, 0);
  atomic_init(&g_ctx.jpeg_decodes, 0);
  atomic_init(&g_ctx.jp2_decodes, 0);
  atomic_init(&g_ctx.fallback_decodes, 0);
  atomic_init(&g_ctx.total_encodes, 0);
  atomic_init(&g_ctx.successful_encodes, 0);
  atomic_init(&g_ctx.jpeg_encodes, 0);
  atomic_init(&g_ctx.jp2_encodes, 0);

  g_ctx.initialized = true;

  verboseLog(
      VERBOSE_DEBUG,
      "nvimgcodec: initialized with %d decode states, %d encode states\n",
      num_decode, num_encode);

  pthread_mutex_unlock(&g_init_mutex);
  return true;
}

void nvimgcodec_cleanup(void) {
  pthread_mutex_lock(&g_init_mutex);

  if (!g_ctx.initialized) {
    pthread_mutex_unlock(&g_init_mutex);
    return;
  }

  // Cleanup encode states
  if (g_ctx.encode_states != NULL) {
    for (int i = 0; i < g_ctx.num_encode_states; i++) {
      cleanup_encode_state(&g_ctx.encode_states[i]);
    }
    free(g_ctx.encode_states);
    g_ctx.encode_states = NULL;
  }

  // Cleanup decode states
  if (g_ctx.decode_states != NULL) {
    for (int i = 0; i < g_ctx.num_decode_states; i++) {
      cleanup_decode_state(&g_ctx.decode_states[i]);
    }
    free(g_ctx.decode_states);
    g_ctx.decode_states = NULL;
  }

  // Destroy instance
  if (g_ctx.instance != NULL) {
    nvimgcodecInstanceDestroy(g_ctx.instance);
    g_ctx.instance = NULL;
  }

  g_ctx.initialized = false;

  pthread_mutex_unlock(&g_init_mutex);
}

bool nvimgcodec_is_available(void) { return g_ctx.initialized; }

bool nvimgcodec_any_available(void) { return g_ctx.initialized; }

bool nvimgcodec_jp2_supported(void) {
  return g_ctx.initialized; // JP2 supported when nvImageCodec is available
}

// ============================================================================
// State Pool Management
// ============================================================================

NvImgCodecDecodeState *nvimgcodec_acquire_decode_state(void) {
  if (!g_ctx.initialized || g_ctx.decode_states == NULL) {
    return NULL;
  }

  for (int i = 0; i < g_ctx.num_decode_states; i++) {
    int expected = SLOT_FREE;
    if (atomic_compare_exchange_strong(&g_ctx.decode_states[i].in_use,
                                       &expected, SLOT_IN_USE)) {
      return &g_ctx.decode_states[i];
    }
  }
  return NULL;
}

void nvimgcodec_release_decode_state(NvImgCodecDecodeState *state) {
  if (state != NULL) {
    atomic_store(&state->in_use, SLOT_FREE);
  }
}

NvImgCodecEncodeState *nvimgcodec_acquire_encode_state(void) {
  if (!g_ctx.initialized || g_ctx.encode_states == NULL) {
    return NULL;
  }

  for (int i = 0; i < g_ctx.num_encode_states; i++) {
    int expected = SLOT_FREE;
    if (atomic_compare_exchange_strong(&g_ctx.encode_states[i].in_use,
                                       &expected, SLOT_IN_USE)) {
      return &g_ctx.encode_states[i];
    }
  }
  return NULL;
}

void nvimgcodec_release_encode_state(NvImgCodecEncodeState *state) {
  if (state != NULL) {
    atomic_store(&state->in_use, SLOT_FREE);
  }
}

// ============================================================================
// Statistics
// ============================================================================

NvImgCodecStats nvimgcodec_get_stats(void) {
  NvImgCodecStats stats = {0};
  if (!g_ctx.initialized) {
    return stats;
  }

  stats.total_decodes = atomic_load(&g_ctx.total_decodes);
  stats.successful_decodes = atomic_load(&g_ctx.successful_decodes);
  stats.jpeg_decodes = atomic_load(&g_ctx.jpeg_decodes);
  stats.jp2_decodes = atomic_load(&g_ctx.jp2_decodes);
  stats.fallback_decodes = atomic_load(&g_ctx.fallback_decodes);
  stats.total_encodes = atomic_load(&g_ctx.total_encodes);
  stats.successful_encodes = atomic_load(&g_ctx.successful_encodes);
  stats.jpeg_encodes = atomic_load(&g_ctx.jpeg_encodes);
  stats.jp2_encodes = atomic_load(&g_ctx.jp2_encodes);
  stats.using_nvimgcodec = true;

  return stats;
}

void nvimgcodec_print_stats(void) {
  NvImgCodecStats stats = nvimgcodec_get_stats();

  fprintf(stderr,
          "nvImageCodec Statistics:\n"
          "  Backend: nvImageCodec (JPEG + JP2)\n"
          "  Total decodes: %zu (JPEG: %zu, JP2: %zu)\n"
          "  Successful: %zu, Fallbacks: %zu\n"
          "  Total encodes: %zu (JPEG: %zu, JP2: %zu)\n"
          "  Successful: %zu\n",
          stats.total_decodes, stats.jpeg_decodes, stats.jp2_decodes,
          stats.successful_decodes, stats.fallback_decodes, stats.total_encodes,
          stats.jpeg_encodes, stats.jp2_encodes, stats.successful_encodes);
}

// ============================================================================
// Image Info
// ============================================================================

bool nvimgcodec_get_image_info(const uint8_t *data, size_t size,
                               NvImgCodecFormat *format, int *width,
                               int *height, int *channels) {
  if (!g_ctx.initialized || data == NULL || size == 0) {
    return false;
  }

  NvImgCodecFormat fmt = nvimgcodec_detect_format(data, size);
  if (format != NULL) {
    *format = fmt;
  }

  // Create code stream from memory
  nvimgcodecCodeStream_t code_stream = NULL;
  nvimgcodecStatus_t status = nvimgcodecCodeStreamCreateFromHostMem(
      g_ctx.instance, &code_stream, data, size);
  if (status != NVIMGCODEC_STATUS_SUCCESS) {
    return false;
  }

  // Get image info
  nvimgcodecImageInfo_t info = {NVIMGCODEC_STRUCTURE_TYPE_IMAGE_INFO,
                                sizeof(info), NULL};
  status = nvimgcodecCodeStreamGetImageInfo(code_stream, &info);

  nvimgcodecCodeStreamDestroy(code_stream);

  if (status != NVIMGCODEC_STATUS_SUCCESS) {
    return false;
  }

  if (width != NULL) {
    *width = (int)info.plane_info[0].width;
  }
  if (height != NULL) {
    *height = (int)info.plane_info[0].height;
  }
  if (channels != NULL) {
    *channels = (int)info.num_planes;
    // For interleaved, count channels from first plane
    if (info.sample_format == NVIMGCODEC_SAMPLEFORMAT_I_RGB ||
        info.sample_format == NVIMGCODEC_SAMPLEFORMAT_I_BGR) {
      *channels = 3;
    } else if (info.sample_format == NVIMGCODEC_SAMPLEFORMAT_P_Y) {
      *channels = 1;
    }
  }

  return true;
}

// ============================================================================
// Decode Operations
// ============================================================================

void nvimgcodec_wait_decode_complete(NvImgCodecDecodedImage *image) {
  if (image == NULL || image->completion_event == NULL) {
    return;
  }

  cudaEvent_t event = (cudaEvent_t)image->completion_event;
  cudaEventSynchronize(event);
  // Don't destroy/release - the event belongs to the decode state
  image->completion_event = NULL;
}

void nvimgcodec_release_completion_event(void *event, bool from_pool) {
  (void)event;
  (void)from_pool;
  // Events are owned by decode states, not released separately
}

bool nvimgcodec_decode(const uint8_t *data, size_t size,
                       NvImgCodecDecodeState *state, UnpaperCudaStream *stream,
                       NvImgCodecOutputFormat output_fmt,
                       NvImgCodecDecodedImage *out) {
  if (!g_ctx.initialized || data == NULL || size == 0 || state == NULL ||
      out == NULL) {
    return false;
  }

  (void)stream; // Use state's dedicated stream

  atomic_fetch_add(&g_ctx.total_decodes, 1);

  // Detect format
  NvImgCodecFormat fmt = nvimgcodec_detect_format(data, size);
  if (fmt == NVIMGCODEC_FORMAT_UNKNOWN) {
    atomic_fetch_add(&g_ctx.fallback_decodes, 1);
    return false;
  }

  // Create or reuse code stream from memory
  // Passing &state->cached_code_stream reuses existing object if non-NULL
  // (nvImageCodec 0.7+ feature: avoids per-operation allocation overhead)
  nvimgcodecStatus_t status = nvimgcodecCodeStreamCreateFromHostMem(
      g_ctx.instance, &state->cached_code_stream, data, size);
  if (status != NVIMGCODEC_STATUS_SUCCESS) {
    atomic_fetch_add(&g_ctx.fallback_decodes, 1);
    return false;
  }

  // Get image info
  nvimgcodecImageInfo_t info = {NVIMGCODEC_STRUCTURE_TYPE_IMAGE_INFO,
                                sizeof(info), NULL};
  status = nvimgcodecCodeStreamGetImageInfo(state->cached_code_stream, &info);
  if (status != NVIMGCODEC_STATUS_SUCCESS) {
    // Don't destroy cached_code_stream - it will be reused on next call
    atomic_fetch_add(&g_ctx.fallback_decodes, 1);
    return false;
  }

  int width = (int)info.plane_info[0].width;
  int height = (int)info.plane_info[0].height;
  int out_channels = (output_fmt == NVIMGCODEC_OUT_GRAY8) ? 1 : 3;

  // Set output format
  nvimgcodecSampleFormat_t sample_fmt;
  switch (output_fmt) {
  case NVIMGCODEC_OUT_GRAY8:
    sample_fmt = NVIMGCODEC_SAMPLEFORMAT_P_Y;
    break;
  case NVIMGCODEC_OUT_RGB:
    sample_fmt = NVIMGCODEC_SAMPLEFORMAT_I_RGB;
    break;
  case NVIMGCODEC_OUT_BGR:
    sample_fmt = NVIMGCODEC_SAMPLEFORMAT_I_BGR;
    break;
  default:
    sample_fmt = NVIMGCODEC_SAMPLEFORMAT_I_RGB;
    break;
  }

  // Calculate pitch (aligned to 256 bytes)
  size_t pitch = (size_t)width * (size_t)out_channels;
  pitch = (pitch + 255) & ~(size_t)255;
  size_t buffer_size = pitch * (size_t)height;

  // Allocate output buffer
  void *gpu_buffer = NULL;
  cudaError_t cuda_err =
      cudaMallocAsync(&gpu_buffer, buffer_size, state->stream);
  if (cuda_err != cudaSuccess) {
    cuda_err = cudaMalloc(&gpu_buffer, buffer_size);
    if (cuda_err != cudaSuccess) {
      // Don't destroy cached_code_stream - it will be reused on next call
      atomic_fetch_add(&g_ctx.fallback_decodes, 1);
      return false;
    }
  }

  // Set up output image info
  nvimgcodecImageInfo_t out_info = {NVIMGCODEC_STRUCTURE_TYPE_IMAGE_INFO,
                                    sizeof(out_info), NULL};
  out_info.sample_format = sample_fmt;
  out_info.num_planes = 1;
  out_info.plane_info[0].width = (uint32_t)width;
  out_info.plane_info[0].height = (uint32_t)height;
  out_info.plane_info[0].num_channels = (uint32_t)out_channels;
  out_info.plane_info[0].sample_type = NVIMGCODEC_SAMPLE_DATA_TYPE_UINT8;
  out_info.plane_info[0].row_stride = (size_t)pitch;
  out_info.buffer = gpu_buffer;
  out_info.buffer_kind = NVIMGCODEC_IMAGE_BUFFER_KIND_STRIDED_DEVICE;
  out_info.cuda_stream = state->stream;

  // Create or reuse output image
  // Passing &state->cached_image reuses existing object if non-NULL
  // (nvImageCodec 0.7+ feature: avoids per-operation allocation overhead)
  status =
      nvimgcodecImageCreate(g_ctx.instance, &state->cached_image, &out_info);
  if (status != NVIMGCODEC_STATUS_SUCCESS) {
    cudaFreeAsync(gpu_buffer, state->stream);
    // Don't destroy cached objects - they will be reused on next call
    atomic_fetch_add(&g_ctx.fallback_decodes, 1);
    return false;
  }

  // Decode
  nvimgcodecDecodeParams_t decode_params = {
      NVIMGCODEC_STRUCTURE_TYPE_DECODE_PARAMS, sizeof(decode_params), NULL};
  decode_params.apply_exif_orientation = 0;

  nvimgcodecFuture_t future = NULL;
  status =
      nvimgcodecDecoderDecode(state->decoder, &state->cached_code_stream,
                              &state->cached_image, 1, &decode_params, &future);
  if (status != NVIMGCODEC_STATUS_SUCCESS) {
    cudaFreeAsync(gpu_buffer, state->stream);
    // Don't destroy cached objects - they will be reused on next call
    atomic_fetch_add(&g_ctx.fallback_decodes, 1);
    return false;
  }

  // Wait for decode to complete
  if (future != NULL) {
    nvimgcodecFutureWaitForAll(future);

    // Check processing status (one status per image in batch)
    nvimgcodecProcessingStatus_t proc_status = 0;
    size_t status_size = 1;
    nvimgcodecFutureGetProcessingStatus(future, &proc_status, &status_size);
    nvimgcodecFutureDestroy(future);

    if (!(proc_status & NVIMGCODEC_PROCESSING_STATUS_SUCCESS)) {
      cudaFreeAsync(gpu_buffer, state->stream);
      // Don't destroy cached objects - they will be reused on next call
      atomic_fetch_add(&g_ctx.fallback_decodes, 1);
      return false;
    }
  }

  // Record completion event
  cudaEventRecord(state->completion_event, state->stream);

  // NOTE: Don't destroy cached_image or cached_code_stream here!
  // They are reused across decode calls to avoid per-operation allocation
  // overhead. They will be destroyed in cleanup_decode_state().

  // Fill output
  out->gpu_ptr = gpu_buffer;
  out->pitch = pitch;
  out->width = width;
  out->height = height;
  out->channels = out_channels;
  out->fmt = output_fmt;
  out->source_fmt = fmt;
  out->completion_event = (void *)state->completion_event;
  out->event_from_pool = false;

  // Update stats
  atomic_fetch_add(&g_ctx.successful_decodes, 1);
  if (fmt == NVIMGCODEC_FORMAT_JPEG) {
    atomic_fetch_add(&g_ctx.jpeg_decodes, 1);
  } else if (fmt == NVIMGCODEC_FORMAT_JPEG2000) {
    atomic_fetch_add(&g_ctx.jp2_decodes, 1);
  }

  return true;
}

bool nvimgcodec_decode_file(const char *filename, UnpaperCudaStream *stream,
                            NvImgCodecOutputFormat output_fmt,
                            NvImgCodecDecodedImage *out) {
  if (filename == NULL || out == NULL) {
    return false;
  }

  // Read file
  FILE *f = fopen(filename, "rb");
  if (f == NULL) {
    return false;
  }

  fseek(f, 0, SEEK_END);
  long file_size = ftell(f);
  fseek(f, 0, SEEK_SET);

  if (file_size <= 0 || file_size > (long)(100 * 1024 * 1024)) {
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

  // Acquire state and decode
  NvImgCodecDecodeState *state = nvimgcodec_acquire_decode_state();
  if (state == NULL) {
    free(data);
    return false;
  }

  bool result = nvimgcodec_decode(data, (size_t)file_size, state, stream,
                                  output_fmt, out);

  nvimgcodec_release_decode_state(state);
  free(data);

  return result;
}

// ============================================================================
// Batch Decode Operations
// ============================================================================

int nvimgcodec_decode_batch(const uint8_t *const *data_ptrs,
                            const size_t *sizes, int batch_size,
                            NvImgCodecOutputFormat output_fmt,
                            NvImgCodecDecodedImage *outputs) {
  if (!g_ctx.initialized || data_ptrs == NULL || sizes == NULL ||
      outputs == NULL || batch_size <= 0) {
    return 0;
  }

  // Initialize all outputs to zero/NULL
  memset(outputs, 0, (size_t)batch_size * sizeof(NvImgCodecDecodedImage));

  // Acquire as many decode states as we can (up to batch_size)
  NvImgCodecDecodeState *states[MAX_DECODE_STATES];
  int num_states = 0;

  for (int i = 0; i < batch_size && i < g_ctx.num_decode_states; i++) {
    NvImgCodecDecodeState *state = nvimgcodec_acquire_decode_state();
    if (state != NULL) {
      states[num_states++] = state;
    }
  }

  if (num_states == 0) {
    verboseLog(VERBOSE_DEBUG,
               "nvimgcodec_decode_batch: no decode states available\n");
    return 0;
  }

  int successful_decodes = 0;

  // Process images in batches based on available states
  // Each iteration processes up to num_states images in parallel
  for (int offset = 0; offset < batch_size; offset += num_states) {
    int chunk_size = batch_size - offset;
    if (chunk_size > num_states) {
      chunk_size = num_states;
    }

    // Launch parallel decodes (each on its own stream via state)
    for (int i = 0; i < chunk_size; i++) {
      int img_idx = offset + i;
      if (data_ptrs[img_idx] == NULL || sizes[img_idx] == 0) {
        continue;
      }

      // Decode using the assigned state (which has its own stream)
      bool result =
          nvimgcodec_decode(data_ptrs[img_idx], sizes[img_idx], states[i], NULL,
                            output_fmt, &outputs[img_idx]);
      if (result) {
        successful_decodes++;
      }
    }

    // Wait for all decodes in this chunk to complete before reusing states
    for (int i = 0; i < chunk_size; i++) {
      int img_idx = offset + i;
      if (outputs[img_idx].completion_event != NULL) {
        nvimgcodec_wait_decode_complete(&outputs[img_idx]);
      }
    }
  }

  // Release all states back to the pool
  for (int i = 0; i < num_states; i++) {
    nvimgcodec_release_decode_state(states[i]);
  }

  return successful_decodes;
}

// ============================================================================
// Encode Operations
// ============================================================================

// Context for resize buffer callback
typedef struct {
  uint8_t *buffer;
  size_t size;
  size_t capacity;
} EncodeBufferContext;

// Resize buffer callback for nvimgcodecCodeStreamCreateToHostMem
static unsigned char *resize_buffer_callback(void *ctx, size_t req_size) {
  EncodeBufferContext *buf_ctx = (EncodeBufferContext *)ctx;

  if (req_size <= buf_ctx->capacity && buf_ctx->buffer != NULL) {
    buf_ctx->size = req_size;
    return buf_ctx->buffer;
  }

  // Need to allocate or grow buffer
  uint8_t *new_buffer = realloc(buf_ctx->buffer, req_size);
  if (new_buffer == NULL) {
    return NULL;
  }

  buf_ctx->buffer = new_buffer;
  buf_ctx->capacity = req_size;
  buf_ctx->size = req_size;
  return new_buffer;
}

bool nvimgcodec_encode(const void *gpu_ptr, size_t pitch, int width, int height,
                       NvImgCodecEncodeInputFormat input_fmt,
                       NvImgCodecEncodeState *state, UnpaperCudaStream *stream,
                       const NvImgCodecEncodeParams *params,
                       NvImgCodecEncodedImage *out) {
  if (!g_ctx.initialized || gpu_ptr == NULL || state == NULL ||
      params == NULL || out == NULL) {
    return false;
  }

  (void)stream; // Use state's dedicated stream

  atomic_fetch_add(&g_ctx.total_encodes, 1);

  int channels = (input_fmt == NVIMGCODEC_ENC_FMT_GRAY8) ? 1 : 3;

  // Set up input image info
  nvimgcodecImageInfo_t in_info = {NVIMGCODEC_STRUCTURE_TYPE_IMAGE_INFO,
                                   sizeof(in_info), NULL};

  switch (input_fmt) {
  case NVIMGCODEC_ENC_FMT_GRAY8:
    in_info.sample_format = NVIMGCODEC_SAMPLEFORMAT_P_Y;
    break;
  case NVIMGCODEC_ENC_FMT_RGB:
    in_info.sample_format = NVIMGCODEC_SAMPLEFORMAT_I_RGB;
    break;
  case NVIMGCODEC_ENC_FMT_BGR:
    in_info.sample_format = NVIMGCODEC_SAMPLEFORMAT_I_BGR;
    break;
  }

  in_info.num_planes = 1;
  in_info.plane_info[0].width = (uint32_t)width;
  in_info.plane_info[0].height = (uint32_t)height;
  in_info.plane_info[0].num_channels = (uint32_t)channels;
  in_info.plane_info[0].sample_type = NVIMGCODEC_SAMPLE_DATA_TYPE_UINT8;
  in_info.plane_info[0].row_stride = pitch;
  in_info.buffer = (void *)gpu_ptr;
  in_info.buffer_kind = NVIMGCODEC_IMAGE_BUFFER_KIND_STRIDED_DEVICE;
  in_info.cuda_stream = state->stream;

  // Create or reuse input image
  // Passing &state->cached_image reuses existing object if non-NULL
  // (nvImageCodec 0.7+ feature: avoids per-operation allocation overhead)
  nvimgcodecStatus_t status =
      nvimgcodecImageCreate(g_ctx.instance, &state->cached_image, &in_info);
  if (status != NVIMGCODEC_STATUS_SUCCESS) {
    return false;
  }

  // Set up output image info for code stream
  nvimgcodecImageInfo_t out_info = {NVIMGCODEC_STRUCTURE_TYPE_IMAGE_INFO,
                                    sizeof(out_info), NULL};
  const char *codec_name =
      params->output_format == NVIMGCODEC_FORMAT_JPEG2000 ? "jpeg2k" : "jpeg";
  strncpy(out_info.codec_name, codec_name, NVIMGCODEC_MAX_CODEC_NAME_SIZE - 1);
  out_info.codec_name[NVIMGCODEC_MAX_CODEC_NAME_SIZE - 1] = '\0';

  // Initialize buffer context for resize callback
  // Note: Each encode gets a fresh buffer context since output ownership
  // transfers to caller
  EncodeBufferContext buf_ctx = {NULL, 0, 0};

  // Create or reuse code stream for encoding
  // Passing &state->cached_code_stream reuses existing object if non-NULL
  status = nvimgcodecCodeStreamCreateToHostMem(
      g_ctx.instance, &state->cached_code_stream, &buf_ctx,
      resize_buffer_callback, &out_info);
  if (status != NVIMGCODEC_STATUS_SUCCESS) {
    // Don't destroy cached_image - it will be reused on next call
    return false;
  }

  // Set up encode params using quality_type and quality_value
  nvimgcodecEncodeParams_t encode_params = {
      NVIMGCODEC_STRUCTURE_TYPE_ENCODE_PARAMS, sizeof(encode_params), NULL};
  encode_params.quality_type = NVIMGCODEC_QUALITY_TYPE_QUALITY;
  encode_params.quality_value = (float)params->quality;

  // Encode
  nvimgcodecFuture_t future = NULL;
  status = nvimgcodecEncoderEncode(state->encoder, &state->cached_image,
                                   &state->cached_code_stream, 1,
                                   &encode_params, &future);
  if (status != NVIMGCODEC_STATUS_SUCCESS) {
    free(buf_ctx.buffer);
    // Don't destroy cached objects - they will be reused on next call
    return false;
  }

  // Wait for encode to complete
  if (future != NULL) {
    nvimgcodecFutureWaitForAll(future);

    // Check processing status
    nvimgcodecProcessingStatus_t proc_status = 0;
    size_t status_size = 1;
    nvimgcodecFutureGetProcessingStatus(future, &proc_status, &status_size);
    nvimgcodecFutureDestroy(future);

    if (!(proc_status & NVIMGCODEC_PROCESSING_STATUS_SUCCESS)) {
      free(buf_ctx.buffer);
      // Don't destroy cached objects - they will be reused on next call
      return false;
    }
  }

  // Sync CUDA stream to ensure encode is complete
  cudaStreamSynchronize(state->stream);

  // NOTE: Don't destroy cached_image or cached_code_stream here!
  // They are reused across encode calls to avoid per-operation allocation
  // overhead. They will be destroyed in cleanup_encode_state().

  // Fill output - buffer ownership transfers to caller
  out->data = buf_ctx.buffer;
  out->size = buf_ctx.size;
  out->width = width;
  out->height = height;
  out->fmt = params->output_format;

  // Update stats
  atomic_fetch_add(&g_ctx.successful_encodes, 1);
  if (params->output_format == NVIMGCODEC_FORMAT_JPEG) {
    atomic_fetch_add(&g_ctx.jpeg_encodes, 1);
  } else if (params->output_format == NVIMGCODEC_FORMAT_JPEG2000) {
    atomic_fetch_add(&g_ctx.jp2_encodes, 1);
  }

  return true;
}

bool nvimgcodec_encode_jpeg(const void *gpu_ptr, size_t pitch, int width,
                            int height, NvImgCodecEncodeInputFormat input_fmt,
                            int quality, NvImgCodecEncodeState *state,
                            UnpaperCudaStream *stream,
                            NvImgCodecEncodedImage *out) {
  NvImgCodecEncodeParams params = nvimgcodec_default_jpeg_params();
  params.quality = (quality > 0) ? quality : g_ctx.jpeg_quality;
  return nvimgcodec_encode(gpu_ptr, pitch, width, height, input_fmt, state,
                           stream, &params, out);
}

bool nvimgcodec_encode_jp2(const void *gpu_ptr, size_t pitch, int width,
                           int height, NvImgCodecEncodeInputFormat input_fmt,
                           bool lossless, NvImgCodecEncodeState *state,
                           UnpaperCudaStream *stream,
                           NvImgCodecEncodedImage *out) {
  NvImgCodecEncodeParams params = {
      .output_format = NVIMGCODEC_FORMAT_JPEG2000,
      .quality = lossless ? 100 : 85,
      .lossless = lossless,
  };
  return nvimgcodec_encode(gpu_ptr, pitch, width, height, input_fmt, state,
                           stream, &params, out);
}

bool nvimgcodec_encode_to_file(const void *gpu_ptr, size_t pitch, int width,
                               int height,
                               NvImgCodecEncodeInputFormat input_fmt,
                               UnpaperCudaStream *stream,
                               const NvImgCodecEncodeParams *params,
                               const char *filename) {
  if (filename == NULL) {
    return false;
  }

  NvImgCodecEncodeState *state = nvimgcodec_acquire_encode_state();
  if (state == NULL) {
    return false;
  }

  NvImgCodecEncodedImage out = {0};
  bool result = nvimgcodec_encode(gpu_ptr, pitch, width, height, input_fmt,
                                  state, stream, params, &out);

  nvimgcodec_release_encode_state(state);

  if (!result) {
    return false;
  }

  // Write to file
  FILE *f = fopen(filename, "wb");
  if (f == NULL) {
    free(out.data);
    return false;
  }

  size_t written = fwrite(out.data, 1, out.size, f);
  fclose(f);
  free(out.data);

  return written == out.size;
}

// ============================================================================
// Quality Control
// ============================================================================

void nvimgcodec_set_jpeg_quality(int quality) {
  g_ctx.jpeg_quality = (quality < 1) ? 1 : (quality > 100) ? 100 : quality;
}

int nvimgcodec_get_jpeg_quality(void) { return g_ctx.jpeg_quality; }

#else // !UNPAPER_WITH_NVIMGCODEC

// ============================================================================
// Stub implementations when nvImageCodec is not available
// ============================================================================
// For CUDA builds, nvImageCodec is now required (see meson.build).
// These stubs are only used for non-CUDA builds.

NvImgCodecFormat nvimgcodec_detect_format(const uint8_t *data, size_t size) {
  (void)data;
  (void)size;
  return NVIMGCODEC_FORMAT_UNKNOWN;
}

bool nvimgcodec_format_decode_supported(NvImgCodecFormat format) {
  (void)format;
  return false;
}

bool nvimgcodec_format_encode_supported(NvImgCodecFormat format) {
  (void)format;
  return false;
}

bool nvimgcodec_init(int num_streams) {
  (void)num_streams;
  return false;
}

void nvimgcodec_cleanup(void) {}

bool nvimgcodec_is_available(void) { return false; }

bool nvimgcodec_any_available(void) { return false; }

bool nvimgcodec_jp2_supported(void) { return false; }

NvImgCodecStats nvimgcodec_get_stats(void) {
  NvImgCodecStats stats = {0};
  return stats;
}

void nvimgcodec_print_stats(void) {}

NvImgCodecDecodeState *nvimgcodec_acquire_decode_state(void) { return NULL; }

void nvimgcodec_release_decode_state(NvImgCodecDecodeState *state) {
  (void)state;
}

NvImgCodecEncodeState *nvimgcodec_acquire_encode_state(void) { return NULL; }

void nvimgcodec_release_encode_state(NvImgCodecEncodeState *state) {
  (void)state;
}

bool nvimgcodec_get_image_info(const uint8_t *data, size_t size,
                               NvImgCodecFormat *format, int *width,
                               int *height, int *channels) {
  (void)data;
  (void)size;
  (void)format;
  (void)width;
  (void)height;
  (void)channels;
  return false;
}

void nvimgcodec_wait_decode_complete(NvImgCodecDecodedImage *image) {
  (void)image;
}

void nvimgcodec_release_completion_event(void *event, bool from_pool) {
  (void)event;
  (void)from_pool;
}

bool nvimgcodec_decode(const uint8_t *data, size_t size,
                       NvImgCodecDecodeState *state, UnpaperCudaStream *stream,
                       NvImgCodecOutputFormat output_fmt,
                       NvImgCodecDecodedImage *out) {
  (void)data;
  (void)size;
  (void)state;
  (void)stream;
  (void)output_fmt;
  (void)out;
  return false;
}

bool nvimgcodec_decode_file(const char *filename, UnpaperCudaStream *stream,
                            NvImgCodecOutputFormat output_fmt,
                            NvImgCodecDecodedImage *out) {
  (void)filename;
  (void)stream;
  (void)output_fmt;
  (void)out;
  return false;
}

int nvimgcodec_decode_batch(const uint8_t *const *data_ptrs,
                            const size_t *sizes, int batch_size,
                            NvImgCodecOutputFormat output_fmt,
                            NvImgCodecDecodedImage *outputs) {
  (void)data_ptrs;
  (void)sizes;
  (void)batch_size;
  (void)output_fmt;
  (void)outputs;
  return 0;
}

bool nvimgcodec_encode(const void *gpu_ptr, size_t pitch, int width, int height,
                       NvImgCodecEncodeInputFormat input_fmt,
                       NvImgCodecEncodeState *state, UnpaperCudaStream *stream,
                       const NvImgCodecEncodeParams *params,
                       NvImgCodecEncodedImage *out) {
  (void)gpu_ptr;
  (void)pitch;
  (void)width;
  (void)height;
  (void)input_fmt;
  (void)state;
  (void)stream;
  (void)params;
  (void)out;
  return false;
}

bool nvimgcodec_encode_jpeg(const void *gpu_ptr, size_t pitch, int width,
                            int height, NvImgCodecEncodeInputFormat input_fmt,
                            int quality, NvImgCodecEncodeState *state,
                            UnpaperCudaStream *stream,
                            NvImgCodecEncodedImage *out) {
  (void)gpu_ptr;
  (void)pitch;
  (void)width;
  (void)height;
  (void)input_fmt;
  (void)quality;
  (void)state;
  (void)stream;
  (void)out;
  return false;
}

bool nvimgcodec_encode_jp2(const void *gpu_ptr, size_t pitch, int width,
                           int height, NvImgCodecEncodeInputFormat input_fmt,
                           bool lossless, NvImgCodecEncodeState *state,
                           UnpaperCudaStream *stream,
                           NvImgCodecEncodedImage *out) {
  (void)gpu_ptr;
  (void)pitch;
  (void)width;
  (void)height;
  (void)input_fmt;
  (void)lossless;
  (void)state;
  (void)stream;
  (void)out;
  return false;
}

bool nvimgcodec_encode_to_file(const void *gpu_ptr, size_t pitch, int width,
                               int height,
                               NvImgCodecEncodeInputFormat input_fmt,
                               UnpaperCudaStream *stream,
                               const NvImgCodecEncodeParams *params,
                               const char *filename) {
  (void)gpu_ptr;
  (void)pitch;
  (void)width;
  (void)height;
  (void)input_fmt;
  (void)stream;
  (void)params;
  (void)filename;
  return false;
}

void nvimgcodec_set_jpeg_quality(int quality) { (void)quality; }

int nvimgcodec_get_jpeg_quality(void) { return 85; }

#endif // UNPAPER_WITH_NVIMGCODEC
