// SPDX-FileCopyrightText: 2025 The unpaper authors
//
// SPDX-License-Identifier: GPL-2.0-only

#include "imageprocess/nvjpeg_decode.h"

#if defined(UNPAPER_WITH_CUDA) && (UNPAPER_WITH_CUDA)

#include <cuda_runtime.h>
#include <nvjpeg.h>
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

// Memory padding to reduce reallocations (1MB)
#define NVJPEG_MEM_PADDING (1024 * 1024)

// Per-stream state for nvJPEG decode operations
struct NvJpegStreamState {
  nvjpegJpegState_t state;          // Decoder state (per-stream)
  nvjpegBufferDevice_t dev_buffer;  // GPU output buffer (per-stream)
  nvjpegBufferPinned_t pin_buffer[2];  // Pinned staging (double-buffer)
  nvjpegJpegStream_t jpeg_stream;   // Bitstream parser (per-stream)
  int current_pin_buffer;           // Toggle for double-buffering
  atomic_int in_use;                // SLOT_FREE or SLOT_IN_USE
};

// Global nvJPEG context (singleton)
typedef struct {
  nvjpegHandle_t handle;              // Global handle (one per process)
  NvJpegStreamState *stream_states;   // Array[num_streams]
  int num_streams;                    // Number of stream states
  bool initialized;                   // Initialization flag

  // Statistics (atomic for thread safety)
  atomic_size_t total_decodes;
  atomic_size_t successful_decodes;
  atomic_size_t fallback_decodes;
  atomic_size_t current_in_use;
  atomic_size_t concurrent_peak;
} NvJpegContext;

static NvJpegContext g_nvjpeg_ctx = {0};
static pthread_mutex_t g_init_mutex = PTHREAD_MUTEX_INITIALIZER;

// ============================================================================
// Custom Allocators for Stream-Ordered Memory
// ============================================================================
// CRITICAL: These allocators use cudaMallocAsync/cudaFreeAsync to avoid
// serialization across CUDA streams. Without this, nvJPEG's internal
// cudaMalloc calls would block ALL streams until complete.

static int nvjpeg_dev_malloc(void *ctx, void **ptr, size_t size,
                             cudaStream_t stream) {
  (void)ctx;
  // Use stream-ordered allocation (CUDA 11.2+)
  // This does NOT serialize across streams
  cudaError_t err = cudaMallocAsync(ptr, size, stream);
  if (err != cudaSuccess) {
    // Fallback to synchronous allocation
    err = cudaMalloc(ptr, size);
  }
  return (err == cudaSuccess) ? 0 : -1;
}

static int nvjpeg_dev_free(void *ctx, void *ptr, size_t size,
                           cudaStream_t stream) {
  (void)ctx;
  (void)size;
  if (ptr == NULL) {
    return 0;
  }
  // Use stream-ordered free
  cudaError_t err = cudaFreeAsync(ptr, stream);
  if (err != cudaSuccess) {
    // Fallback to synchronous free
    err = cudaFree(ptr);
  }
  return (err == cudaSuccess) ? 0 : -1;
}

static int nvjpeg_pinned_malloc(void *ctx, void **ptr, size_t size,
                                cudaStream_t stream) {
  (void)ctx;
  (void)stream;
  // Pinned memory allocation is less critical for scaling
  // but we still want to avoid excessive allocations
  cudaError_t err = cudaMallocHost(ptr, size);
  return (err == cudaSuccess) ? 0 : -1;
}

static int nvjpeg_pinned_free(void *ctx, void *ptr, size_t size,
                              cudaStream_t stream) {
  (void)ctx;
  (void)size;
  (void)stream;
  if (ptr == NULL) {
    return 0;
  }
  cudaError_t err = cudaFreeHost(ptr);
  return (err == cudaSuccess) ? 0 : -1;
}

// ============================================================================
// Helper Functions
// ============================================================================

static const char *nvjpeg_status_string(nvjpegStatus_t status) {
  switch (status) {
  case NVJPEG_STATUS_SUCCESS:
    return "Success";
  case NVJPEG_STATUS_NOT_INITIALIZED:
    return "Not initialized";
  case NVJPEG_STATUS_INVALID_PARAMETER:
    return "Invalid parameter";
  case NVJPEG_STATUS_BAD_JPEG:
    return "Bad JPEG";
  case NVJPEG_STATUS_JPEG_NOT_SUPPORTED:
    return "JPEG not supported";
  case NVJPEG_STATUS_ALLOCATOR_FAILURE:
    return "Allocator failure";
  case NVJPEG_STATUS_EXECUTION_FAILED:
    return "Execution failed";
  case NVJPEG_STATUS_ARCH_MISMATCH:
    return "Architecture mismatch";
  case NVJPEG_STATUS_INTERNAL_ERROR:
    return "Internal error";
  case NVJPEG_STATUS_IMPLEMENTATION_NOT_SUPPORTED:
    return "Implementation not supported";
  default:
    return "Unknown error";
  }
}

static nvjpegOutputFormat_t to_nvjpeg_format(NvJpegOutputFormat fmt) {
  switch (fmt) {
  case NVJPEG_FMT_GRAY8:
    return NVJPEG_OUTPUT_Y;
  case NVJPEG_FMT_RGB:
    return NVJPEG_OUTPUT_RGBI;  // Interleaved RGB
  case NVJPEG_FMT_BGR:
    return NVJPEG_OUTPUT_BGRI;  // Interleaved BGR
  default:
    return NVJPEG_OUTPUT_RGBI;
  }
}

static int format_channels(NvJpegOutputFormat fmt) {
  switch (fmt) {
  case NVJPEG_FMT_GRAY8:
    return 1;
  case NVJPEG_FMT_RGB:
  case NVJPEG_FMT_BGR:
    return 3;
  default:
    return 3;
  }
}

// Update peak concurrent usage atomically
static void update_peak_usage(void) {
  size_t in_use = atomic_load(&g_nvjpeg_ctx.current_in_use);
  size_t peak = atomic_load(&g_nvjpeg_ctx.concurrent_peak);
  while (in_use > peak) {
    if (atomic_compare_exchange_weak(&g_nvjpeg_ctx.concurrent_peak, &peak,
                                     in_use)) {
      break;
    }
  }
}

// ============================================================================
// Stream State Management
// ============================================================================

static bool init_stream_state(NvJpegStreamState *state) {
  nvjpegStatus_t status;

  // Create decoder state
  status = nvjpegJpegStateCreate(g_nvjpeg_ctx.handle, &state->state);
  if (status != NVJPEG_STATUS_SUCCESS) {
    verboseLog(VERBOSE_DEBUG, "nvjpeg: failed to create state: %s\n",
               nvjpeg_status_string(status));
    return false;
  }

  // Create device buffer
  status = nvjpegBufferDeviceCreate(g_nvjpeg_ctx.handle, NULL,
                                    &state->dev_buffer);
  if (status != NVJPEG_STATUS_SUCCESS) {
    nvjpegJpegStateDestroy(state->state);
    verboseLog(VERBOSE_DEBUG, "nvjpeg: failed to create device buffer: %s\n",
               nvjpeg_status_string(status));
    return false;
  }

  // Create pinned buffers (double-buffer for async)
  for (int i = 0; i < 2; i++) {
    status = nvjpegBufferPinnedCreate(g_nvjpeg_ctx.handle, NULL,
                                      &state->pin_buffer[i]);
    if (status != NVJPEG_STATUS_SUCCESS) {
      nvjpegBufferDeviceDestroy(state->dev_buffer);
      nvjpegJpegStateDestroy(state->state);
      for (int j = 0; j < i; j++) {
        nvjpegBufferPinnedDestroy(state->pin_buffer[j]);
      }
      verboseLog(VERBOSE_DEBUG, "nvjpeg: failed to create pinned buffer: %s\n",
                 nvjpeg_status_string(status));
      return false;
    }
  }

  // Create JPEG stream parser
  status = nvjpegJpegStreamCreate(g_nvjpeg_ctx.handle, &state->jpeg_stream);
  if (status != NVJPEG_STATUS_SUCCESS) {
    nvjpegBufferDeviceDestroy(state->dev_buffer);
    for (int i = 0; i < 2; i++) {
      nvjpegBufferPinnedDestroy(state->pin_buffer[i]);
    }
    nvjpegJpegStateDestroy(state->state);
    verboseLog(VERBOSE_DEBUG, "nvjpeg: failed to create JPEG stream: %s\n",
               nvjpeg_status_string(status));
    return false;
  }

  // Attach buffers to state
  status = nvjpegStateAttachDeviceBuffer(state->state, state->dev_buffer);
  if (status != NVJPEG_STATUS_SUCCESS) {
    verboseLog(VERBOSE_DEBUG, "nvjpeg: failed to attach device buffer: %s\n",
               nvjpeg_status_string(status));
  }

  status = nvjpegStateAttachPinnedBuffer(state->state, state->pin_buffer[0]);
  if (status != NVJPEG_STATUS_SUCCESS) {
    verboseLog(VERBOSE_DEBUG, "nvjpeg: failed to attach pinned buffer: %s\n",
               nvjpeg_status_string(status));
  }

  state->current_pin_buffer = 0;
  atomic_init(&state->in_use, SLOT_FREE);

  return true;
}

static void cleanup_stream_state(NvJpegStreamState *state) {
  if (state->jpeg_stream != NULL) {
    nvjpegJpegStreamDestroy(state->jpeg_stream);
    state->jpeg_stream = NULL;
  }
  for (int i = 0; i < 2; i++) {
    if (state->pin_buffer[i] != NULL) {
      nvjpegBufferPinnedDestroy(state->pin_buffer[i]);
      state->pin_buffer[i] = NULL;
    }
  }
  if (state->dev_buffer != NULL) {
    nvjpegBufferDeviceDestroy(state->dev_buffer);
    state->dev_buffer = NULL;
  }
  if (state->state != NULL) {
    nvjpegJpegStateDestroy(state->state);
    state->state = NULL;
  }
}

// ============================================================================
// Global Context Management
// ============================================================================

bool nvjpeg_context_init(int num_streams) {
  pthread_mutex_lock(&g_init_mutex);

  if (g_nvjpeg_ctx.initialized) {
    pthread_mutex_unlock(&g_init_mutex);
    return true;
  }

  // Ensure CUDA is initialized
  UnpaperCudaInitStatus cuda_status = unpaper_cuda_try_init();
  if (cuda_status != UNPAPER_CUDA_INIT_OK) {
    verboseLog(VERBOSE_DEBUG, "nvjpeg: CUDA not available: %s\n",
               unpaper_cuda_init_status_string(cuda_status));
    pthread_mutex_unlock(&g_init_mutex);
    return false;
  }

  nvjpegStatus_t status;

  // Set up custom allocators for stream-ordered memory
  // CRITICAL: This enables linear scaling across streams
  nvjpegDevAllocatorV2_t dev_allocator = {
      .dev_ctx = NULL,
      .dev_malloc = nvjpeg_dev_malloc,
      .dev_free = nvjpeg_dev_free,
  };

  nvjpegPinnedAllocatorV2_t pinned_allocator = {
      .pinned_ctx = NULL,
      .pinned_malloc = nvjpeg_pinned_malloc,
      .pinned_free = nvjpeg_pinned_free,
  };

  // Create nvJPEG handle with custom allocators
  // MUST use nvjpegCreateExV2 for custom allocators!
  status = nvjpegCreateExV2(NVJPEG_BACKEND_GPU_HYBRID, &dev_allocator,
                            &pinned_allocator, NVJPEG_FLAGS_DEFAULT,
                            &g_nvjpeg_ctx.handle);
  if (status != NVJPEG_STATUS_SUCCESS) {
    verboseLog(VERBOSE_DEBUG, "nvjpeg: failed to create handle: %s\n",
               nvjpeg_status_string(status));
    pthread_mutex_unlock(&g_init_mutex);
    return false;
  }

  // Set memory padding to reduce reallocations
  status = nvjpegSetDeviceMemoryPadding(NVJPEG_MEM_PADDING, g_nvjpeg_ctx.handle);
  if (status != NVJPEG_STATUS_SUCCESS) {
    verboseLog(VERBOSE_DEBUG, "nvjpeg: warning - failed to set device padding\n");
  }

  status = nvjpegSetPinnedMemoryPadding(NVJPEG_MEM_PADDING, g_nvjpeg_ctx.handle);
  if (status != NVJPEG_STATUS_SUCCESS) {
    verboseLog(VERBOSE_DEBUG, "nvjpeg: warning - failed to set pinned padding\n");
  }

  // Allocate stream state pool
  g_nvjpeg_ctx.num_streams = num_streams;
  g_nvjpeg_ctx.stream_states =
      calloc((size_t)num_streams, sizeof(NvJpegStreamState));
  if (g_nvjpeg_ctx.stream_states == NULL) {
    nvjpegDestroy(g_nvjpeg_ctx.handle);
    g_nvjpeg_ctx.handle = NULL;
    pthread_mutex_unlock(&g_init_mutex);
    return false;
  }

  // Initialize each stream state
  for (int i = 0; i < num_streams; i++) {
    if (!init_stream_state(&g_nvjpeg_ctx.stream_states[i])) {
      // Cleanup already initialized states
      for (int j = 0; j < i; j++) {
        cleanup_stream_state(&g_nvjpeg_ctx.stream_states[j]);
      }
      free(g_nvjpeg_ctx.stream_states);
      g_nvjpeg_ctx.stream_states = NULL;
      nvjpegDestroy(g_nvjpeg_ctx.handle);
      g_nvjpeg_ctx.handle = NULL;
      pthread_mutex_unlock(&g_init_mutex);
      return false;
    }
  }

  // Initialize statistics
  atomic_init(&g_nvjpeg_ctx.total_decodes, 0);
  atomic_init(&g_nvjpeg_ctx.successful_decodes, 0);
  atomic_init(&g_nvjpeg_ctx.fallback_decodes, 0);
  atomic_init(&g_nvjpeg_ctx.current_in_use, 0);
  atomic_init(&g_nvjpeg_ctx.concurrent_peak, 0);

  g_nvjpeg_ctx.initialized = true;

  verboseLog(VERBOSE_DEBUG, "nvjpeg: initialized with %d stream states\n",
             num_streams);

  pthread_mutex_unlock(&g_init_mutex);
  return true;
}

void nvjpeg_context_cleanup(void) {
  pthread_mutex_lock(&g_init_mutex);

  if (!g_nvjpeg_ctx.initialized) {
    pthread_mutex_unlock(&g_init_mutex);
    return;
  }

  // Cleanup stream states
  if (g_nvjpeg_ctx.stream_states != NULL) {
    for (int i = 0; i < g_nvjpeg_ctx.num_streams; i++) {
      cleanup_stream_state(&g_nvjpeg_ctx.stream_states[i]);
    }
    free(g_nvjpeg_ctx.stream_states);
    g_nvjpeg_ctx.stream_states = NULL;
  }

  // Cleanup global resources
  if (g_nvjpeg_ctx.handle != NULL) {
    nvjpegDestroy(g_nvjpeg_ctx.handle);
    g_nvjpeg_ctx.handle = NULL;
  }

  g_nvjpeg_ctx.num_streams = 0;
  g_nvjpeg_ctx.initialized = false;

  pthread_mutex_unlock(&g_init_mutex);
}

bool nvjpeg_is_available(void) {
  pthread_mutex_lock(&g_init_mutex);
  bool available = g_nvjpeg_ctx.initialized;
  pthread_mutex_unlock(&g_init_mutex);
  return available;
}

NvJpegStats nvjpeg_get_stats(void) {
  NvJpegStats stats = {0};
  if (!g_nvjpeg_ctx.initialized) {
    return stats;
  }

  stats.total_decodes = atomic_load(&g_nvjpeg_ctx.total_decodes);
  stats.successful_decodes = atomic_load(&g_nvjpeg_ctx.successful_decodes);
  stats.fallback_decodes = atomic_load(&g_nvjpeg_ctx.fallback_decodes);
  stats.current_in_use = atomic_load(&g_nvjpeg_ctx.current_in_use);
  stats.concurrent_peak = atomic_load(&g_nvjpeg_ctx.concurrent_peak);
  stats.stream_state_count = (size_t)g_nvjpeg_ctx.num_streams;

  return stats;
}

void nvjpeg_print_stats(void) {
  NvJpegStats stats = nvjpeg_get_stats();

  double success_rate = 0.0;
  if (stats.total_decodes > 0) {
    success_rate =
        100.0 * (double)stats.successful_decodes / (double)stats.total_decodes;
  }

  fprintf(stderr,
          "nvJPEG Decode Statistics:\n"
          "  Stream states: %zu\n"
          "  Total decodes: %zu\n"
          "  Successful: %zu (%.1f%%)\n"
          "  Fallbacks: %zu\n"
          "  Peak concurrent: %zu\n",
          stats.stream_state_count, stats.total_decodes,
          stats.successful_decodes, success_rate, stats.fallback_decodes,
          stats.concurrent_peak);
}

// ============================================================================
// Stream State Pool
// ============================================================================

NvJpegStreamState *nvjpeg_acquire_stream_state(void) {
  if (!g_nvjpeg_ctx.initialized || g_nvjpeg_ctx.stream_states == NULL) {
    return NULL;
  }

  // Lock-free acquisition
  for (int i = 0; i < g_nvjpeg_ctx.num_streams; i++) {
    int expected = SLOT_FREE;
    if (atomic_compare_exchange_strong(&g_nvjpeg_ctx.stream_states[i].in_use,
                                       &expected, SLOT_IN_USE)) {
      atomic_fetch_add(&g_nvjpeg_ctx.current_in_use, 1);
      update_peak_usage();
      return &g_nvjpeg_ctx.stream_states[i];
    }
  }

  // All states in use
  return NULL;
}

void nvjpeg_release_stream_state(NvJpegStreamState *state) {
  if (state == NULL) {
    return;
  }

  // Toggle pinned buffer for next decode
  state->current_pin_buffer = 1 - state->current_pin_buffer;
  nvjpegStateAttachPinnedBuffer(state->state,
                                state->pin_buffer[state->current_pin_buffer]);

  atomic_store(&state->in_use, SLOT_FREE);
  atomic_fetch_sub(&g_nvjpeg_ctx.current_in_use, 1);
}

// ============================================================================
// JPEG Decode Operations
// ============================================================================

bool nvjpeg_get_image_info(const uint8_t *jpeg_data, size_t jpeg_size,
                           int *width, int *height, int *channels) {
  if (!g_nvjpeg_ctx.initialized || jpeg_data == NULL || jpeg_size == 0) {
    return false;
  }

  int nComponents = 0;
  nvjpegChromaSubsampling_t subsampling;
  int widths[NVJPEG_MAX_COMPONENT];
  int heights[NVJPEG_MAX_COMPONENT];

  nvjpegStatus_t status =
      nvjpegGetImageInfo(g_nvjpeg_ctx.handle, jpeg_data, jpeg_size,
                         &nComponents, &subsampling, widths, heights);
  if (status != NVJPEG_STATUS_SUCCESS) {
    return false;
  }

  if (width != NULL) {
    *width = widths[0];
  }
  if (height != NULL) {
    *height = heights[0];
  }
  if (channels != NULL) {
    *channels = nComponents;
  }

  return true;
}

bool nvjpeg_decode_to_gpu(const uint8_t *jpeg_data, size_t jpeg_size,
                          NvJpegStreamState *state, UnpaperCudaStream *stream,
                          NvJpegOutputFormat output_fmt,
                          NvJpegDecodedImage *out) {
  if (!g_nvjpeg_ctx.initialized || jpeg_data == NULL || jpeg_size == 0 ||
      state == NULL || out == NULL) {
    return false;
  }

  atomic_fetch_add(&g_nvjpeg_ctx.total_decodes, 1);

  // Get CUDA stream handle
  cudaStream_t cuda_stream =
      (cudaStream_t)unpaper_cuda_stream_get_raw_handle(stream);

  nvjpegStatus_t status;

  // Get image info first
  int nComponents = 0;
  nvjpegChromaSubsampling_t subsampling;
  int widths[NVJPEG_MAX_COMPONENT];
  int heights[NVJPEG_MAX_COMPONENT];

  status = nvjpegGetImageInfo(g_nvjpeg_ctx.handle, jpeg_data, jpeg_size,
                              &nComponents, &subsampling, widths, heights);
  if (status != NVJPEG_STATUS_SUCCESS) {
    verboseLog(VERBOSE_DEBUG, "nvjpeg: failed to get image info: %s\n",
               nvjpeg_status_string(status));
    atomic_fetch_add(&g_nvjpeg_ctx.fallback_decodes, 1);
    return false;
  }

  int width = widths[0];
  int height = heights[0];
  int out_channels = format_channels(output_fmt);
  nvjpegOutputFormat_t nvfmt = to_nvjpeg_format(output_fmt);

  // Prepare output image structure
  nvjpegImage_t output_image;
  memset(&output_image, 0, sizeof(output_image));

  // For interleaved formats (RGBI, BGRI, Y), we use channel[0] only
  size_t pitch = (size_t)width * (size_t)out_channels;
  // Align pitch to 256 bytes for better memory access patterns
  pitch = (pitch + 255) & ~(size_t)255;

  size_t image_size = pitch * (size_t)height;

  // Allocate output buffer using stream-ordered allocation
  void *output_ptr = NULL;
  cudaError_t cuda_err = cudaMallocAsync(&output_ptr, image_size, cuda_stream);
  if (cuda_err != cudaSuccess) {
    // Fallback to synchronous allocation
    cuda_err = cudaMalloc(&output_ptr, image_size);
    if (cuda_err != cudaSuccess) {
      verboseLog(VERBOSE_DEBUG, "nvjpeg: failed to allocate output buffer\n");
      atomic_fetch_add(&g_nvjpeg_ctx.fallback_decodes, 1);
      return false;
    }
  }

  output_image.channel[0] = (unsigned char *)output_ptr;
  output_image.pitch[0] = (unsigned int)pitch;

  // Use the simple decode API which handles all phases internally
  // This is more robust and well-tested than the phased API
  status = nvjpegDecode(g_nvjpeg_ctx.handle, state->state, jpeg_data, jpeg_size,
                        nvfmt, &output_image, cuda_stream);
  if (status != NVJPEG_STATUS_SUCCESS) {
    verboseLog(VERBOSE_DEBUG, "nvjpeg: decode failed: %s\n",
               nvjpeg_status_string(status));
    cudaFreeAsync(output_ptr, cuda_stream);
    atomic_fetch_add(&g_nvjpeg_ctx.fallback_decodes, 1);
    return false;
  }

  // Synchronize to ensure decode is complete before another thread uses the data
  // This is necessary since the producer thread and worker threads are different
  cudaError_t sync_err = cudaStreamSynchronize(cuda_stream);
  if (sync_err != cudaSuccess) {
    verboseLog(VERBOSE_DEBUG, "nvjpeg: stream sync failed\n");
    cudaFreeAsync(output_ptr, cuda_stream);
    atomic_fetch_add(&g_nvjpeg_ctx.fallback_decodes, 1);
    return false;
  }

  // Fill output structure
  out->gpu_ptr = output_ptr;
  out->pitch = pitch;
  out->width = width;
  out->height = height;
  out->channels = out_channels;
  out->fmt = output_fmt;

  atomic_fetch_add(&g_nvjpeg_ctx.successful_decodes, 1);

  return true;
}

bool nvjpeg_decode_file_to_gpu(const char *filename, UnpaperCudaStream *stream,
                               NvJpegOutputFormat output_fmt,
                               NvJpegDecodedImage *out) {
  if (filename == NULL || out == NULL) {
    return false;
  }

  // Open and read file
  FILE *f = fopen(filename, "rb");
  if (f == NULL) {
    verboseLog(VERBOSE_DEBUG, "nvjpeg: failed to open file: %s\n", filename);
    return false;
  }

  // Get file size
  fseek(f, 0, SEEK_END);
  long file_size = ftell(f);
  fseek(f, 0, SEEK_SET);

  if (file_size <= 0 || file_size > (long)(1024 * 1024 * 100)) {  // Max 100MB
    fclose(f);
    verboseLog(VERBOSE_DEBUG, "nvjpeg: invalid file size: %s\n", filename);
    return false;
  }

  // Allocate buffer and read file
  uint8_t *jpeg_data = malloc((size_t)file_size);
  if (jpeg_data == NULL) {
    fclose(f);
    return false;
  }

  size_t bytes_read = fread(jpeg_data, 1, (size_t)file_size, f);
  fclose(f);

  if (bytes_read != (size_t)file_size) {
    free(jpeg_data);
    return false;
  }

  // Acquire stream state
  NvJpegStreamState *state = nvjpeg_acquire_stream_state();
  if (state == NULL) {
    free(jpeg_data);
    verboseLog(VERBOSE_DEBUG, "nvjpeg: no stream state available\n");
    return false;
  }

  // Decode
  bool result = nvjpeg_decode_to_gpu(jpeg_data, (size_t)file_size, state,
                                     stream, output_fmt, out);

  // Release state and free buffer
  nvjpeg_release_stream_state(state);
  free(jpeg_data);

  return result;
}

#else  // !UNPAPER_WITH_CUDA

// Stub implementations for non-CUDA builds

bool nvjpeg_context_init(int num_streams) {
  (void)num_streams;
  return false;
}

void nvjpeg_context_cleanup(void) {}

bool nvjpeg_is_available(void) { return false; }

NvJpegStats nvjpeg_get_stats(void) {
  NvJpegStats stats = {0};
  return stats;
}

void nvjpeg_print_stats(void) {}

NvJpegStreamState *nvjpeg_acquire_stream_state(void) { return NULL; }

void nvjpeg_release_stream_state(NvJpegStreamState *state) { (void)state; }

bool nvjpeg_get_image_info(const uint8_t *jpeg_data, size_t jpeg_size,
                           int *width, int *height, int *channels) {
  (void)jpeg_data;
  (void)jpeg_size;
  (void)width;
  (void)height;
  (void)channels;
  return false;
}

bool nvjpeg_decode_to_gpu(const uint8_t *jpeg_data, size_t jpeg_size,
                          NvJpegStreamState *state, UnpaperCudaStream *stream,
                          NvJpegOutputFormat output_fmt,
                          NvJpegDecodedImage *out) {
  (void)jpeg_data;
  (void)jpeg_size;
  (void)state;
  (void)stream;
  (void)output_fmt;
  (void)out;
  return false;
}

bool nvjpeg_decode_file_to_gpu(const char *filename, UnpaperCudaStream *stream,
                               NvJpegOutputFormat output_fmt,
                               NvJpegDecodedImage *out) {
  (void)filename;
  (void)stream;
  (void)output_fmt;
  (void)out;
  return false;
}

#endif  // UNPAPER_WITH_CUDA
