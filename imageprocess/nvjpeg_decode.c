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
// When using nvjpegCreateExV2 with custom allocators, nvJPEG manages internal
// buffers automatically via the custom allocators. We only need the state,
// stream, and parser.
struct NvJpegStreamState {
  nvjpegJpegState_t state;        // Decoder state (per-stream)
  nvjpegJpegStream_t jpeg_stream; // Bitstream parser (per-stream)
  cudaStream_t decode_stream; // Dedicated CUDA stream for this state (CRITICAL
                              // for parallelism)
  atomic_int in_use;          // SLOT_FREE or SLOT_IN_USE
};

// Global nvJPEG context (singleton)
typedef struct {
  nvjpegHandle_t handle;            // Global handle (one per process)
  NvJpegStreamState *stream_states; // Array[num_streams]
  int num_streams;                  // Number of stream states
  bool initialized;                 // Initialization flag

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
    return NVJPEG_OUTPUT_RGBI; // Interleaved RGB
  case NVJPEG_FMT_BGR:
    return NVJPEG_OUTPUT_BGRI; // Interleaved BGR
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

  // Create dedicated CUDA stream for this nvJPEG state
  // CRITICAL: Each nvJPEG state MUST have its own stream for true parallelism.
  // Without dedicated streams, all decodes serialize on the default stream,
  // defeating multi-stream scaling.
  cudaError_t cuda_err = cudaStreamCreate(&state->decode_stream);
  if (cuda_err != cudaSuccess) {
    verboseLog(VERBOSE_DEBUG, "nvjpeg: failed to create CUDA stream: %s\n",
               cudaGetErrorString(cuda_err));
    return false;
  }

  // Create decoder state
  // When nvjpegCreateExV2 was used with custom allocators, nvJPEG will use
  // those allocators for internal device and pinned buffers automatically.
  status = nvjpegJpegStateCreate(g_nvjpeg_ctx.handle, &state->state);
  if (status != NVJPEG_STATUS_SUCCESS) {
    cudaStreamDestroy(state->decode_stream);
    state->decode_stream = NULL;
    verboseLog(VERBOSE_DEBUG, "nvjpeg: failed to create state: %s\n",
               nvjpeg_status_string(status));
    return false;
  }

  // Create JPEG stream parser (used for phased decoding, optional for simple
  // API)
  status = nvjpegJpegStreamCreate(g_nvjpeg_ctx.handle, &state->jpeg_stream);
  if (status != NVJPEG_STATUS_SUCCESS) {
    nvjpegJpegStateDestroy(state->state);
    cudaStreamDestroy(state->decode_stream);
    state->decode_stream = NULL;
    verboseLog(VERBOSE_DEBUG, "nvjpeg: failed to create JPEG stream: %s\n",
               nvjpeg_status_string(status));
    return false;
  }

  atomic_init(&state->in_use, SLOT_FREE);

  return true;
}

static void cleanup_stream_state(NvJpegStreamState *state) {
  if (state->jpeg_stream != NULL) {
    nvjpegJpegStreamDestroy(state->jpeg_stream);
    state->jpeg_stream = NULL;
  }
  if (state->state != NULL) {
    nvjpegJpegStateDestroy(state->state);
    state->state = NULL;
  }
  // Destroy dedicated CUDA stream
  if (state->decode_stream != NULL) {
    cudaStreamDestroy(state->decode_stream);
    state->decode_stream = NULL;
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
  status =
      nvjpegSetDeviceMemoryPadding(NVJPEG_MEM_PADDING, g_nvjpeg_ctx.handle);
  if (status != NVJPEG_STATUS_SUCCESS) {
    verboseLog(VERBOSE_DEBUG,
               "nvjpeg: warning - failed to set device padding\n");
  }

  status =
      nvjpegSetPinnedMemoryPadding(NVJPEG_MEM_PADDING, g_nvjpeg_ctx.handle);
  if (status != NVJPEG_STATUS_SUCCESS) {
    verboseLog(VERBOSE_DEBUG,
               "nvjpeg: warning - failed to set pinned padding\n");
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

  // nvJPEG manages internal buffers via custom allocators, no manual buffer
  // management needed here.
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

  // CRITICAL: Use the state's dedicated stream, NOT the passed-in stream.
  // Each nvJPEG stream state has its own CUDA stream to enable true
  // parallelism. Using a shared stream would serialize all decodes, defeating
  // multi-stream scaling. The passed-in stream parameter is kept for API
  // compatibility but ignored.
  (void)stream; // Unused - using state->decode_stream instead
  cudaStream_t cuda_stream = state->decode_stream;

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

  // Synchronize to ensure decode is complete before another thread uses the
  // data This is necessary since the producer thread and worker threads are
  // different
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

  if (file_size <= 0 || file_size > (long)(1024 * 1024 * 100)) { // Max 100MB
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

// ============================================================================
// Batched Decode API (PR36A)
// ============================================================================
// This uses nvjpegDecodeBatched() for efficient parallel decoding.
// Key performance benefits:
// - Single cudaStreamSynchronize per batch (not per image)
// - nvJPEG internal GPU parallelism for batch sizes >50
// - Pre-allocated buffer pool eliminates runtime cudaMalloc

// Batched decoder context (singleton, separate from per-image decoder)
typedef struct {
  // Pre-allocated output buffer pool
  void **gpu_buffers;        // Array of GPU buffer pointers
  size_t *buffer_pitches;    // Pitch for each buffer (aligned)
  int max_batch_size;        // Number of buffers allocated
  int max_width;             // Maximum image width
  int max_height;            // Maximum image height
  NvJpegOutputFormat format; // Output format for all decodes
  size_t buffer_size;        // Size of each buffer in bytes

  // For true batched API (may not be available on all systems)
  nvjpegJpegState_t batch_state; // Dedicated state for batched decode
  cudaStream_t batch_stream;     // Dedicated CUDA stream
  bool batched_api_available;    // True if nvjpegDecodeBatched works

  // Statistics
  size_t total_batch_calls;
  size_t total_images_decoded;
  size_t failed_decodes;
  size_t max_batch_size_used;

  bool initialized;
} NvJpegBatchedContext;

static NvJpegBatchedContext g_batched_ctx = {0};
static pthread_mutex_t g_batched_mutex = PTHREAD_MUTEX_INITIALIZER;

bool nvjpeg_batched_init(int max_batch_size, int max_width, int max_height,
                         NvJpegOutputFormat format) {
  pthread_mutex_lock(&g_batched_mutex);

  // Already initialized?
  if (g_batched_ctx.initialized) {
    // Check if configuration matches
    if (g_batched_ctx.max_batch_size >= max_batch_size &&
        g_batched_ctx.max_width >= max_width &&
        g_batched_ctx.max_height >= max_height &&
        g_batched_ctx.format == format) {
      pthread_mutex_unlock(&g_batched_mutex);
      return true;
    }
    // Configuration changed, need to reinitialize
    pthread_mutex_unlock(&g_batched_mutex);
    nvjpeg_batched_cleanup();
    pthread_mutex_lock(&g_batched_mutex);
  }

  // Require base context to be initialized
  if (!g_nvjpeg_ctx.initialized) {
    verboseLog(VERBOSE_DEBUG, "nvjpeg_batched: base context not initialized\n");
    pthread_mutex_unlock(&g_batched_mutex);
    return false;
  }

  // Cap batch size
  if (max_batch_size > NVJPEG_MAX_BATCH_SIZE) {
    max_batch_size = NVJPEG_MAX_BATCH_SIZE;
  }
  if (max_batch_size < 1) {
    max_batch_size = 1;
  }

  nvjpegStatus_t status;
  g_batched_ctx.batched_api_available = false;

  // Create dedicated CUDA stream for batched operations
  cudaError_t cuda_err = cudaStreamCreate(&g_batched_ctx.batch_stream);
  if (cuda_err != cudaSuccess) {
    verboseLog(VERBOSE_DEBUG,
               "nvjpeg_batched: failed to create CUDA stream: %s\n",
               cudaGetErrorString(cuda_err));
    pthread_mutex_unlock(&g_batched_mutex);
    return false;
  }

  // Try to create state and initialize for batched decoding
  // Note: nvjpegDecodeBatched may not be supported on all systems/images
  status =
      nvjpegJpegStateCreate(g_nvjpeg_ctx.handle, &g_batched_ctx.batch_state);
  if (status == NVJPEG_STATUS_SUCCESS) {
    nvjpegOutputFormat_t nvfmt = to_nvjpeg_format(format);
    // Note: max_cpu_threads must be >= 1 (documentation says deprecated but 0
    // causes INVALID_PARAMETER)
    status = nvjpegDecodeBatchedInitialize(g_nvjpeg_ctx.handle,
                                           g_batched_ctx.batch_state,
                                           max_batch_size, 1, nvfmt);
    if (status == NVJPEG_STATUS_SUCCESS) {
      g_batched_ctx.batched_api_available = true;
      verboseLog(VERBOSE_DEBUG, "nvjpeg_batched: true batched API available\n");
    } else {
      // Batched API not available, cleanup state but continue
      // We'll use concurrent single-image decodes instead
      verboseLog(VERBOSE_DEBUG,
                 "nvjpeg_batched: batched API not available (%s), "
                 "using concurrent single-image fallback\n",
                 nvjpeg_status_string(status));
      nvjpegJpegStateDestroy(g_batched_ctx.batch_state);
      g_batched_ctx.batch_state = NULL;
    }
  } else {
    verboseLog(VERBOSE_DEBUG,
               "nvjpeg_batched: failed to create batch state: %s\n",
               nvjpeg_status_string(status));
    g_batched_ctx.batch_state = NULL;
  }

  // Calculate buffer size with 256-byte pitch alignment
  int channels = format_channels(format);
  size_t pitch = (size_t)max_width * (size_t)channels;
  pitch = (pitch + 255) & ~(size_t)255; // Align to 256 bytes
  size_t buffer_size = pitch * (size_t)max_height;

  // Allocate buffer pool
  g_batched_ctx.gpu_buffers = calloc((size_t)max_batch_size, sizeof(void *));
  g_batched_ctx.buffer_pitches = calloc((size_t)max_batch_size, sizeof(size_t));

  if (g_batched_ctx.gpu_buffers == NULL ||
      g_batched_ctx.buffer_pitches == NULL) {
    verboseLog(VERBOSE_DEBUG,
               "nvjpeg_batched: failed to allocate buffer arrays\n");
    free(g_batched_ctx.gpu_buffers);
    free(g_batched_ctx.buffer_pitches);
    if (g_batched_ctx.batch_state != NULL) {
      nvjpegJpegStateDestroy(g_batched_ctx.batch_state);
      g_batched_ctx.batch_state = NULL;
    }
    cudaStreamDestroy(g_batched_ctx.batch_stream);
    g_batched_ctx.batch_stream = NULL;
    pthread_mutex_unlock(&g_batched_mutex);
    return false;
  }

  // Pre-allocate GPU buffers for output
  for (int i = 0; i < max_batch_size; i++) {
    cuda_err = cudaMalloc(&g_batched_ctx.gpu_buffers[i], buffer_size);
    if (cuda_err != cudaSuccess) {
      verboseLog(VERBOSE_DEBUG,
                 "nvjpeg_batched: failed to allocate GPU buffer %d: %s\n", i,
                 cudaGetErrorString(cuda_err));
      // Free already allocated buffers
      for (int j = 0; j < i; j++) {
        cudaFree(g_batched_ctx.gpu_buffers[j]);
      }
      free(g_batched_ctx.gpu_buffers);
      free(g_batched_ctx.buffer_pitches);
      if (g_batched_ctx.batch_state != NULL) {
        nvjpegJpegStateDestroy(g_batched_ctx.batch_state);
        g_batched_ctx.batch_state = NULL;
      }
      cudaStreamDestroy(g_batched_ctx.batch_stream);
      g_batched_ctx.batch_stream = NULL;
      pthread_mutex_unlock(&g_batched_mutex);
      return false;
    }
    g_batched_ctx.buffer_pitches[i] = pitch;
  }

  // Store configuration
  g_batched_ctx.max_batch_size = max_batch_size;
  g_batched_ctx.max_width = max_width;
  g_batched_ctx.max_height = max_height;
  g_batched_ctx.format = format;
  g_batched_ctx.buffer_size = buffer_size;

  // Reset statistics
  g_batched_ctx.total_batch_calls = 0;
  g_batched_ctx.total_images_decoded = 0;
  g_batched_ctx.failed_decodes = 0;
  g_batched_ctx.max_batch_size_used = 0;

  g_batched_ctx.initialized = true;

  verboseLog(VERBOSE_DEBUG,
             "nvjpeg_batched: initialized with batch_size=%d, max=%dx%d, "
             "buffer_size=%zu bytes\n",
             max_batch_size, max_width, max_height, buffer_size);

  pthread_mutex_unlock(&g_batched_mutex);
  return true;
}

// Fallback: decode images using concurrent single-image nvjpegDecode calls
static int decode_batch_fallback(const uint8_t *const *jpeg_data,
                                 const size_t *jpeg_sizes, int batch_size,
                                 NvJpegDecodedImage *outputs) {
  int success_count = 0;
  int channels = format_channels(g_batched_ctx.format);
  nvjpegOutputFormat_t nvfmt = to_nvjpeg_format(g_batched_ctx.format);

  for (int i = 0; i < batch_size; i++) {
    outputs[i].gpu_ptr = NULL;
    outputs[i].pitch = 0;
    outputs[i].width = 0;
    outputs[i].height = 0;
    outputs[i].channels = 0;
    outputs[i].fmt = g_batched_ctx.format;

    if (jpeg_data[i] == NULL || jpeg_sizes[i] == 0) {
      g_batched_ctx.failed_decodes++;
      continue;
    }

    // Get image info
    int nComponents = 0;
    nvjpegChromaSubsampling_t subsampling;
    int widths[NVJPEG_MAX_COMPONENT];
    int heights[NVJPEG_MAX_COMPONENT];

    nvjpegStatus_t status =
        nvjpegGetImageInfo(g_nvjpeg_ctx.handle, jpeg_data[i], jpeg_sizes[i],
                           &nComponents, &subsampling, widths, heights);
    if (status != NVJPEG_STATUS_SUCCESS) {
      g_batched_ctx.failed_decodes++;
      continue;
    }

    int width = widths[0];
    int height = heights[0];

    if (width > g_batched_ctx.max_width || height > g_batched_ctx.max_height) {
      g_batched_ctx.failed_decodes++;
      continue;
    }

    // Acquire stream state from the per-image pool
    NvJpegStreamState *state = nvjpeg_acquire_stream_state();
    if (state == NULL) {
      g_batched_ctx.failed_decodes++;
      continue;
    }

    // Set up output pointing to pre-allocated buffer
    nvjpegImage_t nv_output = {0};
    nv_output.channel[0] = (unsigned char *)g_batched_ctx.gpu_buffers[i];
    nv_output.pitch[0] = (unsigned int)g_batched_ctx.buffer_pitches[i];

    // Decode using the simple API (each uses its own stream for parallelism)
    status =
        nvjpegDecode(g_nvjpeg_ctx.handle, state->state, jpeg_data[i],
                     jpeg_sizes[i], nvfmt, &nv_output, state->decode_stream);

    if (status == NVJPEG_STATUS_SUCCESS) {
      // Sync this specific stream
      cudaError_t sync_err = cudaStreamSynchronize(state->decode_stream);
      if (sync_err == cudaSuccess) {
        outputs[i].gpu_ptr = g_batched_ctx.gpu_buffers[i];
        outputs[i].pitch = g_batched_ctx.buffer_pitches[i];
        outputs[i].width = width;
        outputs[i].height = height;
        outputs[i].channels = channels;
        outputs[i].fmt = g_batched_ctx.format;
        success_count++;
      } else {
        g_batched_ctx.failed_decodes++;
      }
    } else {
      g_batched_ctx.failed_decodes++;
    }

    nvjpeg_release_stream_state(state);
  }

  g_batched_ctx.total_images_decoded += (size_t)success_count;

  return success_count;
}

int nvjpeg_decode_batch(const uint8_t *const *jpeg_data,
                        const size_t *jpeg_sizes, int batch_size,
                        NvJpegDecodedImage *outputs) {
  if (!g_batched_ctx.initialized) {
    verboseLog(VERBOSE_DEBUG, "nvjpeg_decode_batch: not initialized\n");
    return 0;
  }

  if (jpeg_data == NULL || jpeg_sizes == NULL || outputs == NULL ||
      batch_size <= 0) {
    return 0;
  }

  if (batch_size > g_batched_ctx.max_batch_size) {
    verboseLog(VERBOSE_DEBUG,
               "nvjpeg_decode_batch: batch_size %d exceeds max %d\n",
               batch_size, g_batched_ctx.max_batch_size);
    batch_size = g_batched_ctx.max_batch_size;
  }

  g_batched_ctx.total_batch_calls++;
  if ((size_t)batch_size > g_batched_ctx.max_batch_size_used) {
    g_batched_ctx.max_batch_size_used = (size_t)batch_size;
  }

  // If true batched API not available, use concurrent single-image fallback
  if (!g_batched_ctx.batched_api_available) {
    return decode_batch_fallback(jpeg_data, jpeg_sizes, batch_size, outputs);
  }

  // True batched API path
  nvjpegStatus_t status;
  int channels = format_channels(g_batched_ctx.format);
  int max_batch = g_batched_ctx.max_batch_size;

  // nvjpegDecodeBatched requires arrays sized to match the initialized
  // batch_size.
  nvjpegImage_t *nv_outputs = calloc((size_t)max_batch, sizeof(nvjpegImage_t));
  const uint8_t **padded_data = calloc((size_t)max_batch, sizeof(uint8_t *));
  size_t *padded_sizes = calloc((size_t)max_batch, sizeof(size_t));
  bool *valid = calloc((size_t)max_batch, sizeof(bool));

  if (nv_outputs == NULL || padded_data == NULL || padded_sizes == NULL ||
      valid == NULL) {
    free(nv_outputs);
    free(padded_data);
    free(padded_sizes);
    free(valid);
    return 0;
  }

  // Initialize output structures and prepare padded arrays
  for (int i = 0; i < batch_size; i++) {
    outputs[i].gpu_ptr = NULL;
    outputs[i].pitch = 0;
    outputs[i].width = 0;
    outputs[i].height = 0;
    outputs[i].channels = 0;
    outputs[i].fmt = g_batched_ctx.format;

    padded_data[i] = jpeg_data[i];
    padded_sizes[i] = jpeg_sizes[i];

    if (jpeg_data[i] == NULL || jpeg_sizes[i] == 0) {
      g_batched_ctx.failed_decodes++;
      continue;
    }

    int nComponents = 0;
    nvjpegChromaSubsampling_t subsampling;
    int widths[NVJPEG_MAX_COMPONENT];
    int heights[NVJPEG_MAX_COMPONENT];

    status =
        nvjpegGetImageInfo(g_nvjpeg_ctx.handle, jpeg_data[i], jpeg_sizes[i],
                           &nComponents, &subsampling, widths, heights);
    if (status != NVJPEG_STATUS_SUCCESS) {
      g_batched_ctx.failed_decodes++;
      continue;
    }

    int width = widths[0];
    int height = heights[0];

    if (width > g_batched_ctx.max_width || height > g_batched_ctx.max_height) {
      g_batched_ctx.failed_decodes++;
      continue;
    }

    nv_outputs[i].channel[0] = (unsigned char *)g_batched_ctx.gpu_buffers[i];
    nv_outputs[i].pitch[0] = (unsigned int)g_batched_ctx.buffer_pitches[i];

    outputs[i].gpu_ptr = g_batched_ctx.gpu_buffers[i];
    outputs[i].pitch = g_batched_ctx.buffer_pitches[i];
    outputs[i].width = width;
    outputs[i].height = height;
    outputs[i].channels = channels;
    outputs[i].fmt = g_batched_ctx.format;

    valid[i] = true;
  }

  // Pad remaining slots
  for (int i = batch_size; i < max_batch; i++) {
    padded_data[i] = NULL;
    padded_sizes[i] = 0;
    nv_outputs[i].channel[0] = (unsigned char *)g_batched_ctx.gpu_buffers[i];
    nv_outputs[i].pitch[0] = (unsigned int)g_batched_ctx.buffer_pitches[i];
    valid[i] = false;
  }

  // Perform batched decode
  status = nvjpegDecodeBatched(g_nvjpeg_ctx.handle, g_batched_ctx.batch_state,
                               padded_data, padded_sizes, nv_outputs,
                               g_batched_ctx.batch_stream);

  if (status != NVJPEG_STATUS_SUCCESS) {
    verboseLog(VERBOSE_DEBUG,
               "nvjpeg_decode_batch: nvjpegDecodeBatched failed: %s, "
               "falling back to single-image decode\n",
               nvjpeg_status_string(status));
    // Batched decode failed (e.g., image not supported) - fall back to
    // single-image
    free(valid);
    free(nv_outputs);
    free(padded_data);
    free(padded_sizes);
    return decode_batch_fallback(jpeg_data, jpeg_sizes, batch_size, outputs);
  }

  // Single sync for entire batch
  cudaError_t sync_err = cudaStreamSynchronize(g_batched_ctx.batch_stream);
  if (sync_err != cudaSuccess) {
    for (int i = 0; i < batch_size; i++) {
      if (valid[i]) {
        outputs[i].gpu_ptr = NULL;
        g_batched_ctx.failed_decodes++;
      }
    }
    free(valid);
    free(nv_outputs);
    free(padded_data);
    free(padded_sizes);
    return 0;
  }

  int success_count = 0;
  for (int i = 0; i < batch_size; i++) {
    if (valid[i]) {
      success_count++;
    }
  }

  g_batched_ctx.total_images_decoded += (size_t)success_count;

  free(valid);
  free(nv_outputs);
  free(padded_data);
  free(padded_sizes);

  return success_count;
}

bool nvjpeg_batched_is_ready(void) {
  pthread_mutex_lock(&g_batched_mutex);
  bool ready = g_batched_ctx.initialized;
  pthread_mutex_unlock(&g_batched_mutex);
  return ready;
}

NvJpegOutputFormat nvjpeg_batched_get_format(void) {
  if (!g_batched_ctx.initialized) {
    return NVJPEG_FMT_RGB; // Default
  }
  return g_batched_ctx.format;
}

int nvjpeg_batched_get_max_batch_size(void) {
  if (!g_batched_ctx.initialized) {
    return 0;
  }
  return g_batched_ctx.max_batch_size;
}

void nvjpeg_batched_cleanup(void) {
  pthread_mutex_lock(&g_batched_mutex);

  if (!g_batched_ctx.initialized) {
    pthread_mutex_unlock(&g_batched_mutex);
    return;
  }

  // Free GPU buffer pool
  if (g_batched_ctx.gpu_buffers != NULL) {
    for (int i = 0; i < g_batched_ctx.max_batch_size; i++) {
      if (g_batched_ctx.gpu_buffers[i] != NULL) {
        cudaFree(g_batched_ctx.gpu_buffers[i]);
      }
    }
    free(g_batched_ctx.gpu_buffers);
    g_batched_ctx.gpu_buffers = NULL;
  }

  if (g_batched_ctx.buffer_pitches != NULL) {
    free(g_batched_ctx.buffer_pitches);
    g_batched_ctx.buffer_pitches = NULL;
  }

  // Destroy batch state (if batched API was available)
  if (g_batched_ctx.batch_state != NULL) {
    nvjpegJpegStateDestroy(g_batched_ctx.batch_state);
    g_batched_ctx.batch_state = NULL;
  }

  // Destroy stream
  if (g_batched_ctx.batch_stream != NULL) {
    cudaStreamDestroy(g_batched_ctx.batch_stream);
    g_batched_ctx.batch_stream = NULL;
  }

  g_batched_ctx.max_batch_size = 0;
  g_batched_ctx.max_width = 0;
  g_batched_ctx.max_height = 0;
  g_batched_ctx.buffer_size = 0;
  g_batched_ctx.batched_api_available = false;
  g_batched_ctx.initialized = false;

  verboseLog(VERBOSE_DEBUG, "nvjpeg_batched: cleaned up\n");

  pthread_mutex_unlock(&g_batched_mutex);
}

NvJpegBatchStats nvjpeg_batched_get_stats(void) {
  NvJpegBatchStats stats = {0};

  if (!g_batched_ctx.initialized) {
    return stats;
  }

  stats.total_batch_calls = g_batched_ctx.total_batch_calls;
  stats.total_images_decoded = g_batched_ctx.total_images_decoded;
  stats.failed_decodes = g_batched_ctx.failed_decodes;
  stats.max_batch_size_used = g_batched_ctx.max_batch_size_used;

  return stats;
}

// ============================================================================
// Internal API for encoder context access (PR37)
// ============================================================================

// Get the shared nvJPEG handle for use by nvjpeg_encode.c
// This avoids creating multiple handles which wastes GPU resources.
nvjpegHandle_t nvjpeg_get_handle_internal(void) {
  if (!g_nvjpeg_ctx.initialized) {
    return NULL;
  }
  return g_nvjpeg_ctx.handle;
}

#else // !UNPAPER_WITH_CUDA

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

// Batched decode stubs for non-CUDA builds
bool nvjpeg_batched_init(int max_batch_size, int max_width, int max_height,
                         NvJpegOutputFormat format) {
  (void)max_batch_size;
  (void)max_width;
  (void)max_height;
  (void)format;
  return false;
}

int nvjpeg_decode_batch(const uint8_t *const *jpeg_data,
                        const size_t *jpeg_sizes, int batch_size,
                        NvJpegDecodedImage *outputs) {
  (void)jpeg_data;
  (void)jpeg_sizes;
  (void)batch_size;
  (void)outputs;
  return 0;
}

bool nvjpeg_batched_is_ready(void) { return false; }

NvJpegOutputFormat nvjpeg_batched_get_format(void) { return NVJPEG_FMT_RGB; }

int nvjpeg_batched_get_max_batch_size(void) { return 0; }

void nvjpeg_batched_cleanup(void) {}

NvJpegBatchStats nvjpeg_batched_get_stats(void) {
  NvJpegBatchStats stats = {0};
  return stats;
}

#endif // UNPAPER_WITH_CUDA
