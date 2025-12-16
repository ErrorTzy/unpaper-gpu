// SPDX-FileCopyrightText: 2025 The unpaper authors
//
// SPDX-License-Identifier: GPL-2.0-only

#include "imageprocess/nvjpeg_encode.h"

#if defined(UNPAPER_WITH_CUDA) && (UNPAPER_WITH_CUDA)

#include <cuda_runtime.h>
#include <nvjpeg.h>
#include <pthread.h>
#include <stdatomic.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "imageprocess/cuda_runtime.h"
#include "imageprocess/nvjpeg_decode.h" // For shared nvjpeg handle
#include "lib/logging.h"

// Slot states for lock-free pool
#define SLOT_FREE 0
#define SLOT_IN_USE 1

// Initial bitstream buffer size (4MB - enough for most images)
#define INITIAL_BITSTREAM_SIZE (4 * 1024 * 1024)

// Per-stream encoder state
struct NvJpegEncoderState {
  nvjpegEncoderState_t state;   // nvJPEG encoder state
  nvjpegEncoderParams_t params; // Encoder parameters
  cudaStream_t encode_stream;   // Dedicated CUDA stream
  atomic_int in_use;            // SLOT_FREE or SLOT_IN_USE
};

// Global encoder context
typedef struct {
  nvjpegHandle_t handle;               // Shared with decode (from nvjpeg_decode.c)
  NvJpegEncoderState *encoder_states;  // Array of encoder states
  int num_encoders;                    // Number of encoder states
  int quality;                         // JPEG quality (1-100)
  NvJpegEncodeSubsampling subsampling; // Chroma subsampling
  bool initialized;                    // Initialization flag

  // Statistics (atomic for thread safety)
  atomic_size_t total_encodes;
  atomic_size_t successful_encodes;
  atomic_size_t failed_encodes;
  atomic_size_t total_bytes_out;
  atomic_size_t current_in_use;
  atomic_size_t concurrent_peak;
} NvJpegEncodeContext;

static NvJpegEncodeContext g_encode_ctx = {0};
static pthread_mutex_t g_encode_init_mutex = PTHREAD_MUTEX_INITIALIZER;

// External handle from nvjpeg_decode.c (shared for efficiency)
extern nvjpegHandle_t nvjpeg_get_shared_handle(void);

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

static nvjpegInputFormat_t to_nvjpeg_input_format(NvJpegEncodeFormat fmt) {
  switch (fmt) {
  case NVJPEG_ENC_FMT_GRAY8:
    return NVJPEG_INPUT_RGB; // Will convert to grayscale internally
  case NVJPEG_ENC_FMT_RGB:
    return NVJPEG_INPUT_RGBI; // Interleaved RGB
  case NVJPEG_ENC_FMT_BGR:
    return NVJPEG_INPUT_BGRI; // Interleaved BGR
  default:
    return NVJPEG_INPUT_RGBI;
  }
}

static nvjpegChromaSubsampling_t
to_nvjpeg_subsampling(NvJpegEncodeSubsampling sub) {
  switch (sub) {
  case NVJPEG_ENC_SUBSAMPLING_444:
    return NVJPEG_CSS_444;
  case NVJPEG_ENC_SUBSAMPLING_422:
    return NVJPEG_CSS_422;
  case NVJPEG_ENC_SUBSAMPLING_420:
    return NVJPEG_CSS_420;
  case NVJPEG_ENC_SUBSAMPLING_GRAY:
    return NVJPEG_CSS_GRAY;
  default:
    return NVJPEG_CSS_420;
  }
}

// Update peak concurrent usage atomically
static void update_encode_peak_usage(void) {
  size_t in_use = atomic_load(&g_encode_ctx.current_in_use);
  size_t peak = atomic_load(&g_encode_ctx.concurrent_peak);
  while (in_use > peak) {
    if (atomic_compare_exchange_weak(&g_encode_ctx.concurrent_peak, &peak,
                                     in_use)) {
      break;
    }
  }
}

// ============================================================================
// Encoder State Management
// ============================================================================

static bool init_encoder_state(NvJpegEncoderState *enc_state) {
  nvjpegStatus_t status;

  // Create dedicated CUDA stream for this encoder
  cudaError_t cuda_err = cudaStreamCreate(&enc_state->encode_stream);
  if (cuda_err != cudaSuccess) {
    verboseLog(VERBOSE_DEBUG, "nvjpeg_encode: failed to create CUDA stream: %s\n",
               cudaGetErrorString(cuda_err));
    return false;
  }

  // Create encoder state
  status = nvjpegEncoderStateCreate(g_encode_ctx.handle, &enc_state->state,
                                    enc_state->encode_stream);
  if (status != NVJPEG_STATUS_SUCCESS) {
    verboseLog(VERBOSE_DEBUG, "nvjpeg_encode: failed to create encoder state: %s\n",
               nvjpeg_status_string(status));
    cudaStreamDestroy(enc_state->encode_stream);
    enc_state->encode_stream = NULL;
    return false;
  }

  // Create encoder parameters
  status = nvjpegEncoderParamsCreate(g_encode_ctx.handle, &enc_state->params,
                                     enc_state->encode_stream);
  if (status != NVJPEG_STATUS_SUCCESS) {
    verboseLog(VERBOSE_DEBUG,
               "nvjpeg_encode: failed to create encoder params: %s\n",
               nvjpeg_status_string(status));
    nvjpegEncoderStateDestroy(enc_state->state);
    cudaStreamDestroy(enc_state->encode_stream);
    enc_state->encode_stream = NULL;
    return false;
  }

  // Set quality
  status = nvjpegEncoderParamsSetQuality(enc_state->params, g_encode_ctx.quality,
                                         enc_state->encode_stream);
  if (status != NVJPEG_STATUS_SUCCESS) {
    verboseLog(VERBOSE_DEBUG, "nvjpeg_encode: warning - failed to set quality\n");
  }

  // Set chroma subsampling
  status = nvjpegEncoderParamsSetSamplingFactors(
      enc_state->params, to_nvjpeg_subsampling(g_encode_ctx.subsampling),
      enc_state->encode_stream);
  if (status != NVJPEG_STATUS_SUCCESS) {
    verboseLog(VERBOSE_DEBUG,
               "nvjpeg_encode: warning - failed to set subsampling\n");
  }

  atomic_init(&enc_state->in_use, SLOT_FREE);

  return true;
}

static void cleanup_encoder_state(NvJpegEncoderState *enc_state) {
  if (enc_state->params != NULL) {
    nvjpegEncoderParamsDestroy(enc_state->params);
    enc_state->params = NULL;
  }
  if (enc_state->state != NULL) {
    nvjpegEncoderStateDestroy(enc_state->state);
    enc_state->state = NULL;
  }
  if (enc_state->encode_stream != NULL) {
    cudaStreamDestroy(enc_state->encode_stream);
    enc_state->encode_stream = NULL;
  }
}

// ============================================================================
// Global Context Management
// ============================================================================

// Get shared nvJPEG handle from decode context
// This avoids creating a second handle and saves GPU resources
static nvjpegHandle_t get_shared_handle(void) {
  // The decode context must be initialized first
  if (!nvjpeg_is_available()) {
    return NULL;
  }

  // Access the handle from decode context
  // Note: We're reaching into nvjpeg_decode.c's internal state here
  // In a production codebase, this would be exposed via a proper API
  extern nvjpegHandle_t nvjpeg_get_handle_internal(void);
  return nvjpeg_get_handle_internal();
}

bool nvjpeg_encode_init(int num_encoders, int quality,
                        NvJpegEncodeSubsampling subsampling) {
  pthread_mutex_lock(&g_encode_init_mutex);

  if (g_encode_ctx.initialized) {
    pthread_mutex_unlock(&g_encode_init_mutex);
    return true;
  }

  // Get shared handle from decode context
  g_encode_ctx.handle = get_shared_handle();
  if (g_encode_ctx.handle == NULL) {
    verboseLog(VERBOSE_DEBUG,
               "nvjpeg_encode: nvjpeg context not initialized "
               "(call nvjpeg_context_init first)\n");
    pthread_mutex_unlock(&g_encode_init_mutex);
    return false;
  }

  // Validate and store parameters
  g_encode_ctx.quality = (quality < 1) ? 1 : (quality > 100) ? 100 : quality;
  g_encode_ctx.subsampling = subsampling;

  // Allocate encoder state pool
  g_encode_ctx.num_encoders = num_encoders;
  g_encode_ctx.encoder_states =
      calloc((size_t)num_encoders, sizeof(NvJpegEncoderState));
  if (g_encode_ctx.encoder_states == NULL) {
    pthread_mutex_unlock(&g_encode_init_mutex);
    return false;
  }

  // Initialize each encoder state
  for (int i = 0; i < num_encoders; i++) {
    if (!init_encoder_state(&g_encode_ctx.encoder_states[i])) {
      // Cleanup already initialized states
      for (int j = 0; j < i; j++) {
        cleanup_encoder_state(&g_encode_ctx.encoder_states[j]);
      }
      free(g_encode_ctx.encoder_states);
      g_encode_ctx.encoder_states = NULL;
      pthread_mutex_unlock(&g_encode_init_mutex);
      return false;
    }
  }

  // Initialize statistics
  atomic_init(&g_encode_ctx.total_encodes, 0);
  atomic_init(&g_encode_ctx.successful_encodes, 0);
  atomic_init(&g_encode_ctx.failed_encodes, 0);
  atomic_init(&g_encode_ctx.total_bytes_out, 0);
  atomic_init(&g_encode_ctx.current_in_use, 0);
  atomic_init(&g_encode_ctx.concurrent_peak, 0);

  g_encode_ctx.initialized = true;

  verboseLog(VERBOSE_DEBUG,
             "nvjpeg_encode: initialized with %d encoder states, quality=%d\n",
             num_encoders, g_encode_ctx.quality);

  pthread_mutex_unlock(&g_encode_init_mutex);
  return true;
}

void nvjpeg_encode_cleanup(void) {
  pthread_mutex_lock(&g_encode_init_mutex);

  if (!g_encode_ctx.initialized) {
    pthread_mutex_unlock(&g_encode_init_mutex);
    return;
  }

  // Cleanup encoder states
  if (g_encode_ctx.encoder_states != NULL) {
    for (int i = 0; i < g_encode_ctx.num_encoders; i++) {
      cleanup_encoder_state(&g_encode_ctx.encoder_states[i]);
    }
    free(g_encode_ctx.encoder_states);
    g_encode_ctx.encoder_states = NULL;
  }

  // Don't destroy the handle - it's shared with decode context
  g_encode_ctx.handle = NULL;
  g_encode_ctx.num_encoders = 0;
  g_encode_ctx.initialized = false;

  pthread_mutex_unlock(&g_encode_init_mutex);
}

bool nvjpeg_encode_is_available(void) {
  pthread_mutex_lock(&g_encode_init_mutex);
  bool available = g_encode_ctx.initialized;
  pthread_mutex_unlock(&g_encode_init_mutex);
  return available;
}

NvJpegEncodeStats nvjpeg_encode_get_stats(void) {
  NvJpegEncodeStats stats = {0};
  if (!g_encode_ctx.initialized) {
    return stats;
  }

  stats.total_encodes = atomic_load(&g_encode_ctx.total_encodes);
  stats.successful_encodes = atomic_load(&g_encode_ctx.successful_encodes);
  stats.failed_encodes = atomic_load(&g_encode_ctx.failed_encodes);
  stats.total_bytes_out = atomic_load(&g_encode_ctx.total_bytes_out);
  stats.current_in_use = atomic_load(&g_encode_ctx.current_in_use);
  stats.concurrent_peak = atomic_load(&g_encode_ctx.concurrent_peak);
  stats.encoder_state_count = (size_t)g_encode_ctx.num_encoders;

  return stats;
}

void nvjpeg_encode_print_stats(void) {
  NvJpegEncodeStats stats = nvjpeg_encode_get_stats();

  double success_rate = 0.0;
  if (stats.total_encodes > 0) {
    success_rate =
        100.0 * (double)stats.successful_encodes / (double)stats.total_encodes;
  }

  double avg_size = 0.0;
  if (stats.successful_encodes > 0) {
    avg_size =
        (double)stats.total_bytes_out / (double)stats.successful_encodes / 1024.0;
  }

  fprintf(stderr,
          "nvJPEG Encode Statistics:\n"
          "  Encoder states: %zu\n"
          "  Total encodes: %zu\n"
          "  Successful: %zu (%.1f%%)\n"
          "  Failed: %zu\n"
          "  Total output: %.2f MB\n"
          "  Avg output size: %.1f KB\n"
          "  Peak concurrent: %zu\n",
          stats.encoder_state_count, stats.total_encodes,
          stats.successful_encodes, success_rate, stats.failed_encodes,
          (double)stats.total_bytes_out / (1024.0 * 1024.0), avg_size,
          stats.concurrent_peak);
}

// ============================================================================
// Encoder State Pool
// ============================================================================

NvJpegEncoderState *nvjpeg_encode_acquire_state(void) {
  if (!g_encode_ctx.initialized || g_encode_ctx.encoder_states == NULL) {
    return NULL;
  }

  // Lock-free acquisition
  for (int i = 0; i < g_encode_ctx.num_encoders; i++) {
    int expected = SLOT_FREE;
    if (atomic_compare_exchange_strong(&g_encode_ctx.encoder_states[i].in_use,
                                       &expected, SLOT_IN_USE)) {
      atomic_fetch_add(&g_encode_ctx.current_in_use, 1);
      update_encode_peak_usage();
      return &g_encode_ctx.encoder_states[i];
    }
  }

  // All states in use
  return NULL;
}

void nvjpeg_encode_release_state(NvJpegEncoderState *state) {
  if (state == NULL) {
    return;
  }

  atomic_store(&state->in_use, SLOT_FREE);
  atomic_fetch_sub(&g_encode_ctx.current_in_use, 1);
}

// ============================================================================
// Single Image Encode
// ============================================================================

bool nvjpeg_encode_from_gpu(const void *gpu_ptr, size_t pitch, int width,
                            int height, NvJpegEncodeFormat format,
                            NvJpegEncoderState *state, UnpaperCudaStream *stream,
                            NvJpegEncodedImage *out) {
  if (!g_encode_ctx.initialized || gpu_ptr == NULL || state == NULL ||
      out == NULL) {
    return false;
  }

  if (width <= 0 || height <= 0) {
    return false;
  }

  atomic_fetch_add(&g_encode_ctx.total_encodes, 1);

  // Use the encoder's dedicated stream for consistent behavior
  (void)stream; // Unused - using state->encode_stream
  cudaStream_t cuda_stream = state->encode_stream;

  nvjpegStatus_t status;

  // Prepare input image structure
  nvjpegImage_t input_image;
  memset(&input_image, 0, sizeof(input_image));

  // nvJPEG encoder requires RGB input - handle grayscale by conversion
  void *rgb_buffer = NULL;
  size_t rgb_pitch = 0;
  nvjpegInputFormat_t nvfmt = NVJPEG_INPUT_RGBI;

  if (format == NVJPEG_ENC_FMT_GRAY8) {
    // Convert grayscale to RGB on GPU (replicate Y to R,G,B)
    rgb_pitch = (size_t)width * 3;
    rgb_pitch = (rgb_pitch + 255) & ~(size_t)255; // Align to 256 bytes
    size_t rgb_size = rgb_pitch * (size_t)height;

    cudaError_t cuda_err = cudaMallocAsync(&rgb_buffer, rgb_size, cuda_stream);
    if (cuda_err != cudaSuccess) {
      cuda_err = cudaMalloc(&rgb_buffer, rgb_size);
      if (cuda_err != cudaSuccess) {
        verboseLog(VERBOSE_DEBUG,
                   "nvjpeg_encode: failed to allocate RGB buffer for grayscale\n");
        atomic_fetch_add(&g_encode_ctx.failed_encodes, 1);
        return false;
      }
    }

    // Simple approach: allocate host buffer, convert, copy back
    // This is not optimal but works for initial implementation
    uint8_t *host_gray = malloc((size_t)width * (size_t)height);
    uint8_t *host_rgb = malloc((size_t)width * (size_t)height * 3);
    if (host_gray == NULL || host_rgb == NULL) {
      free(host_gray);
      free(host_rgb);
      cudaFreeAsync(rgb_buffer, cuda_stream);
      atomic_fetch_add(&g_encode_ctx.failed_encodes, 1);
      return false;
    }

    // D2H: copy grayscale
    for (int y = 0; y < height; y++) {
      cudaMemcpy(host_gray + y * width,
                 (const uint8_t *)gpu_ptr + y * pitch,
                 (size_t)width, cudaMemcpyDeviceToHost);
    }

    // Convert to RGB
    for (int i = 0; i < width * height; i++) {
      host_rgb[i * 3 + 0] = host_gray[i];
      host_rgb[i * 3 + 1] = host_gray[i];
      host_rgb[i * 3 + 2] = host_gray[i];
    }

    // H2D: copy RGB
    for (int y = 0; y < height; y++) {
      cudaMemcpy((uint8_t *)rgb_buffer + y * rgb_pitch,
                 host_rgb + y * width * 3,
                 (size_t)width * 3, cudaMemcpyHostToDevice);
    }

    free(host_gray);
    free(host_rgb);

    input_image.channel[0] = (unsigned char *)rgb_buffer;
    input_image.pitch[0] = (unsigned int)rgb_pitch;

    // Set subsampling to gray for grayscale output
    status = nvjpegEncoderParamsSetSamplingFactors(state->params, NVJPEG_CSS_GRAY,
                                                   cuda_stream);
    if (status != NVJPEG_STATUS_SUCCESS) {
      verboseLog(VERBOSE_DEBUG,
                 "nvjpeg_encode: warning - failed to set gray subsampling\n");
    }
  } else {
    // RGB/BGR input - use directly
    input_image.channel[0] = (unsigned char *)gpu_ptr;
    input_image.pitch[0] = (unsigned int)pitch;
    nvfmt = to_nvjpeg_input_format(format);

    // Restore configured subsampling for color images
    status = nvjpegEncoderParamsSetSamplingFactors(
        state->params, to_nvjpeg_subsampling(g_encode_ctx.subsampling),
        cuda_stream);
  }

  // Encode the image
  status = nvjpegEncodeImage(g_encode_ctx.handle, state->state, state->params,
                             &input_image, nvfmt, width, height, cuda_stream);
  if (status != NVJPEG_STATUS_SUCCESS) {
    verboseLog(VERBOSE_DEBUG, "nvjpeg_encode: nvjpegEncodeImage failed: %s\n",
               nvjpeg_status_string(status));
    if (rgb_buffer != NULL) {
      cudaFreeAsync(rgb_buffer, cuda_stream);
    }
    atomic_fetch_add(&g_encode_ctx.failed_encodes, 1);
    return false;
  }

  // Get the size of the encoded bitstream
  size_t bitstream_size = 0;
  status = nvjpegEncodeRetrieveBitstream(g_encode_ctx.handle, state->state, NULL,
                                         &bitstream_size, cuda_stream);
  if (status != NVJPEG_STATUS_SUCCESS) {
    verboseLog(VERBOSE_DEBUG,
               "nvjpeg_encode: failed to get bitstream size: %s\n",
               nvjpeg_status_string(status));
    if (rgb_buffer != NULL) {
      cudaFreeAsync(rgb_buffer, cuda_stream);
    }
    atomic_fetch_add(&g_encode_ctx.failed_encodes, 1);
    return false;
  }

  // Allocate host buffer for JPEG data
  uint8_t *jpeg_data = malloc(bitstream_size);
  if (jpeg_data == NULL) {
    verboseLog(VERBOSE_DEBUG, "nvjpeg_encode: failed to allocate output buffer\n");
    if (rgb_buffer != NULL) {
      cudaFreeAsync(rgb_buffer, cuda_stream);
    }
    atomic_fetch_add(&g_encode_ctx.failed_encodes, 1);
    return false;
  }

  // Retrieve the encoded bitstream
  // Note: This synchronizes the stream
  status = nvjpegEncodeRetrieveBitstream(g_encode_ctx.handle, state->state,
                                         jpeg_data, &bitstream_size, cuda_stream);
  if (status != NVJPEG_STATUS_SUCCESS) {
    verboseLog(VERBOSE_DEBUG, "nvjpeg_encode: failed to retrieve bitstream: %s\n",
               nvjpeg_status_string(status));
    free(jpeg_data);
    if (rgb_buffer != NULL) {
      cudaFreeAsync(rgb_buffer, cuda_stream);
    }
    atomic_fetch_add(&g_encode_ctx.failed_encodes, 1);
    return false;
  }

  // Synchronize to ensure encode is complete
  cudaError_t sync_err = cudaStreamSynchronize(cuda_stream);
  if (sync_err != cudaSuccess) {
    verboseLog(VERBOSE_DEBUG, "nvjpeg_encode: stream sync failed\n");
    free(jpeg_data);
    if (rgb_buffer != NULL) {
      cudaFreeAsync(rgb_buffer, cuda_stream);
    }
    atomic_fetch_add(&g_encode_ctx.failed_encodes, 1);
    return false;
  }

  // Free temporary RGB buffer if allocated
  if (rgb_buffer != NULL) {
    cudaFreeAsync(rgb_buffer, cuda_stream);
  }

  // Fill output structure
  out->jpeg_data = jpeg_data;
  out->jpeg_size = bitstream_size;
  out->width = width;
  out->height = height;

  atomic_fetch_add(&g_encode_ctx.successful_encodes, 1);
  atomic_fetch_add(&g_encode_ctx.total_bytes_out, bitstream_size);

  return true;
}

bool nvjpeg_encode_gpu_to_file(const void *gpu_ptr, size_t pitch, int width,
                               int height, NvJpegEncodeFormat format,
                               UnpaperCudaStream *stream, const char *filename) {
  if (filename == NULL) {
    return false;
  }

  // Acquire encoder state
  NvJpegEncoderState *state = nvjpeg_encode_acquire_state();
  if (state == NULL) {
    verboseLog(VERBOSE_DEBUG, "nvjpeg_encode: no encoder state available\n");
    return false;
  }

  NvJpegEncodedImage out = {0};
  bool result = nvjpeg_encode_from_gpu(gpu_ptr, pitch, width, height, format,
                                       state, stream, &out);

  nvjpeg_encode_release_state(state);

  if (!result) {
    return false;
  }

  // Write to file
  FILE *f = fopen(filename, "wb");
  if (f == NULL) {
    free(out.jpeg_data);
    return false;
  }

  size_t written = fwrite(out.jpeg_data, 1, out.jpeg_size, f);
  fclose(f);
  free(out.jpeg_data);

  return written == out.jpeg_size;
}

// ============================================================================
// Batched Encode API
// ============================================================================

// Batched encoder context
typedef struct {
  int max_batch_size;
  int max_width;
  int max_height;
  bool initialized;
} NvJpegEncodeBatchContext;

static NvJpegEncodeBatchContext g_encode_batch_ctx = {0};
static pthread_mutex_t g_encode_batch_mutex = PTHREAD_MUTEX_INITIALIZER;

bool nvjpeg_encode_batch_init(int max_batch_size, int max_width,
                              int max_height) {
  pthread_mutex_lock(&g_encode_batch_mutex);

  if (g_encode_batch_ctx.initialized) {
    pthread_mutex_unlock(&g_encode_batch_mutex);
    return true;
  }

  if (!g_encode_ctx.initialized) {
    pthread_mutex_unlock(&g_encode_batch_mutex);
    return false;
  }

  // Cap batch size
  if (max_batch_size > NVJPEG_MAX_ENCODE_BATCH_SIZE) {
    max_batch_size = NVJPEG_MAX_ENCODE_BATCH_SIZE;
  }
  if (max_batch_size < 1) {
    max_batch_size = 1;
  }

  g_encode_batch_ctx.max_batch_size = max_batch_size;
  g_encode_batch_ctx.max_width = max_width;
  g_encode_batch_ctx.max_height = max_height;
  g_encode_batch_ctx.initialized = true;

  verboseLog(VERBOSE_DEBUG,
             "nvjpeg_encode_batch: initialized with max_batch=%d, max=%dx%d\n",
             max_batch_size, max_width, max_height);

  pthread_mutex_unlock(&g_encode_batch_mutex);
  return true;
}

int nvjpeg_encode_batch(const void *const *gpu_ptrs, const size_t *pitches,
                        const int *widths, const int *heights,
                        NvJpegEncodeFormat format, int batch_size,
                        NvJpegEncodedImage *outputs) {
  if (!g_encode_batch_ctx.initialized) {
    return 0;
  }

  if (gpu_ptrs == NULL || pitches == NULL || widths == NULL ||
      heights == NULL || outputs == NULL || batch_size <= 0) {
    return 0;
  }

  if (batch_size > g_encode_batch_ctx.max_batch_size) {
    batch_size = g_encode_batch_ctx.max_batch_size;
  }

  int success_count = 0;

  // Encode each image using available encoder states
  // nvJPEG doesn't have a true batched encode API like decode,
  // but we can parallelize by using multiple encoder states
  for (int i = 0; i < batch_size; i++) {
    outputs[i].jpeg_data = NULL;
    outputs[i].jpeg_size = 0;
    outputs[i].width = 0;
    outputs[i].height = 0;

    if (gpu_ptrs[i] == NULL) {
      continue;
    }

    NvJpegEncoderState *state = nvjpeg_encode_acquire_state();
    if (state == NULL) {
      // No state available, try to wait or skip
      continue;
    }

    bool result = nvjpeg_encode_from_gpu(gpu_ptrs[i], pitches[i], widths[i],
                                         heights[i], format, state, NULL,
                                         &outputs[i]);

    nvjpeg_encode_release_state(state);

    if (result) {
      success_count++;
    }
  }

  return success_count;
}

bool nvjpeg_encode_batch_is_ready(void) {
  pthread_mutex_lock(&g_encode_batch_mutex);
  bool ready = g_encode_batch_ctx.initialized;
  pthread_mutex_unlock(&g_encode_batch_mutex);
  return ready;
}

void nvjpeg_encode_batch_cleanup(void) {
  pthread_mutex_lock(&g_encode_batch_mutex);
  g_encode_batch_ctx.initialized = false;
  g_encode_batch_ctx.max_batch_size = 0;
  pthread_mutex_unlock(&g_encode_batch_mutex);
}

// ============================================================================
// Quality Control
// ============================================================================

void nvjpeg_encode_set_quality(int quality) {
  pthread_mutex_lock(&g_encode_init_mutex);

  g_encode_ctx.quality = (quality < 1) ? 1 : (quality > 100) ? 100 : quality;

  // Update all encoder states
  if (g_encode_ctx.initialized && g_encode_ctx.encoder_states != NULL) {
    for (int i = 0; i < g_encode_ctx.num_encoders; i++) {
      NvJpegEncoderState *state = &g_encode_ctx.encoder_states[i];
      nvjpegEncoderParamsSetQuality(state->params, g_encode_ctx.quality,
                                    state->encode_stream);
    }
  }

  pthread_mutex_unlock(&g_encode_init_mutex);
}

int nvjpeg_encode_get_quality(void) {
  return g_encode_ctx.quality;
}

void nvjpeg_encode_set_subsampling(NvJpegEncodeSubsampling subsampling) {
  pthread_mutex_lock(&g_encode_init_mutex);

  g_encode_ctx.subsampling = subsampling;

  // Update all encoder states
  if (g_encode_ctx.initialized && g_encode_ctx.encoder_states != NULL) {
    nvjpegChromaSubsampling_t nvsub = to_nvjpeg_subsampling(subsampling);
    for (int i = 0; i < g_encode_ctx.num_encoders; i++) {
      NvJpegEncoderState *state = &g_encode_ctx.encoder_states[i];
      nvjpegEncoderParamsSetSamplingFactors(state->params, nvsub,
                                            state->encode_stream);
    }
  }

  pthread_mutex_unlock(&g_encode_init_mutex);
}

NvJpegEncodeSubsampling nvjpeg_encode_get_subsampling(void) {
  return g_encode_ctx.subsampling;
}

// ============================================================================
// Internal API for decode context access
// ============================================================================

// This function is called by nvjpeg_encode to get the shared handle
// It's defined in nvjpeg_decode.c but we need to declare it here
nvjpegHandle_t nvjpeg_get_handle_internal(void);

#else // !UNPAPER_WITH_CUDA

// Stub implementations for non-CUDA builds

bool nvjpeg_encode_init(int num_encoders, int quality,
                        NvJpegEncodeSubsampling subsampling) {
  (void)num_encoders;
  (void)quality;
  (void)subsampling;
  return false;
}

void nvjpeg_encode_cleanup(void) {}

bool nvjpeg_encode_is_available(void) { return false; }

NvJpegEncodeStats nvjpeg_encode_get_stats(void) {
  NvJpegEncodeStats stats = {0};
  return stats;
}

void nvjpeg_encode_print_stats(void) {}

NvJpegEncoderState *nvjpeg_encode_acquire_state(void) { return NULL; }

void nvjpeg_encode_release_state(NvJpegEncoderState *state) { (void)state; }

bool nvjpeg_encode_from_gpu(const void *gpu_ptr, size_t pitch, int width,
                            int height, NvJpegEncodeFormat format,
                            NvJpegEncoderState *state, UnpaperCudaStream *stream,
                            NvJpegEncodedImage *out) {
  (void)gpu_ptr;
  (void)pitch;
  (void)width;
  (void)height;
  (void)format;
  (void)state;
  (void)stream;
  (void)out;
  return false;
}

bool nvjpeg_encode_gpu_to_file(const void *gpu_ptr, size_t pitch, int width,
                               int height, NvJpegEncodeFormat format,
                               UnpaperCudaStream *stream, const char *filename) {
  (void)gpu_ptr;
  (void)pitch;
  (void)width;
  (void)height;
  (void)format;
  (void)stream;
  (void)filename;
  return false;
}

bool nvjpeg_encode_batch_init(int max_batch_size, int max_width,
                              int max_height) {
  (void)max_batch_size;
  (void)max_width;
  (void)max_height;
  return false;
}

int nvjpeg_encode_batch(const void *const *gpu_ptrs, const size_t *pitches,
                        const int *widths, const int *heights,
                        NvJpegEncodeFormat format, int batch_size,
                        NvJpegEncodedImage *outputs) {
  (void)gpu_ptrs;
  (void)pitches;
  (void)widths;
  (void)heights;
  (void)format;
  (void)batch_size;
  (void)outputs;
  return 0;
}

bool nvjpeg_encode_batch_is_ready(void) { return false; }

void nvjpeg_encode_batch_cleanup(void) {}

void nvjpeg_encode_set_quality(int quality) { (void)quality; }

int nvjpeg_encode_get_quality(void) { return 85; }

void nvjpeg_encode_set_subsampling(NvJpegEncodeSubsampling subsampling) {
  (void)subsampling;
}

NvJpegEncodeSubsampling nvjpeg_encode_get_subsampling(void) {
  return NVJPEG_ENC_SUBSAMPLING_420;
}

#endif // UNPAPER_WITH_CUDA
