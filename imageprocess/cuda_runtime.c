// SPDX-FileCopyrightText: 2025 The unpaper authors
//
// SPDX-License-Identifier: GPL-2.0-only

#include "imageprocess/cuda_runtime.h"

#include <cuda_runtime.h>
#include <dlfcn.h>
#include <stdbool.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "lib/logging.h"

// Driver API types and functions (for PTX loading only)
typedef int CUresult;
typedef struct CUmod_st *CUmodule;
typedef struct CUfunc_st *CUfunction;

enum { CUDA_SUCCESS = 0 };

typedef CUresult (*cuModuleLoadData_fn)(CUmodule *module, const void *image);
typedef CUresult (*cuModuleUnload_fn)(CUmodule hmod);
typedef CUresult (*cuModuleGetFunction_fn)(CUfunction *hfunc, CUmodule hmod,
                                           const char *name);
typedef CUresult (*cuLaunchKernel_fn)(
    CUfunction f, unsigned int gridDimX, unsigned int gridDimY,
    unsigned int gridDimZ, unsigned int blockDimX, unsigned int blockDimY,
    unsigned int blockDimZ, unsigned int sharedMemBytes, cudaStream_t hStream,
    void **kernelParams, void **extra);
typedef CUresult (*cuGetErrorString_fn)(CUresult error, const char **pStr);

typedef struct {
  void *lib;
  cuModuleLoadData_fn cuModuleLoadData;
  cuModuleUnload_fn cuModuleUnload;
  cuModuleGetFunction_fn cuModuleGetFunction;
  cuLaunchKernel_fn cuLaunchKernel;
  cuGetErrorString_fn cuGetErrorString;
} DriverApiSymbols;

static DriverApiSymbols driver_syms;
static bool driver_syms_loaded = false;
static bool cuda_initialized = false;
static cudaStream_t cuda_stream = NULL;

typedef struct UnpaperCudaStream {
  cudaStream_t stream;
  void *pinned;
  size_t pinned_capacity;
  bool pinned_is_pinned;
} UnpaperCudaStream;

static UnpaperCudaStream default_stream = {0};
static UnpaperCudaStream *current_stream = NULL;
static void *scratch_dptr = NULL;
static size_t scratch_capacity = 0;

static void *load_driver_sym(const char *name) {
  void *p = dlsym(driver_syms.lib, name);
  if (p != NULL) {
    return p;
  }
  char alt[128];
  snprintf(alt, sizeof(alt), "%s_v2", name);
  return dlsym(driver_syms.lib, alt);
}

static const char *cu_err(CUresult res) {
  static char buf[128];
  if (driver_syms.cuGetErrorString != NULL) {
    const char *s = NULL;
    if (driver_syms.cuGetErrorString(res, &s) == CUDA_SUCCESS && s != NULL) {
      return s;
    }
  }
  snprintf(buf, sizeof(buf), "CUDA error %d", res);
  return buf;
}

static bool load_driver_api_symbols(void) {
  if (driver_syms_loaded) {
    return driver_syms.lib != NULL;
  }
  driver_syms_loaded = true;

  driver_syms.lib = dlopen("libcuda.so.1", RTLD_LAZY);
  if (driver_syms.lib == NULL) {
    driver_syms.lib = dlopen("libcuda.so", RTLD_LAZY);
  }
  if (driver_syms.lib == NULL) {
    return false;
  }

  driver_syms.cuModuleLoadData =
      (cuModuleLoadData_fn)load_driver_sym("cuModuleLoadData");
  driver_syms.cuModuleUnload =
      (cuModuleUnload_fn)load_driver_sym("cuModuleUnload");
  driver_syms.cuModuleGetFunction =
      (cuModuleGetFunction_fn)load_driver_sym("cuModuleGetFunction");
  driver_syms.cuLaunchKernel =
      (cuLaunchKernel_fn)load_driver_sym("cuLaunchKernel");
  driver_syms.cuGetErrorString =
      (cuGetErrorString_fn)load_driver_sym("cuGetErrorString");

  return driver_syms.cuModuleLoadData != NULL &&
         driver_syms.cuModuleGetFunction != NULL &&
         driver_syms.cuLaunchKernel != NULL;
}

UnpaperCudaInitStatus unpaper_cuda_try_init(void) {
  if (cuda_initialized) {
    return UNPAPER_CUDA_INIT_OK;
  }

  // Check device count using Runtime API
  int device_count = 0;
  cudaError_t err = cudaGetDeviceCount(&device_count);
  if (err != cudaSuccess) {
    return UNPAPER_CUDA_INIT_ERROR;
  }
  if (device_count <= 0) {
    return UNPAPER_CUDA_INIT_NO_DEVICE;
  }

  // Initialize via Runtime API (sets up primary context)
  err = cudaSetDevice(0);
  if (err != cudaSuccess) {
    return UNPAPER_CUDA_INIT_ERROR;
  }

  // Create a stream
  err = cudaStreamCreate(&cuda_stream);
  if (err != cudaSuccess) {
    return UNPAPER_CUDA_INIT_ERROR;
  }

  default_stream.stream = cuda_stream;
  current_stream = &default_stream;

  // Load Driver API symbols for PTX operations
  if (!load_driver_api_symbols()) {
    // Driver API not available - PTX loading won't work
    // This is an error since we need it for our kernels
    cudaStreamDestroy(cuda_stream);
    cuda_stream = NULL;
    return UNPAPER_CUDA_INIT_ERROR;
  }

  cuda_initialized = true;
  return UNPAPER_CUDA_INIT_OK;
}

const char *unpaper_cuda_init_status_string(UnpaperCudaInitStatus st) {
  switch (st) {
  case UNPAPER_CUDA_INIT_OK:
    return "CUDA is available.";
  case UNPAPER_CUDA_INIT_NO_RUNTIME:
    return "CUDA support is compiled in, but the CUDA driver library is not "
           "available.";
  case UNPAPER_CUDA_INIT_NO_DEVICE:
    return "CUDA support is compiled in, but no CUDA-capable devices were "
           "found.";
  case UNPAPER_CUDA_INIT_ERROR:
    return "CUDA initialization failed.";
  }
  return "CUDA initialization failed.";
}

uint64_t unpaper_cuda_malloc(size_t bytes) {
  UnpaperCudaInitStatus st = unpaper_cuda_try_init();
  if (st != UNPAPER_CUDA_INIT_OK) {
    errOutput("%s", unpaper_cuda_init_status_string(st));
  }

  void *dptr = NULL;
  cudaError_t err = cudaMalloc(&dptr, bytes);
  if (err != cudaSuccess) {
    errOutput("CUDA allocation failed: %s", cudaGetErrorString(err));
  }
  return (uint64_t)(uintptr_t)dptr;
}

void unpaper_cuda_free(uint64_t dptr) {
  if (dptr == 0) {
    return;
  }
  if (!cuda_initialized) {
    return;
  }
  (void)cudaFree((void *)(uintptr_t)dptr);
}

void unpaper_cuda_memcpy_h2d(uint64_t dst, const void *src, size_t bytes) {
  UnpaperCudaInitStatus st = unpaper_cuda_try_init();
  if (st != UNPAPER_CUDA_INIT_OK) {
    errOutput("%s", unpaper_cuda_init_status_string(st));
  }

  cudaError_t err = cudaMemcpy((void *)(uintptr_t)dst, src, bytes,
                               cudaMemcpyHostToDevice);
  if (err != cudaSuccess) {
    errOutput("CUDA memcpy HtoD failed: %s", cudaGetErrorString(err));
  }
}

void unpaper_cuda_memcpy_d2h(void *dst, uint64_t src, size_t bytes) {
  UnpaperCudaInitStatus st = unpaper_cuda_try_init();
  if (st != UNPAPER_CUDA_INIT_OK) {
    errOutput("%s", unpaper_cuda_init_status_string(st));
  }

  cudaError_t err = cudaMemcpy(dst, (const void *)(uintptr_t)src, bytes,
                               cudaMemcpyDeviceToHost);
  if (err != cudaSuccess) {
    errOutput("CUDA memcpy DtoH failed: %s", cudaGetErrorString(err));
  }
}

void unpaper_cuda_memcpy_d2d(uint64_t dst, uint64_t src, size_t bytes) {
  UnpaperCudaInitStatus st = unpaper_cuda_try_init();
  if (st != UNPAPER_CUDA_INIT_OK) {
    errOutput("%s", unpaper_cuda_init_status_string(st));
  }

  cudaError_t err =
      cudaMemcpy((void *)(uintptr_t)dst, (const void *)(uintptr_t)src, bytes,
                 cudaMemcpyDeviceToDevice);
  if (err != cudaSuccess) {
    errOutput("CUDA memcpy DtoD failed: %s", cudaGetErrorString(err));
  }
}

static cudaStream_t stream_handle(UnpaperCudaStream *stream) {
  UnpaperCudaStream *s = stream;
  if (s == NULL) {
    s = current_stream != NULL ? current_stream : &default_stream;
  }
  if (s != NULL && s->stream != NULL) {
    return s->stream;
  }
  return cuda_stream;
}

void unpaper_cuda_memcpy_h2d_async(UnpaperCudaStream *stream, uint64_t dst,
                                   const void *src, size_t bytes) {
  UnpaperCudaInitStatus st = unpaper_cuda_try_init();
  if (st != UNPAPER_CUDA_INIT_OK) {
    errOutput("%s", unpaper_cuda_init_status_string(st));
  }

  cudaStream_t s = stream_handle(stream);
  cudaError_t err = cudaMemcpyAsync((void *)(uintptr_t)dst, src, bytes,
                                    cudaMemcpyHostToDevice, s);
  if (err != cudaSuccess) {
    errOutput("CUDA async memcpy HtoD failed: %s", cudaGetErrorString(err));
  }
}

void unpaper_cuda_memcpy_d2h_async(UnpaperCudaStream *stream, void *dst,
                                   uint64_t src, size_t bytes) {
  UnpaperCudaInitStatus st = unpaper_cuda_try_init();
  if (st != UNPAPER_CUDA_INIT_OK) {
    errOutput("%s", unpaper_cuda_init_status_string(st));
  }

  cudaStream_t s = stream_handle(stream);
  cudaError_t err = cudaMemcpyAsync(dst, (const void *)(uintptr_t)src, bytes,
                                    cudaMemcpyDeviceToHost, s);
  if (err != cudaSuccess) {
    errOutput("CUDA async memcpy DtoH failed: %s", cudaGetErrorString(err));
  }
}

void unpaper_cuda_memcpy_d2d_async(UnpaperCudaStream *stream, uint64_t dst,
                                   uint64_t src, size_t bytes) {
  UnpaperCudaInitStatus st = unpaper_cuda_try_init();
  if (st != UNPAPER_CUDA_INIT_OK) {
    errOutput("%s", unpaper_cuda_init_status_string(st));
  }

  cudaStream_t s = stream_handle(stream);
  cudaError_t err =
      cudaMemcpyAsync((void *)(uintptr_t)dst, (const void *)(uintptr_t)src,
                      bytes, cudaMemcpyDeviceToDevice, s);
  if (err != cudaSuccess) {
    errOutput("CUDA async memcpy DtoD failed: %s", cudaGetErrorString(err));
  }
}

void unpaper_cuda_memset_d8(uint64_t dst, uint8_t value, size_t bytes) {
  UnpaperCudaInitStatus st = unpaper_cuda_try_init();
  if (st != UNPAPER_CUDA_INIT_OK) {
    errOutput("%s", unpaper_cuda_init_status_string(st));
  }

  cudaError_t err = cudaMemset((void *)(uintptr_t)dst, (int)value, bytes);
  if (err != cudaSuccess) {
    errOutput("CUDA memset failed: %s", cudaGetErrorString(err));
  }
}

bool unpaper_cuda_pinned_alloc(UnpaperCudaPinnedBuffer *buf, size_t bytes) {
  if (buf == NULL) {
    return false;
  }
  buf->ptr = NULL;
  buf->bytes = 0;
  buf->is_pinned = false;

  UnpaperCudaInitStatus st = unpaper_cuda_try_init();
  if (st == UNPAPER_CUDA_INIT_OK) {
    void *p = NULL;
    cudaError_t err = cudaMallocHost(&p, bytes);
    if (err == cudaSuccess && p != NULL) {
      buf->ptr = p;
      buf->bytes = bytes;
      buf->is_pinned = true;
      return true;
    }
  }

  void *fallback = malloc(bytes);
  if (fallback == NULL) {
    return false;
  }
  buf->ptr = fallback;
  buf->bytes = bytes;
  buf->is_pinned = false;
  return true;
}

void unpaper_cuda_pinned_free(UnpaperCudaPinnedBuffer *buf) {
  if (buf == NULL || buf->ptr == NULL) {
    return;
  }
  if (buf->is_pinned) {
    (void)cudaFreeHost(buf->ptr);
  } else {
    free(buf->ptr);
  }
  buf->ptr = NULL;
  buf->bytes = 0;
  buf->is_pinned = false;
}

void *unpaper_cuda_stream_pinned_reserve(UnpaperCudaStream *stream,
                                         size_t bytes, size_t *capacity_out) {
  UnpaperCudaInitStatus st = unpaper_cuda_try_init();
  if (st != UNPAPER_CUDA_INIT_OK) {
    return NULL;
  }

  UnpaperCudaStream *s = (stream != NULL) ? stream : &default_stream;
  if (s->pinned_capacity < bytes) {
    if (s->pinned != NULL) {
      UnpaperCudaPinnedBuffer old = {.ptr = s->pinned,
                                     .bytes = s->pinned_capacity,
                                     .is_pinned = s->pinned_is_pinned};
      unpaper_cuda_pinned_free(&old);
    }
    UnpaperCudaPinnedBuffer buf;
    if (!unpaper_cuda_pinned_alloc(&buf, bytes)) {
      return NULL;
    }
    s->pinned = buf.ptr;
    s->pinned_capacity = buf.bytes;
    s->pinned_is_pinned = buf.is_pinned;
  }
  if (capacity_out != NULL) {
    *capacity_out = s->pinned_capacity;
  }
  return s->pinned;
}

uint64_t unpaper_cuda_scratch_reserve(size_t bytes, size_t *capacity_out) {
  UnpaperCudaInitStatus st = unpaper_cuda_try_init();
  if (st != UNPAPER_CUDA_INIT_OK) {
    errOutput("%s", unpaper_cuda_init_status_string(st));
  }

  if (scratch_capacity < bytes) {
    if (scratch_dptr != NULL) {
      unpaper_cuda_free((uint64_t)(uintptr_t)scratch_dptr);
      scratch_dptr = NULL;
      scratch_capacity = 0;
    }
    scratch_dptr = (void *)(uintptr_t)unpaper_cuda_malloc(bytes);
    scratch_capacity = bytes;
  }
  if (capacity_out != NULL) {
    *capacity_out = scratch_capacity;
  }
  return (uint64_t)(uintptr_t)scratch_dptr;
}

void unpaper_cuda_scratch_release_all(void) {
  if (scratch_dptr != NULL) {
    unpaper_cuda_free((uint64_t)(uintptr_t)scratch_dptr);
    scratch_dptr = NULL;
    scratch_capacity = 0;
  }
}

UnpaperCudaStream *unpaper_cuda_stream_create(void) {
  UnpaperCudaInitStatus st = unpaper_cuda_try_init();
  if (st != UNPAPER_CUDA_INIT_OK) {
    return NULL;
  }

  UnpaperCudaStream *s = calloc(1, sizeof(*s));
  if (s == NULL) {
    return NULL;
  }

  cudaStream_t h = NULL;
  cudaError_t err = cudaStreamCreate(&h);
  if (err != cudaSuccess || h == NULL) {
    free(s);
    return NULL;
  }

  s->stream = h;
  s->pinned = NULL;
  s->pinned_capacity = 0;
  s->pinned_is_pinned = false;
  return s;
}

UnpaperCudaStream *unpaper_cuda_stream_get_default(void) {
  UnpaperCudaInitStatus st = unpaper_cuda_try_init();
  if (st != UNPAPER_CUDA_INIT_OK) {
    return NULL;
  }
  return &default_stream;
}

void unpaper_cuda_stream_destroy(UnpaperCudaStream *stream) {
  if (stream == NULL) {
    return;
  }
  if (stream == &default_stream) {
    return;
  }
  if (stream->pinned != NULL) {
    UnpaperCudaPinnedBuffer buf = {.ptr = stream->pinned,
                                   .bytes = stream->pinned_capacity,
                                   .is_pinned = stream->pinned_is_pinned};
    unpaper_cuda_pinned_free(&buf);
    stream->pinned = NULL;
    stream->pinned_capacity = 0;
    stream->pinned_is_pinned = false;
  }
  if (stream->stream != NULL) {
    (void)cudaStreamDestroy(stream->stream);
  }
  free(stream);
}

void unpaper_cuda_set_current_stream(UnpaperCudaStream *stream) {
  if (stream == NULL) {
    current_stream = &default_stream;
  } else {
    current_stream = stream;
  }
}

UnpaperCudaStream *unpaper_cuda_get_current_stream(void) {
  return current_stream;
}

void unpaper_cuda_stream_synchronize_on(UnpaperCudaStream *stream) {
  cudaStream_t s = stream_handle(stream);
  if (s != NULL) {
    (void)cudaStreamSynchronize(s);
  } else {
    (void)cudaDeviceSynchronize();
  }
}

void *unpaper_cuda_module_load_ptx(const char *ptx) {
  UnpaperCudaInitStatus st = unpaper_cuda_try_init();
  if (st != UNPAPER_CUDA_INIT_OK) {
    errOutput("%s", unpaper_cuda_init_status_string(st));
  }
  if (driver_syms.cuModuleLoadData == NULL) {
    errOutput("CUDA module loading is unavailable.");
  }

  CUmodule mod = NULL;
  CUresult res = driver_syms.cuModuleLoadData(&mod, (const void *)ptx);
  if (res != CUDA_SUCCESS || mod == NULL) {
    errOutput("CUDA module load failed: %s", cu_err(res));
  }
  return (void *)mod;
}

void unpaper_cuda_module_unload(void *module) {
  if (module == NULL) {
    return;
  }
  if (!cuda_initialized) {
    return;
  }
  if (driver_syms.cuModuleUnload == NULL) {
    return;
  }
  (void)driver_syms.cuModuleUnload((CUmodule)module);
}

void *unpaper_cuda_module_get_function(void *module, const char *name) {
  UnpaperCudaInitStatus st = unpaper_cuda_try_init();
  if (st != UNPAPER_CUDA_INIT_OK) {
    errOutput("%s", unpaper_cuda_init_status_string(st));
  }
  if (module == NULL || name == NULL) {
    errOutput("invalid CUDA function lookup.");
  }
  if (driver_syms.cuModuleGetFunction == NULL) {
    errOutput("CUDA function lookup is unavailable.");
  }

  CUfunction fn = NULL;
  CUresult res = driver_syms.cuModuleGetFunction(&fn, (CUmodule)module, name);
  if (res != CUDA_SUCCESS || fn == NULL) {
    errOutput("CUDA function lookup failed (%s): %s", name, cu_err(res));
  }
  return (void *)fn;
}

void unpaper_cuda_launch_kernel(void *func, uint32_t grid_x, uint32_t grid_y,
                                uint32_t grid_z, uint32_t block_x,
                                uint32_t block_y, uint32_t block_z,
                                void **kernel_params) {
  UnpaperCudaInitStatus st = unpaper_cuda_try_init();
  if (st != UNPAPER_CUDA_INIT_OK) {
    errOutput("%s", unpaper_cuda_init_status_string(st));
  }
  if (driver_syms.cuLaunchKernel == NULL) {
    errOutput("CUDA kernel launch is unavailable.");
  }

  cudaStream_t s = stream_handle(NULL);
  CUresult res =
      driver_syms.cuLaunchKernel((CUfunction)func, grid_x, grid_y, grid_z,
                                 block_x, block_y, block_z, 0, s,
                                 kernel_params, NULL);
  if (res != CUDA_SUCCESS) {
    errOutput("CUDA kernel launch failed: %s", cu_err(res));
  }

  // Note: No automatic sync here - cudaMemcpy calls are synchronous and will
  // implicitly wait for kernels to complete. This allows kernel pipelining.
}

void unpaper_cuda_launch_kernel_on_stream(UnpaperCudaStream *stream,
                                          void *func, uint32_t grid_x,
                                          uint32_t grid_y, uint32_t grid_z,
                                          uint32_t block_x, uint32_t block_y,
                                          uint32_t block_z,
                                          void **kernel_params) {
  UnpaperCudaInitStatus st = unpaper_cuda_try_init();
  if (st != UNPAPER_CUDA_INIT_OK) {
    errOutput("%s", unpaper_cuda_init_status_string(st));
  }
  if (driver_syms.cuLaunchKernel == NULL) {
    errOutput("CUDA kernel launch is unavailable.");
  }

  cudaStream_t s = stream_handle(stream);
  CUresult res = driver_syms.cuLaunchKernel((CUfunction)func, grid_x, grid_y,
                                            grid_z, block_x, block_y, block_z,
                                            0, s, kernel_params, NULL);
  if (res != CUDA_SUCCESS) {
    errOutput("CUDA kernel launch failed: %s", cu_err(res));
  }
}

bool unpaper_cuda_events_supported(void) {
  return unpaper_cuda_events_supported_on(NULL);
}

bool unpaper_cuda_event_pair_start(void **start, void **stop) {
  return unpaper_cuda_event_pair_start_on(NULL, start, stop);
}

double unpaper_cuda_event_pair_stop_ms(void **start, void **stop) {
  return unpaper_cuda_event_pair_stop_ms_on(NULL, start, stop);
}

void unpaper_cuda_stream_synchronize(void) {
  if (!cuda_initialized) {
    return;
  }
  unpaper_cuda_stream_synchronize_on(NULL);
}

bool unpaper_cuda_events_supported_on(UnpaperCudaStream *stream) {
  if (!cuda_initialized) {
    UnpaperCudaInitStatus st = unpaper_cuda_try_init();
    if (st != UNPAPER_CUDA_INIT_OK) {
      return false;
    }
  }
  cudaStream_t s = stream_handle(stream);
  return (s != NULL);
}

bool unpaper_cuda_event_pair_start_on(UnpaperCudaStream *stream, void **start,
                                      void **stop) {
  if (start == NULL || stop == NULL) {
    return false;
  }
  *start = NULL;
  *stop = NULL;

  if (!unpaper_cuda_events_supported_on(stream)) {
    return false;
  }

  cudaStream_t s = stream_handle(stream);
  cudaEvent_t ev_start = NULL;
  cudaEvent_t ev_stop = NULL;

  if (cudaEventCreate(&ev_start) != cudaSuccess ||
      cudaEventCreate(&ev_stop) != cudaSuccess) {
    if (ev_start != NULL)
      cudaEventDestroy(ev_start);
    if (ev_stop != NULL)
      cudaEventDestroy(ev_stop);
    return false;
  }

  if (cudaEventRecord(ev_start, s) != cudaSuccess) {
    cudaEventDestroy(ev_start);
    cudaEventDestroy(ev_stop);
    return false;
  }

  *start = (void *)ev_start;
  *stop = (void *)ev_stop;
  return true;
}

double unpaper_cuda_event_pair_stop_ms_on(UnpaperCudaStream *stream,
                                          void **start, void **stop) {
  if (start == NULL || stop == NULL || *start == NULL || *stop == NULL) {
    return 0.0;
  }
  cudaEvent_t ev_start = (cudaEvent_t)*start;
  cudaEvent_t ev_stop = (cudaEvent_t)*stop;
  cudaStream_t s = stream_handle(stream);

  double ms = 0.0;
  if (cudaEventRecord(ev_stop, s) == cudaSuccess &&
      cudaEventSynchronize(ev_stop) == cudaSuccess) {
    float elapsed = 0.0f;
    if (cudaEventElapsedTime(&elapsed, ev_start, ev_stop) == cudaSuccess) {
      ms = (double)elapsed;
    }
  }

  cudaEventDestroy(ev_start);
  cudaEventDestroy(ev_stop);
  *start = NULL;
  *stop = NULL;
  return ms;
}

void *unpaper_cuda_stream_get_raw_handle(UnpaperCudaStream *stream) {
  cudaStream_t s = stream_handle(stream);
  return (void *)s;
}
