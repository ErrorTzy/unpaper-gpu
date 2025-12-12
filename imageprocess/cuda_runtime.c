// SPDX-FileCopyrightText: 2025 The unpaper authors
//
// SPDX-License-Identifier: GPL-2.0-only

#include "imageprocess/cuda_runtime.h"

#include <dlfcn.h>
#include <stdbool.h>
#include <stdio.h>
#include <stdint.h>
#include <string.h>

#include "lib/logging.h"

typedef int CUresult;
typedef int CUdevice;
typedef struct CUctx_st *CUcontext;
typedef struct CUstream_st *CUstream;
typedef uint64_t CUdeviceptr;

enum { CUDA_SUCCESS = 0 };

typedef CUresult (*cuInit_fn)(unsigned int flags);
typedef CUresult (*cuDeviceGetCount_fn)(int *count);
typedef CUresult (*cuDeviceGet_fn)(CUdevice *device, int ordinal);
typedef CUresult (*cuCtxCreate_fn)(CUcontext *pctx, unsigned int flags,
                                  CUdevice dev);
typedef CUresult (*cuCtxDestroy_fn)(CUcontext ctx);
typedef CUresult (*cuGetErrorString_fn)(CUresult error, const char **pStr);

typedef CUresult (*cuMemAlloc_fn)(CUdeviceptr *dptr, size_t bytesize);
typedef CUresult (*cuMemFree_fn)(CUdeviceptr dptr);
typedef CUresult (*cuMemcpyHtoD_fn)(CUdeviceptr dstDevice, const void *srcHost,
                                   size_t ByteCount);
typedef CUresult (*cuMemcpyDtoH_fn)(void *dstHost, CUdeviceptr srcDevice,
                                   size_t ByteCount);

typedef CUresult (*cuStreamCreate_fn)(CUstream *phStream, unsigned int flags);
typedef CUresult (*cuStreamDestroy_fn)(CUstream hStream);
typedef CUresult (*cuStreamSynchronize_fn)(CUstream hStream);
typedef CUresult (*cuMemcpyHtoDAsync_fn)(CUdeviceptr dstDevice,
                                        const void *srcHost, size_t ByteCount,
                                        CUstream hStream);
typedef CUresult (*cuMemcpyDtoHAsync_fn)(void *dstHost, CUdeviceptr srcDevice,
                                        size_t ByteCount, CUstream hStream);

typedef struct {
  void *lib;

  cuInit_fn cuInit;
  cuDeviceGetCount_fn cuDeviceGetCount;
  cuDeviceGet_fn cuDeviceGet;
  cuCtxCreate_fn cuCtxCreate;
  cuCtxDestroy_fn cuCtxDestroy;
  cuGetErrorString_fn cuGetErrorString;

  cuMemAlloc_fn cuMemAlloc;
  cuMemFree_fn cuMemFree;
  cuMemcpyHtoD_fn cuMemcpyHtoD;
  cuMemcpyDtoH_fn cuMemcpyDtoH;

  cuStreamCreate_fn cuStreamCreate;
  cuStreamDestroy_fn cuStreamDestroy;
  cuStreamSynchronize_fn cuStreamSynchronize;
  cuMemcpyHtoDAsync_fn cuMemcpyHtoDAsync;
  cuMemcpyDtoHAsync_fn cuMemcpyDtoHAsync;
} CudaSymbols;

static CudaSymbols syms;
static bool syms_loaded = false;
static bool cuda_initialized = false;
static CUcontext cuda_ctx = NULL;
static CUstream cuda_stream = NULL;

static void *load_sym(const char *name) {
  void *p = dlsym(syms.lib, name);
  if (p != NULL) {
    return p;
  }

  char alt[128];
  snprintf(alt, sizeof(alt), "%s_v2", name);
  return dlsym(syms.lib, alt);
}

static const char *cu_err(CUresult res) {
  static char buf[128];
  if (syms.cuGetErrorString != NULL) {
    const char *s = NULL;
    if (syms.cuGetErrorString(res, &s) == CUDA_SUCCESS && s != NULL) {
      return s;
    }
  }
  snprintf(buf, sizeof(buf), "CUDA error %d", res);
  return buf;
}

static bool load_cuda_driver_symbols(void) {
  if (syms_loaded) {
    return syms.lib != NULL;
  }
  syms_loaded = true;

  syms.lib = dlopen("libcuda.so.1", RTLD_LAZY);
  if (syms.lib == NULL) {
    syms.lib = dlopen("libcuda.so", RTLD_LAZY);
  }
  if (syms.lib == NULL) {
    return false;
  }

  syms.cuInit = (cuInit_fn)load_sym("cuInit");
  syms.cuDeviceGetCount = (cuDeviceGetCount_fn)load_sym("cuDeviceGetCount");
  syms.cuDeviceGet = (cuDeviceGet_fn)load_sym("cuDeviceGet");
  syms.cuCtxCreate = (cuCtxCreate_fn)load_sym("cuCtxCreate");
  syms.cuCtxDestroy = (cuCtxDestroy_fn)load_sym("cuCtxDestroy");
  syms.cuGetErrorString = (cuGetErrorString_fn)load_sym("cuGetErrorString");

  syms.cuMemAlloc = (cuMemAlloc_fn)load_sym("cuMemAlloc");
  syms.cuMemFree = (cuMemFree_fn)load_sym("cuMemFree");
  syms.cuMemcpyHtoD = (cuMemcpyHtoD_fn)load_sym("cuMemcpyHtoD");
  syms.cuMemcpyDtoH = (cuMemcpyDtoH_fn)load_sym("cuMemcpyDtoH");

  syms.cuStreamCreate = (cuStreamCreate_fn)load_sym("cuStreamCreate");
  syms.cuStreamDestroy = (cuStreamDestroy_fn)load_sym("cuStreamDestroy");
  syms.cuStreamSynchronize =
      (cuStreamSynchronize_fn)load_sym("cuStreamSynchronize");
  syms.cuMemcpyHtoDAsync = (cuMemcpyHtoDAsync_fn)load_sym("cuMemcpyHtoDAsync");
  syms.cuMemcpyDtoHAsync = (cuMemcpyDtoHAsync_fn)load_sym("cuMemcpyDtoHAsync");

  if (syms.cuInit == NULL || syms.cuDeviceGetCount == NULL ||
      syms.cuDeviceGet == NULL || syms.cuCtxCreate == NULL ||
      syms.cuCtxDestroy == NULL || syms.cuMemAlloc == NULL ||
      syms.cuMemFree == NULL || syms.cuMemcpyHtoD == NULL ||
      syms.cuMemcpyDtoH == NULL) {
    return false;
  }

  return true;
}

UnpaperCudaInitStatus unpaper_cuda_try_init(void) {
  if (cuda_initialized) {
    return UNPAPER_CUDA_INIT_OK;
  }

  if (!load_cuda_driver_symbols()) {
    return UNPAPER_CUDA_INIT_NO_RUNTIME;
  }

  CUresult res = syms.cuInit(0);
  if (res != CUDA_SUCCESS) {
    return UNPAPER_CUDA_INIT_ERROR;
  }

  int device_count = 0;
  res = syms.cuDeviceGetCount(&device_count);
  if (res != CUDA_SUCCESS) {
    return UNPAPER_CUDA_INIT_ERROR;
  }
  if (device_count <= 0) {
    return UNPAPER_CUDA_INIT_NO_DEVICE;
  }

  CUdevice dev = 0;
  res = syms.cuDeviceGet(&dev, 0);
  if (res != CUDA_SUCCESS) {
    return UNPAPER_CUDA_INIT_ERROR;
  }

  res = syms.cuCtxCreate(&cuda_ctx, 0, dev);
  if (res != CUDA_SUCCESS) {
    return UNPAPER_CUDA_INIT_ERROR;
  }

  if (syms.cuStreamCreate != NULL) {
    res = syms.cuStreamCreate(&cuda_stream, 0);
    if (res != CUDA_SUCCESS) {
      (void)syms.cuCtxDestroy(cuda_ctx);
      cuda_ctx = NULL;
      return UNPAPER_CUDA_INIT_ERROR;
    }
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

  CUdeviceptr dptr = 0;
  CUresult res = syms.cuMemAlloc(&dptr, bytes);
  if (res != CUDA_SUCCESS) {
    errOutput("CUDA allocation failed: %s", cu_err(res));
  }
  return (uint64_t)dptr;
}

void unpaper_cuda_free(uint64_t dptr) {
  if (dptr == 0) {
    return;
  }
  if (!cuda_initialized) {
    return;
  }
  (void)syms.cuMemFree((CUdeviceptr)dptr);
}

void unpaper_cuda_memcpy_h2d(uint64_t dst, const void *src, size_t bytes) {
  UnpaperCudaInitStatus st = unpaper_cuda_try_init();
  if (st != UNPAPER_CUDA_INIT_OK) {
    errOutput("%s", unpaper_cuda_init_status_string(st));
  }

  CUresult res;
  if (cuda_stream != NULL && syms.cuMemcpyHtoDAsync != NULL &&
      syms.cuStreamSynchronize != NULL) {
    res = syms.cuMemcpyHtoDAsync((CUdeviceptr)dst, src, bytes, cuda_stream);
    if (res == CUDA_SUCCESS) {
      res = syms.cuStreamSynchronize(cuda_stream);
    }
  } else {
    res = syms.cuMemcpyHtoD((CUdeviceptr)dst, src, bytes);
  }
  if (res != CUDA_SUCCESS) {
    errOutput("CUDA memcpy HtoD failed: %s", cu_err(res));
  }
}

void unpaper_cuda_memcpy_d2h(void *dst, uint64_t src, size_t bytes) {
  UnpaperCudaInitStatus st = unpaper_cuda_try_init();
  if (st != UNPAPER_CUDA_INIT_OK) {
    errOutput("%s", unpaper_cuda_init_status_string(st));
  }

  CUresult res;
  if (cuda_stream != NULL && syms.cuMemcpyDtoHAsync != NULL &&
      syms.cuStreamSynchronize != NULL) {
    res = syms.cuMemcpyDtoHAsync(dst, (CUdeviceptr)src, bytes, cuda_stream);
    if (res == CUDA_SUCCESS) {
      res = syms.cuStreamSynchronize(cuda_stream);
    }
  } else {
    res = syms.cuMemcpyDtoH(dst, (CUdeviceptr)src, bytes);
  }
  if (res != CUDA_SUCCESS) {
    errOutput("CUDA memcpy DtoH failed: %s", cu_err(res));
  }
}
