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
typedef struct CUmod_st *CUmodule;
typedef struct CUfunc_st *CUfunction;
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
typedef CUresult (*cuMemcpyDtoD_fn)(CUdeviceptr dstDevice, CUdeviceptr srcDevice,
                                    size_t ByteCount);
typedef CUresult (*cuMemsetD8_fn)(CUdeviceptr dstDevice, unsigned char uc,
                                 size_t N);

typedef CUresult (*cuStreamCreate_fn)(CUstream *phStream, unsigned int flags);
typedef CUresult (*cuStreamDestroy_fn)(CUstream hStream);
typedef CUresult (*cuStreamSynchronize_fn)(CUstream hStream);
typedef CUresult (*cuMemcpyHtoDAsync_fn)(CUdeviceptr dstDevice,
                                        const void *srcHost, size_t ByteCount,
                                        CUstream hStream);
typedef CUresult (*cuMemcpyDtoHAsync_fn)(void *dstHost, CUdeviceptr srcDevice,
                                        size_t ByteCount, CUstream hStream);

typedef CUresult (*cuCtxSynchronize_fn)(void);
typedef CUresult (*cuModuleLoadData_fn)(CUmodule *module, const void *image);
typedef CUresult (*cuModuleUnload_fn)(CUmodule hmod);
typedef CUresult (*cuModuleGetFunction_fn)(CUfunction *hfunc, CUmodule hmod,
                                           const char *name);
typedef CUresult (*cuLaunchKernel_fn)(
    CUfunction f, unsigned int gridDimX, unsigned int gridDimY,
    unsigned int gridDimZ, unsigned int blockDimX, unsigned int blockDimY,
    unsigned int blockDimZ, unsigned int sharedMemBytes, CUstream hStream,
    void **kernelParams, void **extra);

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
  cuMemcpyDtoD_fn cuMemcpyDtoD;
  cuMemsetD8_fn cuMemsetD8;

  cuStreamCreate_fn cuStreamCreate;
  cuStreamDestroy_fn cuStreamDestroy;
  cuStreamSynchronize_fn cuStreamSynchronize;
  cuMemcpyHtoDAsync_fn cuMemcpyHtoDAsync;
  cuMemcpyDtoHAsync_fn cuMemcpyDtoHAsync;

  cuCtxSynchronize_fn cuCtxSynchronize;
  cuModuleLoadData_fn cuModuleLoadData;
  cuModuleUnload_fn cuModuleUnload;
  cuModuleGetFunction_fn cuModuleGetFunction;
  cuLaunchKernel_fn cuLaunchKernel;
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
  syms.cuMemcpyDtoD = (cuMemcpyDtoD_fn)load_sym("cuMemcpyDtoD");
  syms.cuMemsetD8 = (cuMemsetD8_fn)load_sym("cuMemsetD8");

  syms.cuStreamCreate = (cuStreamCreate_fn)load_sym("cuStreamCreate");
  syms.cuStreamDestroy = (cuStreamDestroy_fn)load_sym("cuStreamDestroy");
  syms.cuStreamSynchronize =
      (cuStreamSynchronize_fn)load_sym("cuStreamSynchronize");
  syms.cuMemcpyHtoDAsync = (cuMemcpyHtoDAsync_fn)load_sym("cuMemcpyHtoDAsync");
  syms.cuMemcpyDtoHAsync = (cuMemcpyDtoHAsync_fn)load_sym("cuMemcpyDtoHAsync");

  syms.cuCtxSynchronize = (cuCtxSynchronize_fn)load_sym("cuCtxSynchronize");
  syms.cuModuleLoadData = (cuModuleLoadData_fn)load_sym("cuModuleLoadData");
  syms.cuModuleUnload = (cuModuleUnload_fn)load_sym("cuModuleUnload");
  syms.cuModuleGetFunction =
      (cuModuleGetFunction_fn)load_sym("cuModuleGetFunction");
  syms.cuLaunchKernel = (cuLaunchKernel_fn)load_sym("cuLaunchKernel");

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

void unpaper_cuda_memcpy_d2d(uint64_t dst, uint64_t src, size_t bytes) {
  UnpaperCudaInitStatus st = unpaper_cuda_try_init();
  if (st != UNPAPER_CUDA_INIT_OK) {
    errOutput("%s", unpaper_cuda_init_status_string(st));
  }
  if (syms.cuMemcpyDtoD == NULL) {
    errOutput("CUDA memcpy DtoD is unavailable.");
  }

  CUresult res = syms.cuMemcpyDtoD((CUdeviceptr)dst, (CUdeviceptr)src, bytes);
  if (res != CUDA_SUCCESS) {
    errOutput("CUDA memcpy DtoD failed: %s", cu_err(res));
  }
}

void unpaper_cuda_memset_d8(uint64_t dst, uint8_t value, size_t bytes) {
  UnpaperCudaInitStatus st = unpaper_cuda_try_init();
  if (st != UNPAPER_CUDA_INIT_OK) {
    errOutput("%s", unpaper_cuda_init_status_string(st));
  }
  if (syms.cuMemsetD8 == NULL) {
    errOutput("CUDA memset is unavailable.");
  }

  CUresult res = syms.cuMemsetD8((CUdeviceptr)dst, value, bytes);
  if (res != CUDA_SUCCESS) {
    errOutput("CUDA memset failed: %s", cu_err(res));
  }
}

void *unpaper_cuda_module_load_ptx(const char *ptx) {
  UnpaperCudaInitStatus st = unpaper_cuda_try_init();
  if (st != UNPAPER_CUDA_INIT_OK) {
    errOutput("%s", unpaper_cuda_init_status_string(st));
  }
  if (syms.cuModuleLoadData == NULL) {
    errOutput("CUDA module loading is unavailable.");
  }

  CUmodule mod = NULL;
  CUresult res = syms.cuModuleLoadData(&mod, (const void *)ptx);
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
  if (syms.cuModuleUnload == NULL) {
    return;
  }
  (void)syms.cuModuleUnload((CUmodule)module);
}

void *unpaper_cuda_module_get_function(void *module, const char *name) {
  UnpaperCudaInitStatus st = unpaper_cuda_try_init();
  if (st != UNPAPER_CUDA_INIT_OK) {
    errOutput("%s", unpaper_cuda_init_status_string(st));
  }
  if (module == NULL || name == NULL) {
    errOutput("invalid CUDA function lookup.");
  }
  if (syms.cuModuleGetFunction == NULL) {
    errOutput("CUDA function lookup is unavailable.");
  }

  CUfunction fn = NULL;
  CUresult res = syms.cuModuleGetFunction(&fn, (CUmodule)module, name);
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
  if (syms.cuLaunchKernel == NULL) {
    errOutput("CUDA kernel launch is unavailable.");
  }

  CUresult res =
      syms.cuLaunchKernel((CUfunction)func, grid_x, grid_y, grid_z, block_x,
                          block_y, block_z, 0, cuda_stream, kernel_params, NULL);
  if (res != CUDA_SUCCESS) {
    errOutput("CUDA kernel launch failed: %s", cu_err(res));
  }

  if (cuda_stream != NULL && syms.cuStreamSynchronize != NULL) {
    res = syms.cuStreamSynchronize(cuda_stream);
    if (res != CUDA_SUCCESS) {
      errOutput("CUDA stream synchronize failed: %s", cu_err(res));
    }
  } else if (syms.cuCtxSynchronize != NULL) {
    res = syms.cuCtxSynchronize();
    if (res != CUDA_SUCCESS) {
      errOutput("CUDA context synchronize failed: %s", cu_err(res));
    }
  }
}
