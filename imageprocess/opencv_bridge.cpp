// SPDX-FileCopyrightText: 2025 The unpaper authors
//
// SPDX-License-Identifier: GPL-2.0-only

#include "imageprocess/opencv_bridge.h"
#include "imageprocess/cuda_kernels_format.h"

#include <cstring>
#include <exception>

#include <opencv2/core/cuda.hpp>
#include <opencv2/core/cuda_stream_accessor.hpp>

#if __has_include(<opencv2/cudaarithm.hpp>)
#include <opencv2/cudaarithm.hpp>
#define HAVE_OPENCV_CUDAARITHM 1
#else
#define HAVE_OPENCV_CUDAARITHM 0
#endif

bool unpaper_opencv_enabled(void) { return true; }

bool unpaper_opencv_cuda_supported(void) {
#if HAVE_OPENCV_CUDAARITHM
  try {
    return cv::cuda::getCudaEnabledDeviceCount() > 0;
  } catch (const std::exception &) {
    return false;
  }
#else
  return false;
#endif
}

#if HAVE_OPENCV_CUDAARITHM
static cv::cuda::Stream wrap_stream(UnpaperCudaStream *stream) {
  if (stream == nullptr) {
    return cv::cuda::Stream::Null();
  }
  void *raw = unpaper_cuda_stream_get_raw_handle(stream);
  if (raw == nullptr) {
    return cv::cuda::Stream::Null();
  }
  cudaStream_t cuda_stream = static_cast<cudaStream_t>(raw);
  return cv::cuda::StreamAccessor::wrapStream(cuda_stream);
}
#endif

bool unpaper_opencv_cuda_ccl(uint64_t mask_device, int width, int height,
                             size_t pitch_bytes, uint8_t foreground_value,
                             uint32_t max_component_size,
                             UnpaperCudaStream *stream,
                             UnpaperOpencvCclStats *stats_out) {
  (void)mask_device;
  (void)width;
  (void)height;
  (void)pitch_bytes;
  (void)foreground_value;
  (void)max_component_size;
  (void)stream;
  if (stats_out != nullptr) {
    std::memset(stats_out, 0, sizeof(*stats_out));
  }
  return false;
}

#if HAVE_OPENCV_CUDAARITHM
static uint8_t compute_lightness(uint8_t r, uint8_t g, uint8_t b) {
  uint8_t max_val = r > g ? (r > b ? r : b) : (g > b ? g : b);
  uint8_t min_val = r < g ? (r < b ? r : b) : (g < b ? g : b);
  return static_cast<uint8_t>((static_cast<uint32_t>(max_val) +
                               static_cast<uint32_t>(min_val)) /
                              2u);
}
#endif

bool unpaper_opencv_extract_dark_mask(uint64_t src_device, int src_width,
                                      int src_height, size_t src_pitch_bytes,
                                      int src_format, uint8_t min_white_level,
                                      UnpaperCudaStream *stream,
                                      UnpaperOpencvMask *mask_out) {
  if (mask_out == nullptr) {
    return false;
  }
  std::memset(mask_out, 0, sizeof(*mask_out));

#if HAVE_OPENCV_CUDAARITHM
  if (src_device == 0 || src_width <= 0 || src_height <= 0) {
    return false;
  }

  try {
    cv::cuda::Stream cv_stream = wrap_stream(stream);
    auto fmt = static_cast<UnpaperCudaFormat>(src_format);

    int cv_type = -1;
    int channels = 1;
    switch (fmt) {
    case UNPAPER_CUDA_FMT_GRAY8:
      cv_type = CV_8UC1;
      channels = 1;
      break;
    case UNPAPER_CUDA_FMT_Y400A:
      cv_type = CV_8UC2;
      channels = 2;
      break;
    case UNPAPER_CUDA_FMT_RGB24:
      cv_type = CV_8UC3;
      channels = 3;
      break;
    default:
      return false;
    }

    cv::cuda::GpuMat src(src_height, src_width, cv_type,
                         reinterpret_cast<void *>(src_device), src_pitch_bytes);

    cv::cuda::GpuMat gray;
    if (channels == 1) {
      gray = src;
    } else if (channels == 2) {
      std::vector<cv::cuda::GpuMat> planes;
      cv::cuda::split(src, planes, cv_stream);
      gray = planes[0];
    } else {
      std::vector<cv::cuda::GpuMat> planes;
      cv::cuda::split(src, planes, cv_stream);
      cv::cuda::GpuMat max_rg, min_rg;
      cv::cuda::max(planes[0], planes[1], max_rg, cv_stream);
      cv::cuda::min(planes[0], planes[1], min_rg, cv_stream);
      cv::cuda::GpuMat max_rgb, min_rgb;
      cv::cuda::max(max_rg, planes[2], max_rgb, cv_stream);
      cv::cuda::min(min_rg, planes[2], min_rgb, cv_stream);
      cv::cuda::GpuMat sum;
      cv::cuda::add(max_rgb, min_rgb, sum, cv::noArray(), CV_16UC1, cv_stream);
      cv::cuda::GpuMat half;
      sum.convertTo(half, CV_8UC1, 0.5, 0.0, cv_stream);
      gray = half;
    }

    cv::cuda::GpuMat mask;
    cv::cuda::compare(gray, cv::Scalar(min_white_level), mask, cv::CMP_LT,
                      cv_stream);

    size_t mask_pitch = mask.step;
    uint64_t mask_dptr = unpaper_cuda_malloc(mask_pitch * src_height);
    if (mask_dptr == 0) {
      return false;
    }

    cv::cuda::GpuMat mask_out_mat(src_height, src_width, CV_8UC1,
                                  reinterpret_cast<void *>(mask_dptr),
                                  mask_pitch);
    mask.copyTo(mask_out_mat, cv_stream);

    cv_stream.waitForCompletion();

    mask_out->device_ptr = mask_dptr;
    mask_out->width = src_width;
    mask_out->height = src_height;
    mask_out->pitch_bytes = mask_pitch;
    return true;
  } catch (const std::exception &) {
    return false;
  }
#else
  (void)src_device;
  (void)src_width;
  (void)src_height;
  (void)src_pitch_bytes;
  (void)src_format;
  (void)min_white_level;
  (void)stream;
  return false;
#endif
}

void unpaper_opencv_mask_free(UnpaperOpencvMask *mask) {
  if (mask == nullptr) {
    return;
  }
  if (mask->device_ptr != 0) {
    unpaper_cuda_free(mask->device_ptr);
  }
  std::memset(mask, 0, sizeof(*mask));
}
