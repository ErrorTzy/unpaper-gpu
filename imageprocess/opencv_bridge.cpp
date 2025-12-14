// SPDX-FileCopyrightText: 2025 The unpaper authors
//
// SPDX-License-Identifier: GPL-2.0-only

#include "imageprocess/opencv_bridge.h"
#include "imageprocess/cuda_kernels_format.h"

#include <cstring>
#include <exception>
#include <unordered_map>
#include <unordered_set>

#include <opencv2/core/cuda.hpp>
#include <opencv2/core/cuda_stream_accessor.hpp>
#include <cuda_runtime.h>

// OpenCV defines HAVE_OPENCV_* in opencv_modules.hpp if modules are present
// (defined without a value, so use #ifdef)
#if __has_include(<opencv2/cudaarithm.hpp>)
#include <opencv2/cudaarithm.hpp>
#endif

#if __has_include(<opencv2/cudaimgproc.hpp>)
#include <opencv2/cudaimgproc.hpp>
#endif

bool unpaper_opencv_enabled(void) { return true; }

bool unpaper_opencv_cuda_supported(void) {
#ifdef HAVE_OPENCV_CUDAARITHM
  try {
    return cv::cuda::getCudaEnabledDeviceCount() > 0;
  } catch (const std::exception &) {
    return false;
  }
#else
  return false;
#endif
}

bool unpaper_opencv_ccl_supported(void) {
#if defined(HAVE_OPENCV_CUDAIMGPROC) && defined(HAVE_OPENCV_CUDAARITHM)
  return unpaper_opencv_cuda_supported();
#else
  return false;
#endif
}

bool unpaper_opencv_cuda_ccl(uint64_t mask_device, int width, int height,
                             size_t pitch_bytes, uint8_t foreground_value,
                             uint32_t max_component_size,
                             UnpaperCudaStream *stream,
                             UnpaperOpencvCclStats *stats_out) {
  if (stats_out != nullptr) {
    std::memset(stats_out, 0, sizeof(*stats_out));
  }

#if defined(HAVE_OPENCV_CUDAIMGPROC) && defined(HAVE_OPENCV_CUDAARITHM)
  if (mask_device == 0 || width <= 0 || height <= 0) {
    return false;
  }

  try {
    // Get OpenCV CUDA stream from our stream handle
    cudaStream_t cuda_stream = nullptr;
    if (stream != nullptr) {
      cuda_stream =
          static_cast<cudaStream_t>(unpaper_cuda_stream_get_raw_handle(stream));
    }
    cv::cuda::Stream cv_stream = cv::cuda::StreamAccessor::wrapStream(cuda_stream);

    // Wrap the mask directly as GpuMat - memory is now compatible (Runtime API)
    cv::cuda::GpuMat mask(height, width, CV_8UC1,
                          reinterpret_cast<void *>(mask_device), pitch_bytes);

    // Convert mask to binary if needed (OpenCV CCL expects non-zero = foreground)
    cv::cuda::GpuMat binary;
    if (foreground_value == 255) {
      binary = mask;
    } else {
      cv::cuda::compare(mask, cv::Scalar(foreground_value), binary, cv::CMP_EQ,
                        cv_stream);
    }

    // Run connected components labeling
    // Note: OpenCV's connectedComponents doesn't accept a stream parameter
    cv::cuda::GpuMat labels;
    cv::cuda::connectedComponents(binary, labels, 8, CV_32S);

    // Download labels to count component sizes (CPU-side for determinism)
    cv::Mat labels_host;
    labels.download(labels_host, cv_stream);
    cv_stream.waitForCompletion();

    // Count component sizes
    std::unordered_map<int32_t, uint32_t> component_sizes;
    for (int y = 0; y < height; y++) {
      const int32_t *row = labels_host.ptr<int32_t>(y);
      for (int x = 0; x < width; x++) {
        int32_t label = row[x];
        if (label > 0) { // label 0 is background
          component_sizes[label]++;
        }
      }
    }

    // Find small components to remove
    std::unordered_set<int32_t> small_labels;
    for (const auto &kv : component_sizes) {
      if (kv.second <= max_component_size) {
        small_labels.insert(kv.first);
      }
    }

    if (stats_out != nullptr) {
      stats_out->label_count = static_cast<int>(component_sizes.size());
      stats_out->removed_components = static_cast<int>(small_labels.size());
    }

    // Apply removal: set small component pixels to background (0 = removed)
    // Download mask, modify, re-upload
    cv::Mat mask_host;
    mask.download(mask_host, cv_stream);
    cv_stream.waitForCompletion();

    for (int y = 0; y < height; y++) {
      const int32_t *label_row = labels_host.ptr<int32_t>(y);
      uint8_t *mask_row = mask_host.ptr<uint8_t>(y);
      for (int x = 0; x < width; x++) {
        int32_t label = label_row[x];
        if (label > 0 && small_labels.count(label)) {
          mask_row[x] = 0; // Mark as removed (will become white)
        }
      }
    }

    // Upload modified mask back to device
    mask.upload(mask_host, cv_stream);
    cv_stream.waitForCompletion();

    return true;
  } catch (const std::exception &e) {
    fprintf(stderr, "OpenCV CUDA CCL failed: %s\n", e.what());
    return false;
  }
#else
  (void)mask_device;
  (void)width;
  (void)height;
  (void)pitch_bytes;
  (void)foreground_value;
  (void)max_component_size;
  (void)stream;
  return false;
#endif
}

bool unpaper_opencv_extract_dark_mask(uint64_t src_device, int src_width,
                                      int src_height, size_t src_pitch_bytes,
                                      int src_format, uint8_t min_white_level,
                                      UnpaperCudaStream *stream,
                                      UnpaperOpencvMask *mask_out) {
  if (mask_out == nullptr) {
    return false;
  }
  std::memset(mask_out, 0, sizeof(*mask_out));

#ifdef HAVE_OPENCV_CUDAARITHM
  if (src_device == 0 || src_width <= 0 || src_height <= 0) {
    return false;
  }

  try {
    // Get OpenCV CUDA stream from our stream handle
    cudaStream_t cuda_stream = nullptr;
    if (stream != nullptr) {
      cuda_stream =
          static_cast<cudaStream_t>(unpaper_cuda_stream_get_raw_handle(stream));
    }
    cv::cuda::Stream cv_stream = cv::cuda::StreamAccessor::wrapStream(cuda_stream);

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

    // Wrap source directly as GpuMat - memory is now compatible (Runtime API)
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
    cv_stream.waitForCompletion();

    // Allocate output mask via cudaMalloc (Runtime API)
    void *d_mask = nullptr;
    size_t mask_pitch = mask.step;
    size_t mask_bytes = mask_pitch * static_cast<size_t>(src_height);
    cudaError_t err = cudaMalloc(&d_mask, mask_bytes);
    if (err != cudaSuccess || d_mask == nullptr) {
      return false;
    }

    // Copy mask data to output buffer
    err = cudaMemcpy2D(d_mask, mask_pitch, mask.data, mask.step, src_width,
                       src_height, cudaMemcpyDeviceToDevice);
    if (err != cudaSuccess) {
      cudaFree(d_mask);
      return false;
    }

    mask_out->device_ptr = reinterpret_cast<uint64_t>(d_mask);
    mask_out->width = src_width;
    mask_out->height = src_height;
    mask_out->pitch_bytes = mask_pitch;
    mask_out->opencv_allocated = true;
    return true;
  } catch (const std::exception &e) {
    fprintf(stderr, "OpenCV CUDA mask extraction failed: %s\n", e.what());
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
    if (mask->opencv_allocated) {
      cudaFree(reinterpret_cast<void *>(mask->device_ptr));
    } else {
      unpaper_cuda_free(mask->device_ptr);
    }
  }
  std::memset(mask, 0, sizeof(*mask));
}
