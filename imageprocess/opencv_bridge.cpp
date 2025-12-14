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

  (void)stream; // Not used in current implementation

  try {
    // Wrap the mask as GpuMat (CV_8UC1 binary mask where foreground_value
    // indicates dark pixels)
    cv::cuda::GpuMat mask(height, width, CV_8UC1,
                          reinterpret_cast<void *>(mask_device), pitch_bytes);

    // Download mask to host, run CCL, then upload results
    // This is needed because cv::cuda::connectedComponents requires contiguous
    // memory and we need to transfer ownership safely
    cv::Mat mask_host;
    mask.download(mask_host);

    // Convert mask to binary (OpenCV CCL expects non-zero = foreground)
    // Our mask has foreground_value for dark pixels
    cv::Mat binary;
    if (foreground_value == 255) {
      binary = mask_host;
    } else {
      cv::compare(mask_host, foreground_value, binary, cv::CMP_EQ);
    }

    // Upload binary mask to GPU via cudaMalloc for OpenCV compatibility
    void *d_binary = nullptr;
    size_t binary_bytes = binary.step * static_cast<size_t>(height);
    cudaError_t err = cudaMalloc(&d_binary, binary_bytes);
    if (err != cudaSuccess || d_binary == nullptr) {
      return false;
    }
    err = cudaMemcpy(d_binary, binary.data, binary_bytes, cudaMemcpyHostToDevice);
    if (err != cudaSuccess) {
      cudaFree(d_binary);
      return false;
    }

    cv::cuda::GpuMat gpu_binary(height, width, CV_8UC1, d_binary, binary.step);

    // Run connected components labeling
    cv::cuda::GpuMat labels;
    cv::cuda::connectedComponents(gpu_binary, labels, 8, CV_32S);

    // Download labels to count component sizes
    cv::Mat labels_host;
    labels.download(labels_host);

    cudaFree(d_binary);

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

    // Apply removal: set small component pixels to background (255 = white)
    // by modifying the original mask in place
    cv::Mat result_mask = mask_host.clone();
    for (int y = 0; y < height; y++) {
      const int32_t *label_row = labels_host.ptr<int32_t>(y);
      uint8_t *mask_row = result_mask.ptr<uint8_t>(y);
      for (int x = 0; x < width; x++) {
        int32_t label = label_row[x];
        if (label > 0 && small_labels.count(label)) {
          mask_row[x] = 0; // Mark as removed (will become white)
        }
      }
    }

    // Upload modified mask back to device
    err = cudaMemcpy(reinterpret_cast<void *>(mask_device), result_mask.data,
                     pitch_bytes * static_cast<size_t>(height),
                     cudaMemcpyHostToDevice);
    if (err != cudaSuccess) {
      return false;
    }

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

  (void)stream; // Not used currently - we download/upload via host to avoid
                // Driver/Runtime API context conflicts

  try {
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

    // Download source from device to host first to avoid Driver/Runtime API
    // context conflicts. The source may be allocated via cuMemAlloc (Driver API)
    // or cudaMalloc (Runtime API). We use cudaMemcpy which handles both via
    // Unified Virtual Addressing (UVA).
    size_t src_bytes = src_pitch_bytes * static_cast<size_t>(src_height);
    cv::Mat src_host(src_height, src_width, cv_type);

    // cudaMemcpy works for both Driver and Runtime API pointers via UVA
    cudaError_t err = cudaMemcpy(src_host.data,
                                 reinterpret_cast<void *>(src_device),
                                 src_bytes, cudaMemcpyDeviceToHost);
    if (err != cudaSuccess) {
      fprintf(stderr,
              "OpenCV CUDA mask extraction: cudaMemcpy D2H failed: %s\n",
              cudaGetErrorString(err));
      return false;
    }

    // Re-upload via cudaMalloc (Runtime API) for OpenCV compatibility
    void *d_src = nullptr;
    err = cudaMalloc(&d_src, src_bytes);
    if (err != cudaSuccess || d_src == nullptr) {
      return false;
    }
    err = cudaMemcpy(d_src, src_host.data, src_bytes, cudaMemcpyHostToDevice);
    if (err != cudaSuccess) {
      cudaFree(d_src);
      return false;
    }

    cv::cuda::GpuMat src(src_height, src_width, cv_type, d_src, src_pitch_bytes);

    cv::cuda::GpuMat gray;
    if (channels == 1) {
      gray = src;
    } else if (channels == 2) {
      std::vector<cv::cuda::GpuMat> planes;
      cv::cuda::split(src, planes);
      gray = planes[0];
    } else {
      std::vector<cv::cuda::GpuMat> planes;
      cv::cuda::split(src, planes);
      cv::cuda::GpuMat max_rg, min_rg;
      cv::cuda::max(planes[0], planes[1], max_rg);
      cv::cuda::min(planes[0], planes[1], min_rg);
      cv::cuda::GpuMat max_rgb, min_rgb;
      cv::cuda::max(max_rg, planes[2], max_rgb);
      cv::cuda::min(min_rg, planes[2], min_rgb);
      cv::cuda::GpuMat sum;
      cv::cuda::add(max_rgb, min_rgb, sum, cv::noArray(), CV_16UC1);
      cv::cuda::GpuMat half;
      sum.convertTo(half, CV_8UC1, 0.5, 0.0);
      gray = half;
    }

    cv::cuda::GpuMat mask;
    cv::cuda::compare(gray, cv::Scalar(min_white_level), mask, cv::CMP_LT);

    // Download mask to CPU and re-upload via cudart for output
    cv::Mat mask_host;
    mask.download(mask_host);

    // Free the temporary source copy
    cudaFree(d_src);

    // Allocate output mask via cudaMalloc (runtime API)
    void *d_mask = nullptr;
    size_t mask_bytes = mask_host.step * static_cast<size_t>(src_height);
    err = cudaMalloc(&d_mask, mask_bytes);
    if (err != cudaSuccess || d_mask == nullptr) {
      return false;
    }

    err = cudaMemcpy(d_mask, mask_host.data, mask_bytes, cudaMemcpyHostToDevice);
    if (err != cudaSuccess) {
      cudaFree(d_mask);
      return false;
    }

    mask_out->device_ptr = reinterpret_cast<uint64_t>(d_mask);
    mask_out->width = src_width;
    mask_out->height = src_height;
    mask_out->pitch_bytes = mask_host.step;
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
#ifdef HAVE_OPENCV_CUDAARITHM
    if (mask->opencv_allocated) {
      cudaFree(reinterpret_cast<void *>(mask->device_ptr));
    } else {
      unpaper_cuda_free(mask->device_ptr);
    }
#else
    unpaper_cuda_free(mask->device_ptr);
#endif
  }
  std::memset(mask, 0, sizeof(*mask));
}
