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

static inline int opencv_type_from_format(UnpaperCudaFormat fmt) {
  switch (fmt) {
  case UNPAPER_CUDA_FMT_GRAY8:
    return CV_8UC1;
  case UNPAPER_CUDA_FMT_Y400A:
    return CV_8UC2;
  case UNPAPER_CUDA_FMT_RGB24:
    return CV_8UC3;
  default:
    return -1;
  }
}

static inline cv::cuda::Stream get_cv_stream(UnpaperCudaStream *stream) {
  cudaStream_t cuda_stream = nullptr;
  if (stream != nullptr) {
    cuda_stream =
        static_cast<cudaStream_t>(unpaper_cuda_stream_get_raw_handle(stream));
  }
  return cv::cuda::StreamAccessor::wrapStream(cuda_stream);
}

// Helper: compute sum of rectangle from integral image (CPU side)
// Integral image is (height+1) x (width+1) with first row and column being 0
static inline int64_t integral_rect_sum(const cv::Mat &integral, int x0, int y0,
                                        int x1, int y1) {
  // Clamp coordinates to valid range
  x0 = std::max(0, x0);
  y0 = std::max(0, y0);
  x1 = std::min(x1, integral.cols - 2);
  y1 = std::min(y1, integral.rows - 2);
  if (x0 > x1 || y0 > y1) {
    return 0;
  }
  // Integral image formula: sum = I[y1+1][x1+1] - I[y0][x1+1] - I[y1+1][x0] + I[y0][x0]
  int64_t a = integral.at<int32_t>(y0, x0);
  int64_t b = integral.at<int32_t>(y0, x1 + 1);
  int64_t c = integral.at<int32_t>(y1 + 1, x0);
  int64_t d = integral.at<int32_t>(y1 + 1, x1 + 1);
  return d - b - c + a;
}

bool unpaper_opencv_grayfilter(uint64_t src_device, int width, int height,
                               size_t pitch_bytes, int format, int tile_width,
                               int tile_height, int step_x, int step_y,
                               uint8_t black_threshold, uint8_t gray_threshold,
                               UnpaperCudaStream *stream) {
#ifdef HAVE_OPENCV_CUDAARITHM
  auto fmt = static_cast<UnpaperCudaFormat>(format);
  int cv_type = opencv_type_from_format(fmt);
  if (cv_type < 0 || src_device == 0) {
    return false;
  }

  if (tile_width <= 0 || tile_height <= 0 || step_x <= 0 || step_y <= 0) {
    return true; // Nothing to do
  }

  try {
    cv::cuda::Stream cv_stream = get_cv_stream(stream);

    // Wrap source image
    cv::cuda::GpuMat src(height, width, cv_type,
                         reinterpret_cast<void *>(src_device), pitch_bytes);

    // Convert to grayscale for analysis
    cv::cuda::GpuMat gray;
    if (cv_type == CV_8UC1) {
      gray = src;
    } else if (cv_type == CV_8UC2) {
      // Y400A: extract Y channel
      std::vector<cv::cuda::GpuMat> channels;
      cv::cuda::split(src, channels, cv_stream);
      gray = channels[0];
    } else {
#ifdef HAVE_OPENCV_CUDAIMGPROC
      // RGB24: use cvtColor for grayscale conversion
      cv::cuda::cvtColor(src, gray, cv::COLOR_RGB2GRAY, 0, cv_stream);
#else
      // Fallback: compute grayscale as average
      std::vector<cv::cuda::GpuMat> channels;
      cv::cuda::split(src, channels, cv_stream);
      cv::cuda::GpuMat c0_32f, c1_32f, c2_32f, sum_32f;
      channels[0].convertTo(c0_32f, CV_32FC1, 1.0, 0, cv_stream);
      channels[1].convertTo(c1_32f, CV_32FC1, 1.0, 0, cv_stream);
      channels[2].convertTo(c2_32f, CV_32FC1, 1.0, 0, cv_stream);
      cv::cuda::add(c0_32f, c1_32f, sum_32f, cv::noArray(), -1, cv_stream);
      cv::cuda::add(sum_32f, c2_32f, sum_32f, cv::noArray(), -1, cv_stream);
      sum_32f.convertTo(gray, CV_8UC1, 1.0 / 3.0, 0, cv_stream);
#endif
    }

    // Create binary mask: pixels <= black_threshold -> 255, else 0
    cv::cuda::GpuMat dark_mask;
    cv::cuda::threshold(gray, dark_mask, static_cast<double>(black_threshold),
                        255.0, cv::THRESH_BINARY_INV, cv_stream);

    // Compute integral images on CPU (OpenCV CUDA integral is limited)
    cv::Mat gray_host, dark_mask_host;
    gray.download(gray_host, cv_stream);
    dark_mask.download(dark_mask_host, cv_stream);
    cv_stream.waitForCompletion();

    cv::Mat gray_integral, dark_integral;
    cv::integral(gray_host, gray_integral, CV_32S);
    cv::integral(dark_mask_host, dark_integral, CV_32S);

    // Scan tiles and collect those that need wiping
    std::vector<cv::Rect> tiles_to_wipe;
    const int tile_pixels = tile_width * tile_height;

    for (int y = 0; y <= height - tile_height; y += step_y) {
      for (int x = 0; x <= width - tile_width; x += step_x) {
        int x1 = x + tile_width - 1;
        int y1 = y + tile_height - 1;

        // Count dark pixels in tile using integral image
        int64_t dark_sum = integral_rect_sum(dark_integral, x, y, x1, y1);
        int64_t dark_count = dark_sum / 255; // Each dark pixel contributes 255

        if (dark_count == 0) {
          // No black pixels, check average lightness
          int64_t lightness_sum = integral_rect_sum(gray_integral, x, y, x1, y1);
          uint8_t avg_lightness =
              static_cast<uint8_t>(lightness_sum / tile_pixels);
          uint8_t inverse_lightness = 255 - avg_lightness;

          if (inverse_lightness < gray_threshold) {
            tiles_to_wipe.push_back(cv::Rect(x, y, tile_width, tile_height));
          }
        }
      }
    }

    // Wipe collected tiles on GPU
    for (const auto &tile : tiles_to_wipe) {
      cv::cuda::GpuMat roi = src(tile);
      if (cv_type == CV_8UC1) {
        roi.setTo(cv::Scalar(255), cv_stream);
      } else if (cv_type == CV_8UC2) {
        roi.setTo(cv::Scalar(255, 255), cv_stream);
      } else {
        roi.setTo(cv::Scalar(255, 255, 255), cv_stream);
      }
    }

    cv_stream.waitForCompletion();
    return true;
  } catch (const std::exception &e) {
    fprintf(stderr, "OpenCV grayfilter failed: %s\n", e.what());
    return false;
  }
#else
  (void)src_device;
  (void)width;
  (void)height;
  (void)pitch_bytes;
  (void)format;
  (void)tile_width;
  (void)tile_height;
  (void)step_x;
  (void)step_y;
  (void)black_threshold;
  (void)gray_threshold;
  (void)stream;
  return false;
#endif
}

bool unpaper_opencv_blurfilter(uint64_t src_device, int width, int height,
                               size_t pitch_bytes, int format, int block_width,
                               int block_height, int step_x, int step_y,
                               uint8_t white_threshold, float intensity,
                               UnpaperCudaStream *stream) {
#ifdef HAVE_OPENCV_CUDAARITHM
  auto fmt = static_cast<UnpaperCudaFormat>(format);
  int cv_type = opencv_type_from_format(fmt);
  if (cv_type < 0 || src_device == 0) {
    return false;
  }

  if (block_width <= 0 || block_height <= 0) {
    return true; // Nothing to do
  }

  try {
    cv::cuda::Stream cv_stream = get_cv_stream(stream);

    // Wrap source image
    cv::cuda::GpuMat src(height, width, cv_type,
                         reinterpret_cast<void *>(src_device), pitch_bytes);

    // Convert to grayscale
    cv::cuda::GpuMat gray;
    if (cv_type == CV_8UC1) {
      gray = src;
    } else if (cv_type == CV_8UC2) {
      std::vector<cv::cuda::GpuMat> channels;
      cv::cuda::split(src, channels, cv_stream);
      gray = channels[0];
    } else {
#ifdef HAVE_OPENCV_CUDAIMGPROC
      // RGB24: use cvtColor for grayscale conversion
      cv::cuda::cvtColor(src, gray, cv::COLOR_RGB2GRAY, 0, cv_stream);
#else
      // Fallback: compute grayscale as average
      std::vector<cv::cuda::GpuMat> channels;
      cv::cuda::split(src, channels, cv_stream);
      cv::cuda::GpuMat c0_32f, c1_32f, c2_32f, sum_32f;
      channels[0].convertTo(c0_32f, CV_32FC1, 1.0, 0, cv_stream);
      channels[1].convertTo(c1_32f, CV_32FC1, 1.0, 0, cv_stream);
      channels[2].convertTo(c2_32f, CV_32FC1, 1.0, 0, cv_stream);
      cv::cuda::add(c0_32f, c1_32f, sum_32f, cv::noArray(), -1, cv_stream);
      cv::cuda::add(sum_32f, c2_32f, sum_32f, cv::noArray(), -1, cv_stream);
      sum_32f.convertTo(gray, CV_8UC1, 1.0 / 3.0, 0, cv_stream);
#endif
    }

    // Create binary mask: pixels <= white_threshold -> 255 (dark), else 0
    cv::cuda::GpuMat dark_mask;
    cv::cuda::threshold(gray, dark_mask, static_cast<double>(white_threshold),
                        255.0, cv::THRESH_BINARY_INV, cv_stream);

    // Download and compute integral
    cv::Mat dark_mask_host;
    dark_mask.download(dark_mask_host, cv_stream);
    cv_stream.waitForCompletion();

    cv::Mat integral_img;
    cv::integral(dark_mask_host, integral_img, CV_32S);

    // Calculate number of blocks per row
    const int blocks_per_row = width / block_width;
    if (blocks_per_row == 0) {
      return true;
    }

    const int64_t total_pixels_in_block =
        static_cast<int64_t>(block_width) * static_cast<int64_t>(block_height);

    // Allocate count arrays for 3 rows of blocks (prev, cur, next)
    // Include boundary blocks (index 0 and blocks_per_row+1) set to max
    std::vector<int64_t> prev_counts(blocks_per_row + 2, total_pixels_in_block);
    std::vector<int64_t> cur_counts(blocks_per_row + 2, total_pixels_in_block);
    std::vector<int64_t> next_counts(blocks_per_row + 2, total_pixels_in_block);

    std::vector<cv::Rect> blocks_to_wipe;

    // Initialize cur_counts for first row
    for (int block = 0; block < blocks_per_row; block++) {
      int x0 = block * block_width;
      int x1 = x0 + block_width - 1;
      int64_t dark_sum = integral_rect_sum(integral_img, x0, 0, x1, block_height - 1);
      cur_counts[block + 1] = dark_sum / 255;
    }

    // Process rows
    const int max_top = height - block_height;
    for (int top = 0; top <= max_top; top += block_height) {
      // Compute next row counts
      int next_top = top + step_y;
      if (next_top <= max_top) {
        for (int block = 0; block < blocks_per_row; block++) {
          int x0 = block * block_width;
          int x1 = x0 + block_width - 1;
          int y0 = next_top;
          int y1 = y0 + block_height - 1;
          int64_t dark_sum = integral_rect_sum(integral_img, x0, y0, x1, y1);
          next_counts[block + 1] = dark_sum / 255;
        }
      } else {
        std::fill(next_counts.begin(), next_counts.end(), total_pixels_in_block);
      }

      // Check each block in current row
      for (int block = 0; block < blocks_per_row; block++) {
        int64_t max_neighbor = std::max({
            prev_counts[block], prev_counts[block + 2],
            next_counts[block], next_counts[block + 2],
            cur_counts[block]  // Left neighbor contributes to isolation check
        });

        // Block is isolated if max neighbor ratio is below intensity
        float ratio = static_cast<float>(max_neighbor) /
                      static_cast<float>(total_pixels_in_block);
        if (ratio <= intensity) {
          int x = block * block_width;
          blocks_to_wipe.push_back(cv::Rect(x, top, block_width, block_height));
          cur_counts[block + 1] = total_pixels_in_block; // Mark as wiped
        }
      }

      // Rotate arrays
      std::swap(prev_counts, cur_counts);
      std::swap(cur_counts, next_counts);
    }

    // Wipe collected blocks on GPU
    for (const auto &block : blocks_to_wipe) {
      cv::cuda::GpuMat roi = src(block);
      if (cv_type == CV_8UC1) {
        roi.setTo(cv::Scalar(255), cv_stream);
      } else if (cv_type == CV_8UC2) {
        roi.setTo(cv::Scalar(255, 255), cv_stream);
      } else {
        roi.setTo(cv::Scalar(255, 255, 255), cv_stream);
      }
    }

    cv_stream.waitForCompletion();
    return true;
  } catch (const std::exception &e) {
    fprintf(stderr, "OpenCV blurfilter failed: %s\n", e.what());
    return false;
  }
#else
  (void)src_device;
  (void)width;
  (void)height;
  (void)pitch_bytes;
  (void)format;
  (void)block_width;
  (void)block_height;
  (void)step_x;
  (void)step_y;
  (void)white_threshold;
  (void)intensity;
  (void)stream;
  return false;
#endif
}

bool unpaper_opencv_blackfilter(uint64_t src_device, int width, int height,
                                size_t pitch_bytes, int format, int scan_size_w,
                                int scan_size_h, int scan_depth_h,
                                int scan_depth_v, int scan_step_h,
                                int scan_step_v, bool scan_dir_h,
                                bool scan_dir_v, uint8_t black_threshold,
                                uint8_t area_threshold, uint64_t intensity,
                                const int32_t *exclusions, int exclusion_count,
                                UnpaperCudaStream *stream) {
  // Blackfilter uses flood-fill which is inherently sequential.
  // The OpenCV CCL approach doesn't directly map to the original algorithm.
  // Return false to fall back to custom CUDA implementation.
  (void)src_device;
  (void)width;
  (void)height;
  (void)pitch_bytes;
  (void)format;
  (void)scan_size_w;
  (void)scan_size_h;
  (void)scan_depth_h;
  (void)scan_depth_v;
  (void)scan_step_h;
  (void)scan_step_v;
  (void)scan_dir_h;
  (void)scan_dir_v;
  (void)black_threshold;
  (void)area_threshold;
  (void)intensity;
  (void)exclusions;
  (void)exclusion_count;
  (void)stream;
  return false;
}

bool unpaper_opencv_sum_rect(uint64_t src_device, int width, int height,
                             size_t pitch_bytes, int format, int x0, int y0,
                             int x1, int y1, UnpaperCudaStream *stream,
                             unsigned long long *result_out) {
  if (result_out == nullptr) {
    return false;
  }
  *result_out = 0;

#ifdef HAVE_OPENCV_CUDAARITHM
  auto fmt = static_cast<UnpaperCudaFormat>(format);
  int cv_type = opencv_type_from_format(fmt);
  if (cv_type < 0 || src_device == 0) {
    return false;
  }

  // Validate and clamp rectangle bounds
  if (x0 > x1 || y0 > y1) {
    return true; // Empty rectangle, sum is 0
  }
  if (x0 < 0) x0 = 0;
  if (y0 < 0) y0 = 0;
  if (x1 >= width) x1 = width - 1;
  if (y1 >= height) y1 = height - 1;
  if (x0 > x1 || y0 > y1) {
    return true; // Rectangle fully outside image
  }

  try {
    cv::cuda::Stream cv_stream = get_cv_stream(stream);

    // Wrap source image
    cv::cuda::GpuMat src(height, width, cv_type,
                         reinterpret_cast<void *>(src_device), pitch_bytes);

    // Create ROI
    int rect_w = x1 - x0 + 1;
    int rect_h = y1 - y0 + 1;
    cv::cuda::GpuMat roi = src(cv::Rect(x0, y0, rect_w, rect_h));

    // Convert to single channel for sum if needed
    cv::cuda::GpuMat gray;
    if (cv_type == CV_8UC1) {
      gray = roi;
    } else if (cv_type == CV_8UC2) {
      // Y400A: use Y channel
      std::vector<cv::cuda::GpuMat> channels;
      cv::cuda::split(roi, channels, cv_stream);
      gray = channels[0];
    } else {
      // RGB24: compute grayscale as average of R, G, B
      std::vector<cv::cuda::GpuMat> channels;
      cv::cuda::split(roi, channels, cv_stream);
      cv::cuda::GpuMat sum_rg, sum_rgb;
      cv::cuda::add(channels[0], channels[1], sum_rg, cv::noArray(), CV_16UC1,
                    cv_stream);
      cv::cuda::add(sum_rg, channels[2], sum_rgb, cv::noArray(), CV_16UC1,
                    cv_stream);
      sum_rgb.convertTo(gray, CV_8UC1, 1.0 / 3.0, 0, cv_stream);
    }

    cv_stream.waitForCompletion();
    cv::Scalar s = cv::cuda::sum(gray);
    *result_out = static_cast<unsigned long long>(s[0]);
    return true;
  } catch (const std::exception &e) {
    fprintf(stderr, "OpenCV sum_rect failed: %s\n", e.what());
    return false;
  }
#else
  (void)src_device;
  (void)width;
  (void)height;
  (void)pitch_bytes;
  (void)format;
  (void)x0;
  (void)y0;
  (void)x1;
  (void)y1;
  (void)stream;
  return false;
#endif
}

bool unpaper_opencv_count_brightness_range(uint64_t src_device, int width,
                                           int height, size_t pitch_bytes,
                                           int format, int x0, int y0, int x1,
                                           int y1, uint8_t min_brightness,
                                           uint8_t max_brightness,
                                           UnpaperCudaStream *stream,
                                           unsigned long long *result_out) {
  if (result_out == nullptr) {
    return false;
  }
  *result_out = 0;

#ifdef HAVE_OPENCV_CUDAARITHM
  auto fmt = static_cast<UnpaperCudaFormat>(format);
  int cv_type = opencv_type_from_format(fmt);
  if (cv_type < 0 || src_device == 0) {
    return false;
  }

  // Validate and clamp rectangle bounds
  if (x0 > x1 || y0 > y1) {
    return true; // Empty rectangle, count is 0
  }
  if (x0 < 0) x0 = 0;
  if (y0 < 0) y0 = 0;
  if (x1 >= width) x1 = width - 1;
  if (y1 >= height) y1 = height - 1;
  if (x0 > x1 || y0 > y1) {
    return true; // Rectangle fully outside image
  }

  try {
    cv::cuda::Stream cv_stream = get_cv_stream(stream);

    // Wrap source image
    cv::cuda::GpuMat src(height, width, cv_type,
                         reinterpret_cast<void *>(src_device), pitch_bytes);

    // Create ROI
    int rect_w = x1 - x0 + 1;
    int rect_h = y1 - y0 + 1;
    cv::cuda::GpuMat roi = src(cv::Rect(x0, y0, rect_w, rect_h));

    // Convert to single channel grayscale
    cv::cuda::GpuMat gray;
    if (cv_type == CV_8UC1) {
      gray = roi;
    } else if (cv_type == CV_8UC2) {
      // Y400A: use Y channel
      std::vector<cv::cuda::GpuMat> channels;
      cv::cuda::split(roi, channels, cv_stream);
      gray = channels[0];
    } else {
      // RGB24: compute grayscale as average
      std::vector<cv::cuda::GpuMat> channels;
      cv::cuda::split(roi, channels, cv_stream);
      cv::cuda::GpuMat sum_rg, sum_rgb;
      cv::cuda::add(channels[0], channels[1], sum_rg, cv::noArray(), CV_16UC1,
                    cv_stream);
      cv::cuda::add(sum_rg, channels[2], sum_rgb, cv::noArray(), CV_16UC1,
                    cv_stream);
      sum_rgb.convertTo(gray, CV_8UC1, 1.0 / 3.0, 0, cv_stream);
    }

    // Create mask for pixels in brightness range using threshold
    // First, threshold for >= min_brightness
    cv::cuda::GpuMat above_min, below_max, in_range;
    cv::cuda::threshold(gray, above_min,
                        static_cast<double>(min_brightness) - 0.5, 255.0,
                        cv::THRESH_BINARY, cv_stream);
    // Then, threshold for <= max_brightness
    cv::cuda::threshold(gray, below_max,
                        static_cast<double>(max_brightness) + 0.5, 255.0,
                        cv::THRESH_BINARY_INV, cv_stream);
    // Combine: pixels that are both >= min and <= max
    cv::cuda::bitwise_and(above_min, below_max, in_range, cv::noArray(),
                          cv_stream);

    cv_stream.waitForCompletion();
    int count = cv::cuda::countNonZero(in_range);
    *result_out = static_cast<unsigned long long>(count);
    return true;
  } catch (const std::exception &e) {
    fprintf(stderr, "OpenCV count_brightness_range failed: %s\n", e.what());
    return false;
  }
#else
  (void)src_device;
  (void)width;
  (void)height;
  (void)pitch_bytes;
  (void)format;
  (void)x0;
  (void)y0;
  (void)x1;
  (void)y1;
  (void)min_brightness;
  (void)max_brightness;
  (void)stream;
  return false;
#endif
}

bool unpaper_opencv_detect_edge_rotation_peaks(
    uint64_t src_device, int width, int height, size_t pitch_bytes, int format,
    const int *base_x, const int *base_y, int scan_size, int max_depth,
    int shift_x, int shift_y, int mask_x0, int mask_y0, int mask_x1,
    int mask_y1, int max_blackness_abs, int rotations_count,
    UnpaperCudaStream *stream, int *peaks_out) {
  // IMPORTANT: This function intentionally returns false to use the custom
  // CUDA kernel instead. The custom kernel is more efficient because:
  //
  // 1. It processes all rotation angles in parallel (one CUDA block per angle)
  // 2. It uses shared memory for fast parallel reduction within each block
  // 3. It avoids downloading the entire image to host memory
  // 4. It only downloads the small peaks array (~400 bytes vs ~4MB image)
  //
  // OpenCV CUDA doesn't have efficient primitives for "gather pixels at
  // arbitrary tilted coordinates and reduce" - we would need multiple kernel
  // launches (remap → reduce → gradient) vs the single optimized custom kernel.
  //
  // The custom kernel unpaper_detect_edge_rotation_peaks in cuda_kernels.cu
  // is specifically designed for this operation and outperforms any OpenCV-based
  // implementation.

  (void)src_device;
  (void)width;
  (void)height;
  (void)pitch_bytes;
  (void)format;
  (void)base_x;
  (void)base_y;
  (void)scan_size;
  (void)max_depth;
  (void)shift_x;
  (void)shift_y;
  (void)mask_x0;
  (void)mask_y0;
  (void)mask_x1;
  (void)mask_y1;
  (void)max_blackness_abs;
  (void)rotations_count;
  (void)stream;
  (void)peaks_out;

  // Return false to fall back to the optimized custom CUDA kernel
  return false;
}
