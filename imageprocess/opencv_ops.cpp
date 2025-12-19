// SPDX-FileCopyrightText: 2025 The unpaper authors
//
// SPDX-License-Identifier: GPL-2.0-only

#include "imageprocess/opencv_ops.h"
#include "imageprocess/cuda_kernels_format.h"
#include "lib/logging.h"

#include <cstring>
#include <exception>

#include <cuda_runtime.h>
#include <opencv2/core/cuda.hpp>
#include <opencv2/core/cuda_stream_accessor.hpp>

#if __has_include(<opencv2/cudaarithm.hpp>)
#include <opencv2/cudaarithm.hpp>
#endif

#if __has_include(<opencv2/cudawarping.hpp>)
#include <opencv2/cudawarping.hpp>
#endif

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

static inline bool is_mono_format(UnpaperCudaFormat fmt) {
  return fmt == UNPAPER_CUDA_FMT_MONOWHITE || fmt == UNPAPER_CUDA_FMT_MONOBLACK;
}

static inline cv::cuda::Stream get_cv_stream(UnpaperCudaStream *stream) {
  cudaStream_t cuda_stream = nullptr;
  if (stream != nullptr) {
    cuda_stream =
        static_cast<cudaStream_t>(unpaper_cuda_stream_get_raw_handle(stream));
  }
  return cv::cuda::StreamAccessor::wrapStream(cuda_stream);
}

bool unpaper_opencv_wipe_rect(uint64_t dst_device, int dst_width,
                              int dst_height, size_t dst_pitch, int dst_format,
                              int x0, int y0, int x1, int y1, uint8_t r,
                              uint8_t g, uint8_t b, UnpaperCudaStream *stream) {
#ifdef HAVE_OPENCV_CUDAARITHM
  auto fmt = static_cast<UnpaperCudaFormat>(dst_format);
  if (is_mono_format(fmt)) {
    return false; // Mono formats handled by custom kernel
  }

  int cv_type = opencv_type_from_format(fmt);
  if (cv_type < 0 || dst_device == 0) {
    return false;
  }

  // Validate and clamp rectangle bounds
  if (x0 > x1 || y0 > y1) {
    return true; // Empty rectangle, nothing to do
  }
  if (x0 < 0)
    x0 = 0;
  if (y0 < 0)
    y0 = 0;
  if (x1 >= dst_width)
    x1 = dst_width - 1;
  if (y1 >= dst_height)
    y1 = dst_height - 1;
  if (x0 > x1 || y0 > y1) {
    return true; // Rectangle fully outside image
  }

  try {
    cv::cuda::Stream cv_stream = get_cv_stream(stream);

    // Wrap the full image as GpuMat
    cv::cuda::GpuMat dst(dst_height, dst_width, cv_type,
                         reinterpret_cast<void *>(dst_device), dst_pitch);

    // Create ROI for the rectangle
    int rect_w = x1 - x0 + 1;
    int rect_h = y1 - y0 + 1;
    cv::cuda::GpuMat roi = dst(cv::Rect(x0, y0, rect_w, rect_h));

    // Set the ROI to the fill color
    cv::Scalar color;
    switch (fmt) {
    case UNPAPER_CUDA_FMT_GRAY8:
      color = cv::Scalar((r + g + b) / 3);
      break;
    case UNPAPER_CUDA_FMT_Y400A:
      color = cv::Scalar((r + g + b) / 3, 255);
      break;
    case UNPAPER_CUDA_FMT_RGB24:
      color = cv::Scalar(r, g, b);
      break;
    default:
      return false;
    }

    roi.setTo(color, cv_stream);
    return true;
  } catch (const std::exception &e) {
    fprintf(stderr, "OpenCV wipe_rect failed: %s\n", e.what());
    return false;
  }
#else
  (void)dst_device;
  (void)dst_width;
  (void)dst_height;
  (void)dst_pitch;
  (void)dst_format;
  (void)x0;
  (void)y0;
  (void)x1;
  (void)y1;
  (void)r;
  (void)g;
  (void)b;
  (void)stream;
  return false;
#endif
}

bool unpaper_opencv_copy_rect(uint64_t src_device, int src_width,
                              int src_height, size_t src_pitch, int src_format,
                              uint64_t dst_device, int dst_width,
                              int dst_height, size_t dst_pitch, int dst_format,
                              int src_x0, int src_y0, int dst_x0, int dst_y0,
                              int copy_w, int copy_h,
                              UnpaperCudaStream *stream) {
#ifdef HAVE_OPENCV_CUDAARITHM
  auto sfmt = static_cast<UnpaperCudaFormat>(src_format);
  auto dfmt = static_cast<UnpaperCudaFormat>(dst_format);

  // Mono formats require custom kernel for bit manipulation
  if (is_mono_format(sfmt) || is_mono_format(dfmt)) {
    return false;
  }

  // Format conversion between different byte formats requires custom kernel
  if (sfmt != dfmt) {
    return false;
  }

  int cv_type = opencv_type_from_format(sfmt);
  if (cv_type < 0 || src_device == 0 || dst_device == 0) {
    return false;
  }

  if (copy_w <= 0 || copy_h <= 0) {
    return true; // Nothing to copy
  }

  // Validate bounds
  if (src_x0 < 0 || src_y0 < 0 || src_x0 + copy_w > src_width ||
      src_y0 + copy_h > src_height) {
    return false; // Source out of bounds
  }
  if (dst_x0 < 0 || dst_y0 < 0 || dst_x0 + copy_w > dst_width ||
      dst_y0 + copy_h > dst_height) {
    return false; // Destination out of bounds
  }

  try {
    cv::cuda::Stream cv_stream = get_cv_stream(stream);

    // Wrap images as GpuMat
    cv::cuda::GpuMat src(src_height, src_width, cv_type,
                         reinterpret_cast<void *>(src_device), src_pitch);
    cv::cuda::GpuMat dst(dst_height, dst_width, cv_type,
                         reinterpret_cast<void *>(dst_device), dst_pitch);

    // Create ROIs
    cv::cuda::GpuMat src_roi = src(cv::Rect(src_x0, src_y0, copy_w, copy_h));
    cv::cuda::GpuMat dst_roi = dst(cv::Rect(dst_x0, dst_y0, copy_w, copy_h));

    // Copy using OpenCV
    src_roi.copyTo(dst_roi, cv_stream);
    return true;
  } catch (const std::exception &e) {
    fprintf(stderr, "OpenCV copy_rect failed: %s\n", e.what());
    return false;
  }
#else
  (void)src_device;
  (void)src_width;
  (void)src_height;
  (void)src_pitch;
  (void)src_format;
  (void)dst_device;
  (void)dst_width;
  (void)dst_height;
  (void)dst_pitch;
  (void)dst_format;
  (void)src_x0;
  (void)src_y0;
  (void)dst_x0;
  (void)dst_y0;
  (void)copy_w;
  (void)copy_h;
  (void)stream;
  return false;
#endif
}

bool unpaper_opencv_mirror(uint64_t src_device, uint64_t dst_device, int width,
                           int height, size_t pitch, int format,
                           bool horizontal, bool vertical,
                           UnpaperCudaStream *stream) {
#ifdef HAVE_OPENCV_CUDAARITHM
  auto fmt = static_cast<UnpaperCudaFormat>(format);
  if (is_mono_format(fmt)) {
    return false; // Mono formats handled by custom kernel
  }

  int cv_type = opencv_type_from_format(fmt);
  if (cv_type < 0 || src_device == 0 || dst_device == 0) {
    return false;
  }

  if (!horizontal && !vertical) {
    // No flip needed, just copy if src != dst
    if (src_device != dst_device) {
      // Use async copy with stream to avoid serializing all streams
      cudaStream_t cuda_stream = nullptr;
      if (stream != nullptr) {
        cuda_stream = static_cast<cudaStream_t>(
            unpaper_cuda_stream_get_raw_handle(stream));
      }
      cudaMemcpyAsync(reinterpret_cast<void *>(dst_device),
                      reinterpret_cast<void *>(src_device), pitch * height,
                      cudaMemcpyDeviceToDevice, cuda_stream);
      if (cuda_stream != nullptr) {
        cudaStreamSynchronize(cuda_stream);
      }
    }
    return true;
  }

  try {
    cv::cuda::Stream cv_stream = get_cv_stream(stream);

    cv::cuda::GpuMat src(height, width, cv_type,
                         reinterpret_cast<void *>(src_device), pitch);
    cv::cuda::GpuMat dst(height, width, cv_type,
                         reinterpret_cast<void *>(dst_device), pitch);

    // OpenCV flip codes:
    // 0 = vertical flip (around x-axis)
    // 1 = horizontal flip (around y-axis)
    // -1 = both horizontal and vertical flip
    int flip_code;
    if (horizontal && vertical) {
      flip_code = -1;
    } else if (horizontal) {
      flip_code = 1;
    } else {
      flip_code = 0;
    }

    cv::cuda::flip(src, dst, flip_code, cv_stream);
    return true;
  } catch (const std::exception &e) {
    fprintf(stderr, "OpenCV mirror failed: %s\n", e.what());
    return false;
  }
#else
  (void)src_device;
  (void)dst_device;
  (void)width;
  (void)height;
  (void)pitch;
  (void)format;
  (void)horizontal;
  (void)vertical;
  (void)stream;
  return false;
#endif
}

bool unpaper_opencv_rotate90(uint64_t src_device, int src_width, int src_height,
                             size_t src_pitch, uint64_t dst_device,
                             size_t dst_pitch, int format, bool clockwise,
                             UnpaperCudaStream *stream) {
#ifdef HAVE_OPENCV_CUDAARITHM
  auto fmt = static_cast<UnpaperCudaFormat>(format);
  if (is_mono_format(fmt)) {
    return false; // Mono formats handled by custom kernel
  }

  int cv_type = opencv_type_from_format(fmt);
  if (cv_type < 0 || src_device == 0 || dst_device == 0) {
    return false;
  }

  // Destination dimensions are swapped for 90-degree rotation
  int dst_width = src_height;
  int dst_height = src_width;

  try {
    cv::cuda::Stream cv_stream = get_cv_stream(stream);

    cv::cuda::GpuMat src(src_height, src_width, cv_type,
                         reinterpret_cast<void *>(src_device), src_pitch);
    cv::cuda::GpuMat dst(dst_height, dst_width, cv_type,
                         reinterpret_cast<void *>(dst_device), dst_pitch);

    // Strategy depends on format due to OpenCV limitations:
    // - transpose() only supports elemSize 1, 4, or 8 bytes
    // - rotate() supports 1, 3, or 4 channels
    // - GRAY8 (1 byte, 1 channel): use transpose + flip (fastest)
    // - RGB24 (3 bytes, 3 channels): use rotate() with INTER_NEAREST
    // - Y400A (2 bytes, 2 channels): not supported by OpenCV, use custom kernel

    if (fmt == UNPAPER_CUDA_FMT_GRAY8) {
      // GRAY8: Use transpose + flip (elemSize=1 is supported)
      cv::cuda::GpuMat transposed;
      cv::cuda::transpose(src, transposed, cv_stream);
      int flip_code = clockwise ? 1 : 0;
      cv::cuda::flip(transposed, dst, flip_code, cv_stream);
      return true;
    }

#ifdef HAVE_OPENCV_CUDAWARPING
    if (fmt == UNPAPER_CUDA_FMT_RGB24) {
      // RGB24: Use cv::cuda::warpAffine() which supports 3 channels
      // warpAffine uses inverse mapping: for each dst pixel, compute src coords
      //
      // IMPORTANT: By default, warpAffine expects a FORWARD transformation
      // matrix (source to destination) and internally inverts it. Since we
      // provide an INVERSE matrix (destination to source), we must set
      // WARP_INVERSE_MAP.
      //
      // For 90° clockwise rotation of WxH image to HxW:
      //   src_col = dst_row, src_row = src_height - 1 - dst_col
      // For 90° counter-clockwise rotation:
      //   src_col = src_width - 1 - dst_row, src_row = dst_col
      cv::Mat M(2, 3, CV_64F);
      if (clockwise) {
        M.at<double>(0, 0) = 0;
        M.at<double>(0, 1) = 1;
        M.at<double>(0, 2) = 0;
        M.at<double>(1, 0) = -1;
        M.at<double>(1, 1) = 0;
        M.at<double>(1, 2) = src_height - 1;
      } else {
        M.at<double>(0, 0) = 0;
        M.at<double>(0, 1) = -1;
        M.at<double>(0, 2) = src_width - 1;
        M.at<double>(1, 0) = 1;
        M.at<double>(1, 1) = 0;
        M.at<double>(1, 2) = 0;
      }

      // Use WARP_INVERSE_MAP because our matrix is already in inverse form
      cv::cuda::warpAffine(src, dst, M, cv::Size(dst_width, dst_height),
                           cv::INTER_NEAREST | cv::WARP_INVERSE_MAP,
                           cv::BORDER_CONSTANT, cv::Scalar(0, 0, 0), cv_stream);
      return true;
    }
#endif

    // Y400A (2 channels): Not supported by OpenCV rotate/warpAffine
    // Fall back to custom kernel
    return false;

  } catch (const std::exception &e) {
    fprintf(stderr, "OpenCV rotate90 failed: %s\n", e.what());
    return false;
  }
#else
  (void)src_device;
  (void)src_width;
  (void)src_height;
  (void)src_pitch;
  (void)dst_device;
  (void)dst_pitch;
  (void)format;
  (void)clockwise;
  (void)stream;
  return false;
#endif
}

// Convert unpaper interpolation type to OpenCV interpolation flag
static inline int opencv_interp_from_unpaper(int interp_type) {
  switch (interp_type) {
  case 0:
    return cv::INTER_NEAREST;
  case 1:
    return cv::INTER_LINEAR;
  case 2:
  default:
    return cv::INTER_CUBIC;
  }
}

bool unpaper_opencv_resize(uint64_t src_device, int src_width, int src_height,
                           size_t src_pitch, uint64_t dst_device, int dst_width,
                           int dst_height, size_t dst_pitch, int format,
                           int interp_type, UnpaperCudaStream *stream) {
#ifdef HAVE_OPENCV_CUDAWARPING
  auto fmt = static_cast<UnpaperCudaFormat>(format);
  if (is_mono_format(fmt)) {
    return false; // Mono formats handled by custom kernel
  }

  int cv_type = opencv_type_from_format(fmt);
  if (cv_type < 0 || src_device == 0 || dst_device == 0) {
    return false;
  }

  // Y400A (2 channels) is not well supported by OpenCV resize
  if (fmt == UNPAPER_CUDA_FMT_Y400A) {
    return false;
  }

  // Get bytes per pixel for this format
  int elem_size = 1;
  if (fmt == UNPAPER_CUDA_FMT_RGB24) {
    elem_size = 3;
  }

  try {
    cv::cuda::Stream cv_stream = get_cv_stream(stream);

    cv::cuda::GpuMat src(src_height, src_width, cv_type,
                         reinterpret_cast<void *>(src_device), src_pitch);

    int cv_interp = opencv_interp_from_unpaper(interp_type);

    // Resize to temporary buffer first
    cv::cuda::GpuMat resized;
    cv::cuda::resize(src, resized, cv::Size(dst_width, dst_height), 0, 0,
                     cv_interp, cv_stream);

    // Copy to destination with correct pitch using cudaMemcpy2DAsync
    // Use the same stream to avoid serializing all streams (default stream
    // sync) The async copy will wait for resize to complete since it's on the
    // same stream
    cudaStream_t cuda_stream = cv::cuda::StreamAccessor::getStream(cv_stream);
    cudaError_t err = cudaMemcpy2DAsync(
        reinterpret_cast<void *>(dst_device), dst_pitch, resized.data,
        resized.step, (size_t)dst_width * elem_size, (size_t)dst_height,
        cudaMemcpyDeviceToDevice, cuda_stream);
    if (err != cudaSuccess) {
      fprintf(stderr, "OpenCV resize cudaMemcpy2DAsync failed: %s\n",
              cudaGetErrorString(err));
      return false;
    }
    // Sync this stream to ensure copy completes before resized buffer is freed
    cv_stream.waitForCompletion();

    return true;
  } catch (const std::exception &e) {
    verboseLog(VERBOSE_DEBUG, "OpenCV resize failed: %s\n", e.what());
    return false;
  }
#else
  (void)src_device;
  (void)src_width;
  (void)src_height;
  (void)src_pitch;
  (void)dst_device;
  (void)dst_width;
  (void)dst_height;
  (void)dst_pitch;
  (void)format;
  (void)interp_type;
  (void)stream;
  return false;
#endif
}

bool unpaper_opencv_deskew(uint64_t src_device, int src_width, int src_height,
                           size_t src_pitch, uint64_t dst_device, int dst_width,
                           int dst_height, size_t dst_pitch, int format,
                           float src_center_x, float src_center_y,
                           float dst_center_x, float dst_center_y, float cosval,
                           float sinval, int interp_type,
                           UnpaperCudaStream *stream) {
#ifdef HAVE_OPENCV_CUDAWARPING
  auto fmt = static_cast<UnpaperCudaFormat>(format);
  if (is_mono_format(fmt)) {
    return false; // Mono formats handled by custom kernel
  }

  int cv_type = opencv_type_from_format(fmt);
  if (cv_type < 0 || src_device == 0 || dst_device == 0) {
    return false;
  }

  // Y400A (2 channels) is not supported by warpAffine
  if (fmt == UNPAPER_CUDA_FMT_Y400A) {
    return false;
  }

  // Get bytes per pixel for this format
  int elem_size = 1;
  if (fmt == UNPAPER_CUDA_FMT_RGB24) {
    elem_size = 3;
  }

  try {
    cv::cuda::Stream cv_stream = get_cv_stream(stream);

    cv::cuda::GpuMat src(src_height, src_width, cv_type,
                         reinterpret_cast<void *>(src_device), src_pitch);

    // Unpaper deskew coordinate mapping (inverse, dst -> src):
    //   sx = src_center_x + (x - dst_center_x) * cosval + (y - dst_center_y) *
    //   sinval sy = src_center_y + (y - dst_center_y) * cosval - (x -
    //   dst_center_x) * sinval
    //
    // Expanding:
    //   sx = cosval * x + sinval * y + (src_center_x - dst_center_x * cosval -
    //   dst_center_y * sinval) sy = -sinval * x + cosval * y + (src_center_y +
    //   dst_center_x * sinval - dst_center_y * cosval)
    //
    // Affine matrix for inverse mapping:
    // | cosval,   sinval,   tx |
    // | -sinval,  cosval,   ty |
    double tx = (double)src_center_x - (double)dst_center_x * (double)cosval -
                (double)dst_center_y * (double)sinval;
    double ty = (double)src_center_y + (double)dst_center_x * (double)sinval -
                (double)dst_center_y * (double)cosval;

    cv::Mat M(2, 3, CV_64F);
    M.at<double>(0, 0) = (double)cosval;
    M.at<double>(0, 1) = (double)sinval;
    M.at<double>(0, 2) = tx;
    M.at<double>(1, 0) = -(double)sinval;
    M.at<double>(1, 1) = (double)cosval;
    M.at<double>(1, 2) = ty;

    int cv_interp = opencv_interp_from_unpaper(interp_type);
    cv::Scalar border_color = (fmt == UNPAPER_CUDA_FMT_GRAY8)
                                  ? cv::Scalar(255)
                                  : cv::Scalar(255, 255, 255);

    // Warp to temporary buffer first
    cv::cuda::GpuMat warped;
    cv::cuda::warpAffine(src, warped, M, cv::Size(dst_width, dst_height),
                         cv_interp | cv::WARP_INVERSE_MAP, cv::BORDER_CONSTANT,
                         border_color, cv_stream);

    // Copy to destination with correct pitch using cudaMemcpy2DAsync
    // Use the same stream to avoid serializing all streams (default stream
    // sync) The async copy will wait for warp to complete since it's on the
    // same stream
    cudaStream_t cuda_stream = cv::cuda::StreamAccessor::getStream(cv_stream);
    cudaError_t err = cudaMemcpy2DAsync(
        reinterpret_cast<void *>(dst_device), dst_pitch, warped.data,
        warped.step, (size_t)dst_width * elem_size, (size_t)dst_height,
        cudaMemcpyDeviceToDevice, cuda_stream);
    if (err != cudaSuccess) {
      fprintf(stderr, "OpenCV deskew cudaMemcpy2DAsync failed: %s\n",
              cudaGetErrorString(err));
      return false;
    }
    // Sync this stream to ensure copy completes before warped buffer is freed
    cv_stream.waitForCompletion();

    return true;
  } catch (const std::exception &e) {
    fprintf(stderr, "OpenCV deskew failed: %s\n", e.what());
    return false;
  }
#else
  (void)src_device;
  (void)src_width;
  (void)src_height;
  (void)src_pitch;
  (void)dst_device;
  (void)dst_width;
  (void)dst_height;
  (void)dst_pitch;
  (void)format;
  (void)src_center_x;
  (void)src_center_y;
  (void)dst_center_x;
  (void)dst_center_y;
  (void)cosval;
  (void)sinval;
  (void)interp_type;
  (void)stream;
  return false;
#endif
}
