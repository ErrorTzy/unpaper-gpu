// SPDX-FileCopyrightText: 2025 The unpaper authors
//
// SPDX-License-Identifier: GPL-2.0-only

#pragma once

#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>

#include "imageprocess/cuda_runtime.h"

#ifdef __cplusplus
extern "C" {
#endif

/**
 * Wipe a rectangle region to a solid color using OpenCV CUDA.
 *
 * @param dst_device  Device pointer to destination image
 * @param dst_width   Image width in pixels
 * @param dst_height  Image height in pixels
 * @param dst_pitch   Row stride in bytes
 * @param dst_format  Pixel format (UnpaperCudaFormat)
 * @param x0, y0      Top-left corner of rectangle (inclusive)
 * @param x1, y1      Bottom-right corner of rectangle (inclusive)
 * @param r, g, b     Fill color components
 * @param stream      Optional CUDA stream (may be NULL)
 * @return true on success, false if operation not supported (e.g., mono format)
 */
bool unpaper_opencv_wipe_rect(uint64_t dst_device, int dst_width, int dst_height,
                              size_t dst_pitch, int dst_format, int x0, int y0,
                              int x1, int y1, uint8_t r, uint8_t g, uint8_t b,
                              UnpaperCudaStream *stream);

/**
 * Copy a rectangle region from source to destination using OpenCV CUDA.
 *
 * @param src_device  Device pointer to source image
 * @param src_width   Source image width in pixels
 * @param src_height  Source image height in pixels
 * @param src_pitch   Source row stride in bytes
 * @param src_format  Source pixel format (UnpaperCudaFormat)
 * @param dst_device  Device pointer to destination image
 * @param dst_width   Destination image width in pixels
 * @param dst_height  Destination image height in pixels
 * @param dst_pitch   Destination row stride in bytes
 * @param dst_format  Destination pixel format (UnpaperCudaFormat)
 * @param src_x0, src_y0  Top-left corner of source rectangle
 * @param dst_x0, dst_y0  Top-left corner in destination
 * @param copy_w, copy_h  Size of region to copy
 * @param stream      Optional CUDA stream (may be NULL)
 * @return true on success, false if operation not supported
 */
bool unpaper_opencv_copy_rect(uint64_t src_device, int src_width, int src_height,
                              size_t src_pitch, int src_format,
                              uint64_t dst_device, int dst_width, int dst_height,
                              size_t dst_pitch, int dst_format, int src_x0,
                              int src_y0, int dst_x0, int dst_y0, int copy_w,
                              int copy_h, UnpaperCudaStream *stream);

/**
 * Mirror an image horizontally and/or vertically using OpenCV CUDA.
 *
 * @param src_device  Device pointer to source image
 * @param dst_device  Device pointer to destination image (may be same as src)
 * @param width       Image width in pixels
 * @param height      Image height in pixels
 * @param pitch       Row stride in bytes
 * @param format      Pixel format (UnpaperCudaFormat)
 * @param horizontal  If true, mirror horizontally
 * @param vertical    If true, mirror vertically
 * @param stream      Optional CUDA stream (may be NULL)
 * @return true on success, false if operation not supported (e.g., mono format)
 */
bool unpaper_opencv_mirror(uint64_t src_device, uint64_t dst_device, int width,
                           int height, size_t pitch, int format, bool horizontal,
                           bool vertical, UnpaperCudaStream *stream);

/**
 * Rotate an image by 90 degrees clockwise or counter-clockwise using OpenCV CUDA.
 *
 * @param src_device   Device pointer to source image
 * @param src_width    Source image width in pixels
 * @param src_height   Source image height in pixels
 * @param src_pitch    Source row stride in bytes
 * @param dst_device   Device pointer to destination image
 * @param dst_pitch    Destination row stride in bytes
 * @param format       Pixel format (UnpaperCudaFormat)
 * @param clockwise    If true, rotate clockwise; otherwise counter-clockwise
 * @param stream       Optional CUDA stream (may be NULL)
 * @return true on success, false if operation not supported (e.g., mono format)
 */
bool unpaper_opencv_rotate90(uint64_t src_device, int src_width, int src_height,
                             size_t src_pitch, uint64_t dst_device,
                             size_t dst_pitch, int format, bool clockwise,
                             UnpaperCudaStream *stream);

/**
 * Resize an image using OpenCV CUDA warpAffine.
 *
 * Uses the same coordinate mapping as unpaper CPU: src = dst * scale
 * This matches unpaper's convention exactly for pixel-perfect parity.
 *
 * @param src_device   Device pointer to source image
 * @param src_width    Source image width in pixels
 * @param src_height   Source image height in pixels
 * @param src_pitch    Source row stride in bytes
 * @param dst_device   Device pointer to destination image
 * @param dst_width    Destination image width in pixels
 * @param dst_height   Destination image height in pixels
 * @param dst_pitch    Destination row stride in bytes
 * @param format       Pixel format (UnpaperCudaFormat)
 * @param interp_type  Interpolation type (0=NN, 1=linear, 2=cubic)
 * @param stream       Optional CUDA stream (may be NULL)
 * @return true on success, false if operation not supported (e.g., mono/Y400A)
 */
bool unpaper_opencv_resize(uint64_t src_device, int src_width, int src_height,
                           size_t src_pitch, uint64_t dst_device, int dst_width,
                           int dst_height, size_t dst_pitch, int format,
                           int interp_type, UnpaperCudaStream *stream);

/**
 * Deskew (rotate) an image using OpenCV CUDA warpAffine.
 *
 * Uses the same coordinate mapping as unpaper CPU deskew:
 *   sx = src_center_x + (x - dst_center_x) * cos + (y - dst_center_y) * sin
 *   sy = src_center_y + (y - dst_center_y) * cos - (x - dst_center_x) * sin
 *
 * @param src_device     Device pointer to source image
 * @param src_width      Source image width in pixels
 * @param src_height     Source image height in pixels
 * @param src_pitch      Source row stride in bytes
 * @param dst_device     Device pointer to destination image
 * @param dst_width      Destination image width in pixels
 * @param dst_height     Destination image height in pixels
 * @param dst_pitch      Destination row stride in bytes
 * @param format         Pixel format (UnpaperCudaFormat)
 * @param src_center_x   Source image rotation center X
 * @param src_center_y   Source image rotation center Y
 * @param dst_center_x   Destination image rotation center X
 * @param dst_center_y   Destination image rotation center Y
 * @param cosval         Cosine of rotation angle
 * @param sinval         Sine of rotation angle
 * @param interp_type    Interpolation type (0=NN, 1=linear, 2=cubic)
 * @param stream         Optional CUDA stream (may be NULL)
 * @return true on success, false if operation not supported (e.g., mono/Y400A)
 */
bool unpaper_opencv_deskew(uint64_t src_device, int src_width, int src_height,
                           size_t src_pitch, uint64_t dst_device, int dst_width,
                           int dst_height, size_t dst_pitch, int format,
                           float src_center_x, float src_center_y,
                           float dst_center_x, float dst_center_y, float cosval,
                           float sinval, int interp_type,
                           UnpaperCudaStream *stream);

#ifdef __cplusplus
}
#endif
