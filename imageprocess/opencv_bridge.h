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

typedef struct {
  int label_count;
  int removed_components;
} UnpaperOpencvCclStats;

typedef struct {
  uint64_t device_ptr;
  int width;
  int height;
  size_t pitch_bytes;
  bool opencv_allocated; // true if allocated via OpenCV/cudart, false via
                         // unpaper
} UnpaperOpencvMask;

bool unpaper_opencv_enabled(void);
bool unpaper_opencv_cuda_supported(void);
bool unpaper_opencv_ccl_supported(void);

bool unpaper_opencv_cuda_ccl(uint64_t mask_device, int width, int height,
                             size_t pitch_bytes, uint8_t foreground_value,
                             uint32_t max_component_size,
                             UnpaperCudaStream *stream,
                             UnpaperOpencvCclStats *stats_out);

bool unpaper_opencv_extract_dark_mask(uint64_t src_device, int src_width,
                                      int src_height, size_t src_pitch_bytes,
                                      int src_format, uint8_t min_white_level,
                                      UnpaperCudaStream *stream,
                                      UnpaperOpencvMask *mask_out);

void unpaper_opencv_mask_free(UnpaperOpencvMask *mask);

/**
 * Grayfilter using OpenCV CUDA with integral images.
 * Scans tiles and wipes those that are uniformly gray (no black pixels
 * but average lightness below threshold).
 *
 * @param src_device     Device pointer to image
 * @param width          Image width in pixels
 * @param height         Image height in pixels
 * @param pitch_bytes    Row stride in bytes
 * @param format         Pixel format (UnpaperCudaFormat)
 * @param tile_width     Width of scan tile
 * @param tile_height    Height of scan tile
 * @param step_x         Horizontal step between tiles
 * @param step_y         Vertical step between tiles
 * @param black_threshold Pixels with grayscale <= this are considered black
 * @param gray_threshold  Tiles with inverse lightness < this are wiped
 * @param stream         Optional CUDA stream (may be NULL)
 * @param sync_after     If true, synchronize stream before returning.
 *                       Set to false for batch processing to defer sync.
 * @return true on success, false if operation not supported
 */
bool unpaper_opencv_grayfilter(uint64_t src_device, int width, int height,
                               size_t pitch_bytes, int format, int tile_width,
                               int tile_height, int step_x, int step_y,
                               uint8_t black_threshold, uint8_t gray_threshold,
                               UnpaperCudaStream *stream, bool sync_after);

/**
 * Blurfilter using OpenCV CUDA with integral images.
 * Scans blocks and wipes isolated dark regions.
 *
 * @param src_device      Device pointer to image
 * @param width           Image width in pixels
 * @param height          Image height in pixels
 * @param pitch_bytes     Row stride in bytes
 * @param format          Pixel format (UnpaperCudaFormat)
 * @param block_width     Width of scan block
 * @param block_height    Height of scan block
 * @param step_x          Horizontal step between blocks
 * @param step_y          Vertical step between blocks
 * @param white_threshold Pixels with grayscale <= this are counted as dark
 * @param intensity       Max ratio of dark pixels to keep a block
 * @param stream          Optional CUDA stream (may be NULL)
 * @param sync_after      If true, synchronize stream before returning.
 *                        Set to false for batch processing to defer sync.
 * @return true on success, false if operation not supported
 */
bool unpaper_opencv_blurfilter(uint64_t src_device, int width, int height,
                               size_t pitch_bytes, int format, int block_width,
                               int block_height, int step_x, int step_y,
                               uint8_t white_threshold, float intensity,
                               UnpaperCudaStream *stream, bool sync_after);

/**
 * Blackfilter using OpenCV CUDA with CCL.
 * Detects and removes large connected black regions.
 *
 * @param src_device       Device pointer to image
 * @param width            Image width in pixels
 * @param height           Image height in pixels
 * @param pitch_bytes      Row stride in bytes
 * @param format           Pixel format (UnpaperCudaFormat)
 * @param scan_size_w      Width of scan region
 * @param scan_size_h      Height of scan region
 * @param scan_depth_h     Horizontal scan depth
 * @param scan_depth_v     Vertical scan depth
 * @param scan_step_h      Horizontal scan step
 * @param scan_step_v      Vertical scan step
 * @param scan_dir_h       Scan horizontally
 * @param scan_dir_v       Scan vertically
 * @param black_threshold  Pixels with grayscale <= this are considered black
 * @param area_threshold   Darkness threshold to trigger removal (0-255)
 * @param intensity        Intensity for flood-fill tolerance
 * @param exclusions       Array of exclusion rectangles (x0,y0,x1,y1 per rect)
 * @param exclusion_count  Number of exclusion rectangles
 * @param stream           Optional CUDA stream (may be NULL)
 * @return true on success, false if operation not supported
 */
bool unpaper_opencv_blackfilter(
    uint64_t src_device, int width, int height, size_t pitch_bytes, int format,
    int scan_size_w, int scan_size_h, int scan_depth_h, int scan_depth_v,
    int scan_step_h, int scan_step_v, bool scan_dir_h, bool scan_dir_v,
    uint8_t black_threshold, uint8_t area_threshold, uint64_t intensity,
    const int32_t *exclusions, int exclusion_count, UnpaperCudaStream *stream);

/**
 * Sum grayscale values in a rectangle using OpenCV CUDA.
 *
 * @param src_device     Device pointer to image
 * @param width          Image width in pixels
 * @param height         Image height in pixels
 * @param pitch_bytes    Row stride in bytes
 * @param format         Pixel format (UnpaperCudaFormat)
 * @param x0, y0         Top-left corner of rectangle (inclusive)
 * @param x1, y1         Bottom-right corner of rectangle (inclusive)
 * @param stream         Optional CUDA stream (may be NULL)
 * @param result_out     Output: sum of grayscale values
 * @return true on success, false if operation not supported
 */
bool unpaper_opencv_sum_rect(uint64_t src_device, int width, int height,
                             size_t pitch_bytes, int format, int x0, int y0,
                             int x1, int y1, UnpaperCudaStream *stream,
                             unsigned long long *result_out);

/**
 * Count pixels in a brightness range using OpenCV CUDA.
 *
 * @param src_device     Device pointer to image
 * @param width          Image width in pixels
 * @param height         Image height in pixels
 * @param pitch_bytes    Row stride in bytes
 * @param format         Pixel format (UnpaperCudaFormat)
 * @param x0, y0         Top-left corner of rectangle (inclusive)
 * @param x1, y1         Bottom-right corner of rectangle (inclusive)
 * @param min_brightness Minimum brightness (inclusive)
 * @param max_brightness Maximum brightness (inclusive)
 * @param stream         Optional CUDA stream (may be NULL)
 * @param result_out     Output: count of pixels in range
 * @return true on success, false if operation not supported
 */
bool unpaper_opencv_count_brightness_range(uint64_t src_device, int width,
                                           int height, size_t pitch_bytes,
                                           int format, int x0, int y0, int x1,
                                           int y1, uint8_t min_brightness,
                                           uint8_t max_brightness,
                                           UnpaperCudaStream *stream,
                                           unsigned long long *result_out);

/**
 * Detect edge rotation peaks using OpenCV CUDA.
 * Computes blackness sums along scan lines at different angles.
 *
 * @param src_device     Device pointer to image
 * @param width          Image width in pixels
 * @param height         Image height in pixels
 * @param pitch_bytes    Row stride in bytes
 * @param format         Pixel format (UnpaperCudaFormat)
 * @param base_x         Array of X coordinates for base scan line points
 *                       [rotations_count * scan_size]
 * @param base_y         Array of Y coordinates for base scan line points
 * @param scan_size      Number of points in each scan line
 * @param max_depth      Maximum depth to scan perpendicular to line
 * @param shift_x        X direction for perpendicular scan
 * @param shift_y        Y direction for perpendicular scan
 * @param mask_x0        Mask region left boundary
 * @param mask_y0        Mask region top boundary
 * @param mask_x1        Mask region right boundary
 * @param mask_y1        Mask region bottom boundary
 * @param max_blackness_abs Maximum accumulated blackness threshold
 * @param rotations_count Number of rotation angles to test
 * @param stream         Optional CUDA stream (may be NULL)
 * @param peaks_out      Output: array of peak values [rotations_count]
 * @return true on success, false if operation not supported
 */
bool unpaper_opencv_detect_edge_rotation_peaks(
    uint64_t src_device, int width, int height, size_t pitch_bytes, int format,
    const int *base_x, const int *base_y, int scan_size, int max_depth,
    int shift_x, int shift_y, int mask_x0, int mask_y0, int mask_x1,
    int mask_y1, int max_blackness_abs, int rotations_count,
    UnpaperCudaStream *stream, int *peaks_out);

#ifdef __cplusplus
}
#endif
