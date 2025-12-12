// SPDX-FileCopyrightText: 2025 The unpaper authors
//
// SPDX-License-Identifier: GPL-2.0-only

#include "imageprocess/backend.h"

#include <inttypes.h>

#include "lib/logging.h"

void wipe_rectangle_cpu(Image image, Rectangle input_area, Pixel color);
void copy_rectangle_cpu(Image source, Image target, Rectangle source_area,
                        Point target_coords);
void center_image_cpu(Image source, Image target, Point target_origin,
                      RectangleSize target_size);
void stretch_and_replace_cpu(Image *pImage, RectangleSize size,
                             Interpolation interpolate_type);
void resize_and_replace_cpu(Image *pImage, RectangleSize size,
                            Interpolation interpolate_type);
void flip_rotate_90_cpu(Image *pImage, RotationDirection direction);
void mirror_cpu(Image image, Direction direction);
void shift_image_cpu(Image *pImage, Delta d);

void apply_masks_cpu(Image image, const Rectangle masks[], size_t masks_count,
                     Pixel color);
void apply_wipes_cpu(Image image, Wipes wipes, Pixel color);
void apply_border_cpu(Image image, const Border border, Pixel color);
size_t detect_masks_cpu(Image image, MaskDetectionParameters params,
                        const Point points[], size_t points_count,
                        Rectangle masks[]);
void align_mask_cpu(Image image, const Rectangle inside_area,
                    const Rectangle outside, MaskAlignmentParameters params);
Border detect_border_cpu(Image image, BorderScanParameters params,
                         const Rectangle outside_mask);

void blackfilter_cpu(Image image, BlackfilterParameters params);
void blurfilter_cpu(Image image, BlurfilterParameters params,
                    uint8_t abs_white_threshold);
void noisefilter_cpu(Image image, uint64_t intensity, uint8_t min_white_level);
void grayfilter_cpu(Image image, GrayfilterParameters params);

float detect_rotation_cpu(Image image, Rectangle mask,
                          const DeskewParameters params);
void deskew_cpu(Image source, Rectangle mask, float radians,
                Interpolation interpolate_type);

static const ImageBackend backend_cpu = {
    .name = "cpu",

    .wipe_rectangle = wipe_rectangle_cpu,
    .copy_rectangle = copy_rectangle_cpu,
    .center_image = center_image_cpu,
    .stretch_and_replace = stretch_and_replace_cpu,
    .resize_and_replace = resize_and_replace_cpu,
    .flip_rotate_90 = flip_rotate_90_cpu,
    .mirror = mirror_cpu,
    .shift_image = shift_image_cpu,

    .apply_masks = apply_masks_cpu,
    .apply_wipes = apply_wipes_cpu,
    .apply_border = apply_border_cpu,
    .detect_masks = detect_masks_cpu,
    .align_mask = align_mask_cpu,
    .detect_border = detect_border_cpu,

    .blackfilter = blackfilter_cpu,
    .blurfilter = blurfilter_cpu,
    .noisefilter = noisefilter_cpu,
    .grayfilter = grayfilter_cpu,

    .detect_rotation = detect_rotation_cpu,
    .deskew = deskew_cpu,
};

static const ImageBackend *backend = &backend_cpu;

const ImageBackend *image_backend_get(void) { return backend; }

void image_backend_select(UnpaperDevice device) {
  switch (device) {
  case UNPAPER_DEVICE_CPU:
    backend = &backend_cpu;
    return;

  case UNPAPER_DEVICE_CUDA:
#if defined(UNPAPER_WITH_CUDA) && (UNPAPER_WITH_CUDA)
    extern const ImageBackend backend_cuda;
    backend = &backend_cuda;
    return;
#else
    errOutput("CUDA backend requested, but this build has no CUDA support.");
#endif
    return;
  }

  errOutput("unknown device value: %" PRId32, (int32_t)device);
}
