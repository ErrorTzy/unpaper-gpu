// SPDX-FileCopyrightText: 2025 The unpaper authors
//
// SPDX-License-Identifier: GPL-2.0-only

#include "imageprocess/backend.h"

#include "lib/logging.h"

static void cuda_unimplemented(const char *op_name) {
  errOutput("CUDA backend selected, but it is not implemented yet (%s).",
            op_name);
}

static void wipe_rectangle_cuda(Image image, Rectangle input_area,
                                Pixel color) {
  (void)image;
  (void)input_area;
  (void)color;
  cuda_unimplemented("wipe_rectangle");
}

static void copy_rectangle_cuda(Image source, Image target,
                                Rectangle source_area, Point target_coords) {
  (void)source;
  (void)target;
  (void)source_area;
  (void)target_coords;
  cuda_unimplemented("copy_rectangle");
}

static void center_image_cuda(Image source, Image target, Point target_origin,
                              RectangleSize target_size) {
  (void)source;
  (void)target;
  (void)target_origin;
  (void)target_size;
  cuda_unimplemented("center_image");
}

static void stretch_and_replace_cuda(Image *pImage, RectangleSize size,
                                     Interpolation interpolate_type) {
  (void)pImage;
  (void)size;
  (void)interpolate_type;
  cuda_unimplemented("stretch_and_replace");
}

static void resize_and_replace_cuda(Image *pImage, RectangleSize size,
                                    Interpolation interpolate_type) {
  (void)pImage;
  (void)size;
  (void)interpolate_type;
  cuda_unimplemented("resize_and_replace");
}

static void flip_rotate_90_cuda(Image *pImage, RotationDirection direction) {
  (void)pImage;
  (void)direction;
  cuda_unimplemented("flip_rotate_90");
}

static void mirror_cuda(Image image, Direction direction) {
  (void)image;
  (void)direction;
  cuda_unimplemented("mirror");
}

static void shift_image_cuda(Image *pImage, Delta d) {
  (void)pImage;
  (void)d;
  cuda_unimplemented("shift_image");
}

static void apply_masks_cuda(Image image, const Rectangle masks[],
                             size_t masks_count, Pixel color) {
  (void)image;
  (void)masks;
  (void)masks_count;
  (void)color;
  cuda_unimplemented("apply_masks");
}

static void apply_wipes_cuda(Image image, Wipes wipes, Pixel color) {
  (void)image;
  (void)wipes;
  (void)color;
  cuda_unimplemented("apply_wipes");
}

static void apply_border_cuda(Image image, const Border border, Pixel color) {
  (void)image;
  (void)border;
  (void)color;
  cuda_unimplemented("apply_border");
}

static size_t detect_masks_cuda(Image image, MaskDetectionParameters params,
                                const Point points[], size_t points_count,
                                Rectangle masks[]) {
  (void)image;
  (void)params;
  (void)points;
  (void)points_count;
  (void)masks;
  cuda_unimplemented("detect_masks");
  return 0;
}

static void align_mask_cuda(Image image, const Rectangle inside_area,
                            const Rectangle outside,
                            MaskAlignmentParameters params) {
  (void)image;
  (void)inside_area;
  (void)outside;
  (void)params;
  cuda_unimplemented("align_mask");
}

static Border detect_border_cuda(Image image, BorderScanParameters params,
                                 const Rectangle outside_mask) {
  (void)image;
  (void)params;
  (void)outside_mask;
  cuda_unimplemented("detect_border");
  return (Border){0};
}

static void blackfilter_cuda(Image image, BlackfilterParameters params) {
  (void)image;
  (void)params;
  cuda_unimplemented("blackfilter");
}

static void blurfilter_cuda(Image image, BlurfilterParameters params,
                            uint8_t abs_white_threshold) {
  (void)image;
  (void)params;
  (void)abs_white_threshold;
  cuda_unimplemented("blurfilter");
}

static void noisefilter_cuda(Image image, uint64_t intensity,
                             uint8_t min_white_level) {
  (void)image;
  (void)intensity;
  (void)min_white_level;
  cuda_unimplemented("noisefilter");
}

static void grayfilter_cuda(Image image, GrayfilterParameters params) {
  (void)image;
  (void)params;
  cuda_unimplemented("grayfilter");
}

static float detect_rotation_cuda(Image image, Rectangle mask,
                                  const DeskewParameters params) {
  (void)image;
  (void)mask;
  (void)params;
  cuda_unimplemented("detect_rotation");
  return 0.0f;
}

static void deskew_cuda(Image source, Rectangle mask, float radians,
                        Interpolation interpolate_type) {
  (void)source;
  (void)mask;
  (void)radians;
  (void)interpolate_type;
  cuda_unimplemented("deskew");
}

const ImageBackend backend_cuda = {
    .name = "cuda",

    .wipe_rectangle = wipe_rectangle_cuda,
    .copy_rectangle = copy_rectangle_cuda,
    .center_image = center_image_cuda,
    .stretch_and_replace = stretch_and_replace_cuda,
    .resize_and_replace = resize_and_replace_cuda,
    .flip_rotate_90 = flip_rotate_90_cuda,
    .mirror = mirror_cuda,
    .shift_image = shift_image_cuda,

    .apply_masks = apply_masks_cuda,
    .apply_wipes = apply_wipes_cuda,
    .apply_border = apply_border_cuda,
    .detect_masks = detect_masks_cuda,
    .align_mask = align_mask_cuda,
    .detect_border = detect_border_cuda,

    .blackfilter = blackfilter_cuda,
    .blurfilter = blurfilter_cuda,
    .noisefilter = noisefilter_cuda,
    .grayfilter = grayfilter_cuda,

    .detect_rotation = detect_rotation_cuda,
    .deskew = deskew_cuda,
};
