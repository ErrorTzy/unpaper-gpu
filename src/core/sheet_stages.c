// SPDX-FileCopyrightText: 2025 The unpaper authors
//
// SPDX-License-Identifier: GPL-2.0-only

#include "src/core/sheet_stages.h"

#include "imageprocess/backend.h"
#include "imageprocess/blit.h"
#include "imageprocess/deskew.h"
#include "imageprocess/filters.h"
#include "imageprocess/masks.h"
#include "imageprocess/pixel.h"
#include "lib/encode_queue.h"
#include "lib/logging.h"
#include "unpaper.h"

#include <libswscale/swscale.h>
#include <string.h>

// isExcluded is already defined in parse.h (included via options.h)

static Layout sheet_stage_resolve_layout(const SheetProcessState *state,
                                         const Options *options) {
  Layout layout = options->layout;
  if (state->layout_override >= 0 && state->layout_override < LAYOUTS_COUNT) {
    layout = (Layout)state->layout_override;
  }
  return layout;
}

void sheet_stage_context_init(SheetStageContext *ctx, SheetProcessState *state,
                              const SheetProcessConfig *config) {
  const Options *options = config->options;

  ctx->layout = sheet_stage_resolve_layout(state, options);
  ctx->output_pixel_format = options->output_pixel_format;
  ctx->mask_params = options->mask_detection_parameters;
  ctx->blackfilter_params = options->blackfilter_parameters;

  perf_recorder_init(&state->perf, options->perf,
                     options->device == UNPAPER_DEVICE_CUDA);
}

static bool sheet_stage_decode(SheetProcessState *state,
                               const SheetProcessConfig *config,
                               SheetStageContext *ctx) {
  const Options *options = config->options;
  enum AVPixelFormat output_pixel_format = ctx->output_pixel_format;

  perf_stage_begin(&state->perf, PERF_STAGE_DECODE);

  for (int j = 0; j < state->input_count; j++) {
    if (state->input_files[j] != NULL ||
        (state->use_decoded_frames && state->decoded_frames[j] != NULL) ||
        (state->use_gpu_decoded_images &&
         state->gpu_decoded_images[j].frame != NULL)) {
      // Priority: GPU-decoded images > CPU-decoded frames > file load
      if (state->use_gpu_decoded_images &&
          state->gpu_decoded_images[j].frame != NULL) {
        // Use GPU-decoded image directly (already GPU-resident)
        state->page = state->gpu_decoded_images[j];
        state->gpu_decoded_images[j] = EMPTY_IMAGE; // Transfer ownership
      } else if (state->use_decoded_frames &&
                 state->decoded_frames[j] != NULL) {
        // Create image from pre-decoded frame
        AVFrame *decoded = state->decoded_frames[j];
        Rectangle area = rectangle_from_size(
            POINT_ORIGIN, (RectangleSize){.width = decoded->width,
                                          .height = decoded->height});

        switch (decoded->format) {
        case AV_PIX_FMT_Y400A:
        case AV_PIX_FMT_GRAY8:
        case AV_PIX_FMT_RGB24:
        case AV_PIX_FMT_MONOBLACK:
        case AV_PIX_FMT_MONOWHITE:
          state->page = create_image(size_of_rectangle(area), decoded->format,
                                     false, options->sheet_background,
                                     options->abs_black_threshold);
          av_frame_free(&state->page.frame);
          // Transfer ownership of the decoded frame into the Image.
          state->page.frame = decoded;
          state->decoded_frames[j] = NULL;
          break;
        case AV_PIX_FMT_PAL8: {
          state->page = create_image(size_of_rectangle(area), AV_PIX_FMT_RGB24,
                                     false, options->sheet_background,
                                     options->abs_black_threshold);
          const uint32_t *palette = (const uint32_t *)decoded->data[1];
          scan_rectangle(area) {
            const uint8_t palette_index =
                decoded->data[0][decoded->linesize[0] * y + x];
            set_pixel(state->page, (Point){x, y},
                      pixel_from_value(palette[palette_index]));
          }
          av_frame_free(&decoded);
          state->decoded_frames[j] = NULL;
        } break;
        default: {
          // FFmpeg decoders may output YUV and other formats. Convert to RGB24
          // so the rest of the pipeline can operate on it.
          state->page = create_image(size_of_rectangle(area), AV_PIX_FMT_RGB24,
                                     false, options->sheet_background,
                                     options->abs_black_threshold);

          struct SwsContext *sws_ctx = sws_getContext(
              decoded->width, decoded->height, (enum AVPixelFormat)decoded->format,
              decoded->width, decoded->height, AV_PIX_FMT_RGB24, SWS_BILINEAR,
              NULL, NULL, NULL);
          if (sws_ctx == NULL) {
            free_image(&state->page);
            state->page = EMPTY_IMAGE;
          } else {
            sws_scale(sws_ctx, (const uint8_t *const *)decoded->data,
                      decoded->linesize, 0, decoded->height,
                      state->page.frame->data, state->page.frame->linesize);
            sws_freeContext(sws_ctx);
          }

          av_frame_free(&decoded);
          state->decoded_frames[j] = NULL;
        } break;
        }
      } else {
        // Load from file
        loadImage(state->input_files[j], &state->page,
                  options->sheet_background, options->abs_black_threshold);
      }

      if (output_pixel_format == AV_PIX_FMT_NONE && state->page.frame != NULL) {
        output_pixel_format = state->page.frame->format;
      }

      // Pre-rotate
      if (options->pre_rotate != 0) {
        flip_rotate_90(&state->page, options->pre_rotate / 90);
      }

      // Set sheet size based on input
      RectangleSize inputSheetSize = {
          .width = state->page.frame->width * state->input_count,
          .height = state->page.frame->height,
      };
      state->input_size = coerce_size(
          state->input_size, coerce_size(options->sheet_size, inputSheetSize));
    } else {
      state->page = EMPTY_IMAGE;
    }

    // Allocate sheet buffer if needed
    if ((state->sheet.frame == NULL) && (state->input_size.width != -1) &&
        (state->input_size.height != -1)) {
      state->sheet =
          create_image(state->input_size, AV_PIX_FMT_RGB24, true,
                       options->sheet_background, options->abs_black_threshold);
    }

    // Place page into sheet buffer
    if (state->page.frame != NULL) {
      center_image(
          state->page, state->sheet,
          (Point){(state->input_size.width * j / state->input_count), 0},
          (RectangleSize){(state->input_size.width / state->input_count),
                          state->input_size.height});
    }
  }

  // Handle all-blank case
  if (state->sheet.frame == NULL) {
    state->input_size = state->previous_size;
    if ((state->input_size.width == -1) || (state->input_size.height == -1)) {
      verboseLog(VERBOSE_NORMAL, "ERROR: sheet size unknown\n");
      return false;
    }
    state->sheet =
        create_image(state->input_size, AV_PIX_FMT_RGB24, true,
                     options->sheet_background, options->abs_black_threshold);
  }

  state->previous_size = state->input_size;
  perf_stage_end(&state->perf, PERF_STAGE_DECODE);

  ctx->output_pixel_format = output_pixel_format;
  return true;
}

static bool sheet_stage_pre(SheetProcessState *state,
                            const SheetProcessConfig *config,
                            SheetStageContext *ctx) {
  const Options *options = config->options;
  Layout layout = ctx->layout;

  // Upload to GPU if using CUDA
  if (options->device == UNPAPER_DEVICE_CUDA) {
    perf_stage_begin(&state->perf, PERF_STAGE_UPLOAD);
    image_ensure_cuda(&state->sheet);
    perf_stage_end(&state->perf, PERF_STAGE_UPLOAD);
  }

  // Pre-mirroring
  if (options->pre_mirror.horizontal || options->pre_mirror.vertical) {
    mirror(state->sheet, options->pre_mirror);
  }

  // Pre-shifting
  if (options->pre_shift.horizontal != 0 || options->pre_shift.vertical != 0) {
    shift_image(&state->sheet, options->pre_shift);
  }

  // Pre-masking
  if (config->pre_mask_count > 0) {
    apply_masks(state->sheet, config->pre_masks, config->pre_mask_count,
                options->mask_color);
  }

  // Stretch
  state->input_size =
      coerce_size(options->stretch_size, size_of_image(state->sheet));
  state->input_size.width *= options->pre_zoom_factor;
  state->input_size.height *= options->pre_zoom_factor;
  stretch_and_replace(&state->sheet, state->input_size,
                      options->interpolate_type);

  // Resize if needed
  if (options->page_size.width != -1 || options->page_size.height != -1) {
    state->input_size =
        coerce_size(options->page_size, size_of_image(state->sheet));
    resize_and_replace(&state->sheet, state->input_size,
                       options->interpolate_type);
  }

  // Handle layout - set auto points and masks
  if (layout == LAYOUT_SINGLE) {
    if (state->point_count == 0) {
      state->points[state->point_count++] = (Point){
          state->sheet.frame->width / 2, state->sheet.frame->height / 2};
    }
    if (state->mask_max_width == -1) {
      state->mask_max_width = state->sheet.frame->width;
    }
    if (state->mask_max_height == -1) {
      state->mask_max_height = state->sheet.frame->height;
    }
    if (state->outside_borderscan_mask_count == 0) {
      state->outside_borderscan_mask[state->outside_borderscan_mask_count++] =
          full_image(state->sheet);
    }
  } else if (layout == LAYOUT_DOUBLE) {
    if (state->point_count == 0) {
      state->points[state->point_count++] = (Point){
          state->sheet.frame->width / 4, state->sheet.frame->height / 2};
      state->points[state->point_count++] =
          (Point){state->sheet.frame->width - state->sheet.frame->width / 4,
                  state->sheet.frame->height / 2};
    }
    if (state->mask_max_width == -1) {
      state->mask_max_width = state->sheet.frame->width / 2;
    }
    if (state->mask_max_height == -1) {
      state->mask_max_height = state->sheet.frame->height;
    }
    if (state->outside_borderscan_mask_count == 0) {
      state->outside_borderscan_mask[state->outside_borderscan_mask_count++] =
          (Rectangle){{POINT_ORIGIN,
                       {state->sheet.frame->width / 2,
                        state->sheet.frame->height - 1}}};
      state->outside_borderscan_mask[state->outside_borderscan_mask_count++] =
          (Rectangle){{{state->sheet.frame->width / 2, 0},
                       {state->sheet.frame->width - 1,
                        state->sheet.frame->height - 1}}};
    }
  }

  if (state->mask_max_width == -1) {
    state->mask_max_width = state->sheet.frame->width;
  }
  if (state->mask_max_height == -1) {
    state->mask_max_height = state->sheet.frame->height;
  }

  // Pre-wipe
  if (!isExcluded(state->sheet_nr, options->no_wipe_multi_index,
                  options->ignore_multi_index)) {
    apply_wipes(state->sheet, options->pre_wipes, options->mask_color);
  }

  // Pre-border
  if (!isExcluded(state->sheet_nr, options->no_border_multi_index,
                  options->ignore_multi_index)) {
    apply_border(state->sheet, options->pre_border, options->mask_color);
  }

  // Create local copy of mask detection params with our max sizes
  ctx->mask_params = options->mask_detection_parameters;
  ctx->mask_params.maximum_width = state->mask_max_width;
  ctx->mask_params.maximum_height = state->mask_max_height;

  // Create local copy of blackfilter params
  ctx->blackfilter_params = options->blackfilter_parameters;
  if (config->blackfilter_exclude_count == 0 && layout != LAYOUT_NONE) {
    // Auto-generate exclusion zones
    RectangleSize sheetSize = size_of_image(state->sheet);
    if (layout == LAYOUT_SINGLE) {
      ctx->blackfilter_params
          .exclusions[ctx->blackfilter_params.exclusions_count++] =
          rectangle_from_size((Point){sheetSize.width / 4, sheetSize.height / 4},
                              (RectangleSize){.width = sheetSize.width / 2,
                                              .height = sheetSize.height / 2});
    } else if (layout == LAYOUT_DOUBLE) {
      RectangleSize filterSize = {.width = sheetSize.width / 4,
                                  .height = sheetSize.height / 2};
      Point firstFilterOrigin = {sheetSize.width / 8, sheetSize.height / 4};
      Point secondFilterOrigin = shift_point(
          firstFilterOrigin, (Delta){state->sheet.frame->width / 2});
      ctx->blackfilter_params
          .exclusions[ctx->blackfilter_params.exclusions_count++] =
          rectangle_from_size(firstFilterOrigin, filterSize);
      ctx->blackfilter_params
          .exclusions[ctx->blackfilter_params.exclusions_count++] =
          rectangle_from_size(secondFilterOrigin, filterSize);
    }
  }

  return true;
}

static bool sheet_stage_filters(SheetProcessState *state,
                                const SheetProcessConfig *config,
                                SheetStageContext *ctx) {
  const Options *options = config->options;
  int nr = state->sheet_nr;

  perf_stage_begin(&state->perf, PERF_STAGE_FILTERS);

  // Black area filter
  if (!isExcluded(nr, options->no_blackfilter_multi_index,
                  options->ignore_multi_index)) {
    blackfilter(state->sheet, ctx->blackfilter_params);
  }

  // Noise filter
  if (!isExcluded(nr, options->no_noisefilter_multi_index,
                  options->ignore_multi_index)) {
    noisefilter(state->sheet, options->noisefilter_intensity,
                options->abs_white_threshold);
  }

  // Blur filter
  if (!isExcluded(nr, options->no_blurfilter_multi_index,
                  options->ignore_multi_index)) {
    blurfilter(state->sheet, options->blurfilter_parameters,
               options->abs_white_threshold);
  }

  perf_stage_end(&state->perf, PERF_STAGE_FILTERS);
  return true;
}

static bool sheet_stage_masks(SheetProcessState *state,
                              const SheetProcessConfig *config,
                              SheetStageContext *ctx) {
  const Options *options = config->options;
  int nr = state->sheet_nr;

  perf_stage_begin(&state->perf, PERF_STAGE_MASKS);

  // Mask detection
  if (!isExcluded(nr, options->no_mask_scan_multi_index,
                  options->ignore_multi_index)) {
    detect_masks(state->sheet, ctx->mask_params, state->points,
                 state->point_count, state->masks);
  }

  // Apply permanent masks
  if (state->mask_count > 0) {
    apply_masks(state->sheet, state->masks, state->mask_count,
                options->mask_color);
  }

  // Gray filter
  if (!isExcluded(nr, options->no_grayfilter_multi_index,
                  options->ignore_multi_index)) {
    grayfilter(state->sheet, options->grayfilter_parameters);
  }

  perf_stage_end(&state->perf, PERF_STAGE_MASKS);
  return true;
}

static bool sheet_stage_deskew(SheetProcessState *state,
                               const SheetProcessConfig *config,
                               SheetStageContext *ctx) {
  const Options *options = config->options;
  int nr = state->sheet_nr;

  perf_stage_begin(&state->perf, PERF_STAGE_DESKEW);

  // Re-detect masks for better accuracy
  if (!isExcluded(nr, options->no_mask_scan_multi_index,
                  options->ignore_multi_index)) {
    state->mask_count = detect_masks(state->sheet, ctx->mask_params,
                                     state->points, state->point_count,
                                     state->masks);
  }

  // Auto-deskew each mask
  for (size_t i = 0; i < state->mask_count; i++) {
    float rotation = detect_rotation(state->sheet, state->masks[i],
                                     options->deskew_parameters);
    if (rotation != 0.0) {
      deskew(state->sheet, state->masks[i], rotation,
             options->interpolate_type);
    }
  }

  perf_stage_end(&state->perf, PERF_STAGE_DESKEW);
  return true;
}

static bool sheet_stage_post(SheetProcessState *state,
                             const SheetProcessConfig *config,
                             SheetStageContext *ctx) {
  const Options *options = config->options;
  int nr = state->sheet_nr;
  Layout layout = ctx->layout;

  // Mask centering
  perf_stage_begin(&state->perf, PERF_STAGE_MASKS);
  if (!isExcluded(nr, options->no_mask_center_multi_index,
                  options->ignore_multi_index)) {
    if (!isExcluded(nr, options->no_mask_scan_multi_index,
                    options->ignore_multi_index)) {
      state->mask_count = detect_masks(state->sheet, ctx->mask_params,
                                       state->points, state->point_count,
                                       state->masks);
    }
    for (size_t i = 0; i < state->mask_count; i++) {
      center_mask(state->sheet, state->points[i], state->masks[i]);
    }
  }

  // Explicit wipe
  if (!isExcluded(nr, options->no_wipe_multi_index,
                  options->ignore_multi_index)) {
    Wipes wipes = options->wipes;
    // Add middle wipe for double layout
    if (layout == LAYOUT_DOUBLE &&
        (config->middle_wipe[0] > 0 || config->middle_wipe[1] > 0)) {
      wipes.areas[wipes.count++] = (Rectangle){{
          {state->sheet.frame->width / 2 - config->middle_wipe[0], 0},
          {state->sheet.frame->width / 2 + config->middle_wipe[1],
           state->sheet.frame->height - 1},
      }};
    }
    apply_wipes(state->sheet, wipes, options->mask_color);
  }

  // Explicit border
  if (!isExcluded(nr, options->no_border_multi_index,
                  options->ignore_multi_index)) {
    apply_border(state->sheet, options->border, options->mask_color);
  }

  // Border detection
  if (!isExcluded(nr, options->no_border_scan_multi_index,
                  options->ignore_multi_index)) {
    Rectangle autoborderMask[state->outside_borderscan_mask_count];
    for (size_t i = 0; i < state->outside_borderscan_mask_count; i++) {
      autoborderMask[i] = border_to_mask(
          state->sheet,
          detect_border(state->sheet, options->border_scan_parameters,
                        state->outside_borderscan_mask[i]));
    }
    apply_masks(state->sheet, autoborderMask,
                state->outside_borderscan_mask_count, options->mask_color);
    for (size_t i = 0; i < state->outside_borderscan_mask_count; i++) {
      if (!isExcluded(nr, options->no_border_align_multi_index,
                      options->ignore_multi_index)) {
        align_mask(state->sheet, autoborderMask[i],
                   state->outside_borderscan_mask[i],
                   options->mask_alignment_parameters);
      }
    }
  }

  // Post-wipe
  if (!isExcluded(nr, options->no_wipe_multi_index,
                  options->ignore_multi_index)) {
    apply_wipes(state->sheet, options->post_wipes, options->mask_color);
  }

  // Post-border
  if (!isExcluded(nr, options->no_border_multi_index,
                  options->ignore_multi_index)) {
    apply_border(state->sheet, options->post_border, options->mask_color);
  }

  perf_stage_end(&state->perf, PERF_STAGE_MASKS);

  // Post-mirroring
  if (options->post_mirror.horizontal || options->post_mirror.vertical) {
    mirror(state->sheet, options->post_mirror);
  }

  // Post-shifting
  if (options->post_shift.horizontal != 0 ||
      options->post_shift.vertical != 0) {
    shift_image(&state->sheet, options->post_shift);
  }

  // Post-rotating
  if (options->post_rotate != 0) {
    flip_rotate_90(&state->sheet, options->post_rotate / 90);
  }

  // Post-stretch
  state->input_size =
      coerce_size(options->post_stretch_size, size_of_image(state->sheet));
  state->input_size.width *= options->post_zoom_factor;
  state->input_size.height *= options->post_zoom_factor;
  stretch_and_replace(&state->sheet, state->input_size,
                      options->interpolate_type);

  // Post-size
  if (options->post_page_size.width != -1 ||
      options->post_page_size.height != -1) {
    state->input_size =
        coerce_size(options->post_page_size, size_of_image(state->sheet));
    resize_and_replace(&state->sheet, state->input_size,
                       options->interpolate_type);
  }

  return true;
}

static bool sheet_stage_output(SheetProcessState *state,
                               const SheetProcessConfig *config,
                               SheetStageContext *ctx) {
  const Options *options = config->options;
  enum AVPixelFormat output_pixel_format = ctx->output_pixel_format;

  if (output_pixel_format == AV_PIX_FMT_NONE) {
    // Try to detect output format from file extension first
    if (state->output_count > 0 && state->output_files[0] != NULL) {
      output_pixel_format =
          detectPixelFormatFromExtension(state->output_files[0]);
    }
    // Fall back to image format if extension not recognized
    if (output_pixel_format == AV_PIX_FMT_NONE) {
      output_pixel_format = state->sheet.frame->format;
    }
  }

  // Check if GPU encode path can be used (auto-detected)
  // GPU pipeline: skip D2H transfer, encode directly from GPU memory
  bool use_gpu_encode = false;
  if (state->encode_queue != NULL &&
      encode_queue_gpu_enabled(state->encode_queue) &&
      image_is_gpu_resident(&state->sheet)) {
    // GPU-resident image with GPU encoding enabled - use zero-copy path
    use_gpu_encode = true;
  }

  if (use_gpu_encode) {
    // GPU pipeline path: encode directly from GPU memory (zero-copy)
    // No D2H transfer needed - skip PERF_STAGE_DOWNLOAD entirely
    void *gpu_ptr = image_get_gpu_ptr(&state->sheet);
    size_t pitch = image_get_gpu_pitch(&state->sheet);
    int width = state->sheet.frame->width;
    int height = state->sheet.frame->height;
    int channels = (state->sheet.frame->format == AV_PIX_FMT_GRAY8) ? 1 : 3;

    if (gpu_ptr != NULL) {
      encode_queue_submit_gpu(state->encode_queue, gpu_ptr, pitch, width, height,
                              channels, state->output_files, state->output_count,
                              output_pixel_format, state->job_index);
    } else {
      // Fallback: GPU pointer not available, use CPU path
      use_gpu_encode = false;
    }
  }

  if (!use_gpu_encode) {
    // CPU path: download from GPU and encode on CPU
    perf_stage_begin(&state->perf, PERF_STAGE_DOWNLOAD);
    image_ensure_cpu(&state->sheet);
    perf_stage_end(&state->perf, PERF_STAGE_DOWNLOAD);

    // Check if async encoding is enabled
    if (state->encode_queue != NULL) {
      // Async path: submit to encode queue
      // Clone the frame since encode queue takes ownership
      AVFrame *frame_copy = av_frame_clone(state->sheet.frame);
      if (frame_copy) {
        // Note: pinned memory optimization would require changes to
        // image_ensure_cpu For now, we use non-pinned memory for the encode
        // queue submission
        encode_queue_submit(state->encode_queue, frame_copy,
                            state->output_files, state->output_count,
                            output_pixel_format, state->job_index, false);
      }
      // Encode timing is handled by encode queue, so skip perf stage
    } else {
      // Sync path: encode directly
      perf_stage_begin(&state->perf, PERF_STAGE_ENCODE);
      if (state->output_count == 1) {
        saveImage(state->output_files[0], state->sheet, output_pixel_format);
      } else {
        for (int j = 0; j < state->output_count; j++) {
          state->page = create_compatible_image(
              state->sheet,
              (RectangleSize){state->sheet.frame->width / state->output_count,
                              state->sheet.frame->height},
              false);
          copy_rectangle(state->sheet, state->page,
                         (Rectangle){{{state->page.frame->width * j, 0},
                                      {state->page.frame->width * j +
                                           state->page.frame->width,
                                       state->page.frame->height}}},
                         POINT_ORIGIN);
          saveImage(state->output_files[j], state->page, output_pixel_format);
          free_image(&state->page);
        }
      }
      perf_stage_end(&state->perf, PERF_STAGE_ENCODE);
    }
  }

  ctx->output_pixel_format = output_pixel_format;
  return true;
}

typedef bool (*SheetStageShouldRunFn)(const SheetProcessState *state,
                                      const SheetProcessConfig *config,
                                      const SheetStageContext *ctx);

typedef struct {
  const char *name;
  SheetStageShouldRunFn should_run;
  bool (*run)(SheetProcessState *state, const SheetProcessConfig *config,
              SheetStageContext *ctx);
} SheetStage;

static bool sheet_stage_should_run_deskew(const SheetProcessState *state,
                                          const SheetProcessConfig *config,
                                          const SheetStageContext *ctx) {
  (void)ctx;
  return !isExcluded(state->sheet_nr, config->options->no_deskew_multi_index,
                     config->options->ignore_multi_index);
}

static bool sheet_stage_should_run_output(const SheetProcessState *state,
                                          const SheetProcessConfig *config,
                                          const SheetStageContext *ctx) {
  (void)state;
  (void)ctx;
  return config->options->write_output;
}

static const SheetStage sheet_stages[] = {
    {.name = "decode", .should_run = NULL, .run = sheet_stage_decode},
    {.name = "pre", .should_run = NULL, .run = sheet_stage_pre},
    {.name = "filters", .should_run = NULL, .run = sheet_stage_filters},
    {.name = "masks", .should_run = NULL, .run = sheet_stage_masks},
    {.name = "deskew",
     .should_run = sheet_stage_should_run_deskew,
     .run = sheet_stage_deskew},
    {.name = "post", .should_run = NULL, .run = sheet_stage_post},
    {.name = "output",
     .should_run = sheet_stage_should_run_output,
     .run = sheet_stage_output},
};

bool sheet_stages_run(SheetProcessState *state,
                      const SheetProcessConfig *config,
                      SheetStageContext *ctx) {
  for (size_t i = 0; i < sizeof(sheet_stages) / sizeof(sheet_stages[0]); i++) {
    const SheetStage *stage = &sheet_stages[i];
    if (stage->should_run != NULL &&
        !stage->should_run(state, config, ctx)) {
      continue;
    }
    if (!stage->run(state, config, ctx)) {
      return false;
    }
  }

  // Print perf if enabled
  if (config->options->perf) {
    perf_recorder_print(&state->perf, state->sheet_nr,
                        config->options->device == UNPAPER_DEVICE_CUDA ? "cuda"
                                                                       : "cpu");
  }

  return true;
}
