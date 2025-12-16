// SPDX-FileCopyrightText: 2025 The unpaper authors
//
// SPDX-License-Identifier: GPL-2.0-only

#include "sheet_process.h"

#include "imageprocess/backend.h"
#include "imageprocess/blit.h"
#include "imageprocess/deskew.h"
#include "imageprocess/filters.h"
#include "imageprocess/masks.h"
#include "lib/encode_queue.h"
#include "lib/logging.h"
#include "unpaper.h"

#include <libavutil/pixfmt.h>
#include <string.h>

// isExcluded is already defined in parse.h (included via options.h)

void sheet_process_config_init(
    SheetProcessConfig *config, const Options *options,
    const Rectangle *pre_masks, size_t pre_mask_count,
    const Point *initial_points, size_t initial_point_count,
    const int32_t middle_wipe[2], const Rectangle *blackfilter_exclude,
    size_t blackfilter_exclude_count) {
  config->options = options;
  config->pre_masks = pre_masks;
  config->pre_mask_count = pre_mask_count;
  config->initial_points = initial_points;
  config->initial_point_count = initial_point_count;
  config->middle_wipe[0] = middle_wipe[0];
  config->middle_wipe[1] = middle_wipe[1];
  config->blackfilter_exclude = blackfilter_exclude;
  config->blackfilter_exclude_count = blackfilter_exclude_count;
}

void sheet_process_state_init(SheetProcessState *state,
                              const SheetProcessConfig *config, BatchJob *job) {
  memset(state, 0, sizeof(*state));

  state->sheet_nr = job->sheet_nr;
  state->input_count = job->input_count;
  state->output_count = job->output_count;

  for (int i = 0; i < job->input_count; i++) {
    state->input_files[i] = job->input_files[i];
  }
  for (int i = 0; i < job->output_count; i++) {
    state->output_files[i] = job->output_files[i];
  }

  // Initialize pre-decoded frames
  for (int i = 0; i < BATCH_MAX_FILES_PER_SHEET; i++) {
    state->decoded_frames[i] = NULL;
  }
  state->use_decoded_frames = false;

  // Initialize GPU-decoded images
  for (int i = 0; i < BATCH_MAX_FILES_PER_SHEET; i++) {
    state->gpu_decoded_images[i] = EMPTY_IMAGE;
  }
  state->use_gpu_decoded_images = false;

  // Copy initial points
  state->point_count = config->initial_point_count;
  for (size_t i = 0; i < config->initial_point_count; i++) {
    state->points[i] = config->initial_points[i];
  }

  state->sheet = EMPTY_IMAGE;
  state->page = EMPTY_IMAGE;
  state->mask_count = 0;
  state->outside_borderscan_mask_count = 0;

  state->input_size = (RectangleSize){-1, -1};
  state->previous_size = (RectangleSize){-1, -1};

  // Initialize encode queue to NULL (sync mode)
  state->encode_queue = NULL;
  state->job_index = 0;

  // Copy default max dimensions (will be adjusted per-sheet)
  state->mask_max_width =
      config->options->mask_detection_parameters.maximum_width;
  state->mask_max_height =
      config->options->mask_detection_parameters.maximum_height;

  perf_recorder_init(&state->perf, config->options->perf,
                     config->options->device == UNPAPER_DEVICE_CUDA);
}

void sheet_process_state_set_decoded(SheetProcessState *state, AVFrame *frame,
                                     int input_index) {
  if (!state || input_index < 0 || input_index >= BATCH_MAX_FILES_PER_SHEET) {
    return;
  }
  state->decoded_frames[input_index] = frame;
  state->use_decoded_frames = true;
}

void sheet_process_state_set_gpu_decoded_image(SheetProcessState *state,
                                               Image image, int input_index) {
  if (!state || input_index < 0 || input_index >= BATCH_MAX_FILES_PER_SHEET) {
    return;
  }
  state->gpu_decoded_images[input_index] = image;
  state->use_gpu_decoded_images = true;
}

void sheet_process_state_set_encode_queue(SheetProcessState *state,
                                          struct EncodeQueue *encode_queue,
                                          int job_index) {
  if (!state) {
    return;
  }
  state->encode_queue = encode_queue;
  state->job_index = job_index;
}

void sheet_process_state_cleanup(SheetProcessState *state) {
  free_image(&state->sheet);
  free_image(&state->page);

  // Free any remaining pre-decoded frames
  for (int i = 0; i < BATCH_MAX_FILES_PER_SHEET; i++) {
    if (state->decoded_frames[i] != NULL) {
      av_frame_free(&state->decoded_frames[i]);
      state->decoded_frames[i] = NULL;
    }
  }

  // Free any remaining GPU-decoded images
  for (int i = 0; i < BATCH_MAX_FILES_PER_SHEET; i++) {
    if (state->gpu_decoded_images[i].frame != NULL) {
      free_image(&state->gpu_decoded_images[i]);
    }
  }
}

// coerce_size is already declared in primitives.h

bool process_sheet(SheetProcessState *state, const SheetProcessConfig *config) {
  const Options *options = config->options;
  int nr = state->sheet_nr;

  perf_recorder_init(&state->perf, options->perf,
                     options->device == UNPAPER_DEVICE_CUDA);

  // Load input images
  perf_stage_begin(&state->perf, PERF_STAGE_DECODE);

  enum AVPixelFormat output_pixel_format = options->output_pixel_format;

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
          state->page.frame = av_frame_clone(decoded);
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
        } break;
        default:
          state->page = EMPTY_IMAGE;
          break;
        }
        // Clear the decoded frame pointer (ownership moved)
        state->decoded_frames[j] = NULL;
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
  if (options->layout == LAYOUT_SINGLE) {
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
  } else if (options->layout == LAYOUT_DOUBLE) {
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
  if (!isExcluded(nr, options->no_wipe_multi_index,
                  options->ignore_multi_index)) {
    apply_wipes(state->sheet, options->pre_wipes, options->mask_color);
  }

  // Pre-border
  if (!isExcluded(nr, options->no_border_multi_index,
                  options->ignore_multi_index)) {
    apply_border(state->sheet, options->pre_border, options->mask_color);
  }

  // Create local copy of mask detection params with our max sizes
  MaskDetectionParameters mask_params = options->mask_detection_parameters;
  mask_params.maximum_width = state->mask_max_width;
  mask_params.maximum_height = state->mask_max_height;

  // Create local copy of blackfilter params
  BlackfilterParameters bf_params = options->blackfilter_parameters;
  if (config->blackfilter_exclude_count == 0 &&
      options->layout != LAYOUT_NONE) {
    // Auto-generate exclusion zones
    RectangleSize sheetSize = size_of_image(state->sheet);
    if (options->layout == LAYOUT_SINGLE) {
      bf_params.exclusions[bf_params.exclusions_count++] = rectangle_from_size(
          (Point){sheetSize.width / 4, sheetSize.height / 4},
          (RectangleSize){.width = sheetSize.width / 2,
                          .height = sheetSize.height / 2});
    } else if (options->layout == LAYOUT_DOUBLE) {
      RectangleSize filterSize = {.width = sheetSize.width / 4,
                                  .height = sheetSize.height / 2};
      Point firstFilterOrigin = {sheetSize.width / 8, sheetSize.height / 4};
      Point secondFilterOrigin = shift_point(
          firstFilterOrigin, (Delta){state->sheet.frame->width / 2});
      bf_params.exclusions[bf_params.exclusions_count++] =
          rectangle_from_size(firstFilterOrigin, filterSize);
      bf_params.exclusions[bf_params.exclusions_count++] =
          rectangle_from_size(secondFilterOrigin, filterSize);
    }
  }

  // Filters stage
  perf_stage_begin(&state->perf, PERF_STAGE_FILTERS);

  // Black area filter
  if (!isExcluded(nr, options->no_blackfilter_multi_index,
                  options->ignore_multi_index)) {
    blackfilter(state->sheet, bf_params);
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

  // Masks stage
  perf_stage_begin(&state->perf, PERF_STAGE_MASKS);

  // Mask detection
  if (!isExcluded(nr, options->no_mask_scan_multi_index,
                  options->ignore_multi_index)) {
    detect_masks(state->sheet, mask_params, state->points, state->point_count,
                 state->masks);
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

  // Deskew
  if (!isExcluded(nr, options->no_deskew_multi_index,
                  options->ignore_multi_index)) {
    perf_stage_begin(&state->perf, PERF_STAGE_DESKEW);

    // Re-detect masks for better accuracy
    if (!isExcluded(nr, options->no_mask_scan_multi_index,
                    options->ignore_multi_index)) {
      state->mask_count = detect_masks(state->sheet, mask_params, state->points,
                                       state->point_count, state->masks);
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
  }

  // Mask centering
  perf_stage_begin(&state->perf, PERF_STAGE_MASKS);
  if (!isExcluded(nr, options->no_mask_center_multi_index,
                  options->ignore_multi_index)) {
    if (!isExcluded(nr, options->no_mask_scan_multi_index,
                    options->ignore_multi_index)) {
      state->mask_count = detect_masks(state->sheet, mask_params, state->points,
                                       state->point_count, state->masks);
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
    if (options->layout == LAYOUT_DOUBLE &&
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

  // Write output
  if (options->write_output) {
    if (output_pixel_format == AV_PIX_FMT_NONE) {
      output_pixel_format = state->sheet.frame->format;
    }

    // Check if GPU pipeline path can be used (PR38)
    // GPU pipeline: skip D2H transfer, encode directly from GPU memory
    bool use_gpu_encode = false;
    if (options->gpu_pipeline && state->encode_queue != NULL &&
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
        encode_queue_submit_gpu(state->encode_queue, gpu_ptr, pitch, width,
                                height, channels, state->output_files,
                                state->output_count, state->job_index);
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
  }

  // Print perf if enabled
  if (options->perf) {
    perf_recorder_print(&state->perf, nr,
                        options->device == UNPAPER_DEVICE_CUDA ? "cuda"
                                                               : "cpu");
  }

  return true;
}
