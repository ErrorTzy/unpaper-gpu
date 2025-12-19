// SPDX-FileCopyrightText: 2025 The unpaper authors
//
// SPDX-License-Identifier: GPL-2.0-only

#include "sheet_process.h"

#include "src/core/sheet_stages.h"

#include <libavutil/frame.h>
#include <string.h>

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
  state->layout_override = job->layout_override;

  for (int i = 0; i < job->input_count; i++) {
    const BatchInput *input = batch_job_input(job, i);
    state->input_files[i] = batch_input_path(input);
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

bool process_sheet(SheetProcessState *state, const SheetProcessConfig *config) {
  SheetStageContext ctx;

  sheet_stage_context_init(&ctx, state, config);
  return sheet_stages_run(state, config, &ctx);
}
