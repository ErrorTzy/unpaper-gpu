// SPDX-FileCopyrightText: 2025 The unpaper authors
//
// SPDX-License-Identifier: GPL-2.0-only

#include <assert.h>
#include <stdbool.h>
#include <stdint.h>
#include <string.h>

#include "imageprocess/backend.h"
#include "imageprocess/image.h"
#include "imageprocess/pixel.h"
#include "lib/options.h"
#include "sheet_process.h"

static void set_multi_index_all(struct MultiIndex *idx) {
  idx->count = -1;
  idx->indexes = NULL;
}

static AVFrame *make_test_frame(int width, int height) {
  Image image = create_image((RectangleSize){width, height}, AV_PIX_FMT_GRAY8,
                             true, PIXEL_WHITE, 0);
  set_pixel(image, (Point){width / 2, height / 2}, PIXEL_BLACK);

  AVFrame *frame = image.frame;
  image.frame = NULL;
  return frame;
}

static void init_options(Options *options, Rectangle *blackfilter_exclude) {
  options_init(options);
  options_init_filter_defaults(options, blackfilter_exclude);
  options->device = UNPAPER_DEVICE_CPU;
  options->write_output = false;
  options->abs_black_threshold = 0;
  options->abs_white_threshold = WHITE;
}

static void init_job(BatchJob *job) {
  memset(job, 0, sizeof(*job));
  job->sheet_nr = 1;
  job->input_count = 1;
  job->output_count = 1;
  job->layout_override = -1;
  job->input_files[0] = NULL;
  job->output_files[0] = NULL;
}

static void configure_skip_filters_and_borders(Options *options) {
  set_multi_index_all(&options->no_blackfilter_multi_index);
  set_multi_index_all(&options->no_noisefilter_multi_index);
  set_multi_index_all(&options->no_blurfilter_multi_index);
  set_multi_index_all(&options->no_grayfilter_multi_index);
  set_multi_index_all(&options->no_wipe_multi_index);
  set_multi_index_all(&options->no_border_multi_index);
  set_multi_index_all(&options->no_border_scan_multi_index);
  set_multi_index_all(&options->no_border_align_multi_index);
  set_multi_index_all(&options->no_deskew_multi_index);
}

static void assert_rectangle(Rectangle rect, int x0, int y0, int x1, int y1) {
  assert(rect.vertex[0].x == x0);
  assert(rect.vertex[0].y == y0);
  assert(rect.vertex[1].x == x1);
  assert(rect.vertex[1].y == y1);
}

static void test_layout_single_defaults(void) {
  Options options;
  Rectangle blackfilter_exclude[MAX_MASKS];
  init_options(&options, blackfilter_exclude);
  options.layout = LAYOUT_SINGLE;

  configure_skip_filters_and_borders(&options);
  set_multi_index_all(&options.no_mask_scan_multi_index);
  set_multi_index_all(&options.no_mask_center_multi_index);

  Rectangle pre_masks[MAX_MASKS];
  Point points[MAX_POINTS];
  int32_t middle_wipe[2] = {0, 0};

  SheetProcessConfig config;
  sheet_process_config_init(&config, &options, pre_masks, 0, points, 0,
                            middle_wipe, blackfilter_exclude, 0);

  BatchJob job;
  init_job(&job);

  SheetProcessState state;
  sheet_process_state_init(&state, &config, &job);
  sheet_process_state_set_decoded(&state, make_test_frame(10, 8), 0);

  assert(process_sheet(&state, &config));

  assert(state.point_count == 1);
  assert(state.points[0].x == state.sheet.frame->width / 2);
  assert(state.points[0].y == state.sheet.frame->height / 2);
  assert(state.mask_max_width == state.sheet.frame->width);
  assert(state.mask_max_height == state.sheet.frame->height);
  assert(state.outside_borderscan_mask_count == 1);
  assert_rectangle(state.outside_borderscan_mask[0], 0, 0,
                   state.sheet.frame->width - 1,
                   state.sheet.frame->height - 1);

  sheet_process_state_cleanup(&state);
}

static void test_layout_double_defaults(void) {
  Options options;
  Rectangle blackfilter_exclude[MAX_MASKS];
  init_options(&options, blackfilter_exclude);
  options.layout = LAYOUT_DOUBLE;

  configure_skip_filters_and_borders(&options);
  set_multi_index_all(&options.no_mask_scan_multi_index);
  set_multi_index_all(&options.no_mask_center_multi_index);

  Rectangle pre_masks[MAX_MASKS];
  Point points[MAX_POINTS];
  int32_t middle_wipe[2] = {0, 0};

  SheetProcessConfig config;
  sheet_process_config_init(&config, &options, pre_masks, 0, points, 0,
                            middle_wipe, blackfilter_exclude, 0);

  BatchJob job;
  init_job(&job);

  SheetProcessState state;
  sheet_process_state_init(&state, &config, &job);
  sheet_process_state_set_decoded(&state, make_test_frame(12, 6), 0);

  assert(process_sheet(&state, &config));

  int width = state.sheet.frame->width;
  int height = state.sheet.frame->height;

  assert(state.point_count == 2);
  assert(state.points[0].x == width / 4);
  assert(state.points[0].y == height / 2);
  assert(state.points[1].x == width - width / 4);
  assert(state.points[1].y == height / 2);
  assert(state.mask_max_width == width / 2);
  assert(state.mask_max_height == height);
  assert(state.outside_borderscan_mask_count == 2);
  assert_rectangle(state.outside_borderscan_mask[0], 0, 0, width / 2,
                   height - 1);
  assert_rectangle(state.outside_borderscan_mask[1], width / 2, 0, width - 1,
                   height - 1);

  sheet_process_state_cleanup(&state);
}

static void test_mask_detection_updates_count(void) {
  Options options;
  Rectangle blackfilter_exclude[MAX_MASKS];
  init_options(&options, blackfilter_exclude);
  options.layout = LAYOUT_SINGLE;

  configure_skip_filters_and_borders(&options);
  options.no_mask_center_multi_index.count = 0;
  options.no_mask_center_multi_index.indexes = NULL;
  options.no_mask_scan_multi_index.count = 0;
  options.no_mask_scan_multi_index.indexes = NULL;

  Rectangle pre_masks[MAX_MASKS];
  Point points[MAX_POINTS];
  int32_t middle_wipe[2] = {0, 0};

  SheetProcessConfig config;
  sheet_process_config_init(&config, &options, pre_masks, 0, points, 0,
                            middle_wipe, blackfilter_exclude, 0);

  BatchJob job;
  init_job(&job);

  SheetProcessState state;
  sheet_process_state_init(&state, &config, &job);
  sheet_process_state_set_decoded(&state, make_test_frame(9, 9), 0);

  assert(process_sheet(&state, &config));

  assert(state.point_count == 1);
  assert(state.mask_count == state.point_count);
  assert(!(state.masks[0].vertex[0].x == -1 &&
           state.masks[0].vertex[0].y == -1 &&
           state.masks[0].vertex[1].x == -1 &&
           state.masks[0].vertex[1].y == -1));

  sheet_process_state_cleanup(&state);
}

int main(void) {
  image_backend_select(UNPAPER_DEVICE_CPU);

  test_layout_single_defaults();
  test_layout_double_defaults();
  test_mask_detection_updates_count();

  return 0;
}
