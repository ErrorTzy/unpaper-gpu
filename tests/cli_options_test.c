// SPDX-FileCopyrightText: 2025 The unpaper authors
//
// SPDX-License-Identifier: GPL-2.0-only

#include <assert.h>
#include <stdbool.h>
#include <stdint.h>
#include <stdlib.h>

#include "imageprocess/deskew.h"
#include "imageprocess/filters.h"
#include "imageprocess/masks.h"
#include "imageprocess/pixel.h"
#include "lib/options.h"
#include "parse.h"

// Stub out dependencies referenced by lib/options.c but not needed for these
// unit tests.
bool validate_blackfilter_parameters(BlackfilterParameters *params,
                                     RectangleSize scan_size, Delta scan_step,
                                     uint32_t scan_depth_h,
                                     uint32_t scan_depth_v,
                                     Direction scan_direction, float threshold,
                                     int32_t intensity,
                                     size_t exclusions_count,
                                     Rectangle *exclusions) {
  (void)params;
  (void)scan_size;
  (void)scan_step;
  (void)scan_depth_h;
  (void)scan_depth_v;
  (void)scan_direction;
  (void)threshold;
  (void)intensity;
  (void)exclusions_count;
  (void)exclusions;
  return true;
}

bool validate_blurfilter_parameters(BlurfilterParameters *params,
                                    RectangleSize scan_size, Delta scan_step,
                                    float intensity) {
  (void)params;
  (void)scan_size;
  (void)scan_step;
  (void)intensity;
  return true;
}

bool validate_grayfilter_parameters(GrayfilterParameters *params,
                                    RectangleSize scan_size, Delta scan_step,
                                    float threshold) {
  (void)params;
  (void)scan_size;
  (void)scan_step;
  (void)threshold;
  return true;
}

bool validate_deskew_parameters(DeskewParameters *params, float deskewScanRange,
                                float deskewScanStep,
                                float deskewScanDeviation,
                                int deskewScanSize, float deskewScanDepth,
                                Edges deskewScanEdges) {
  (void)params;
  (void)deskewScanRange;
  (void)deskewScanStep;
  (void)deskewScanDeviation;
  (void)deskewScanSize;
  (void)deskewScanDepth;
  (void)deskewScanEdges;
  return true;
}

bool validate_mask_detection_parameters(
    MaskDetectionParameters *params, Direction scan_direction,
    RectangleSize scan_size, const int32_t scan_depth[DIRECTIONS_COUNT],
    Delta scan_step, const float scan_threshold[DIRECTIONS_COUNT],
    const int scan_mininum[DIMENSIONS_COUNT],
    const int scan_maximum[DIMENSIONS_COUNT]) {
  (void)params;
  (void)scan_direction;
  (void)scan_size;
  (void)scan_depth;
  (void)scan_step;
  (void)scan_threshold;
  (void)scan_mininum;
  (void)scan_maximum;
  return true;
}

bool validate_mask_alignment_parameters(MaskAlignmentParameters *params,
                                        Edges alignment, Delta margin) {
  (void)params;
  (void)alignment;
  (void)margin;
  return true;
}

bool validate_border_scan_parameters(
    BorderScanParameters *params, Direction scan_direction,
    RectangleSize scan_size, Delta scan_step,
    const int32_t scan_threshold[DIRECTIONS_COUNT]) {
  (void)params;
  (void)scan_direction;
  (void)scan_size;
  (void)scan_step;
  (void)scan_threshold;
  return true;
}

Pixel pixel_from_value(uint32_t value) {
  (void)value;
  return (Pixel){0, 0, 0};
}

int compare_pixel(Pixel a, Pixel b) {
  (void)a;
  (void)b;
  return 0;
}

static void free_multi_index(struct MultiIndex *idx) {
  free(idx->indexes);
  idx->indexes = NULL;
}

static void test_parse_multi_index(void) {
  struct MultiIndex idx;

  parseMultiIndex("1,3-4,7", &idx);
  assert(idx.count == 4);
  assert(idx.indexes[0] == 1);
  assert(idx.indexes[1] == 3);
  assert(idx.indexes[2] == 4);
  assert(idx.indexes[3] == 7);
  free_multi_index(&idx);

  parseMultiIndex("2-4", &idx);
  assert(idx.count == 3);
  assert(idx.indexes[0] == 2);
  assert(idx.indexes[1] == 3);
  assert(idx.indexes[2] == 4);
  free_multi_index(&idx);

  parseMultiIndex("foo", &idx);
  assert(idx.count == -1);
  free_multi_index(&idx);

  parseMultiIndex(NULL, &idx);
  assert(idx.count == -1);
  free_multi_index(&idx);
}

static void test_parse_layout(void) {
  Layout layout = LAYOUT_SINGLE;

  assert(parse_layout("single", &layout));
  assert(layout == LAYOUT_SINGLE);

  assert(parse_layout("DOUBLE", &layout));
  assert(layout == LAYOUT_DOUBLE);

  assert(parse_layout("none", &layout));
  assert(layout == LAYOUT_NONE);

  assert(!parse_layout("weird", &layout));
}

static void test_validation_errors(void) {
  Delta delta;
  Border border;

  assert(!parse_scan_step("0,5", &delta));
  assert(!parse_scan_step("-1,5", &delta));
  assert(parse_scan_step("1,2", &delta));

  assert(!parse_border("1,2,3", &border));
  assert(!parse_border("1,2,-3,4", &border));
  assert(parse_border("1,2,3,4", &border));
}

int main(void) {
  test_parse_multi_index();
  test_parse_layout();
  test_validation_errors();
  return 0;
}
