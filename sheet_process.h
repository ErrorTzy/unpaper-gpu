// SPDX-FileCopyrightText: 2025 The unpaper authors
//
// SPDX-License-Identifier: GPL-2.0-only

#pragma once

#include "constants.h"
#include "imageprocess/image.h"
#include "imageprocess/primitives.h"
#include "lib/batch.h"
#include "lib/options.h"
#include "lib/perf.h"

#include <stdbool.h>
#include <stddef.h>

// Forward declaration for encode queue
struct EncodeQueue;

// Read-only context shared between all workers
// Set once during command-line parsing, never modified during processing
typedef struct {
  const Options *options;

  // Pre-set masks and points from command line
  const Rectangle *pre_masks;
  size_t pre_mask_count;

  const Point *initial_points;
  size_t initial_point_count;

  int32_t middle_wipe[2];

  // Blackfilter exclusions
  const Rectangle *blackfilter_exclude;
  size_t blackfilter_exclude_count;
} SheetProcessConfig;

// Forward declaration
struct AVFrame;

// Per-job state - each worker gets its own instance
typedef struct {
  // Input/output file paths (borrowed from BatchJob)
  char *input_files[BATCH_MAX_FILES_PER_SHEET];
  int input_count;
  char *output_files[BATCH_MAX_FILES_PER_SHEET];
  int output_count;

  // Pre-decoded frames (optional, for decode queue integration)
  // If set, process_sheet uses these instead of calling loadImage
  struct AVFrame *decoded_frames[BATCH_MAX_FILES_PER_SHEET];
  bool use_decoded_frames;

  // GPU-decoded images (optional, for nvJPEG decode integration)
  // If set, process_sheet uses these directly (already GPU-resident)
  Image gpu_decoded_images[BATCH_MAX_FILES_PER_SHEET];
  bool use_gpu_decoded_images;

  // Sheet identification
  int sheet_nr;
  int layout_override; // -1 = use options->layout, otherwise Layout enum value

  // Working images (allocated per-job)
  Image sheet;
  Image page;

  // Per-sheet computed state
  Point points[MAX_POINTS];
  size_t point_count;
  Rectangle masks[MAX_MASKS];
  size_t mask_count;
  Rectangle outside_borderscan_mask[MAX_PAGES];
  size_t outside_borderscan_mask_count;

  // Mutable copy of options fields that may be modified per-sheet
  int mask_max_width;
  int mask_max_height;

  // Size tracking
  RectangleSize input_size;
  RectangleSize previous_size;

  // Performance recording
  PerfRecorder perf;

  // Async encode queue (optional, set via sheet_process_state_set_encode_queue)
  struct EncodeQueue *encode_queue;
  int job_index; // For encode queue submission
} SheetProcessState;

// Initialize shared config (call once after parsing)
void sheet_process_config_init(
    SheetProcessConfig *config, const Options *options,
    const Rectangle *pre_masks, size_t pre_mask_count,
    const Point *initial_points, size_t initial_point_count,
    const int32_t middle_wipe[2], const Rectangle *blackfilter_exclude,
    size_t blackfilter_exclude_count);

// Initialize per-job state (call for each job)
void sheet_process_state_init(SheetProcessState *state,
                              const SheetProcessConfig *config, BatchJob *job);

// Clean up per-job state
void sheet_process_state_cleanup(SheetProcessState *state);

// Set a pre-decoded frame for an input (for decode queue integration)
// frame: Decoded AVFrame (ownership transferred to state)
// input_index: Which input (0 or 1)
void sheet_process_state_set_decoded(SheetProcessState *state,
                                     struct AVFrame *frame, int input_index);

// Set a GPU-decoded image for an input (for nvJPEG decode integration)
// image: Image created from GPU memory (ownership transferred to state)
// input_index: Which input (0 or 1)
void sheet_process_state_set_gpu_decoded_image(SheetProcessState *state,
                                               Image image, int input_index);

// Set encode queue for async encoding (optional)
// If set, process_sheet will submit encoded frames to this queue
// instead of calling saveImage directly
void sheet_process_state_set_encode_queue(SheetProcessState *state,
                                          struct EncodeQueue *encode_queue,
                                          int job_index);

// Process a single sheet
// Returns true on success, false on failure
bool process_sheet(SheetProcessState *state, const SheetProcessConfig *config);
