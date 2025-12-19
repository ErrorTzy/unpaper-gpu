// SPDX-FileCopyrightText: 2025 The unpaper authors
//
// SPDX-License-Identifier: GPL-2.0-only

#include "src/cli/cli_options.h"

#include <getopt.h>
#include <inttypes.h>
#include <stdbool.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <strings.h>

#include <libavutil/pixfmt.h>

#include "imageprocess/cuda_runtime.h"
#include "lib/logging.h"
#include "lib/physical.h"
#include "parse.h"
#include "version.h"

#define WELCOME                                                                \
  "unpaper " VERSION_STR "\n"                                                  \
  "License GPLv2: GNU GPL version 2.\n"                                        \
  "This is free software: you are free to change and redistribute it.\n"       \
  "There is NO WARRANTY, to the extent permitted by law.\n"

#define USAGE                                                                  \
  WELCOME "\n"                                                                 \
          "Usage: unpaper [options] <input-file(s)> <output-file(s)>\n"        \
          "\n"                                                                 \
          "Options: --device=cpu|cuda (default: auto; cuda if available)\n"   \
          "         --perf (print per-stage timings)\n"                        \
          "         --batch, -B (enable batch processing mode)\n"              \
          "         --jobs=N, -j N (parallel workers, 0=auto, default: 0)\n"   \
          "         --cuda-streams=N (CUDA streams, 0=auto, default: 0)\n"     \
          "         --jpeg-quality=N (JPEG output quality, 1-100)\n"           \
          "         --pdf-quality=fast|high (PDF: fast=JPEG, high=JP2)\n"      \
          "         --pdf-dpi=N (PDF render DPI, 72-1200, default: 300)\n"     \
          "         --progress (show batch progress)\n"                        \
          "         --split (shortcut: double-page -> single-page A4)\n"      \
          "         --skip-split=RANGE (PDF: pages to skip splitting)\n"       \
          "\n"                                                                 \
          "Filenames may contain a formatting placeholder starting with '%%' " \
          "to insert a\n"                                                      \
          "page counter for multi-page processing. E.g.: 'scan%%03d.pbm' to "  \
          "process files\n"                                                    \
          "scan001.pbm, scan002.pbm, scan003.pbm etc.\n"                       \
          "\n"                                                                 \
          "See 'man unpaper' for options details\n"                            \
          "Report bugs at https://github.com/unpaper/unpaper/issues\n"

// We use these for the "val" field in struct option, for getopt_long_only().
// These are for the options that do not have single characters as short
// options.
//
// The values start at 0x7e because this is above all the values for the
// short-option characters (e.g. 0x7e is '~', but there is no '~" short option,
// so we start with that).
enum LONG_OPTION_VALUES {
  OPT_START_SHEET = 0x7e,
  OPT_END_SHEET,
  OPT_START_INPUT,
  OPT_START_OUTPUT,
  OPT_SHEET_BACKGROUND,
  OPT_PRE_ROTATE,
  OPT_POST_ROTATE,
  OPT_POST_MIRROR,
  OPT_PRE_SHIFT,
  OPT_POST_SHIFT,
  OPT_PRE_MASK,
  OPT_POST_SIZE,
  OPT_STRETCH,
  OPT_POST_STRETCH,
  OPT_POST_ZOOM,
  OPT_PRE_WIPE,
  OPT_POST_WIPE,
  OPT_MIDDLE_WIPE,
  OPT_PRE_BORDER,
  OPT_POST_BORDER,
  OPT_NO_BLACK_FILTER,
  OPT_BLACK_FILTER_SCAN_DIRECTION,
  OPT_BLACK_FILTER_SCAN_SIZE,
  OPT_BLACK_FILTER_SCAN_DEPTH,
  OPT_BLACK_FILTER_SCAN_STEP,
  OPT_BLACK_FILTER_SCAN_THRESHOLD,
  OPT_BLACK_FILTER_SCAN_EXCLUDE,
  OPT_BLACK_FILTER_INTENSITY,
  OPT_NO_NOISE_FILTER,
  OPT_NOISE_FILTER_INTENSITY,
  OPT_NO_BLUR_FILTER,
  OPT_BLUR_FILTER_SIZE,
  OPT_BLUR_FILTER_STEP,
  OPT_BLUR_FILTER_INTENSITY,
  OPT_NO_GRAY_FILTER,
  OPT_GRAY_FILTER_SIZE,
  OPT_GRAY_FILTER_STEP,
  OPT_GRAY_FILTER_THRESHOLD,
  OPT_NO_MASK_SCAN,
  OPT_MASK_SCAN_DIRECTION,
  OPT_MASK_SCAN_SIZE,
  OPT_MASK_SCAN_DEPTH,
  OPT_MASK_SCAN_STEP,
  OPT_MASK_SCAN_THRESHOLD,
  OPT_MASK_SCAN_MINIMUM,
  OPT_MASK_SCAN_MAXIMUM,
  OPT_MASK_COLOR,
  OPT_NO_MASK_CENTER,
  OPT_NO_DESKEW,
  OPT_DESKEW_SCAN_DIRECTION,
  OPT_DESKEW_SCAN_SIZE,
  OPT_DESKEW_SCAN_DEPTH,
  OPT_DESKEW_SCAN_RANGE,
  OPT_DESKEW_SCAN_STEP,
  OPT_DESKEW_SCAN_DEVIATION,
  OPT_NO_BORDER_SCAN,
  OPT_BORDER_SCAN_DIRECTION,
  OPT_BORDER_SCAN_SIZE,
  OPT_BORDER_SCAN_STEP,
  OPT_BORDER_SCAN_THRESHOLD,
  OPT_BORDER_ALIGN,
  OPT_BORDER_MARGIN,
  OPT_NO_BORDER_ALIGN,
  OPT_NO_WIPE,
  OPT_NO_BORDER,
  OPT_INPUT_PAGES,
  OPT_OUTPUT_PAGES,
  OPT_INPUT_FILE_SEQUENCE,
  OPT_OUTPUT_FILE_SEQUENCE,
  OPT_INSERT_BLANK,
  OPT_REPLACE_BLANK,
  OPT_NO_MULTI_PAGES,
  OPT_PPI,
  OPT_SPLIT,
  OPT_SKIP_SPLIT,
  OPT_OVERWRITE,
  OPT_VERBOSE_MORE,
  OPT_DEBUG,
  OPT_DEBUG_SAVE,
  OPT_INTERPOLATE,
  OPT_DEVICE,
  OPT_PERF,
  OPT_BATCH,
  OPT_JOBS,
  OPT_PROGRESS,
  OPT_CUDA_STREAMS,
  OPT_JPEG_QUALITY,
  OPT_PDF_QUALITY,
  OPT_PDF_DPI,
};

const char *cli_welcome_message(void) { return WELCOME; }

bool cli_is_jpeg_filename(const char *filename) {
  if (!filename)
    return false;
  size_t len = strlen(filename);
  if (len < 4)
    return false;
  const char *ext = filename + len - 4;
  if (strcasecmp(ext, ".jpg") == 0)
    return true;
  if (len >= 5) {
    ext = filename + len - 5;
    if (strcasecmp(ext, ".jpeg") == 0)
      return true;
  }
  return false;
}

static void cli_options_init(OptionsResolved *out) {
  options_init(&out->options);
  out->device_explicit = false;
  out->optind = 0;
  out->point_count = 0;
  out->mask_count = 0;
  out->pre_mask_count = 0;
  out->middle_wipe[0] = 0;
  out->middle_wipe[1] = 0;
  out->blackfilter_exclude_count = 0;
}

static void cli_options_infer_device(OptionsResolved *out) {
  if (out->device_explicit) {
    return;
  }

#if defined(UNPAPER_WITH_CUDA) && (UNPAPER_WITH_CUDA)
  if (unpaper_cuda_try_init() == UNPAPER_CUDA_INIT_OK) {
    out->options.device = UNPAPER_DEVICE_CUDA;
  }
#endif
}

static void cli_options_infer_batch_mode(int argc, char **argv,
                                         OptionsResolved *out) {
#ifdef UNPAPER_WITH_CUDA
  if (out->options.device != UNPAPER_DEVICE_CUDA || out->options.batch_mode) {
    return;
  }

  int first_output_idx = optind + out->options.input_count;
  for (int i = first_output_idx; i < argc; i++) {
    if (cli_is_jpeg_filename(argv[i])) {
      out->options.batch_mode = true;
      verboseLog(VERBOSE_NORMAL,
                 "Auto-enabled batch mode for GPU JPEG pipeline\n");
      break;
    }
  }
#else
  (void)argc;
  (void)argv;
#endif
}

CliParseResult cli_options_parse(int argc, char **argv, OptionsResolved *out) {
  CliParseResult result = {.status = CLI_PARSE_OK, .exit_code = 0};

  if (!out) {
    errOutput("internal error: options output is NULL");
  }

  memset(out, 0, sizeof(*out));
  cli_options_init(out);

  float whiteThreshold = 0.9;
  float blackThreshold = 0.33;

  Edges deskewScanEdges = {
      .left = true, .top = false, .right = true, .bottom = false};
  int deskewScanSize = 1500;
  float deskewScanDepth = 0.5;
  float deskewScanRange = 5.0;
  float deskewScanStep = 0.1;
  float deskewScanDeviation = 1.0;
  Direction maskScanDirections = DIRECTION_HORIZONTAL;
  RectangleSize maskScanSize = {50, 50};
  int32_t maskScanDepth[DIRECTIONS_COUNT] = {-1, -1};
  Delta maskScanStep = {5, 5};
  float maskScanThreshold[DIRECTIONS_COUNT] = {0.1, 0.1};
  int maskScanMinimum[DIMENSIONS_COUNT] = {100, 100};
  int maskScanMaximum[DIMENSIONS_COUNT] = {-1, -1}; // set default later
  Direction borderScanDirections = DIRECTION_VERTICAL;
  RectangleSize borderScanSize = {5, 5};
  Delta borderScanStep = {5, 5};
  int32_t borderScanThreshold[DIRECTIONS_COUNT] = {5, 5};
  Edges borderAlign = {
      .left = false, .top = false, .right = false, .bottom = false}; // center
  MilsDelta borderAlignMarginPhysical = {0, 0, false};               // center

  int16_t ppi = 300;
  MilsSize sheetSizePhysical = {-1, -1, false};
  MilsDelta preShiftPhysical = {0, 0, false};
  MilsDelta postShiftPhysical = {0, 0, false};
  MilsSize sizePhysical = {-1, -1, false};
  MilsSize postSizePhysical = {-1, -1, false};
  MilsSize stretchSizePhysical = {-1, -1, false};
  MilsSize postStretchSizePhysical = {-1, -1, false};

  Direction blackfilterScanDirections = DIRECTION_BOTH;
  RectangleSize blackfilterScanSize = {20, 20};
  int32_t blackfilterScanDepth[DIRECTIONS_COUNT] = {500, 500};
  Delta blackfilterScanStep = {5, 5};
  float blackfilterScanThreshold = 0.95;
  int blackfilterIntensity = 20;
  RectangleSize blurfilterScanSize = {100, 100};
  Delta blurfilterScanStep = {50, 50};
  float blurfilterIntensity = 0.01;
  RectangleSize grayfilterScanSize = {50, 50};
  Delta grayfilterScanStep = {20, 20};
  float grayfilterThreshold = 0.5;

  int option_index = 0;
  while (true) {
    int c;

    static const struct option long_options[] = {
        {"help", no_argument, NULL, 'h'},
        {"?", no_argument, NULL, 'h'},
        {"version", no_argument, NULL, 'V'},
        {"layout", required_argument, NULL, 'l'},
        {"#", required_argument, NULL, '#'},
        {"sheet", required_argument, NULL, '#'},
        {"start", required_argument, NULL, OPT_START_SHEET},
        {"start-sheet", required_argument, NULL, OPT_START_SHEET},
        {"end", required_argument, NULL, OPT_END_SHEET},
        {"end-sheet", required_argument, NULL, OPT_END_SHEET},
        {"start-input", required_argument, NULL, OPT_START_INPUT},
        {"si", required_argument, NULL, OPT_START_INPUT},
        {"start-output", required_argument, NULL, OPT_START_OUTPUT},
        {"so", required_argument, NULL, OPT_START_OUTPUT},
        {"sheet-size", required_argument, NULL, 'S'},
        {"sheet-background", required_argument, NULL, OPT_SHEET_BACKGROUND},
        {"exclude", optional_argument, NULL, 'x'},
        {"no-processing", required_argument, NULL, 'n'},
        {"pre-rotate", required_argument, NULL, OPT_PRE_ROTATE},
        {"post-rotate", required_argument, NULL, OPT_POST_ROTATE},
        {"pre-mirror", required_argument, NULL, 'M'},
        {"post-mirror", required_argument, NULL, OPT_POST_MIRROR},
        {"pre-shift", required_argument, NULL, OPT_PRE_SHIFT},
        {"post-shift", required_argument, NULL, OPT_POST_SHIFT},
        {"pre-mask", required_argument, NULL, OPT_PRE_MASK},
        {"size", required_argument, NULL, 's'},
        {"post-size", required_argument, NULL, OPT_POST_SIZE},
        {"stretch", required_argument, NULL, OPT_STRETCH},
        {"post-stretch", required_argument, NULL, OPT_POST_STRETCH},
        {"zoom", required_argument, NULL, 'z'},
        {"post-zoom", required_argument, NULL, OPT_POST_ZOOM},
        {"mask-scan-point", required_argument, NULL, 'p'},
        {"mask", required_argument, NULL, 'm'},
        {"wipe", required_argument, NULL, 'W'},
        {"pre-wipe", required_argument, NULL, OPT_PRE_WIPE},
        {"post-wipe", required_argument, NULL, OPT_POST_WIPE},
        {"middle-wipe", required_argument, NULL, OPT_MIDDLE_WIPE},
        {"mw", required_argument, NULL, OPT_MIDDLE_WIPE},
        {"split", no_argument, NULL, OPT_SPLIT},
        {"skip-split", required_argument, NULL, OPT_SKIP_SPLIT},
        {"border", required_argument, NULL, 'B'},
        {"pre-border", required_argument, NULL, OPT_PRE_BORDER},
        {"post-border", required_argument, NULL, OPT_POST_BORDER},
        {"no-blackfilter", optional_argument, NULL, OPT_NO_BLACK_FILTER},
        {"blackfilter-scan-direction", required_argument, NULL,
         OPT_BLACK_FILTER_SCAN_DIRECTION},
        {"bn", required_argument, NULL, OPT_BLACK_FILTER_SCAN_DIRECTION},
        {"blackfilter-scan-size", required_argument, NULL,
         OPT_BLACK_FILTER_SCAN_SIZE},
        {"bs", required_argument, NULL, OPT_BLACK_FILTER_SCAN_SIZE},
        {"blackfilter-scan-depth", required_argument, NULL,
         OPT_BLACK_FILTER_SCAN_DEPTH},
        {"bd", required_argument, NULL, OPT_BLACK_FILTER_SCAN_DEPTH},
        {"blackfilter-scan-step", required_argument, NULL,
         OPT_BLACK_FILTER_SCAN_STEP},
        {"bp", required_argument, NULL, OPT_BLACK_FILTER_SCAN_STEP},
        {"blackfilter-scan-threshold", required_argument, NULL,
         OPT_BLACK_FILTER_SCAN_THRESHOLD},
        {"bt", required_argument, NULL, OPT_BLACK_FILTER_SCAN_THRESHOLD},
        {"blackfilter-scan-exclude", required_argument, NULL,
         OPT_BLACK_FILTER_SCAN_EXCLUDE},
        {"bx", required_argument, NULL, OPT_BLACK_FILTER_SCAN_EXCLUDE},
        {"blackfilter-intensity", required_argument, NULL,
         OPT_BLACK_FILTER_INTENSITY},
        {"bi", required_argument, NULL, OPT_BLACK_FILTER_INTENSITY},
        {"no-noisefilter", optional_argument, NULL, OPT_NO_NOISE_FILTER},
        {"noisefilter-intensity", required_argument, NULL,
         OPT_NOISE_FILTER_INTENSITY},
        {"ni", required_argument, NULL, OPT_NOISE_FILTER_INTENSITY},
        {"no-blurfilter", optional_argument, NULL, OPT_NO_BLUR_FILTER},
        {"blurfilter-size", required_argument, NULL, OPT_BLUR_FILTER_SIZE},
        {"ls", required_argument, NULL, OPT_BLUR_FILTER_SIZE},
        {"blurfilter-step", required_argument, NULL, OPT_BLUR_FILTER_STEP},
        {"lp", required_argument, NULL, OPT_BLUR_FILTER_STEP},
        {"blurfilter-intensity", required_argument, NULL,
         OPT_BLUR_FILTER_INTENSITY},
        {"li", required_argument, NULL, OPT_BLUR_FILTER_INTENSITY},
        {"no-grayfilter", optional_argument, NULL, OPT_NO_GRAY_FILTER},
        {"grayfilter-size", required_argument, NULL, OPT_GRAY_FILTER_SIZE},
        {"gs", required_argument, NULL, OPT_GRAY_FILTER_SIZE},
        {"grayfilter-step", required_argument, NULL, OPT_GRAY_FILTER_STEP},
        {"gp", required_argument, NULL, OPT_GRAY_FILTER_STEP},
        {"grayfilter-threshold", required_argument, NULL,
         OPT_GRAY_FILTER_THRESHOLD},
        {"gt", required_argument, NULL, OPT_GRAY_FILTER_THRESHOLD},
        {"no-mask-scan", optional_argument, NULL, OPT_NO_MASK_SCAN},
        {"mask-scan-direction", required_argument, NULL,
         OPT_MASK_SCAN_DIRECTION},
        {"mn", required_argument, NULL, OPT_MASK_SCAN_DIRECTION},
        {"mask-scan-size", required_argument, NULL, OPT_MASK_SCAN_SIZE},
        {"ms", required_argument, NULL, OPT_MASK_SCAN_SIZE},
        {"mask-scan-depth", required_argument, NULL, OPT_MASK_SCAN_DEPTH},
        {"md", required_argument, NULL, OPT_MASK_SCAN_DEPTH},
        {"mask-scan-step", required_argument, NULL, OPT_MASK_SCAN_STEP},
        {"mp", required_argument, NULL, OPT_MASK_SCAN_STEP},
        {"mask-scan-threshold", required_argument, NULL,
         OPT_MASK_SCAN_THRESHOLD},
        {"mt", required_argument, NULL, OPT_MASK_SCAN_THRESHOLD},
        {"mask-scan-minimum", required_argument, NULL, OPT_MASK_SCAN_MINIMUM},
        {"mm", required_argument, NULL, OPT_MASK_SCAN_MINIMUM},
        {"mask-scan-maximum", required_argument, NULL, OPT_MASK_SCAN_MAXIMUM},
        {"mM", required_argument, NULL, OPT_MASK_SCAN_MAXIMUM},
        {"mask-color", required_argument, NULL, OPT_MASK_COLOR},
        {"mc", required_argument, NULL, OPT_MASK_COLOR},
        {"no-mask-center", optional_argument, NULL, OPT_NO_MASK_CENTER},
        {"no-deskew", optional_argument, NULL, OPT_NO_DESKEW},
        {"deskew-scan-direction", required_argument, NULL,
         OPT_DESKEW_SCAN_DIRECTION},
        {"dn", required_argument, NULL, OPT_DESKEW_SCAN_DIRECTION},
        {"deskew-scan-size", required_argument, NULL, OPT_DESKEW_SCAN_SIZE},
        {"ds", required_argument, NULL, OPT_DESKEW_SCAN_SIZE},
        {"deskew-scan-depth", required_argument, NULL, OPT_DESKEW_SCAN_DEPTH},
        {"dd", required_argument, NULL, OPT_DESKEW_SCAN_DEPTH},
        {"deskew-scan-range", required_argument, NULL, OPT_DESKEW_SCAN_RANGE},
        {"dr", required_argument, NULL, OPT_DESKEW_SCAN_RANGE},
        {"deskew-scan-step", required_argument, NULL, OPT_DESKEW_SCAN_STEP},
        {"dp", required_argument, NULL, OPT_DESKEW_SCAN_STEP},
        {"deskew-scan-deviation", required_argument, NULL,
         OPT_DESKEW_SCAN_DEVIATION},
        {"dv", required_argument, NULL, OPT_DESKEW_SCAN_DEVIATION},
        {"no-border-scan", optional_argument, NULL, OPT_NO_BORDER_SCAN},
        {"border-scan-direction", required_argument, NULL,
         OPT_BORDER_SCAN_DIRECTION},
        {"Bn", required_argument, NULL, OPT_BORDER_SCAN_DIRECTION},
        {"border-scan-size", required_argument, NULL, OPT_BORDER_SCAN_SIZE},
        {"Bs", required_argument, NULL, OPT_BORDER_SCAN_SIZE},
        {"border-scan-step", required_argument, NULL, OPT_BORDER_SCAN_STEP},
        {"Bp", required_argument, NULL, OPT_BORDER_SCAN_STEP},
        {"border-scan-threshold", required_argument, NULL,
         OPT_BORDER_SCAN_THRESHOLD},
        {"Bt", required_argument, NULL, OPT_BORDER_SCAN_THRESHOLD},
        {"border-align", required_argument, NULL, OPT_BORDER_ALIGN},
        {"Ba", required_argument, NULL, OPT_BORDER_ALIGN},
        {"border-margin", required_argument, NULL, OPT_BORDER_MARGIN},
        {"Bm", required_argument, NULL, OPT_BORDER_MARGIN},
        {"no-border-align", optional_argument, NULL, OPT_NO_BORDER_ALIGN},
        {"no-wipe", optional_argument, NULL, OPT_NO_WIPE},
        {"no-border", optional_argument, NULL, OPT_NO_BORDER},
        {"white-threshold", required_argument, NULL, 'w'},
        {"black-threshold", required_argument, NULL, 'b'},
        {"input-pages", required_argument, NULL, OPT_INPUT_PAGES},
        {"ip", required_argument, NULL, OPT_INPUT_PAGES},
        {"output-pages", required_argument, NULL, OPT_OUTPUT_PAGES},
        {"op", required_argument, NULL, OPT_OUTPUT_PAGES},
        {"input-file-sequence", required_argument, NULL,
         OPT_INPUT_FILE_SEQUENCE},
        {"if", required_argument, NULL, OPT_INPUT_FILE_SEQUENCE},
        {"output-file-sequence", required_argument, NULL,
         OPT_OUTPUT_FILE_SEQUENCE},
        {"of", required_argument, NULL, OPT_OUTPUT_FILE_SEQUENCE},
        {"insert-blank", required_argument, NULL, OPT_INSERT_BLANK},
        {"replace-blank", required_argument, NULL, OPT_REPLACE_BLANK},
        {"test-only", no_argument, NULL, 'T'},
        {"no-multi-pages", no_argument, NULL, OPT_NO_MULTI_PAGES},
        {"dpi", required_argument, NULL, OPT_PPI},
        {"ppi", required_argument, NULL, OPT_PPI},
        {"type", required_argument, NULL, 't'},
        {"quiet", no_argument, NULL, 'q'},
        {"overwrite", no_argument, NULL, OPT_OVERWRITE},
        {"verbose", no_argument, NULL, 'v'},
        {"vv", no_argument, NULL, OPT_VERBOSE_MORE},
        {"debug", no_argument, NULL, OPT_DEBUG},
        {"vvv", no_argument, NULL, OPT_DEBUG},
        {"debug-save", no_argument, NULL, OPT_DEBUG_SAVE},
        {"vvvv", no_argument, NULL, OPT_DEBUG_SAVE},
        {"interpolate", required_argument, NULL, OPT_INTERPOLATE},
        {"device", required_argument, NULL, OPT_DEVICE},
        {"perf", no_argument, NULL, OPT_PERF},
        {"batch", no_argument, NULL, OPT_BATCH},
        {"B", no_argument, NULL, OPT_BATCH},
        {"jobs", required_argument, NULL, OPT_JOBS},
        {"j", required_argument, NULL, OPT_JOBS},
        {"progress", no_argument, NULL, OPT_PROGRESS},
        {"cuda-streams", required_argument, NULL, OPT_CUDA_STREAMS},
        {"jpeg-quality", required_argument, NULL, OPT_JPEG_QUALITY},
        {"pdf-quality", required_argument, NULL, OPT_PDF_QUALITY},
        {"pdf-dpi", required_argument, NULL, OPT_PDF_DPI},
        {NULL, no_argument, NULL, 0}};

    c = getopt_long_only(argc, argv, "hVl:S:x::n::M:s:z:p:m:W:B:w:b:Tt:qv",
                         long_options, &option_index);
    if (c == -1)
      break;

    switch (c) {
    case 'h':
    case '?':
      puts(USAGE);
      result.status = CLI_PARSE_EXIT;
      result.exit_code = c == '?' ? 1 : 0;
      return result;

    case 'V':
      puts(VERSION_STR);
      result.status = CLI_PARSE_EXIT;
      result.exit_code = 0;
      return result;

    case 'l':
      if (!parse_layout(optarg, &out->options.layout)) {
        errOutput("unable to parse layout: '%s'", optarg);
      }
      break;

    case '#':
      parseMultiIndex(optarg, &out->options.sheet_multi_index);
      // allow 0 as start sheet, might be overwritten by --start-sheet again
      if (out->options.sheet_multi_index.count > 0 &&
          out->options.start_sheet >
              out->options.sheet_multi_index.indexes[0])
        out->options.start_sheet = out->options.sheet_multi_index.indexes[0];
      break;

    case OPT_START_SHEET:
      sscanf(optarg, "%d", &out->options.start_sheet);
      break;

    case OPT_END_SHEET:
      sscanf(optarg, "%d", &out->options.end_sheet);
      break;

    case OPT_START_INPUT:
      sscanf(optarg, "%d", &out->options.start_input);
      break;

    case OPT_START_OUTPUT:
      sscanf(optarg, "%d", &out->options.start_output);
      break;

    case 'S':
      parse_physical_size(optarg, &sheetSizePhysical);
      break;

    case OPT_SHEET_BACKGROUND:
      if (!parse_color(optarg, &out->options.sheet_background)) {
        errOutput("invalid value for sheet-background: '%s'", optarg);
      }
      break;

    case 'x':
      parseMultiIndex(optarg, &out->options.exclude_multi_index);
      if (out->options.exclude_multi_index.count == -1)
        out->options.exclude_multi_index.count = 0; // 'exclude all' makes no
                                                    // sense
      break;

    case 'n':
      parseMultiIndex(optarg, &out->options.ignore_multi_index);
      break;

    case OPT_PRE_ROTATE:
      sscanf(optarg, "%hd", &out->options.pre_rotate);
      if ((out->options.pre_rotate != 0) &&
          (abs(out->options.pre_rotate) != 90)) {
        fprintf(stderr,
                "cannot set --pre-rotate value other than -90 or 90, "
                "ignoring.\n");
        out->options.pre_rotate = 0;
      }
      break;

    case OPT_POST_ROTATE:
      sscanf(optarg, "%hd", &out->options.post_rotate);
      if ((out->options.post_rotate != 0) &&
          (abs(out->options.post_rotate) != 90)) {
        fprintf(stderr,
                "cannot set --post-rotate value other than -90 or "
                "90, ignoring.\n");
        out->options.post_rotate = 0;
      }
      break;

    case 'M':
      if (!parse_direction(optarg, &out->options.pre_mirror)) {
        errOutput("unable to parse pre-mirror directions: '%s'", optarg);
      };
      break;

    case OPT_POST_MIRROR:
      if (!parse_direction(optarg, &out->options.post_mirror)) {
        errOutput("unable to parse post-mirror directions: '%s'", optarg);
      }
      break;

    case OPT_PRE_SHIFT:
      parse_physical_delta(optarg, &preShiftPhysical);
      break;

    case OPT_POST_SHIFT:
      parse_physical_delta(optarg, &postShiftPhysical);
      break;

    case OPT_PRE_MASK:
      if (out->pre_mask_count < MAX_MASKS) {
        if (parse_rectangle(optarg, &out->pre_masks[out->pre_mask_count])) {
          out->pre_mask_count++;
        }
      } else {
        fprintf(stderr,
                "maximum number of masks (%d) exceeded, ignoring mask %s\n",
                MAX_MASKS, optarg);
      }
      break;

    case 's':
      parse_physical_size(optarg, &sizePhysical);
      break;

    case OPT_POST_SIZE:
      parse_physical_size(optarg, &postSizePhysical);
      break;

    case OPT_STRETCH:
      parse_physical_size(optarg, &stretchSizePhysical);
      break;

    case OPT_POST_STRETCH:
      parse_physical_size(optarg, &postStretchSizePhysical);
      break;

    case 'z':
      sscanf(optarg, "%f", &out->options.pre_zoom_factor);
      break;

    case OPT_POST_ZOOM:
      sscanf(optarg, "%f", &out->options.post_zoom_factor);
      break;

    case 'p':
      if (out->point_count < MAX_POINTS) {
        int x = -1;
        int y = -1;
        sscanf(optarg, "%d,%d", &x, &y);
        out->points[out->point_count++] = (Point){x, y};
      } else {
        fprintf(stderr,
                "maximum number of scan points (%d) exceeded, ignoring scan "
                "point %s\n",
                MAX_POINTS, optarg);
      }
      break;

    case 'm':
      if (out->mask_count < MAX_MASKS) {
        if (parse_rectangle(optarg, &out->masks[out->mask_count])) {
          out->mask_count++;
        }
      } else {
        fprintf(stderr,
                "maximum number of masks (%d) exceeded, ignoring mask %s\n",
                MAX_MASKS, optarg);
      }
      break;

    case 'W':
      parse_wipe("wipe", optarg, &out->options.wipes);
      break;

    case OPT_PRE_WIPE:
      parse_wipe("pre-wipe", optarg, &out->options.pre_wipes);
      break;

    case OPT_POST_WIPE:
      parse_wipe("post-wipe", optarg, &out->options.post_wipes);
      break;

    case OPT_MIDDLE_WIPE:
      if (!parse_symmetric_integers(optarg, &out->middle_wipe[0],
                                    &out->middle_wipe[1])) {
        errOutput("unable to parse middle-wipe: '%s'", optarg);
      }
      break;

    case 'B':
      if (!parse_border(optarg, &out->options.border)) {
        errOutput("unable to parse border: '%s'", optarg);
      }
      break;

    case OPT_PRE_BORDER:
      if (!parse_border(optarg, &out->options.pre_border)) {
        errOutput("unable to parse pre-border: '%s'", optarg);
      }
      break;

    case OPT_POST_BORDER:
      if (!parse_border(optarg, &out->options.post_border)) {
        errOutput("unable to parse post-border: '%s'", optarg);
      }
      break;

    case OPT_NO_BLACK_FILTER:
      parseMultiIndex(optarg, &out->options.no_blackfilter_multi_index);
      break;

    case OPT_BLACK_FILTER_SCAN_DIRECTION:
      if (!parse_direction(optarg, &blackfilterScanDirections)) {
        errOutput("unable to parse blackfilter-scan-direction: '%s'", optarg);
      }
      break;

    case OPT_BLACK_FILTER_SCAN_SIZE:
      if (!parse_rectangle_size(optarg, &blackfilterScanSize)) {
        errOutput("unable to parse blackfilter-scan-size: '%s'", optarg);
      }
      break;

    case OPT_BLACK_FILTER_SCAN_DEPTH:
      if (!parse_symmetric_integers(optarg, &blackfilterScanDepth[0],
                                    &blackfilterScanDepth[1]) ||
          blackfilterScanDepth[0] <= 0 || blackfilterScanDepth[1] <= 0) {
        errOutput("unable to parse blackfilter-scan-depth: '%s'", optarg);
      }
      break;

    case OPT_BLACK_FILTER_SCAN_STEP:
      if (!parse_scan_step(optarg, &blackfilterScanStep)) {
        errOutput("unable to parse blackfilter-scan-step: '%s'", optarg);
      }
      break;

    case OPT_BLACK_FILTER_SCAN_THRESHOLD:
      sscanf(optarg, "%f", &blackfilterScanThreshold);
      break;

    case OPT_BLACK_FILTER_SCAN_EXCLUDE:
      if (out->blackfilter_exclude_count < MAX_MASKS) {
        if (parse_rectangle(
                optarg, &out->blackfilter_exclude[out->blackfilter_exclude_count])) {
          out->blackfilter_exclude_count++;
        }
      } else {
        fprintf(stderr,
                "maximum number of blackfilter exclusion (%d) exceeded, "
                "ignoring mask %s\n",
                MAX_MASKS, optarg);
      }
      break;

    case OPT_BLACK_FILTER_INTENSITY:
      sscanf(optarg, "%d", &blackfilterIntensity);
      break;

    case OPT_NO_NOISE_FILTER:
      parseMultiIndex(optarg, &out->options.no_noisefilter_multi_index);
      break;

    case OPT_NOISE_FILTER_INTENSITY:
      sscanf(optarg, "%" SCNu64, &out->options.noisefilter_intensity);
      break;

    case OPT_NO_BLUR_FILTER:
      parseMultiIndex(optarg, &out->options.no_blurfilter_multi_index);
      break;

    case OPT_BLUR_FILTER_SIZE:
      if (!parse_rectangle_size(optarg, &blurfilterScanSize)) {
        errOutput("unable to parse blurfilter-scan-size: '%s'", optarg);
      }
      break;

    case OPT_BLUR_FILTER_STEP:
      if (!parse_scan_step(optarg, &blurfilterScanStep)) {
        errOutput("unable to parse blurfilter-scan-step: '%s'", optarg);
      }
      break;

    case OPT_BLUR_FILTER_INTENSITY:
      sscanf(optarg, "%f", &blurfilterIntensity);
      break;

    case OPT_NO_GRAY_FILTER:
      parseMultiIndex(optarg, &out->options.no_grayfilter_multi_index);
      break;

    case OPT_GRAY_FILTER_SIZE:
      if (!parse_rectangle_size(optarg, &grayfilterScanSize)) {
        errOutput("unable to parse grayfilter-scan-size: '%s'", optarg);
      }
      break;

    case OPT_GRAY_FILTER_STEP:
      if (!parse_scan_step(optarg, &grayfilterScanStep)) {
        errOutput("unable to parse grayfilter-scan-step: '%s'", optarg);
      }
      break;

    case OPT_GRAY_FILTER_THRESHOLD:
      sscanf(optarg, "%f", &grayfilterThreshold);
      break;

    case OPT_NO_MASK_SCAN:
      parseMultiIndex(optarg, &out->options.no_mask_scan_multi_index);
      break;

    case OPT_MASK_SCAN_DIRECTION:
      if (!parse_direction(optarg, &maskScanDirections)) {
        errOutput("unable to parse mask-scan-direction: '%s'", optarg);
      }
      break;

    case OPT_MASK_SCAN_SIZE:
      if (!parse_rectangle_size(optarg, &maskScanSize)) {
        errOutput("unable to parse mask-scan-size: '%s'", optarg);
      }
      break;

    case OPT_MASK_SCAN_DEPTH:
      if (!parse_symmetric_integers(optarg, &maskScanDepth[0],
                                    &maskScanDepth[1]) ||
          maskScanDepth[0] <= 0 || maskScanDepth[1] <= 0) {
        errOutput("unable to parse mask-scan-depth: '%s'", optarg);
      }
      break;

    case OPT_MASK_SCAN_STEP:
      if (!parse_scan_step(optarg, &maskScanStep)) {
        errOutput("unable to parse mask-scan-step");
      }
      break;

    case OPT_MASK_SCAN_THRESHOLD:
      if (!parse_symmetric_floats(optarg, &maskScanThreshold[0],
                                  &maskScanThreshold[1]) ||
          maskScanThreshold[0] <= 0 || maskScanThreshold[1] <= 0) {
        errOutput("unable to parse mask-scan-threshold: '%s'", optarg);
      }
      break;

    case OPT_MASK_SCAN_MINIMUM:
      sscanf(optarg, "%d,%d", &maskScanMinimum[WIDTH],
             &maskScanMinimum[HEIGHT]);
      break;

    case OPT_MASK_SCAN_MAXIMUM:
      sscanf(optarg, "%d,%d", &maskScanMaximum[WIDTH],
             &maskScanMaximum[HEIGHT]);
      break;

    case OPT_MASK_COLOR:
      if (!parse_color(optarg, &out->options.mask_color)) {
        errOutput("invalid value for mask-color: '%s'", optarg);
      }
      break;

    case OPT_NO_MASK_CENTER:
      parseMultiIndex(optarg, &out->options.no_mask_center_multi_index);
      break;

    case OPT_NO_DESKEW:
      parseMultiIndex(optarg, &out->options.no_deskew_multi_index);
      break;

    case OPT_DESKEW_SCAN_DIRECTION:
      if (!parse_edges(optarg, &deskewScanEdges)) {
        errOutput("uanble to parse deskew-scan-direction: '%s'", optarg);
      }
      break;

    case OPT_DESKEW_SCAN_SIZE:
      sscanf(optarg, "%d", &deskewScanSize);
      break;

    case OPT_DESKEW_SCAN_DEPTH:
      sscanf(optarg, "%f", &deskewScanDepth);
      break;

    case OPT_DESKEW_SCAN_RANGE:
      sscanf(optarg, "%f", &deskewScanRange);
      break;

    case OPT_DESKEW_SCAN_STEP:
      sscanf(optarg, "%f", &deskewScanStep);
      break;

    case OPT_DESKEW_SCAN_DEVIATION:
      sscanf(optarg, "%f", &deskewScanDeviation);
      break;

    case OPT_NO_BORDER_SCAN:
      parseMultiIndex(optarg, &out->options.no_border_scan_multi_index);
      break;

    case OPT_BORDER_SCAN_DIRECTION:
      if (!parse_direction(optarg, &borderScanDirections)) {
        errOutput("unable to parse border-scan-direction: '%s'", optarg);
      }
      break;

    case OPT_BORDER_SCAN_SIZE:
      if (!parse_rectangle_size(optarg, &borderScanSize)) {
        errOutput("unable to parse border-scan-size: '%s'", optarg);
      }
      break;

    case OPT_BORDER_SCAN_STEP:
      if (!parse_scan_step(optarg, &borderScanStep)) {
        errOutput("unable to parse border-scan-step: '%s'", optarg);
      }
      break;

    case OPT_BORDER_SCAN_THRESHOLD:
      if (!parse_symmetric_integers(optarg, &borderScanThreshold[0],
                                    &borderScanThreshold[1]) ||
          borderScanThreshold[0] <= 0 || borderScanThreshold <= 0) {
        errOutput("unable to parse border-scan-threshold: '%s'", optarg);
      }
      break;

    case OPT_BORDER_ALIGN:
      if (!parse_edges(optarg, &borderAlign)) {
        errOutput("unable to parse border-align: '%s'", optarg);
      }
      break;

    case OPT_BORDER_MARGIN:
      parse_physical_delta(optarg, &borderAlignMarginPhysical);
      break;

    case OPT_NO_BORDER_ALIGN:
      parseMultiIndex(optarg, &out->options.no_border_align_multi_index);
      break;

    case OPT_NO_WIPE:
      parseMultiIndex(optarg, &out->options.no_wipe_multi_index);
      break;

    case OPT_NO_BORDER:
      parseMultiIndex(optarg, &out->options.no_border_multi_index);
      break;

    case 'w':
      sscanf(optarg, "%f", &whiteThreshold);
      break;

    case 'b':
      sscanf(optarg, "%f", &blackThreshold);
      break;

    case OPT_INPUT_PAGES:
      sscanf(optarg, "%d", &out->options.input_count);
      if (!(out->options.input_count >= 1 && out->options.input_count <= 2)) {
        fprintf(stderr,
                "cannot set --input-pages value other than 1 or 2, "
                "ignoring.\n");
        out->options.input_count = 1;
      }

      break;

    case OPT_OUTPUT_PAGES:
      sscanf(optarg, "%d", &out->options.output_count);
      if (!(out->options.output_count >= 1 && out->options.output_count <= 2)) {
        fprintf(stderr,
                "cannot set --output-pages value other than 1 or 2, "
                "ignoring.\n");
        out->options.output_count = 1;
      }

      break;

    case OPT_INPUT_FILE_SEQUENCE:
    case OPT_OUTPUT_FILE_SEQUENCE:
      errOutput(
          "--input-file-sequence and --output-file-sequence are deprecated "
          "and "
          "unimplemented.\n"
          "Please pass input output pairs as arguments to unpaper instead.");
      break;

    case OPT_INSERT_BLANK:
      parseMultiIndex(optarg, &out->options.insert_blank);
      break;

    case OPT_REPLACE_BLANK:
      parseMultiIndex(optarg, &out->options.replace_blank);
      break;

    case 'T':
      out->options.write_output = false;
      break;

    case OPT_NO_MULTI_PAGES:
      out->options.multiple_sheets = false;
      break;

    case OPT_PPI:
      sscanf(optarg, "%hd", &ppi);
      break;

    case OPT_SPLIT:
      out->options.layout = LAYOUT_DOUBLE;
      out->options.output_count = 2;
      if (!parse_physical_size("a3-landscape", &postSizePhysical)) {
        errOutput("unable to apply --split shortcut (invalid size)");
      }
      break;

    case OPT_SKIP_SPLIT:
      parseMultiIndex(optarg, &out->options.skip_split);
      break;

    case 't':
      if (strcmp(optarg, "pbm") == 0) {

        out->options.output_pixel_format = AV_PIX_FMT_MONOWHITE;
      } else if (strcmp(optarg, "pgm") == 0) {
        out->options.output_pixel_format = AV_PIX_FMT_GRAY8;
      } else if (strcmp(optarg, "ppm") == 0) {
        out->options.output_pixel_format = AV_PIX_FMT_RGB24;
      }
      break;

    case 'q':
      verbose = VERBOSE_QUIET;
      break;

    case OPT_OVERWRITE:
      out->options.overwrite_output = true;
      break;

    case 'v':
      verbose = VERBOSE_NORMAL;
      break;

    case OPT_VERBOSE_MORE:
      verbose = VERBOSE_MORE;
      break;

    case OPT_DEBUG:
      verbose = VERBOSE_DEBUG;
      break;

    case OPT_DEBUG_SAVE:
      verbose = VERBOSE_DEBUG_SAVE;
      break;

    case OPT_INTERPOLATE:
      if (!parse_interpolate(optarg, &out->options.interpolate_type)) {
        errOutput("unable to parse interpolate: '%s'", optarg);
      }
      break;

    case OPT_DEVICE:
      out->device_explicit = true;
      if (strcmp(optarg, "cpu") == 0) {
        out->options.device = UNPAPER_DEVICE_CPU;
      } else if (strcmp(optarg, "cuda") == 0) {
#if defined(UNPAPER_WITH_CUDA) && (UNPAPER_WITH_CUDA)
        out->options.device = UNPAPER_DEVICE_CUDA;
#else
        errOutput("CUDA backend requested, but this build has no CUDA "
                  "support.");
#endif
      } else {
        errOutput("invalid value for --device: '%s' (expected cpu or cuda)",
                  optarg);
      }
      break;

    case OPT_PERF:
      out->options.perf = true;
      break;

    case OPT_BATCH:
      out->options.batch_mode = true;
      break;

    case OPT_JOBS:
      if (sscanf(optarg, "%d", &out->options.batch_jobs) != 1 ||
          out->options.batch_jobs < 0) {
        errOutput("invalid value for --jobs: '%s'", optarg);
      }
      break;

    case OPT_PROGRESS:
      out->options.batch_progress = true;
      break;

    case OPT_CUDA_STREAMS:
      if (sscanf(optarg, "%d", &out->options.cuda_streams) != 1 ||
          out->options.cuda_streams < 0) {
        errOutput("invalid value for --cuda-streams: '%s'", optarg);
      }
      break;

    case OPT_JPEG_QUALITY:
      if (sscanf(optarg, "%d", &out->options.jpeg_quality) != 1 ||
          out->options.jpeg_quality < 1 || out->options.jpeg_quality > 100) {
        errOutput("invalid value for --jpeg-quality: '%s' (valid: 1-100)",
                  optarg);
      }
      break;

    case OPT_PDF_QUALITY:
      if (strcasecmp(optarg, "fast") == 0) {
        out->options.pdf_quality_mode = PDF_QUALITY_FAST;
      } else if (strcasecmp(optarg, "high") == 0) {
        out->options.pdf_quality_mode = PDF_QUALITY_HIGH;
      } else {
        errOutput("invalid value for --pdf-quality: '%s' (valid: fast, high)",
                  optarg);
      }
      break;

    case OPT_PDF_DPI:
      if (sscanf(optarg, "%d", &out->options.pdf_render_dpi) != 1 ||
          out->options.pdf_render_dpi < 72 || out->options.pdf_render_dpi > 1200) {
        errOutput("invalid value for --pdf-dpi: '%s' (valid: 72-1200)",
                  optarg);
      }
      break;
    }
  }

  // Expand any physical size to their pixel equivalents.
  out->options.pre_shift = mils_delta_to_pixels(preShiftPhysical, ppi);
  out->options.post_shift = mils_delta_to_pixels(postShiftPhysical, ppi);

  out->options.sheet_size = mils_size_to_pixels(sheetSizePhysical, ppi);
  out->options.page_size = mils_size_to_pixels(sizePhysical, ppi);
  out->options.post_page_size = mils_size_to_pixels(postSizePhysical, ppi);
  out->options.stretch_size = mils_size_to_pixels(stretchSizePhysical, ppi);
  out->options.post_stretch_size =
      mils_size_to_pixels(postStretchSizePhysical, ppi);

  // Calculate the constant absolute values based on the relative parameters.
  out->options.abs_black_threshold = WHITE * (1.0 - blackThreshold);
  out->options.abs_white_threshold = WHITE * (whiteThreshold);

  if (!validate_deskew_parameters(&out->options.deskew_parameters,
                                  deskewScanRange, deskewScanStep,
                                  deskewScanDeviation, deskewScanSize,
                                  deskewScanDepth, deskewScanEdges)) {
    errOutput("deskew parameters are not valid.");
  }
  if (!validate_mask_detection_parameters(
          &out->options.mask_detection_parameters, maskScanDirections,
          maskScanSize, maskScanDepth, maskScanStep, maskScanThreshold,
          maskScanMinimum, maskScanMaximum)) {
    errOutput("mask detection parameters are not valid.");
  }
  if (!validate_mask_alignment_parameters(
          &out->options.mask_alignment_parameters, borderAlign,
          mils_delta_to_pixels(borderAlignMarginPhysical, ppi))) {
    errOutput("mask alignment parameters are not valid.");
  };
  if (!validate_border_scan_parameters(&out->options.border_scan_parameters,
                                       borderScanDirections, borderScanSize,
                                       borderScanStep, borderScanThreshold)) {
    errOutput("border scan parameters are not valid.");
  };
  if (!validate_grayfilter_parameters(&out->options.grayfilter_parameters,
                                      grayfilterScanSize, grayfilterScanStep,
                                      grayfilterThreshold)) {
    errOutput("grayfilter parameters are not valid.");
  }
  if (!validate_blackfilter_parameters(
          &out->options.blackfilter_parameters, blackfilterScanSize,
          blackfilterScanStep, blackfilterScanDepth[HORIZONTAL],
          blackfilterScanDepth[VERTICAL], blackfilterScanDirections,
          blackfilterScanThreshold, blackfilterIntensity,
          out->blackfilter_exclude_count, out->blackfilter_exclude)) {
    errOutput("blackfilter parameters are not valid.");
  }
  if (!validate_blurfilter_parameters(&out->options.blurfilter_parameters,
                                      blurfilterScanSize, blurfilterScanStep,
                                      blurfilterIntensity)) {
    errOutput("blurfilter parameters are not valid.");
  }

  if (out->options.start_input == -1)
    out->options.start_input =
        (out->options.start_sheet - 1) * out->options.input_count + 1;
  if (out->options.start_output == -1)
    out->options.start_output =
        (out->options.start_sheet - 1) * out->options.output_count + 1;

  if (!out->options.multiple_sheets && out->options.end_sheet == -1)
    out->options.end_sheet = out->options.start_sheet;

  out->optind = optind;

  cli_options_infer_device(out);
  cli_options_infer_batch_mode(argc, argv, out);

  return result;
}
