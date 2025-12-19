// Copyright © 2005-2007 Jens Gulden
// Copyright © 2011-2011 Diego Elio Pettenò
// SPDX-FileCopyrightText: 2005 The unpaper authors
//
// SPDX-License-Identifier: GPL-2.0-only

/* --- The main program  -------------------------------------------------- */

#include <stdbool.h>

#include "src/cli/cli_options.h"
#include "src/pipeline/image_pipeline.h"
#include "src/pipeline/pdf_pipeline.h"

#include "imageprocess/backend.h"
#include "lib/logging.h"

/****************************************************************************
 * MAIN()                                                                   *
 ****************************************************************************/

/**
 * The main program.
 */
int main(int argc, char *argv[]) {
  CliParseResult parse_result;
  OptionsResolved resolved;
  parse_result = cli_options_parse(argc, argv, &resolved);
  if (parse_result.status == CLI_PARSE_EXIT) {
    return parse_result.exit_code;
  }

  int optind = resolved.optind;

  /* make sure we have at least two arguments after the options, as
     that's the minimum amount of parameters we need (one input and
     one output, or a wildcard of inputs and a wildcard of
     outputs.
  */
  if (optind + 2 > argc)
    errOutput("no input or output files given.\n");

#if defined(UNPAPER_WITH_PDF) && (UNPAPER_WITH_PDF)
  const char *input_file = argv[optind];
  const char *output_file = (optind + 1 < argc) ? argv[optind + 1] : NULL;
  bool pdf_mode_requested = pdf_pipeline_requested(input_file, output_file);

  if (resolved.options.skip_split.count != 0 && !pdf_mode_requested) {
    errOutput("--skip-split requires PDF input and output files.");
  }
#else
  if (resolved.options.skip_split.count != 0) {
    errOutput("--skip-split requires PDF input and output files.");
  }
#endif

  verboseLog(VERBOSE_NORMAL, "%s", cli_welcome_message()); // welcome message

  image_backend_select(resolved.options.device);

#if defined(UNPAPER_WITH_PDF) && (UNPAPER_WITH_PDF)
  // Check for PDF input/output - use dedicated PDF pipeline
  if (pdf_mode_requested) {
    return pdf_pipeline_run(argc, argv, optind, &resolved);
  }
#endif

  return image_pipeline_run(argc, argv, &resolved);
}
