// SPDX-FileCopyrightText: 2025 The unpaper authors
//
// SPDX-License-Identifier: GPL-2.0-only

#include <assert.h>
#include <stdbool.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include <sys/stat.h>
#include <unistd.h>

#include <libavutil/avutil.h>

#include "src/pipeline/image_pipeline.h"

#include "imageprocess/backend.h"
#include "imageprocess/blit.h"
#include "imageprocess/cuda_mempool.h"
#include "imageprocess/cuda_runtime.h"
#include "imageprocess/cuda_stream_pool.h"
#include "imageprocess/deskew.h"
#include "imageprocess/filters.h"
#include "imageprocess/image.h"
#include "imageprocess/interpolate.h"
#include "imageprocess/masks.h"
#include "imageprocess/nvimgcodec.h"
#include "imageprocess/opencv_bridge.h"
#include "imageprocess/pixel.h"
#include "lib/batch.h"
#include "lib/batch_worker.h"
#include "lib/decode_queue.h"
#include "lib/encode_queue.h"
#include "lib/gpu_monitor.h"
#include "lib/perf.h"
#include "lib/threadpool.h"
#include "parse.h"
#include "sheet_process.h"
#include "unpaper.h"

/**
 * Check if any job in the batch queue has JPEG output files.
 * Used to auto-enable GPU encode for JPEG outputs.
 */
static bool batch_has_jpeg_output(BatchQueue *queue) {
  for (size_t i = 0; i < queue->count; i++) {
    BatchJob *job = batch_queue_get(queue, i);
    for (int j = 0; j < job->output_count; j++) {
      if (cli_is_jpeg_filename(job->output_files[j]))
        return true;
    }
  }
  return false;
}

int image_pipeline_run(int argc, char **argv, OptionsResolved *resolved) {
  Options options = resolved->options;
  size_t pointCount = resolved->point_count;
  Point *points = resolved->points;
  size_t maskCount = resolved->mask_count;
  Rectangle *masks = resolved->masks;
  size_t preMaskCount = resolved->pre_mask_count;
  Rectangle *preMasks = resolved->pre_masks;
  int32_t *middleWipe = resolved->middle_wipe;
  Rectangle *blackfilterExclude = resolved->blackfilter_exclude;
  Rectangle outsideBorderscanMask[MAX_PAGES]; // set by --layout
  size_t outsideBorderscanMaskCount = 0;
  int optind = resolved->optind;

  // Initialize batch queue
  BatchQueue batch_queue;
  batch_queue_init(&batch_queue);
  batch_queue.progress = options.batch_progress;
  // Auto-detect parallelism based on device
  if (options.batch_jobs > 0) {
    batch_queue.parallelism = options.batch_jobs;
  } else if (options.device == UNPAPER_DEVICE_CUDA) {
    // For CUDA: use optimized parallelism for GPU workloads
    batch_queue.parallelism = batch_detect_cuda_parallelism();
  } else {
    batch_queue.parallelism = batch_detect_parallelism();
  }

  // Initialize batch-level performance tracking (used in both batch modes)
  BatchPerfRecorder batch_perf;
  batch_perf_init(&batch_perf, options.perf && options.batch_mode);

  // In batch mode, enumerate all jobs upfront
  if (options.batch_mode) {
    verboseLog(VERBOSE_NORMAL, "Batch mode enabled, enumerating jobs...\n");

    int enum_input_nr = options.start_input;
    int enum_output_nr = options.start_output;
    int enum_optind = optind;

    bool inputWildcard =
        options.multiple_sheets && (strchr(argv[enum_optind], '%') != NULL);

    for (int nr = options.start_sheet;
         (options.end_sheet == -1) || (nr <= options.end_sheet); nr++) {

      // Skip excluded sheets
      if (!isInMultiIndex(nr, options.sheet_multi_index) ||
          isInMultiIndex(nr, options.exclude_multi_index)) {
        continue;
      }

      BatchJob *job = batch_queue_add(&batch_queue);
      if (!job) {
        errOutput("failed to allocate batch job");
      }
      job->sheet_nr = nr;
      job->input_nr = enum_input_nr;
      job->output_nr = enum_output_nr;
      job->input_count = options.input_count;
      job->output_count = options.output_count;

      // Enumerate input files
      for (int i = 0; i < options.input_count; i++) {
        bool ins = isInMultiIndex(enum_input_nr, options.insert_blank);
        bool repl = isInMultiIndex(enum_input_nr, options.replace_blank);
        BatchInput *input = batch_job_input_mut(job, i);

        if (repl) {
          if (input) {
            input->type = BATCH_INPUT_NONE;
            input->path = NULL;
            input->pdf_page_index = -1;
          }
          enum_input_nr++;
        } else if (ins) {
          if (input) {
            input->type = BATCH_INPUT_NONE;
            input->path = NULL;
            input->pdf_page_index = -1;
          }
        } else if (inputWildcard) {
          char buf[PATH_MAX];
          sprintf(buf, argv[enum_optind], enum_input_nr++);
          if (input) {
            input->type = BATCH_INPUT_FILE;
            input->path = strdup(buf);
            input->pdf_page_index = -1;
          }

          // Check if file exists
          struct stat statBuf;
          if (!input || stat(input->path, &statBuf) != 0) {
            if (options.end_sheet == -1) {
              // End of files - remove this job and stop
              batch_job_free(job);
              batch_queue.count--;
              goto batch_enum_done;
            } else {
              errOutput("unable to open file %s.", input ? input->path : "");
            }
          }
        } else if (enum_optind >= argc) {
          if (options.end_sheet == -1) {
            batch_job_free(job);
            batch_queue.count--;
            goto batch_enum_done;
          } else {
            errOutput("not enough input files given.");
          }
        } else {
          if (input) {
            input->type = BATCH_INPUT_FILE;
            input->path = strdup(argv[enum_optind++]);
            input->pdf_page_index = -1;
          }
        }
      }
      if (inputWildcard)
        enum_optind++;

      // Enumerate output files
      bool outputWildcard =
          options.multiple_sheets && (strchr(argv[enum_optind], '%') != NULL);
      for (int i = 0; i < options.output_count; i++) {
        if (outputWildcard) {
          char buf[PATH_MAX];
          sprintf(buf, argv[enum_optind], enum_output_nr++);
          job->output_files[i] = strdup(buf);
        } else if (enum_optind >= argc) {
          errOutput("not enough output files given.");
        } else {
          job->output_files[i] = strdup(argv[enum_optind++]);
        }

        // Check for existing output file
        if (!options.overwrite_output) {
          struct stat statbuf;
          if (stat(job->output_files[i], &statbuf) == 0) {
            errOutput("output file '%s' already present.\n",
                      job->output_files[i]);
          }
        }
      }
      if (outputWildcard)
        enum_optind++;

      // Reset optind for next iteration in wildcard mode
      if (inputWildcard) {
        enum_optind = optind;
      }
    }

  batch_enum_done:
    verboseLog(VERBOSE_NORMAL, "Batch queue: %zu jobs to process\n",
               batch_queue_count(&batch_queue));

    if (batch_queue_count(&batch_queue) == 0) {
      verboseLog(VERBOSE_NORMAL, "No jobs to process.\n");
      batch_queue_free(&batch_queue);
      return 0;
    }

    batch_progress_start(&batch_queue);

    // Start batch-level performance timing
    batch_perf_start(&batch_perf);

    // Batch processing with decode queue (works for any parallelism >= 1)
    // The old sequential path (parallelism == 1) doesn't support GPU pipeline
    // or GPU decode, so we always use the batch processing path.
    if (batch_queue.parallelism >= 1) {
      // Initialize GPU memory pool and stream pool for CUDA batch processing
      // Memory pool eliminates per-image cudaMalloc overhead
      // Stream pool enables concurrent GPU operations across multiple jobs
#ifdef UNPAPER_WITH_CUDA
      bool mempool_active = false;
      bool integralpool_active = false;
      bool streampool_active = false;
      size_t auto_stream_count = 4;  // Default for non-GPU pipeline (PNG)
      size_t auto_buffer_count = 12; // 3x streams for triple-buffering
      if (options.device == UNPAPER_DEVICE_CUDA) {
        // Check available GPU memory before starting batch
        GpuMemoryInfo mem_info = {0};
        if (gpu_monitor_get_memory_info(&mem_info)) {
          verboseLog(VERBOSE_NORMAL,
                     "GPU memory: %.1f MB free of %.1f MB total\n",
                     (double)mem_info.free_bytes / (1024.0 * 1024.0),
                     (double)mem_info.total_bytes / (1024.0 * 1024.0));

          // Auto-tune pool sizes based on available GPU memory
          // Scale: For every 3GB of VRAM, multiply parallelism tier
          // This allows high-end GPUs (RTX 5090 24GB) to use more resources
          size_t vram_gb = mem_info.total_bytes / (1024 * 1024 * 1024);
          size_t tier = vram_gb / 3; // 1 tier per 3GB
          if (tier < 1)
            tier = 1;
          if (tier > 8)
            tier = 8; // Cap at 8x scaling

          // Scale streams: 1 per tier (1..8; capped below)
          auto_stream_count = tier;
          if (auto_stream_count > 32)
            auto_stream_count = 32;

          // Scale buffers: 3x streams for triple-buffering
          // (decode+process+encode) 2x was insufficient - peak usage exceeded
          // pool size causing cudaMalloc fallback
          auto_buffer_count = auto_stream_count * 3;
          if (auto_buffer_count > 96)
            auto_buffer_count = 96;

          verboseLog(VERBOSE_NORMAL,
                     "GPU auto-tune: %zu GB VRAM -> tier %zu -> "
                     "%zu streams, %zu buffers\n",
                     vram_gb, tier, auto_stream_count, auto_buffer_count);
        }

        // Override auto-tuned stream count if --cuda-streams specified
        if (options.cuda_streams > 0) {
          auto_stream_count = (size_t)options.cuda_streams;
          auto_buffer_count = auto_stream_count * 3; // 3x for triple-buffering
          if (auto_buffer_count > 96)
            auto_buffer_count = 96;
          verboseLog(VERBOSE_NORMAL,
                     "GPU manual override: %zu streams, %zu buffers\n",
                     auto_stream_count, auto_buffer_count);
        }

        // Buffer size: 32MB covers A1 images (2500x3500 RGB24 = ~26MB)
        const size_t buffer_size = 32 * 1024 * 1024;
        const size_t total_pool_size = auto_buffer_count * buffer_size;

        // Warn if GPU memory seems low for the batch
        if (mem_info.free_bytes > 0 &&
            mem_info.free_bytes < total_pool_size * 2) {
          fprintf(
              stderr,
              "WARNING: Low GPU memory available (%.1f MB free).\n"
              "         Pool requires %.1f MB. Batch processing may fail.\n"
              "         Consider reducing --jobs or using smaller images.\n",
              (double)mem_info.free_bytes / (1024.0 * 1024.0),
              (double)total_pool_size / (1024.0 * 1024.0));
        }

        if (cuda_mempool_global_init(auto_buffer_count, buffer_size)) {
          mempool_active = true;
          verboseLog(
              VERBOSE_NORMAL,
              "GPU memory pool: %zu buffers x %zu bytes (%.1f MB total)\n",
              auto_buffer_count, buffer_size,
              (double)total_pool_size / (1024.0 * 1024.0));
        } else {
          verboseLog(VERBOSE_NORMAL, "GPU memory pool initialization failed, "
                                     "using direct allocation\n");
        }

        // Integral buffer pool for GPU integral image computation
        // Integral buffers are int32, ~36MB for A1 images (2500x3500x4 aligned
        // to 512) Pool same count as image buffers for matching throughput
        const size_t integral_buffer_size = 36 * 1024 * 1024;
        const size_t total_integral_pool_size =
            auto_buffer_count * integral_buffer_size;
        if (cuda_mempool_integral_global_init(auto_buffer_count,
                                              integral_buffer_size)) {
          integralpool_active = true;
          verboseLog(
              VERBOSE_NORMAL,
              "GPU integral pool: %zu buffers x %zu bytes (%.1f MB total)\n",
              auto_buffer_count, integral_buffer_size,
              (double)total_integral_pool_size / (1024.0 * 1024.0));
        } else {
          verboseLog(VERBOSE_NORMAL, "GPU integral pool initialization failed, "
                                     "using direct allocation\n");
        }

        // Scratch buffer pool for temporary GPU allocations
        // Largest user is blackfilter stack: w*h*8 bytes = ~70MB for A1 images
        // Use 80MB buffers to handle A1-sized images with some margin
        // Need 2x streams since some operations need multiple scratch buffers
        // (e.g., grayfilter needs 2 integrals + mask buffer concurrently)
        const size_t scratch_buffer_size = 80 * 1024 * 1024;
        const size_t scratch_buffer_count = auto_stream_count * 2;
        const size_t total_scratch_pool_size =
            scratch_buffer_count * scratch_buffer_size;
        if (cuda_mempool_scratch_global_init(scratch_buffer_count,
                                             scratch_buffer_size)) {
          verboseLog(
              VERBOSE_NORMAL,
              "GPU scratch pool: %zu buffers x %zu bytes (%.1f MB total)\n",
              scratch_buffer_count, scratch_buffer_size,
              (double)total_scratch_pool_size / (1024.0 * 1024.0));
        } else {
          verboseLog(VERBOSE_NORMAL, "GPU scratch pool initialization failed, "
                                     "using direct allocation\n");
        }

        // Initialize stream pool for concurrent GPU operations
        if (cuda_stream_pool_global_init(auto_stream_count)) {
          streampool_active = true;
          verboseLog(VERBOSE_NORMAL, "GPU stream pool: %zu streams\n",
                     auto_stream_count);
        } else {
          verboseLog(
              VERBOSE_NORMAL,
              "GPU stream pool initialization failed, using default stream\n");
        }

        // Initialize GPU monitor for concurrent job tracking and occupancy
        // stats
        if (gpu_monitor_global_init()) {
          verboseLog(VERBOSE_NORMAL, "GPU occupancy monitoring enabled\n");
          gpu_monitor_global_batch_start();
        }
      }
#endif

      // Set up sheet processing configuration
      SheetProcessConfig config;
      sheet_process_config_init(&config, &options, preMasks, preMaskCount,
                                points, pointCount, middleWipe,
                                blackfilterExclude, 0);

      // Create decode queue for pre-decoded image pipeline
      // Queue depth scales with parallelism and GPU buffer count
      size_t decode_queue_depth = (size_t)batch_queue.parallelism * 2;
      bool use_pinned_memory = false;
      DecodeQueue *decode_queue = NULL;

#ifdef UNPAPER_WITH_CUDA
      use_pinned_memory = (options.device == UNPAPER_DEVICE_CUDA);

      // Per-image decode is used for all GPU workloads (faster, better scaling)
      {
        int num_decode_threads = 1;
        // Scale decode threads with CUDA stream count to keep GPU fed
        if (options.device == UNPAPER_DEVICE_CUDA && auto_stream_count >= 2) {
          long cpu_cores = sysconf(_SC_NPROCESSORS_ONLN);
          if (cpu_cores < 1)
            cpu_cores = 8;
          int stream_based = (int)auto_stream_count;
          int cpu_max = (int)(cpu_cores * 3 / 4);
          num_decode_threads =
              (stream_based < cpu_max) ? stream_based : cpu_max;
          if (num_decode_threads < 2)
            num_decode_threads = 2;
          if (num_decode_threads > 20)
            num_decode_threads = 20;
        }
#else
      {
        int num_decode_threads = 1;
#endif
        decode_queue =
            (num_decode_threads > 1)
                ? decode_queue_create_parallel(
                      decode_queue_depth, use_pinned_memory, num_decode_threads)
                : decode_queue_create(decode_queue_depth, use_pinned_memory);
        if (decode_queue) {
#ifdef UNPAPER_WITH_CUDA
          // Enable GPU decode for JPEG files when using CUDA
          if (options.device == UNPAPER_DEVICE_CUDA) {
            int nvimgcodec_streams =
                num_decode_threads > 1 ? num_decode_threads : 4;
            if (nvimgcodec_init(nvimgcodec_streams)) {
              decode_queue_enable_gpu_decode(decode_queue, true);
              verboseLog(VERBOSE_NORMAL,
                         "nvImageCodec GPU decode: enabled (%d streams)\n",
                         nvimgcodec_streams);
            } else {
              verboseLog(
                  VERBOSE_NORMAL,
                  "nvImageCodec GPU decode: unavailable, using CPU decode\n");
            }
          }
#endif
          verboseLog(VERBOSE_NORMAL, "Decode queue: %zu slots%s%s\n",
                     decode_queue_depth,
                     use_pinned_memory ? " (pinned memory)" : "",
                     num_decode_threads > 1 ? ", parallel decode" : "");
          if (num_decode_threads > 1) {
            verboseLog(VERBOSE_NORMAL, "Decode threads: %d\n",
                       num_decode_threads);
          }
          // Start producer thread
          if (!decode_queue_start_producer(decode_queue, &batch_queue,
                                           &options)) {
            verboseLog(VERBOSE_NORMAL,
                       "Failed to start decode producer, falling "
                       "back to inline decode\n");
            decode_queue_destroy(decode_queue);
            decode_queue = NULL;
          }
        }
      }

      // Create encode queue for async encoding pipeline
      // Queue depth scales with parallelism and GPU buffer count
      size_t encode_queue_depth = (size_t)batch_queue.parallelism * 2;
#ifdef UNPAPER_WITH_CUDA
      // For CUDA with auto-tuned buffers, ensure encode queue can handle
      // throughput
      if (options.device == UNPAPER_DEVICE_CUDA &&
          auto_buffer_count > encode_queue_depth) {
        encode_queue_depth = auto_buffer_count;
      }
#endif
      // Scale encoder threads with parallelism to avoid I/O bottleneck on fast
      // GPUs Also consider available CPU cores for multi-core systems
      int num_encoder_threads = 2;
#ifdef UNPAPER_WITH_CUDA
      if (options.device == UNPAPER_DEVICE_CUDA &&
          batch_queue.parallelism >= 8) {
        // Get CPU core count for scaling
        long cpu_cores = sysconf(_SC_NPROCESSORS_ONLN);
        if (cpu_cores < 1)
          cpu_cores = 8;

        // Scale encoder threads aggressively to saturate I/O
        // Use 1 encoder per stream to avoid output bottleneck
        int parallel_based = batch_queue.parallelism; // 1:1 with workers
        int cpu_max =
            (int)(cpu_cores * 3 / 4); // Use up to 3/4 of CPU cores for encode
        num_encoder_threads =
            (parallel_based < cpu_max) ? parallel_based : cpu_max;

        if (num_encoder_threads < 2)
          num_encoder_threads = 2;
        if (num_encoder_threads > 20)
          num_encoder_threads = 20;
      }
#endif
      EncodeQueue *encode_queue =
          encode_queue_create(encode_queue_depth, num_encoder_threads);
      if (encode_queue) {
        verboseLog(VERBOSE_NORMAL,
                   "Encode queue: %zu slots, %d encoder threads\n",
                   encode_queue_depth, num_encoder_threads);

#ifdef UNPAPER_WITH_CUDA
        // Auto-enable GPU encode for JPEG outputs when using CUDA backend
        if (options.device == UNPAPER_DEVICE_CUDA &&
            batch_has_jpeg_output(&batch_queue)) {
          int jpeg_quality =
              options.jpeg_quality > 0 ? options.jpeg_quality : 85;
          // nvimgcodec is already initialized and handles encoding
          if (nvimgcodec_any_available()) {
            encode_queue_enable_gpu(encode_queue, true, jpeg_quality);
            verboseLog(VERBOSE_NORMAL,
                       "nvImageCodec GPU encode: auto-enabled for JPEG outputs "
                       "(quality %d)\n",
                       jpeg_quality);
          } else {
            verboseLog(VERBOSE_NORMAL,
                       "GPU encode: unavailable, using CPU encode\n");
          }
        }
#endif

        if (!encode_queue_start(encode_queue)) {
          verboseLog(VERBOSE_NORMAL, "Failed to start encoder threads, falling "
                                     "back to inline encode\n");
          encode_queue_destroy(encode_queue);
          encode_queue = NULL;
        }
      }

      // Create thread pool
      ThreadPool *pool = threadpool_create(batch_queue.parallelism);
      if (!pool) {
        if (encode_queue) {
          encode_queue_signal_done(encode_queue);
          encode_queue_wait(encode_queue);
          encode_queue_destroy(encode_queue);
        }
        if (decode_queue) {
          decode_queue_stop_producer(decode_queue);
          decode_queue_destroy(decode_queue);
        }
        errOutput("Failed to create thread pool");
      }

      verboseLog(VERBOSE_NORMAL, "Parallel batch processing with %d workers\n",
                 threadpool_get_num_threads(pool));

      // Set up batch worker context
      BatchWorkerContext worker_ctx;
      batch_worker_init(&worker_ctx, &options, &batch_queue);
      batch_worker_set_config(&worker_ctx, &config);
      batch_worker_set_decode_queue(&worker_ctx, decode_queue);
      batch_worker_set_encode_queue(&worker_ctx, encode_queue);
#ifdef UNPAPER_WITH_CUDA
      // Enable stream pooling if the stream pool was initialized
      batch_worker_enable_stream_pool(&worker_ctx, streampool_active);
#endif

      // Process all jobs in parallel
      int failed = batch_process_parallel(&worker_ctx, pool);

      // Cleanup
      batch_worker_cleanup(&worker_ctx);
      threadpool_destroy(pool);

      // Cleanup decode queue
      if (decode_queue) {
        decode_queue_stop_producer(decode_queue);
        if (options.perf) {
          decode_queue_print_stats(decode_queue);
        }
        decode_queue_destroy(decode_queue);
      }
#ifdef UNPAPER_WITH_CUDA
      // Print nvImageCodec decode statistics
      if (options.perf) {
        nvimgcodec_print_stats();
      }
#endif

      // Cleanup encode queue - wait for all pending encodes to complete
      if (encode_queue) {
        encode_queue_signal_done(encode_queue);
        encode_queue_wait(encode_queue);
        if (options.perf) {
          encode_queue_print_stats(encode_queue);
        }
        encode_queue_destroy(encode_queue);
      }

#ifdef UNPAPER_WITH_CUDA
      // Cleanup nvImageCodec context (handles both decode and encode)
      nvimgcodec_cleanup();
#endif

#ifdef UNPAPER_WITH_CUDA
      // Print GPU pool statistics and cleanup
      if (mempool_active || streampool_active) {
        // Mark end of batch for GPU monitor
        if (gpu_monitor_global_active()) {
          gpu_monitor_global_batch_end();
        }

        if (options.perf) {
          if (mempool_active) {
            cuda_mempool_global_print_stats();
          }
          if (integralpool_active) {
            cuda_mempool_integral_global_print_stats();
          }
          if (streampool_active) {
            cuda_stream_pool_global_print_stats();
          }
          // Print GPU occupancy statistics
          if (gpu_monitor_global_active()) {
            gpu_monitor_global_print_stats();
          }
        }
        // Cleanup in reverse order of initialization
        if (gpu_monitor_global_active()) {
          gpu_monitor_global_cleanup();
        }
        if (streampool_active) {
          cuda_stream_pool_global_cleanup();
        }
        if (integralpool_active) {
          cuda_mempool_integral_global_cleanup();
        }
        // Scratch pool cleanup (safe to call even if not initialized)
        cuda_mempool_scratch_global_cleanup();
        if (mempool_active) {
          cuda_mempool_global_cleanup();
        }
      }
#endif

      // End batch performance timing and print summary
      batch_perf_end(&batch_perf, batch_queue.completed, batch_queue.failed);
      batch_perf_print(&batch_perf,
                       options.device == UNPAPER_DEVICE_CUDA ? "cuda" : "cpu");

      // Print async allocation stats for CUDA debugging
      if (options.device == UNPAPER_DEVICE_CUDA && options.perf) {
        unpaper_cuda_print_async_stats();
      }

      batch_progress_finish(&batch_queue);
      batch_queue_free(&batch_queue);

      return (failed > 0) ? 1 : 0;
    }
  }

  int inputNr = options.start_input;
  int outputNr = options.start_output;

  RectangleSize inputSize = {-1, -1};
  RectangleSize previousSize = {-1, -1};
  Image sheet = EMPTY_IMAGE;
  Image page = EMPTY_IMAGE;
  PerfRecorder perf;

  // Track current job index for batch mode progress
  size_t batch_job_index = 0;

  for (int nr = options.start_sheet;
       (options.end_sheet == -1) || (nr <= options.end_sheet); nr++) {
    char inputFilesBuffer[2][PATH_MAX];
    char outputFilesBuffer[2][PATH_MAX];
    char *inputFileNames[2];
    char *outputFileNames[2];

    // -------------------------------------------------------------------
    // --- begin processing                                            ---
    // -------------------------------------------------------------------

    bool inputWildcard =
        options.multiple_sheets && (strchr(argv[optind], '%') != NULL);
    perf_recorder_init(&perf, options.perf,
                       options.device == UNPAPER_DEVICE_CUDA);

    if (options.perf) {
#ifdef UNPAPER_WITH_OPENCV
      printf("perf backends: device=%s opencv=%s ccl=%s\n",
             options.device == UNPAPER_DEVICE_CUDA ? "cuda" : "cpu",
             unpaper_opencv_enabled() ? "yes" : "no",
             unpaper_opencv_ccl_supported() ? "yes" : "no");
#else
      printf("perf backends: device=%s opencv=n/a ccl=n/a\n",
             options.device == UNPAPER_DEVICE_CUDA ? "cuda" : "cpu");
#endif
    }

    bool outputWildcard = false;

    for (int i = 0; i < options.input_count; i++) {
      bool ins = isInMultiIndex(inputNr, options.insert_blank);
      bool repl = isInMultiIndex(inputNr, options.replace_blank);

      if (repl) {
        inputFileNames[i] = NULL;
        inputNr++; /* replace */
      } else if (ins) {
        inputFileNames[i] = NULL; /* insert */
      } else if (inputWildcard) {
        sprintf(inputFilesBuffer[i], argv[optind], inputNr++);
        inputFileNames[i] = inputFilesBuffer[i];
      } else if (optind >= argc) {
        if (options.end_sheet == -1) {
          options.end_sheet = nr - 1;
          goto sheet_end;
        } else {
          errOutput("not enough input files given.");
        }
      } else {
        inputFileNames[i] = argv[optind++];
      }
      if (inputFileNames[i] == NULL) {
        verboseLog(VERBOSE_DEBUG, "added blank input file\n");
      } else {
        verboseLog(VERBOSE_DEBUG, "added input file %s\n", inputFileNames[i]);
      }

      if (inputFileNames[i] != NULL) {
        struct stat statBuf;
        if (stat(inputFileNames[i], &statBuf) != 0) {
          if (options.end_sheet == -1) {
            options.end_sheet = nr - 1;
            goto sheet_end;
          } else {
            errOutput("unable to open file %s.", inputFileNames[i]);
          }
        }
      }
    }
    if (inputWildcard)
      optind++;

    if (optind >= argc) { // see if any one of the last two optind++ has pushed
                          // it over the array boundary
      errOutput("not enough output files given.");
    }
    outputWildcard =
        options.multiple_sheets && (strchr(argv[optind], '%') != NULL);
    for (int i = 0; i < options.output_count; i++) {
      if (outputWildcard) {
        sprintf(outputFilesBuffer[i], argv[optind], outputNr++);
        outputFileNames[i] = outputFilesBuffer[i];
      } else if (optind >= argc) {
        errOutput("not enough output files given.");
      } else {
        outputFileNames[i] = argv[optind++];
      }
      verboseLog(VERBOSE_DEBUG, "added output file %s\n", outputFileNames[i]);

      if (!options.overwrite_output) {
        struct stat statbuf;
        if (stat(outputFileNames[i], &statbuf) == 0) {
          errOutput("output file '%s' already present.\n", outputFileNames[i]);
        }
      }
    }
    if (outputWildcard)
      optind++;

    // ---------------------------------------------------------------
    // --- process single sheet                                    ---
    // ---------------------------------------------------------------

    if (isInMultiIndex(nr, options.sheet_multi_index) &&
        (!isInMultiIndex(nr, options.exclude_multi_index))) {
      char s1[1023]; // buffers for result of implode()

      // Update batch progress
      if (options.batch_mode &&
          batch_job_index < batch_queue_count(&batch_queue)) {
        batch_progress_update(&batch_queue, (int)batch_job_index,
                              BATCH_JOB_IN_PROGRESS);
      }
      char s2[1023];

      verboseLog(
          VERBOSE_NORMAL,
          "\n-------------------------------------------------------------"
          "------------------\n");

      if (options.multiple_sheets) {
        verboseLog(
            VERBOSE_NORMAL, "Processing sheet #%d: %s -> %s\n", nr,
            implode(s1, (const char **)inputFileNames, options.input_count),
            implode(s2, (const char **)outputFileNames, options.output_count));
      } else {
        verboseLog(
            VERBOSE_NORMAL, "Processing sheet: %s -> %s\n",
            implode(s1, (const char **)inputFileNames, options.input_count),
            implode(s2, (const char **)outputFileNames, options.output_count));
      }

      // load input image(s)
      perf_stage_begin(&perf, PERF_STAGE_DECODE);
      for (int j = 0; j < options.input_count; j++) {
        if (inputFileNames[j] !=
            NULL) { // may be null if --insert-blank or --replace-blank
          verboseLog(VERBOSE_MORE, "loading file %s.\n", inputFileNames[j]);

          loadImage(inputFileNames[j], &page, options.sheet_background,
                    options.abs_black_threshold);
          saveDebug("_loaded_%d.pnm", inputNr - options.input_count + j, page);

          if (options.output_pixel_format == AV_PIX_FMT_NONE &&
              page.frame != NULL) {
            // Try to detect output format from file extension first
            options.output_pixel_format =
                detectPixelFormatFromExtension(outputFileNames[0]);
            // Fall back to image format if extension not recognized
            if (options.output_pixel_format == AV_PIX_FMT_NONE) {
              options.output_pixel_format = page.frame->format;
            }
          }

          if (options.device == UNPAPER_DEVICE_CUDA &&
              isInMultiIndex(nr, options.ignore_multi_index) &&
              options.input_count == 1 && options.output_count == 1 &&
              options.sheet_size.width == -1 &&
              options.sheet_size.height == -1 &&
              options.page_size.width == -1 && options.page_size.height == -1 &&
              options.post_page_size.width == -1 &&
              options.post_page_size.height == -1 &&
              options.stretch_size.width == -1 &&
              options.stretch_size.height == -1 &&
              options.post_stretch_size.width == -1 &&
              options.post_stretch_size.height == -1 &&
              options.pre_rotate == 0 && options.post_rotate == 0 &&
              !options.pre_mirror.horizontal && !options.pre_mirror.vertical &&
              !options.post_mirror.horizontal &&
              !options.post_mirror.vertical &&
              options.pre_shift.horizontal == 0 &&
              options.pre_shift.vertical == 0 &&
              options.post_shift.horizontal == 0 &&
              options.post_shift.vertical == 0 &&
              options.pre_zoom_factor == 1.0 &&
              options.post_zoom_factor == 1.0 && options.pre_wipes.count == 0 &&
              options.wipes.count == 0 && options.post_wipes.count == 0 &&
              options.pre_border.left == 0 && options.pre_border.top == 0 &&
              options.pre_border.right == 0 && options.pre_border.bottom == 0 &&
              options.border.left == 0 && options.border.top == 0 &&
              options.border.right == 0 && options.border.bottom == 0 &&
              options.post_border.left == 0 && options.post_border.top == 0 &&
              options.post_border.right == 0 &&
              options.post_border.bottom == 0 &&
              options.layout == LAYOUT_SINGLE) {
            if (options.write_output) {
              image_ensure_cuda(&page);
              image_mark_cuda_dirty(&page);
              image_ensure_cpu(&page);

              verboseLog(VERBOSE_MORE, "saving file %s.\n", outputFileNames[0]);
              saveImage(outputFileNames[0], page, options.output_pixel_format);
            }
            free_image(&page);
            perf_stage_end(&perf, PERF_STAGE_DECODE);
            goto sheet_end;
          }

          // pre-rotate
          if (options.pre_rotate != 0) {
            verboseLog(VERBOSE_NORMAL, "pre-rotating %hd degrees.\n",
                       options.pre_rotate);

            flip_rotate_90(&page, options.pre_rotate / 90);
          }

          // if sheet-size is not known yet (and not forced by --sheet-size),
          // set now based on size of (first) input image
          RectangleSize inputSheetSize = {
              .width = page.frame->width * options.input_count,
              .height = page.frame->height,
          };
          inputSize = coerce_size(
              inputSize, coerce_size(options.sheet_size, inputSheetSize));
        } else { // inputFiles[j] == NULL
          page = EMPTY_IMAGE;
        }

        // place image into sheet buffer
        // allocate sheet-buffer if not done yet
        if ((sheet.frame == NULL) && (inputSize.width != -1) &&
            (inputSize.height != -1)) {
          sheet = create_image(inputSize, AV_PIX_FMT_RGB24, true,
                               options.sheet_background,
                               options.abs_black_threshold);
        }
        if (page.frame != NULL) {
          saveDebug("_page%d.pnm", inputNr - options.input_count + j, page);
          saveDebug("_before_center_page%d.pnm",
                    inputNr - options.input_count + j, sheet);

          center_image(page, sheet,
                       (Point){(inputSize.width * j / options.input_count), 0},
                       (RectangleSize){(inputSize.width / options.input_count),
                                       inputSize.height});

          saveDebug("_after_center_page%d.pnm",
                    inputNr - options.input_count + j, sheet);
        }
      }

      // the only case that buffer is not yet initialized is if all blank pages
      // have been inserted
      if (sheet.frame == NULL) {
        // last chance: try to get previous (unstretched/not zoomed) sheet size
        inputSize = previousSize;
        verboseLog(VERBOSE_NORMAL,
                   "need to guess sheet size from previous sheet: %dx%d\n",
                   inputSize.width, inputSize.height);

        if ((inputSize.width == -1) || (inputSize.height == -1)) {
          errOutput("sheet size unknown, use at least one input file per "
                    "sheet, or force using --sheet-size.");
        } else {
          sheet = create_image(inputSize, AV_PIX_FMT_RGB24, true,
                               options.sheet_background,
                               options.abs_black_threshold);
        }
      }

      previousSize = inputSize;
      perf_stage_end(&perf, PERF_STAGE_DECODE);

      if (options.device == UNPAPER_DEVICE_CUDA) {
        perf_stage_begin(&perf, PERF_STAGE_UPLOAD);
        image_ensure_cuda(&sheet);
        perf_stage_end(&perf, PERF_STAGE_UPLOAD);
      }

      // pre-mirroring
      if (options.pre_mirror.horizontal || options.pre_mirror.vertical) {
        verboseLog(VERBOSE_NORMAL, "pre-mirroring %s\n",
                   direction_to_string(options.pre_mirror));

        mirror(sheet, options.pre_mirror);
      }

      // pre-shifting
      if (options.pre_shift.horizontal != 0 ||
          options.pre_shift.vertical != 0) {
        verboseLog(VERBOSE_NORMAL, "pre-shifting [%" PRId32 ",%" PRId32 "]\n",
                   options.pre_shift.horizontal, options.pre_shift.vertical);

        shift_image(&sheet, options.pre_shift);
      }

      // pre-masking
      if (preMaskCount > 0) {
        verboseLog(VERBOSE_NORMAL, "pre-masking\n ");

        apply_masks(sheet, preMasks, preMaskCount, options.mask_color);
      }

      // --------------------------------------------------------------
      // --- verbose parameter output,                              ---
      // --------------------------------------------------------------

      // parameters and size are known now

      if (verbose >= VERBOSE_MORE) {
        switch (options.layout) {
        case LAYOUT_NONE:
          printf("layout: none\n");
          break;
        case LAYOUT_SINGLE:
          printf("layout: single\n");
          break;
        case LAYOUT_DOUBLE:
          printf("layout: double\n");
          break;
        default:
          assert(false); // unreachable
        }

        if (options.pre_rotate != 0) {
          printf("pre-rotate: %d\n", options.pre_rotate);
        }
        printf("pre-mirror: %s\n", direction_to_string(options.pre_mirror));
        if (options.pre_shift.horizontal != 0 ||
            options.pre_shift.vertical != 0) {
          printf("pre-shift: [%" PRId32 ",%" PRId32 "]\n",
                 options.pre_shift.horizontal, options.pre_shift.vertical);
        }
        if (options.pre_wipes.count > 0) {
          printf("pre-wipe: ");
          for (size_t i = 0; i < options.pre_wipes.count; i++) {
            print_rectangle(options.pre_wipes.areas[i]);
          }
          printf("\n");
        }
        if (memcmp(&options.pre_border, &BORDER_NULL, sizeof(BORDER_NULL)) !=
            0) {
          printf("pre-border: ");
          print_border(options.pre_border);
          printf("\n");
        }
        if (preMaskCount > 0) {
          printf("pre-masking: ");
          for (int i = 0; i < preMaskCount; i++) {
            print_rectangle(preMasks[i]);
          }
          printf("\n");
        }
        if (options.stretch_size.width != -1 ||
            options.stretch_size.height != -1) {
          printf("stretch to: %" PRId32 "x%" PRId32 "\n",
                 options.stretch_size.width, options.stretch_size.height);
        }
        if (options.post_stretch_size.width != -1 ||
            options.post_stretch_size.height != -1) {
          printf("post-stretch to: %" PRId32 "x%" PRId32 "d\n",
                 options.post_stretch_size.width,
                 options.post_stretch_size.height);
        }
        if (options.pre_zoom_factor != 1.0) {
          printf("zoom: %f\n", options.pre_zoom_factor);
        }
        if (options.post_zoom_factor != 1.0) {
          printf("post-zoom: %f\n", options.post_zoom_factor);
        }
        if (options.no_blackfilter_multi_index.count != -1) {
          printf("blackfilter-scan-direction: %s\n",
                 direction_to_string(
                     options.blackfilter_parameters.scan_direction));
          printf("blackfilter-scan-size: ");
          print_rectangle_size(options.blackfilter_parameters.scan_size);
          printf("\nblackfilter-scan-depth: [%d,%d]\n",
                 options.blackfilter_parameters.scan_depth.horizontal,
                 options.blackfilter_parameters.scan_depth.vertical);
          printf("blackfilter-scan-step: ");
          print_delta(options.blackfilter_parameters.scan_step);
          printf("\nblackfilter-scan-threshold: %d\n",
                 options.blackfilter_parameters.abs_threshold);
          if (options.blackfilter_parameters.exclusions_count > 0) {
            printf("blackfilter-scan-exclude: ");
            for (size_t i = 0;
                 i < options.blackfilter_parameters.exclusions_count; i++) {
              print_rectangle(options.blackfilter_parameters.exclusions[i]);
            }
            printf("\n");
          }
          printf("blackfilter-intensity: %d\n",
                 options.blackfilter_parameters.intensity);
          if (options.no_blackfilter_multi_index.count > 0) {
            printf("blackfilter DISABLED for sheets: ");
            printMultiIndex(options.no_blackfilter_multi_index);
          }
        } else {
          printf("blackfilter DISABLED for all sheets.\n");
        }
        if (options.no_noisefilter_multi_index.count != -1) {
          printf("noisefilter-intensity: %" PRIu64 "\n",
                 options.noisefilter_intensity);
          if (options.no_noisefilter_multi_index.count > 0) {
            printf("noisefilter DISABLED for sheets: ");
            printMultiIndex(options.no_noisefilter_multi_index);
          }
        } else {
          printf("noisefilter DISABLED for all sheets.\n");
        }
        if (options.no_blurfilter_multi_index.count != -1) {
          printf("blurfilter-size: ");
          print_rectangle_size(options.blurfilter_parameters.scan_size);
          printf("\nblurfilter-step: ");
          print_delta(options.blurfilter_parameters.scan_step);
          printf("\nblurfilter-intensity: %f\n",
                 options.blurfilter_parameters.intensity);
          if (options.no_blurfilter_multi_index.count > 0) {
            printf("blurfilter DISABLED for sheets: ");
            printMultiIndex(options.no_blurfilter_multi_index);
          }
        } else {
          printf("blurfilter DISABLED for all sheets.\n");
        }
        if (options.no_grayfilter_multi_index.count != -1) {
          printf("grayfilter-size: ");
          print_rectangle_size(options.grayfilter_parameters.scan_size);
          printf("\ngrayfilter-step: ");
          print_delta(options.grayfilter_parameters.scan_step);
          printf("\ngrayfilter-threshold: %d\n",
                 options.grayfilter_parameters.abs_threshold);
          if (options.no_grayfilter_multi_index.count > 0) {
            printf("grayfilter DISABLED for sheets: ");
            printMultiIndex(options.no_grayfilter_multi_index);
          }
        } else {
          printf("grayfilter DISABLED for all sheets.\n");
        }
        if (options.no_mask_scan_multi_index.count != -1) {
          printf("mask points: ");
          for (int i = 0; i < pointCount; i++) {
            printf("(%d,%d) ", points[i].x, points[i].y);
          }
          printf("\n");
          printf("mask-scan-direction: %s\n",
                 direction_to_string(
                     options.mask_detection_parameters.scan_direction));
          printf("mask-scan-size: ");
          print_rectangle_size(options.mask_detection_parameters.scan_size);
          printf("\nmask-scan-depth: [%d,%d]\n",
                 options.mask_detection_parameters.scan_depth.horizontal,
                 options.mask_detection_parameters.scan_depth.vertical);
          printf("mask-scan-step: ");
          print_delta(options.mask_detection_parameters.scan_step);
          printf("\nmask-scan-threshold: [%f,%f]\n",
                 options.mask_detection_parameters.scan_threshold.horizontal,
                 options.mask_detection_parameters.scan_threshold.vertical);
          printf("mask-scan-minimum: [%d,%d]\n",
                 options.mask_detection_parameters.minimum_width,
                 options.mask_detection_parameters.minimum_height);
          printf("mask-scan-maximum: [%d,%d]\n",
                 options.mask_detection_parameters.maximum_width,
                 options.mask_detection_parameters.maximum_height);
          printf("mask-color: ");
          print_color(options.mask_color);
          printf("\n");
          if (options.no_mask_scan_multi_index.count > 0) {
            printf("mask-scan DISABLED for sheets: ");
            printMultiIndex(options.no_mask_scan_multi_index);
          }
        } else {
          printf("mask-scan DISABLED for all sheets.\n");
        }
        if (options.no_deskew_multi_index.count != -1) {
          printf("deskew-scan-direction: ");
          print_edges(options.deskew_parameters.scan_edges);
          printf("deskew-scan-size: %d\n",
                 options.deskew_parameters.deskewScanSize);
          printf("deskew-scan-depth: %f\n",
                 options.deskew_parameters.deskewScanDepth);
          printf("deskew-scan-range: %f\n",
                 options.deskew_parameters.deskewScanRangeRad);
          printf("deskew-scan-step: %f\n",
                 options.deskew_parameters.deskewScanStepRad);
          printf("deskew-scan-deviation: %f\n",
                 options.deskew_parameters.deskewScanDeviationRad);
          if (options.no_deskew_multi_index.count > 0) {
            printf("deskew-scan DISABLED for sheets: ");
            printMultiIndex(options.no_deskew_multi_index);
          }
        } else {
          printf("deskew-scan DISABLED for all sheets.\n");
        }
        if (options.no_wipe_multi_index.count != -1) {
          if (options.wipes.count > 0) {
            printf("wipe areas: ");
            for (size_t i = 0; i < options.wipes.count; i++) {
              print_rectangle(options.wipes.areas[i]);
            }
            printf("\n");
          }
        } else {
          printf("wipe DISABLED for all sheets.\n");
        }
        if (middleWipe[0] > 0 || middleWipe[1] > 0) {
          printf("middle-wipe (l,r): %d,%d\n", middleWipe[0], middleWipe[1]);
        }
        if (options.no_border_multi_index.count != -1) {
          if (memcmp(&options.border, &BORDER_NULL, sizeof(BORDER_NULL)) != 0) {
            printf("explicit border: ");
            print_border(options.border);
            printf("\n");
          }
        } else {
          printf("border DISABLED for all sheets.\n");
        }
        if (options.no_border_scan_multi_index.count != -1) {
          printf("border-scan-direction: %s\n",
                 direction_to_string(
                     options.border_scan_parameters.scan_direction));
          printf("border-scan-size: ");
          print_rectangle_size(options.border_scan_parameters.scan_size);
          printf("\nborder-scan-step: ");
          print_delta(options.border_scan_parameters.scan_step);
          printf("\nborder-scan-threshold: [%d,%d]\n",
                 options.border_scan_parameters.scan_threshold.horizontal,
                 options.border_scan_parameters.scan_threshold.vertical);
          if (options.no_border_scan_multi_index.count > 0) {
            printf("border-scan DISABLED for sheets: ");
            printMultiIndex(options.no_border_scan_multi_index);
          }
          printf("border-align: ");
          print_edges(options.mask_alignment_parameters.alignment);
          printf("border-margin: [%d,%d]\n",
                 options.mask_alignment_parameters.margin.horizontal,
                 options.mask_alignment_parameters.margin.vertical);
        } else {
          printf("border-scan DISABLED for all sheets.\n");
        }
        if (options.post_wipes.count > 0) {
          printf("post-wipe: ");
          for (size_t i = 0; i < options.post_wipes.count; i++) {
            print_rectangle(options.post_wipes.areas[i]);
          }
          printf("\n");
        }
        if (memcmp(&options.post_border, &BORDER_NULL, sizeof(BORDER_NULL)) !=
            0) {
          printf("post-border: ");
          print_border(options.post_border);
          printf("\n");
        }
        printf("post-mirror: %s\n", direction_to_string(options.post_mirror));
        if (options.post_shift.horizontal != 0 ||
            options.post_shift.vertical != 0) {
          printf("post-shift: [%" PRId32 ",%" PRId32 "]\n",
                 options.post_shift.horizontal, options.post_shift.vertical);
        }
        if (options.post_rotate != 0) {
          printf("post-rotate: %d\n", options.post_rotate);
        }
        // if (options.ignoreMultiIndex.count > 0) {
        //    printf("EXCLUDE sheets: ");
        //    printMultiIndex(options.ignoreMultiIndex);
        //}
        printf("white-threshold: %d\n", options.abs_white_threshold);
        printf("black-threshold: %d\n", options.abs_black_threshold);
        printf("sheet-background: ");
        print_color(options.sheet_background);
        printf("\n");
        printf("input-files per sheet: %d\n", options.input_count);
        printf("output-files per sheet: %d\n", options.output_count);
        if (options.sheet_size.width != -1 || options.sheet_size.height != -1) {
          printf("sheet size forced to: %" PRId32 " x %" PRId32 " pixels\n",
                 options.sheet_size.width, options.sheet_size.height);
        }
        printf("input-file-sequence:  %s\n",
               implode(s1, (const char **)inputFileNames, options.input_count));
        printf(
            "output-file-sequence: %s\n",
            implode(s1, (const char **)outputFileNames, options.output_count));
        if (options.overwrite_output) {
          printf("OVERWRITING EXISTING FILES\n");
        }
        printf("\n");
      }
      verboseLog(
          VERBOSE_NORMAL, "input-file%s for sheet %d: %s\n",
          pluralS(options.input_count), nr,
          implode(s1, (const char **)inputFileNames, options.input_count));
      verboseLog(
          VERBOSE_NORMAL, "output-file%s for sheet %d: %s\n",
          pluralS(options.output_count), nr,
          implode(s1, (const char **)outputFileNames, options.output_count));
      verboseLog(VERBOSE_NORMAL, "sheet size: %dx%d\n", sheet.frame->width,
                 sheet.frame->height);
      verboseLog(VERBOSE_NORMAL, "...\n");

      // -------------------------------------------------------
      // --- process image data                              ---
      // -------------------------------------------------------

      // stretch
      inputSize = coerce_size(options.stretch_size, size_of_image(sheet));

      inputSize.width *= options.pre_zoom_factor;
      inputSize.height *= options.pre_zoom_factor;

      saveDebug("_before-stretch%d.pnm", nr, sheet);
      stretch_and_replace(&sheet, inputSize, options.interpolate_type);
      saveDebug("_after-stretch%d.pnm", nr, sheet);

      // size
      if (options.page_size.width != -1 || options.page_size.height != -1) {
        inputSize = coerce_size(options.page_size, size_of_image(sheet));
        saveDebug("_before-resize%d.pnm", nr, sheet);
        resize_and_replace(&sheet, inputSize, options.interpolate_type);
        saveDebug("_after-resize%d.pnm", nr, sheet);
      }

      // handle sheet layout

      // LAYOUT_SINGLE
      if (options.layout == LAYOUT_SINGLE) {
        // set middle of sheet as single starting point for mask detection
        if (pointCount == 0) { // no manual settings, use auto-values
          points[pointCount++] =
              (Point){sheet.frame->width / 2, sheet.frame->height / 2};
        }
        if (options.mask_detection_parameters.maximum_width == -1) {
          options.mask_detection_parameters.maximum_width = sheet.frame->width;
        }
        if (options.mask_detection_parameters.maximum_height == -1) {
          options.mask_detection_parameters.maximum_height =
              sheet.frame->height;
        }
        // avoid inner half of the sheet to be blackfilter-detectable
        if (options.blackfilter_parameters.exclusions_count == 0) {
          // no manual settings, use auto-values
          RectangleSize sheetSize = size_of_image(sheet);
          options.blackfilter_parameters
              .exclusions[options.blackfilter_parameters.exclusions_count++] =
              rectangle_from_size(
                  (Point){sheetSize.width / 4, sheetSize.height / 4},
                  (RectangleSize){.width = sheetSize.width / 2,
                                  .height = sheetSize.height / 2});
        }
        // set single outside border to start scanning for final border-scan
        if (outsideBorderscanMaskCount ==
            0) { // no manual settings, use auto-values
          outsideBorderscanMask[outsideBorderscanMaskCount++] =
              full_image(sheet);
        }

        // LAYOUT_DOUBLE
      } else if (options.layout == LAYOUT_DOUBLE) {
        // set two middle of left/right side of sheet as starting points for
        // mask detection
        if (pointCount == 0) { // no manual settings, use auto-values
          points[pointCount++] =
              (Point){sheet.frame->width / 4, sheet.frame->height / 2};
          points[pointCount++] =
              (Point){sheet.frame->width - sheet.frame->width / 4,
                      sheet.frame->height / 2};
        }
        if (options.mask_detection_parameters.maximum_width == -1) {
          options.mask_detection_parameters.maximum_width =
              sheet.frame->width / 2;
        }
        if (options.mask_detection_parameters.maximum_height == -1) {
          options.mask_detection_parameters.maximum_height =
              sheet.frame->height;
        }
        if (middleWipe[0] > 0 || middleWipe[1] > 0) { // left, right
          options.wipes.areas[options.wipes.count++] = (Rectangle){{
              {sheet.frame->width / 2 - middleWipe[0], 0},
              {sheet.frame->width / 2 + middleWipe[1], sheet.frame->height - 1},
          }};
        }
        // avoid inner half of each page to be blackfilter-detectable
        if (options.blackfilter_parameters.exclusions_count == 0) {
          // no manual settings, use auto-values
          RectangleSize sheetSize = size_of_image(sheet);
          RectangleSize filterSize = {
              .width = sheetSize.width / 4,
              .height = sheetSize.height / 2,
          };
          Point firstFilterOrigin = {sheetSize.width / 8, sheetSize.height / 4};
          Point secondFilterOrigin =
              shift_point(firstFilterOrigin, (Delta){sheet.frame->width / 2});

          options.blackfilter_parameters
              .exclusions[options.blackfilter_parameters.exclusions_count++] =
              rectangle_from_size(firstFilterOrigin, filterSize);
          options.blackfilter_parameters
              .exclusions[options.blackfilter_parameters.exclusions_count++] =
              rectangle_from_size(secondFilterOrigin, filterSize);
        }
        // set two outside borders to start scanning for final border-scan
        if (outsideBorderscanMaskCount ==
            0) { // no manual settings, use auto-values
          outsideBorderscanMask[outsideBorderscanMaskCount++] =
              (Rectangle){{POINT_ORIGIN,
                           {sheet.frame->width / 2, sheet.frame->height - 1}}};
          outsideBorderscanMask[outsideBorderscanMaskCount++] =
              (Rectangle){{{sheet.frame->width / 2, 0},
                           {sheet.frame->width - 1, sheet.frame->height - 1}}};
        }
      }
      // if maskScanMaximum still unset (no --layout specified), set to full
      // sheet size now
      if (options.mask_detection_parameters.maximum_width == -1) {
        options.mask_detection_parameters.maximum_width = sheet.frame->width;
      }
      if (options.mask_detection_parameters.maximum_height == -1) {
        options.mask_detection_parameters.maximum_height = sheet.frame->height;
      }

      // pre-wipe
      if (!isExcluded(nr, options.no_wipe_multi_index,
                      options.ignore_multi_index)) {
        apply_wipes(sheet, options.pre_wipes, options.mask_color);
      }

      // pre-border
      if (!isExcluded(nr, options.no_border_multi_index,
                      options.ignore_multi_index)) {
        apply_border(sheet, options.pre_border, options.mask_color);
      }

      perf_stage_begin(&perf, PERF_STAGE_FILTERS);
      // black area filter
      if (!isExcluded(nr, options.no_blackfilter_multi_index,
                      options.ignore_multi_index)) {
        saveDebug("_before-blackfilter%d.pnm", nr, sheet);
        blackfilter(sheet, options.blackfilter_parameters);
        saveDebug("_after-blackfilter%d.pnm", nr, sheet);
      } else {
        verboseLog(VERBOSE_MORE, "+ blackfilter DISABLED for sheet %d\n", nr);
      }

      // noise filter
      if (!isExcluded(nr, options.no_noisefilter_multi_index,
                      options.ignore_multi_index)) {
        saveDebug("_before-noisefilter%d.pnm", nr, sheet);
        noisefilter(sheet, options.noisefilter_intensity,
                    options.abs_white_threshold);
        saveDebug("_after-noisefilter%d.pnm", nr, sheet);
      } else {
        verboseLog(VERBOSE_MORE, "+ noisefilter DISABLED for sheet %d\n", nr);
      }

      // blur filter
      if (!isExcluded(nr, options.no_blurfilter_multi_index,
                      options.ignore_multi_index)) {
        saveDebug("_before-blurfilter%d.pnm", nr, sheet);
        blurfilter(sheet, options.blurfilter_parameters,
                   options.abs_white_threshold);
        saveDebug("_after-blurfilter%d.pnm", nr, sheet);
      } else {
        verboseLog(VERBOSE_MORE, "+ blurfilter DISABLED for sheet %d\n", nr);
      }
      perf_stage_end(&perf, PERF_STAGE_FILTERS);

      perf_stage_begin(&perf, PERF_STAGE_MASKS);
      // mask-detection
      if (!isExcluded(nr, options.no_mask_scan_multi_index,
                      options.ignore_multi_index)) {
        detect_masks(sheet, options.mask_detection_parameters, points,
                     pointCount, masks);
      } else {
        verboseLog(VERBOSE_MORE, "+ mask-scan DISABLED for sheet %d\n", nr);
      }

      // permanently apply masks
      if (maskCount > 0) {
        saveDebug("_before-masking%d.pnm", nr, sheet);
        apply_masks(sheet, masks, maskCount, options.mask_color);
        saveDebug("_after-masking%d.pnm", nr, sheet);
      }

      // gray filter
      if (!isExcluded(nr, options.no_grayfilter_multi_index,
                      options.ignore_multi_index)) {
        saveDebug("_before-grayfilter%d.pnm", nr, sheet);
        grayfilter(sheet, options.grayfilter_parameters);
        saveDebug("_after-grayfilter%d.pnm", nr, sheet);
      } else {
        verboseLog(VERBOSE_MORE, "+ grayfilter DISABLED for sheet %d\n", nr);
      }

      // rotation-detection
      perf_stage_end(&perf, PERF_STAGE_MASKS);
      if ((!isExcluded(nr, options.no_deskew_multi_index,
                       options.ignore_multi_index))) {
        perf_stage_begin(&perf, PERF_STAGE_DESKEW);
        saveDebug("_before-deskew%d.pnm", nr, sheet);

        // detect masks again, we may get more precise results now after first
        // masking and grayfilter
        if (!isExcluded(nr, options.no_mask_scan_multi_index,
                        options.ignore_multi_index)) {
          maskCount = detect_masks(sheet, options.mask_detection_parameters,
                                   points, pointCount, masks);
        } else {
          verboseLog(VERBOSE_MORE, "(mask-scan before deskewing disabled)\n");
        }

        // auto-deskew each mask
        for (size_t i = 0; i < maskCount; i++) {
          float rotation =
              detect_rotation(sheet, masks[i], options.deskew_parameters);

          verboseLog(VERBOSE_NORMAL, "rotate (%d,%d): %f\n", points[i].x,
                     points[i].y, rotation);

          if (rotation != 0.0) {
            saveDebug("_before-deskew-detect%d.pnm", nr * maskCount + i, sheet);
            deskew(sheet, masks[i], rotation, options.interpolate_type);
            saveDebug("_after-deskew-detect%d.pnm", nr * maskCount + i, sheet);
          }
        }

        saveDebug("_after-deskew%d.pnm", nr, sheet);
        perf_stage_end(&perf, PERF_STAGE_DESKEW);
      } else {
        verboseLog(VERBOSE_MORE, "+ deskewing DISABLED for sheet %d\n", nr);
      }

      // auto-center masks on either single-page or double-page layout
      perf_stage_begin(&perf, PERF_STAGE_MASKS);
      if (!isExcluded(
              nr, options.no_mask_center_multi_index,
              options.ignore_multi_index)) { // (maskCount==pointCount to
                                             // make sure all masks had
                                             // correctly been detected)
        // perform auto-masking again to get more precise masks after rotation
        if (!isExcluded(nr, options.no_mask_scan_multi_index,
                        options.ignore_multi_index)) {
          maskCount = detect_masks(sheet, options.mask_detection_parameters,
                                   points, pointCount, masks);
        } else {
          verboseLog(VERBOSE_MORE, "(mask-scan before centering disabled)\n");
        }

        saveDebug("_before-centering%d.pnm", nr, sheet);
        // center masks on the sheet, according to their page position
        for (int i = 0; i < maskCount; i++) {
          center_mask(sheet, points[i], masks[i]);
        }
        saveDebug("_after-centering%d.pnm", nr, sheet);
      } else {
        verboseLog(VERBOSE_MORE, "+ auto-centering DISABLED for sheet %d\n",
                   nr);
      }

      // explicit wipe
      if (!isExcluded(nr, options.no_wipe_multi_index,
                      options.ignore_multi_index)) {
        apply_wipes(sheet, options.wipes, options.mask_color);
      } else {
        verboseLog(VERBOSE_MORE, "+ wipe DISABLED for sheet %d\n", nr);
      }

      // explicit border
      if (!isExcluded(nr, options.no_border_multi_index,
                      options.ignore_multi_index)) {
        apply_border(sheet, options.border, options.mask_color);
      } else {
        verboseLog(VERBOSE_MORE, "+ border DISABLED for sheet %d\n", nr);
      }

      // border-detection
      if (!isExcluded(nr, options.no_border_scan_multi_index,
                      options.ignore_multi_index)) {
        Rectangle autoborderMask[outsideBorderscanMaskCount];
        saveDebug("_before-border%d.pnm", nr, sheet);
        for (int i = 0; i < outsideBorderscanMaskCount; i++) {
          autoborderMask[i] = border_to_mask(
              sheet, detect_border(sheet, options.border_scan_parameters,
                                   outsideBorderscanMask[i]));
        }
        apply_masks(sheet, autoborderMask, outsideBorderscanMaskCount,
                    options.mask_color);
        for (int i = 0; i < outsideBorderscanMaskCount; i++) {
          // border-centering
          if (!isExcluded(nr, options.no_border_align_multi_index,
                          options.ignore_multi_index)) {
            align_mask(sheet, autoborderMask[i], outsideBorderscanMask[i],
                       options.mask_alignment_parameters);
          } else {
            verboseLog(VERBOSE_MORE,
                       "+ border-centering DISABLED for sheet %d\n", nr);
          }
        }
        saveDebug("_after-border%d.pnm", nr, sheet);
      } else {
        verboseLog(VERBOSE_MORE, "+ border-scan DISABLED for sheet %d\n", nr);
      }

      // post-wipe
      if (!isExcluded(nr, options.no_wipe_multi_index,
                      options.ignore_multi_index)) {
        apply_wipes(sheet, options.post_wipes, options.mask_color);
      }

      // post-border
      if (!isExcluded(nr, options.no_border_multi_index,
                      options.ignore_multi_index)) {
        apply_border(sheet, options.post_border, options.mask_color);
      }

      perf_stage_end(&perf, PERF_STAGE_MASKS);

      // post-mirroring
      if (options.post_mirror.horizontal || options.post_mirror.vertical) {
        verboseLog(VERBOSE_NORMAL, "post-mirroring %s\n",
                   direction_to_string(options.post_mirror));
        mirror(sheet, options.post_mirror);
      }

      // post-shifting
      if ((options.post_shift.horizontal != 0) ||
          ((options.post_shift.vertical != 0))) {
        verboseLog(VERBOSE_NORMAL, "post-shifting [%" PRId32 ",%" PRId32 "]\n",
                   options.post_shift.horizontal, options.post_shift.vertical);

        shift_image(&sheet, options.post_shift);
      }

      // post-rotating
      if (options.post_rotate != 0) {
        verboseLog(VERBOSE_NORMAL, "post-rotating %d degrees.\n",
                   options.post_rotate);
        flip_rotate_90(&sheet, options.post_rotate / 90);
      }

      // post-stretch
      inputSize = coerce_size(options.post_stretch_size, size_of_image(sheet));

      inputSize.width *= options.post_zoom_factor;
      inputSize.height *= options.post_zoom_factor;

      stretch_and_replace(&sheet, inputSize, options.interpolate_type);

      // post-size
      if (options.post_page_size.width != -1 ||
          options.post_page_size.height != -1) {
        inputSize = coerce_size(options.post_page_size, size_of_image(sheet));
        resize_and_replace(&sheet, inputSize, options.interpolate_type);
      }

      // --- write output file ---

      // write split pages output

      if (options.write_output) {
        verboseLog(VERBOSE_NORMAL, "writing output.\n");
        // write files
        saveDebug("_before-save%d.pnm", nr, sheet);

        if (options.output_pixel_format == AV_PIX_FMT_NONE) {
          // Try to detect output format from file extension first
          options.output_pixel_format =
              detectPixelFormatFromExtension(outputFileNames[0]);
          // Fall back to image format if extension not recognized
          if (options.output_pixel_format == AV_PIX_FMT_NONE) {
            options.output_pixel_format = sheet.frame->format;
          }
        }

        perf_stage_begin(&perf, PERF_STAGE_DOWNLOAD);
        image_ensure_cpu(&sheet);
        perf_stage_end(&perf, PERF_STAGE_DOWNLOAD);

        perf_stage_begin(&perf, PERF_STAGE_ENCODE);
        if (options.output_count == 1) {
          verboseLog(VERBOSE_MORE, "saving file %s.\n", outputFileNames[0]);
          saveImage(outputFileNames[0], sheet, options.output_pixel_format);
        } else {
          for (int j = 0; j < options.output_count; j++) {
            // get pagebuffer
            page = create_compatible_image(
                sheet,
                (RectangleSize){sheet.frame->width / options.output_count,
                                sheet.frame->height},
                false);
            copy_rectangle(
                sheet, page,
                (Rectangle){{{page.frame->width * j, 0},
                             {page.frame->width * j + page.frame->width,
                              page.frame->height}}},
                POINT_ORIGIN);

            verboseLog(VERBOSE_MORE, "saving file %s.\n", outputFileNames[j]);

            saveImage(outputFileNames[j], page, options.output_pixel_format);

            free_image(&page);
          }
        }
        perf_stage_end(&perf, PERF_STAGE_ENCODE);

        free_image(&sheet);
      }
    }

  sheet_end:
    if (options.perf) {
      perf_recorder_print(
          &perf, nr, options.device == UNPAPER_DEVICE_CUDA ? "cuda" : "cpu");
    }

    // Update batch progress - mark as completed
    if (options.batch_mode &&
        batch_job_index < batch_queue_count(&batch_queue)) {
      batch_progress_update(&batch_queue, (int)batch_job_index,
                            BATCH_JOB_COMPLETED);
      batch_job_index++;
    }

    /* if we're not given an input wildcard, and we finished the
     * arguments, we don't want to keep looping.
     */
    if (optind >= argc && !inputWildcard)
      break;
    else if (inputWildcard && outputWildcard)
      optind -= 2;
  }

  // Finish batch progress reporting and cleanup
  if (options.batch_mode) {
    // End batch performance timing for sequential mode and print summary
    batch_perf_end(&batch_perf, batch_queue.completed, batch_queue.failed);
    batch_perf_print(&batch_perf,
                     options.device == UNPAPER_DEVICE_CUDA ? "cuda" : "cpu");
    batch_progress_finish(&batch_queue);

    // Print async allocation stats for CUDA debugging
    if (options.device == UNPAPER_DEVICE_CUDA && options.perf) {
      unpaper_cuda_print_async_stats();
    }
  }
  batch_queue_free(&batch_queue);

  return 0;
}
