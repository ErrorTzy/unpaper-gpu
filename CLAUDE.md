# Repository Guidelines

## Project Structure & Module Organization

```
unpaper/
├── unpaper.c          # Main CLI entry point and processing loop
├── parse.c/.h         # Command-line option parsing
├── file.c             # Image file I/O (FFmpeg-based decode/encode, direct PNM writer)
├── imageprocess/      # Image processing backends and algorithms
│   ├── backend.c/.h       # Backend vtable and dispatch (CPU/CUDA selection)
│   ├── backend_cuda.c     # CUDA backend implementation
│   ├── image.c/.h         # Image struct and CPU memory management
│   ├── image_cuda.c       # GPU residency, dirty flags, upload/download helpers
│   ├── cuda_runtime.c/.h  # CUDA runtime abstraction (streams, memory, kernel launch)
│   ├── cuda_kernels.cu    # Custom CUDA kernels (mono formats, blackfilter, deskew)
│   ├── opencv_bridge.cpp/.h  # OpenCV CUDA integration (CCL, filters)
│   ├── opencv_ops.cpp/.h     # OpenCV CUDA primitives (wipe, copy, resize, rotate)
│   ├── blit.c             # Rectangle operations (wipe, copy, center)
│   ├── deskew.c           # Rotation detection and correction
│   ├── filters.c          # blackfilter, blurfilter, grayfilter, noisefilter
│   ├── masks.c            # Mask detection, alignment, border detection
│   ├── interpolate.c/.h   # Interpolation types (NN, linear, cubic)
│   └── primitives.h       # Core types (Point, Rectangle, Pixel, etc.)
├── lib/               # Shared utilities
│   ├── batch.c/.h         # Batch job queue and progress reporting
│   ├── logging.c/.h       # Verbose output and error handling
│   ├── options.c/.h       # Options struct definition and parsing
│   └── physical.c/.h      # Physical dimension helpers (DPI, mm)
├── tests/             # Test suite
│   ├── unpaper_tests.py   # Pytest golden image tests
│   ├── source_images/     # Test input images
│   └── golden_images/     # Expected outputs
├── tools/             # Benchmarking tools
│   ├── bench_a1.py        # Single-page benchmark
│   └── bench_double.py    # Double-page benchmark
└── doc/               # Documentation
    ├── unpaper.1.rst      # Man page source (Sphinx)
    └── CUDA_BACKEND_HISTORY.md  # Completed CUDA implementation history (PR1-18)
```

## Architecture Overview

### Backend System

unpaper uses a backend vtable (`ImageBackend` in `backend.h`) to dispatch image operations:

```c
typedef struct {
  const char *name;
  void (*wipe_rectangle)(Image, Rectangle, Pixel);
  void (*copy_rectangle)(Image, Image, Rectangle, Point);
  // ... 20+ operations for transforms, filters, masks, deskew
} ImageBackend;
```

- **CPU backend** (`backend.c`): Default, uses FFmpeg `AVFrame` for all operations
- **CUDA backend** (`backend_cuda.c`): GPU-accelerated, requires OpenCV with CUDA modules

Selection via `--device=cpu|cuda` (default: `cpu`).

### Image Memory Model

The `Image` struct supports dual CPU/GPU residency:

```c
typedef struct {
  AVFrame *frame;              // CPU data (always present for I/O)
  Pixel background;
  uint8_t abs_black_threshold;
  // GPU state stored in frame->opaque_ref (ImageCudaState)
} Image;
```

Key helpers:
- `image_ensure_cuda(Image*)`: Upload to GPU if CPU-dirty
- `image_ensure_cpu(Image*)`: Download from GPU if GPU-dirty
- Dirty flags track which copy is current; minimize transfers

### CUDA Backend Design

When `-Dcuda=enabled`:
- **OpenCV CUDA** is mandatory for `cudaarithm`, `cudaimgproc`, `cudawarping` modules
- **Runtime API** (`cudaMalloc`) used for OpenCV compatibility
- **Driver API** only for PTX kernel loading (custom kernels)
- **Custom kernels retained** for: mono formats (1-bit packed), blackfilter flood-fill, rotation detection

Performance (A1 benchmark): CUDA ~880ms vs CPU ~6.1s (~7x speedup)

## Build, Test, and Development Commands

When you are building with meson, set PATH="/home/scott/Documents/unpaper/.venv/bin:/usr/bin:$PATH" and use meson in .venv.

### CPU-only Build (default)

```bash
meson setup builddir/ --buildtype=debugoptimized
meson compile -C builddir/
meson test -C builddir/ -v
```

### CUDA Build (requires OpenCV with CUDA)

```bash
meson setup builddir-cuda/ -Dcuda=enabled --buildtype=debugoptimized
meson compile -C builddir-cuda/
meson test -C builddir-cuda/ -v
```

### Other Commands

```bash
# Build man page
meson compile -C builddir/ man

# Staged install
DESTDIR=/tmp/unpaper-staging meson install -C builddir/

# Run benchmarks
python tools/bench_a1.py --devices cpu,cuda
python tools/bench_double.py --devices cpu,cuda

# Pre-commit checks
pre-commit run -a
```

### Dependencies

- **Required**: FFmpeg libraries (`libavformat`, `libavcodec`, `libavutil`), Meson, Ninja
- **For CUDA builds**: CUDA toolkit, OpenCV 4.x with CUDA modules (`cudaarithm`, `cudaimgproc`, `cudawarping`)
- **For tests**: Python 3, pytest, Pillow
- **For docs**: Sphinx

## Coding Style & Naming Conventions

- Language: C11 (see `meson.build`)
- Indentation: 2 spaces; LF line endings; trim trailing whitespace (see `.editorconfig`)
- Formatting: `clang-format` enforced via `pre-commit`
- Backend functions: `*_cpu()` and `*_cuda()` suffixes
- Keep changes focused: prefer small helpers in `lib/` over duplicating logic

## Testing Guidelines

- Tests are `pytest`-based (`tests/unpaper_tests.py`) comparing outputs to golden images
- CUDA tests run automatically when CUDA runtime is available
- When changing image-processing behavior, update `tests/source_images/` and `tests/golden_images/` together
- Keep test output deterministic (CUDA must match CPU within tolerance)

## Commit & Pull Request Guidelines

- Commits: short, imperative subject lines; optional scope prefixes (`tests:`, `cuda:`, `perf:`)
- PRs: describe behavioral change, include relevant CLI flags, note any golden image updates
- Ensure `meson test -C builddir/ -v` and `pre-commit run -a` pass

## Licensing & Compliance

Files use SPDX headers. Add SPDX headers to new files and validate with `reuse lint` (via `pre-commit`).

---

## Active Development Roadmap

### Native Batch Processing (PR19-PR27)

**Goal**: Add native batch processing to efficiently process multiple images in a single invocation. Primary focus is GPU batch optimization to amortize CUDA overhead and exploit GPU parallelism.

**Performance target**: 10x faster than sequential CPU for batch workloads (100+ images).

**Speedup sources**:
1. Amortize CUDA initialization across entire batch (not per-image)
2. Overlap decode/upload/process/download/encode stages (pipelining)
3. Process multiple images concurrently via multiple CUDA streams
4. Thread-parallel CPU processing as baseline comparison

#### Architecture Analysis

**Current limitations**:
- Main loop processes one sheet at a time sequentially (`unpaper.c:1208-2212`)
- File I/O is synchronous (no pipelining yet)
- Image memory allocated/freed per-sheet, no pooling

**Existing infrastructure to leverage**:
- CUDA stream API: `unpaper_cuda_stream_create()`, async memcpy
- Pinned memory: `unpaper_cuda_pinned_alloc()`
- Backend vtable: clean dispatch, can be made thread-safe
- Wildcard patterns: already support `%d` for multi-file operations
- Batch queue: `BatchJob` struct and `BatchQueue` for job management (`lib/batch.h`)

#### PR-by-PR Roadmap

**PR 19: Batch CLI infrastructure + job queue**

- Status: complete
- Scope:
  - Add `--batch` / `-B` flag to enable batch processing mode
  - Add `--jobs N` / `-j N` to control parallelism (default: auto-detect)
  - Implement job queue abstraction: `BatchJob` struct with input/output paths (`lib/batch.c`, `lib/batch.h`)
  - Refactor main loop to populate job queue upfront in batch mode
  - Add progress reporting infrastructure (`--progress` flag)
- Acceptance:
  - CLI accepts new flags; `--help` documents them
  - Job queue correctly enumerates all files upfront
  - No behavior change for existing single-image invocations

**PR 20: CPU batch processing with thread pool**

- Status: complete
- Scope:
  - Implement pthread-based thread pool (`lib/threadpool.c`, `lib/threadpool.h`)
  - Worker threads process jobs concurrently with thread-local image buffers
  - Sheet processing extracted to `sheet_process.c`/`.h` for parallel execution
  - Batch worker coordination in `lib/batch_worker.c`/`.h`
  - Add batch benchmark script (`tools/bench_batch.py`)
- Results:
  - CPU batch with N threads shows near-linear speedup:
    - 1 thread: baseline (6117ms/image)
    - 2 threads: 1.89x speedup (3245ms/image)
    - 4 threads: 2.55x speedup (2402ms/image)
  - All existing tests pass

**PR 21: GPU batch - persistent context + memory pool**

- Status: complete
- Scope:
  - Implement GPU memory pool (`imageprocess/cuda_mempool.c`, `imageprocess/cuda_mempool.h`)
  - Pre-allocate N image-sized buffers; return to pool instead of `cudaFree()`
  - Pool statistics logging (allocations, reuses, peak usage)
  - Integration with `image_cuda.c` for transparent pool usage
  - Global pool management with thread-safe acquire/release
- Results:
  - Pool pre-allocates 8 buffers x 32MB = 256MB for A1 images
  - 100% pool hit rate for homogeneous batches (same image size)
  - Pool statistics output with `--perf` flag
  - Atomic operations for lock-free fast path acquire
  - All CPU and CUDA tests pass
- Acceptance:
  - Per-image CUDA malloc overhead eliminated for homogeneous batches
  - Flat GPU memory usage during batch processing

**PR 22: GPU batch - multi-stream pipeline infrastructure**

- Status: complete
- Scope:
  - Create stream pool: N `UnpaperCudaStream` instances (default N=4)
  - Associate each in-flight job with a stream
  - Stream synchronization points before download and stream reuse
- Results:
  - Stream pool (`imageprocess/cuda_stream_pool.c`, `imageprocess/cuda_stream_pool.h`)
  - Lock-free fast path acquire with atomic operations
  - Blocking wait with condition variable when all streams busy
  - Stream synchronization before release to ensure work completion
  - Integration with batch worker via `batch_worker_enable_stream_pool()`
  - Statistics: total acquisitions, wait events (contention), peak concurrent usage
  - All CPU (24 tests) and CUDA (9 tests + 34 pytest) pass
- Acceptance:
  - Infrastructure supports concurrent GPU operations across streams
  - No correctness issues from stream interleaving

**PR 23: GPU batch - decode/upload overlap (producer-consumer)**

- Status: complete
- Scope:
  - Producer thread: decode images (FFmpeg) -> queue decoded frames
  - Consumer (GPU): upload -> process -> download
  - Use pinned host memory for async H2D; double/triple buffering
- Results:
  - Decode queue (`lib/decode_queue.c`, `lib/decode_queue.h`)
  - Producer thread runs ahead, decoding images before workers need them
  - Bounded queue depth (2x parallelism) to control memory usage
  - Pinned memory support for CUDA builds (async H2D transfers)
  - Lock-free fast path for slot acquisition with atomic operations
  - Blocking wait with condition variables when queue full/empty
  - Integration with batch worker via `batch_worker_set_decode_queue()`
  - Statistics: images decoded/consumed, producer/consumer waits, pinned allocations
  - All CPU (24 tests) and CUDA (9 tests + 34 pytest) pass
- Acceptance:
  - Decode latency hidden behind GPU processing
  - Memory bounded by queue size x image size

**PR 24: GPU batch - concurrent multi-image GPU processing**

- Status: complete
- Scope:
  - Process multiple images concurrently using different streams
  - Work scheduling to balance load across streams
  - GPU occupancy monitoring
- Results:
  - GPU monitor module (`lib/gpu_monitor.c`, `lib/gpu_monitor.h`)
  - Concurrent job tracking with atomic counters for lock-free updates
  - Per-job GPU timing via CUDA events (start/stop pairs)
  - Memory usage monitoring via `cudaMemGetInfo()`
  - Statistics: total/peak/average concurrent jobs, GPU time (total/avg/min/max)
  - Peak concurrent jobs matches stream pool size (4 streams -> 4 concurrent)
  - Integration with batch worker via global GPU monitor
  - Batch start/end markers for accurate memory delta tracking
  - All CPU (24 tests) and CUDA (9 tests + 34 pytest) pass
- Acceptance:
  - GPU utilization increases during batch processing
  - NVIDIA Nsight shows concurrent kernel execution

**PR 25: GPU batch - download/encode overlap**

- Status: complete
- Scope:
  - Async D2H with pinned memory (`cudaMemcpyAsync`)
  - Encoder thread consumes downloaded frames
  - Pipeline: while GPU processes N, CPU encodes N-1
- Results:
  - Encode queue module (`lib/encode_queue.c`, `lib/encode_queue.h`)
  - Bounded queue with configurable depth (default: 2x parallelism)
  - Multiple encoder threads (default: 2) for I/O-bound encoding
  - Lock-free fast path for slot acquisition with atomic operations
  - Fast path format conversions: RGB24->MONOWHITE, RGB24->GRAY8, GRAY8->MONOWHITE
  - Statistics: images queued/encoded, waits, queue depth, encode time
  - Integration via `sheet_process_state_set_encode_queue()`
  - Frame cloning for async ownership transfer to encoder threads
  - All CPU (24 tests) and CUDA (9 tests + 34 pytest) pass
- Acceptance:
  - Encode latency hidden behind GPU processing
  - Full 4-stage pipeline operational

**PR 26: End-to-end batch pipeline + benchmarking**

- Status: complete
- Scope:
  - Integrate all pipeline stages
  - Comprehensive batch benchmark (10, 50, 100 images)
  - `--perf` output for batch mode: total time, images/second
  - Error handling: per-image failures logged, batch continues
  - Progress reporting: `[42/100] processing image042.png...`
- Results:
  - Batch performance summary output (`BatchPerfRecorder` in `lib/perf.c`)
  - Shows total time, images count (completed/failed), throughput, avg per image
  - Enhanced benchmark script (`tools/bench_batch.py`) with:
    - CUDA device support via `--devices cpu,cuda`
    - Multiple image counts via `--images 10,50,100`
    - 10x speedup verification via `--verify-10x`
    - Comparison against sequential CPU baseline
  - Per-image error logging in `lib/batch_worker.c` with input/output file details
  - Verified 10x speedup target:
    - 10 images: 13.33x speedup (CUDA batch vs sequential CPU)
    - 50 images: 14.74x speedup (CUDA batch vs sequential CPU)
  - All CPU (24 tests) and CUDA (9 tests + 34 pytest) pass
- Acceptance:
  - **Primary gate**: 100 images batch CUDA < sequential CPU / 10 [PASSED]
  - All test images produce correct output [PASSED]

**PR 27: Documentation + edge cases**

- Status: complete
- Scope:
  - Document batch processing in `doc/unpaper.1.rst`
  - Handle edge cases: mixed image sizes, GPU memory exhaustion
  - Graceful degradation when GPU memory exhausted
- Results:
  - Full documentation for `--batch`, `--jobs`, `--progress` options
  - New "Batch Processing Considerations" section covering:
    - Homogeneous image sizes for optimal GPU performance
    - GPU memory requirements and troubleshooting
    - Error handling behavior (failed jobs don't stop batch)
  - Memory pool statistics now show breakdown of miss reasons:
    - Size mismatches (images larger than pool buffer)
    - Pool exhaustion (all buffers in use)
  - GPU memory check at batch start with warning for low memory
  - All CPU (24 tests) and CUDA (9 tests + 34 pytest) pass
- Acceptance:
  - Documentation complete [DONE]
  - Robust handling of edge cases [DONE]

**PR 28: GPU auto-tuning for high-end GPUs**

- Status: complete
- Scope:
  - Auto-scale GPU resources based on available VRAM
  - Parallel decode threads to feed high-throughput GPUs
  - Scale encoder threads for fast GPU output
- Results:
  - GPU auto-tuning based on VRAM size (`unpaper.c`):
    - Formula: `tier = min(8, VRAM_GB / 3)`
    - Streams: `4 × tier` (4 to 32 streams)
    - Buffers: `streams × 2` (8 to 64 buffers)
    - Example: 24GB GPU → tier 7 → 28 streams, 56 buffers, 1792MB pool
  - Parallel decode queue (`lib/decode_queue.c`, `lib/decode_queue.h`):
    - `decode_queue_create_parallel()` for multi-threaded decoding
    - Work stealing via atomic job counter for load balancing
    - Decode threads scale with stream count: `streams / 4` (2 to 8 threads)
  - Encoder thread scaling for CUDA mode:
    - `parallelism / 4` threads (2 to 8) for high-throughput batches
  - Verbose output shows auto-tuned values:
    ```
    GPU auto-tune: 23 GB VRAM -> tier 7 -> 28 streams, 56 buffers
    GPU memory pool: 56 buffers x 33554432 bytes (1792.0 MB total)
    GPU stream pool: 28 streams
    Decode queue: 56 slots (pinned memory), parallel decode
    Encode queue: 56 slots, 4 encoder threads
    ```
  - Benchmark results (50 images on RTX 5090 24GB):
    - Sequential CPU: 307.9s (baseline)
    - Batch CUDA (j=8): 20.5s (**15.03x speedup**)
  - All CPU (24 tests) and CUDA (9 tests + 34 pytest) pass
- Acceptance:
  - High-end GPUs automatically utilize more resources [DONE]
  - Parallel decode reduces GPU starvation [DONE]
  - 15x speedup achieved (exceeds 10x target) [DONE]

#### Implementation Notes

**Thread safety**:
- `Options` struct: read-only after parsing
- `ImageBackend` pointer: set once at startup
- `Image` struct: per-job, not shared
- CUDA state: per-stream isolation

**Memory budget for GPU batch** (auto-scaled based on VRAM):
- Typical A1 image: ~26MB (2500x3500 RGB24), buffer size 32MB
- Default (small GPU): 4 streams, 8 buffers = 256MB pool
- High-end (24GB GPU): 28 streams, 56 buffers = 1792MB pool
- Auto-tuning formula: `tier = VRAM_GB / 3`, streams = `4 × tier`, buffers = `2 × streams`

---

### Stream Concurrency Optimization (PR29-PR34)

**Problem**: Despite having 28 CUDA streams on high-end GPUs (RTX 5090), batch processing only achieves ~15x speedup vs CPU baseline, far below the theoretical ~50x with near-linear stream scaling.

**Root Cause Analysis**: The `opencv_bridge.cpp` file contains `waitForCompletion()` sync points that serialize all streams on CPU-bound operations:

| Filter | Sync Point | CPU Operation | Impact |
|--------|-----------|---------------|--------|
| blurfilter | Line 514 | Download + integral + scan | **Critical** - 4 syncs per call |
| grayfilter | Line 389 | Download + 2× integral + scan | **Critical** - 1 sync per call |
| noisefilter | Lines 93, 124, 139 | CCL stats + mask modification | Moderate |
| sum_rect | Line 715 | cuda::sum() setup | Minor |

The critical bottleneck is the **integral image computation + scan loop** in blurfilter and grayfilter. Each stream must:
1. Download entire mask to CPU (~5-10ms)
2. Compute CPU integral (~10-20ms)
3. Run CPU scan loop (~20-50ms)

With 28 streams, each waiting for its CPU section, massive serialization occurs.

**Solution**: NPP GPU Integral + Custom GPU Scan Kernels

Move all computation to GPU using:
1. **NPP integral**: `nppiIntegral_8u32s_C1R_Ctx` computes integral on GPU with stream support
2. **Custom scan kernel**: Find threshold regions on GPU, output coordinate list
3. **Minimal transfer**: Only download coordinate list (<1KB) vs full image (~35MB)

**Performance Targets**:
- Single image (A1 bench): <1.5s (currently ~880ms, 2× regression acceptable)
- Batch scaling: ≥50x vs CPU baseline with 28 streams (currently ~15x)
- Expected single image: ~800-850ms (likely improvement, not regression)

#### PR-by-PR Roadmap

**PR 29: NPP Build Integration + Infrastructure**

- Status: complete
- Scope:
  - Add NPP library detection to `meson.build`:
    - Link `libnppc` (core), `libnppist` (statistics - contains integral)
    - Library path: `/usr/local/cuda-*/lib64/`
    - Header: `<nppi_statistics_functions.h>`
  - Create `imageprocess/npp_wrapper.c/.h`:
    - `NppStreamContext` initialization helper from CUDA stream
    - Device property caching (one-time init)
    - Error checking macros (`NPP_CHECK`, `NPP_CHECK_CTX`)
  - Create `imageprocess/npp_integral.c/.h`:
    - `npp_integral_create_context()` - initialize NppStreamContext
    - `npp_integral_8u32s()` - wrapper around `nppiIntegral_8u32s_C1R_Ctx`
    - Buffer management helpers for integral output
  - No functional changes to image processing yet
- Results:
  - NPP libraries detected and linked (libnppc, libnppist)
  - `UnpaperNppContext` wraps NppStreamContext with device property caching
  - `unpaper_npp_integral_8u32s()` computes integral on GPU with stream support
  - NPP format differs from OpenCV: first row/column zeros, output is width×height
  - Unit tests verify GPU integral matches CPU reference
  - All 10 CUDA tests + 34 pytest pass
- Files:
  - `meson.build` - add NPP dependency
  - `imageprocess/npp_wrapper.c/.h` - NPP infrastructure
  - `imageprocess/npp_integral.c/.h` - integral wrapper
  - `tests/npp_integral_test.c` - unit tests
- Acceptance:
  - Build succeeds with NPP libraries linked [DONE]
  - NPP functions callable from C code [DONE]
  - Unit test verifies NPP integral matches CPU integral [DONE]

**PR 30: Integral Buffer Pool + GPU Integral Implementation**

- Status: planned
- Scope:
  - Extend `cuda_mempool` to support integral buffers:
    - Integral output size: (width+1) × (height+1) × 4 bytes (int32)
    - For A1 (2500×3500): 2501×3501×4 = ~35MB per buffer
    - Pool 2× stream count buffers for double-buffering
  - Implement full GPU integral pipeline:
    - Allocate output buffer from pool
    - Call NPP integral with stream context
    - Return buffer to pool after use
  - Add integral buffer statistics to `--perf` output
- Files:
  - `imageprocess/cuda_mempool.c/.h` - extend for integral buffers
  - `imageprocess/npp_integral.c/.h` - complete implementation
- Acceptance:
  - GPU integral produces identical results to CPU `cv::integral()`
  - Integral buffers pooled (no per-call allocation)
  - Statistics show integral buffer hits/misses

**PR 31: Blurfilter GPU Scan Kernel**

- Status: planned
- Scope:
  - Design GPU scan kernel for blurfilter block isolation detection:
    ```cuda
    __global__ void unpaper_blurfilter_scan(
        const int32_t *integral,     // GPU integral image
        int integral_step,           // Bytes per row
        int width, int height,       // Image dimensions
        int block_w, int block_h,    // Block size
        int64_t total_pixels,        // block_w × block_h
        float intensity_threshold,   // Isolation threshold
        int2 *out_blocks,            // Output: block coordinates
        int *out_count               // Output: number of blocks found
    );
    ```
  - Algorithm:
    - Each thread processes one potential block position
    - Compute sum from integral using standard formula
    - Check neighbor sums for isolation criterion
    - Use atomic counter to collect matching blocks
  - Kernel launch parameters:
    - Grid: `(width / block_w, height / block_h)`
    - Block: `(16, 16)` or tuned for occupancy
- Files:
  - `imageprocess/cuda_kernels.cu` - add `unpaper_blurfilter_scan` kernel
  - `imageprocess/cuda_kernels_format.h` - declare kernel
- Acceptance:
  - Kernel produces same block list as CPU scan
  - Kernel runs asynchronously on stream (no sync required)
  - Output coordinate list small (<10KB for typical images)

**PR 32: Blurfilter Integration + Single Image Validation**

- Status: planned
- Scope:
  - Modify `unpaper_opencv_blurfilter()` in `opencv_bridge.cpp`:
    - Replace CPU integral with `npp_integral_8u32s()`
    - Replace CPU scan with `unpaper_blurfilter_scan` kernel
    - Download only coordinate list (not full image)
    - Keep final wipe operation on GPU
  - Sync point reduction:
    - Before: 2 syncs (download for integral, final wipe)
    - After: 1 sync (download coordinate list ~1KB)
  - Single image benchmark validation:
    - A1 bench target: <1.5s
    - Golden image tests must pass
- Files:
  - `imageprocess/opencv_bridge.cpp` - blurfilter modification
  - May need `imageprocess/cuda_runtime.c` for coordinate download
- Acceptance:
  - Single image: A1 bench <1.5s (target: ~850ms)
  - All pytest golden image tests pass
  - Blurfilter output identical to CPU reference

**PR 33: Grayfilter GPU Optimization**

- Status: planned
- Scope:
  - Similar optimization pattern for grayfilter:
    - GPU integral for both gray and dark_mask images
    - Custom scan kernel for tile detection
  - Grayfilter scan kernel differences from blurfilter:
    - Tile-based (not block-based)
    - Uses two integrals (gray sum + dark pixel count)
    - Different threshold logic (no dark pixels + low gray level)
  - Sync point reduction:
    - Before: 2 syncs (download for integrals, final wipe)
    - After: 1 sync (download coordinate list)
- Files:
  - `imageprocess/cuda_kernels.cu` - add `unpaper_grayfilter_scan` kernel
  - `imageprocess/opencv_bridge.cpp` - grayfilter modification
- Acceptance:
  - All grayfilter tests pass
  - Single image regression still <1.5s
  - Combined blurfilter + grayfilter optimization shows improvement

**PR 34: Batch Pipeline Optimization + Performance Validation**

- Status: planned
- Scope:
  - Comprehensive batch benchmarking with stream scaling:
    - Test: 4, 8, 16, 28 streams
    - Measure: actual concurrent GPU utilization
    - Target: near-linear scaling up to GPU compute saturation
  - NVIDIA Nsight profiling to verify:
    - Streams run concurrently (not serialized)
    - Kernel overlap visible in timeline
    - Reduced CPU activity during batch
  - Performance validation targets:
    - 28 streams: ≥50x vs sequential CPU baseline
    - Minimum: ≥40x (acknowledging some overhead)
  - Documentation update with final results
- Files:
  - `tools/bench_batch.py` - add stream scaling measurement
  - `CLAUDE.md` - document final performance numbers
- Acceptance:
  - **Primary gate**: 28 streams batch CUDA ≥50× sequential CPU
  - Stream utilization visible in Nsight timeline
  - All tests pass (no regressions)

#### Technical Details

**NPP Integral Image Notes**:
- NPP output dimensions: (width) × (height), NOT (width+1) × (height+1) like OpenCV
- First row/column implicitly zero in integral formula
- Use `nVal = 0` parameter for standard integral
- Stream context required for async execution

**Memory Budget** (additional for integral buffers):
- Per integral buffer: ~35MB for A1 images
- Double-buffering: 2× stream_count buffers
- 28 streams: 56 buffers × 35MB = ~2GB additional
- Total with 24GB VRAM: 1.8GB (images) + 2GB (integrals) = ~3.8GB used

**Kernel Design Considerations**:
- Integral sum formula: `sum = I[y1+1][x1+1] - I[y0][x1+1] - I[y1+1][x0] + I[y0][x0]`
- Coordinate clamping required for boundary blocks
- Atomic counter for output list (contention acceptable for small output)
- Consider warp-level primitives for neighbor checks

**Risk Mitigation**:
1. **Single image regression**: If NPP overhead exceeds benefit, add fast path for single-image mode
2. **NPP unavailable**: Graceful fallback to CPU integral (existing code)
3. **Kernel correctness**: Extensive golden image testing before merge

---

## Historical Documentation

For the completed CUDA backend implementation history (PR1-PR18), see [doc/CUDA_BACKEND_HISTORY.md](doc/CUDA_BACKEND_HISTORY.md).
