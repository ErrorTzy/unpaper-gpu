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

- Status: not started
- Scope:
  - Implement GPU memory pool (`imageprocess/cuda_mempool.c`)
  - Pre-allocate N image-sized buffers; return to pool instead of `cudaFree()`
  - Pool statistics logging (allocations, reuses, peak usage)
- Acceptance:
  - Per-image CUDA malloc overhead eliminated for homogeneous batches
  - Flat GPU memory usage during batch processing

**PR 22: GPU batch - multi-stream pipeline infrastructure**

- Status: not started
- Scope:
  - Create stream pool: N `UnpaperCudaStream` instances (default N=4)
  - Associate each in-flight job with a stream
  - Stream synchronization points before download and stream reuse
- Acceptance:
  - Infrastructure supports concurrent GPU operations across streams
  - No correctness issues from stream interleaving

**PR 23: GPU batch - decode/upload overlap (producer-consumer)**

- Status: not started
- Scope:
  - Producer thread: decode images (FFmpeg) -> queue decoded frames
  - Consumer (GPU): upload -> process -> download
  - Use pinned host memory for async H2D; double/triple buffering
- Acceptance:
  - Decode latency hidden behind GPU processing
  - Memory bounded by queue size x image size

**PR 24: GPU batch - concurrent multi-image GPU processing**

- Status: not started
- Scope:
  - Process multiple images concurrently using different streams
  - Work scheduling to balance load across streams
  - GPU occupancy monitoring
- Acceptance:
  - GPU utilization increases during batch processing
  - NVIDIA Nsight shows concurrent kernel execution

**PR 25: GPU batch - download/encode overlap**

- Status: not started
- Scope:
  - Async D2H with pinned memory (`cudaMemcpyAsync`)
  - Encoder thread consumes downloaded frames
  - Pipeline: while GPU processes N, CPU encodes N-1
- Acceptance:
  - Encode latency hidden behind GPU processing
  - Full 4-stage pipeline operational

**PR 26: End-to-end batch pipeline + benchmarking**

- Status: not started
- Scope:
  - Integrate all pipeline stages
  - Comprehensive batch benchmark (10, 50, 100 images)
  - `--perf` output for batch mode: total time, images/second
  - Error handling: per-image failures logged, batch continues
  - Progress reporting: `[42/100] processing image042.png...`
- Acceptance:
  - **Primary gate**: 100 images batch CUDA < sequential CPU / 10
  - All test images produce correct output

**PR 27: Documentation + edge cases**

- Status: not started
- Scope:
  - Document batch processing in `doc/unpaper.1.rst`
  - Handle edge cases: mixed image sizes, GPU memory exhaustion
  - Graceful degradation when GPU memory exhausted
- Acceptance:
  - Documentation complete
  - Robust handling of edge cases

#### Implementation Notes

**Thread safety**:
- `Options` struct: read-only after parsing
- `ImageBackend` pointer: set once at startup
- `Image` struct: per-job, not shared
- CUDA state: per-stream isolation

**Memory budget for GPU batch**:
- Typical A1 image: ~4MB (2500x3500 RGB24)
- 4 concurrent images: ~16MB GPU memory
- Pool should support ~8 buffers for triple-buffered 4-stream operation

---

## Historical Documentation

For the completed CUDA backend implementation history (PR1-PR18), see [doc/CUDA_BACKEND_HISTORY.md](doc/CUDA_BACKEND_HISTORY.md).
