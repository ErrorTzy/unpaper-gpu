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
│   ├── cuda_mempool.c/.h  # GPU memory pool (image buffers, integral buffers)
│   ├── npp_integral.c/.h  # NPP GPU integral image computation
│   ├── nvjpeg_decode.c/.h # nvJPEG GPU decode (PR35+)
│   ├── nvjpeg_encode.c/.h # nvJPEG GPU encode (PR37+)
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
│   ├── bench_double.py    # Double-page benchmark
│   ├── bench_batch.py     # Batch processing benchmark
│   └── bench_jpeg_pipeline.py  # JPEG GPU pipeline benchmark (PR38+)
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
- **For CUDA builds**: CUDA toolkit (includes nvJPEG, NPP), OpenCV 4.x with CUDA modules (`cudaarithm`, `cudaimgproc`, `cudawarping`)
- **For JPEG GPU pipeline** (PR35+): nvJPEG library (part of CUDA toolkit, no separate install)
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

---

### GPU Batch Scaling Pipeline (PR36-PR42)

**Problem**: Despite nvJPEG GPU decode being fast (~15ms/image), batch scaling is limited to ~2x with 8 streams. Analysis reveals the root cause might be **excessive CUDA stream synchronization** during GPU processing, not decode/encode bottlenecks.

#### Root Cause Analysis

Profiling with 8 streams shows:
- GPU utilization: only ~50% (should be >90% for good scaling)
- Per-image timing variability: 1 worker = consistent 78ms; 8 workers = 54-306ms (4x variance!)
- Optimal scaling at 3 workers (2.08x), **degradation** with more workers

**The bottleneck**: 25 `cudaStreamSynchronize()` / `waitForCompletion()` calls in GPU processing code:
- `opencv_bridge.cpp`: 14 sync points (filters, masks, CCL)
- `opencv_ops.cpp`: 3 sync points (resize, deskew)
- `backend_cuda.c`: 3 sync points (blackfilter)
- `npp_integral.c`: 2 sync points
- Other: 3 sync points

Each sync blocks the CPU thread. With 8 workers hitting syncs, they serialize behind each other even though their GPU streams are independent.

#### Sync Point Categories

| Category | Count | Can Remove? | Example |
|----------|-------|-------------|---------|
| CPU needs intermediate GPU result | ~10 | No (need algorithm rewrite) | CCL component counting, filter tile selection |
| Buffer lifetime management | ~5 | Yes (use events) | Temp buffer freed after scope |
| Conservative/unnecessary | ~6 | Yes (same stream ordering) | Sync before NPP on same stream |
| Thread handoff | ~1 | No | Decode→worker handoff |
| Data transfer completion | ~3 | No | Upload/download must complete |

#### Performance Requirements

**Single-Image Baseline (No Regression Gate)**

All PRs must maintain single-image CUDA performance. Current baselines (A1-size 2480x3507):

| Format | CUDA Time | Threshold |
|--------|-----------|-----------|
| PNG | ~545ms | ≤650ms |
| JPEG (grayscale) | ~620ms | ≤700ms |
| PBM (1-bit) | ~575ms | ≤650ms |

**Batch Scaling Targets**

| PR | Expected Scaling (8 streams) | GPU Utilization | Approach |
|----|------------------------------|-----------------|----------|
| PR36 | ≥2.5x | ~60% | Remove unnecessary syncs |
| PR37 | ≥3.0x | ~70% | GPU-resident filter algorithms |
| PR38 | ≥3.5x | ~75% | Batched nvJPEG decode |
| PR39 | ≥4.0x | ~80% | Worker sync consolidation |
| PR40 | ≥4.5x | ~85% | nvJPEG GPU encode |
| PR41 | ≥6.0x | >95% | Batched GPU processing architecture |

**Scaling Cap Requirement**: When batch scaling plateaus (diminishing returns from adding more streams), GPU utilization must be ~100%. This proves the GPU is fully utilized, not blocked on CPU sync points.

---

**PR 36: Deferred Sync Architecture**

- Status: complete (partial scaling improvement)
- Goal: Remove/defer ~10 unnecessary sync points to improve multi-stream scaling
- Implemented:
  - Removed syncs before NPP calls (lines 483, 711 in `opencv_bridge.cpp`) - NPP uses same stream, ordering is automatic
  - Added `sync_after` parameter to `unpaper_opencv_grayfilter()` and `unpaper_opencv_blurfilter()` - let caller decide when to sync
  - Added CUDA event API (`unpaper_cuda_event_create/destroy/record/sync/query`) for future deferred sync patterns
  - Changed D2H memcpy in filters to use `unpaper_cuda_memcpy_d2h_async()` with stream parameter - critical fix to avoid device-wide synchronization from streamless cudaMemcpy
  - Changed memset to `cudaMemsetAsync()` on stream - avoid device-wide sync
  - Added memory pool usage for filter output buffers - avoid cudaMalloc/cudaFree serialization
- Files: `opencv_bridge.cpp`, `opencv_bridge.h`, `cuda_runtime.c`, `cuda_runtime.h`, `backend_cuda.c`
- Results:
  - All pytest tests pass ✓
  - **Single-image**: 580ms (≤650ms threshold) ✓
  - **Batch scaling**: 1.24x with 8 streams (target was 2.5x) - partial improvement
  - No change in output quality ✓
- Analysis:
  - Main bottleneck discovered: The ~10 sync points in filters are mostly unavoidable (need intermediate GPU results for tile/block coordinate download)
  - Remaining ~15 sync points in other operations (masks, deskew, etc.) not yet addressed
  - PR37 (GPU-resident filter algorithms) will eliminate the need to download tile/block coordinates by doing scan+wipe in a single kernel pass

---

**PR 37: GPU-Resident Filter Algorithms**

- Status: planned
- Goal: Eliminate D2H→CPU→H2D round-trips in grayfilter/blurfilter
- Problem: Current flow: GPU kernel finds tiles → sync → download count → CPU loops → upload wipes
- Approach:
  - New `unpaper_grayfilter_scan_and_wipe` kernel that scans AND wipes in one pass
  - New `unpaper_blurfilter_scan_and_wipe` kernel (same pattern)
  - Eliminate 4 sync points per image (2 per filter)
- Files: `cuda_kernels.cu`, `opencv_bridge.cpp`
- Acceptance:
  - All pytest tests pass
  - **Single-image**: ≤650ms for PNG/PBM, ≤700ms for JPEG
  - **Batch scaling**: 32 images, 8 streams achieves ≥3.0x speedup
  - Filter output matches baseline (<0.1% pixel difference)
  - GPU utilization increases to >70%

---

**PR 38: Batched nvJPEG Decode**

- Status: planned
- Goal: Decode all batch images with single `nvjpegDecodeBatched()` call instead of per-image decode
- Approach:
  - Use `nvjpegDecodeBatchedInitialize()` + `nvjpegDecodeBatched()` API
  - Collect all JPEG file data before starting batch
  - Single sync at end of batch decode
  - Alternative: phased API (`nvjpegDecodeJpegHost` → `nvjpegDecodeJpegTransferToDevice` → `nvjpegDecodeJpegDevice`) with overlap
- Files: `nvjpeg_decode.c`, `decode_queue.c`
- Acceptance:
  - All pytest tests pass
  - **Single-image**: ≤650ms for PNG/PBM, ≤700ms for JPEG
  - **Batch scaling**: ≥3.5x scaling with 8 streams
  - Decode phase 3x faster for 32 images vs per-image decode
  - Graceful fallback if batched decode fails

---

**PR 39: Worker Sync Consolidation**

- Status: planned
- Goal: Reduce per-image syncs from ~8 to 1-2 by deferring all syncs to end of processing
- Approach:
  - Add `bool sync_after` parameter to all backend GPU functions (default true for compatibility)
  - Modify `process_sheet()` to call all GPU operations with `sync_after=false`
  - Single sync before D2H transfer for output
- Files: `backend_cuda.c`, `backend.h`, `sheet_process.c`
- Acceptance:
  - All pytest tests pass
  - **Single-image**: ≤650ms for PNG/PBM, ≤700ms for JPEG
  - **Batch scaling**: ≥4.0x scaling with 8 streams
  - GPU utilization reaches >80%

---

**PR 40: nvJPEG GPU Encode**

- Status: planned
- Goal: Eliminate D2H transfer for JPEG output by encoding directly on GPU
- Approach:
  - Add `nvjpeg_encode_from_gpu()` using nvJPEG encoder API
  - Add `--output-format=jpeg` CLI option
  - Encoder state pool (same pattern as decoder)
- Files: `nvjpeg_encode.c/.h`, `encode_queue.c`, `options.h`, `parse.c`, `file.c`
- Acceptance:
  - All pytest tests pass
  - **Single-image**: ≤650ms for PNG/PBM, ≤700ms for JPEG
  - **Batch scaling**: ≥4.5x scaling with 8 streams
  - JPEG output quality matches ImageMagick (SSIM >0.99)
  - No D2H transfer for JPEG→JPEG workflow
  - GPU utilization reaches >85%

---

**PR 41: Batched GPU Processing Architecture**

- Status: planned
- Goal: Process all batch images as a GPU pipeline with minimal sync
- Approach:
  - Instead of N workers each processing 1 image with syncs
  - Single control thread orchestrating N images across M streams
  - Phase 1: All uploads (async); Phase 2: All processing (no sync); Phase 3: Single sync; Phase 4: All downloads
- Files: `batch_worker.c`, `unpaper.c`
- Acceptance:
  - All pytest tests pass
  - **Single-image**: ≤650ms for PNG/PBM, ≤700ms for JPEG
  - **Batch scaling**: ≥6.0x scaling with 8 streams
  - **GPU utilization >95%** when scaling plateaus (proves full GPU utilization)

---

**PR 42+: PDF Integration**

Once batch scaling is achieved, PDF workflow becomes straightforward via `pdfimages -j` for extraction and `img2pdf` for reassembly.

#### Verification Commands

```bash
# Single-image regression test (must pass for every PR)
python tools/bench_a1.py --devices cuda --iterations 5 --warmup 2
# Expected: ≤650ms for PNG

python tools/bench_a1.py --devices cuda --iterations 5 --warmup 2 --input /path/to/grayscale.jpg
# Expected: ≤700ms for JPEG

# Batch scaling benchmark
python tools/bench_batch.py --device cuda --streams 1,2,4,8 --images 32

# Profile with Nsight to verify sync reduction and GPU utilization
nsys profile -o profile ./builddir-cuda/unpaper --batch --device=cuda \
    --cuda-streams=8 input-%d.jpg output-%d.pnm

# Look for: fewer cudaStreamSynchronize gaps, overlapping kernel execution, high GPU utilization
```

---

## Historical Documentation

For the completed CUDA backend implementation history (PR1-PR18), see [doc/CUDA_BACKEND_HISTORY.md](doc/CUDA_BACKEND_HISTORY.md).
