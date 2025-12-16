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

### GPU Batch Scaling Pipeline (PR36-PR39) — REVISED DECEMBER 2024

**Scope**: JPEG-only workflow. PNG/PNM use FFmpeg CPU decode which doesn't scale.

---

#### Analysis Summary: Why Per-Image Decode Cannot Scale

**Problem Analysis (December 2024)**:

Systematic testing revealed that the current per-image nvjpegDecode() architecture **cannot achieve >~1x scaling** regardless of optimizations attempted:

| Test Configuration | 1 Stream | 8 Streams | Scaling | Root Cause |
|-------------------|----------|-----------|---------|------------|
| JPEG, no processing | 3.7s | 1.5s | **2.6x** | Per-image cudaStreamSynchronize |
| JPEG, decode-only (event-based) | 1.1s | 1.1s | **~1x** | nvJPEG internal serialization |
| JPEG, full processing | 19.1s | 14.2s | **1.35x** | blackfilter + decode combined |
| PNG, no processing | 1.6s | 1.7s | **0.97x** | FFmpeg CPU decode |

**Approaches Tested and Results**:

1. **Custom stream-ordered allocators** (`cudaMallocAsync`): Implemented but didn't help. nvJPEG still serializes internally.

2. **Per-stream CUDA streams for nvJPEG states**: Implemented - each `NvJpegStreamState` has dedicated CUDA stream. Still didn't scale.

3. **Event-based sync** (move sync from producer to worker): Tested and **failed** - achieved ~1x scaling. The sync point location doesn't matter because nvJPEG's internal decode phases are the bottleneck.

**Root Cause Analysis**:

The fundamental problem is that `nvjpegDecode()` (the simple API) performs:
```c
// nvJPEG internal flow - all serialized on the same state
1. Parse JPEG header (CPU)
2. Allocate device buffers (internal cudaMalloc - serializes!)
3. Transfer data H2D
4. GPU Huffman decode
5. GPU IDCT/color conversion
```

Even with custom allocators, the **Huffman decode phase uses GPU compute** that serializes across concurrent decodes. The GPU_HYBRID backend only triggers parallel GPU Huffman decode when batch size > 100 images AND using `nvjpegDecodeBatched()`.

**Conclusion**: Must migrate to `nvjpegDecodeBatched()` architecture.

---

#### nvjpegDecodeBatched() API Requirements

**Key API Constraints** (from NVIDIA documentation):

1. **All images in a batch must use IDENTICAL output format** - set once during `nvjpegDecodeBatchedInitialize()`
2. **Batch size >50 triggers optimized GPU decode** (HARDWARE backend); **>100 for GPU_HYBRID backend**
3. **State handles must be per-thread** - cannot share `nvjpegJpegState_t` across threads
4. **Pre-allocation only works with HARDWARE backend** (A100/H100) - not available on consumer GPUs

**Function Signatures**:
```c
// Initialize batched decoder (call once per batch configuration)
nvjpegStatus_t nvjpegDecodeBatchedInitialize(
    nvjpegHandle_t handle,
    nvjpegJpegState_t jpeg_handle,
    int batch_size,
    int max_cpu_threads,     // Deprecated, use 0
    nvjpegOutputFormat_t output_format);

// Decode entire batch (single call, single sync)
nvjpegStatus_t nvjpegDecodeBatched(
    nvjpegHandle_t handle,
    nvjpegJpegState_t jpeg_handle,
    const unsigned char *const *data,    // Array of JPEG data pointers
    const size_t *lengths,                // Array of data sizes
    nvjpegImage_t *destinations,          // Array of output buffer descriptors
    cudaStream_t stream);
```

**Memory Requirements**:
- Input: Array of host pointers to JPEG data (pinned memory preferred)
- Output: Pre-allocated GPU buffers for each image
- Pitch alignment: 256 bytes recommended for optimal memory access

---

#### Architecture Transformation Required

**Current Architecture** (per-image, producer-consumer):
```
┌─────────────────────────────────────────────────────────────────┐
│  Producer Thread 0          Producer Thread 1          ...     │
│  ┌─────────────────┐       ┌─────────────────┐                 │
│  │ fread(jpeg[0])  │       │ fread(jpeg[1])  │                 │
│  │ nvjpegDecode()  │       │ nvjpegDecode()  │    ← SERIALIZED │
│  │ sync            │       │ sync            │                 │
│  │ → slot[0]       │       │ → slot[1]       │                 │
│  └─────────────────┘       └─────────────────┘                 │
│           │                         │                          │
│           ▼                         ▼                          │
│  ┌─────────────────────────────────────────────────┐           │
│  │              Decode Queue (slots)               │           │
│  └─────────────────────────────────────────────────┘           │
│           │                         │                          │
│           ▼                         ▼                          │
│  Worker Thread 0            Worker Thread 1           ...      │
└─────────────────────────────────────────────────────────────────┘
```

**Required Architecture** (batch-collect-decode-distribute):
```
┌─────────────────────────────────────────────────────────────────┐
│  PHASE 1: COLLECT (parallel file I/O)                          │
│  ┌─────────────────────────────────────────────────────────┐   │
│  │ I/O Thread 0: fread → jpeg_data[0], jpeg_data[4], ...   │   │
│  │ I/O Thread 1: fread → jpeg_data[1], jpeg_data[5], ...   │   │
│  │ I/O Thread 2: fread → jpeg_data[2], jpeg_data[6], ...   │   │
│  │ I/O Thread 3: fread → jpeg_data[3], jpeg_data[7], ...   │   │
│  └─────────────────────────────────────────────────────────┘   │
│                            │                                    │
│                            ▼                                    │
│  PHASE 2: BATCHED DECODE (single API call)                     │
│  ┌─────────────────────────────────────────────────────────┐   │
│  │ nvjpegDecodeBatchedInitialize(batch_size, RGB)          │   │
│  │ nvjpegDecodeBatched(data[], lengths[], outputs[])       │   │
│  │ cudaStreamSynchronize(stream) ← ONE sync for ALL images │   │
│  └─────────────────────────────────────────────────────────┘   │
│                            │                                    │
│                            ▼                                    │
│  PHASE 3: DISTRIBUTE (workers pull from decoded pool)          │
│  ┌─────────────────────────────────────────────────────────┐   │
│  │ Worker 0: process(outputs[0]) → encode                  │   │
│  │ Worker 1: process(outputs[1]) → encode                  │   │
│  │ Worker 2: process(outputs[2]) → encode                  │   │
│  │ ...                                                      │   │
│  └─────────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────────┘
```

---

#### Implementation Challenges

**Challenge 1: Memory Constraints**

Decoding ALL images at once for large batches is not feasible:
- 32 images × 4000×6000 × 3 channels = **2.3 GB GPU RAM** just for decode output
- Plus processing buffers, integral images, etc.

**Solution**: Chunk-based processing
```c
#define MAX_DECODE_CHUNK 8  // Decode 8 images at a time

for (chunk = 0; chunk < num_images; chunk += MAX_DECODE_CHUNK) {
    int chunk_size = min(MAX_DECODE_CHUNK, num_images - chunk);

    // Phase 1: Collect this chunk's JPEG data (parallel I/O)
    collect_jpeg_data(&jpeg_data[chunk], &lengths[chunk], chunk_size);

    // Phase 2: Batch decode this chunk
    nvjpegDecodeBatched(handle, state, &jpeg_data[chunk], &lengths[chunk],
                        &outputs[0], chunk_size, stream);
    cudaStreamSynchronize(stream);  // ONE sync per chunk

    // Phase 3: Submit to workers (async processing)
    for (i = 0; i < chunk_size; i++) {
        submit_to_worker_pool(&outputs[i], chunk + i);
    }
}
```

**Challenge 2: Mixed Output Formats**

nvjpegDecodeBatched requires single format for entire batch. Some images are grayscale, others RGB.

**Solutions** (choose one):
- **Option A**: Always decode to RGB, convert grayscale on GPU after (slight overhead)
- **Option B**: Separate batches for gray vs RGB (more complex)
- **Option C**: Query format before batching, group by format (requires pre-scan)

**Recommendation**: Option A - decode everything as RGB. The conversion overhead (~0.5ms/image) is negligible compared to scaling gains.

**Challenge 3: Non-JPEG Files**

PNG files must still use FFmpeg CPU decode. Mixed batches need special handling.

**Solution**: Pre-filter files by extension
```c
// Separate JPEG and non-JPEG files
for (i = 0; i < num_files; i++) {
    if (is_jpeg(files[i])) {
        jpeg_files[jpeg_count++] = files[i];
        jpeg_job_indices[jpeg_count-1] = i;
    } else {
        // Decode immediately via FFmpeg (CPU path)
        cpu_decode_and_submit(files[i], i);
    }
}

// Batch decode all JPEGs
if (jpeg_count > 0) {
    nvjpeg_decode_batch(jpeg_files, jpeg_count, outputs);
    for (i = 0; i < jpeg_count; i++) {
        submit_to_worker(outputs[i], jpeg_job_indices[i]);
    }
}
```

**Challenge 4: Worker Synchronization**

Workers can't start until their image is decoded. With chunked batching, need coordination.

**Solution**: Completion flags per image
```c
typedef struct {
    atomic_bool decoded;     // Set true when decode complete
    void *gpu_ptr;           // GPU buffer pointer
    size_t pitch;
    int width, height;
} DecodedImageSlot;

// Decode thread sets flag after batch complete
for (i = 0; i < chunk_size; i++) {
    slots[chunk + i].gpu_ptr = outputs[i].channel[0];
    atomic_store(&slots[chunk + i].decoded, true);
}

// Worker waits for its slot
while (!atomic_load(&slots[job_index].decoded)) {
    sched_yield();  // Or use condition variable
}
```

---

#### PR Roadmap (Revised)

---

**PR 36A: Batched Decode Infrastructure**

- Status: **completed**
- Goal: Implement nvjpegDecodeBatched() wrapper with output buffer pool
- Scope:
  - Add `nvjpeg_decode_batched()` function in `nvjpeg_decode.c`
  - Pre-allocate output buffer pool (MAX_DECODE_CHUNK buffers)
  - Handle re-initialization when batch size or format changes
  - Unit tests for batched decode

**New Functions**:
```c
// Initialize batched decoder for given configuration
bool nvjpeg_batched_init(int max_batch_size, int max_width, int max_height,
                         NvJpegOutputFormat format);

// Decode a batch of JPEG images
// Returns number of successfully decoded images
int nvjpeg_decode_batch(
    const uint8_t *const *jpeg_data,  // Array of JPEG data pointers
    const size_t *jpeg_sizes,          // Array of data sizes
    int batch_size,                    // Number of images
    NvJpegDecodedImage *outputs);      // Output array (pre-allocated)

// Cleanup batched decoder resources
void nvjpeg_batched_cleanup(void);
```

- Files: `imageprocess/nvjpeg_decode.c/.h`, `tests/nvjpeg_batched_test.c`
- Acceptance:
  - Batched decode of 8 images works correctly ✓
  - Output matches single-image decode (pixel-level verification) ✓
  - No memory leaks (valgrind clean) ✓
  - Unit tests pass ✓
- Implementation notes:
  - Uses `nvjpegDecodeBatchedInitialize()` with `max_cpu_threads=1` (0 causes INVALID_PARAMETER)
  - `nvjpegDecodeBatchedSupported()` returns 0 for supported images, non-zero for unsupported
  - Pre-allocates GPU buffer pool with 256-byte pitch alignment
  - Fallback to concurrent single-image decode if batched API fails
  - Statistics tracking: total calls, images decoded, failures, max batch size used

---

**PR 36B: Batch-Oriented Decode Queue**

- Status: **planned** (depends on PR36A)
- Goal: Replace per-image producer model with batch-collect-decode-distribute
- Scope:
  - New `BatchDecodeQueue` that collects JPEG data first, then batch decodes
  - Parallel file I/O threads for collection phase
  - Chunked processing for memory efficiency
  - Integration with existing `BatchWorkerContext`

**Architecture Change**:
```c
// OLD: decode_queue.c producer_thread_fn()
for each job:
    for each input:
        fread() → nvjpegDecode() → sync → slot  // SERIALIZED

// NEW: batch_decode_queue.c
Phase 1 - Collect (parallel):
    io_threads[] → fread() → jpeg_data[]  // PARALLEL I/O

Phase 2 - Decode (batched):
    nvjpegDecodeBatched(jpeg_data[], outputs[])  // ONE CALL
    cudaStreamSynchronize()                       // ONE SYNC

Phase 3 - Distribute:
    for each output → worker_pool.submit()  // PARALLEL WORKERS
```

- Files:
  - `lib/batch_decode_queue.c/.h` (new)
  - `lib/batch_worker.c` - integrate new queue
  - `unpaper.c` - initialize new queue type
- Acceptance:
  - JPEG batch processing works end-to-end
  - Mixed JPEG+PNG batches handled correctly
  - Memory usage bounded (chunked processing)
  - No deadlocks or race conditions

---

**PR 36C: Performance Validation**

- Status: **planned** (depends on PR36B)
- Goal: Verify **≥3x scaling** with 8 streams for decode-only pipeline
- Scope:
  - Performance benchmarks comparing old vs new architecture
  - Nsight Systems profiling to verify single sync point
  - Tune chunk size for optimal throughput
  - Update `bench_batch.py` with new decode mode

**Verification**:
```bash
# Old architecture (for comparison)
./builddir-cuda/unpaper --batch --device=cuda --cuda-streams=8 \
    --decode-mode=per-image input%02d.jpg output%02d.pbm

# New architecture
./builddir-cuda/unpaper --batch --device=cuda --cuda-streams=8 \
    --decode-mode=batched input%02d.jpg output%02d.pbm
```

- Acceptance:
  - JPEG decode-only: **≥3x scaling** with 8 streams (up from ~1x)
  - Single sync point per chunk (verified via Nsight)
  - No regression for non-JPEG files
  - All pytest tests pass

---

**PR 37: Batched Blackfilter (unchanged)**

- Status: planned (after PR36 complete)
- Goal: **≥2.5x scaling** for blackfilter
- Approach: Batch all darkness computations into single kernel
- See existing PR37 documentation below

---

**PR 38: nvJPEG GPU Encode (unchanged)**

- Status: planned (after PR37)
- Goal: Full GPU-resident JPEG→JPEG pipeline
- See existing PR38 documentation below

---

#### Performance Targets Summary (Revised)

| PR | Component | Current | Target | Approach |
|----|-----------|---------|--------|----------|
| PR36A-C | Decode pipeline | ~1x | **≥3x** | nvjpegDecodeBatched |
| PR37 | Blackfilter | 1.38x | **≥2.5x** | Batched darkness kernel |
| PR37 | Full processing | 1.35x | **≥3x** | Combined effect |
| PR38 | JPEG encode | N/A | -10ms/img | nvJPEG GPU encode |

#### Key Implementation Notes

**Why nvjpegDecodeBatched over Phased API?**

The phased API (`nvjpegDecodeJpegHost` → `nvjpegDecodeJpegTransferToDevice` → `nvjpegDecodeJpegDevice`) was considered but rejected:
- Requires managing explicit state per image across phases
- Complex error handling if one image fails mid-phase
- nvjpegDecodeBatched handles all this internally
- Phased API doesn't provide better scaling for our use case

**Batch Size Threshold**:

nvJPEG GPU_HYBRID backend triggers optimized GPU Huffman decode only when `batch_size > 100`. For smaller batches:
- Still benefits from single sync point (vs N syncs)
- Still benefits from pre-allocated buffers
- May not see full GPU parallelism

Recommendation: Use chunk sizes of 8-16 for memory efficiency, accept that each chunk has a sync point.

**Hardware vs GPU_HYBRID Backend**:

- `NVJPEG_BACKEND_HARDWARE`: Uses dedicated JPEG decoder (A100/H100 only). Up to 20x faster.
- `NVJPEG_BACKEND_GPU_HYBRID`: Uses CUDA SMs for decode. Available on all GPUs.

We use GPU_HYBRID since consumer GPUs don't have hardware decoder. Performance will be better on datacenter GPUs.

#### Verification Commands (Revised)

```bash
# Test batched decode scaling
for streams in 1 2 4 8; do
  echo "Streams: $streams"
  time ./builddir-cuda/unpaper --batch --jobs=$streams --device=cuda \
       --cuda-streams=$streams --overwrite \
       --no-blackfilter --no-blurfilter --no-noisefilter --no-grayfilter \
       --no-mask-scan --no-mask-center --no-border --no-border-scan \
       --no-border-align --no-deskew \
       /tmp/test/input%02d.jpg /tmp/test/output%02d.pbm
done

# Profile with Nsight to verify batched decode
nsys profile -o nvjpeg_batched ./builddir-cuda/unpaper --batch --device=cuda \
    --cuda-streams=8 input%02d.jpg output%02d.pbm

# Expected Nsight output:
# - nvjpegDecodeBatched() calls (few, batched)
# - cudaStreamSynchronize() calls (one per chunk, not per image)
# - Overlapping worker kernel execution
```

#### References

- [nvJPEG Documentation (v13.1)](https://docs.nvidia.com/cuda/nvjpeg/index.html)
- [NVIDIA Blog: Leveraging Hardware JPEG Decoder](https://developer.nvidia.com/blog/leveraging-hardware-jpeg-decoder-and-nvjpeg-on-a100/)

#### PR-by-PR Roadmap

---

**PR 35: nvJPEG Build Integration + Core Infrastructure**

- Status: completed
- Scope:
  - Add nvJPEG library detection to `meson.build`:
    ```meson
    # nvJPEG is part of CUDA Toolkit, no separate install needed
    nvjpeg_dep = dependency('nvjpeg', required: cuda_enabled)
    # Fallback: find library directly
    if not nvjpeg_dep.found()
      nvjpeg_dep = cc.find_library('nvjpeg',
        dirs: cuda_libdir,
        required: cuda_enabled)
    endif
    ```
  - Create `imageprocess/nvjpeg_decode.c/.h` with core infrastructure:
    - `NvJpegContext` global context with lazy initialization
    - `NvJpegStreamState` per-stream state pool
    - **CRITICAL: Custom stream-ordered allocators** (see "Custom Allocator for Linear Scaling" section above):
      - `nvjpegDevAllocatorV2_t` using `cudaMallocAsync`/`cudaFreeAsync`
      - `nvjpegPinnedAllocatorV2_t` for pinned memory
      - Initialize with `nvjpegCreateExV2()` (NOT `nvjpegCreateSimple()`)
      - Set memory padding: `nvjpegSetDeviceMemoryPadding(1MB)`
    - Backend selection: `NVJPEG_BACKEND_GPU_HYBRID` for CUDA-based decode
    - Memory pre-allocation to avoid runtime `cudaMalloc`:
      ```c
      // Pre-allocate device buffer for max expected image size
      nvjpegBufferDeviceCreate(handle, &dev_buffer, NULL);
      nvjpegStateAttachDeviceBuffer(state, dev_buffer);

      // Pre-allocate pinned buffers (double-buffer for async)
      nvjpegBufferPinnedCreate(handle, &pin_buffer[0], NULL);
      nvjpegBufferPinnedCreate(handle, &pin_buffer[1], NULL);
      nvjpegStateAttachPinnedBuffer(state, pin_buffer[0]);
      ```
  - Implement `nvjpeg_decode_to_gpu()` with multi-phase pipeline:
    ```c
    // Phase 1: Host-side JPEG parsing (synchronous)
    nvjpegJpegStreamParse(handle, jpeg_data, jpeg_size, 0, 0, jpeg_stream);
    nvjpegDecodeJpegHost(handle, decoder, state, decode_params, jpeg_stream);

    // Phase 2: Transfer to device (async on stream)
    nvjpegDecodeJpegTransferToDevice(handle, decoder, state, jpeg_stream, cuda_stream);

    // Phase 3: GPU decode (async on stream)
    nvjpegDecodeJpegDevice(handle, decoder, state, &output_image, cuda_stream);
    ```
  - Add unit tests for nvJPEG decode:
    - Test grayscale JPEG decode (`NVJPEG_OUTPUT_Y`)
    - Test RGB JPEG decode (`NVJPEG_OUTPUT_RGB` / `NVJPEG_OUTPUT_RGBI`)
    - Test concurrent decode on multiple streams
    - Verify output matches FFmpeg decode (pixel-level comparison)
- Files:
  - `meson.build` - nvJPEG dependency detection
  - `imageprocess/nvjpeg_decode.c` - decode implementation
  - `imageprocess/nvjpeg_decode.h` - public API
  - `tests/nvjpeg_decode_test.c` - unit tests
  - `tests/source_images/test_jpeg.jpg` - test JPEG image
- Implementation notes:
  - Use `nvjpegGetImageInfo()` to query image dimensions before decode
  - Handle chroma subsampling: 4:4:4, 4:2:2, 4:2:0 supported
  - nvJPEG supports baseline and progressive JPEG, 1-4 channels
  - Error handling: check `nvjpegStatus_t` return codes, fall back to FFmpeg on failure
- Acceptance:
  - Build succeeds with nvJPEG linked (CUDA builds only)
  - `nvjpeg_context_init()` creates global handle + stream state pool
  - `nvjpeg_decode_to_gpu()` decodes JPEG directly to GPU memory
  - Unit test passes: decoded image matches FFmpeg reference within tolerance
  - **Custom allocator verified**: No `cudaMalloc` calls during decode (verify with `CUDA_LAUNCH_BLOCKING=1` or Nsight)
  - **Parallel scaling test**: 4 concurrent decodes complete in <2x time of 1 decode (proves no serialization)

---

**PR 36: Decode Queue GPU Integration**

- Status: completed (infrastructure in place with per-stream CUDA streams, GPU decode path enabled but nvJPEG batch decode has allocator issues)
- Scope:
  - Extend `DecodedImage` struct to track GPU residency:
    ```c
    typedef struct {
        AVFrame *frame;           // CPU frame (may be NULL for GPU-only)
        int job_index;
        int input_index;
        bool valid;
        bool uses_pinned_memory;
        // NEW fields for GPU decode:
        bool on_gpu;              // True if decoded directly to GPU
        void *gpu_ptr;            // GPU memory pointer (if on_gpu)
        size_t gpu_pitch;         // Row pitch in bytes
        int gpu_width, gpu_height;
        int gpu_format;           // NVJPEG_OUTPUT_* format
    } DecodedImage;
    ```
  - Modify `decode_queue` producer to use nvJPEG for JPEG files:
    ```c
    // In producer_thread_fn():
    const char *ext = strrchr(filename, '.');
    if (ext && (strcasecmp(ext, ".jpg") == 0 || strcasecmp(ext, ".jpeg") == 0)) {
        // GPU decode path
        NvJpegStreamState *nvstate = nvjpeg_acquire_stream_state();
        if (nvjpeg_decode_to_gpu(filename, nvstate, cuda_stream, &slot->image)) {
            slot->image.on_gpu = true;
            // No CPU frame needed
        }
        nvjpeg_release_stream_state(nvstate);
    } else {
        // Fallback: FFmpeg decode (existing path)
        decoded = decode_image_file(filename);
        slot->image.on_gpu = false;
    }
    ```
  - Modify `batch_worker` to handle GPU-decoded images:
    ```c
    // In batch_process_job():
    if (decoded_images[i]->on_gpu) {
        // Create Image struct pointing to GPU memory
        Image input = create_image_from_gpu(
            decoded_images[i]->gpu_ptr,
            decoded_images[i]->gpu_pitch,
            decoded_images[i]->gpu_width,
            decoded_images[i]->gpu_height,
            decoded_images[i]->gpu_format
        );
        // Mark as GPU-resident (skip image_ensure_cuda)
        image_set_gpu_resident(&input, true);
    } else {
        // Existing path: upload to GPU
        image_ensure_cuda(&input);
    }
    ```
  - Add helper in `image_cuda.c`:
    ```c
    // Create Image from existing GPU memory (no allocation)
    Image create_image_from_gpu(void *gpu_ptr, size_t pitch,
                                 int width, int height, int format);

    // Mark image as already on GPU (skip upload)
    void image_set_gpu_resident(Image *img, bool resident);
    ```
  - nvJPEG stream state pool management:
    - Pool size = num_streams (matches CUDA stream pool)
    - Lock-free acquire/release with atomic operations
    - Statistics: acquisitions, wait events, peak concurrent usage
- Files:
  - `lib/decode_queue.c/.h` - GPU decode path, DecodedImage extensions
  - `lib/batch_worker.c` - handle GPU-decoded images
  - `imageprocess/image_cuda.c` - `create_image_from_gpu()`, `image_set_gpu_resident()`
  - `imageprocess/nvjpeg_decode.c` - stream state pool management
- Implementation notes:
  - nvJPEG output format must match unpaper's internal format:
    - Grayscale: `NVJPEG_OUTPUT_Y` → `AV_PIX_FMT_GRAY8`
    - Color: `NVJPEG_OUTPUT_RGBI` → `AV_PIX_FMT_RGB24` (interleaved)
  - GPU memory ownership: decode queue owns GPU buffer until `decode_queue_release()`
  - Handle mixed batches: some JPEG (GPU decode), some PNG (CPU decode)
- Acceptance:
  - ~~JPEG files decoded directly to GPU (verified via `--perf` output)~~ (nvJPEG allocator failure in batch mode)
  - ~~No H2D transfer for JPEG input (verified via CUDA profiler)~~ (blocked by above)
  - ~~Mixed JPEG+PNG batches work correctly~~ (blocked by above)
  - ~~Batch benchmark shows improved scaling (target: ≥2.5x with 8 streams)~~ (blocked by above)
  - All existing tests pass (no regression) ✓
  - JPEG format comparison tests added (<10% dissimilarity) ✓
- Implementation notes (PR36 completion):
  - Infrastructure implemented and tested:
    - `DecodedImage` struct extended with GPU fields
    - `decode_jpeg_to_gpu()` function implemented
    - `create_image_from_gpu()` helper implemented
    - `image_is_gpu_resident()` and `image_set_gpu_resident()` implemented
    - nvJPEG context initialization integrated into unpaper.c
    - batch_worker updated to handle GPU-decoded images
    - **Each NvJpegStreamState now has its own dedicated CUDA stream** for true parallel decode
  - GPU decode path enabled in decode_queue.c, but nvJPEG decode fails with "Allocator failure" in batch mode
  - Remaining issues to investigate:
    - nvJPEG custom allocator not working correctly with nvjpegCreateExV2
    - May need to use nvjpegCreateSimple or different allocator approach
  - nvJPEG decode unit tests pass (concurrent decode verified)
  - All 38 pytest tests pass (including 4 new JPEG comparison tests)

---

**PR 37: nvJPEG Encode + GPU-Resident Output**

- Status: planned
- Scope:
  - Add nvJPEG encode support in `imageprocess/nvjpeg_encode.c/.h`:
    ```c
    typedef struct {
        nvjpegEncoderState_t state;
        nvjpegEncoderParams_t params;
    } NvJpegEncoderState;

    // Encode from GPU memory to JPEG bytes
    bool nvjpeg_encode_from_gpu(
        const Image *img,           // GPU-resident image
        int quality,                // JPEG quality (1-100)
        uint8_t **jpeg_data,        // Output: JPEG bytes (host memory)
        size_t *jpeg_size,          // Output: size in bytes
        cudaStream_t stream         // CUDA stream for async
    );
    ```
  - Implement encode pipeline:
    ```c
    // Setup encoder params
    nvjpegEncoderParamsSetQuality(params, quality, stream);
    nvjpegEncoderParamsSetSamplingFactors(params, NVJPEG_CSS_444, stream);

    // Encode from GPU (async on stream)
    nvjpegEncodeImage(handle, encoder_state, params,
                      &nv_image, input_format, width, height, stream);

    // Retrieve compressed data (requires sync)
    nvjpegEncodeRetrieveBitstream(handle, encoder_state, jpeg_data, &jpeg_size, stream);
    ```
  - Extend `encode_queue` to support GPU-resident images:
    ```c
    typedef struct {
        // Existing fields...
        bool source_on_gpu;         // True if source is GPU-resident
        void *gpu_ptr;              // GPU source pointer
        size_t gpu_pitch;
        // Output options:
        bool encode_to_jpeg;        // Use nvJPEG encode
        int jpeg_quality;
    } EncodeJob;
    ```
  - Add JPEG output format option:
    - CLI: `--output-format=pnm|jpeg` (default: pnm for compatibility)
    - When `--output-format=jpeg`: use nvJPEG encode, skip D2H transfer
  - Modify `saveImage()` to use nvJPEG when appropriate:
    ```c
    void saveImage(char *filename, Image input, int outputPixFmt) {
        const char *ext = strrchr(filename, '.');
        if (ext && strcasecmp(ext, ".jpg") == 0 && image_is_gpu_resident(&input)) {
            // GPU-direct JPEG encode
            nvjpeg_encode_from_gpu(&input, jpeg_quality, &data, &size, stream);
            write_file(filename, data, size);
            return;
        }
        // Existing path: D2H + PNM/FFmpeg encode
    }
    ```
- Files:
  - `imageprocess/nvjpeg_encode.c/.h` - encode implementation
  - `lib/encode_queue.c/.h` - GPU-resident source support
  - `lib/options.c/.h` - add `--output-format` option
  - `file.c` - integrate nvJPEG encode path
  - `parse.c` - parse `--output-format` option
  - `doc/unpaper.1.rst` - document new option
- Implementation notes:
  - nvJPEG encode requires contiguous RGB or YUV input
  - If internal format differs, convert on GPU before encode
  - Quality parameter: default 90 for document scanning
  - Encoder state pool: same pattern as decoder (per-stream)
- Acceptance:
  - JPEG output encoded directly from GPU memory
  - No D2H transfer for JPEG→JPEG workflow (verified via profiler)
  - Output JPEG visually identical to PNM→JPEG conversion
  - `--output-format=jpeg` documented in man page
  - A1 benchmark with JPEG output: no regression vs PNM output

---

**PR 38: Full GPU Pipeline + Performance Validation**

- Status: planned
- Scope:
  - Complete GPU-resident pipeline for JPEG→JPEG workflow:
    ```
    ┌─────────────────────────────────────────────────────────────┐
    │  JPEG file ──> nvJPEG decode ──> GPU memory                │
    │                                      │                      │
    │                              GPU Processing                 │
    │                       (filters, deskew, masks)              │
    │                                      │                      │
    │                                      ▼                      │
    │  JPEG file <── nvJPEG encode <── GPU memory                │
    │                                                             │
    │  *** Image NEVER leaves GPU memory ***                     │
    └─────────────────────────────────────────────────────────────┘
    ```
  - Optimize memory management:
    - nvJPEG decode buffer pool (pre-allocated per stream)
    - nvJPEG encode buffer pool (pre-allocated per stream)
    - Zero runtime `cudaMalloc` for JPEG workflow
  - Add comprehensive benchmarking:
    - New benchmark: `tools/bench_jpeg_pipeline.py`
    - Measure: decode time, process time, encode time, total time
    - Compare: JPEG pipeline vs PNM pipeline vs FFmpeg baseline
  - Batch scaling validation:
    - Test with 50images
    - Test with 1, 4, 8, 16 streams
    - Verify near-linear scaling
  - Profile with NVIDIA Nsight:
    - Verify no unexpected sync points
    - Verify concurrent kernel execution across streams
    - Verify H2D/D2H transfers only for non-JPEG formats
- Files:
  - `unpaper.c` - integrate full pipeline, pool initialization
  - `tools/bench_jpeg_pipeline.py` - JPEG-specific benchmark
  - `tools/bench_batch.py` - add JPEG format option
- Implementation notes:
  - Memory budget for JPEG pipeline (auto-scaled):
    - Decode buffers: num_streams × max_image_size
    - Encode buffers: num_streams × max_jpeg_size (compressed, ~1/10 of raw)
    - Total: ~40MB per stream for A1 images
  - Handle decode/encode errors gracefully:
    - nvJPEG error → fall back to FFmpeg path
    - Log warning, continue processing
- Acceptance:
  - **Primary gate**: JPEG batch scaling ≥3.5x with 8 streams (vs 1 stream)
  - **Secondary gate**: JPEG→JPEG workflow has zero H2D/D2H transfers
  - Single image (A1 bench): <500ms total (decode+process+encode)
  - Batch (50 images, 8 streams): <5s total (<100ms/image)
  - All existing tests pass
  - **Bottleneck verification** (all items from checklist above):
    - Nsight trace shows NO `cudaMalloc`/`cudaFree` during steady-state processing
    - Nsight trace shows overlapping kernel execution across streams
    - No stream stalls >10ms waiting for other streams
    - CPU utilization spread across multiple cores (not single-threaded)
  - **Scaling regression test** added to CI:
    ```bash
    # This test MUST pass for PR to merge
    python tools/bench_jpeg_pipeline.py --streams 1,4,8 --images 50
    # Expected output:
    # 1 stream:  X.XX img/s (baseline)
    # 4 streams: Y.YY img/s (≥2.5x baseline)
    # 8 streams: Z.ZZ img/s (≥3.5x baseline)  <-- PRIMARY GATE
    ```

---

#### Performance Targets Summary

| Metric | Current | Target | Notes |
|--------|---------|--------|-------|
| Single image (A1) | 486ms | <500ms | No regression |
| Batch scaling (8 streams) | 1.80x | **≥3.5x** | Primary goal |
| JPEG decode | 98ms (FFmpeg) | **15-25ms** (nvJPEG) | 4-6x improvement |
| JPEG encode | 34ms (direct PNM) | **10-15ms** (nvJPEG) | 2-3x improvement |
| H2D transfer (JPEG) | 5-10ms | **0ms** | Eliminated |
| D2H transfer (JPEG) | 5-10ms | **0ms** | Eliminated |

#### Thread Safety & Concurrency Model

```
┌────────────────────────────────────────────────────────────────────┐
│                     PROCESS-LEVEL (Shared)                         │
├────────────────────────────────────────────────────────────────────┤
│  nvjpegHandle_t (one per process, thread-safe)                    │
│  NvJpegStreamState pool[num_streams] (pre-allocated)              │
│  NvJpegEncoderState pool[num_streams] (pre-allocated)             │
└────────────────────────────────────────────────────────────────────┘
                              │
          ┌───────────────────┼───────────────────┐
          ▼                   ▼                   ▼
┌─────────────────┐ ┌─────────────────┐ ┌─────────────────┐
│   Stream 0      │ │   Stream 1      │ │   Stream N      │
├─────────────────┤ ├─────────────────┤ ├─────────────────┤
│ NvJpegStreamState│ │ NvJpegStreamState│ │ NvJpegStreamState│
│ - state         │ │ - state         │ │ - state         │
│ - dev_buffer    │ │ - dev_buffer    │ │ - dev_buffer    │
│ - pin_buffer[2] │ │ - pin_buffer[2] │ │ - pin_buffer[2] │
│ - jpeg_stream   │ │ - jpeg_stream   │ │ - jpeg_stream   │
├─────────────────┤ ├─────────────────┤ ├─────────────────┤
│ NvJpegEncoder   │ │ NvJpegEncoder   │ │ NvJpegEncoder   │
│ - enc_state     │ │ - enc_state     │ │ - enc_state     │
│ - enc_params    │ │ - enc_params    │ │ - enc_params    │
└─────────────────┘ └─────────────────┘ └─────────────────┘
```

**Key invariants**:
1. Each CUDA stream has dedicated nvJPEG state (no sharing)
2. Double-buffered pinned memory enables async H2D overlap
3. Stream state acquired atomically before decode/encode
4. No cross-stream resource sharing (eliminates sync points)

#### Future: PDF Integration (PR39+)

Once JPEG pipeline is complete, PDF workflow becomes straightforward:

```bash
# Extract JPEGs from PDF (zero-transcode, uses embedded JPEG directly)
pdfimages -j input.pdf page

# Process with unpaper (full GPU pipeline)
unpaper --batch --device=cuda --output-format=jpeg page-*.jpg output-%d.jpg

# Reassemble PDF (optional)
img2pdf output-*.jpg -o output.pdf
```

For programmatic PDF handling, consider libpoppler or MuPDF integration in future PRs.

---

## Historical Documentation

For the completed CUDA backend implementation history (PR1-PR18), see [doc/CUDA_BACKEND_HISTORY.md](doc/CUDA_BACKEND_HISTORY.md).
