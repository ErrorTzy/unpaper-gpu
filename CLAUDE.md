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

### GPU Decode/Encode Pipeline with nvJPEG (PR35-PR38)

**Problem**: Despite optimizations in PR34 (see /doc/CUDA_BACKEND_HISTORY.md), batch scaling is limited to 1.80x with 8 streams because CPU decode (FFmpeg) is slower than GPU processing:

| Stage | Time/image | Bottleneck? |
|-------|------------|-------------|
| FFmpeg decode (CPU) | 98ms | **YES - workers starve** |
| GPU processing | 57ms (8 streams) | No |
| H2D transfer | ~5-10ms | Adds latency |

#### Format Selection Analysis

**Why JPEG (nvJPEG) is the optimal choice:**

1. **PDF Workflow Optimization**: PDFs internally store images as compressed streams. Most scanned PDFs use DCTDecode (JPEG). Direct extraction via `pdfimages -j` yields the original JPEG bytes without transcoding.

2. **GPU Acceleration**: nvJPEG uses CUDA for parallel JPEG decoding, significantly faster than CPU-based FFmpeg decode.

3. **Performance Comparison**:
   | Format | Library | Decode Time | GPU Accel? | In PDFs? |
   |--------|---------|-------------|------------|----------|
   | **JPEG** | nvJPEG | **15-25ms** | Yes (CUDA) | Yes (common) |
   | JPEG 2000 | nvJPEG2000 | 20-30ms | Yes (CUDA) | Yes (archival) |
   | TIFF | nvTIFF | 15-25ms | Yes (CUDA) | No |
   | PNG | None | 50-80ms | **No** | No |
   | PNM | None | ~5ms | **No** (trivial) | No |

4. **Library Maturity**:
   - nvJPEG: Stable, part of CUDA Toolkit, extensive documentation
   - nvImageCodec: Beta (v0.7), unified API but PNG/PNM still CPU-only
   - Recommendation: Use nvJPEG directly for production stability

**Expected Performance with nvJPEG**:
- Decode: **15-25ms/image** (vs FFmpeg 98ms = **4-6x faster**)
- Eliminates CPU decode bottleneck
- Good stream scaling possible (≥3.5x with 8 streams)

#### Architecture: Per-Stream State Management

nvJPEG requires separate state objects per CUDA stream for concurrent decode:

```c
// Thread-safety model (CRITICAL for stream scaling):
// - nvjpegHandle_t: Thread-safe, ONE per process (shared)
// - nvjpegJpegState_t: NOT thread-safe, ONE per stream
// - nvjpegBufferDevice_t: NOT thread-safe, ONE per stream
// - nvjpegBufferPinned_t: NOT thread-safe, TWO per stream (double-buffer)

typedef struct {
    nvjpegJpegState_t state;           // Decoder state (per-stream)
    nvjpegBufferDevice_t dev_buffer;   // GPU output buffer (per-stream)
    nvjpegBufferPinned_t pin_buffer[2]; // Pinned staging (double-buffer)
    nvjpegJpegStream_t jpeg_stream;    // Bitstream parser (per-stream)
    int current_pin_buffer;            // Toggle for double-buffering
} NvJpegStreamState;

typedef struct {
    nvjpegHandle_t handle;             // Global handle (one per process)
    NvJpegStreamState *stream_states;  // Array[num_streams]
    int num_streams;
    pthread_mutex_t init_mutex;        // Protect lazy initialization
} NvJpegContext;
```

#### CRITICAL: Custom Allocator for Linear Scaling

**Problem**: Default nvJPEG uses `cudaMalloc()` internally, which is **globally serializing** - it blocks ALL threads/streams until complete. This prevents linear scaling even with per-stream state.

**Solution**: Provide custom stream-ordered allocators via `nvjpegCreateEx()`:

```c
// Custom device allocator using cudaMallocAsync (CUDA 11.2+)
// This is REQUIRED for linear stream scaling!
static int nvjpeg_dev_malloc(void *ctx, void **ptr, size_t size, cudaStream_t stream) {
    // Stream-ordered allocation - does NOT serialize across streams
    return cudaMallocAsync(ptr, size, stream) == cudaSuccess ? 0 : -1;
}

static int nvjpeg_dev_free(void *ctx, void *ptr, size_t size, cudaStream_t stream) {
    return cudaFreeAsync(ptr, stream) == cudaSuccess ? 0 : -1;
}

// Custom pinned allocator (pinned memory is less critical but still helps)
static int nvjpeg_pinned_malloc(void *ctx, void **ptr, size_t size, cudaStream_t stream) {
    // Use async if available (CUDA 11.2+), else fall back to sync
    cudaError_t err = cudaMallocHost(ptr, size);
    return err == cudaSuccess ? 0 : -1;
}

static int nvjpeg_pinned_free(void *ctx, void *ptr, size_t size, cudaStream_t stream) {
    return cudaFreeHost(ptr) == cudaSuccess ? 0 : -1;
}

// Initialize nvJPEG with custom allocators
nvjpegDevAllocatorV2_t dev_allocator = {
    .dev_ctx = NULL,
    .dev_malloc = nvjpeg_dev_malloc,
    .dev_free = nvjpeg_dev_free
};

nvjpegPinnedAllocatorV2_t pinned_allocator = {
    .pinned_ctx = NULL,
    .pinned_malloc = nvjpeg_pinned_malloc,
    .pinned_free = nvjpeg_pinned_free
};

// MUST use nvjpegCreateExV2 with custom allocators!
nvjpegCreateExV2(NVJPEG_BACKEND_GPU_HYBRID, &dev_allocator, &pinned_allocator,
                  NVJPEG_FLAGS_DEFAULT, &handle);

// Also set memory padding to reduce reallocations
nvjpegSetDeviceMemoryPadding(1024 * 1024, handle);  // 1MB padding
nvjpegSetPinnedMemoryPadding(1024 * 1024, handle);
```

#### Scaling Analysis: Why This Enables ≥3.5x with 8 Streams

| Factor | Without Custom Allocator | With Custom Allocator |
|--------|--------------------------|----------------------|
| Internal cudaMalloc | Serializes ALL streams | Stream-ordered, parallel |
| 8 concurrent decodes | Effectively 1 stream | True 8-stream parallelism |
| Expected scaling (8 streams) | **~1.5x** (serialized) | **≥3.5x** (target) |

**Why 3.5x and not 8x?** nvJPEG uses CUDA SMs for decode, which:
- Competes with processing kernels for SM resources
- Has diminishing returns as SM occupancy increases
- Memory bandwidth becomes limiting factor at high parallelism

Expected scaling curve:
- 1→4 streams: ~2.5-3x (SMs not saturated)
- 4→8 streams: ~1.3-1.5x additional (diminishing returns)
- **Total 8 streams: ≥3.5x** (our target)

#### Complete Bottleneck Checklist

Before claiming linear scaling, verify ALL of these are addressed:

| # | Bottleneck | How to Verify | Solution |
|---|------------|---------------|----------|
| 1 | `cudaMalloc` in nvJPEG | Nsight Systems trace, look for `cudaMalloc` during decode | Custom allocator with `cudaMallocAsync` |
| 2 | `cudaMalloc` in unpaper | Nsight trace | Use existing `cuda_mempool` |
| 3 | Per-stream state sharing | Code review | Separate `NvJpegStreamState` per stream |
| 4 | Global mutex in decode queue | Code review | Lock-free atomic acquire/release |
| 5 | File I/O serialization | Profile with `strace` | Memory-map files or use async I/O |
| 6 | CPU host phase contention | Profile CPU usage | Ensure num_decode_threads ≤ CPU cores |
| 7 | Pinned memory allocation | Nsight trace for `cudaMallocHost` | Pre-allocate pinned buffers |
| 8 | CUDA context switches | Nsight trace | All streams on same device |

**Verification command**:
```bash
# Profile with Nsight Systems to detect serialization
nsys profile -o nvjpeg_scaling ./builddir-cuda/unpaper --batch --device=cuda \
    --cuda-streams=8 input-%d.jpg output-%d.pnm

# Look for:
# - cudaMalloc/cudaFree during decode (BAD - should use async)
# - Long gaps between stream activity (BAD - serialization)
# - Overlapping kernel execution across streams (GOOD)
```

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

- Status: planned
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
  - JPEG files decoded directly to GPU (verified via `--perf` output)
  - No H2D transfer for JPEG input (verified via CUDA profiler)
  - Mixed JPEG+PNG batches work correctly
  - Batch benchmark shows improved scaling (target: ≥2.5x with 8 streams)
  - All existing tests pass (no regression)

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
