<!--
SPDX-FileCopyrightText: 2025 The unpaper authors

SPDX-License-Identifier: GPL-2.0-only
-->

# Repository Guidelines

## Project Structure

```
unpaper/
├── unpaper.c              # Main CLI entry point
├── parse.c/.h             # Command-line parsing
├── file.c                 # Image I/O (FFmpeg decode/encode)
├── sheet_process.c/.h     # Per-sheet processing logic
├── imageprocess/          # Image processing
│   ├── backend.c/.h       # Backend vtable (CPU/CUDA dispatch)
│   ├── backend_cuda.c     # CUDA backend: infrastructure, vtable
│   ├── backend_cuda_internal.h # CUDA backend: shared declarations
│   ├── backend_cuda_blit.c    # CUDA backend: rectangle ops, transforms
│   ├── backend_cuda_masks.c   # CUDA backend: mask/border operations
│   ├── backend_cuda_filters.c # CUDA backend: blackfilter, blurfilter, etc.
│   ├── backend_cuda_deskew.c  # CUDA backend: rotation detection/deskew
│   ├── image.c/.h         # Image struct, CPU memory
│   ├── image_cuda.c       # GPU residency, dirty flags
│   ├── cuda_runtime.c/.h  # CUDA abstraction (streams, memory)
│   ├── cuda_kernels.cu    # CUDA kernels: main entry, grayscale ops
│   ├── cuda_kernels_common.cuh  # CUDA kernels: shared macros/helpers
│   ├── cuda_kernels_blit.cu     # CUDA kernels: rectangle ops, transforms
│   ├── cuda_kernels_masks.cu    # CUDA kernels: mask/border operations
│   ├── cuda_kernels_filters.cu  # CUDA kernels: blackfilter, flood-fill
│   ├── cuda_kernels_deskew.cu   # CUDA kernels: rotation detection
│   ├── cuda_mempool.c/.h  # GPU memory pool (images + integrals)
│   ├── cuda_stream_pool.c/.h # CUDA stream pool
│   ├── nvjpeg_decode.c/.h # nvJPEG GPU decode
│   ├── nvjpeg_encode.c/.h # nvJPEG GPU encode
│   ├── npp_wrapper.c/.h   # NPP infrastructure
│   ├── npp_integral.c/.h  # NPP integral image computation
│   ├── opencv_bridge.cpp/.h # OpenCV CUDA bridge (filters)
│   ├── opencv_ops.cpp/.h  # OpenCV CUDA operations
│   ├── blit.c             # CPU: rectangle operations
│   ├── deskew.c           # CPU: rotation detection/correction
│   ├── filters.c          # CPU: blackfilter, blurfilter, etc.
│   └── masks.c            # CPU: mask detection, borders
├── lib/                   # Utilities
│   ├── batch.c/.h         # Batch job queue
│   ├── batch_worker.c/.h  # Batch worker coordination
│   ├── batch_decode_queue.c/.h # Batched decode queue
│   ├── decode_queue.c/.h  # Decode queue with GPU integration
│   ├── encode_queue.c/.h  # Encode queue
│   ├── logging.c/.h       # Verbose output
│   └── options.c/.h       # Options struct
├── tests/                 # Test suite
│   ├── unpaper_tests.py   # Pytest golden tests
│   └── source_images/     # Test inputs
├── tools/                 # Benchmark and utility scripts
│   ├── bench_a1.py        # Single-image A1 benchmark
│   ├── bench_batch.py     # Batch processing benchmark
│   └── bench_jpeg_pipeline.py # JPEG pipeline benchmark
├── pdf/                   # PDF support
│   ├── pdf_reader.c/.h    # PDF reading (MuPDF)
│   ├── pdf_writer.c/.h    # PDF writing (MuPDF)
│   ├── pdf_pipeline_cpu.c/.h  # Sequential CPU PDF pipeline
│   ├── pdf_pipeline_gpu.c/.h  # Sequential GPU PDF pipeline
│   ├── pdf_pipeline_batch.c/.h # Parallel batch PDF pipeline
│   └── pdf_page_accumulator.c/.h # Thread-safe page accumulator
└── doc/                   # Documentation
    ├── CUDA_BACKEND_HISTORY.md  # Completed PR history
    └── nvimgcodec_migration_baseline.md  # nvImageCodec migration baseline
```

## Architecture Overview

### Backend System

Backend vtable (`ImageBackend` in `backend.h`) dispatches operations:
- **CPU backend**: Default, FFmpeg-based
- **CUDA backend**: GPU-accelerated via OpenCV CUDA modules

Selection: `--device=cpu|cuda`

### Image Memory Model

`Image` struct supports dual CPU/GPU residency:
- `image_ensure_cuda()`: Upload to GPU if needed
- `image_ensure_cpu()`: Download from GPU if needed
- Dirty flags minimize transfers

### CUDA Backend

Requires OpenCV with CUDA modules (`cudaarithm`, `cudaimgproc`, `cudawarping`).
Custom kernels for: mono formats, blackfilter flood-fill, rotation detection.
Performance: ~7x speedup vs CPU (A1 benchmark).

The CUDA backend is split into modular files mirroring the CPU structure:
- `backend_cuda.c` - Infrastructure, kernel loading, shared helpers, vtable
- `backend_cuda_blit.c` - Rectangle ops (wipe, copy, rotate90, mirror, stretch, resize)
- `backend_cuda_masks.c` - Mask detection and border operations
- `backend_cuda_filters.c` - blackfilter, blurfilter, noisefilter, grayfilter
- `backend_cuda_deskew.c` - Rotation detection and deskew
- `backend_cuda_internal.h` - Shared declarations across modules

### Pipeline Auto-Selection

The system automatically selects the optimal pipeline based on device and file formats:

| Backend | Input Format | Output Format | Pipeline |
|---------|--------------|---------------|----------|
| CPU | Any | Any | FFmpeg decode → CPU → FFmpeg encode |
| GPU | JPEG | JPEG | nvJPEG decode → GPU → nvJPEG encode (zero-copy) |
| GPU | JPEG | non-JPEG | nvJPEG decode → GPU → D2H → FFmpeg encode |
| GPU | non-JPEG | JPEG | FFmpeg decode → H2D → GPU → nvJPEG encode |
| GPU | non-JPEG | non-JPEG | FFmpeg decode → H2D → GPU → D2H → FFmpeg encode |

**User controls:**
- `--device=cpu|cuda` - Backend selection
- `--jpeg-quality=N` - JPEG output quality (1-100, default 85)

**Automatic behaviors:**
- GPU backend auto-enables nvJPEG decode for JPEG inputs
- GPU backend auto-enables nvJPEG encode when any output is JPEG
- Per-image decode mode is always used (faster than batched)

## Build Commands

**Set PATH for meson**: `PATH="/home/scott/Documents/unpaper/.venv/bin:/usr/bin:$PATH"`

### CPU Build
```bash
meson setup builddir/ --buildtype=debugoptimized
meson compile -C builddir/
meson test -C builddir/ -v
```

### CUDA Build
```bash
meson setup builddir-cuda/ -Dcuda=enabled --buildtype=debugoptimized
meson compile -C builddir-cuda/
meson test -C builddir-cuda/ -v
```

### PDF Build
```bash
meson setup builddir-pdf/ -Dpdf=enabled --buildtype=debugoptimized
meson compile -C builddir-pdf/
meson test -C builddir-pdf/ -v
```

### Dependencies
- **Required**: FFmpeg libs, Meson, Ninja
- **CUDA**: CUDA toolkit (nvJPEG, NPP), OpenCV 4.x with CUDA
- **PDF**: MuPDF library (`libmupdf-dev` on Debian/Ubuntu)
- **Tests**: Python 3, pytest, Pillow

## Coding Style

- C11, 2-space indent, LF endings (see `.editorconfig`)
- `clang-format` enforced via `pre-commit`
- Backend functions: `*_cpu()` / `*_cuda()` suffixes

## Testing

- Pytest-based golden image comparison
- Update `source_images/` and `golden_images/` together
- CUDA must match CPU within tolerance

## Commits

- Short imperative subjects; scope prefixes: `tests:`, `cuda:`, `perf:`
- Run `meson test -C builddir/ -v` and `pre-commit run -a` before commit
- SPDX headers required on new files

---

## Completed Development

The CUDA backend implementation is complete. Key capabilities:

| Feature | Description | Performance |
|---------|-------------|-------------|
| CUDA backend | OpenCV CUDA + custom kernels | ~7x vs CPU (single image) |
| Batch processing | Multi-stream pipeline | ~15x vs sequential CPU |
| nvJPEG decode | GPU-resident JPEG decode | Per-image mode (20% faster than batched) |
| nvJPEG encode | GPU-resident JPEG encode | Zero-copy for JPEG→JPEG workflows |
| NPP integral | GPU integral images | Async filter operations |
| Stream pool | Auto-scaled by VRAM | Up to 28 streams on high-end GPUs |

### Known Limitations

**Stream scaling is limited** (~1.8x with 8 streams vs theoretical 8x):
- `cudaStreamSynchronize()` calls after decode/encode serialize GPU work
- Workers wait for specific jobs by index, preventing out-of-order processing
- `cudaMalloc`/`cudaFree` are inherently synchronous and serialize all streams

---

## PDF Support Roadmap

**Goal**: PDF in → processing → PDF out, with maximum GPU performance and CPU fallback.

### Design Decisions

| Choice | Decision | Rationale |
|--------|----------|-----------|
| PDF library | MuPDF | 3x faster than Poppler, handles read+write |
| GPU codec | nvImageCodec | Adds JPEG2000 GPU decode/encode to existing JPEG |
| JBIG2 | jbig2dec | Native B&W support, no lossy conversion |
| Output modes | `--pdf-quality=high\|fast` | JP2 lossless vs JPEG lossy |

### Pipeline Architecture

**GPU Pipeline** (optimized):
```
PDF → MuPDF extract raw bytes → nvImageCodec GPU decode → GPU process → GPU encode → MuPDF embed → PDF
```

**CPU Pipeline** (fallback):
```
PDF → MuPDF extract/render → FFmpeg decode → CPU process → FFmpeg encode → MuPDF embed → PDF
```

---

### PR 1: MuPDF Integration + PDF Reader [COMPLETE]

**Status**: Implemented and tested.

**Why**: Foundation for PDF I/O. MuPDF extracts embedded images in native format (no re-decode).

**Implementation**:
- Add MuPDF dependency in `meson.build` (`-Dpdf=enabled`)
- New `pdf/pdf_reader.c/.h`: open PDF, iterate pages, extract raw image bytes
- Detect image format (JPEG, JP2, JBIG2, CCITT, raw) from PDF stream
- Add `pdf_render_page()` fallback: render page to RGB pixels at specified DPI
- Extract page dimensions, DPI, metadata

**Key API** (implemented in `pdf/pdf_reader.h`):
```c
PdfDocument *pdf_open(const char *path);
PdfDocument *pdf_open_memory(const uint8_t *data, size_t size);
void pdf_close(PdfDocument *doc);
int pdf_page_count(PdfDocument *doc);
bool pdf_get_page_info(PdfDocument *doc, int page, PdfPageInfo *info);
bool pdf_extract_page_image(PdfDocument *doc, int page, PdfImage *image);
uint8_t *pdf_render_page(PdfDocument *doc, int page, int dpi, int *w, int *h, int *stride);
uint8_t *pdf_render_page_gray(PdfDocument *doc, int page, int dpi, int *w, int *h, int *stride);
PdfMetadata pdf_get_metadata(PdfDocument *doc);
```

**Tests**: `tests/pdf_reader_test.c` - Unit tests for all API functions.
- Test PDFs in `tests/pdf_samples/`
- Verifies page count, page info, image extraction, rendering, and metadata

**Build**: `meson setup builddir -Dpdf=enabled && meson compile -C builddir`

**Success**: Can extract raw JPEG bytes from PDF without re-encoding. Rendering fallback works for complex pages.

---

### PR 2: PDF Writer + Metadata Preservation [COMPLETE]

**Status**: Implemented and tested.

**Why**: Create output PDFs with direct image embedding (no transcoding overhead).

**Implementation**:
- New `pdf/pdf_writer.c/.h`: create PDF, add pages, embed images
- Support JPEG and JP2 direct embedding (zero-copy path)
- Add `pdf_writer_add_page_pixels()` for raw RGB/grayscale data (Flate compression)
- Copy metadata from input PDF (title, author, subject, keywords, dates)
- Page size computed from image dimensions + DPI
- Abort support for error handling without partial file creation

**Key API** (implemented in `pdf/pdf_writer.h`):
```c
PdfWriter *pdf_writer_create(const char *path, const PdfMetadata *meta, int dpi);
bool pdf_writer_add_page_jpeg(PdfWriter *w, const uint8_t *data, size_t len, int width, int height, int dpi);
bool pdf_writer_add_page_jp2(PdfWriter *w, const uint8_t *data, size_t len, int width, int height, int dpi);
bool pdf_writer_add_page_pixels(PdfWriter *w, const uint8_t *pixels, int w, int h, int stride, PdfPixelFormat fmt, int dpi);
bool pdf_writer_close(PdfWriter *w);
void pdf_writer_abort(PdfWriter *w);
int pdf_writer_page_count(const PdfWriter *w);
```

**Tests**: `tests/pdf_writer_test.c` - Unit tests for all API functions.
- Create PDF with embedded JPEG
- Create PDF with pixel data (grayscale and RGB)
- Multi-page PDF creation
- Metadata preservation verification
- Extract JPEG from source PDF and re-embed

**Build**: `meson setup builddir-pdf -Dpdf=enabled && meson compile -C builddir-pdf`

**Success**: Output PDF contains directly embedded images, metadata matches input.

---

### PR 3: CPU PDF Pipeline [COMPLETE]

**Status**: Implemented and tested.

**Why**: Functional PDF support without GPU. Uses existing CPU backend with FFmpeg decode/encode.

**Implementation**:
- New `pdf/pdf_pipeline_cpu.c/.h`: orchestrate CPU PDF processing
- Wire PDF detection in `unpaper.c` main to auto-switch to PDF pipeline
- For each page: extract raw bytes or render → FFmpeg decode to AVFrame → CPU process → FFmpeg encode → embed
- Sequential processing (optimized for throughput)

**Processing path**:
```
PDF page → MuPDF extract → FFmpeg decode (JPEG/PNG) → Image struct → CPU process → JPEG encode → PDF
                 ↓ (fallback if extraction fails)
         MuPDF render to pixels → Image struct → CPU process → JPEG encode → PDF
```

**Key API** (implemented in `pdf/pdf_pipeline_cpu.h`):
```c
int pdf_pipeline_cpu_process(const char *input_path, const char *output_path,
                             const Options *options,
                             const SheetProcessConfig *config);
bool pdf_pipeline_is_pdf(const char *filename);
```

**Tests**: `tests/pdf_pipeline_cpu_test.c` - Unit tests for PDF pipeline.
- Single-page PDF processing
- Multi-page PDF processing
- Invalid input handling
- Output quality verification

**Build**: `meson setup builddir-pdf -Dpdf=enabled && meson compile -C builddir-pdf`

**Usage**: `unpaper input.pdf output.pdf` - Auto-detects PDF files and uses PDF pipeline.

**Success**: PDF processing works on CPU-only systems. JPEG and rendered pages handled correctly.

---

### PR 4: nvImageCodec Integration [COMPLETE]

**Status**: Implemented and tested.

**Why**: Adds JPEG2000 GPU decode/encode. Unified API replaces nvJPEG.

**Implementation**:
- Add nvImageCodec dependency (`-Dnvimgcodec=enabled`, falls back to nvJPEG if unavailable)
- New `imageprocess/nvimgcodec.c/.h`: wrapper with format detection
- Decode JPEG/JP2 to GPU-resident buffer
- Encode GPU buffer to JPEG (quality 1-100) or JP2 (lossless/near-lossless)
- Update `decode_queue.c` and `encode_queue.c` to use new API

**Key API** (implemented in `imageprocess/nvimgcodec.h`):
```c
// Format detection
NvImgCodecFormat nvimgcodec_detect_format(const uint8_t *data, size_t size);

// Initialization
bool nvimgcodec_init(int num_streams);
void nvimgcodec_cleanup(void);
bool nvimgcodec_is_available(void);      // True if nvImageCodec loaded
bool nvimgcodec_any_available(void);     // True if any codec available (nvJPEG fallback)
bool nvimgcodec_jp2_supported(void);     // True only with full nvImageCodec

// Decode/encode state management (pool-based, thread-safe)
NvImgCodecDecodeState *nvimgcodec_acquire_decode_state(void);
void nvimgcodec_release_decode_state(NvImgCodecDecodeState *state);
NvImgCodecEncodeState *nvimgcodec_acquire_encode_state(void);
void nvimgcodec_release_encode_state(NvImgCodecEncodeState *state);

// Decode to GPU
bool nvimgcodec_decode(const uint8_t *data, size_t size,
                       NvImgCodecDecodeState *state, UnpaperCudaStream *stream,
                       NvImgCodecOutputFormat output_fmt, NvImgCodecDecodedImage *out);

// Encode from GPU
bool nvimgcodec_encode_jpeg(const void *gpu_ptr, size_t pitch, int width, int height,
                            NvImgCodecEncodeInputFormat input_fmt, int quality,
                            NvImgCodecEncodeState *state, UnpaperCudaStream *stream,
                            NvImgCodecEncodedImage *out);
bool nvimgcodec_encode_jp2(const void *gpu_ptr, size_t pitch, int width, int height,
                           NvImgCodecEncodeInputFormat input_fmt, bool lossless,
                           NvImgCodecEncodeState *state, UnpaperCudaStream *stream,
                           NvImgCodecEncodedImage *out);
```

**Tests**: `tests/nvimgcodec_test.c` - Unit tests for wrapper API.
- Format detection tests (JPEG, JP2, unknown)
- State acquire/release tests
- JPEG decode to GPU test
- JPEG encode from GPU test
- JP2 support status verification

**Build**: `meson setup builddir-cuda -Dcuda=enabled && meson compile -C builddir-cuda`

**Success**: JPEG2000 decodes on GPU when nvImageCodec is available. Falls back to nvJPEG for JPEG-only when nvImageCodec is not installed. Existing JPEG tests pass. No performance regression.

---

### PR 5: JBIG2 Native Support [COMPLETE]

**Status**: Implemented and tested.

**Why**: B&W scanned documents use JBIG2. Native decode avoids lossy JPEG conversion.

**Implementation**:
- Add jbig2dec dependency (`-Djbig2=enabled`, auto-detected if available)
- New `lib/jbig2_decode.c/.h`: decode JBIG2 to 1-bit bitmap with globals support
- Extended `PdfImage` struct to include JBIG2 globals data
- Modified `pdf_reader.c` to extract JBIG2 globals when present
- Modified `pdf_pipeline_cpu.c` to use jbig2_decode for JBIG2 images
- Automatic expansion from 1-bit packed to 8-bit grayscale for processing pipeline

**Key API** (implemented in `lib/jbig2_decode.h`):
```c
// Decode JBIG2 data to 1-bit bitmap
bool jbig2_decode(const uint8_t *data, size_t size, const uint8_t *globals,
                  size_t globals_size, Jbig2DecodedImage *out);

// Expand 1-bit to 8-bit grayscale
bool jbig2_expand_to_gray8(const Jbig2DecodedImage *jbig2, uint8_t *gray_out,
                           size_t gray_stride, bool invert);

// Check runtime availability
bool jbig2_is_available(void);
void jbig2_free_image(Jbig2DecodedImage *image);
```

**Processing path**:
```
PDF JBIG2 → MuPDF extract (data + globals) → jbig2dec → 1-bit bitmap → expand to 8-bit gray → process → encode
```

**Tests**: `tests/jbig2_decode_test.c` - Unit tests for JBIG2 decode.
- Test PDF with JBIG2 content in `tests/pdf_samples/test_jbig2.pdf`
- JBIG2 availability check
- Null safety tests
- Decode from PDF with globals
- Expand to grayscale verification

**Build**: `meson setup builddir-pdf -Dpdf=enabled -Djbig2=enabled && meson compile -C builddir-pdf`

**Success**: JBIG2 PDFs process without quality loss. B&W pages decoded and expanded to grayscale correctly. Globals extraction from PDF works. All tests pass.

---

### PR 5.1: Establish nvImageCodec Migration Baseline [COMPLETE]

**Status**: Complete.

**Why**: Before migrating from nvJPEG to nvImageCodec, we need quantitative baseline performance numbers to ensure no regression.

**Background**: nvImageCodec is NVIDIA's unified codec framework that internally uses nvJPEG (via `nvjpeg_ext` plugin) for JPEG support. The current codebase has:
- `nvjpeg_decode.c` (1498 lines) - Direct nvJPEG wrapper with optimizations
- `nvjpeg_encode.c` (894 lines) - Direct nvJPEG encoder wrapper
- `nvimgcodec.c` (1604 lines) - nvImageCodec wrapper with redundant nvJPEG fallback

The fallback architecture is unnecessary since nvImageCodec handles JPEG through its nvjpeg_ext plugin.

**Baseline Results** (50 JPEG images, 8 streams/jobs):

| Benchmark | Per Image | Throughput |
|-----------|-----------|------------|
| Full processing (nvJPEG decode → GPU → nvJPEG encode) | 252.8ms | 3.96 img/s |
| Decode-only (no processing filters) | 40.4ms | 24.8 img/s |

**Key findings**:
- Processing filters dominate runtime (~84% of total time)
- nvJPEG encode provides modest 5% speedup over D2H + CPU encode
- Raw decode/encode throughput is ~25 img/s

**Results documented in**: `doc/nvimgcodec_migration_baseline.md`

**Success criteria**: Documented baseline numbers before any code changes. ✓

---

### PR 5.2: Remove nvJPEG Fallback from nvimgcodec.c [COMPLETE]

**Status**: Implemented and tested.

**Why**: nvImageCodec is the unified framework - the nvJPEG fallback path was unnecessary complexity.

**Implementation**:
- Removed the `#else // !UNPAPER_WITH_NVIMGCODEC` section that wrapped nvJPEG (lines 970-1433)
- Kept only: nvImageCodec implementation + non-CUDA stubs
- Updated `meson.build`: nvImageCodec now required for `-Dcuda=enabled` builds
- Updated `meson_options.txt`: removed redundant `nvimgcodec` option, updated `cuda` description
- Updated to nvImageCodec 0.7.0 API (new function signatures, quality_type/quality_value, futures)

**Files changed**:
- `imageprocess/nvimgcodec.c` - Removed ~460 lines of fallback code, updated to 0.7.0 API
- `meson.build` - nvImageCodec required for CUDA builds, added include paths
- `meson_options.txt` - Removed `nvimgcodec` option

**Result**: nvimgcodec.c dropped from 1604 to ~1160 lines.

**Success**: CUDA build requires nvImageCodec. All 17 CUDA tests pass.

---

### PR 5.3: Add Batch Decode API to nvimgcodec.c [COMPLETE]

**Status**: Implemented and tested.

**Why**: `batch_decode_queue.c` needs a batch decode function. Uses per-image decode internally (batched decode mode was proven slower and is being dropped).

**Implementation**:
- Added `nvimgcodec_decode_batch()` function that acquires N decode states from pool
- Launches parallel decodes on separate CUDA streams (one per state)
- Processes images in chunks when batch size exceeds available states
- Returns count of successful decodes
- Added stub implementation for non-CUDA builds

**Key API** (implemented in `imageprocess/nvimgcodec.h`):
```c
// Batch decode - processes images in parallel using per-image decode
int nvimgcodec_decode_batch(
    const uint8_t *const *data_ptrs,    // Array of image data pointers
    const size_t *sizes,                 // Array of data sizes
    int batch_size,                      // Number of images
    NvImgCodecOutputFormat output_fmt,   // Output format
    NvImgCodecDecodedImage *outputs      // Output array
);
```

**Files changed**:
- `imageprocess/nvimgcodec.h` - Added batch API declaration (~20 lines)
- `imageprocess/nvimgcodec.c` - Added batch implementation (~70 lines) + stub (~10 lines)
- `tests/nvimgcodec_test.c` - Added 3 unit tests (~170 lines)

**Tests**: `tests/nvimgcodec_test.c` - Unit tests for batch decode.
- `test_decode_batch` - Batch decode 3 images
- `test_decode_batch_null_safety` - NULL input handling
- `test_decode_batch_partial` - Mixed valid/NULL entries

**Success**: Batch API works. All 17 CUDA tests pass including 3 new batch decode tests.

---

### PR 5.4: Migrate batch_decode_queue.c to nvimgcodec [COMPLETE]

**Status**: Implemented and tested.

**Why**: Currently uses nvjpeg_decode.c directly. Must use unified nvimgcodec API.

**Implementation**:
- Replaced `#include "nvjpeg_decode.h"` with `#include "nvimgcodec.h"`
- Replaced `nvjpeg_batched_is_ready()` → `nvimgcodec_is_available()`
- Replaced `nvjpeg_decode_batch()` → `nvimgcodec_decode_batch()`
- Updated `NvJpegDecodedImage` → `NvImgCodecDecodedImage`
- Updated single decode fallback to use nvimgcodec API
- Removed `nvjpeg_batched_init()` - nvimgcodec handles initialization internally
- Removed `nvjpeg_batched_cleanup()` - nvimgcodec cleanup is handled separately
- Updated `gpu_pool_owned` to `false` since nvimgcodec allocates per-image (no pool)

**Files changed**:
- `lib/batch_decode_queue.c` - Updated ~80 lines

**Success**: All 17 CUDA tests pass. `bench_batch.py` and `bench_jpeg_pipeline.py` run successfully.

---

### PR 5.5: Migrate decode_queue.c and Remove Redundant Fallback [COMPLETE]

**Status**: Implemented and tested.

**Why**: Had redundant manual fallback to nvjpeg that duplicates nvimgcodec's internal handling.

**Implementation**:
- Removed `#include "imageprocess/nvjpeg_decode.h"` from decode_queue.c
- Removed `goto legacy_nvjpeg` fallback logic (~10 lines)
- Removed `legacy_nvjpeg:` label and ~66 lines of fallback code
- Replaced `nvjpeg_release_completion_event()` with `nvimgcodec_release_completion_event()`
- Updated unpaper.c to use nvimgcodec for decode context:
  - Changed `#include "imageprocess/nvjpeg_decode.h"` to `#include "imageprocess/nvimgcodec.h"`
  - Replaced `nvjpeg_context_init()` with `nvimgcodec_init()`
  - Replaced `nvjpeg_context_cleanup()` with `nvimgcodec_cleanup()`
  - Replaced `nvjpeg_print_stats()` with `nvimgcodec_print_stats()`
  - Removed batch stats printing (nvimgcodec handles stats internally)

**Files changed**:
- `lib/decode_queue.c` - Removed ~70 lines of redundant fallback
- `unpaper.c` - Updated to use nvimgcodec for decode initialization/cleanup/stats

**Success**: decode_queue.c uses only nvimgcodec. All 17 CUDA tests pass.

---

### PR 5.6: Delete nvjpeg_decode.c and nvjpeg_encode.c [COMPLETE]

**Status**: Implemented and tested.

**Why**: No longer needed - nvimgcodec handles everything through nvImageCodec's nvjpeg_ext plugin.

**Implementation**:
- Deleted `imageprocess/nvjpeg_decode.c` (1498 lines)
- Deleted `imageprocess/nvjpeg_decode.h`
- Deleted `imageprocess/nvjpeg_encode.c` (894 lines)
- Deleted `imageprocess/nvjpeg_encode.h`
- Deleted test files: `tests/nvjpeg_decode_test.c`, `tests/nvjpeg_batched_test.c`, `tests/nvjpeg_encode_test.c`
- Updated `meson.build` to remove nvjpeg source files and tests
- Updated `lib/encode_queue.c` to use nvimgcodec exclusively (removed legacy nvJPEG path)
- Updated `unpaper.c` to use nvimgcodec for GPU encode initialization and cleanup
- Updated `tests/nvimgcodec_test.c` to not depend on nvjpeg

**Files changed**:
- `lib/encode_queue.c` - Removed nvjpeg_encode.h include, simplified GPU encode path (~50 lines removed)
- `unpaper.c` - Removed nvjpeg_encode.h include, updated GPU encode init/cleanup (~15 lines simplified)
- `tests/nvimgcodec_test.c` - Removed nvjpeg_decode.h include, updated init/cleanup tests (~20 lines simplified)
- `meson.build` - Removed nvjpeg source files and tests (~70 lines removed)

**Total lines removed**: ~2500+ lines

**Success**: Build succeeds without nvjpeg_*.c files. All 14 CUDA tests pass.

---

### PR 5.7: Verify Performance + Add Async Allocators [COMPLETE]

**Status**: Complete. Performance regression documented; async allocators added as mitigation.

**Why**: Ensure migration didn't regress performance; optimize where possible.

**Benchmark Results** (50 JPEG images, 8 streams/jobs):

| Metric | Baseline (nvJPEG) | HEAD (nvImageCodec) | Regression |
|--------|-------------------|---------------------|------------|
| GPU pipeline | 264.8ms/img | 344.4ms/img | **~30%** |
| Throughput | 3.77 img/s | 2.9 img/s | **-23%** |

**Root Cause Analysis** (via nsys profiling):

| CUDA API Call | Baseline | HEAD | Issue |
|---------------|----------|------|-------|
| cuMemFree_v2 | 58 calls, 2ms | 258 calls, 1.15s | 4.4x more calls, synchronous |
| cuStreamDestroy_v2 | 38 calls, 0.2ms | 138 calls, 252ms | 3.6x more calls |

**Optimization Attempted**:

Added custom async allocators (`cudaMallocAsync`/`cudaFreeAsync`) to nvImageCodec exec_params:
- Device allocator: uses `cudaMallocAsync` with sync fallback
- Pinned allocator: uses `cudaMallocHost`/`cudaFreeHost`

nsys profiling confirms async allocators ARE working (12k+ `cudaMallocAsync` calls). However, nvImageCodec's API-level object creation/destruction still causes synchronous `cuMemFree_v2` overhead that cannot be avoided.

**Other Attempts (reverted as ineffective)**:
- Object reuse via cached_image/cached_code_stream: nvImageCodec's "reuse" is broken (recreates internal resources)
- pre_init=1, skip_pre_sync=1: No measurable improvement
- Deferred cleanup queue: Background thread to destroy CodeStream/Image objects asynchronously. Does not help because `cuMemFree_v2` is GPU-global - it synchronizes the entire GPU regardless of which CPU thread calls it. Moving destroys to a background thread still blocks all GPU streams.

**Root Cause**:

nvImageCodec's architecture requires per-operation CodeStream and Image object creation. The `nvimgcodecCodeStreamCreateFromHostMem` function always creates new internal resources (confirmed in nvImageCodec source: `nvimgcodec_capi.cpp:325`). Object destruction triggers synchronous CUDA memory frees that cannot be avoided with the current API.

**Conclusion**:

The ~30% regression is inherent to nvImageCodec's API design. The async allocators provide partial mitigation for nvJPEG's internal allocations but cannot address the nvImageCodec wrapper overhead. Future options:
1. Accept regression for unified API benefits (JP2 support)
2. Restore direct nvJPEG for JPEG, use nvImageCodec only for JP2

**Implementation**: Custom async allocators added to `nvimgcodec.c` (lines 28-95, 257-258, 317-318).

---

### PR 5.x Summary: nvImageCodec Migration

| PR | Description | Lines Changed | Status |
|----|-------------|---------------|--------|
| 5.1 | Baseline benchmarks | 0 (documentation only) | Complete |
| 5.2 | Remove nvJPEG fallback from nvimgcodec.c | -460 | Complete |
| 5.3 | Add batch decode API to nvimgcodec | +100 | Complete |
| 5.4 | Migrate batch_decode_queue.c | ~100 modified | Complete |
| 5.5 | Migrate decode_queue.c | -70 | Complete |
| 5.6 | Delete nvjpeg_decode.c/nvjpeg_encode.c | -2400 | Complete |
| 5.7 | Verify performance + async allocators | +70 | Complete |

**Net result**: ~2800 lines removed, cleaner architecture, unified codec API.

**Known limitation**: ~30% performance regression vs direct nvJPEG due to nvImageCodec API overhead (per-operation object creation with synchronous cleanup). This is inherent to nvImageCodec's architecture and cannot be avoided without reverting to direct nvJPEG for JPEG.

**Key optimizations in nvimgcodec.c**:

| Optimization | Implementation |
|--------------|----------------|
| Custom async allocators | `cudaMallocAsync`/`cudaFreeAsync` for nvJPEG internals (lines 28-95) |
| Dedicated CUDA streams | Per-state `cudaStream_t` (line 34) |
| Pre-allocated events | `completion_event` in state (line 35) |
| Lock-free state pool | `atomic_compare_exchange` (lines 418-425) |
| Memory alignment | 256-byte pitch alignment (line 632) |

---

### PR 6: GPU PDF Pipeline Integration [COMPLETE]

**Status**: Implemented and tested.

**Why**: Wire GPU decode/encode to PDF pipeline for maximum performance. Both CPU and GPU pipelines now use the full `process_sheet()` processing path.

**Implementation**:
- New `pdf/pdf_pipeline_gpu.c`: orchestrate GPU PDF processing
- Updated `pdf/pdf_pipeline_cpu.c` to use `process_sheet()` instead of duplicating filter logic
- Both pipelines now support: blackfilter, blurfilter, grayfilter, noisefilter, deskew, mask detection, border detection, etc.
- Added `options_init_filter_defaults()` helper function in `lib/options.c` for proper filter parameter initialization
- Fixed backend selection: CPU pipeline calls `image_backend_select(UNPAPER_DEVICE_CPU)`
- GPU pipeline uses `sheet_process_state_set_gpu_decoded_image()` for pre-decoded GPU images
- Auto-selects GPU vs CPU pipeline based on `--device` flag

**Key architectural change**:
```c
// Old approach (duplicated filter logic in each pipeline):
mirror(state.sheet, options->pre_mirror);
blackfilter(state.sheet, bf_params);
noisefilter(state.sheet, ...);
// ... 200+ lines of duplicated code

// New approach (reuse existing infrastructure):
sheet_process_state_set_decoded(&state, page_frame, 0);  // CPU
// OR: sheet_process_state_set_gpu_decoded_image(&state, page_image, 0);  // GPU
process_sheet(&state, &pdf_config);  // All processing handled
```

**Processing path**:
```
CPU: PDF page → MuPDF extract → FFmpeg decode → AVFrame → process_sheet() → JPEG encode → PDF
GPU: PDF page → MuPDF extract → nvImageCodec decode → GPU Image → process_sheet() → nvImageCodec encode → PDF
```

**Tests**: `tests/pdf_pipeline_cpu_test.c` - All tests pass.
- PDF detection tests
- Single-page PDF processing
- Multi-page PDF processing
- Output quality verification

**Build**: `meson setup builddir-cuda-pdf -Dcuda=enabled -Dpdf=enabled && meson compile -C builddir-cuda-pdf`

**Usage**: `unpaper --device=cpu input.pdf output.pdf` or `unpaper --device=cuda input.pdf output.pdf`

**Success**: Full PDF processing pipeline works on both CPU and GPU. All filters and deskew applied correctly.

---

### PR 7: Batch PDF Processing [COMPLETE]

**Status**: Implemented and tested.

**Why**: Multi-page PDFs need parallel processing for throughput.

**Implementation**:
- New `pdf/pdf_page_accumulator.c/.h`: Thread-safe accumulator for ordered page write
  - Collects encoded pages from parallel workers
  - Writes pages to PDF in sequential order (0, 1, 2, ...)
  - Writer thread processes pages as they arrive, waits for missing pages
- New `pdf/pdf_pipeline_batch.c/.h`: Batch PDF processing pipeline
  - Decode producer thread: extracts/decodes pages in parallel
  - Worker pool: processes pages concurrently using `process_sheet()`
  - Page accumulator: ensures ordered PDF output
  - Progress reporting: `Processing page X/Y...`
- Updated `unpaper.c`: Routes to batch pipeline when `--batch` or `-j N` flags used

**Architecture**:
```
PDF pages 0..N → Decode Queue (4-8 slots) → Worker Pool → Page Accumulator → PDF Writer
                     |                           |               |
              (parallel decode)         (parallel process)  (sequential write)
```

**Key API** (implemented in `pdf/pdf_pipeline_batch.h`):
```c
// Configuration
typedef struct {
  int parallelism;        // Number of worker threads (0 = auto)
  int decode_queue_depth; // Decode queue depth (0 = auto: parallelism * 2)
  bool progress;          // Show progress output
  bool use_gpu;           // Use GPU pipeline (requires CUDA)
} PdfBatchConfig;

void pdf_batch_config_init(PdfBatchConfig *config);
int pdf_pipeline_batch_process(const char *input_path, const char *output_path,
                               const Options *options,
                               const SheetProcessConfig *sheet_config,
                               const PdfBatchConfig *batch_config);
bool pdf_pipeline_batch_available(void);
```

**Usage**: `unpaper --batch -j 4 --progress input.pdf output.pdf`

**Tests**: All 5 PDF tests pass. Functional test with 2-page PDF verified.

**Files changed**:
- `pdf/pdf_page_accumulator.h` - Page accumulator API (~90 lines)
- `pdf/pdf_page_accumulator.c` - Thread-safe accumulator implementation (~250 lines)
- `pdf/pdf_pipeline_batch.h` - Batch pipeline API (~70 lines)
- `pdf/pdf_pipeline_batch.c` - Batch pipeline implementation (~760 lines)
- `meson.build` - Added new source files
- `unpaper.c` - Added batch pipeline routing

**Success**: Multi-page PDFs process in parallel with memory-bounded decode queue.

---

### PR 8: Performance Optimization + Polish

**Why**: Maximize throughput, minimize latency.

**Optimizations**:
- Async metadata read (don't block decode)
- Page-level stream assignment (one stream per in-flight page)
- Zero-copy paths: JPEG/JP2 bytes stay in pinned memory through pipeline
- Memory pool for encoded output buffers

**Benchmarks to hit**:
| Metric | Target |
|--------|--------|
| Single page (A4 300dpi color) | <50ms GPU, <200ms CPU |
| 100-page PDF throughput | >50 pages/sec GPU |
| Peak GPU memory | <1GB for 8 workers |

**Tests**: `tools/bench_pdf.py` automated benchmarks. CI performance regression check.

**Success**: Benchmarks met. No sync points in hot path. Stable memory usage.

---

### Testing Strategy

**Test PDFs** (create in `tests/pdf_samples/`):
- `jpeg_color_10page.pdf` - Standard color scans
- `jp2_archival.pdf` - JPEG2000 archival format
- `jbig2_bw_contract.pdf` - B&W document with JBIG2
- `mixed_formats.pdf` - Different formats per page
- `large_100page.pdf` - Throughput stress test

**Golden tests**: Process each test PDF, compare output page-by-page to golden images (existing pytest infrastructure).

**Benchmarks**: `tools/bench_pdf.py` with standardized test corpus, track in CI.

---

### Dependencies (add to meson.build)

```meson
mupdf_dep = dependency('mupdf', required: get_option('pdf'))
jbig2dec_dep = dependency('jbig2dec', required: false)
# nvImageCodec: manual detection (not in pkg-config)
nvimgcodec_dep = cc.find_library('nvimgcodec', required: false)
```

---

## Historical Documentation

For completed PR history, see `doc/CUDA_BACKEND_HISTORY.md`.
