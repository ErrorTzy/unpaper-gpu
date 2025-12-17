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
│   ├── backend_cuda.c     # CUDA backend implementation
│   ├── image.c/.h         # Image struct, CPU memory
│   ├── image_cuda.c       # GPU residency, dirty flags
│   ├── cuda_runtime.c/.h  # CUDA abstraction (streams, memory)
│   ├── cuda_kernels.cu    # Custom CUDA kernels
│   ├── cuda_mempool.c/.h  # GPU memory pool (images + integrals)
│   ├── cuda_stream_pool.c/.h # CUDA stream pool
│   ├── nvjpeg_decode.c/.h # nvJPEG GPU decode
│   ├── nvjpeg_encode.c/.h # nvJPEG GPU encode
│   ├── npp_wrapper.c/.h   # NPP infrastructure
│   ├── npp_integral.c/.h  # NPP integral image computation
│   ├── opencv_bridge.cpp/.h # OpenCV CUDA bridge (filters)
│   ├── opencv_ops.cpp/.h  # OpenCV CUDA operations
│   ├── blit.c             # Rectangle operations
│   ├── deskew.c           # Rotation detection/correction
│   ├── filters.c          # blackfilter, blurfilter, etc.
│   └── masks.c            # Mask detection, borders
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
│   └── pdf_writer.c/.h    # PDF writing (MuPDF)
└── doc/                   # Documentation
    └── CUDA_BACKEND_HISTORY.md  # Completed PR history
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

### PR 6: GPU PDF Pipeline Integration

**Why**: Wire GPU decode/encode to PDF pipeline for maximum performance.

**Implementation**:
- New `pdf/pdf_pipeline_gpu.c`: orchestrate GPU PDF processing
- Update `parse.c`: add options:
  - `--pdf-quality=high|fast` (high=JP2 lossless, fast=JPEG 85)
  - `--pdf-dpi=N` (for rendered fallback, default 300)
- Wire: PDF extract → nvImageCodec decode → GPU process → nvImageCodec encode → PDF embed
- Auto-select GPU vs CPU pipeline based on `--device` flag

**Pipeline selection**:
| Input Format | `--pdf-quality` | Output Format |
|--------------|-----------------|---------------|
| JPEG | fast | JPEG |
| JPEG | high | JP2 lossless |
| JP2 | fast | JPEG |
| JP2 | high | JP2 (preserve) |
| JBIG2 | fast | Grayscale JPEG |
| JBIG2 | high | 1-bit PNG in PDF |

**Tests**: `unpaper --device=cuda input.pdf output.pdf` works. GPU path used for JPEG/JP2.

**Success**: Full GPU PDF→PDF pipeline works. `--pdf-quality` produces expected output sizes.

---

### PR 7: Batch PDF Processing

**Why**: Multi-page PDFs need parallel processing for throughput.

**Implementation**:
- Extend `BatchQueue` to handle PDF pages as jobs
- Pre-fetch N pages into decode queue (hide I/O latency)
- Parallel GPU processing across pages
- Sequential PDF write (accumulate encoded pages, write in order)
- Progress reporting: `Processing page 5/100...`

**Architecture**:
```
PDF pages 0..N → Decode Queue (4-8 slots) → Worker Pool → Encode Queue → PDF Writer (sequential)
```

**Tests**: 100-page PDF benchmark. Memory stays bounded. Progress accurate.

**Success**: Multi-page PDFs 3-5x faster than sequential. Memory doesn't grow with page count.

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
