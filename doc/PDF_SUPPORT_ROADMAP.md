# PDF Support Roadmap for unpaper

## Executive Summary

This document outlines a PR-by-PR roadmap for adding PDF input/output support to unpaper, with maximum GPU acceleration for scanned document processing.

**Goal**: PDF in → GPU processing → PDF out, with pure GPU pipelines where possible.

---

## Research Findings

### PDF Libraries Comparison

| Library | License | PDF Read | PDF Write | Speed | Raw Image Extract |
|---------|---------|----------|-----------|-------|-------------------|
| **MuPDF** | AGPL v3 | ✓ | ✓ | Fastest (3x Poppler) | ✓ Native bytes |
| **Poppler** | GPL v3 | ✓ | ✗ | Good | ✓ Native bytes |
| **PDFium** | BSD-like | ✓ | ✗ | Good | ✓ Render only |
| **Cairo** | LGPL 2.1 | ✗ | ✓ | Good | N/A |

**Recommendation**: Use **Poppler** for reading + **Cairo** for writing. Both are GPL-compatible with unpaper's MIT license (must document the combined work's license requirements). Alternative: MuPDF for both (requires unpaper to adopt AGPL or purchase commercial license).

### Image Formats in Scanned PDFs

| Format | Usage | GPU Support (nvImageCodec) |
|--------|-------|---------------------------|
| **JPEG** | Most common (color scans) | ✓ Full GPU decode/encode |
| **JPEG2000** | Modern PDFs, archival | ✓ Full GPU decode/encode |
| **JBIG2** | B&W documents | ✗ CPU only |
| **CCITT** | Fax documents | ✗ CPU only |
| **Raw bitmap** | Uncompressed | ✗ CPU only |

### nvImageCodec vs Current nvJPEG

| Feature | nvJPEG (current) | nvImageCodec |
|---------|------------------|--------------|
| JPEG decode/encode | GPU | GPU |
| JPEG2000 decode/encode | ✗ | GPU |
| TIFF decode/encode | ✗ | GPU |
| PNG/BMP/WebP | ✗ | CPU fallback |
| Unified API | No | Yes |
| Dependencies | CUDA Toolkit only | CUDA 12.1+, optional nvJPEG2000 |

**nvImageCodec is the strategic choice** for broader format support and future-proofing.

### Performance Analysis

**Optimal GPU Pipeline** (JPEG/JP2 embedded in PDF):
```
PDF → Extract raw bytes → nvImageCodec GPU decode → GPU process → GPU encode → Embed in PDF
     (CPU, ~5ms/page)    (~2ms/image)              (existing)    (~3ms)       (~5ms/page)
```

**Fallback Pipeline** (JBIG2/CCITT/other):
```
PDF → Extract raw bytes → CPU decode → H2D transfer → GPU process → GPU encode → Embed in PDF
     (CPU, ~5ms/page)     (~20ms)      (~5ms)         (existing)    (~3ms)       (~5ms/page)
```

---

## Architecture Design

### High-Level Data Flow

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                              PDF INPUT LAYER                                 │
├─────────────────────────────────────────────────────────────────────────────┤
│  pdf_reader.c/.h                                                            │
│  ├── Open PDF with Poppler                                                  │
│  ├── Iterate pages                                                          │
│  ├── For each page:                                                         │
│  │   ├── Try extract embedded images (preserves JPEG/JP2 raw bytes)        │
│  │   └── Fallback: render page to bitmap at specified DPI                  │
│  └── Return: PdfPageImage { format, raw_bytes, width, height, dpi }        │
└─────────────────────────────────────────────────────────────────────────────┘
                                      │
                                      ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                            DECODE LAYER                                      │
├─────────────────────────────────────────────────────────────────────────────┤
│  Extend decode_queue.c with PDF source support:                             │
│  ├── JPEG/JPEG2000 raw bytes → nvImageCodec GPU decode → GPU-resident Image│
│  └── Other formats → CPU decode (FFmpeg/jbig2dec) → H2D → GPU-resident Image│
└─────────────────────────────────────────────────────────────────────────────┘
                                      │
                                      ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                       EXISTING GPU PROCESSING PIPELINE                       │
├─────────────────────────────────────────────────────────────────────────────┤
│  sheet_process.c (unchanged)                                                │
│  ├── Filters: blackfilter, blurfilter, noisefilter, grayfilter             │
│  ├── Masks: detect, apply                                                   │
│  ├── Deskew: rotation detection, correction                                │
│  └── Output: processed GPU-resident Image                                   │
└─────────────────────────────────────────────────────────────────────────────┘
                                      │
                                      ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                            ENCODE LAYER                                      │
├─────────────────────────────────────────────────────────────────────────────┤
│  Extend encode_queue.c with PDF destination support:                        │
│  ├── GPU Image → nvImageCodec GPU encode → JPEG/JP2 raw bytes              │
│  └── Submit to PDF writer queue                                             │
└─────────────────────────────────────────────────────────────────────────────┘
                                      │
                                      ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                             PDF OUTPUT LAYER                                 │
├─────────────────────────────────────────────────────────────────────────────┐
│  pdf_writer.c/.h                                                            │
│  ├── Create PDF with Cairo                                                  │
│  ├── For each processed page:                                               │
│  │   ├── Receive encoded JPEG/JP2 bytes from encode queue                  │
│  │   └── Embed as image with cairo_surface_set_mime_data()                 │
│  ├── Preserve original PDF metadata (optional)                              │
│  └── Write to output file                                                   │
└─────────────────────────────────────────────────────────────────────────────┘
```

### New Files

```
unpaper/
├── pdf/
│   ├── pdf_reader.c/.h       # Poppler-based PDF reading
│   ├── pdf_writer.c/.h       # Cairo-based PDF writing
│   └── pdf_job.c/.h          # PDF batch job management
├── imageprocess/
│   ├── nvimgcodec.c/.h       # nvImageCodec wrapper (replaces nvjpeg_*.c)
│   └── nvimgcodec_pool.c/.h  # Decoder/encoder state pool
└── lib/
    └── jbig2_decode.c/.h     # Optional JBIG2 CPU decoder (jbig2dec)
```

---

## PR-by-PR Roadmap

### Phase 1: Foundation (PRs 1-3)

#### PR 1: Add Poppler Dependency and PDF Reader Infrastructure
**Scope**: Add Poppler-glib dependency, create basic PDF reading capability.

**Files**:
- `meson.build` - Add poppler-glib dependency
- `meson_options.txt` - Add `pdf` option (auto/enabled/disabled)
- `pdf/pdf_reader.c/.h` - Basic PDF reading API

**API Design**:
```c
typedef struct {
    int page_count;
    void *poppler_doc;  // Opaque handle
} PdfDocument;

typedef struct {
    int page_index;
    int width, height;
    double dpi_x, dpi_y;
    int image_count;  // Number of embedded images
} PdfPageInfo;

typedef struct {
    enum { PDF_IMG_JPEG, PDF_IMG_JP2, PDF_IMG_JBIG2, PDF_IMG_CCITT, PDF_IMG_RAW } format;
    uint8_t *data;
    size_t data_size;
    int width, height;
    int bits_per_component;
    int components;  // 1=gray, 3=RGB, 4=CMYK
} PdfEmbeddedImage;

// API
PdfDocument *pdf_open(const char *filename, char **error);
void pdf_close(PdfDocument *doc);
int pdf_get_page_count(PdfDocument *doc);
PdfPageInfo pdf_get_page_info(PdfDocument *doc, int page_index);
int pdf_extract_images(PdfDocument *doc, int page_index, PdfEmbeddedImage **images, int *count);
void pdf_free_images(PdfEmbeddedImage *images, int count);
uint8_t *pdf_render_page(PdfDocument *doc, int page_index, int dpi, int *width, int *height, int *stride);
```

**Tests**:
- Unit test: Open sample PDF, verify page count
- Unit test: Extract known embedded JPEG, verify raw bytes match
- Unit test: Render page at 300 DPI, verify dimensions

**Acceptance Criteria**:
- [x] `meson setup` with `-Dpdf=enabled` finds poppler-glib
- [x] Can open multi-page PDF and iterate pages
- [x] Can extract embedded JPEG images in raw form
- [x] Can render pages that have non-extractable images
- [x] Memory leaks checked with valgrind

---

#### PR 2: Add Cairo Dependency and PDF Writer Infrastructure
**Scope**: Add Cairo PDF surface support, create basic PDF writing capability.

**Files**:
- `meson.build` - Add cairo dependency (likely already present via poppler)
- `pdf/pdf_writer.c/.h` - PDF writing API

**API Design**:
```c
typedef struct {
    void *cairo_surface;  // PDF surface
    void *cairo_ctx;      // Drawing context
    char *filename;
    double width_pt, height_pt;  // Current page size
} PdfWriter;

typedef struct {
    enum { PDF_OUT_JPEG, PDF_OUT_JP2 } format;
    uint8_t *data;
    size_t data_size;
    int width, height;
} PdfOutputImage;

// API
PdfWriter *pdf_writer_create(const char *filename, char **error);
void pdf_writer_close(PdfWriter *writer);
int pdf_writer_add_page(PdfWriter *writer, double width_pt, double height_pt);
int pdf_writer_embed_jpeg(PdfWriter *writer, const uint8_t *data, size_t size,
                          int img_width, int img_height);
int pdf_writer_embed_jp2(PdfWriter *writer, const uint8_t *data, size_t size,
                         int img_width, int img_height);
int pdf_writer_finish_page(PdfWriter *writer);
```

**Tests**:
- Unit test: Create single-page PDF with embedded JPEG
- Unit test: Create multi-page PDF
- Unit test: Verify output PDF opens in standard viewers
- Round-trip test: Extract JPEG from PDF A, embed in PDF B, compare

**Acceptance Criteria**:
- [x] Can create valid single-page PDF with embedded JPEG
- [x] Can create valid multi-page PDF
- [x] JPEG data is stored directly (no re-encoding)
- [x] PDF validates with `pdfinfo` and opens in viewers

---

#### PR 3: CLI Integration for PDF I/O
**Scope**: Add command-line support for PDF input and output files.

**Files**:
- `parse.c/.h` - Recognize `.pdf` extension
- `unpaper.c` - Handle PDF files in main loop
- `lib/options.h` - Add PDF-specific options

**New CLI Options**:
```
--pdf-dpi=N          Render DPI for non-extractable pages (default: 300)
--pdf-output-format  Output image format in PDF: jpeg, jp2 (default: jpeg)
--pdf-quality=N      JPEG quality for PDF output (default: 85)
```

**Behavior**:
- Single PDF input → process all pages → single PDF output
- Page range selection: `input.pdf[1-5]` syntax
- Progress reporting per page

**Tests**:
- Integration test: `unpaper input.pdf output.pdf` produces valid PDF
- Integration test: Page range selection works
- Integration test: CPU backend works with PDF

**Acceptance Criteria**:
- [x] `unpaper input.pdf output.pdf` works (CPU mode)
- [x] Page count preserved
- [x] Basic filters work on PDF pages
- [x] Error messages for invalid PDFs are clear

---

### Phase 2: GPU Integration (PRs 4-6)

#### PR 4: nvImageCodec Integration (Replace nvJPEG)
**Scope**: Replace nvJPEG with nvImageCodec for broader format support.

**Files**:
- `meson.build` - Add nvImageCodec dependency
- `imageprocess/nvimgcodec.c/.h` - New unified codec interface
- `imageprocess/nvimgcodec_pool.c/.h` - Decoder/encoder state pooling
- Update: `lib/decode_queue.c` - Use nvImageCodec
- Update: `lib/encode_queue.c` - Use nvImageCodec
- Deprecate: `imageprocess/nvjpeg_decode.c/.h`
- Deprecate: `imageprocess/nvjpeg_encode.c/.h`

**API Design**:
```c
typedef enum {
    NVIMGCODEC_FMT_JPEG,
    NVIMGCODEC_FMT_JPEG2000,
    NVIMGCODEC_FMT_TIFF,
    NVIMGCODEC_FMT_PNG,   // CPU fallback
    NVIMGCODEC_FMT_UNKNOWN
} NvImgCodecFormat;

// Decode raw bytes to GPU-resident image
int nvimgcodec_decode_to_gpu(
    const uint8_t *data, size_t size,
    NvImgCodecFormat format,
    CUstream stream,
    void **gpu_ptr, int *width, int *height, int *pitch,
    CUevent *completion_event
);

// Encode GPU-resident image to raw bytes
int nvimgcodec_encode_from_gpu(
    void *gpu_ptr, int width, int height, int pitch,
    NvImgCodecFormat format,
    int quality,  // 1-100 for JPEG/JP2
    CUstream stream,
    uint8_t **data, size_t *size,
    CUevent *completion_event
);

// Format detection from raw bytes
NvImgCodecFormat nvimgcodec_detect_format(const uint8_t *data, size_t size);
```

**Tests**:
- Unit test: Decode JPEG via nvImageCodec, compare to nvJPEG baseline
- Unit test: Decode JPEG2000 (new capability)
- Unit test: Encode to JPEG, verify quality
- Unit test: Encode to JPEG2000 (new capability)
- Performance test: Throughput comparison vs old nvJPEG path

**Acceptance Criteria**:
- [x] All existing JPEG tests pass with nvImageCodec
- [x] JPEG2000 decode works on GPU
- [x] JPEG2000 encode works on GPU
- [x] Performance is equal or better than nvJPEG for JPEG
- [x] Graceful CPU fallback for unsupported formats

---

#### PR 5: GPU-Accelerated PDF Decode Pipeline
**Scope**: Wire PDF reader to GPU decode path.

**Files**:
- Update: `pdf/pdf_reader.c` - Add GPU decode integration
- Update: `lib/decode_queue.c` - Add PDF page source type
- New: `pdf/pdf_decode_slot.c/.h` - PDF-specific decode slot management

**Data Flow**:
```
PdfDocument
     │
     ├── pdf_extract_images() → PdfEmbeddedImage[]
     │        │
     │        ├── JPEG/JP2 → nvimgcodec_decode_to_gpu() → GPU Image (zero-copy)
     │        │
     │        └── JBIG2/CCITT → jbig2dec/fax → CPU Image → cudaMemcpyAsync → GPU Image
     │
     └── pdf_render_page() (fallback) → CPU bitmap → cudaMemcpyAsync → GPU Image
```

**Tests**:
- Integration test: PDF with JPEG images → GPU decode
- Integration test: PDF with JPEG2000 images → GPU decode
- Integration test: PDF with mixed formats → hybrid decode
- Performance benchmark: Compare PDF decode GPU vs CPU

**Acceptance Criteria**:
- [x] JPEG/JP2 images stay GPU-resident (no D2H/H2D)
- [x] Mixed-format PDFs process correctly
- [x] GPU memory pool integration works
- [x] Stream pool utilized for concurrent decode

---

#### PR 6: GPU-Accelerated PDF Encode Pipeline
**Scope**: Wire GPU encode to PDF writer.

**Files**:
- Update: `pdf/pdf_writer.c` - Accept GPU-encoded bytes
- Update: `lib/encode_queue.c` - Add PDF output support
- New: `pdf/pdf_encode_slot.c/.h` - PDF page accumulation

**Data Flow**:
```
Processed GPU Image
         │
         ├── nvimgcodec_encode_from_gpu(JPEG/JP2)
         │         │
         │         └── Raw JPEG/JP2 bytes (GPU memory)
         │                    │
         │                    └── cudaMemcpyAsync(D2H)
         │                               │
         └───────────────────────────────┘
                              │
                    pdf_writer_embed_jpeg/jp2()
                              │
                        Cairo PDF surface
```

**Tests**:
- Integration test: GPU process → JPEG in PDF
- Integration test: GPU process → JP2 in PDF
- Round-trip test: PDF → GPU process → PDF, compare quality
- Performance benchmark: Full pipeline throughput

**Acceptance Criteria**:
- [x] Output PDF contains directly embedded JPEG/JP2 (no transcoding)
- [x] Quality matches `--pdf-quality` setting
- [x] GPU encode completes asynchronously
- [x] Full GPU pipeline has no unnecessary sync points

---

### Phase 3: Optimization & Polish (PRs 7-9)

#### PR 7: Batch PDF Processing
**Scope**: Optimize for multi-page PDFs with concurrent processing.

**Files**:
- Update: `pdf/pdf_job.c/.h` - PDF batch job management
- Update: `lib/batch_worker.c` - PDF-aware worker logic
- Update: `unpaper.c` - PDF batch mode

**Architecture**:
```
PDF Document (N pages)
        │
        ├── Page 0 ──┐
        ├── Page 1 ──┼── Decode Queue (pre-fetch 4-8 pages)
        ├── Page 2 ──┤         │
        └── ...     ─┘         │
                               ▼
                    Worker Pool (K workers)
                          │ │ │
                          ▼ ▼ ▼
                    Encode Queue (buffer encoded pages)
                               │
                               ▼
                    PDF Writer (sequential page append)
```

**Key Optimizations**:
- Pre-fetch PDF pages to hide I/O latency
- Concurrent GPU decode/process/encode
- Sequential PDF write (required by Cairo)
- Memory-bounded queue depths

**Tests**:
- Benchmark: 100-page PDF processing time
- Stress test: Very large PDFs (1000+ pages)
- Memory test: Peak memory usage scales linearly

**Acceptance Criteria**:
- [x] Multi-page PDFs process in parallel
- [x] Memory usage is bounded
- [x] Progress reporting per page
- [x] 3-5x throughput improvement over sequential

---

#### PR 8: JBIG2 CPU Decoder (Optional)
**Scope**: Add native JBIG2 decode support for B&W scanned documents.

**Files**:
- `meson.build` - Optional jbig2dec dependency
- `lib/jbig2_decode.c/.h` - JBIG2 decode wrapper

**Note**: JBIG2 is CPU-only. This PR improves compatibility with B&W archival PDFs. Skip if targeting color/grayscale documents only.

**Tests**:
- Unit test: Decode sample JBIG2 stream
- Integration test: PDF with JBIG2 images processes correctly

**Acceptance Criteria**:
- [x] JBIG2 images decode without errors
- [x] Falls back gracefully if jbig2dec not available
- [x] Performance acceptable for B&W documents

---

#### PR 9: Documentation and Polish
**Scope**: User documentation, performance tuning guide, example workflows.

**Files**:
- `doc/pdf-processing.rst` - User guide
- `doc/performance-tuning.rst` - Optimization tips
- `README.md` - Update with PDF capabilities
- Example scripts in `tools/`

**Documentation Topics**:
- Supported PDF input types
- Output format selection (JPEG vs JP2)
- Performance expectations
- Memory requirements
- Troubleshooting common issues

**Acceptance Criteria**:
- [x] Clear documentation for PDF workflow
- [x] Performance benchmarks published
- [x] Example scripts work out of the box

---

## Performance Expectations

### Single-Page Latency (A4 @ 300 DPI, color)

| Stage | GPU Pipeline | CPU Pipeline |
|-------|--------------|--------------|
| PDF page extract | 5ms | 5ms |
| Decode (JPEG) | 2ms (GPU) | 15ms (CPU) |
| H2D transfer | 0ms (GPU-resident) | 8ms |
| Processing | 20ms | 140ms |
| Encode (JPEG) | 3ms (GPU) | 25ms (CPU) |
| D2H + PDF embed | 8ms | 0ms |
| **Total** | **~38ms** | **~193ms** |

### Batch Throughput (100-page PDF)

| Mode | Time | Pages/sec |
|------|------|-----------|
| CPU sequential | ~19s | 5.3 |
| GPU sequential | ~4s | 25 |
| GPU parallel (8 workers) | ~1.2s | 83 |

### Memory Requirements

| Component | Per-Page | Total (8 workers) |
|-----------|----------|-------------------|
| GPU image buffer | 32MB | 256MB |
| Decode queue (4 slots) | 128MB | 128MB |
| Encode queue (4 slots) | 128MB | 128MB |
| PDF reader overhead | 10MB | 10MB |
| **Total GPU** | - | **~520MB** |

---

## Risks and Mitigations

### Risk 1: nvImageCodec Availability
**Risk**: nvImageCodec requires CUDA 12.1+ and may not be available on all systems.
**Mitigation**: Keep nvJPEG as fallback, auto-detect nvImageCodec at runtime.

### Risk 2: Poppler API Stability
**Risk**: Poppler's internal API for raw image extraction may change.
**Mitigation**: Use stable glib API where possible, version-pin in meson.

### Risk 3: Cairo JPEG2000 Support
**Risk**: Cairo's JP2 MIME type support varies by version.
**Mitigation**: Detect at runtime, fall back to JPEG if JP2 embedding fails.

### Risk 4: License Compatibility
**Risk**: GPL (Poppler) infects MIT codebase when distributed together.
**Mitigation**: Document license requirements clearly. Alternatively, use PDFium (BSD) for reading + Cairo (LGPL) for writing.

---

## Testing Strategy

### Unit Tests (per PR)
- Each new module has corresponding `tests/test_<module>.c`
- Mock dependencies where needed
- Target: 80%+ code coverage

### Integration Tests
- `tests/pdf_pipeline_tests.py` - Full pipeline tests
- Golden image comparison for processed PDF pages
- Compare GPU vs CPU output quality

### Performance Tests
- `tools/bench_pdf.py` - Benchmark script
- Automated regression detection
- Track: latency, throughput, memory

### Test PDFs
Create `tests/pdf_samples/` with:
- `jpeg_color.pdf` - Color JPEG embedded
- `jp2_grayscale.pdf` - Grayscale JPEG2000
- `jbig2_bw.pdf` - Black & white JBIG2
- `mixed_formats.pdf` - Multiple format types
- `large_100page.pdf` - Multi-page stress test
- `scanned_a4_300dpi.pdf` - Real-world scan

---

## Dependencies Summary

| Dependency | Version | License | Required |
|------------|---------|---------|----------|
| poppler-glib | ≥0.90 | GPL v3 | Yes (PDF read) |
| cairo | ≥1.16 | LGPL 2.1 | Yes (PDF write) |
| nvImageCodec | ≥0.3 | Proprietary (NVIDIA) | Optional (GPU codec) |
| jbig2dec | ≥0.19 | AGPL v3 | Optional (JBIG2) |

---

## Timeline Estimate

| Phase | PRs | Complexity |
|-------|-----|------------|
| Phase 1: Foundation | 1-3 | Medium |
| Phase 2: GPU Integration | 4-6 | High |
| Phase 3: Optimization | 7-9 | Medium |

Each PR should be independently testable and mergeable.

---

## Open Questions

1. **License decision**: Accept GPL from Poppler, or investigate PDFium alternative?
2. **JPEG2000 priority**: Is JP2 output important, or JPEG-only sufficient?
3. **JBIG2 support**: Include jbig2dec (AGPL), or skip B&W PDF support?
4. **Metadata preservation**: Copy PDF metadata (title, author) to output?
5. **Page reordering**: Support `--pages` option for subset/reorder?

---

## References

- [nvImageCodec Documentation](https://docs.nvidia.com/cuda/nvimagecodec/index.html)
- [Poppler API Reference](https://poppler.freedesktop.org/api/glib/)
- [Cairo PDF Surfaces](https://www.cairographics.org/manual/cairo-PDF-Surfaces.html)
- [PDF Image Formats (pdfimages man)](https://manpages.debian.org/testing/poppler-utils/pdfimages.1.en.html)
- [MuPDF C API](https://mupdf.readthedocs.io/en/latest/reference/c/fitz/pixmap.html)
