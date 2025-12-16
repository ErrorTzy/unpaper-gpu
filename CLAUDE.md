# Repository Guidelines

## Project Structure

```
unpaper/
├── unpaper.c              # Main CLI entry point
├── parse.c/.h             # Command-line parsing
├── file.c                 # Image I/O (FFmpeg decode/encode)
├── imageprocess/          # Image processing
│   ├── backend.c/.h       # Backend vtable (CPU/CUDA dispatch)
│   ├── backend_cuda.c     # CUDA backend implementation
│   ├── image.c/.h         # Image struct, CPU memory
│   ├── image_cuda.c       # GPU residency, dirty flags
│   ├── cuda_runtime.c/.h  # CUDA abstraction (streams, memory)
│   ├── cuda_kernels.cu    # Custom CUDA kernels
│   ├── cuda_mempool.c/.h  # GPU memory pool
│   ├── nvjpeg_decode.c/.h # nvJPEG GPU decode
│   ├── opencv_ops.cpp/.h  # OpenCV CUDA operations
│   ├── blit.c             # Rectangle operations
│   ├── deskew.c           # Rotation detection/correction
│   ├── filters.c          # blackfilter, blurfilter, etc.
│   └── masks.c            # Mask detection, borders
├── lib/                   # Utilities
│   ├── batch.c/.h         # Batch job queue
│   ├── batch_decode_queue.c/.h # Batched decode queue (PR36B)
│   ├── decode_queue.c/.h  # Decode queue with GPU integration
│   ├── encode_queue.c/.h  # Encode queue
│   ├── logging.c/.h       # Verbose output
│   └── options.c/.h       # Options struct
├── tests/                 # Test suite
│   ├── unpaper_tests.py   # Pytest golden tests
│   └── source_images/     # Test inputs
└── doc/                   # Documentation
    └── CUDA_BACKEND_HISTORY.md  # PR1-18 history
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

### Dependencies
- **Required**: FFmpeg libs, Meson, Ninja
- **CUDA**: CUDA toolkit (nvJPEG, NPP), OpenCV 4.x with CUDA
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

## Development Roadmap

### PR Status Summary

| PR | Description | Status |
|----|-------------|--------|
| PR1-18 | CUDA backend foundation | **completed** |
| PR19-27 | Native batch processing | **completed** |
| PR35 | nvJPEG decode infrastructure | **completed** |
| PR36 | nvJPEG decode queue integration | **completed** |
| PR36A | nvjpegDecodeBatched infrastructure | **completed** |
| PR36B | Batch-oriented decode queue | **completed** |
| PR36C | Performance validation | planned |
| PR37 | nvJPEG encode | planned |
| PR38 | Full GPU pipeline | planned |

---

## GPU Batch Scaling Pipeline (PR36-PR38)

### Problem: Per-Image Decode Cannot Scale

Testing revealed that per-image `nvjpegDecode()` **cannot achieve >~1x scaling** regardless of optimizations:

| Configuration | 1 Stream | 8 Streams | Scaling |
|--------------|----------|-----------|---------|
| JPEG decode-only | 1.1s | 1.1s | **~1x** |
| JPEG full processing | 19.1s | 14.2s | **1.35x** |

**Root cause**: `nvjpegDecode()` performs internal `cudaMalloc` calls that serialize across streams. Even with custom stream-ordered allocators, the GPU Huffman decode phase serializes.

**Solution**: Migrate to `nvjpegDecodeBatched()` which decodes multiple images in a single API call with a single sync point.

### Architecture Transformation

**Current (per-image, serialized):**
```
Producer 0: fread→nvjpegDecode→sync    ← SERIALIZED
Producer 1: fread→nvjpegDecode→sync
...
Workers pull from decode queue
```

**Target (batch-collect-decode-distribute):**
```
Phase 1 - Collect: I/O threads read JPEG files in parallel
Phase 2 - Decode:  nvjpegDecodeBatched() - ONE sync for ALL images
Phase 3 - Distribute: Workers process decoded images in parallel
```

---

### PR36A: Batched Decode Infrastructure (COMPLETED)

**Motivation**: Provide `nvjpegDecodeBatched()` wrapper that PR36B can use.

**Scope**:
- `nvjpeg_batched_init()`: Initialize batched decoder with buffer pool
- `nvjpeg_decode_batch()`: Decode array of JPEG data pointers
- Pre-allocated GPU buffer pool with 256-byte pitch alignment
- Fallback to single-image decode if batched API fails

**API:**
```c
bool nvjpeg_batched_init(int max_batch_size, int max_width, int max_height,
                         NvJpegOutputFormat format);
int nvjpeg_decode_batch(const uint8_t *const *jpeg_data, const size_t *jpeg_sizes,
                        int batch_size, NvJpegDecodedImage *outputs);
void nvjpeg_batched_cleanup(void);
```

**Key implementation notes:**
- `nvjpegDecodeBatchedInitialize()`: Use `max_cpu_threads=1` (0 causes INVALID_PARAMETER)
- `nvjpegDecodeBatchedSupported()`: Returns 0 for supported, non-zero for unsupported
- Statistics tracking: total calls, images decoded, failures

**Files:** `imageprocess/nvjpeg_decode.c/.h`, `tests/nvjpeg_batched_test.c`

**Acceptance:** ✓ Batched decode works, output matches single-image decode, no memory leaks

---

### PR36B: Batch-Oriented Decode Queue (COMPLETED)

**Motivation**: Current decode_queue uses per-image nvjpegDecode() which serializes. Must replace with batch-collect-decode-distribute architecture to achieve scaling.

**Scope**:
- New `BatchDecodeQueue` that collects JPEG data first, then batch decodes
- Parallel file I/O threads for collection phase
- Chunked processing (8 images at a time) for memory efficiency
- Integration with existing `BatchWorkerContext`

**Architecture Change:**
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

**Challenges and Solutions:**

1. **Memory constraints**: 32 images × 4000×6000 × 3ch = 2.3GB GPU RAM
   - Solution: Chunk-based processing (BATCH_DECODE_CHUNK_SIZE = 8)

2. **Mixed formats**: nvjpegDecodeBatched requires single format
   - Solution: Always decode to RGB, convert grayscale after (negligible overhead)

3. **Non-JPEG files**: PNG must use FFmpeg CPU decode
   - Solution: Pre-filter by extension, process JPEG batches separately

**Files:**
- `lib/batch_decode_queue.c/.h` (new)
- `lib/batch_worker.c` - integrate new queue
- `unpaper.c` - initialize new queue type

**Implementation Details:**
- `BatchDecodeQueue` struct with orchestrator thread and I/O thread pool
- Lock-free slot acquisition for output queue
- Statistics tracking for I/O time, decode time, batch calls
- Automatic fallback to single-image nvjpegDecode when batched API fails
- Graceful fallback to legacy DecodeQueue if batch queue init fails

**API:**
```c
BatchDecodeQueue *batch_decode_queue_create(size_t queue_depth, int num_io_threads,
                                             int max_width, int max_height);
void batch_decode_queue_enable_gpu(BatchDecodeQueue *queue, bool enable);
bool batch_decode_queue_start(BatchDecodeQueue *queue, BatchQueue *batch_queue,
                               const Options *options);
BatchDecodedImage *batch_decode_queue_get(BatchDecodeQueue *queue,
                                           int job_index, int input_index);
void batch_decode_queue_release(BatchDecodeQueue *queue, BatchDecodedImage *image);
void batch_decode_queue_stop(BatchDecodeQueue *queue);
void batch_decode_queue_destroy(BatchDecodeQueue *queue);
```

**Acceptance:** ✓ All criteria met:
- ✓ JPEG batch processing works end-to-end
- ✓ Mixed JPEG+PNG batches handled correctly
- ✓ Memory usage bounded (chunked processing)
- ✓ No deadlocks or race conditions
- ✓ All 14 tests pass (including pytest suite)

---

### PR36C: Performance Validation (PLANNED)

**Motivation**: Verify that batched decode architecture achieves the target scaling improvement.

**Scope**:
- Performance benchmarks comparing old vs new architecture
- Nsight Systems profiling to verify single sync point per chunk
- Tune chunk size for optimal throughput
- Update `bench_batch.py` with decode mode selection

**Verification:**
```bash
# Compare old vs new
./builddir-cuda/unpaper --batch --device=cuda --cuda-streams=8 \
    --decode-mode=per-image input%02d.jpg output%02d.pbm   # Old

./builddir-cuda/unpaper --batch --device=cuda --cuda-streams=8 \
    --decode-mode=batched input%02d.jpg output%02d.pbm     # New
```

**Acceptance:**
- JPEG decode-only: **≥3x scaling** with 8 streams (up from ~1x)
- Single sync point per chunk verified via Nsight
- No regression for non-JPEG files
- All pytest tests pass

---

### PR37: nvJPEG GPU Encode (PLANNED)

**Motivation**: Currently, output encoding uses FFmpeg which requires D2H transfer. GPU-resident encoding eliminates this transfer for JPEG output.

**Scope**:
- Add `nvjpeg_encode_from_gpu()` in `nvjpeg_encode.c/.h`
- Batched encoding support for multiple images
- Quality parameter mapping
- Integration with encode_queue

**API:**
```c
bool nvjpeg_encode_init(int quality, int max_width, int max_height);
int nvjpeg_encode_batch(const NvJpegDecodedImage *inputs, int batch_size,
                        uint8_t **jpeg_data, size_t *jpeg_sizes);
void nvjpeg_encode_cleanup(void);
```

**Files:** `imageprocess/nvjpeg_encode.c/.h`, `tests/nvjpeg_encode_test.c`

**Acceptance:**
- GPU-resident JPEG encoding works
- Output quality matches FFmpeg encoder
- Saves ~10ms/image (D2H transfer eliminated)
- Integration with encode_queue

---

### PR38: Full GPU Pipeline (PLANNED)

**Motivation**: Complete JPEG→processing→JPEG pipeline where image data never leaves GPU memory, eliminating all H2D/D2H transfers.

**Scope**:
- Connect nvJPEG decode → processing → nvJPEG encode pipeline
- Zero-copy path for JPEG-to-JPEG workflows
- Fallback to CPU for non-JPEG formats
- CLI flag: `--gpu-pipeline` for JPEG-only mode

**Pipeline:**
```
JPEG file → [nvjpegDecode] → GPU buffer → [processing] → GPU buffer → [nvjpegEncode] → JPEG file
                         ↑                                        ↓
                    No H2D transfer                          No D2H transfer
```

**Files:**
- `imageprocess/gpu_pipeline.c/.h` (new)
- `unpaper.c` - add `--gpu-pipeline` flag
- `tools/bench_jpeg_pipeline.py` (new)

**Acceptance:**
- Full JPEG→JPEG processing without CPU memory touch
- **Target**: <500ms single image, <5s for 50 images (8 streams)
- Graceful fallback for non-JPEG input/output

---

### Performance Targets Summary

| PR | Component | Current | Target | Approach |
|----|-----------|---------|--------|----------|
| PR36B-C | Decode pipeline | ~1x | **≥3x** | nvjpegDecodeBatched |
| PR37 | Encode | N/A | -10ms/img | nvJPEG GPU encode |
| PR38 | Full pipeline | N/A | <500ms/img | Zero-copy GPU path |

---

## nvJPEG API Notes

### Creating Handle
```c
// With custom allocators (recommended for stream-ordered allocation):
nvjpegCreateExV2(NVJPEG_BACKEND_GPU_HYBRID, &dev_alloc, &pin_alloc, flags, &handle);

// Simple (uses default allocators):
nvjpegCreateSimple(&handle);
```

### Batched Decode
```c
nvjpegJpegStateCreate(handle, &state);
nvjpegDecodeBatchedInitialize(handle, state, batch_size, 1, NVJPEG_OUTPUT_RGBI);
nvjpegDecodeBatched(handle, state, data_ptrs, sizes, outputs, stream);
cudaStreamSynchronize(stream);  // Single sync for entire batch
```

### Thread Safety
- `nvjpegHandle_t`: Thread-safe, one per process
- `nvjpegJpegState_t`: NOT thread-safe, one per stream
- Each CUDA stream needs its own nvJPEG state for parallelism

### Backend Selection
- `NVJPEG_BACKEND_GPU_HYBRID`: Uses CUDA SMs (all GPUs)
- `NVJPEG_BACKEND_HARDWARE`: Uses dedicated decoder (A100/H100 only, ~20x faster)

---

## Historical Documentation

For completed CUDA backend history (PR1-PR18), see `doc/CUDA_BACKEND_HISTORY.md`.
