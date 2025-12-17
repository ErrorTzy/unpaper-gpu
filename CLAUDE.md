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
│   ├── nvjpeg_encode.c/.h # nvJPEG GPU encode
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
| PR36C | Performance validation | **completed** |
| PR37 | nvJPEG encode | **completed** |
| PR38 | Full GPU pipeline | **completed** |
| PR39 | Async stream scaling | **in progress** |

---

## GPU Stream Scaling Status

### Current State: Stream Scaling Does NOT Work

Testing revealed that **neither decode mode achieves stream scaling**:

| Mode | 1 Stream | 8 Streams | Scaling | Expected |
|------|----------|-----------|---------|----------|
| Decode-only (batched) | 1504ms | 1517ms | **0.99x** | 4x+ |
| Decode-only (per-image) | 1249ms | 1263ms | **0.99x** | 4x+ |
| Full pipeline (batched) | 3030ms | 2520ms | **1.20x** | 4x+ |
| Full pipeline (per-image) | 2757ms | 2536ms | **1.09x** | 4x+ |

**Root cause**: `cudaStreamSynchronize()` calls after each decode/encode operation serialize GPU work on the CPU side. Even with 8 CUDA streams, only ONE stream is ever active because the CPU blocks after each operation.

### Per-Image vs Batched Decode

**Per-image decode is recommended** (now the default):
- 20% faster than batched (1249ms vs 1504ms for decode-only)
- Simpler architecture, easier to optimize
- Each worker can operate independently on its own stream
- Amenable to async/event-based fixes

**Batched decode has architectural issues**:
- Sequential phases: collect ALL files → decode ALL → distribute
- Workers cannot start until all decoding completes
- No overlap possible between decode and processing
- Mutex contention when distributing decoded images

### Bottleneck Locations

| File | Line | Issue |
|------|------|-------|
| `nvjpeg_decode.c` | 1114 | `cudaStreamSynchronize()` in batch fallback path |
| `nvjpeg_encode.c` | 566 | `cudaStreamSynchronize()` after each encode |
| `batch_decode_queue.c` | 776-782 | Blocking wait for all files (batched mode) |
| `decode_queue.c` | 360-374 | **Job-specific waiting** (see below) |

### Key Discovery: Job-Specific Waiting Pattern

The per-image decode queue (`decode_queue.c`) has event-based async tracking implemented,
BUT workers wait for **specific images** by job_index, not any available image:

```c
// find_ready_slot() at line 360-374
if (img->job_index == job_index && img->input_index == input_index) {
  // Only return THIS specific image, wait otherwise
}
```

This means:
- If Worker 0 wants job 0, but jobs 1-7 finish first, Worker 0 blocks
- Workers cannot process jobs out of order
- Decode parallelism is wasted due to in-order consumption

**Impact**: Even with 8 parallel async decodes, workers serialize waiting for their specific jobs.

### Planned Fixes for Stream Scaling (PR39)

**Phase 1: Already Implemented**
- Event-based async decode in `nvjpeg_decode_to_gpu()` (events recorded, sync deferred)
- Event pool to avoid cudaEventCreate/Destroy overhead
- Per-stream nvJPEG states with dedicated CUDA streams

**Phase 2: Required Changes**

1. **Work-stealing queue** - Workers take ANY available decoded image, not specific jobs:
   ```c
   // BEFORE (job-specific):
   find_ready_slot(queue, job_index, input_index);  // Wait for THIS job

   // AFTER (work-stealing):
   find_any_ready_slot(queue);  // Take any available, sort output later
   ```

2. **Remove encode sync** - Use events in `nvjpeg_encode.c:566`

3. **Pipeline overlap** - Allow decode, process, encode to run concurrently:
   ```
   Stream 0: [Decode A] → [Process A] → [Encode A]
   Stream 1:    [Decode B] → [Process B] → [Encode B]
   Stream 2:       [Decode C] → [Process C] → [Encode C]
   ```

Target: 8 streams should achieve 4x+ speedup over 1 stream.

---

## GPU Decode Infrastructure (PR36-PR38)

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

### PR36C: Performance Validation (COMPLETED)

**Motivation**: Verify that batched decode architecture achieves the target scaling improvement.

**Scope**:
- Performance benchmarks comparing old vs new architecture
- Bug fixes for memory management and deadlocks
- Update `bench_batch.py` and `bench_a1.py` with decode mode selection

**Bug Fixes:**

1. **Pool Buffer cudaFree Bug** (`lib/batch_decode_queue.c`):
   - `batch_decode_queue_release()` was calling `cudaFree()` on nvjpegDecodeBatched pool buffers
   - Each `cudaFree()` is synchronous (~10-50ms overhead per call)
   - With 50 images: 50 × sync overhead = 500-2500ms extra latency
   - **Fix**: Added `gpu_pool_owned` flag to `BatchDecodedImage` struct
   - Only free non-pool-managed GPU memory in release path

2. **Multi-Stream Deadlock** (`unpaper.c`):
   - Queue depth was `parallelism * 2` (~16 slots for 8 workers)
   - With "decode all at once", orchestrator places images in I/O completion order
   - Workers wait for specific `job_index` images
   - If queue full before worker's image placed → DEADLOCK
   - **Fix**: Ensure `queue_depth >= batch_queue.count` for batch decode

**Implementation:**

1. **Pipeline Auto-Selection** (replaces manual CLI options):
   - Per-image decode is always used (faster than batched)
   - Batched decode code retained internally for future experimentation

2. **Benchmark Tools**:
   - `tools/bench_batch.py`: Performance benchmarks
   - `tools/bench_a1.py`: A1 benchmark suite

**Files Modified:**
- `lib/batch_decode_queue.h` - Added `gpu_pool_owned` field to `BatchDecodedImage`
- `lib/batch_decode_queue.c` - Fixed pool buffer handling, queue depth logic (internal only)
- `unpaper.c` - Simplified pipeline selection with auto-detection

**Performance Results:**

| Benchmark | Batched | Per-Image | Difference |
|-----------|---------|-----------|------------|
| Decode-only (30 images) | 1504ms | 1249ms | **Per-image 20% faster** |
| Full pipeline (30 images) | 2520ms | 2536ms | ~comparable |

**Scaling Analysis** (with `--no-processing`):
- Neither mode scales with additional streams (0.99x at 8 streams)
- Root cause: `cudaStreamSynchronize()` after each decode serializes GPU work
- Per-image is faster due to less coordination overhead

**Acceptance:** ✓ All criteria met:
- ✓ Pool buffer memory leak fixed
- ✓ Multi-stream deadlock fixed
- ✓ All 14 tests pass
- ⚠ Stream scaling NOT achieved (documented as known issue)

---

### PR37: nvJPEG GPU Encode (COMPLETED)

**Motivation**: Currently, output encoding uses FFmpeg which requires D2H transfer. GPU-resident encoding eliminates this transfer for JPEG output.

**Scope**:
- Add `nvjpeg_encode_from_gpu()` in `nvjpeg_encode.c/.h`
- Per-stream encoder state pool for concurrent encoding
- Quality parameter mapping (1-100)
- Chroma subsampling control (444/422/420/gray)
- Integration with encode_queue for JPEG outputs

**Implementation:**

1. **Encoder State Pool** (`imageprocess/nvjpeg_encode.c`):
   - Pre-allocated `NvJpegEncoderState` pool (one per CUDA stream)
   - Lock-free acquisition via atomic compare-exchange
   - Shared nvJPEG handle with decode context (avoids duplicate handle)

2. **Single Image Encode**:
   - `nvjpeg_encode_from_gpu()`: Encode GPU buffer directly to JPEG
   - `nvjpeg_encode_gpu_to_file()`: Convenience wrapper for file output
   - Grayscale support via RGB conversion + CSS_GRAY subsampling

3. **Batched Encode** (for PR38):
   - `nvjpeg_encode_batch()`: Encode multiple images using encoder pool
   - Note: nvJPEG doesn't have true batched encode API like decode,
     but concurrent single-image encodes achieve good parallelism

4. **encode_queue Integration** (`lib/encode_queue.c`):
   - `encode_queue_enable_gpu()`: Enable GPU encoding for JPEG outputs
   - `encode_queue_submit_gpu()`: Submit GPU-resident image for encoding
   - Automatic fallback to D2H + FFmpeg for non-JPEG outputs

**API:**
```c
// Initialization
bool nvjpeg_encode_init(int num_encoders, int quality,
                        NvJpegEncodeSubsampling subsampling);
void nvjpeg_encode_cleanup(void);

// State pool
NvJpegEncoderState *nvjpeg_encode_acquire_state(void);
void nvjpeg_encode_release_state(NvJpegEncoderState *state);

// Single image encode
bool nvjpeg_encode_from_gpu(const void *gpu_ptr, size_t pitch,
                            int width, int height, NvJpegEncodeFormat format,
                            NvJpegEncoderState *state, UnpaperCudaStream *stream,
                            NvJpegEncodedImage *out);
bool nvjpeg_encode_gpu_to_file(..., const char *filename);

// Batched encode
bool nvjpeg_encode_batch_init(int max_batch_size, int max_width, int max_height);
int nvjpeg_encode_batch(const void *const *gpu_ptrs, ...);

// encode_queue GPU support
void encode_queue_enable_gpu(EncodeQueue *queue, bool enable, int quality);
bool encode_queue_submit_gpu(EncodeQueue *queue, void *gpu_ptr, ...);
```

**Files:**
- `imageprocess/nvjpeg_encode.c/.h` (new)
- `tests/nvjpeg_encode_test.c` (new)
- `lib/encode_queue.c/.h` (updated)
- `meson.build` (updated)

**Performance:**
- RGB encode: ~1.4MB output for 2480x3507 image at quality 85
- Quality 50 vs 95: 917KB vs 2MB (expected JPEG compression behavior)
- Grayscale: Uses RGB conversion + CSS_GRAY (not optimal, but functional)

**Acceptance:** ✓ All criteria met:
- ✓ GPU-resident JPEG encoding works (RGB and grayscale)
- ✓ Quality control (1-100) working
- ✓ Concurrent encoding via state pool
- ✓ Integration with encode_queue
- ✓ All 15 tests pass

---

### PR38: Full GPU Pipeline (COMPLETED)

**Motivation**: Complete JPEG→processing→JPEG pipeline where image data never leaves GPU memory, eliminating all H2D/D2H transfers.

**Scope**:
- Connect nvJPEG decode → processing → nvJPEG encode pipeline
- Zero-copy path for JPEG-to-JPEG workflows
- Fallback to CPU for non-JPEG formats
- **Auto-detected** based on input/output file formats

**Pipeline:**
```
JPEG file → [nvjpegDecode] → GPU buffer → [processing] → GPU buffer → [nvjpegEncode] → JPEG file
                         ↑                                        ↓
                    No H2D transfer                          No D2H transfer
```

**Implementation:**

1. **Auto-Detection** (`unpaper.c`):
   - GPU backend auto-enables nvJPEG decode for JPEG inputs
   - GPU backend auto-enables nvJPEG encode when any output is JPEG
   - `--jpeg-quality=N`: JPEG output quality (1-100, default 85)

2. **GPU Encode Path** (`sheet_process.c`):
   - Detects when GPU pipeline can be used:
     - `encode_queue_gpu_enabled()` returns true
     - `image_is_gpu_resident()` returns true (GPU buffer has valid data)
   - Skips `image_ensure_cpu()` D2H transfer
   - Submits GPU pointer directly via `encode_queue_submit_gpu()`
   - Automatic fallback to CPU path if conditions not met

3. **Image GPU Access** (`imageprocess/image.h/.c`):
   - `image_get_gpu_ptr()`: Get GPU device pointer from Image
   - `image_get_gpu_pitch()`: Get GPU pitch from Image
   - Used by sheet_process.c for direct GPU encode submission

4. **Integration** (`unpaper.c`):
   - Initializes nvJPEG encode when JPEG outputs detected
   - Configures encode queue with GPU encoding
   - Cleanup of nvJPEG encode resources on shutdown

**Files Modified:**
- `lib/options.h/.c` - Added `jpeg_quality` field
- `unpaper.c` - Auto-detection logic, nvJPEG encode init/cleanup
- `sheet_process.c` - GPU encode path detection and submission
- `imageprocess/image.h/.c` - GPU pointer access functions
- `tools/bench_jpeg_pipeline.py` - GPU pipeline benchmark

**Usage:**
```bash
# Full GPU pipeline with JPEG output (auto-detected)
./builddir-cuda/unpaper --batch --device=cuda --jpeg-quality 85 \
    input%02d.jpg output%02d.jpg

# Benchmark GPU pipeline
python tools/bench_jpeg_pipeline.py --images 50
```

**Acceptance Criteria:**
- [x] Full JPEG→JPEG processing without CPU memory touch
- [x] D2H transfer eliminated for JPEG outputs (~6ms/image saved)
- [x] Graceful fallback for non-JPEG input/output
- [x] All 15 tests pass

**Benchmark Results** (2480x3507 test images, 8 streams):
| Metric | Standard Path | GPU Pipeline | Improvement |
|--------|---------------|--------------|-------------|
| 1 image (total) | 1994ms | 1999ms | ~same (startup dominated) |
| 50 images (total) | 21745ms | 21449ms | 1.4% faster |
| Per-image | 434.9ms | 429.0ms | ~6ms/image saved |
| Output size | 13929 KB/img (PBM) | 1298 KB/img (JPEG) | 10.7x smaller |

**Note**: The D2H transfer elimination saves ~6ms/image. The total processing time
is dominated by image processing (filters, deskew, etc.), not memory transfers.
For JPEG output workflows, the GPU pipeline also provides ~10x smaller output files.

---

### Performance Targets Summary

| PR | Component | Status | Result | Notes |
|----|-----------|--------|--------|-------|
| PR36B-C | Decode pipeline | **partial** | ~1.2x scaling | Stream scaling blocked by cudaStreamSynchronize |
| PR37 | Encode | **achieved** | GPU encode working | nvJPEG encoder pool |
| PR38 | Full pipeline | **achieved** | ~6ms/img saved | Zero-copy GPU path |
| PR39 | Stream scaling | **TODO** | Target: 4x+ | Remove blocking syncs, use events |

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
