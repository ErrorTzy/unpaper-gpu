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

## Historical Documentation

For completed PR history, see `doc/CUDA_BACKEND_HISTORY.md`.
