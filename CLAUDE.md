# Repository Guidelines

## Project Structure & Module Organization

- `unpaper.c`, `parse.c/.h`, `file.c`: core CLI, option parsing, and file I/O.
- `imageprocess/`: image-processing primitives and algorithms (deskew, filters, masks, etc.).
- `lib/`: shared utilities (logging, options, physical dimension helpers).
- `tests/`: `pytest` suite plus `source_images/` inputs and `golden_images/` expected outputs.
- `doc/`: Sphinx sources used to build the `unpaper(1)` man page and additional Markdown docs.

## Build, Test, and Development Commands

This project uses Meson (with Ninja).

- Configure: `meson setup builddir/ --buildtype=debugoptimized`
- Build: `meson compile -C builddir/`
- Run tests: `meson test -C builddir/ -v`
- Build man page: `meson compile -C builddir/ man`
- Install (staged): `DESTDIR=/tmp/unpaper-staging meson install -C builddir/`

System dependencies include FFmpeg libraries (`libavformat`, `libavcodec`, `libavutil`) and Python packages in `requirements.txt` (Meson, Sphinx, pytest, Pillow).

## Coding Style & Naming Conventions

- Language: C11 (see `meson.build`).
- Indentation: 2 spaces; LF line endings; trim trailing whitespace (see `.editorconfig`).
- Formatting: `clang-format` is enforced via `pre-commit` (no repo-specific config file).
- Keep changes focused: prefer small helpers in `lib/` over duplicating logic across modules.

## Testing Guidelines

- Tests are `pytest`-based (`tests/unpaper_tests.py`) and compare outputs to golden images.
- When changing image-processing behavior, update/extend `tests/source_images/` and `tests/golden_images/` together and keep test output deterministic.
- Prefer adding a focused regression case over broad rewrites.

## Commit & Pull Request Guidelines

- Commits: follow existing history—short, imperative subject lines; optional scope prefixes like `tests: ...`.
- PRs: describe the behavioral change, include reproduction steps and relevant CLI flags, and note any golden image updates. Ensure `meson test -C builddir/` and `pre-commit run -a` are clean.

## Licensing & Compliance

Files use SPDX headers and the project uses REUSE tooling. Add SPDX headers to new files and validate with `reuse lint` (via `pre-commit`).

## GPU/CUDA Backend Rewrite Plan

Goal: add a fully functioning CUDA backend with `--device=cpu|cuda` such that CPU remains the default and GPU mode preserves the same CLI semantics and output behavior (within the project’s test tolerances unless explicitly tightened).

### Non-negotiables

- Keep `--device=cpu` behavior identical to today.
- `--device=cuda` must support the same flags and processing steps as CPU mode (no “partial pipeline” or silent fallbacks).
- In CUDA mode, minimize CPU↔GPU transfers: upload once after `loadImage()`, keep processing GPU-resident, download once before `saveImage()` (debug outputs may force extra downloads).
- Build must remain CPU-only by default; CUDA support is optional and should be feature-detected at build time.

### High-level approach

- Keep the existing CLI/options/tests pipeline (`unpaper.c`, `parse.c/.h`) and implement a second image-processing backend.
- Refactor `imageprocess/` entry points to dispatch through a backend vtable selected by `--device`.
- Extend the `Image` abstraction to support GPU residency while keeping CPU `AVFrame*` as the I/O anchor.

### Backend boundaries (API surface)

Introduce a backend interface covering all operations used by `unpaper.c`, including (names approximate):

- Image transforms: `stretch_and_replace`, `resize_and_replace`, `flip_rotate_90`, `mirror`, `shift_image`
- Blit/utility: `wipe_rectangle`, `copy_rectangle`, `center_image`
- Masks/borders/wipes: `apply_masks`, `apply_wipes`, `apply_border`, `detect_masks`, `align_mask`, `detect_border`
- Filters: `blackfilter`, `blurfilter`, `noisefilter`, `grayfilter`
- Deskew: `detect_rotation`, `deskew`

CPU backend uses existing code; CUDA backend provides equivalent implementations.

### `Image` and memory model

- Keep `Image.frame` (`AVFrame*`) for decoded/encoded CPU frames and metadata.
- Add CUDA-owned storage for pixel data plus sync flags (CPU dirty / GPU dirty).
- Provide helpers:
  - `image_ensure_cuda(Image*)` (upload/convert as needed)
  - `image_ensure_cpu(Image*)` (download/convert as needed)

### CUDA implementation strategy (parity-first)

Implement CUDA in layers, validating against CPU behavior at each step:

1. CUDA runtime scaffolding:
   - context/stream init, error handling, allocation helpers
   - support current pixel formats used by unpaper (`GRAY8`, `RGB24`, `MONO*`, `Y400A`)
2. GPU “primitives” first (unblocks most of pipeline):
   - fill/wipe rectangle, copy rectangle, mirror, shift, rotate90
   - stretch/resize with NN/bilinear/bicubic matching CPU coordinate mapping/clamping
3. Filters:
   - implement GPU kernels for per-pixel / local-neighborhood ops
   - avoid in-place update hazards (use two-pass where required to match CPU semantics)
4. Mask/border detection:
   - compute scan statistics on GPU (parallel reductions), keep small decision logic on CPU if it helps parity/determinism
5. Deskew:
   - parity-first: compute per-angle/per-depth metrics on GPU, run selection logic on CPU
   - apply final rotation/warp on GPU

Only after parity is demonstrated, optimize (kernel fusion, integral images for window sums, fewer intermediates, stream overlap).

### Build system changes (Meson)

- Add an optional CUDA feature (build option like `-Dcuda=true/false`).
- When enabled, compile/link CUDA sources and expose capability to the binary.
- When disabled or not available, `--device=cuda` must fail with a clear error message (or support an explicit `--device=auto` fallback if added).

### Tests (required)

- Extend `tests/unpaper_tests.py` to run key cases under both `--device=cpu` and `--device=cuda`.
- Ensure CUDA mode is deterministic run-to-run; avoid nondeterministic reductions unless ordering is fixed.
- Keep golden comparisons stable; update golden images only if the project agrees on the acceptable behavior change.

### PR-by-PR roadmap (execute in order)

This section is the execution checklist for adding CUDA support while keeping CPU behavior unchanged. Each PR should be reviewable, keep diffs focused, and include a clear acceptance gate (tests + behavior).

**PR 1: Add `--device` CLI option (CPU-only)**

- Status: completed (2025-12-12)

- Scope:
  - Add `--device=cpu|cuda` parsing (default `cpu`) and propagate into the options struct.
  - If `--device=cuda` is requested but CUDA support is not compiled in, fail fast with a clear error.
- Tests:
  - Extend `tests/unpaper_tests.py` with a small CLI-argument test that `--device=cpu` works and `--device=cuda` errors when CUDA is unavailable.
  - Run `meson test -C builddir/ -v`.
- Acceptance:
  - `--device` appears in `--help`.
  - No output changes for default runs (CPU path untouched).

**PR 2: Introduce backend vtable + CPU backend (no behavior change)**

- Status: completed (2025-12-12)

- Scope:
  - Add a backend interface (vtable) covering every operation called from `unpaper.c`:
    `wipe_rectangle`, `copy_rectangle`, `center_image`, `stretch_and_replace`,
    `resize_and_replace`, `flip_rotate_90`, `mirror`, `shift_image`,
    `apply_masks`, `apply_wipes`, `apply_border`, `detect_masks`, `align_mask`,
    `detect_border`, `blackfilter`, `blurfilter`, `noisefilter`, `grayfilter`,
    `detect_rotation`, `deskew`.
  - Refactor `imageprocess/*.c` public entry points into dispatch wrappers that call the selected backend.
  - Implement a CPU backend that uses the current code paths (parity-first, minimal edits).
- Tests:
  - Existing `pytest` golden tests must pass unchanged.
- Acceptance:
  - `--device=cpu` produces byte-identical or within-existing-tolerance outputs.
  - No CUDA code is introduced yet.

**PR 3: Add Meson CUDA feature option + compile-time capability flag**

- Status: completed (2025-12-12)

- Scope:
  - Add `meson_options.txt` with `option('cuda', type: 'feature', value: 'disabled', ...)` (CPU-only remains default).
  - When enabled, compile CUDA sources and define a capability macro (e.g., `UNPAPER_WITH_CUDA=1`).
  - Keep `--device=cuda` failing with a clear error unless built with CUDA enabled.
- Tests:
  - CPU-only build: `meson setup builddir/ && meson test -C builddir/ -v`.
  - CUDA build: `meson setup builddir-cuda/ -Dcuda=enabled && meson test -C builddir-cuda/ -v`.
- Acceptance:
  - Default build has no CUDA dependency.
  - CUDA-enabled build produces a binary that reports CUDA capability (at least via accepting `--device=cuda` and failing later if unimplemented).

**PR 4: Extend `Image` for GPU residency + sync helpers**

- Status: completed (2025-12-12)

- Scope:
  - Extend `Image` to support backend-owned pixel storage and residency/sync flags:
    `Image.frame` remains the CPU I/O anchor (`AVFrame*`).
  - Add helpers:
    - `image_ensure_cuda(Image*)`: allocate/upload/convert once as needed
    - `image_ensure_cpu(Image*)`: download/convert once as needed
  - Add CUDA runtime scaffolding (device init, stream, error handling) behind `-Dcuda=enabled`.
- Tests:
  - Add a small C unit-test binary that verifies CPU↔CUDA round-trip copies for the supported formats used by unpaper.
  - Keep existing `pytest` golden tests passing on CPU.
- Acceptance:
  - In CUDA builds, selecting `--device=cuda -n` (no processing) can run end-to-end (load → optional upload → download → save) with stable output.

**PR 5: CUDA primitives (required by most of the pipeline)**

- Status: completed (2025-12-12)

- Scope (CUDA backend parity-first):
  - Implement: `wipe_rectangle`, `copy_rectangle`, `mirror`, `shift_image`, `flip_rotate_90`, `center_image`.
  - Ensure wrappers call CUDA implementations when `--device=cuda`.
  - Enforce “no silent fallback”: if an op is missing in CUDA mode, error out with the op name.
- Tests:
  - Unit tests comparing CPU vs CUDA outputs on synthetic images for each primitive.
  - Add/extend `pytest` cases that exercise `--pre-rotate`, `--pre-mirror`, `--pre-shift` under `--device=cuda`.
- Acceptance:
  - CUDA output matches CPU within the existing image-diff tolerance for these operations.
  - Deterministic run-to-run (identical output on repeated runs).

**PR 6: CUDA resize/stretch + interpolation parity**

- Status: completed (2025-12-12)

- Scope:
  - Implement `stretch_and_replace`, `resize_and_replace` in CUDA for NN/linear/cubic, matching CPU coordinate mapping and clamping.
  - Minimize transfers: keep data GPU-resident; only download for save/debug.
- Tests:
  - Add unit tests for scale-up/scale-down cases and each interpolation type (CPU vs CUDA).
  - Add a `pytest` case that uses `--stretch`/`--post-size` under CUDA.
- Acceptance:
  - CPU-vs-CUDA diffs remain within tolerance on existing golden inputs.

**PR 7: CUDA filters**

- Status: completed (2025-12-13)

- Scope:
  - Implement: `blackfilter`, `noisefilter`, `blurfilter`, `grayfilter`.
  - Avoid in-place hazards: use ping-pong buffers where CPU semantics imply read-before-write behavior.
- Tests:
  - Add focused regression inputs (small, synthetic or minimal real images) that isolate each filter and threshold edge cases.
  - Add a determinism check: run the same CUDA invocation twice and assert identical output.
- Acceptance:
  - CUDA runs full filter pipeline with no fallback and stable results.

**PR 8: CUDA masks/borders/wipes (detection + application)**

- Status: completed (2025-12-13)

- Scope:
  - Implement: `detect_masks`, `align_mask`, `apply_masks`, `apply_wipes`, `apply_border`, `detect_border`.
  - GPU does bulk scanning/reduction; CPU may do final selection logic if needed for determinism (this is not a fallback; it is control logic).
- Tests:
  - Add at least 1 new integration fixture covering tricky borders/masks.
  - Ensure both CPU and CUDA runs stay deterministic.
- Acceptance:
  - Mask/border-related golden tests pass under `--device=cuda`.

**PR 9: CUDA deskew (detect + apply)**

- Status: completed (2025-12-13)

- Scope:
  - Implement `detect_rotation` and `deskew` in CUDA mode.
  - Strategy: GPU computes per-angle metrics; CPU selects best angle deterministically; GPU applies the final warp using the same interpolation kernels.
- Tests:
  - Add a deskew-focused fixture (slight known rotation) and compare CPU vs CUDA output within tolerance.
- Acceptance:
  - Deskew-enabled runs under CUDA match CPU within tolerance and are deterministic.

**PR 10: Test matrix + docs polishing**

- Status: completed (2025-12-13)

- Scope:
  - Update `tests/unpaper_tests.py` to parameterize device runs (CPU always; CUDA only when enabled/available).
  - Update `doc/unpaper.1.rst` for `--device` (including error behavior when CUDA is not compiled in).
- Acceptance:
  - `meson test -C builddir/ -v` runs CPU suite everywhere.
  - `meson test -C builddir-cuda/ -v` runs CPU + CUDA parity checks where CUDA is enabled.

### Performance PR roadmap (PR11+)

Goal: significantly accelerate `--device=cuda` end-to-end throughput by removing CPU-driven inner loops, eliminating serial CUDA kernels, and overlapping decode/compute/encode. Performance is prioritized as long as CUDA output remains deterministic run-to-run and within the project’s existing test tolerances (unless explicitly tightened).

#### Performance target (this machine)

- Primary gate: `A1` (`tests/source_images/imgsrc001.png`) via `./builddir-cuda/unpaper --device cuda` mean wall time **< 2.5s** over repeated runs (after 1 warmup), writing output to a tmpfs path (e.g. `/dev/shm`) to avoid disk bottlenecks.

**PR 11: Benchmark harness + stage timing (no behavior change)**

- Status: completed (2025-12-13)
- Scope:
  - Add a small benchmark runner (e.g. `tools/bench_a1.py`) that runs warmups + N iterations and prints mean/stdev (CPU and CUDA).
  - Add an optional `--perf` (or env-gated) stage timing output (decode, upload, filters, masks/borders, deskew, download, encode).
  - Add CUDA event timing for kernel-heavy stages (with explicit stream sync for accurate reporting).
- Tests:
  - `meson test -C builddir/ -v`
  - `meson test -C builddir-cuda/ -v`
- Acceptance:
  - No output changes unless `--perf` is enabled.
  - Benchmark runner is stable and reproducible.

**PR 12: CUDA throughput scaffolding (streams + async + pooling)**

- Status: completed (2025-12-13)
- Scope:
  - Extend `imageprocess/cuda_runtime.*` to support CUDA streams and stream sync.
  - Add async H2D/D2H/D2D memcpy APIs and pinned host buffers for transfers.
  - Add a simple device scratch allocator (or CUDA async mempool) to avoid per-call device allocations for small reductions.
  - Make CUDA state safe for per-page concurrency (stream-per-job).
- Tests:
  - Existing CPU + CUDA suites.
  - Add a small C test covering stream correctness + determinism (identical output across repeated runs).
- Acceptance:
  - No behavior changes.
  - Enables later PRs to overlap decode/compute/encode without data races.

**PR 12.1: Add optional OpenCV dependency hook (C++ bridge only)**

- Status: completed (2025-12-13)
- Scope:
  - Add a Meson feature option `opencv` (default `disabled`).
  - Detect `opencv4` via pkg-config; if enabled, switch build to allow C++ and expose a small C API shim for CUDA CCL.
  - No functional changes; just build plumbing and CI knobs.
- Tests:
  - CPU + CUDA suites still pass when `-Dopencv=disabled` (default).
  - Configure/build with `-Dopencv=enabled` (skip runtime use for now).
- Acceptance:
  - Build remains C-only when OpenCV is off; adds C++ compilation path when on.

**PR 12.2: CUDA stream interop + mask adapter**

- Status: completed (2025-12-14)
- Scope:
  - Add a C shim wrapping OpenCV's CUDA stream handle from our `UnpaperCudaStream`.
  - Implement GPU mask extraction (lightness < `min_white_level`) into `cv::cuda::GpuMat` without extra H2D copies.
  - No noisefilter behavioral change yet; just utilities callable from backend.
- Tests:
  - New unit test ensuring stream interop and mask extraction round-trip equals our existing mask on CPU.
- Acceptance:
  - Utilities compile and run under `-Dopencv=enabled`; CPU/CUDA pipelines unchanged.

**PR 12.3: OpenCV CUDA CCL noisefilter path (GRAY8)**

- Status: completed (2025-12-14)
- Scope:
  - Implement noisefilter GPU path using `cv::cuda::connectedComponents` on GRAY8 images, with label counting and apply on GPU.
  - Keep existing custom CCL as fallback or when OpenCV disabled.
  - Ensure determinism and parity with CPU within tolerance.
- Implementation notes:
  - OpenCV CUDA CCL implemented in `opencv_bridge.cpp` via `unpaper_opencv_cuda_ccl()`.
  - Due to Driver API vs Runtime API context incompatibility, the OpenCV path currently falls back to custom CCL when called from the main binary (which uses cuCtxCreate). Works correctly in standalone tests using cudaMalloc.
  - Fallback to custom CUDA CCL is automatic and transparent.
- Tests:
  - `opencv_bridge_test` verifies CCL functionality in Runtime API context.
  - `cuda_filters_test` verifies fallback to custom CCL works correctly.
  - All 9 tests pass; golden pytest suite passes.
- Acceptance:
  - GRAY8 CUDA noisefilter has OpenCV path available; falls back to custom CCL when context incompatible; outputs match CPU within tolerance.

**PR 12.4: Format coverage + perf gate**

- Status: completed (2025-12-14)
- Scope:
  - Support RGB24 and Y400A by generating masks on GPU, reusing OpenCV CCL.
  - Add A1 benchmark run with `--device=cuda` + OpenCV path; target <3.0s on this machine.
  - Add a build/CLI capability flag showing which path is active.
- Implementation notes:
  - Extended `noisefilter_cuda_opencv()` to support GRAY8, Y400A, and RGB24 formats.
  - OpenCV path extracts dark mask and runs CCL; fallback to custom CUDA CCL when Driver API context used.
  - Mask application loop handles all three formats with proper lightness calculations.
  - Added Y400A noisefilter test to `cuda_filters_test.c`.
  - Added `--perf` capability flag output: `perf backends: device=<device> opencv=<yes|no> ccl=<yes|no>`.
  - Fixed critical kernel name mismatch bug (init/union/compress → build_labels/propagate) from prior commit.
  - A1 benchmark: ~1.40s average, well under 3.0s target.
- Tests:
  - New regression for RGB24 noisefilter parity; re-run benchmark harness.
- Acceptance:
  - A1 CUDA runtime <3.0s with OpenCV path enabled; parity holds across formats.

**PR 12.5: Packaging + fallback polish**

- Status: completed (2025-12-14)
- Scope:
  - Document the optional OpenCV dependency and how to enable it.
  - Ensure clean fallbacks (clear error when `--device=cuda` + OpenCV path requested but unavailable).
  - Trim extra allocations; prefer stream reuse; add perf logging guardable by `--perf`.
- Implementation notes:
  - Updated README.md with build instructions for CUDA (`-Dcuda=enabled`) and OpenCV (`-Dopencv=enabled`).
  - Updated doc/unpaper.1.rst to document OpenCV acceleration under `--device` option and capability output under `--perf`.
  - Fallback behavior already implemented: when OpenCV is unavailable or CCL fails, noisefilter falls back to built-in CUDA CCL automatically.
  - Stub file (`opencv_bridge_stub.c`) returns `false` for all capability queries when OpenCV disabled.
  - Capability flags visible via `--perf`: `perf backends: device=<device> opencv=<yes|no> ccl=<yes|no>`.
  - All files have proper SPDX headers (no new files created).
- Tests:
  - CPU + CUDA suites with and without OpenCV; `reuse lint` still passes.
- Acceptance:
  - Optional dependency well-documented; behavior predictable when absent.

### OpenCV-Based GPU Backend Rewrite (PR13-PR18)

**Strategy pivot**: Make OpenCV with CUDA a mandatory dependency for `--device=cuda` builds. This allows replacing custom CUDA kernels with highly-optimized OpenCV CUDA functions, reducing maintenance burden and improving performance.

**Key technical considerations**:
1. **CUDA context model**: The current code uses `cuCtxCreate` (Driver API) which is incompatible with OpenCV's Runtime API (`cudaMalloc`). Must switch to Runtime API for OpenCV compatibility.
2. **Memory management**: Use `cv::cuda::GpuMat` for all GPU-resident images; wrap existing allocations or re-allocate via `cudaMalloc`.
3. **Mono format handling**: OpenCV doesn't natively support 1-bit packed images. Strategy: convert mono ↔ 8-bit on GPU or keep minimal custom kernels for mono.
4. **Format mapping**: GRAY8 → CV_8UC1, Y400A → CV_8UC2, RGB24 → CV_8UC3.

**OpenCV CUDA modules required**:
- `cudaarithm`: arithmetic, comparisons, reductions, threshold
- `cudaimgproc`: connected components, color conversion, histograms
- `cudawarping`: resize, warpAffine, rotate
- `cudafilters`: morphology (optional, for blackfilter optimization)

**Custom kernel → OpenCV replacement mapping**:
| Custom Kernel | OpenCV Replacement | Module |
|---------------|-------------------|--------|
| wipe_rect | `GpuMat::setTo()` with ROI | core |
| copy_rect | `GpuMat::copyTo()` with ROI | core |
| mirror | `cv::cuda::flip()` | cudaarithm |
| rotate90 | `cv::cuda::rotate()` or transpose+flip | cudawarping |
| stretch/resize | `cv::cuda::resize()` | cudawarping |
| rotate (deskew) | `cv::cuda::warpAffine()` | cudawarping |
| count_brightness | `cv::cuda::threshold()` + `countNonZero()` | cudaarithm |
| sum_lightness | `cv::cuda::min()` + `sum()` | cudaarithm |
| noisefilter CCL | `cv::cuda::connectedComponents()` | cudaimgproc |
| blackfilter | `cv::cuda::threshold()` + morphology + CCL | cudaimgproc |

**PR 13: Make OpenCV mandatory + Fix CUDA context model**

- Status: completed (2025-12-14)
- Scope:
  - Make OpenCV (with CUDA modules) a required dependency for `-Dcuda=enabled` builds.
  - Remove `opencv_bridge_stub.c` and the `-Dopencv=` option; OpenCV is now implicit with CUDA.
  - Switch CUDA runtime from Driver API (`cuCtxCreate`, `cuMemAlloc`) to Runtime API (`cudaMalloc`, `cudaFree`) for OpenCV compatibility.
  - Update `cuda_runtime.c` to use `cudaMalloc`/`cudaFree` instead of `cuMemAlloc`/`cuMemFree`.
  - Update `opencv_bridge.cpp` to work directly with our GpuMat-compatible allocations (no D2H→H2D round-trips).
  - Update build system: require `opencv4` with `cudaarithm`, `cudaimgproc`, `cudawarping` when `-Dcuda=enabled`.
  - Update documentation (README.md, man page) to reflect mandatory OpenCV dependency.
- Implementation notes:
  - Rewrote `cuda_runtime.c` to use CUDA Runtime API (`cudaMalloc`, `cudaFree`, `cudaMemcpy`, `cudaStreamCreate`, etc.) instead of Driver API. Driver API is only used for PTX module loading (`cuModuleLoadData`, `cuModuleGetFunction`, `cuLaunchKernel`).
  - Removed `opencv_bridge_stub.c` and `-Dopencv=` meson option. OpenCV is now mandatory when `-Dcuda=enabled`.
  - Updated `meson.build` to consolidate CUDA and OpenCV setup in a single block. OpenCV is detected and required after CUDA is detected.
  - Updated `opencv_bridge.cpp` to include `<opencv2/core/cuda_stream_accessor.hpp>` for `cv::cuda::StreamAccessor`. Fixed `connectedComponents` call (doesn't take stream parameter).
  - Updated `unpaper.c` to guard `unpaper_opencv_*` calls with `#ifdef UNPAPER_WITH_OPENCV` for CPU-only builds.
  - Updated README.md and doc/unpaper.1.rst to document mandatory OpenCV dependency for CUDA builds.
  - Memory allocations via `cudaMalloc` are now directly compatible with OpenCV's `cv::cuda::GpuMat` (no context conflicts).
- Tests:
  - All 9 CUDA tests pass (cuda roundtrip, primitives, resize, filters, masks/borders, deskew, stream, opencv bridge, pytest suite).
  - All 24 CPU-only pytest tests pass; 10 CUDA tests skipped.
  - `--device=cuda` in CPU-only builds fails with clear error message.
  - A1 benchmark: ~1.0s with CUDA+OpenCV, performance maintained.
- Acceptance:
  - `--device=cuda` requires OpenCV at build time; fails clearly if unavailable.
  - No Driver API usage for memory operations (only PTX loading uses Driver API).
  - A1 benchmark performance maintained.

**PR 14: OpenCV primitives (wipe, copy, mirror, rotate90)**

- Status: completed (2025-12-14)
- Scope:
  - Create `opencv_ops.cpp` with C API wrappers for OpenCV CUDA primitives.
  - Replace `wipe_rectangle_cuda` with `cv::cuda::GpuMat::setTo()` via ROI.
  - Replace `copy_rectangle_cuda` with `cv::cuda::GpuMat::copyTo()` via ROI.
  - Replace `mirror_cuda` with `cv::cuda::flip()`.
  - Replace `flip_rotate_90_cuda` with `cv::cuda::rotate()` or transpose+flip combination.
  - For mono formats: convert to 8-bit, process, convert back (or keep minimal custom kernels if performance-critical).
  - Remove corresponding custom CUDA kernels from `cuda_kernels.cu`.
- Implementation notes:
  - Created `opencv_ops.h` and `opencv_ops.cpp` with C API wrappers for OpenCV CUDA primitives.
  - OpenCV path used for GRAY8, Y400A, RGB24 formats; mono formats fall back to custom kernels.
  - OpenCV `cv::cuda::transpose()` only supports element sizes of 1, 4, or 8 bytes. RGB24 (3 bytes) and Y400A (2 bytes) are not supported, so `rotate90` only uses OpenCV for GRAY8 and falls back to custom kernel for other formats.
  - Custom CUDA kernels retained for: mono formats (all operations), RGB24/Y400A rotate90.
  - Updated `backend_cuda.c` to try OpenCV path first, falling back to custom kernels when OpenCV returns false.
- Tests:
  - Extend `cuda_primitives_test.c` to verify parity with CPU for all replaced operations.
  - Ensure determinism across repeated runs.
- Acceptance:
  - All primitive operations work correctly for GRAY8, Y400A, RGB24, and mono formats.
  - Custom kernels retained where OpenCV limitations exist (mono formats, RGB24/Y400A rotate90).

**PR 15: OpenCV resize and deskew (cudawarping)**

- Status: planned
- Scope:
  - Replace `stretch_and_replace_cuda` and `resize_and_replace_cuda` with `cv::cuda::resize()`.
  - Map interpolation modes: NEAREST → INTER_NEAREST, LINEAR → INTER_LINEAR, CUBIC → INTER_CUBIC.
  - Replace `deskew_cuda` rotation with `cv::cuda::warpAffine()`.
  - Handle coordinate mapping to match CPU behavior exactly (center-based rotation).
  - For mono: convert to 8-bit, resize/rotate, threshold back to mono.
  - Remove custom kernels `stretch_*`, `rotate_*` from `cuda_kernels.cu`.
- Tests:
  - Extend `cuda_resize_test.c` and `cuda_deskew_test.c` for all interpolation modes.
  - Verify sub-pixel accuracy matches CPU within tolerance.
- Acceptance:
  - Resize and deskew produce identical output to CPU within existing tolerances.
  - Custom stretch/rotate kernels removed.
  - A1 benchmark: deskew stage time reduced by ≥50%.

**PR 16: OpenCV-based filters (noisefilter, grayfilter, blurfilter, blackfilter)**

- Status: planned
- Scope:
  - **noisefilter**: Already uses `cv::cuda::connectedComponents()`; optimize to eliminate any CPU loops and D2H transfers. Use `cv::cuda::countNonZero()` for component size counting if possible, or GPU histogram.
  - **grayfilter**: Replace tile-based CPU loop with:
    - `cv::cuda::threshold()` to create dark-pixel mask
    - Use integral images or tiled `cv::cuda::sum()` for per-tile statistics
    - Single-pass wipe kernel for tiles below threshold
  - **blurfilter**: Similar to grayfilter:
    - `cv::cuda::threshold()` for brightness mask
    - Integral images for efficient window sums
    - Batch wipe for isolated clusters
  - **blackfilter**: Replace flood-fill with:
    - `cv::cuda::threshold()` to create near-black mask
    - `cv::cuda::connectedComponents()` for region detection
    - Optionally use morphological operations (`cv::cuda::dilate`) for intensity tolerance
    - Batch wipe of detected black regions
  - Remove custom kernels: `noisefilter_*`, `count_brightness_range`, `sum_lightness_rect`, `sum_grayscale_rect`, `blackfilter_floodfill_rect`.
- Tests:
  - Extend `cuda_filters_test.c` with comprehensive coverage.
  - Verify determinism and parity with CPU.
- Acceptance:
  - All filters work without CPU-driven inner loops.
  - A1 benchmark: filters stage time reduced by ≥70%.
  - Primary gate: `A1` CUDA mean **< 1.0s** on this machine.

**PR 17: OpenCV-based detection (masks, borders, deskew angle)**

- Status: planned
- Scope:
  - **detect_masks_cuda**: Replace custom reduction with:
    - `cv::cuda::threshold()` for dark-pixel mask
    - `cv::cuda::sum()` or `cv::cuda::countNonZero()` with ROI for bar scanning
    - Keep control logic on CPU for determinism
  - **detect_border_cuda**: Similar approach using OpenCV reductions.
  - **detect_rotation_cuda**: Replace `detect_edge_rotation_peaks` kernel with:
    - `cv::cuda::threshold()` for edge mask
    - Pre-computed rotation matrices for angle sampling
    - Parallel `cv::cuda::sum()` calls for each angle
    - CPU-side angle selection for determinism
  - Remove custom kernel `detect_edge_rotation_peaks`.
- Tests:
  - Verify mask/border detection matches CPU exactly.
  - Verify deskew angle detection produces identical angles.
- Acceptance:
  - Detection operations have no custom CUDA kernels.
  - Deskew angle detection time reduced by ≥50%.

**PR 18: Cleanup, optimization, and final benchmarking**

- Status: planned
- Scope:
  - Remove all unused custom CUDA kernels from `cuda_kernels.cu`.
  - Remove PTX compilation if no custom kernels remain (or keep minimal set for mono format handling).
  - Add CUDA stream overlap: decode next page while processing current, encode previous while processing next.
  - Implement memory pooling: reuse `cv::cuda::GpuMat` buffers across operations to reduce allocation overhead.
  - Add `cv::cuda::Stream` integration for async operations.
  - Final performance tuning and profiling.
  - Update documentation with final architecture and performance characteristics.
- Tests:
  - Full test suite passes.
  - Determinism verified across all operations.
  - Memory leak check with valgrind/cuda-memcheck.
- Acceptance:
  - Minimal or no custom CUDA kernels remain.
  - A1 benchmark: CUDA mean **< 0.5s** on this machine (target: 3-4× faster than current ~1.4s).
  - Multi-page throughput ≥2× compared to single-threaded CPU.
