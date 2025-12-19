# Repository Guidelines

## Project Structure & Module Organization

- `unpaper.c`, `parse.c/.h`, `file.c`: core CLI, option parsing, and file I/O.
- `imageprocess/`: image-processing primitives and algorithms (deskew, filters, masks, etc.).
- `lib/`: shared utilities (logging, options, physical dimension helpers).
- `pdf/`: PDF reader/writer and pipelines. CPU PDF processing is implemented as a thin wrapper around the generic batch infrastructure (`lib/decode_queue` + `batch_process_parallel()`), with pages written via `pdf/pdf_page_accumulator.c`. GPU PDF processing remains a specialized pipeline.
- `tests/`: `pytest` suite plus `source_images/` inputs and `golden_images/` expected outputs.
- `doc/`: Sphinx sources used to build the `unpaper(1)` man page and additional Markdown docs.

## Build, Test, and Development Commands

This project uses Meson (with Ninja). **Set PATH for meson**: `PATH="/home/scott/Documents/unpaper/.venv/bin:/usr/bin:$PATH"`

- Configure: `meson setup builddir/ --buildtype=debugoptimized`
- Build: `meson compile -C builddir/`
- Run tests: `meson test -C builddir/ -v`
- Build man page: `meson compile -C builddir/ man`
- Install (staged): `DESTDIR=/tmp/unpaper-staging meson install -C builddir/`

System dependencies include FFmpeg libraries (`libavformat`, `libavcodec`, `libavutil`) and Python packages in `requirements.txt` (Meson, Sphinx, pytest, Pillow). PDF support is optional (`-Dpdf=enabled`) and uses MuPDF.

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

## Architecture Refactor Roadmap (Modularization + Test Strategy)

### Why change the current structure?

Key maintainability issues observed:
- `unpaper.c` mixes CLI parsing, validation, pipeline selection, and batch orchestration.
- `process_sheet()` is monolithic and mixes decode, transforms, filters, masks, deskew, and output paths.
- Batch decode handling is duplicated for legacy vs batched queues.
- PDF batch pipeline fakes input filenames instead of using first-class input descriptors.
- Logging/observability is inconsistent (`verboseLog` vs `fprintf`) and lacks job context.
- Tests are redundant and slow (GPU/JPEG/PDF goldens re-run the same cases).

### Target design (no behavior changes)

Layered structure with explicit stage pipeline:
- **CLI layer**: parse + validate + resolve options.
- **Core pipeline**: stage runner with explicit stages (decode, pre, filters, masks, deskew, post, output).
- **Pipelines**: image and PDF orchestration using shared core.
- **Adapters**: decode/encode providers (queue abstractions), PDF input adapter.
- **Infra**: logging, threadpool, queues.

Proposed file layout (guide, not a strict requirement):
```
src/cli/*              # option parsing + validation + main
src/core/*             # sheet pipeline + stages + state
src/pipeline/*         # image/pdf orchestration
src/adapters/*         # decode/encode providers, pdf input adapter
src/infra/*            # logging + threading utils
```

### PR-by-PR refactor roadmap (execute in order)

**PR A1: Test suite split (fast vs slow)**
- Scope:
  - Add pytest markers: `fast`, `slow`, `gpu`, `pdf`.
  - Move full PDF matrix + multistream scaling to `slow`.
  - Keep a minimal fast subset: 3–5 goldens + 1 GPU equivalence + 1 PDF smoke.
- Acceptance:
  - Default `pytest -m "not slow"` finishes quickly.
  - Full coverage still available in `slow`.

**PR A2: CLI parsing extraction**
- Scope:
  - Split `unpaper.c` into `src/cli/cli_options.c` (parse/validate) and
    `src/cli/cli_main.c` (orchestration).
  - Introduce `OptionsResolved` (derived defaults, device/batch inference).
- Tests:
  - Add unit tests for `parseMultiIndex`, layout parsing, and validation errors.
- Acceptance:
  - No CLI behavior change; `--help` output unchanged.
- Status: complete.

**PR A3: Core pipeline API**
- Scope:
  - Create `src/core/sheet_pipeline.c` exposing `sheet_pipeline_run()`.
  - Initially call existing `process_sheet()` (no logic change).
- Acceptance:
  - Pipelines call the new API, behavior unchanged.
- Status: complete.

**PR A4: Stage extraction from `process_sheet()`**
- Scope:
  - Extract stages into `src/core/sheet_stages.c`:
    decode, pre, filters, masks, deskew, post, output.
  - Stage table controls order + skip logic.
- Tests:
  - Add focused unit tests for 1–2 stages (mask detection, layout).
- Acceptance:
  - Bit‑for‑bit output identical for all existing goldens.

**PR A5: Decode provider abstraction**
- Scope:
  - Introduce `DecodedImageProvider` interface with `get/release`.
  - Remove duplicated decode handling in `lib/batch_worker.c`.
- Acceptance:
  - Batch/per-image decode queues are interchangeable.

**PR A6: First-class batch inputs (PDF pages)**
- Scope:
  - Add `BatchInput` type to `BatchJob` (file path / PDF page index).
  - PDF pipeline uses real page descriptors; no fake filenames.
- Tests:
  - Existing PDF tests pass unchanged.

**PR A7: Logging unification + job context**
- Scope:
  - Central logger wrapper with `job/sheet/device` context.
  - Replace ad‑hoc `fprintf` in batch worker & pipelines.
- Acceptance:
  - Log output unchanged in content but consistently formatted.

**PR A8: Pipeline split (image vs PDF)**
- Scope:
  - Move orchestration into `src/pipeline/image_pipeline.c` and
    `src/pipeline/pdf_pipeline.c`.
  - `cli_main` only selects pipeline.
- Acceptance:
  - No behavior changes; reduces `unpaper.c` complexity.

**PR A9: Test redundancy cleanup**
- Scope:
  - Remove duplicate GPU JPEG golden comparisons where image goldens already cover
    the same transformations.
  - Reduce multistream scaling test to 1 vs 4 streams.
  - Keep one CPU↔CUDA equivalence for minimal and full pipelines.
- Acceptance:
  - Coverage preserved with fewer, more targeted tests.
