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
