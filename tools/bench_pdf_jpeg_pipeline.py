#!/usr/bin/env python3
# SPDX-FileCopyrightText: 2025 The unpaper authors
#
# SPDX-License-Identifier: GPL-2.0-only

"""
Benchmark PDF vs batch JPEG pipeline performance for unpaper.

This script builds a PDF from JPEG images created the same way as
`tools/bench_jpeg_pipeline.py`, then compares:
  1) Batch JPEG processing (JPEG -> GPU -> JPEG)
  2) PDF processing with JPEG-encoded pages (PDF -> GPU -> PDF)

The goal is to ensure PDF GPU pipeline throughput is close to the batch
pipeline on the same images.
"""

import argparse
import statistics
import subprocess
import sys
import tempfile
import time
from pathlib import Path


def create_test_jpegs(
    source_image: Path, output_dir: Path, count: int, jpeg_quality: int
) -> list[Path]:
    """Create JPEG test images from a source image."""
    try:
        from PIL import Image
    except ImportError:
        print("ERROR: PIL/Pillow required for JPEG conversion", file=sys.stderr)
        sys.exit(1)

    images = []
    src_img = Image.open(source_image)
    if src_img.mode == "RGBA":
        src_img = src_img.convert("RGB")
    for i in range(1, count + 1):
        dest = output_dir / f"input{i:04d}.jpg"
        src_img.save(dest, "JPEG", quality=jpeg_quality)
        images.append(dest)
    return images


def check_binary_support(binary: Path) -> tuple[bool, bool]:
    """Return (pdf_supported, cuda_supported)."""
    proc = subprocess.run(
        [str(binary), "--help"],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )
    help_text = (proc.stdout or b"") + (proc.stderr or b"")
    pdf_supported = b"--pdf-quality" in help_text
    cuda_supported = b"--device=cpu|cuda" in help_text
    return pdf_supported, cuda_supported


def create_pdf_from_jpegs(jpegs: list[Path], output_pdf: Path, dpi: int) -> None:
    """Create a PDF with one JPEG per page using mutool."""
    import shutil

    if shutil.which("mutool") is None:
        raise RuntimeError("missing external tool: mutool")

    try:
        from PIL import Image
    except ImportError:
        print("ERROR: PIL/Pillow required for PDF creation", file=sys.stderr)
        sys.exit(1)

    page_files: list[Path] = []
    workdir = output_pdf.parent

    for i, img_path in enumerate(jpegs, start=1):
        with Image.open(img_path) as img:
            width_px, height_px = img.size
        width_pt = width_px * 72.0 / dpi
        height_pt = height_px * 72.0 / dpi

        page_txt = workdir / f"page-{i:03d}.txt"
        page_txt.write_text(
            "\n".join(
                [
                    f"%%MediaBox 0 0 {width_pt:.6f} {height_pt:.6f}",
                    f"%%Image Im{i} {img_path.name}",
                    "q",
                    f"{width_pt:.6f} 0 0 {height_pt:.6f} 0 0 cm",
                    f"/Im{i} Do",
                    "Q",
                    "",
                ]
            ),
            encoding="utf-8",
        )
        page_files.append(page_txt)

    subprocess.run(
        ["mutool", "create", "-o", str(output_pdf), *[str(p) for p in page_files]],
        cwd=str(workdir),
        check=True,
    )


def run_unpaper(cmd: list[str]) -> float:
    start = time.perf_counter()
    proc = subprocess.run(
        cmd,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.PIPE,
        check=False,
    )
    elapsed = (time.perf_counter() - start) * 1000.0

    if proc.returncode != 0:
        err = proc.stderr.decode(errors="replace").strip()
        raise RuntimeError(err or "unpaper failed")

    return elapsed


def bench_configuration(
    run_cmd: list[str], warmup: int, iterations: int
) -> tuple[float, float]:
    samples: list[float] = []
    for i in range(warmup + iterations):
        elapsed = run_unpaper(run_cmd)
        if i >= warmup:
            samples.append(elapsed)

    mean = statistics.mean(samples)
    stdev = statistics.pstdev(samples) if len(samples) > 1 else 0.0
    return mean, stdev


def main() -> int:
    repo_root = Path(__file__).resolve().parent.parent
    default_input = repo_root / "tests" / "source_images" / "imgsrc001.png"

    parser = argparse.ArgumentParser(
        description="Benchmark PDF vs batch JPEG pipeline performance."
    )
    parser.add_argument(
        "--builddir",
        default=repo_root / "builddir-unified-pdf-cuda",
        type=Path,
        help="Meson builddir containing unpaper binary",
    )
    parser.add_argument(
        "--source",
        default=default_input,
        type=Path,
        help="Source image to convert to JPEG for testing",
    )
    parser.add_argument(
        "--images",
        default=50,
        type=int,
        help="Number of test images (default: 50)",
    )
    parser.add_argument(
        "--warmup",
        default=1,
        type=int,
        help="Warmup runs before measurement",
    )
    parser.add_argument(
        "--iterations",
        default=3,
        type=int,
        help="Measurement iterations",
    )
    parser.add_argument(
        "--jpeg-quality",
        default=95,
        type=int,
        help="JPEG quality for input/output (default: 95)",
    )
    parser.add_argument(
        "--pdf-dpi",
        default=300,
        type=int,
        help="PDF page DPI when building the input PDF (default: 300)",
    )
    parser.add_argument(
        "--jobs",
        default=8,
        type=int,
        help="Number of batch jobs/threads (default: 8)",
    )
    parser.add_argument(
        "--streams",
        default=8,
        type=int,
        help="Number of CUDA streams (default: 8)",
    )
    parser.add_argument(
        "--max-slowdown",
        default=1.15,
        type=float,
        help="Fail if PDF is slower than batch by this factor (default: 1.15)",
    )
    parser.add_argument(
        "--verify-close",
        action="store_true",
        help="Exit non-zero if PDF is slower than --max-slowdown",
    )
    parser.add_argument(
        "--keep-temp",
        action="store_true",
        help="Keep temporary files for inspection",
    )

    args = parser.parse_args()

    binary = args.builddir / "unpaper"
    if not binary.exists():
        print(f"ERROR: unpaper binary not found at {binary}", file=sys.stderr)
        return 1

    pdf_ok, cuda_ok = check_binary_support(binary)
    if not pdf_ok:
        print("ERROR: unpaper binary lacks PDF support", file=sys.stderr)
        return 1
    if not cuda_ok:
        print("ERROR: unpaper binary lacks CUDA support", file=sys.stderr)
        return 1

    temp_ctx = (
        tempfile.TemporaryDirectory(prefix="unpaper-pdf-bench-")
        if not args.keep_temp
        else None
    )
    try:
        workdir = Path(temp_ctx.name) if temp_ctx else Path(
            tempfile.mkdtemp(prefix="unpaper-pdf-bench-")
        )

        # Create JPEG inputs the same way as bench_jpeg_pipeline.py
        jpegs = create_test_jpegs(
            args.source, workdir, args.images, args.jpeg_quality
        )

        # Build a PDF using those JPEGs (one image per page)
        input_pdf = workdir / "input.pdf"
        create_pdf_from_jpegs(jpegs, input_pdf, args.pdf_dpi)

        # Batch JPEG processing (JPEG -> GPU -> JPEG)
        batch_output_pattern = str(workdir / "output%04d.jpg")
        batch_cmd = [
            str(binary),
            "--batch",
            f"--jobs={args.jobs}",
            "--device=cuda",
            f"--cuda-streams={args.streams}",
            f"--jpeg-quality={args.jpeg_quality}",
            "--overwrite",
            str(workdir / "input%04d.jpg"),
            batch_output_pattern,
        ]

        batch_mean, batch_stdev = bench_configuration(
            batch_cmd, args.warmup, args.iterations
        )

        # PDF processing (PDF -> GPU -> PDF)
        output_pdf = workdir / "output.pdf"
        pdf_cmd = [
            str(binary),
            "--batch",
            f"--jobs={args.jobs}",
            "--device=cuda",
            f"--cuda-streams={args.streams}",
            "--pdf-quality=fast",
            "--overwrite",
            str(input_pdf),
            str(output_pdf),
        ]

        pdf_mean, pdf_stdev = bench_configuration(
            pdf_cmd, args.warmup, args.iterations
        )

        ratio = pdf_mean / batch_mean if batch_mean > 0 else float("inf")

        print("Benchmark results (ms):")
        print(
            f"  Batch JPEG: {batch_mean:.2f} ± {batch_stdev:.2f}"
        )
        print(
            f"  PDF JPEG:   {pdf_mean:.2f} ± {pdf_stdev:.2f}"
        )
        print(f"  PDF / Batch ratio: {ratio:.3f}x")

        if args.verify_close and ratio > args.max_slowdown:
            print(
                f"ERROR: PDF slower than batch by {ratio:.3f}x (limit {args.max_slowdown:.3f}x)",
                file=sys.stderr,
            )
            return 2

        if args.keep_temp:
            print(f"Temporary files kept at: {workdir}")

    finally:
        if temp_ctx is not None:
            temp_ctx.cleanup()

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
