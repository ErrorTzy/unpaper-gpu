#!/usr/bin/env python3
# SPDX-FileCopyrightText: 2025 The unpaper authors
#
# SPDX-License-Identifier: GPL-2.0-only

"""
PDF processing benchmark for unpaper.

This script measures the performance of unpaper's PDF processing pipeline
with different configurations. It measures:
- Single-page processing latency
- Multi-page PDF throughput
- GPU vs CPU performance
- Batch mode scaling

Usage:
    python tools/bench_pdf.py                     # Basic benchmark with test PDFs
    python tools/bench_pdf.py --pages 100         # Create and test 100-page PDF
    python tools/bench_pdf.py --devices cpu,cuda  # Compare CPU vs GPU
    python tools/bench_pdf.py --verify-targets    # Verify PR8 performance targets
"""

import argparse
import os
import shutil
import statistics
import subprocess
import sys
import tempfile
import time
from pathlib import Path


def check_pdf_support(binary: Path) -> bool:
    """Check if the binary has PDF support compiled in."""
    proc = subprocess.run(
        [str(binary), "--help"],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )
    return b".pdf" in proc.stdout.lower() or b"pdf" in proc.stderr.lower()


def check_cuda_available(binary: Path) -> bool:
    """Check if the binary supports CUDA."""
    proc = subprocess.run(
        [str(binary), "--help"],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )
    return b"--device=cpu|cuda" in proc.stdout


def create_test_pdf(source_image: Path, output_pdf: Path, page_count: int) -> bool:
    """Create a test PDF with repeated pages from a source image.

    Uses PIL/Pillow to create a multi-page PDF.
    """
    try:
        from PIL import Image
    except ImportError:
        print("ERROR: PIL/Pillow not installed. Run: pip install Pillow", file=sys.stderr)
        return False

    # Load source image
    img = Image.open(source_image)
    if img.mode == "RGBA":
        img = img.convert("RGB")

    # Create multi-page PDF
    images = [img.copy() for _ in range(page_count - 1)]
    img.save(output_pdf, "PDF", save_all=True, append_images=images, resolution=300.0)

    return True


def get_pdf_page_count(pdf_path: Path) -> int:
    """Get the number of pages in a PDF."""
    try:
        from pypdf import PdfReader
        reader = PdfReader(str(pdf_path))
        return len(reader.pages)
    except ImportError:
        pass

    # Fallback: try with pikepdf
    try:
        import pikepdf
        with pikepdf.open(str(pdf_path)) as pdf:
            return len(pdf.pages)
    except ImportError:
        pass

    # Fallback: guess 1 page for test files
    if "2page" in str(pdf_path):
        return 2
    return 1


def run_pdf_process(binary: Path, input_pdf: Path, output_pdf: Path,
                    device: str = "cpu", batch: bool = False,
                    jobs: int = 1) -> tuple[float, bool]:
    """Run unpaper on a PDF and return elapsed time in ms.

    Returns (elapsed_ms, success).
    """
    # Remove output if exists
    if output_pdf.exists():
        output_pdf.unlink()

    cmd = [
        str(binary),
        f"--device={device}",
        "--overwrite",
    ]

    if batch:
        cmd.extend(["--batch", f"--jobs={jobs}"])

    cmd.extend([str(input_pdf), str(output_pdf)])

    start = time.perf_counter()
    proc = subprocess.run(
        cmd,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.PIPE,
        check=False,
    )
    elapsed = (time.perf_counter() - start) * 1000.0

    success = proc.returncode == 0 and output_pdf.exists()

    if not success and proc.stderr:
        err_msg = proc.stderr.decode().strip()[:200]
        print(f"  [ERROR] {err_msg}", file=sys.stderr)

    return elapsed, success


def bench_pdf(binary: Path, input_pdf: Path, output_dir: Path,
              device: str, batch: bool, jobs: int,
              warmup: int, iterations: int) -> tuple[float, float, int]:
    """Benchmark PDF processing.

    Returns (mean_ms, stdev_ms, page_count).
    """
    output_pdf = output_dir / "output.pdf"
    page_count = get_pdf_page_count(input_pdf)

    samples = []
    for i in range(warmup + iterations):
        elapsed, success = run_pdf_process(
            binary, input_pdf, output_pdf, device, batch, jobs
        )

        if not success:
            return float("nan"), float("nan"), page_count

        if i >= warmup:
            samples.append(elapsed)

    mean = statistics.mean(samples)
    stdev = statistics.pstdev(samples) if len(samples) > 1 else 0.0

    return mean, stdev, page_count


def main() -> int:
    repo_root = Path(__file__).resolve().parent.parent
    default_source = repo_root / "tests" / "source_images" / "imgsrc001.png"
    default_test_pdf = repo_root / "tests" / "pdf_samples" / "test_jpeg.pdf"

    parser = argparse.ArgumentParser(
        description="Benchmark unpaper PDF processing performance."
    )
    parser.add_argument(
        "--builddir",
        default=repo_root / "builddir-cuda",
        type=Path,
        help="Meson builddir containing unpaper binary",
    )
    parser.add_argument(
        "--source",
        default=default_source,
        type=Path,
        help="Source image for generating test PDFs",
    )
    parser.add_argument(
        "--pdf",
        default=None,
        type=Path,
        help="Use existing PDF instead of generating one",
    )
    parser.add_argument(
        "--pages",
        default="1,10,50",
        help="Comma-separated page counts to test (when generating PDFs)",
    )
    parser.add_argument(
        "--devices",
        default="cpu",
        help="Comma-separated devices to test (cpu, cuda)",
    )
    parser.add_argument(
        "--batch",
        action="store_true",
        help="Use batch mode for multi-page PDFs",
    )
    parser.add_argument(
        "--jobs",
        default="4",
        help="Comma-separated job counts to test in batch mode",
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
        "--verify-targets",
        action="store_true",
        help="Verify PR8 performance targets",
    )

    args = parser.parse_args()

    # Find binary
    binary = args.builddir / "unpaper"
    if not binary.exists():
        # Try builddir-pdf for PDF-only builds
        for candidate in ["builddir-cuda", "builddir-pdf", "builddir"]:
            binary = repo_root / candidate / "unpaper"
            if binary.exists():
                break

    if not binary.exists():
        print(f"Binary not found. Tried: {args.builddir / 'unpaper'}", file=sys.stderr)
        return 1

    # Check PDF support
    # Note: PDF support detection is implicit - just try to run

    # Check CUDA
    devices = [d.strip() for d in args.devices.split(",") if d.strip()]
    cuda_available = check_cuda_available(binary)
    if "cuda" in devices and not cuda_available:
        print("WARNING: CUDA requested but not available", file=sys.stderr)
        devices = [d for d in devices if d != "cuda"]

    if not devices:
        print("No valid devices to test", file=sys.stderr)
        return 1

    page_counts = [int(p.strip()) for p in args.pages.split(",") if p.strip()]
    job_counts = [int(j.strip()) for j in args.jobs.split(",") if j.strip()]

    # Use tmpfs if available
    tmpdir_base = Path("/dev/shm") if Path("/dev/shm").is_dir() else Path(tempfile.gettempdir())

    all_results = {}  # (pages, device, batch, jobs) -> (mean, stdev, pages)

    print(f"\n{'='*70}")
    print("PDF Processing Benchmark")
    print(f"{'='*70}")
    print(f"  Binary: {binary}")
    print(f"  Devices: {', '.join(devices)}")
    print(f"  Page counts: {', '.join(map(str, page_counts))}")
    print(f"  Warmup: {args.warmup}, Iterations: {args.iterations}")
    print()

    with tempfile.TemporaryDirectory(dir=tmpdir_base) as tmpdir:
        tmpdir = Path(tmpdir)

        for page_count in page_counts:
            print(f"\n{'-'*60}")
            print(f"Testing {page_count}-page PDF")
            print(f"{'-'*60}")

            # Create or use existing PDF
            if args.pdf:
                test_pdf = args.pdf
                actual_pages = get_pdf_page_count(test_pdf)
                print(f"  Using: {test_pdf} ({actual_pages} pages)")
            else:
                test_pdf = tmpdir / f"test_{page_count}page.pdf"
                print(f"  Creating {page_count}-page test PDF...", end=" ", flush=True)
                if not create_test_pdf(args.source, test_pdf, page_count):
                    print("FAILED")
                    continue
                print("done")

            # Test each device
            for device in devices:
                device_label = device.upper()

                # Non-batch mode
                print(f"  {device_label} (sequential)...", end=" ", flush=True)
                mean, stdev, pages = bench_pdf(
                    binary, test_pdf, tmpdir, device, False, 1,
                    args.warmup, args.iterations
                )

                if mean != mean:  # NaN check
                    print("FAILED")
                else:
                    per_page = mean / pages
                    throughput = (pages / mean) * 1000.0
                    print(f"{mean:.0f}ms ({per_page:.1f}ms/page, {throughput:.1f} pages/s)")
                    all_results[(pages, device, False, 1)] = (mean, stdev, pages)

                # Batch mode
                if args.batch and pages > 1:
                    for jobs in job_counts:
                        print(f"  {device_label} batch (jobs={jobs})...", end=" ", flush=True)
                        mean, stdev, pages = bench_pdf(
                            binary, test_pdf, tmpdir, device, True, jobs,
                            args.warmup, args.iterations
                        )

                        if mean != mean:
                            print("FAILED")
                        else:
                            per_page = mean / pages
                            throughput = (pages / mean) * 1000.0
                            print(f"{mean:.0f}ms ({per_page:.1f}ms/page, {throughput:.1f} pages/s)")
                            all_results[(pages, device, True, jobs)] = (mean, stdev, pages)

    # Summary
    print(f"\n{'='*70}")
    print("Summary")
    print(f"{'='*70}")

    print(f"\n{'Pages':<8} {'Config':<25} {'Time (ms)':<12} {'ms/page':<10} {'pages/s':<10}")
    print("-" * 65)

    for (pages, device, batch, jobs), (mean, stdev, _) in sorted(all_results.items()):
        if mean != mean:
            continue
        config = f"{device}"
        if batch:
            config += f" batch(j={jobs})"
        per_page = mean / pages
        throughput = (pages / mean) * 1000.0
        print(f"{pages:<8} {config:<25} {mean:<12.0f} {per_page:<10.1f} {throughput:<10.1f}")

    # Verify PR8 targets
    if args.verify_targets:
        print(f"\n{'='*70}")
        print("PR8 Performance Target Verification")
        print(f"{'='*70}")

        targets_passed = True

        # Target 1: Single page GPU < 50ms
        single_gpu = all_results.get((1, "cuda", False, 1))
        if single_gpu:
            mean, _, _ = single_gpu
            target = 50.0
            status = "PASS" if mean < target else "FAIL"
            if mean >= target:
                targets_passed = False
            print(f"  Single page GPU < 50ms: {mean:.0f}ms [{status}]")
        elif "cuda" in devices:
            print(f"  Single page GPU < 50ms: NO DATA")
            targets_passed = False

        # Target 2: Single page CPU < 200ms
        single_cpu = all_results.get((1, "cpu", False, 1))
        if single_cpu:
            mean, _, _ = single_cpu
            target = 200.0
            status = "PASS" if mean < target else "FAIL"
            if mean >= target:
                targets_passed = False
            print(f"  Single page CPU < 200ms: {mean:.0f}ms [{status}]")
        else:
            print(f"  Single page CPU < 200ms: NO DATA")

        # Target 3: 100-page throughput > 50 pages/sec on GPU
        # Find best GPU result for ~100 pages
        best_gpu_100 = None
        for (pages, device, batch, jobs), (mean, _, _) in all_results.items():
            if device == "cuda" and pages >= 50:
                throughput = (pages / mean) * 1000.0
                if best_gpu_100 is None or throughput > best_gpu_100:
                    best_gpu_100 = throughput

        if best_gpu_100:
            target = 50.0
            status = "PASS" if best_gpu_100 >= target else "FAIL"
            if best_gpu_100 < target:
                targets_passed = False
            print(f"  Multi-page GPU > 50 pages/s: {best_gpu_100:.1f} pages/s [{status}]")
        elif "cuda" in devices:
            print(f"  Multi-page GPU > 50 pages/s: NO DATA (test with --pages 100)")

        print()
        if targets_passed:
            print("  *** ALL TARGETS PASSED ***")
            return 0
        else:
            print("  *** SOME TARGETS NOT MET ***")
            return 1

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
