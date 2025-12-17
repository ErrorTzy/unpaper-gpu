#!/usr/bin/env python3
# SPDX-FileCopyrightText: 2025 The unpaper authors
# SPDX-License-Identifier: GPL-2.0-only

"""
Benchmark JBIG2 PDF batch processing.

Usage: python bench_jbig2_pdf.py [--device cpu|cuda] [--runs N] [--pdf PATH]

Measures processing time for multi-page JBIG2 PDFs.
"""

import argparse
import os
import shutil
import subprocess
import sys
import tempfile
import time
from pathlib import Path


def find_unpaper() -> str:
    """Find unpaper binary in build directories."""
    candidates = [
        "builddir-cuda-pdf/unpaper",
        "builddir-cuda/unpaper",
        "builddir-pdf/unpaper",
        "builddir/unpaper",
    ]
    for c in candidates:
        if os.path.exists(c):
            return c
    raise FileNotFoundError("No unpaper binary found. Build with: meson compile -C builddir-cuda-pdf/")


def run_benchmark(unpaper_path: str, pdf_path: str, device: str, runs: int = 3) -> dict:
    """Run benchmark and return timing statistics."""
    results = {
        "device": device,
        "pdf_path": pdf_path,
        "runs": runs,
        "times": [],
        "pages_per_sec": [],
    }

    # Get page count
    # Use pdfinfo or parse from unpaper output
    cmd = [unpaper_path, "-v", pdf_path, "/dev/null"]
    proc = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
    output = proc.stdout + proc.stderr

    page_count = 50  # Default
    for line in output.split('\n'):
        if 'pages to process' in line:
            try:
                page_count = int(line.split()[2])
            except (ValueError, IndexError):
                pass
            break

    results["page_count"] = page_count

    print(f"\nBenchmark: {pdf_path}")
    print(f"Device: {device}")
    print(f"Pages: {page_count}")
    print(f"Runs: {runs}")
    print("-" * 50)

    for i in range(runs):
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp_path = os.path.join(tmpdir, "output.pdf")

            cmd = [unpaper_path, f"--device={device}", pdf_path, tmp_path]

            start = time.perf_counter()
            proc = subprocess.run(cmd, capture_output=True, text=True, timeout=600)
            elapsed = time.perf_counter() - start

            if proc.returncode != 0:
                print(f"Run {i+1}: FAILED")
                print(f"  stderr: {proc.stderr[:500]}")
                continue

            pages_per_sec = page_count / elapsed
            results["times"].append(elapsed)
            results["pages_per_sec"].append(pages_per_sec)

            print(f"Run {i+1}: {elapsed:.3f}s ({pages_per_sec:.1f} pages/sec)")

    if results["times"]:
        results["mean_time"] = sum(results["times"]) / len(results["times"])
        results["mean_pps"] = sum(results["pages_per_sec"]) / len(results["pages_per_sec"])
        results["min_time"] = min(results["times"])
        results["max_time"] = max(results["times"])

        print("-" * 50)
        print(f"Mean: {results['mean_time']:.3f}s ({results['mean_pps']:.1f} pages/sec)")
        print(f"Best: {results['min_time']:.3f}s ({page_count / results['min_time']:.1f} pages/sec)")

    return results


def main():
    parser = argparse.ArgumentParser(description="Benchmark JBIG2 PDF processing")
    parser.add_argument("--device", choices=["cpu", "cuda"], default="cpu",
                       help="Device to use (default: cpu)")
    parser.add_argument("--runs", type=int, default=3,
                       help="Number of benchmark runs (default: 3)")
    parser.add_argument("--pdf", type=str,
                       default="tests/pdf_samples/benchmark_jbig2_50page.pdf",
                       help="PDF file to benchmark")
    args = parser.parse_args()

    if not os.path.exists(args.pdf):
        print(f"Error: PDF not found: {args.pdf}")
        print("Create it with: python tools/create_jbig2_benchmark_pdf.py")
        sys.exit(1)

    unpaper_path = find_unpaper()
    print(f"Using: {unpaper_path}")

    # Check if CUDA is available when requested
    if args.device == "cuda":
        proc = subprocess.run([unpaper_path, "--help"], capture_output=True, text=True)
        if "cuda" not in proc.stdout.lower():
            print("Warning: CUDA support may not be available in this build")

    results = run_benchmark(unpaper_path, args.pdf, args.device, args.runs)

    # Summary
    print("\n" + "=" * 50)
    print("BENCHMARK SUMMARY")
    print("=" * 50)
    print(f"PDF: {results['pdf_path']}")
    print(f"Device: {results['device']}")
    print(f"Pages: {results['page_count']}")
    if results.get("mean_time"):
        print(f"Mean time: {results['mean_time']:.3f}s")
        print(f"Throughput: {results['mean_pps']:.1f} pages/sec")
        print(f"Per-page latency: {results['mean_time'] / results['page_count'] * 1000:.1f}ms")


if __name__ == "__main__":
    main()
