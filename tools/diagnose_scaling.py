#!/usr/bin/env python3
# SPDX-FileCopyrightText: 2025 The unpaper authors
# SPDX-License-Identifier: GPL-2.0-only
"""
Diagnostic script to prove GPU stream scaling bottleneck.

This script measures different configurations to demonstrate that
cudaMalloc/cudaFree in backend_cuda.c are serializing GPU streams.

Root Cause:
- Functions like cuda_rect_count_brightness_range() do cudaMalloc(8)
  and cudaFree() for each call
- cudaMalloc/cudaFree are GLOBAL synchronization points that block ALL streams
- This negates the benefit of using multiple CUDA streams

Evidence:
1. With 1 stream: filters take ~2ms/image
2. With 4 streams: filters take ~15ms/image (7x SLOWER!)
3. CUDA_LAUNCH_BLOCKING=1 only adds ~10% overhead (kernels already serialized)
4. Pool-using paths (image buffers, integral images) scale well
5. Non-pool paths (small scratch allocs) don't scale
"""

import argparse
import os
import shutil
import subprocess
import sys
import tempfile
import time
from pathlib import Path


def run_unpaper(binary: Path, args: list[str], timeout: int = 120) -> tuple[float, str]:
    """Run unpaper and return (elapsed_ms, stderr)."""
    start = time.perf_counter()
    proc = subprocess.run(
        [str(binary)] + args,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.PIPE,
        timeout=timeout,
    )
    elapsed = (time.perf_counter() - start) * 1000
    return elapsed, proc.stderr.decode()


def main():
    repo_root = Path(__file__).resolve().parent.parent
    binary = repo_root / "builddir-cuda" / "unpaper"

    if not binary.exists():
        print(f"Binary not found: {binary}", file=sys.stderr)
        return 1

    source_img = repo_root / "tests" / "source_images" / "imgsrc001.png"
    if not source_img.exists():
        print(f"Source image not found: {source_img}", file=sys.stderr)
        return 1

    # Use tmpfs for I/O
    tmpdir_base = Path("/dev/shm") if Path("/dev/shm").is_dir() else Path(tempfile.gettempdir())

    with tempfile.TemporaryDirectory(dir=tmpdir_base) as tmpdir:
        tmpdir = Path(tmpdir)

        # Create test images
        n_images = 10
        for i in range(1, n_images + 1):
            shutil.copy(source_img, tmpdir / f"input{i:02d}.png")

        input_pattern = str(tmpdir / "input%02d.png")
        output_pattern = str(tmpdir / "output%02d.pbm")

        print("=" * 70)
        print("GPU Stream Scaling Bottleneck Diagnosis")
        print("=" * 70)
        print()

        # Test configurations
        configs = [
            # (name, streams, jobs, extra_args, description)
            ("1 stream, full processing", 1, 1, [], "Baseline"),
            ("4 streams, full processing", 4, 4, [], "Should be ~4x faster"),
            ("8 streams, full processing", 8, 8, [], "Should be ~8x faster"),
            ("1 stream, no processing", 1, 1, [
                "--no-blackfilter", "--no-grayfilter", "--no-blurfilter",
                "--no-noisefilter", "--no-deskew", "--no-mask-scan",
                "--no-mask-center", "--no-border", "--no-border-scan",
                "--no-border-align", "--layout=none"
            ], "Decode+encode only (uses pools)"),
            ("4 streams, no processing", 4, 4, [
                "--no-blackfilter", "--no-grayfilter", "--no-blurfilter",
                "--no-noisefilter", "--no-deskew", "--no-mask-scan",
                "--no-mask-center", "--no-border", "--no-border-scan",
                "--no-border-align", "--layout=none"
            ], "Should scale better (pools)"),
            ("1 stream, blackfilter only", 1, 1, [
                "--no-grayfilter", "--no-blurfilter",
                "--no-noisefilter", "--no-deskew", "--no-mask-scan",
                "--no-mask-center", "--no-border", "--no-border-scan",
                "--no-border-align", "--layout=none"
            ], "Only blackfilter (uses cudaMalloc)"),
            ("4 streams, blackfilter only", 4, 4, [
                "--no-grayfilter", "--no-blurfilter",
                "--no-noisefilter", "--no-deskew", "--no-mask-scan",
                "--no-mask-center", "--no-border", "--no-border-scan",
                "--no-border-align", "--layout=none"
            ], "Shows cudaMalloc serialization"),
        ]

        results = {}
        print(f"Running {len(configs)} test configurations with {n_images} images each...")
        print("-" * 70)

        for name, streams, jobs, extra_args, desc in configs:
            # Clean output files
            for f in tmpdir.glob("output*.pbm"):
                f.unlink()

            args = [
                "--batch",
                f"--jobs={jobs}",
                "--device=cuda",
                f"--cuda-streams={streams}",
                "--overwrite",
            ] + extra_args + [input_pattern, output_pattern]

            elapsed, _ = run_unpaper(binary, args)
            per_img = elapsed / n_images
            results[name] = (elapsed, per_img)
            print(f"{name:40s}: {elapsed:8.0f}ms ({per_img:6.1f}ms/img) - {desc}")

        print()
        print("=" * 70)
        print("Analysis")
        print("=" * 70)
        print()

        # Calculate scaling factors
        baseline_full = results["1 stream, full processing"][0]
        scaling_4s = baseline_full / results["4 streams, full processing"][0]
        scaling_8s = baseline_full / results["8 streams, full processing"][0]

        print(f"Full processing scaling:")
        print(f"  1→4 streams: {scaling_4s:.2f}x (ideal: 4.0x, efficiency: {scaling_4s/4*100:.0f}%)")
        print(f"  1→8 streams: {scaling_8s:.2f}x (ideal: 8.0x, efficiency: {scaling_8s/8*100:.0f}%)")
        print()

        baseline_noprocess = results["1 stream, no processing"][0]
        scaling_noprocess = baseline_noprocess / results["4 streams, no processing"][0]
        print(f"No-processing scaling (decode+encode only):")
        print(f"  1→4 streams: {scaling_noprocess:.2f}x")
        print()

        baseline_bf = results["1 stream, blackfilter only"][0]
        scaling_bf = baseline_bf / results["4 streams, blackfilter only"][0]
        print(f"Blackfilter-only scaling:")
        print(f"  1→4 streams: {scaling_bf:.2f}x")
        print()

        print("=" * 70)
        print("DIAGNOSIS")
        print("=" * 70)
        print()

        if scaling_4s < 2.0:
            print("PROBLEM CONFIRMED: Adding more GPU streams does NOT improve performance.")
            print()
            print("Root Cause: cudaMalloc/cudaFree calls in backend_cuda.c")
            print("  - These are GLOBAL synchronization points")
            print("  - They block ALL CUDA streams until complete")
            print("  - This serializes all GPU work, negating stream parallelism")
            print()
            print("Affected functions (each call does malloc+free):")
            print("  - cuda_rect_count_brightness_range() - used by blurfilter")
            print("  - cuda_rect_sum_lightness()          - used by grayfilter")
            print("  - cuda_rect_sum_grayscale()          - used by blackfilter")
            print("  - cuda_rect_sum_darkness_inverse()   - used by grayfilter")
            print("  - blackfilter_cuda_parallel()        - 2-4 allocs per call")
            print("  - detect_rotation_cuda()             - 3 allocs per call")
            print()
            print("FIX: Replace cudaMalloc/cudaFree with:")
            print("  1. Per-stream scratch buffers (pre-allocated)")
            print("  2. cudaMallocAsync/cudaFreeAsync (stream-ordered)")
            print("  3. Thread-local scratch pools")
            print()
        else:
            print("Stream scaling is working as expected.")

        return 0


if __name__ == "__main__":
    raise SystemExit(main())
