#!/usr/bin/env python3
# SPDX-FileCopyrightText: 2025 The unpaper authors
#
# SPDX-License-Identifier: GPL-2.0-only

"""
Batch processing benchmark for unpaper.

This script measures the performance of unpaper's batch processing mode
with different parallelism settings. It compares:
- Sequential processing (baseline)
- Single-threaded batch mode (--jobs=1)
- Multi-threaded batch mode (--jobs=N)

Usage:
    python tools/bench_batch.py --images 10
    python tools/bench_batch.py --images 50 --threads 1,2,4,8
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


def create_test_images(source_image: Path, output_dir: Path, count: int) -> list[Path]:
    """Create multiple copies of a source image for testing."""
    images = []
    for i in range(1, count + 1):
        dest = output_dir / f"input{i:04d}.png"
        shutil.copy(source_image, dest)
        images.append(dest)
    return images


def run_sequential(binary: Path, input_pattern: str, output_pattern: str,
                   count: int) -> float:
    """Run unpaper sequentially (no batch mode)."""
    start = time.perf_counter()

    for i in range(1, count + 1):
        input_file = input_pattern % i
        output_file = output_pattern % i

        # Remove output if exists
        if os.path.exists(output_file):
            os.unlink(output_file)

        proc = subprocess.run(
            [str(binary), input_file, output_file],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.PIPE,
            check=False,
        )
        if proc.returncode != 0:
            raise RuntimeError(
                f"Sequential unpaper failed: {proc.stderr.decode().strip()}"
            )

    return (time.perf_counter() - start) * 1000.0


def run_batch(binary: Path, input_pattern: str, output_pattern: str,
              jobs: int) -> float:
    """Run unpaper in batch mode with specified parallelism."""
    # Remove existing output files
    output_dir = Path(output_pattern).parent
    for f in output_dir.glob("output*.pbm"):
        f.unlink()

    start = time.perf_counter()

    cmd = [
        str(binary),
        "--batch",
        f"--jobs={jobs}",
        "--overwrite",
        input_pattern,
        output_pattern,
    ]

    proc = subprocess.run(
        cmd,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.PIPE,
        check=False,
    )

    elapsed = (time.perf_counter() - start) * 1000.0

    if proc.returncode != 0:
        raise RuntimeError(
            f"Batch unpaper (jobs={jobs}) failed: {proc.stderr.decode().strip()}"
        )

    return elapsed


def bench_configuration(binary: Path, input_pattern: str, output_pattern: str,
                        count: int, jobs: int | None, warmup: int,
                        iterations: int) -> tuple[float, float]:
    """Benchmark a specific configuration and return mean/stdev."""
    samples = []

    for i in range(warmup + iterations):
        try:
            if jobs is None:
                elapsed = run_sequential(binary, input_pattern, output_pattern, count)
            else:
                elapsed = run_batch(binary, input_pattern, output_pattern, jobs)

            if i >= warmup:
                samples.append(elapsed)
        except RuntimeError as e:
            print(f"  [ERROR] {e}", file=sys.stderr)
            return float("nan"), float("nan")

    mean = statistics.mean(samples)
    stdev = statistics.pstdev(samples) if len(samples) > 1 else 0.0
    return mean, stdev


def main() -> int:
    repo_root = Path(__file__).resolve().parent.parent
    default_input = repo_root / "tests" / "source_images" / "imgsrc001.png"

    parser = argparse.ArgumentParser(
        description="Benchmark unpaper batch processing performance."
    )
    parser.add_argument(
        "--builddir",
        default=repo_root / "builddir",
        type=Path,
        help="Meson builddir containing unpaper binary",
    )
    parser.add_argument(
        "--source",
        default=default_input,
        type=Path,
        help="Source image to replicate for batch testing",
    )
    parser.add_argument(
        "--images",
        default=10,
        type=int,
        help="Number of images to process in batch",
    )
    parser.add_argument(
        "--threads",
        default="1,2,4",
        help="Comma-separated list of thread counts to test",
    )
    parser.add_argument(
        "--warmup",
        default=0,
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
        "--sequential",
        action="store_true",
        help="Include sequential (non-batch) baseline",
    )

    args = parser.parse_args()

    binary = args.builddir / "unpaper"
    if not binary.exists():
        print(f"Binary not found: {binary}", file=sys.stderr)
        return 1

    if not args.source.exists():
        print(f"Source image not found: {args.source}", file=sys.stderr)
        return 1

    thread_counts = [int(t.strip()) for t in args.threads.split(",") if t.strip()]

    # Use tmpfs if available for better I/O performance
    tmpdir_base = Path("/dev/shm") if Path("/dev/shm").is_dir() else Path(tempfile.gettempdir())

    with tempfile.TemporaryDirectory(dir=tmpdir_base) as tmpdir:
        tmpdir = Path(tmpdir)

        print(f"Creating {args.images} test images...")
        create_test_images(args.source, tmpdir, args.images)

        input_pattern = str(tmpdir / "input%04d.png")
        output_pattern = str(tmpdir / "output%04d.pbm")

        print(f"\nBenchmarking with {args.images} images:")
        print(f"  Binary: {binary}")
        print(f"  Warmup: {args.warmup}, Iterations: {args.iterations}")
        print()

        results = []

        # Sequential baseline (optional)
        if args.sequential:
            print("Sequential (no batch)...", end=" ", flush=True)
            mean, stdev = bench_configuration(
                binary, input_pattern, output_pattern, args.images,
                None, args.warmup, args.iterations
            )
            print(f"{mean:.0f}ms (stdev={stdev:.0f}ms)")
            results.append(("sequential", mean, stdev))

        # Batch with different thread counts
        for threads in thread_counts:
            print(f"Batch (jobs={threads})...", end=" ", flush=True)
            mean, stdev = bench_configuration(
                binary, input_pattern, output_pattern, args.images,
                threads, args.warmup, args.iterations
            )
            print(f"{mean:.0f}ms (stdev={stdev:.0f}ms)")
            results.append((f"batch-j{threads}", mean, stdev))

        # Summary
        print("\n" + "=" * 60)
        print("Summary:")
        print("=" * 60)

        baseline = results[0][1] if results else 1.0
        for name, mean, stdev in results:
            speedup = baseline / mean if mean > 0 else 0
            per_image = mean / args.images
            print(f"  {name:<15} {mean:>8.0f}ms  "
                  f"({per_image:.0f}ms/image, {speedup:.2f}x vs baseline)")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
