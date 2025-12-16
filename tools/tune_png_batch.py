#!/usr/bin/env python3
# SPDX-FileCopyrightText: 2025 The unpaper authors
#
# SPDX-License-Identifier: GPL-2.0-only

"""
PNG batch processing parameter tuning for unpaper.

This script benchmarks different combinations of batch processing parameters
to find the optimal configuration for PNG files (which use FFmpeg CPU decode/encode).

Key parameters tuned:
- --jobs: Number of worker threads for processing
- --cuda-streams: Number of CUDA streams for GPU parallelism
- Decode/encode threads are auto-scaled internally based on these

For PNG files:
- Decode uses FFmpeg (CPU-bound)
- Processing uses CUDA (GPU-bound)
- Encode uses FFmpeg (CPU-bound)

The optimal balance depends on:
- CPU core count
- GPU compute capacity
- I/O bandwidth (SSD vs HDD)

Usage:
    python tools/tune_png_batch.py --images 50
    python tools/tune_png_batch.py --images 50 --quick
    python tools/tune_png_batch.py --images 100 --thorough
"""

import argparse
import os
import shutil
import statistics
import subprocess
import sys
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path


@dataclass
class BenchResult:
    jobs: int
    streams: int
    elapsed_ms: float
    per_image_ms: float
    throughput: float  # images/sec
    stdev_ms: float


def create_test_images(source_image: Path, output_dir: Path, count: int) -> list[Path]:
    """Create multiple copies of a source PNG image for testing."""
    images = []
    for i in range(1, count + 1):
        dest = output_dir / f"input{i:04d}.png"
        shutil.copy(source_image, dest)
        images.append(dest)
    return images


def check_cuda_available(binary: Path) -> bool:
    """Check if the binary supports CUDA and CUDA runtime is available."""
    proc = subprocess.run(
        [str(binary), "--help"],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )
    if b"--device=cpu|cuda" not in proc.stdout:
        return False
    # Try a quick CUDA run
    proc = subprocess.run(
        [str(binary), "--device=cuda", "--help"],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.PIPE,
        check=False,
    )
    return proc.returncode == 0


def get_cpu_cores() -> int:
    """Get number of CPU cores."""
    return os.cpu_count() or 8


def run_batch(binary: Path, input_pattern: str, output_pattern: str,
              jobs: int, streams: int, verbose: bool = False) -> float:
    """Run unpaper in batch mode with specified parameters."""
    # Remove existing output files
    output_dir = Path(output_pattern).parent
    for f in output_dir.glob("output*.pbm"):
        f.unlink()

    start = time.perf_counter()

    cmd = [
        str(binary),
        "--batch",
        f"--jobs={jobs}",
        "--device=cuda",
        f"--cuda-streams={streams}",
        "--overwrite",
        input_pattern,
        output_pattern,
    ]

    if verbose:
        cmd.append("--verbose")

    proc = subprocess.run(
        cmd,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.PIPE if not verbose else None,
        check=False,
    )

    elapsed = (time.perf_counter() - start) * 1000.0

    if proc.returncode != 0:
        stderr_text = proc.stderr.decode().strip() if proc.stderr else "unknown error"
        raise RuntimeError(f"Batch failed (jobs={jobs}, streams={streams}): {stderr_text}")

    return elapsed


def benchmark_config(binary: Path, input_pattern: str, output_pattern: str,
                     image_count: int, jobs: int, streams: int,
                     warmup: int = 0, iterations: int = 3) -> BenchResult:
    """Benchmark a specific configuration."""
    samples = []

    for i in range(warmup + iterations):
        try:
            elapsed = run_batch(binary, input_pattern, output_pattern, jobs, streams)
            if i >= warmup:
                samples.append(elapsed)
        except RuntimeError as e:
            print(f"  [ERROR] {e}", file=sys.stderr)
            return BenchResult(
                jobs=jobs, streams=streams,
                elapsed_ms=float("nan"), per_image_ms=float("nan"),
                throughput=0.0, stdev_ms=0.0
            )

    mean = statistics.mean(samples)
    stdev = statistics.pstdev(samples) if len(samples) > 1 else 0.0
    per_image = mean / image_count
    throughput = (image_count / mean) * 1000.0  # images/sec

    return BenchResult(
        jobs=jobs, streams=streams,
        elapsed_ms=mean, per_image_ms=per_image,
        throughput=throughput, stdev_ms=stdev
    )


def generate_parameter_grid(cpu_cores: int, mode: str) -> list[tuple[int, int]]:
    """Generate (jobs, streams) parameter combinations to test.

    Returns list of (jobs, streams) tuples.
    """
    if mode == "quick":
        # Quick: test a few key points
        jobs_list = [4, 8, cpu_cores]
        streams_list = [4, 8]
    elif mode == "thorough":
        # Thorough: comprehensive grid search
        jobs_list = list(range(2, min(cpu_cores * 2, 32) + 1, 2))
        streams_list = [1, 2, 4, 6, 8, 12, 16]
    else:
        # Default: balanced coverage
        jobs_list = [2, 4, 6, 8, 10, 12, 16]
        if cpu_cores >= 16:
            jobs_list.extend([20, 24])
        streams_list = [2, 4, 6, 8]

    grid = []
    for jobs in jobs_list:
        for streams in streams_list:
            grid.append((jobs, streams))

    return grid


def main() -> int:
    repo_root = Path(__file__).resolve().parent.parent
    default_input = repo_root / "tests" / "source_images" / "imgsrc001.png"

    parser = argparse.ArgumentParser(
        description="Tune PNG batch processing parameters for optimal performance."
    )
    parser.add_argument(
        "--builddir",
        default=repo_root / "builddir-cuda",
        type=Path,
        help="Meson builddir containing unpaper binary",
    )
    parser.add_argument(
        "--source",
        default=default_input,
        type=Path,
        help="Source PNG image to replicate for testing",
    )
    parser.add_argument(
        "--images",
        default=50,
        type=int,
        help="Number of images to process in batch (default: 50)",
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
        "--quick",
        action="store_true",
        help="Quick mode: test fewer parameter combinations",
    )
    parser.add_argument(
        "--thorough",
        action="store_true",
        help="Thorough mode: comprehensive parameter search",
    )
    parser.add_argument(
        "--jobs",
        default="",
        help="Override: comma-separated jobs values to test",
    )
    parser.add_argument(
        "--streams",
        default="",
        help="Override: comma-separated streams values to test",
    )

    args = parser.parse_args()

    binary = args.builddir / "unpaper"
    if not binary.exists():
        print(f"Binary not found: {binary}", file=sys.stderr)
        return 1

    if not args.source.exists():
        print(f"Source image not found: {args.source}", file=sys.stderr)
        return 1

    if not check_cuda_available(binary):
        print("ERROR: CUDA not available", file=sys.stderr)
        return 1

    cpu_cores = get_cpu_cores()
    print(f"{'='*70}")
    print(f"PNG Batch Processing Parameter Tuning")
    print(f"{'='*70}")
    print(f"CPU cores: {cpu_cores}")
    print(f"Images: {args.images}")
    print(f"Warmup: {args.warmup}, Iterations: {args.iterations}")
    print()

    # Determine mode
    if args.quick:
        mode = "quick"
    elif args.thorough:
        mode = "thorough"
    else:
        mode = "default"

    # Generate parameter grid
    if args.jobs and args.streams:
        jobs_list = [int(x.strip()) for x in args.jobs.split(",")]
        streams_list = [int(x.strip()) for x in args.streams.split(",")]
        grid = [(j, s) for j in jobs_list for s in streams_list]
    else:
        grid = generate_parameter_grid(cpu_cores, mode)

    print(f"Testing {len(grid)} parameter combinations ({mode} mode)")
    print()

    # Use tmpfs if available
    tmpdir_base = Path("/dev/shm") if Path("/dev/shm").is_dir() else Path(tempfile.gettempdir())

    results: list[BenchResult] = []

    with tempfile.TemporaryDirectory(dir=tmpdir_base) as tmpdir:
        tmpdir = Path(tmpdir)

        # Create test images
        print(f"Creating {args.images} test images...", end=" ", flush=True)
        create_test_images(args.source, tmpdir, args.images)
        print("done")
        print()

        input_pattern = str(tmpdir / "input%04d.png")
        output_pattern = str(tmpdir / "output%04d.pbm")

        # Run benchmarks
        print(f"{'Jobs':<6} {'Streams':<8} {'Time (ms)':<12} {'ms/img':<10} {'img/s':<10} {'stdev':<10}")
        print("-" * 60)

        for jobs, streams in grid:
            result = benchmark_config(
                binary, input_pattern, output_pattern, args.images,
                jobs, streams, args.warmup, args.iterations
            )
            results.append(result)

            if result.throughput > 0:
                print(f"{jobs:<6} {streams:<8} {result.elapsed_ms:<12.0f} "
                      f"{result.per_image_ms:<10.1f} {result.throughput:<10.2f} "
                      f"{result.stdev_ms:<10.1f}")
            else:
                print(f"{jobs:<6} {streams:<8} FAILED")

    # Find best configuration
    valid_results = [r for r in results if r.throughput > 0]
    if not valid_results:
        print("\nNo valid results!")
        return 1

    # Sort by throughput (higher is better)
    valid_results.sort(key=lambda r: r.throughput, reverse=True)

    print()
    print("=" * 70)
    print("RESULTS SUMMARY")
    print("=" * 70)
    print()

    # Top 5 configurations
    print("Top 5 configurations (by throughput):")
    print("-" * 60)
    print(f"{'Rank':<6} {'Jobs':<6} {'Streams':<8} {'ms/img':<10} {'img/s':<10}")
    print("-" * 60)
    for i, r in enumerate(valid_results[:5], 1):
        print(f"{i:<6} {r.jobs:<6} {r.streams:<8} {r.per_image_ms:<10.1f} {r.throughput:<10.2f}")

    best = valid_results[0]
    print()
    print(f"OPTIMAL CONFIGURATION:")
    print(f"  --jobs={best.jobs} --cuda-streams={best.streams}")
    print()
    print(f"  Throughput: {best.throughput:.2f} images/sec")
    print(f"  Per-image:  {best.per_image_ms:.1f} ms")
    print(f"  Total time: {best.elapsed_ms:.0f} ms for {args.images} images")
    print()

    # Analysis
    print("ANALYSIS:")
    print("-" * 60)

    # Find best by jobs count
    jobs_best = {}
    for r in valid_results:
        if r.jobs not in jobs_best or r.throughput > jobs_best[r.jobs].throughput:
            jobs_best[r.jobs] = r

    print("Best streams per jobs setting:")
    for jobs in sorted(jobs_best.keys()):
        r = jobs_best[jobs]
        print(f"  jobs={jobs}: streams={r.streams} ({r.throughput:.2f} img/s)")

    # Find best by streams count
    streams_best = {}
    for r in valid_results:
        if r.streams not in streams_best or r.throughput > streams_best[r.streams].throughput:
            streams_best[r.streams] = r

    print()
    print("Best jobs per streams setting:")
    for streams in sorted(streams_best.keys()):
        r = streams_best[streams]
        print(f"  streams={streams}: jobs={r.jobs} ({r.throughput:.2f} img/s)")

    # Recommendations
    print()
    print("RECOMMENDATIONS FOR YOUR SYSTEM:")
    print("-" * 60)
    print(f"For {args.images} PNG images with {cpu_cores} CPU cores:")
    print()
    print(f"  Best overall: --jobs={best.jobs} --cuda-streams={best.streams}")
    print()

    # Check if there's a simpler config that's nearly as good (within 5%)
    for r in valid_results[1:10]:
        if r.throughput >= best.throughput * 0.95:
            if r.jobs < best.jobs or r.streams < best.streams:
                print(f"  Alternative (similar performance, simpler): "
                      f"--jobs={r.jobs} --cuda-streams={r.streams}")
                break

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
