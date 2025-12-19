#!/usr/bin/env python3
# SPDX-FileCopyrightText: 2025 The unpaper authors
#
# SPDX-License-Identifier: GPL-2.0-only

"""
JPEG GPU Pipeline benchmark for unpaper.

This script measures the performance improvement from the full GPU pipeline
(JPEG decode -> GPU processing -> JPEG encode) vs the standard path
(JPEG decode -> GPU processing -> D2H transfer -> CPU encode).

The GPU pipeline is auto-enabled when output files are JPEG format.

Key measurements:
- Standard CUDA batch: D2H transfer + CPU encode (PBM output)
- GPU pipeline: JPEG-to-JPEG zero-copy path (JPEG output, auto-detected)
- Performance gain from eliminating D2H transfer

Usage:
    python tools/bench_jpeg_pipeline.py --images 50
    python tools/bench_jpeg_pipeline.py --images 50 --verify-speedup
    python tools/bench_jpeg_pipeline.py --images 100 --jpeg-quality 85
"""

import argparse
import statistics
import subprocess
import sys
import tempfile
import time
from pathlib import Path


def create_test_jpegs(source_image: Path, output_dir: Path, count: int) -> list[Path]:
    """Create JPEG test images from a source image."""
    images = []

    try:
        from PIL import Image
        src_img = Image.open(source_image)
        if src_img.mode == "RGBA":
            src_img = src_img.convert("RGB")
        for i in range(1, count + 1):
            dest = output_dir / f"input{i:04d}.jpg"
            src_img.save(dest, "JPEG", quality=95)
            images.append(dest)
        return images
    except ImportError:
        print("ERROR: PIL/Pillow required for JPEG conversion", file=sys.stderr)
        sys.exit(1)


def check_cuda_available(binary: Path) -> bool:
    """Check if the binary supports CUDA."""
    proc = subprocess.run(
        [str(binary), "--help"],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )
    return b"--device=cpu|cuda" in proc.stdout


def run_batch(binary: Path, input_pattern: str, output_pattern: str,
              jobs: int, streams: int = 8, jpeg_quality: int = 85) -> float:
    """Run unpaper in batch mode with CUDA.

    GPU encode is auto-enabled when output files have .jpg extension.

    Args:
        binary: Path to unpaper binary
        input_pattern: Input file pattern with %d placeholder (must be JPEG)
        output_pattern: Output file pattern with %d placeholder
        jobs: Number of parallel workers
        streams: Number of CUDA streams
        jpeg_quality: JPEG quality for JPEG output
    """
    # Remove existing output files
    output_dir = Path(output_pattern).parent
    for f in list(output_dir.glob("output*.pbm")) + list(output_dir.glob("output*.jpg")):
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

    # Add JPEG quality for JPEG outputs
    if output_pattern.endswith(".jpg"):
        cmd.insert(5, f"--jpeg-quality={jpeg_quality}")

    proc = subprocess.run(
        cmd,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.PIPE,
        check=False,
    )

    elapsed = (time.perf_counter() - start) * 1000.0

    if proc.returncode != 0:
        raise RuntimeError(
            f"Batch unpaper failed: {proc.stderr.decode().strip()}"
        )

    return elapsed


def bench_configuration(binary: Path, input_pattern: str, output_pattern: str,
                        count: int, warmup: int, iterations: int,
                        jpeg_quality: int = 85, jobs: int = 8,
                        streams: int = 8) -> tuple[float, float]:
    """Benchmark a specific configuration and return mean/stdev."""
    samples = []

    for i in range(warmup + iterations):
        try:
            elapsed = run_batch(binary, input_pattern, output_pattern,
                                jobs, streams, jpeg_quality)

            if i >= warmup:
                samples.append(elapsed)
        except RuntimeError as e:
            print(f"  [ERROR] {e}", file=sys.stderr)
            return float("nan"), float("nan")

    mean = statistics.mean(samples)
    stdev = statistics.pstdev(samples) if len(samples) > 1 else 0.0
    return mean, stdev


def get_output_file_sizes(output_pattern: str, count: int) -> tuple[int, int]:
    """Get total and average output file size in bytes."""
    total_size = 0
    found = 0
    output_dir = Path(output_pattern).parent

    # Check for both .pbm and .jpg outputs
    for ext in [".pbm", ".jpg"]:
        pattern = output_pattern.replace("%04d", "*").replace(".pbm", ext).replace(".jpg", ext)
        for f in output_dir.glob(Path(pattern).name):
            total_size += f.stat().st_size
            found += 1

    avg_size = total_size // found if found > 0 else 0
    return total_size, avg_size


def main() -> int:
    repo_root = Path(__file__).resolve().parent.parent
    default_input = repo_root / "tests" / "source_images" / "imgsrc001.png"

    parser = argparse.ArgumentParser(
        description="Benchmark unpaper JPEG GPU pipeline performance."
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
        default=85,
        type=int,
        help="JPEG quality for GPU pipeline output (1-100, default: 85)",
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
        "--verify-speedup",
        action="store_true",
        help="Verify GPU pipeline achieves performance improvement",
    )

    args = parser.parse_args()

    binary = args.builddir / "unpaper"
    if not binary.exists():
        print(f"Binary not found: {binary}", file=sys.stderr)
        return 1

    if not args.source.exists():
        print(f"Source image not found: {args.source}", file=sys.stderr)
        return 1

    # Check prerequisites
    if not check_cuda_available(binary):
        print("ERROR: CUDA support required for GPU pipeline benchmark", file=sys.stderr)
        return 1

    # Use tmpfs if available for better I/O performance
    tmpdir_base = Path("/dev/shm") if Path("/dev/shm").is_dir() else Path(tempfile.gettempdir())

    print(f"{'='*70}")
    print(f"JPEG GPU Pipeline Benchmark")
    print(f"{'='*70}")
    print(f"  Binary: {binary}")
    print(f"  Images: {args.images}")
    print(f"  JPEG quality: {args.jpeg_quality}")
    print(f"  Warmup: {args.warmup}, Iterations: {args.iterations}")
    print(f"  Note: GPU encode is auto-enabled for JPEG outputs")
    print()

    with tempfile.TemporaryDirectory(dir=tmpdir_base) as tmpdir:
        tmpdir = Path(tmpdir)

        # Create JPEG test images
        print(f"Creating {args.images} JPEG test images...", end=" ", flush=True)
        create_test_jpegs(args.source, tmpdir, args.images)
        print("done")
        print()

        input_pattern = str(tmpdir / "input%04d.jpg")

        # Test 1: Standard path (JPEG -> GPU -> D2H -> CPU encode to PBM)
        output_pattern_pbm = str(tmpdir / "output%04d.pbm")
        print("Standard CUDA batch (JPEG -> GPU -> D2H -> PBM)...", end=" ", flush=True)
        std_mean, std_stdev = bench_configuration(
            binary, input_pattern, output_pattern_pbm,
            args.images, args.warmup, args.iterations,
            jobs=args.jobs, streams=args.streams,
        )
        print(f"{std_mean:.0f}ms (stdev={std_stdev:.0f}ms)")

        # Get output sizes for standard path
        std_total, std_avg = get_output_file_sizes(output_pattern_pbm, args.images)

        # Test 2: GPU pipeline (JPEG -> GPU -> nvJPEG encode -> JPEG)
        # GPU encode is auto-enabled because output is .jpg
        output_pattern_jpg = str(tmpdir / "output%04d.jpg")
        print("GPU pipeline (JPEG -> GPU -> JPEG, auto-enabled)...", end=" ", flush=True)
        gpu_mean, gpu_stdev = bench_configuration(
            binary, input_pattern, output_pattern_jpg,
            args.images, args.warmup, args.iterations,
            jpeg_quality=args.jpeg_quality,
            jobs=args.jobs, streams=args.streams,
        )
        print(f"{gpu_mean:.0f}ms (stdev={gpu_stdev:.0f}ms)")

        # Get output sizes for GPU pipeline
        gpu_total, gpu_avg = get_output_file_sizes(output_pattern_jpg, args.images)

        # Results summary
        print()
        print(f"{'-'*70}")
        print(f"Results Summary")
        print(f"{'-'*70}")

        if std_mean > 0 and gpu_mean > 0:
            std_per_img = std_mean / args.images
            gpu_per_img = gpu_mean / args.images
            speedup = std_mean / gpu_mean
            time_saved = std_mean - gpu_mean
            time_saved_per_img = time_saved / args.images

            print(f"  Standard path:  {std_mean:>8.0f}ms total ({std_per_img:>6.1f}ms/img)")
            print(f"  GPU pipeline:   {gpu_mean:>8.0f}ms total ({gpu_per_img:>6.1f}ms/img)")
            print()
            print(f"  Speedup:        {speedup:>8.2f}x")
            print(f"  Time saved:     {time_saved:>8.0f}ms ({time_saved_per_img:>6.1f}ms/img)")
            print()
            print(f"  Output sizes:")
            print(f"    Standard (PBM): {std_total/(1024*1024):>6.1f} MB total ({std_avg/1024:>5.1f} KB/img)")
            print(f"    GPU (JPEG):     {gpu_total/(1024*1024):>6.1f} MB total ({gpu_avg/1024:>5.1f} KB/img)")

            # Verify speedup if requested
            if args.verify_speedup:
                print()
                print(f"{'-'*70}")
                print(f"Speedup Verification")
                print(f"{'-'*70}")

                # Target: GPU pipeline should be at least as fast (speedup >= 1.0)
                # Ideally faster due to eliminated D2H transfer
                target = 1.0
                status = "PASS" if speedup >= target else "FAIL"

                print(f"  GPU pipeline speedup: {speedup:.2f}x (target: >= {target}x) [{status}]")

                if speedup >= target:
                    print()
                    print("  *** GPU PIPELINE PERFORMANCE VERIFIED ***")
                    return 0
                else:
                    print()
                    print("  *** GPU PIPELINE SLOWER THAN STANDARD PATH ***")
                    print("  Note: This may indicate nvJPEG encode overhead or fallback to CPU.")
                    return 1
        else:
            print("  ERROR: One or both benchmarks failed")
            return 1

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
