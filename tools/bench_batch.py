#!/usr/bin/env python3
# SPDX-FileCopyrightText: 2025 The unpaper authors
#
# SPDX-License-Identifier: GPL-2.0-only

"""
Batch processing benchmark for unpaper.

This script measures the performance of unpaper's batch processing mode
with different parallelism settings and devices. It compares:
- Sequential processing (baseline)
- Single-threaded batch mode (--jobs=1)
- Multi-threaded batch mode (--jobs=N)
- CUDA-accelerated batch mode (if available)
- GPU stream scaling (--streams mode)

Usage:
    python tools/bench_batch.py --images 10
    python tools/bench_batch.py --images 100 --threads 1,4,8 --devices cpu,cuda
    python tools/bench_batch.py --images 50,100 --devices cuda --verify-10x
    python tools/bench_batch.py --images 50 --devices cuda --streams 4,8,16,28
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


def create_test_images(source_image: Path, output_dir: Path, count: int,
                       output_format: str = "png") -> list[Path]:
    """Create multiple copies of a source image for testing.

    Args:
        source_image: Source image to copy/convert
        output_dir: Directory to create images in
        count: Number of images to create
        output_format: Output format ("png" or "jpg")
    """
    images = []
    ext = "jpg" if output_format == "jpg" else "png"

    # If converting to JPEG, use PIL
    if output_format == "jpg" and source_image.suffix.lower() != ".jpg":
        try:
            from PIL import Image
            src_img = Image.open(source_image)
            if src_img.mode == "RGBA":
                src_img = src_img.convert("RGB")
            for i in range(1, count + 1):
                dest = output_dir / f"input{i:04d}.{ext}"
                src_img.save(dest, "JPEG", quality=95)
                images.append(dest)
            return images
        except ImportError:
            print("WARNING: PIL not available, falling back to PNG", file=sys.stderr)
            ext = "png"

    # Simple copy for same format or PNG fallback
    for i in range(1, count + 1):
        dest = output_dir / f"input{i:04d}.{ext}"
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
    # Try a quick CUDA run to verify runtime
    proc = subprocess.run(
        [str(binary), "--device=cuda", "--help"],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.PIPE,
        check=False,
    )
    return proc.returncode == 0


def run_sequential(binary: Path, input_pattern: str, output_pattern: str,
                   count: int, device: str = "cpu") -> float:
    """Run unpaper sequentially (no batch mode)."""
    start = time.perf_counter()

    for i in range(1, count + 1):
        input_file = input_pattern % i
        output_file = output_pattern % i

        # Remove output if exists
        if os.path.exists(output_file):
            os.unlink(output_file)

        cmd = [str(binary)]
        if device != "cpu":
            cmd.extend([f"--device={device}"])
        cmd.extend([input_file, output_file])

        proc = subprocess.run(
            cmd,
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
              jobs: int, device: str = "cpu", streams: int | None = None,
              no_processing: bool = False) -> float:
    """Run unpaper in batch mode with specified parallelism.

    Args:
        binary: Path to unpaper binary
        input_pattern: Input file pattern with %d placeholder
        output_pattern: Output file pattern with %d placeholder
        jobs: Number of parallel workers
        device: Device to use (cpu or cuda)
        streams: Number of CUDA streams (None = auto)
        no_processing: If True, disable all processing filters (decode-only test)
    """
    # Remove existing output files
    output_dir = Path(output_pattern).parent
    for f in output_dir.glob("output*.pbm"):
        f.unlink()

    start = time.perf_counter()

    cmd = [
        str(binary),
        "--batch",
        f"--jobs={jobs}",
        f"--device={device}",
        "--overwrite",
        input_pattern,
        output_pattern,
    ]

    # Add stream count for CUDA device
    if streams is not None and device == "cuda":
        cmd.insert(4, f"--cuda-streams={streams}")

    # Disable all processing filters for decode-only benchmarks
    if no_processing:
        cmd.extend([
            "--no-blackfilter", "--no-grayfilter", "--no-blurfilter",
            "--no-noisefilter", "--no-deskew", "--no-mask-scan",
            "--no-mask-center", "--no-border", "--no-border-scan",
            "--no-border-align", "--layout=none"
        ])

    proc = subprocess.run(
        cmd,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.PIPE,
        check=False,
    )

    elapsed = (time.perf_counter() - start) * 1000.0

    if proc.returncode != 0:
        raise RuntimeError(
            f"Batch unpaper (jobs={jobs}, device={device}, streams={streams}) failed: "
            f"{proc.stderr.decode().strip()}"
        )

    return elapsed


def bench_configuration(binary: Path, input_pattern: str, output_pattern: str,
                        count: int, jobs: int | None, warmup: int,
                        iterations: int, device: str = "cpu",
                        streams: int | None = None,
                        no_processing: bool = False) -> tuple[float, float]:
    """Benchmark a specific configuration and return mean/stdev."""
    samples = []

    for i in range(warmup + iterations):
        try:
            if jobs is None:
                elapsed = run_sequential(binary, input_pattern, output_pattern, count, device)
            else:
                elapsed = run_batch(binary, input_pattern, output_pattern, jobs, device, streams,
                                    no_processing)

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
        default=repo_root / "builddir-cuda",
        type=Path,
        help="Meson builddir containing unpaper binary (default: builddir-cuda for CUDA support)",
    )
    parser.add_argument(
        "--source",
        default=default_input,
        type=Path,
        help="Source image to replicate for batch testing",
    )
    parser.add_argument(
        "--images",
        default="10",
        help="Comma-separated list of image counts to test (e.g., '10,50,100')",
    )
    parser.add_argument(
        "--threads",
        default="1,4,8",
        help="Comma-separated list of thread counts to test",
    )
    parser.add_argument(
        "--devices",
        default="cpu",
        help="Comma-separated list of devices to test (cpu, cuda)",
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
        help="Include sequential (non-batch) CPU baseline for speedup comparison",
    )
    parser.add_argument(
        "--verify-10x",
        action="store_true",
        help="Verify that CUDA batch achieves 10x speedup over sequential CPU",
    )
    parser.add_argument(
        "--streams",
        default="",
        help="Comma-separated list of CUDA stream counts to test (e.g., '4,8,16,28'). "
             "When specified, tests stream scaling with jobs=streams for each stream count.",
    )
    parser.add_argument(
        "--verify-linear",
        action="store_true",
        help="Verify near-linear scaling with GPU streams (requires --streams)",
    )
    parser.add_argument(
        "--jpeg",
        action="store_true",
        help="Use JPEG format for test images (converts PNG source to JPEG)",
    )
    parser.add_argument(
        "--no-processing",
        action="store_true",
        help="Disable all processing filters (decode-only benchmark)",
    )

    args = parser.parse_args()

    binary = args.builddir / "unpaper"
    if not binary.exists():
        # Try fallback to builddir
        binary = repo_root / "builddir" / "unpaper"
        if not binary.exists():
            print(f"Binary not found: {args.builddir / 'unpaper'}", file=sys.stderr)
            return 1

    if not args.source.exists():
        print(f"Source image not found: {args.source}", file=sys.stderr)
        return 1

    thread_counts = [int(t.strip()) for t in args.threads.split(",") if t.strip()]
    devices = [d.strip() for d in args.devices.split(",") if d.strip()]
    image_counts = [int(c.strip()) for c in args.images.split(",") if c.strip()]
    stream_counts = [int(s.strip()) for s in args.streams.split(",") if s.strip()]

    # Check CUDA availability
    cuda_available = check_cuda_available(binary)
    if "cuda" in devices and not cuda_available:
        print("WARNING: CUDA requested but not available, skipping CUDA tests", file=sys.stderr)
        devices = [d for d in devices if d != "cuda"]

    if not devices:
        print("No valid devices to test", file=sys.stderr)
        return 1

    # Use tmpfs if available for better I/O performance
    tmpdir_base = Path("/dev/shm") if Path("/dev/shm").is_dir() else Path(tempfile.gettempdir())

    all_results = {}  # (image_count, config_name) -> (mean, stdev)
    sequential_cpu_baseline = {}  # image_count -> mean_time

    with tempfile.TemporaryDirectory(dir=tmpdir_base) as tmpdir:
        tmpdir = Path(tmpdir)

        for image_count in image_counts:
            print(f"\n{'='*70}")
            print(f"Benchmarking with {image_count} images")
            print(f"{'='*70}")
            print(f"  Binary: {binary}")
            print(f"  Devices: {', '.join(devices)}")
            print(f"  Threads: {', '.join(map(str, thread_counts))}")
            print(f"  Warmup: {args.warmup}, Iterations: {args.iterations}")
            print()

            # Create test images (reuse if same count)
            img_format = "jpg" if args.jpeg else "png"
            print(f"Creating {image_count} test images ({img_format.upper()})...", end=" ", flush=True)
            for f in tmpdir.glob("input*.*"):
                f.unlink()
            create_test_images(args.source, tmpdir, image_count, img_format)
            print("done")

            input_pattern = str(tmpdir / f"input%04d.{img_format}")
            output_pattern = str(tmpdir / "output%04d.pbm")

            results = []

            # Sequential CPU baseline (always run for --verify-10x or --sequential)
            if args.sequential or args.verify_10x:
                print("Sequential CPU (no batch)...", end=" ", flush=True)
                mean, stdev = bench_configuration(
                    binary, input_pattern, output_pattern, image_count,
                    None, args.warmup, args.iterations, "cpu",
                    no_processing=args.no_processing
                )
                print(f"{mean:.0f}ms (stdev={stdev:.0f}ms)")
                results.append(("sequential-cpu", mean, stdev, "cpu"))
                sequential_cpu_baseline[image_count] = mean
                all_results[(image_count, "sequential-cpu")] = (mean, stdev)

            # Test each device
            for device in devices:
                device_label = device.upper()

                # Batch with different thread counts (skip if stream scaling mode)
                if not stream_counts:
                    for threads in thread_counts:
                        config_name = f"batch-{device}-j{threads}"
                        print(f"Batch {device_label} (jobs={threads})...", end=" ", flush=True)
                        mean, stdev = bench_configuration(
                            binary, input_pattern, output_pattern, image_count,
                            threads, args.warmup, args.iterations, device,
                            no_processing=args.no_processing
                        )
                        print(f"{mean:.0f}ms (stdev={stdev:.0f}ms)")
                        results.append((config_name, mean, stdev, device))
                        all_results[(image_count, config_name)] = (mean, stdev)

                # Stream scaling tests (CUDA only)
                if stream_counts and device == "cuda":
                    print(f"\n  GPU Stream Scaling:")
                    for streams in stream_counts:
                        # Use jobs=streams to ensure enough work submitted
                        jobs = streams
                        config_name = f"cuda-s{streams}-j{jobs}"
                        print(f"    CUDA (streams={streams}, jobs={jobs})...", end=" ", flush=True)
                        mean, stdev = bench_configuration(
                            binary, input_pattern, output_pattern, image_count,
                            jobs, args.warmup, args.iterations, device, streams,
                            no_processing=args.no_processing
                        )
                        print(f"{mean:.0f}ms (stdev={stdev:.0f}ms)")
                        results.append((config_name, mean, stdev, device))
                        all_results[(image_count, config_name)] = (mean, stdev)

            # Summary for this image count
            print(f"\n{'-'*60}")
            print(f"Summary for {image_count} images:")
            print(f"{'-'*60}")

            baseline = sequential_cpu_baseline.get(image_count, results[0][1]) if results else 1.0
            for name, mean, stdev, device in results:
                if mean > 0 and not (mean != mean):  # Check for NaN
                    speedup = baseline / mean
                    per_image = mean / image_count
                    throughput = (image_count / mean) * 1000.0  # images/sec
                    print(f"  {name:<20} {mean:>8.0f}ms  "
                          f"({per_image:>5.1f}ms/img, {throughput:>5.1f} img/s, {speedup:>5.2f}x)")
                else:
                    print(f"  {name:<20}      FAILED")

    # Final summary across all image counts
    if len(image_counts) > 1:
        print(f"\n{'='*70}")
        print("Overall Summary (speedup vs sequential CPU)")
        print(f"{'='*70}")
        print(f"{'Config':<25}", end="")
        for count in image_counts:
            print(f"{count:>10} img", end="")
        print()
        print("-" * (25 + 14 * len(image_counts)))

        # Get all unique config names (excluding sequential)
        config_names = sorted(set(name for (_, name) in all_results.keys() if name != "sequential-cpu"))
        for config in config_names:
            print(f"{config:<25}", end="")
            for count in image_counts:
                result = all_results.get((count, config))
                baseline = sequential_cpu_baseline.get(count)
                if result and baseline and result[0] > 0:
                    speedup = baseline / result[0]
                    print(f"{speedup:>10.2f}x", end="")
                else:
                    print(f"{'N/A':>11}", end="")
            print()

    # Stream scaling summary and verification
    if stream_counts:
        print(f"\n{'='*70}")
        print("GPU Stream Scaling Analysis")
        print(f"{'='*70}")

        for count in image_counts:
            print(f"\n  {count} images:")
            print(f"  {'Streams':<10} {'Time (ms)':<12} {'ms/img':<10} {'Speedup':<10} {'Efficiency':<10}")
            print(f"  {'-'*52}")

            # Get baseline (lowest stream count)
            stream_results = []
            for streams in stream_counts:
                config = f"cuda-s{streams}-j{streams}"
                result = all_results.get((count, config))
                if result and result[0] > 0:
                    stream_results.append((streams, result[0]))

            if not stream_results:
                print(f"  No stream scaling results")
                continue

            # Sort by stream count
            stream_results.sort(key=lambda x: x[0])
            baseline_streams, baseline_time = stream_results[0]

            for streams, elapsed in stream_results:
                per_img = elapsed / count
                # Speedup relative to lowest stream count
                speedup = baseline_time / elapsed
                # Efficiency: actual speedup / ideal speedup (ideal = streams/baseline_streams)
                ideal_speedup = streams / baseline_streams
                efficiency = (speedup / ideal_speedup) * 100.0 if ideal_speedup > 0 else 0
                print(f"  {streams:<10} {elapsed:<12.0f} {per_img:<10.1f} {speedup:<10.2f}x {efficiency:<10.1f}%")

        # Verify near-linear scaling
        if args.verify_linear and len(stream_counts) >= 2:
            print(f"\n{'-'*60}")
            print("Linear Scaling Verification (PR33 Target)")
            print(f"{'-'*60}")

            passed = True
            for count in image_counts:
                # Get first and last stream count results
                first_config = f"cuda-s{stream_counts[0]}-j{stream_counts[0]}"
                last_config = f"cuda-s{stream_counts[-1]}-j{stream_counts[-1]}"
                first_result = all_results.get((count, first_config))
                last_result = all_results.get((count, last_config))

                if not first_result or not last_result:
                    print(f"  {count} images: SKIP (missing results)")
                    continue

                first_time = first_result[0]
                last_time = last_result[0]
                actual_speedup = first_time / last_time
                ideal_speedup = stream_counts[-1] / stream_counts[0]
                efficiency = (actual_speedup / ideal_speedup) * 100.0

                # Target: at least 50% efficiency for near-linear scaling
                target_efficiency = 50.0
                status = "PASS" if efficiency >= target_efficiency else "FAIL"
                passed = passed and (efficiency >= target_efficiency)

                print(f"  {count} images: {stream_counts[0]}→{stream_counts[-1]} streams = "
                      f"{actual_speedup:.2f}x (ideal: {ideal_speedup:.1f}x, efficiency: {efficiency:.0f}%) [{status}]")

            print()
            if passed:
                print("  *** LINEAR SCALING VERIFIED ***")
            else:
                print("  *** LINEAR SCALING BELOW TARGET ***")

    # Verify 10x speedup target for PR26
    if args.verify_10x:
        print(f"\n{'='*70}")
        print("10x Speedup Verification (PR26 Target)")
        print(f"{'='*70}")

        passed = True
        for count in image_counts:
            baseline = sequential_cpu_baseline.get(count)
            if not baseline:
                print(f"  {count} images: SKIP (no sequential baseline)")
                continue

            # Find best CUDA batch result for this image count
            best_cuda = None
            best_cuda_name = None
            for (img_count, name), (mean, _) in all_results.items():
                if img_count == count and "cuda" in name and mean > 0:
                    if best_cuda is None or mean < best_cuda:
                        best_cuda = mean
                        best_cuda_name = name

            if best_cuda:
                speedup = baseline / best_cuda
                target = 10.0
                status = "PASS" if speedup >= target else "FAIL"
                passed = passed and (speedup >= target)
                print(f"  {count} images: {best_cuda_name} -> {speedup:.2f}x (target: {target}x) [{status}]")
            else:
                print(f"  {count} images: No CUDA results available")
                if "cuda" in devices:
                    passed = False

        print()
        if passed:
            print("  *** ALL TARGETS PASSED ***")
            return 0
        else:
            print("  *** SOME TARGETS FAILED ***")
            return 1

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
