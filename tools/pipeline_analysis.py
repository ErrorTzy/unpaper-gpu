#!/usr/bin/env python3
"""
Pipeline analysis tool for nvJPEG batch processing.

This script isolates each component of the batch processing pipeline
to identify bottlenecks and measure scaling potential.
"""

import subprocess
import sys
import os
import time
import tempfile
import statistics
from pathlib import Path

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent
BUILD_DIR = PROJECT_ROOT / "builddir-cuda"
UNPAPER = BUILD_DIR / "unpaper"

def run_command(cmd, capture=True):
    """Run a command and return stdout."""
    result = subprocess.run(
        cmd,
        capture_output=capture,
        text=True,
        cwd=PROJECT_ROOT
    )
    return result

def create_test_images(count=32, output_dir=None):
    """Create test JPEG images."""
    if output_dir is None:
        output_dir = tempfile.mkdtemp(prefix="unpaper_test_")

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Use ImageMagick to create test images
    images = []
    for i in range(count):
        img_path = output_dir / f"test_{i:04d}.jpg"
        if not img_path.exists():
            # Create A4-ish test image at 300 DPI (2480x3508 pixels)
            # Use a simpler pattern for faster generation
            cmd = [
                "convert", "-size", "2480x3508",
                "xc:white",
                "-fill", "gray90",
                "-draw", f"rectangle 100,100 2380,3408",
                "-fill", "black",
                "-pointsize", "72",
                "-annotate", f"+200+{200 + (i % 30) * 100}", f"Test Page {i+1}",
                "-quality", "90",
                str(img_path)
            ]
            subprocess.run(cmd, capture_output=True)
        images.append(str(img_path))

    return output_dir, images

def measure_decode_only(images, streams):
    """Measure just the decode component using a minimal unpaper run."""
    # This will use the decode queue but minimal processing
    with tempfile.TemporaryDirectory() as tmpdir:
        output_pattern = Path(tmpdir) / "out_%d.pnm"
        input_pattern = images[0].replace("_0000.jpg", "_%d.jpg")

        cmd = [
            str(UNPAPER),
            "--batch",
            f"--start-input=1",
            f"--end-input={len(images)}",
            "--device=cuda",
            f"--cuda-streams={streams}",
            "--no-deskew",
            "--no-blackfilter",
            "--no-noisefilter",
            "--no-blurfilter",
            "--no-grayfilter",
            "--no-mask-scan",
            "--no-mask-center",
            "--no-border-scan",
            "--no-border-align",
            "-v", "-v",
            input_pattern,
            str(output_pattern)
        ]

        start = time.time()
        result = run_command(cmd)
        elapsed = time.time() - start

        return elapsed, result.stderr

def measure_full_pipeline(images, streams, workers=None):
    """Measure the full pipeline."""
    with tempfile.TemporaryDirectory() as tmpdir:
        output_pattern = Path(tmpdir) / "out_%d.pnm"
        input_pattern = images[0].replace("_0000.jpg", "_%d.jpg")

        cmd = [
            str(UNPAPER),
            "--batch",
            f"--start-input=1",
            f"--end-input={len(images)}",
            "--device=cuda",
            f"--cuda-streams={streams}",
            "-v", "-v",
            input_pattern,
            str(output_pattern)
        ]

        start = time.time()
        result = run_command(cmd)
        elapsed = time.time() - start

        return elapsed, result.stderr

def parse_nvjpeg_stats(stderr):
    """Parse nvJPEG statistics from stderr."""
    stats = {}
    for line in stderr.split('\n'):
        if 'nvJPEG' in line:
            stats['nvjpeg_line'] = line
        if 'Peak concurrent' in line:
            try:
                stats['peak_concurrent'] = int(line.split(':')[-1].strip())
            except:
                pass
        if 'GPU decodes' in line:
            try:
                parts = line.split(':')
                if len(parts) >= 2:
                    stats['gpu_decodes'] = int(parts[1].split()[0])
            except:
                pass
        if 'Decode threads:' in line:
            try:
                stats['decode_threads'] = int(line.split(':')[-1].strip())
            except:
                pass
        if 'GPU stream pool:' in line:
            try:
                stats['stream_pool'] = int(line.split(':')[1].split()[0])
            except:
                pass
    return stats

def analyze_scaling(image_count=32):
    """Analyze scaling across different stream counts."""
    print(f"Creating {image_count} test images...")
    test_dir, images = create_test_images(image_count)
    print(f"Test images in: {test_dir}")

    stream_counts = [1, 2, 4, 8]
    results = []

    print("\n" + "="*70)
    print("PIPELINE SCALING ANALYSIS")
    print("="*70)

    # Baseline: 1 stream
    print("\n--- Baseline: 1 stream ---")
    baseline_time, baseline_stderr = measure_full_pipeline(images, 1)
    baseline_stats = parse_nvjpeg_stats(baseline_stderr)
    print(f"Time: {baseline_time:.2f}s ({baseline_time/len(images)*1000:.1f}ms/image)")
    print(f"Stats: {baseline_stats}")

    results.append({
        'streams': 1,
        'time': baseline_time,
        'speedup': 1.0,
        'stats': baseline_stats
    })

    # Test different stream counts
    for streams in stream_counts[1:]:
        print(f"\n--- {streams} streams ---")
        elapsed, stderr = measure_full_pipeline(images, streams)
        stats = parse_nvjpeg_stats(stderr)
        speedup = baseline_time / elapsed

        print(f"Time: {elapsed:.2f}s ({elapsed/len(images)*1000:.1f}ms/image)")
        print(f"Speedup: {speedup:.2f}x")
        print(f"Stats: {stats}")

        results.append({
            'streams': streams,
            'time': elapsed,
            'speedup': speedup,
            'stats': stats
        })

    # Summary
    print("\n" + "="*70)
    print("SUMMARY")
    print("="*70)
    print(f"{'Streams':<10} {'Time (s)':<12} {'ms/img':<12} {'Speedup':<10}")
    print("-"*44)
    for r in results:
        ms_per_img = r['time'] / len(images) * 1000
        print(f"{r['streams']:<10} {r['time']:<12.2f} {ms_per_img:<12.1f} {r['speedup']:<10.2f}x")

    return test_dir, results

def measure_component_timing(images):
    """Break down timing by component using verbose output."""
    print("\n" + "="*70)
    print("COMPONENT TIMING ANALYSIS")
    print("="*70)

    with tempfile.TemporaryDirectory() as tmpdir:
        output_pattern = Path(tmpdir) / "out_%d.pnm"
        input_pattern = images[0].replace("_0000.jpg", "_%d.jpg")

        # Run with high verbosity to get timing
        cmd = [
            str(UNPAPER),
            "--batch",
            f"--start-input=1",
            f"--end-input={len(images)}",
            "--device=cuda",
            "--cuda-streams=8",
            "-v", "-v", "-v",
            input_pattern,
            str(output_pattern)
        ]

        start = time.time()
        result = run_command(cmd)
        elapsed = time.time() - start

        # Parse timing info from stderr
        print(f"\nTotal time: {elapsed:.2f}s")
        print(f"\nRelevant output:")
        for line in result.stderr.split('\n'):
            if any(x in line.lower() for x in ['queue', 'decode', 'nvjpeg', 'gpu', 'worker', 'stream', 'time', 'ms']):
                print(f"  {line}")

def main():
    if not UNPAPER.exists():
        print(f"Error: unpaper not found at {UNPAPER}")
        print("Build with: meson compile -C builddir-cuda/")
        sys.exit(1)

    # Check CUDA availability
    result = run_command([str(UNPAPER), "--help"])

    print("Unpaper Pipeline Analysis Tool")
    print("="*50)

    image_count = 32
    if len(sys.argv) > 1:
        image_count = int(sys.argv[1])

    # Run scaling analysis
    test_dir, results = analyze_scaling(image_count)

    # Component timing with 8 streams
    print("\nGetting component timing breakdown...")
    test_dir_path, images = create_test_images(image_count, test_dir)
    measure_component_timing(images)

    print(f"\nTest images kept at: {test_dir}")
    print("You can re-run tests manually with:")
    print(f"  {UNPAPER} --batch --device=cuda --cuda-streams=N -v -v \\")
    print(f"    {test_dir}/test_%d.jpg /tmp/out_%d.pnm")

if __name__ == "__main__":
    main()
