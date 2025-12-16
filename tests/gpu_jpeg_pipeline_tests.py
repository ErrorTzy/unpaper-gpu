# SPDX-FileCopyrightText: 2025 The unpaper authors
#
# SPDX-License-Identifier: GPL-2.0-only

"""
GPU JPEG Pipeline Tests (PR38)

This test suite validates that the GPU JPEG pipeline produces output
consistent with the CPU backend. Since JPEG is lossy, we use image
similarity metrics rather than exact pixel comparison.

Test categories:
1. Single image tests - compare GPU JPEG output with PBM/PPM golden images
2. Batch processing tests - validate batch mode with GPU pipeline
3. CPU vs GPU pipeline comparison tests
"""

import logging
import math
import os
import pathlib
import shlex
import subprocess
import sys
import tempfile
from functools import lru_cache
from typing import Optional, Tuple

import pytest

try:
    import PIL.Image
    import PIL.ImageFilter
    HAS_PIL = True
except ImportError:
    HAS_PIL = False

try:
    import numpy as np
    HAS_NUMPY = True
except ImportError:
    HAS_NUMPY = False

_LOGGER = logging.getLogger(__name__)


def compute_ssim(img1: "PIL.Image.Image", img2: "PIL.Image.Image") -> float:
    """Compute Structural Similarity Index (SSIM) between two images.

    Returns a value between 0 and 1, where 1 means identical images.
    This is more appropriate than pixel-wise comparison for lossy formats.
    """
    if not HAS_NUMPY:
        # Fallback to simple comparison if numpy not available
        return 1.0 - compute_pixel_difference_ratio(img1, img2)

    # Convert to grayscale numpy arrays
    arr1 = np.array(img1.convert('L'), dtype=np.float64)
    arr2 = np.array(img2.convert('L'), dtype=np.float64)

    if arr1.shape != arr2.shape:
        return 0.0

    # SSIM constants
    C1 = (0.01 * 255) ** 2
    C2 = (0.03 * 255) ** 2

    # Compute means
    mu1 = arr1.mean()
    mu2 = arr2.mean()

    # Compute variances and covariance
    sigma1_sq = ((arr1 - mu1) ** 2).mean()
    sigma2_sq = ((arr2 - mu2) ** 2).mean()
    sigma12 = ((arr1 - mu1) * (arr2 - mu2)).mean()

    # SSIM formula
    numerator = (2 * mu1 * mu2 + C1) * (2 * sigma12 + C2)
    denominator = (mu1 ** 2 + mu2 ** 2 + C1) * (sigma1_sq + sigma2_sq + C2)

    return numerator / denominator


def compute_pixel_difference_ratio(img1: "PIL.Image.Image", img2: "PIL.Image.Image",
                                   threshold: int = 30) -> float:
    """Compute ratio of pixels that differ by more than threshold.

    This is more lenient than exact comparison and accounts for JPEG artifacts.
    """
    if img1.size != img2.size:
        return 1.0

    # Convert both to same mode for comparison
    if img1.mode != img2.mode:
        img1 = img1.convert('RGB')
        img2 = img2.convert('RGB')

    total_pixels = img1.width * img1.height
    different_pixels = 0

    for y in range(img1.height):
        for x in range(img1.width):
            p1 = img1.getpixel((x, y))
            p2 = img2.getpixel((x, y))

            # Handle grayscale vs RGB
            if isinstance(p1, int):
                diff = abs(p1 - p2)
            else:
                # RGB comparison - check if any channel differs significantly
                diff = max(abs(c1 - c2) for c1, c2 in zip(p1[:3], p2[:3]))

            if diff > threshold:
                different_pixels += 1

    return different_pixels / total_pixels


def compute_mse(img1: "PIL.Image.Image", img2: "PIL.Image.Image") -> float:
    """Compute Mean Squared Error between two images."""
    if not HAS_NUMPY:
        return 0.0

    arr1 = np.array(img1.convert('L'), dtype=np.float64)
    arr2 = np.array(img2.convert('L'), dtype=np.float64)

    if arr1.shape != arr2.shape:
        return float('inf')

    return np.mean((arr1 - arr2) ** 2)


def compute_psnr(img1: "PIL.Image.Image", img2: "PIL.Image.Image") -> float:
    """Compute Peak Signal-to-Noise Ratio between two images.

    Higher values indicate more similarity. Typical values:
    - PSNR > 40 dB: Excellent quality, nearly identical
    - PSNR 30-40 dB: Good quality
    - PSNR 20-30 dB: Acceptable quality
    - PSNR < 20 dB: Poor quality
    """
    mse = compute_mse(img1, img2)
    if mse == 0:
        return float('inf')
    if mse == float('inf'):
        return 0.0

    max_pixel = 255.0
    return 20 * math.log10(max_pixel / math.sqrt(mse))


def compare_images_similarity(golden_path: pathlib.Path, result_path: pathlib.Path,
                              min_ssim: float = 0.85, max_diff_ratio: float = 0.15,
                              diff_threshold: int = 30) -> Tuple[bool, str]:
    """Compare images using multiple similarity metrics.

    Returns (passed, message) tuple.
    """
    golden_image = PIL.Image.open(golden_path)
    result_image = PIL.Image.open(result_path)

    # Check sizes match (allow small differences due to rounding)
    size_diff = abs(golden_image.width - result_image.width) + abs(golden_image.height - result_image.height)
    if size_diff > 2:
        return False, f"Size mismatch: golden {golden_image.size} vs result {result_image.size}"

    # Resize result to match golden if slightly different
    if golden_image.size != result_image.size:
        result_image = result_image.resize(golden_image.size, PIL.Image.LANCZOS)

    # Compute metrics
    ssim = compute_ssim(golden_image, result_image)
    diff_ratio = compute_pixel_difference_ratio(golden_image, result_image, diff_threshold)
    psnr = compute_psnr(golden_image, result_image)

    message = f"SSIM={ssim:.4f}, diff_ratio={diff_ratio:.4f}, PSNR={psnr:.1f}dB"

    # Pass if either metric is good
    if ssim >= min_ssim or diff_ratio <= max_diff_ratio:
        return True, message

    return False, f"FAILED: {message} (need SSIM>={min_ssim} or diff<={max_diff_ratio})"


def run_unpaper(*cmdline, check: bool = True) -> subprocess.CompletedProcess:
    """Run unpaper with given arguments."""
    unpaper_path = os.getenv("TEST_UNPAPER_BINARY", "unpaper")
    full_cmdline = [unpaper_path, "-vvv"] + list(cmdline)
    print(f"Running {shlex.join(full_cmdline)}")

    return subprocess.run(
        full_cmdline,
        stdout=sys.stdout,
        stderr=sys.stderr,
        check=check,
    )


@lru_cache(maxsize=1)
def gpu_pipeline_available() -> bool:
    """Check if GPU pipeline is available."""
    unpaper_path = os.getenv("TEST_UNPAPER_BINARY", "unpaper")

    proc = subprocess.run(
        [unpaper_path, "--help"],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        check=False,
    )
    return "--gpu-pipeline" in proc.stdout


@lru_cache(maxsize=1)
def cuda_runtime_available() -> bool:
    """Check if CUDA runtime is available."""
    unpaper_path = os.getenv("TEST_UNPAPER_BINARY", "unpaper")
    imgsrc_path = pathlib.Path(os.getenv("TEST_IMGSRC_DIR", "tests/source_images/"))
    source_path = imgsrc_path / "imgsrc003.png"

    if not source_path.exists():
        return False

    with tempfile.TemporaryDirectory(prefix="unpaper-cuda-probe-") as tmpdir:
        result_path = pathlib.Path(tmpdir) / "cuda-probe.ppm"
        proc = subprocess.run(
            [unpaper_path, "-vvv", "--device", "cuda", "-n",
             str(source_path), str(result_path)],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            check=False,
        )
        return proc.returncode == 0


def convert_to_jpeg(source_path: pathlib.Path, dest_path: pathlib.Path,
                   quality: int = 95) -> None:
    """Convert image to JPEG format."""
    img = PIL.Image.open(source_path)
    if img.mode == 'RGBA':
        img = img.convert('RGB')
    elif img.mode == '1':  # Binary
        img = img.convert('L')  # Convert to grayscale first
    elif img.mode == 'L':  # Grayscale
        pass  # Keep as is, JPEG supports grayscale
    img.save(dest_path, "JPEG", quality=quality)


@pytest.fixture(name="imgsrc_path")
def get_imgsrc_directory() -> pathlib.Path:
    return pathlib.Path(os.getenv("TEST_IMGSRC_DIR", "tests/source_images/"))


@pytest.fixture(name="goldendir_path")
def get_golden_directory() -> pathlib.Path:
    return pathlib.Path(os.getenv("TEST_GOLDEN_DIR", "tests/golden_images/"))


# =============================================================================
# Single Image GPU Pipeline Tests
# =============================================================================

class TestGpuPipelineSingleImage:
    """Test GPU JPEG pipeline with single images against golden references."""

    def test_a1_gpu_pipeline_vs_golden(self, imgsrc_path, goldendir_path, tmp_path):
        """[A1] Full processing with GPU JPEG pipeline vs golden PBM."""
        if not cuda_runtime_available():
            pytest.skip("CUDA runtime not available")
        if not gpu_pipeline_available():
            pytest.skip("GPU pipeline not available")

        # Convert source to JPEG
        source_png = imgsrc_path / "imgsrc001.png"
        source_jpeg = tmp_path / "source.jpg"
        convert_to_jpeg(source_png, source_jpeg)

        result_path = tmp_path / "result.jpg"
        golden_path = goldendir_path / "goldenA1.pbm"

        run_unpaper(
            "--device", "cuda",
            "--gpu-pipeline",
            "--jpeg-quality", "95",
            str(source_jpeg),
            str(result_path),
        )

        passed, message = compare_images_similarity(golden_path, result_path,
                                                    min_ssim=0.80, max_diff_ratio=0.20)
        assert passed, f"A1 GPU pipeline failed: {message}"
        print(f"A1 GPU pipeline: {message}")

    def test_b1_gpu_pipeline_vs_golden(self, imgsrc_path, goldendir_path, tmp_path):
        """[B1] Combined color/gray with GPU pipeline vs golden PPM."""
        if not cuda_runtime_available():
            pytest.skip("CUDA runtime not available")
        if not gpu_pipeline_available():
            pytest.skip("GPU pipeline not available")

        source1_png = imgsrc_path / "imgsrc003.png"
        source2_png = imgsrc_path / "imgsrc004.png"
        source1_jpeg = tmp_path / "source1.jpg"
        source2_jpeg = tmp_path / "source2.jpg"

        convert_to_jpeg(source1_png, source1_jpeg)
        convert_to_jpeg(source2_png, source2_jpeg)

        result_path = tmp_path / "result.jpg"
        golden_path = goldendir_path / "goldenB1.ppm"

        run_unpaper(
            "--device", "cuda",
            "--gpu-pipeline",
            "--jpeg-quality", "95",
            "-n",
            "--input-pages", "2",
            str(source1_jpeg),
            str(source2_jpeg),
            str(result_path),
        )

        passed, message = compare_images_similarity(golden_path, result_path,
                                                    min_ssim=0.85, max_diff_ratio=0.15)
        assert passed, f"B1 GPU pipeline failed: {message}"
        print(f"B1 GPU pipeline: {message}")

    def test_d1_crop_gpu_pipeline(self, imgsrc_path, goldendir_path, tmp_path):
        """[D1] Crop to sheet size with GPU pipeline."""
        if not cuda_runtime_available():
            pytest.skip("CUDA runtime not available")
        if not gpu_pipeline_available():
            pytest.skip("GPU pipeline not available")

        source_png = imgsrc_path / "imgsrc003.png"
        source_jpeg = tmp_path / "source.jpg"
        convert_to_jpeg(source_png, source_jpeg)

        result_path = tmp_path / "result.jpg"
        golden_path = goldendir_path / "goldenD1.ppm"

        run_unpaper(
            "--device", "cuda",
            "--gpu-pipeline",
            "--jpeg-quality", "95",
            "-n",
            "--sheet-size", "20cm,10cm",
            str(source_jpeg),
            str(result_path),
        )

        passed, message = compare_images_similarity(golden_path, result_path,
                                                    min_ssim=0.85, max_diff_ratio=0.15)
        assert passed, f"D1 GPU pipeline failed: {message}"
        print(f"D1 GPU pipeline: {message}")


# =============================================================================
# GPU Pipeline vs CPU Comparison Tests
# =============================================================================

class TestGpuPipelineVsCpu:
    """Test GPU JPEG pipeline produces similar results to CPU backend."""

    def test_minimal_processing_gpu_vs_cpu(self, imgsrc_path, tmp_path):
        """Minimal processing: GPU pipeline should match CPU output."""
        if not cuda_runtime_available():
            pytest.skip("CUDA runtime not available")
        if not gpu_pipeline_available():
            pytest.skip("GPU pipeline not available")

        source_png = imgsrc_path / "imgsrc001.png"
        source_jpeg = tmp_path / "source.jpg"
        convert_to_jpeg(source_png, source_jpeg)

        cpu_result = tmp_path / "cpu_result.ppm"
        gpu_result = tmp_path / "gpu_result.jpg"

        common_args = [
            "-n",
            "--no-blackfilter",
            "--no-noisefilter",
            "--no-blurfilter",
            "--no-grayfilter",
            "--no-deskew",
        ]

        # CPU processing
        run_unpaper("--device", "cpu", *common_args,
                   str(source_jpeg), str(cpu_result))

        # GPU pipeline processing
        run_unpaper("--device", "cuda", "--gpu-pipeline", "--jpeg-quality", "95",
                   *common_args, str(source_jpeg), str(gpu_result))

        passed, message = compare_images_similarity(cpu_result, gpu_result,
                                                    min_ssim=0.90, max_diff_ratio=0.10)
        assert passed, f"GPU vs CPU minimal processing failed: {message}"
        print(f"GPU vs CPU minimal: {message}")

    def test_full_processing_gpu_vs_cpu(self, imgsrc_path, tmp_path):
        """Full processing: GPU pipeline should match CPU output."""
        if not cuda_runtime_available():
            pytest.skip("CUDA runtime not available")
        if not gpu_pipeline_available():
            pytest.skip("GPU pipeline not available")

        source_png = imgsrc_path / "imgsrc001.png"
        source_jpeg = tmp_path / "source.jpg"
        convert_to_jpeg(source_png, source_jpeg)

        cpu_result = tmp_path / "cpu_result.pbm"
        gpu_result = tmp_path / "gpu_result.jpg"

        # CPU full processing
        run_unpaper("--device", "cpu", str(source_jpeg), str(cpu_result))

        # GPU pipeline full processing
        run_unpaper("--device", "cuda", "--gpu-pipeline", "--jpeg-quality", "95",
                   str(source_jpeg), str(gpu_result))

        passed, message = compare_images_similarity(cpu_result, gpu_result,
                                                    min_ssim=0.80, max_diff_ratio=0.20)
        assert passed, f"GPU vs CPU full processing failed: {message}"
        print(f"GPU vs CPU full: {message}")

    def test_blackfilter_only_gpu_vs_cpu(self, imgsrc_path, tmp_path):
        """Blackfilter only: test parallel blackfilter produces similar results."""
        if not cuda_runtime_available():
            pytest.skip("CUDA runtime not available")
        if not gpu_pipeline_available():
            pytest.skip("GPU pipeline not available")

        source_png = imgsrc_path / "imgsrc001.png"
        source_jpeg = tmp_path / "source.jpg"
        convert_to_jpeg(source_png, source_jpeg)

        cpu_result = tmp_path / "cpu_result.pbm"
        gpu_result = tmp_path / "gpu_result.jpg"

        common_args = [
            "--no-noisefilter",
            "--no-blurfilter",
            "--no-grayfilter",
            "--no-deskew",
        ]

        run_unpaper("--device", "cpu", *common_args,
                   str(source_jpeg), str(cpu_result))
        run_unpaper("--device", "cuda", "--gpu-pipeline", "--jpeg-quality", "95",
                   *common_args, str(source_jpeg), str(gpu_result))

        passed, message = compare_images_similarity(cpu_result, gpu_result,
                                                    min_ssim=0.85, max_diff_ratio=0.15)
        assert passed, f"Blackfilter GPU vs CPU failed: {message}"
        print(f"Blackfilter GPU vs CPU: {message}")


# =============================================================================
# Batch Processing Tests
# =============================================================================

class TestGpuPipelineBatch:
    """Test GPU JPEG pipeline in batch mode."""

    def test_batch_processing_consistency(self, imgsrc_path, tmp_path):
        """Batch processing should produce consistent results."""
        if not cuda_runtime_available():
            pytest.skip("CUDA runtime not available")
        if not gpu_pipeline_available():
            pytest.skip("GPU pipeline not available")

        # Create multiple JPEG test images
        source_png = imgsrc_path / "imgsrc001.png"
        num_images = 5

        for i in range(1, num_images + 1):
            source_jpeg = tmp_path / f"input{i:04d}.jpg"
            convert_to_jpeg(source_png, source_jpeg)

        input_pattern = str(tmp_path / "input%04d.jpg")
        output_pattern = str(tmp_path / "output%04d.jpg")

        run_unpaper(
            "--batch",
            "--jobs", "4",
            "--device", "cuda",
            "--gpu-pipeline",
            "--jpeg-quality", "95",
            "--cuda-streams", "4",
            "--overwrite",
            input_pattern,
            output_pattern,
        )

        # Verify all outputs exist and are similar to each other
        outputs = list(tmp_path.glob("output*.jpg"))
        assert len(outputs) == num_images, f"Expected {num_images} outputs, got {len(outputs)}"

        # Compare all outputs to the first one - they should be nearly identical
        first_output = outputs[0]
        for output in outputs[1:]:
            passed, message = compare_images_similarity(first_output, output,
                                                        min_ssim=0.99, max_diff_ratio=0.01)
            assert passed, f"Batch output inconsistency: {first_output} vs {output}: {message}"

        print(f"Batch processing: {num_images} images processed consistently")

    def test_batch_vs_single_consistency(self, imgsrc_path, tmp_path):
        """Batch processing should produce same results as single processing."""
        if not cuda_runtime_available():
            pytest.skip("CUDA runtime not available")
        if not gpu_pipeline_available():
            pytest.skip("GPU pipeline not available")

        source_png = imgsrc_path / "imgsrc001.png"
        source_jpeg = tmp_path / "source.jpg"
        convert_to_jpeg(source_png, source_jpeg)

        # Single image processing
        single_result = tmp_path / "single_result.jpg"
        run_unpaper(
            "--device", "cuda",
            "--gpu-pipeline",
            "--jpeg-quality", "95",
            str(source_jpeg),
            str(single_result),
        )

        # Batch processing with single image
        batch_input = tmp_path / "batch_input0001.jpg"
        convert_to_jpeg(source_png, batch_input)

        batch_pattern = str(tmp_path / "batch_input%04d.jpg")
        batch_output = str(tmp_path / "batch_output%04d.jpg")

        run_unpaper(
            "--batch",
            "--jobs", "1",
            "--device", "cuda",
            "--gpu-pipeline",
            "--jpeg-quality", "95",
            "--overwrite",
            batch_pattern,
            batch_output,
        )

        batch_result = tmp_path / "batch_output0001.jpg"

        passed, message = compare_images_similarity(single_result, batch_result,
                                                    min_ssim=0.99, max_diff_ratio=0.01)
        assert passed, f"Batch vs single inconsistency: {message}"
        print(f"Batch vs single: {message}")

    def test_batch_multistream_scaling(self, imgsrc_path, tmp_path):
        """Test that batch processing works with multiple CUDA streams."""
        if not cuda_runtime_available():
            pytest.skip("CUDA runtime not available")
        if not gpu_pipeline_available():
            pytest.skip("GPU pipeline not available")

        source_png = imgsrc_path / "imgsrc001.png"
        num_images = 10

        for i in range(1, num_images + 1):
            source_jpeg = tmp_path / f"input{i:04d}.jpg"
            convert_to_jpeg(source_png, source_jpeg)

        input_pattern = str(tmp_path / "input%04d.jpg")

        # Test with different stream counts
        for streams in [1, 2, 4, 8]:
            output_dir = tmp_path / f"output_streams{streams}"
            output_dir.mkdir(exist_ok=True)
            output_pattern = str(output_dir / "output%04d.jpg")

            run_unpaper(
                "--batch",
                f"--jobs={streams}",
                "--device", "cuda",
                "--gpu-pipeline",
                "--jpeg-quality", "95",
                f"--cuda-streams={streams}",
                "--overwrite",
                input_pattern,
                output_pattern,
            )

            outputs = list(output_dir.glob("output*.jpg"))
            assert len(outputs) == num_images, \
                f"With {streams} streams: expected {num_images} outputs, got {len(outputs)}"

        # Compare outputs from different stream counts - should be identical
        ref_output = tmp_path / "output_streams1" / "output0001.jpg"
        for streams in [2, 4, 8]:
            test_output = tmp_path / f"output_streams{streams}" / "output0001.jpg"
            passed, message = compare_images_similarity(ref_output, test_output,
                                                        min_ssim=0.99, max_diff_ratio=0.01)
            assert passed, f"Stream scaling inconsistency ({streams} streams): {message}"

        print("Multi-stream batch processing: consistent across 1,2,4,8 streams")


# =============================================================================
# Quality Control Tests
# =============================================================================

class TestGpuPipelineQuality:
    """Test JPEG quality settings affect output correctly."""

    def test_quality_affects_file_size(self, imgsrc_path, tmp_path):
        """Higher JPEG quality should produce larger files."""
        if not cuda_runtime_available():
            pytest.skip("CUDA runtime not available")
        if not gpu_pipeline_available():
            pytest.skip("GPU pipeline not available")

        source_png = imgsrc_path / "imgsrc001.png"
        source_jpeg = tmp_path / "source.jpg"
        convert_to_jpeg(source_png, source_jpeg)

        sizes = {}
        for quality in [50, 75, 95]:
            result_path = tmp_path / f"result_q{quality}.jpg"
            run_unpaper(
                "--device", "cuda",
                "--gpu-pipeline",
                f"--jpeg-quality={quality}",
                "-n",
                str(source_jpeg),
                str(result_path),
            )
            sizes[quality] = result_path.stat().st_size

        # Higher quality should produce larger files
        assert sizes[50] < sizes[75] < sizes[95], \
            f"Quality vs size unexpected: q50={sizes[50]}, q75={sizes[75]}, q95={sizes[95]}"

        print(f"Quality vs size: q50={sizes[50]/1024:.0f}KB, " +
              f"q75={sizes[75]/1024:.0f}KB, q95={sizes[95]/1024:.0f}KB")

    def test_high_quality_preserves_detail(self, imgsrc_path, tmp_path):
        """High quality JPEG should preserve details similar to original."""
        if not cuda_runtime_available():
            pytest.skip("CUDA runtime not available")
        if not gpu_pipeline_available():
            pytest.skip("GPU pipeline not available")

        source_png = imgsrc_path / "imgsrc001.png"
        source_jpeg = tmp_path / "source.jpg"
        convert_to_jpeg(source_png, source_jpeg, quality=100)

        result_path = tmp_path / "result.jpg"
        run_unpaper(
            "--device", "cuda",
            "--gpu-pipeline",
            "--jpeg-quality", "95",
            "-n",
            "--no-blackfilter",
            "--no-noisefilter",
            "--no-blurfilter",
            "--no-grayfilter",
            "--no-deskew",
            str(source_jpeg),
            str(result_path),
        )

        # High quality should have high SSIM with source
        passed, message = compare_images_similarity(source_jpeg, result_path,
                                                    min_ssim=0.90, max_diff_ratio=0.10)
        assert passed, f"High quality JPEG degradation: {message}"
        print(f"High quality preservation: {message}")


# =============================================================================
# Edge Cases and Error Handling
# =============================================================================

class TestGpuPipelineEdgeCases:
    """Test edge cases and error handling."""

    def test_non_jpeg_input_with_gpu_pipeline(self, imgsrc_path, tmp_path):
        """GPU pipeline with PNG input should still work (convert internally)."""
        if not cuda_runtime_available():
            pytest.skip("CUDA runtime not available")
        if not gpu_pipeline_available():
            pytest.skip("GPU pipeline not available")

        source_png = imgsrc_path / "imgsrc001.png"
        result_path = tmp_path / "result.jpg"

        # GPU pipeline with PNG input - should work
        run_unpaper(
            "--device", "cuda",
            "--gpu-pipeline",
            "--jpeg-quality", "95",
            "-n",
            str(source_png),
            str(result_path),
        )

        assert result_path.exists()
        img = PIL.Image.open(result_path)
        assert img.width > 0 and img.height > 0

    def test_grayscale_jpeg_processing(self, imgsrc_path, tmp_path):
        """Grayscale JPEG should be processed correctly."""
        if not cuda_runtime_available():
            pytest.skip("CUDA runtime not available")
        if not gpu_pipeline_available():
            pytest.skip("GPU pipeline not available")

        source_png = imgsrc_path / "imgsrc001.png"

        # Create grayscale JPEG
        img = PIL.Image.open(source_png).convert('L')
        source_jpeg = tmp_path / "source_gray.jpg"
        img.save(source_jpeg, "JPEG", quality=95)

        result_path = tmp_path / "result.jpg"

        run_unpaper(
            "--device", "cuda",
            "--gpu-pipeline",
            "--jpeg-quality", "95",
            str(source_jpeg),
            str(result_path),
        )

        assert result_path.exists()
        result_img = PIL.Image.open(result_path)
        assert result_img.width > 0 and result_img.height > 0


# =============================================================================
# Parallel Test Execution
# =============================================================================

@pytest.mark.parametrize("test_case,source_file,golden_file", [
    ("A1", "imgsrc001.png", "goldenA1.pbm"),
    ("B1_src1", "imgsrc003.png", None),  # Multi-input, tested separately
    ("D1", "imgsrc003.png", "goldenD1.ppm"),
    ("D2", "imgsrc003.png", "goldenD2.ppm"),
    ("D3", "imgsrc003.png", "goldenD3.ppm"),
])
def test_gpu_pipeline_golden_comparison(test_case, source_file, golden_file,
                                        imgsrc_path, goldendir_path, tmp_path):
    """Parametrized test comparing GPU pipeline output to golden images."""
    if not cuda_runtime_available():
        pytest.skip("CUDA runtime not available")
    if not gpu_pipeline_available():
        pytest.skip("GPU pipeline not available")
    if golden_file is None:
        pytest.skip("Multi-input test handled separately")

    source_path = imgsrc_path / source_file
    source_jpeg = tmp_path / "source.jpg"
    convert_to_jpeg(source_path, source_jpeg)

    result_path = tmp_path / "result.jpg"
    golden_path = goldendir_path / golden_file

    # Map test case to processing args
    extra_args = []
    if test_case == "D1":
        extra_args = ["-n", "--sheet-size", "20cm,10cm"]
    elif test_case == "D2":
        extra_args = ["-n", "--size", "20cm,10cm"]
    elif test_case == "D3":
        extra_args = ["-n", "--stretch", "20cm,10cm"]

    run_unpaper(
        "--device", "cuda",
        "--gpu-pipeline",
        "--jpeg-quality", "95",
        *extra_args,
        str(source_jpeg),
        str(result_path),
    )

    passed, message = compare_images_similarity(golden_path, result_path,
                                                min_ssim=0.80, max_diff_ratio=0.20)
    assert passed, f"{test_case} GPU pipeline failed: {message}"
    print(f"{test_case} GPU pipeline: {message}")


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
