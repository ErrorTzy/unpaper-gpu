# SPDX-FileCopyrightText: 2025 The unpaper authors
#
# SPDX-License-Identifier: GPL-2.0-only

"""
GPU JPEG Pipeline Tests (PR A9)

Focused checks for the CUDA JPEG output path:
- GPU JPEG output should match CPU processing for minimal and full pipelines.
- Batch JPEG mode should be consistent with single-image output.
- JPEG quality should affect encoded file size.

Redundant golden comparisons and multistream scaling tests are removed
because image goldens already cover the same transformations.
"""

import os
import pathlib
import shlex
import shutil
import subprocess
import sys
import tempfile
from functools import lru_cache
from typing import Tuple

import pytest
import PIL.Image
import PIL.ImageChops

try:
    import numpy as np
    HAS_NUMPY = True
except ImportError:
    HAS_NUMPY = False

pytestmark = [pytest.mark.gpu, pytest.mark.slow]


def compute_ssim(img1: "PIL.Image.Image", img2: "PIL.Image.Image") -> float:
    """Compute a lightweight SSIM approximation for grayscale images."""
    if not HAS_NUMPY:
        return 1.0 - compute_pixel_difference_ratio(img1, img2)

    arr1 = np.array(img1.convert("L"), dtype=np.float64)
    arr2 = np.array(img2.convert("L"), dtype=np.float64)
    if arr1.shape != arr2.shape:
        return 0.0

    C1 = (0.01 * 255) ** 2
    C2 = (0.03 * 255) ** 2

    mu1 = arr1.mean()
    mu2 = arr2.mean()
    sigma1_sq = ((arr1 - mu1) ** 2).mean()
    sigma2_sq = ((arr2 - mu2) ** 2).mean()
    sigma12 = ((arr1 - mu1) * (arr2 - mu2)).mean()

    numerator = (2 * mu1 * mu2 + C1) * (2 * sigma12 + C2)
    denominator = (mu1 ** 2 + mu2 ** 2 + C1) * (sigma1_sq + sigma2_sq + C2)
    return numerator / denominator


def compute_pixel_difference_ratio(
    img1: "PIL.Image.Image", img2: "PIL.Image.Image", threshold: int = 30
) -> float:
    """Compute ratio of pixels that differ by more than threshold."""
    if img1.size != img2.size:
        return 1.0

    img1_l = img1.convert("L")
    img2_l = img2.convert("L")

    if HAS_NUMPY:
        arr1 = np.array(img1_l, dtype=np.int16)
        arr2 = np.array(img2_l, dtype=np.int16)
        diff = np.abs(arr1 - arr2)
        return float(np.mean(diff > threshold))

    diff = PIL.ImageChops.difference(img1_l, img2_l)
    mask = diff.point(lambda p: 255 if p > threshold else 0)
    hist = mask.histogram()
    total_pixels = img1.width * img1.height
    different_pixels = total_pixels - hist[0]
    return different_pixels / total_pixels


def compare_images_similarity(
    golden_path: pathlib.Path,
    result_path: pathlib.Path,
    *,
    min_ssim: float,
    max_diff_ratio: float,
    diff_threshold: int = 30,
) -> Tuple[bool, str]:
    """Compare images using SSIM and per-pixel difference ratio."""
    golden_image = PIL.Image.open(golden_path)
    result_image = PIL.Image.open(result_path)

    size_diff = abs(golden_image.width - result_image.width) + abs(
        golden_image.height - result_image.height
    )
    if size_diff > 2:
        return (
            False,
            f"Size mismatch: golden {golden_image.size} vs result {result_image.size}",
        )

    if golden_image.size != result_image.size:
        result_image = result_image.resize(golden_image.size, PIL.Image.LANCZOS)

    ssim = compute_ssim(golden_image, result_image)
    diff_ratio = compute_pixel_difference_ratio(
        golden_image, result_image, diff_threshold
    )
    message = f"SSIM={ssim:.4f}, diff_ratio={diff_ratio:.4f}"

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
def cuda_backend_available() -> bool:
    """Check if the binary advertises CUDA support."""
    unpaper_path = os.getenv("TEST_UNPAPER_BINARY", "unpaper")

    proc = subprocess.run(
        [unpaper_path, "--help"],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        check=False,
    )
    return "--device=cpu|cuda" in proc.stdout


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
            [
                unpaper_path,
                "-vvv",
                "--device",
                "cuda",
                "-n",
                str(source_path),
                str(result_path),
            ],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            check=False,
        )
        return proc.returncode == 0


def require_cuda() -> None:
    if not cuda_runtime_available():
        pytest.skip("CUDA runtime not available")
    if not cuda_backend_available():
        pytest.skip("CUDA backend not available")


@pytest.fixture(name="imgsrc_path")
def get_imgsrc_directory() -> pathlib.Path:
    return pathlib.Path(os.getenv("TEST_IMGSRC_DIR", "tests/source_images/"))


# ---------------------------------------------------------------------------
# Focused GPU JPEG pipeline tests
# ---------------------------------------------------------------------------

def test_gpu_jpeg_minimal_matches_cpu(imgsrc_path, tmp_path):
    """Minimal processing: GPU JPEG output should match CPU output."""
    require_cuda()

    source_path = imgsrc_path / "imgsrc001.png"
    cpu_result = tmp_path / "cpu_min.ppm"
    gpu_result = tmp_path / "gpu_min.jpg"

    common_args = [
        "-n",
        "--no-blackfilter",
        "--no-noisefilter",
        "--no-blurfilter",
        "--no-grayfilter",
        "--no-deskew",
    ]

    run_unpaper("--device", "cpu", *common_args, str(source_path), str(cpu_result))
    run_unpaper(
        "--device",
        "cuda",
        "--jpeg-quality",
        "95",
        *common_args,
        str(source_path),
        str(gpu_result),
    )

    passed, message = compare_images_similarity(
        cpu_result,
        gpu_result,
        min_ssim=0.90,
        max_diff_ratio=0.10,
    )
    assert passed, f"GPU vs CPU minimal processing failed: {message}"


def test_gpu_jpeg_full_matches_cpu(imgsrc_path, tmp_path):
    """Full processing: GPU JPEG output should match CPU output."""
    require_cuda()

    source_path = imgsrc_path / "imgsrc001.png"
    cpu_result = tmp_path / "cpu_full.pbm"
    gpu_result = tmp_path / "gpu_full.jpg"

    run_unpaper("--device", "cpu", str(source_path), str(cpu_result))
    run_unpaper(
        "--device",
        "cuda",
        "--jpeg-quality",
        "95",
        str(source_path),
        str(gpu_result),
    )

    passed, message = compare_images_similarity(
        cpu_result,
        gpu_result,
        min_ssim=0.80,
        max_diff_ratio=0.20,
    )
    assert passed, f"GPU vs CPU full processing failed: {message}"


def test_gpu_jpeg_batch_matches_single(imgsrc_path, tmp_path):
    """Batch mode should match single-image output."""
    require_cuda()

    source_path = imgsrc_path / "imgsrc001.png"
    single_result = tmp_path / "single.jpg"

    run_unpaper(
        "--device",
        "cuda",
        "--jpeg-quality",
        "90",
        str(source_path),
        str(single_result),
    )

    batch_input = tmp_path / "batch_input0001.png"
    shutil.copyfile(source_path, batch_input)

    batch_pattern = str(tmp_path / "batch_input%04d.png")
    batch_output = str(tmp_path / "batch_output%04d.jpg")

    run_unpaper(
        "--batch",
        "--jobs",
        "1",
        "--device",
        "cuda",
        "--jpeg-quality",
        "90",
        "--overwrite",
        batch_pattern,
        batch_output,
    )

    batch_result = tmp_path / "batch_output0001.jpg"
    passed, message = compare_images_similarity(
        single_result,
        batch_result,
        min_ssim=0.99,
        max_diff_ratio=0.01,
    )
    assert passed, f"Batch vs single inconsistency: {message}"


def test_gpu_jpeg_quality_affects_size(imgsrc_path, tmp_path):
    """Higher JPEG quality should produce larger files."""
    require_cuda()

    source_path = imgsrc_path / "imgsrc001.png"

    sizes = {}
    for quality in [60, 95]:
        result_path = tmp_path / f"result_q{quality}.jpg"
        run_unpaper(
            "--device",
            "cuda",
            f"--jpeg-quality={quality}",
            "-n",
            str(source_path),
            str(result_path),
        )
        sizes[quality] = result_path.stat().st_size

    assert sizes[60] < sizes[95], (
        f"Quality vs size unexpected: q60={sizes[60]}, q95={sizes[95]}"
    )


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
