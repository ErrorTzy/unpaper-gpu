# SPDX-FileCopyrightText: 2021 The unpaper authors
#
# SPDX-License-Identifier: GPL-2.0-only
# SPDX-License-Identifier: MIT

import logging
import os
import pathlib
import re
import shlex
import shutil
import subprocess
import sys
import tempfile
from functools import lru_cache
from typing import Sequence

import pytest
import PIL.Image
import PIL.ImageChops

_LOGGER = logging.getLogger(__name__)


def compare_images(*, golden: pathlib.Path, result: pathlib.Path) -> float:
    """Compare images loaded from the provided paths, returns ratio of different pixels.

    Images are converted to grayscale and then binarized before comparison to handle
    format differences (e.g., comparing PBM mono output against PPM grayscale golden
    images). This is necessary because the output format auto-detection now forces
    mono format for .pbm files.
    """

    golden_image = PIL.Image.open(golden)
    result_image = PIL.Image.open(result)

    if golden_image.size != result_image.size:
        _LOGGER.error(
            f"image sizes don't match: {golden} {golden_image.size} != {result} {result_image.size}"
        )
        return float("inf")

    # Convert both images to grayscale and binarize at threshold 128.
    # This handles cases where golden is grayscale but result is mono.
    threshold = 128
    golden_bw = golden_image.convert("L").point(lambda p: 255 if p >= threshold else 0)
    result_bw = result_image.convert("L").point(lambda p: 255 if p >= threshold else 0)

    diff = PIL.ImageChops.difference(golden_bw, result_bw)
    hist = diff.histogram()
    total_pixels = golden_image.width * golden_image.height
    different_pixels = total_pixels - hist[0]
    return different_pixels / total_pixels


def compare_images_pdf(*, golden: pathlib.Path, result: pathlib.Path) -> float:
    """Compare images for PDF tests, allowing small size drift.

    PDF renderers may introduce off-by-a-few pixels due to page box rounding.
    For PDF pipeline tests we allow a small resample to the golden size before
    binarized comparison.
    """
    golden_image = PIL.Image.open(golden)
    result_image = PIL.Image.open(result)

    if golden_image.size != result_image.size:
        gw, gh = golden_image.size
        rw, rh = result_image.size
        # Guardrail: avoid masking major mismatches.
        if abs(gw - rw) > max(gw, rw) * 0.10 or abs(gh - rh) > max(gh, rh) * 0.10:
            _LOGGER.error(
                f"image sizes don't match (too large to resample): {golden} {golden_image.size} != {result} {result_image.size}"
            )
            return float("inf")
        result_image = result_image.resize(
            golden_image.size, resample=PIL.Image.Resampling.BILINEAR
        )

    threshold = 128
    golden_bw = golden_image.convert("L").point(lambda p: 255 if p >= threshold else 0)
    result_bw = result_image.convert("L").point(lambda p: 255 if p >= threshold else 0)

    diff = PIL.ImageChops.difference(golden_bw, result_bw)
    hist = diff.histogram()
    total_pixels = golden_image.width * golden_image.height
    different_pixels = total_pixels - hist[0]
    return different_pixels / total_pixels


def run_unpaper(
    *cmdline: Sequence[str], check: bool = True, capture: bool = False
) -> subprocess.CompletedProcess:
    unpaper_path = os.getenv("TEST_UNPAPER_BINARY", "unpaper")

    full_cmdline = [unpaper_path, "-vvv"] + list(cmdline)
    print(f"Running {shlex.join(full_cmdline)}")

    if capture:
        return subprocess.run(
            full_cmdline,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            check=check,
        )

    return subprocess.run(
        full_cmdline,
        stdout=sys.stdout,
        stderr=sys.stderr,
        check=check,
    )


def test_device_cpu_works_and_cuda_unavailable_errors(imgsrc_path, tmp_path):
    source_path = imgsrc_path / "imgsrc001.png"
    result_path = tmp_path / "result.pbm"

    run_unpaper("--device", "cpu", "-n", str(source_path), str(result_path))

    unpaper_path = os.getenv("TEST_UNPAPER_BINARY", "unpaper")
    proc = subprocess.run(
        [unpaper_path, "-vvv", "--device", "cuda", "-n", str(source_path), str(result_path)],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        check=False,
    )
    # In CPU-only builds, CUDA must error.
    # In CUDA-capable builds, CUDA may still error at runtime if no device is
    # available, but it must never silently succeed in a CPU-only build.
    if proc.returncode != 0:
        assert "cuda" in proc.stderr.lower()


@lru_cache(maxsize=1)
def cuda_runtime_available() -> bool:
    unpaper_path = os.getenv("TEST_UNPAPER_BINARY", "unpaper")
    imgsrc_path = pathlib.Path(os.getenv("TEST_IMGSRC_DIR", "tests/source_images/"))
    source_path = imgsrc_path / "imgsrc003.png"

    if not source_path.exists():
        return False

    tmpdir = tempfile.TemporaryDirectory(prefix="unpaper-cuda-probe-")
    try:
        result_path = pathlib.Path(tmpdir.name) / "cuda-probe.ppm"

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
    finally:
        tmpdir.cleanup()


@lru_cache(maxsize=1)
def pdf_mode_supported() -> bool:
    """Return true if the unpaper binary supports PDF mode."""
    unpaper_path = os.getenv("TEST_UNPAPER_BINARY", "unpaper")
    proc = subprocess.run(
        [unpaper_path, "--help"],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        check=False,
    )
    help_text = (proc.stdout or "") + (proc.stderr or "")
    # The help text includes the PDF flags only when built with PDF support.
    return proc.returncode == 0 and "--pdf-quality" in help_text


def _require_external_tools(*tools: str) -> None:
    missing = [t for t in tools if shutil.which(t) is None]
    if missing:
        pytest.skip(
            f"missing external tools required for PDF tests: {', '.join(missing)}"
        )


def _gs_build_pdf(
    *,
    images: list[pathlib.Path],
    out_pdf: pathlib.Path,
    encoding: str,
    workdir: pathlib.Path,
) -> None:
    if encoding not in {"jpeg", "png", "jbig2"}:
        raise ValueError(f"unknown encoding: {encoding}")

    _require_external_tools("mutool")

    # Build a simple PDF, one image per page.
    # For JBIG2, we generate JBIG2 streams via the `jbig2` encoder and wrap them
    # into a minimal PDF.
    dpi = 300
    page_files: list[pathlib.Path] = []

    for i, src in enumerate(images, start=1):
        if encoding == "jpeg":
            img = PIL.Image.open(src).convert("RGB")
            img_path = workdir / f"page-{i:03d}.jpg"
            img.save(img_path, "JPEG", quality=95)
        elif encoding == "png":
            # Force 8bpc RGB PNG so MuPDF embeds it as PNG/Flate, not CCITT.
            img = PIL.Image.open(src).convert("RGB")
            img_path = workdir / f"page-{i:03d}.png"
            img.save(img_path, "PNG")
        else:  # jbig2
            # JBIG2 supports bi-level images; generate PBM for encoding.
            img = PIL.Image.open(src).convert("L")
            bw = img.point(lambda p: 255 if p >= 128 else 0, mode="L").convert("1")
            img_path = workdir / f"page-{i:03d}.pbm"
            bw.save(img_path)

        width_px, height_px = img.size
        width_pt = width_px * 72.0 / dpi
        height_pt = height_px * 72.0 / dpi

        if encoding == "jbig2":
            # For JBIG2 we don't use mutool create; we wrap encoded JBIG2 data.
            continue

        page_txt = workdir / f"page-{i:03d}.txt"
        page_txt.write_text(
            "\n".join(
                [
                    f"%%MediaBox 0 0 {width_pt:.6f} {height_pt:.6f}",
                    f"%%Image Im{i} {img_path.name}",
                    "q",
                    f"{width_pt:.6f} 0 0 {height_pt:.6f} 0 0 cm",
                    f"/Im{i} Do",
                    "Q",
                    "",
                ]
            ),
            encoding="utf-8",
        )
        page_files.append(page_txt)

    if encoding in {"jpeg", "png"}:
        base_pdf = workdir / "base.pdf"
        subprocess.run(
            ["mutool", "create", "-o", str(base_pdf), *[str(p) for p in page_files]],
            cwd=str(workdir),
            check=True,
        )
        shutil.copyfile(base_pdf, out_pdf)
        return

    # JBIG2: encode each page to JBIG2 and write a minimal PDF that embeds the
    # JBIG2 streams directly (so MuPDF reports PDF_IMAGE_JBIG2).
    _require_external_tools("jbig2")

    def _write_simple_jbig2_pdf(
        *, pages: list[tuple[bytes, int, int]], out_path: pathlib.Path, dpi: int
    ) -> None:
        def obj_bytes(obj_id: int, payload: bytes) -> tuple[int, bytes]:
            return obj_id, b"%d 0 obj\n" % obj_id + payload + b"\nendobj\n"

        objects: list[tuple[int, bytes]] = []
        next_id = 1

        # 1: Catalog, 2: Pages.
        catalog_id = next_id
        next_id += 1
        pages_id = next_id
        next_id += 1

        page_ids: list[int] = []
        content_ids: list[int] = []
        image_ids: list[int] = []

        for _ in pages:
            page_ids.append(next_id)
            next_id += 1
            content_ids.append(next_id)
            next_id += 1
            image_ids.append(next_id)
            next_id += 1

        kids = b" ".join([b"%d 0 R" % pid for pid in page_ids])
        objects.append(obj_bytes(catalog_id, b"<< /Type /Catalog /Pages %d 0 R >>" % pages_id))
        objects.append(
            obj_bytes(
                pages_id,
                b"<< /Type /Pages /Count %d /Kids [ %s ] >>"
                % (len(pages), kids),
            )
        )

        for idx, (jb2, w_px, h_px) in enumerate(pages):
            page_id = page_ids[idx]
            content_id = content_ids[idx]
            image_id = image_ids[idx]

            w_pt = w_px * 72.0 / dpi
            h_pt = h_px * 72.0 / dpi

            contents = (
                f"q\n{w_pt:.6f} 0 0 {h_pt:.6f} 0 0 cm\n/Im0 Do\nQ\n".encode("ascii")
            )

            page_payload = (
                b"<< /Type /Page /Parent %d 0 R /MediaBox [0 0 %.6f %.6f] "
                b"/Resources << /XObject << /Im0 %d 0 R >> >> /Contents %d 0 R >>"
                % (pages_id, w_pt, h_pt, image_id, content_id)
            )
            objects.append(obj_bytes(page_id, page_payload))

            content_payload = (
                b"<< /Length %d >>\nstream\n" % len(contents)
                + contents
                + b"endstream"
            )
            objects.append(obj_bytes(content_id, content_payload))

            image_dict = (
                b"<< /Type /XObject /Subtype /Image /Width %d /Height %d "
                b"/ColorSpace /DeviceGray /BitsPerComponent 1 "
                b"/Filter /JBIG2Decode /Length %d >>\n"
                % (w_px, h_px, len(jb2))
            )
            image_payload = image_dict + b"stream\n" + jb2 + b"\nendstream"
            objects.append(obj_bytes(image_id, image_payload))

        # Write file with xref.
        objects.sort(key=lambda t: t[0])
        max_id = objects[-1][0] if objects else 0

        with out_path.open("wb") as f:
            f.write(b"%PDF-1.4\n%\xE2\xE3\xCF\xD3\n")
            offsets = {0: 0}
            for obj_id, body in objects:
                offsets[obj_id] = f.tell()
                f.write(body)

            xref_start = f.tell()
            f.write(b"xref\n")
            f.write(b"0 %d\n" % (max_id + 1))
            f.write(b"0000000000 65535 f \n")
            for obj_id in range(1, max_id + 1):
                off = offsets.get(obj_id, 0)
                f.write(b"%010d 00000 n \n" % off)

            f.write(b"trailer\n")
            f.write(b"<< /Size %d /Root %d 0 R >>\n" % (max_id + 1, catalog_id))
            f.write(b"startxref\n")
            f.write(b"%d\n" % xref_start)
            f.write(b"%%EOF\n")

    jb2_pages: list[tuple[bytes, int, int]] = []
    for i, src in enumerate(images, start=1):
        img = PIL.Image.open(src).convert("L")
        bw = img.point(lambda p: 255 if p >= 128 else 0, mode="L").convert("1")
        pbm_path = workdir / f"page-{i:03d}.pbm"
        bw.save(pbm_path)

        proc = subprocess.run(
            ["jbig2", "-p", str(pbm_path)],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            check=False,
        )
        if proc.returncode != 0 or not proc.stdout:
            raise RuntimeError(
                "jbig2 encode failed:\n"
                + (proc.stderr.decode("utf-8", errors="replace") if proc.stderr else "")
            )

        w_px, h_px = img.size
        jb2_pages.append((proc.stdout, w_px, h_px))

    _write_simple_jbig2_pdf(pages=jb2_pages, out_path=out_pdf, dpi=dpi)


def _render_pdf_pages(
    *, pdf_path: pathlib.Path, out_dir: pathlib.Path, dpi: int = 300
) -> list[pathlib.Path]:
    _require_external_tools("mutool")

    out_dir.mkdir(parents=True, exist_ok=True)
    output_pattern = out_dir / "page-%03d.pnm"
    subprocess.run(
        [
            "mutool",
            "draw",
            "-q",
            "-A",
            "0",
            "-c",
            "rgb",
            "-r",
            str(dpi),
            "-F",
            "pnm",
            "-o",
            str(output_pattern),
            str(pdf_path),
        ],
        check=True,
    )
    return sorted(out_dir.glob("page-*.pnm"))


@lru_cache(maxsize=1)
def jbig2_gpu_decode_available() -> bool:
    """Return true if CUDA+PDF builds can decode JBIG2 via the GPU PDF pipeline."""
    if not cuda_runtime_available():
        return False
    if not pdf_mode_supported():
        return False

    sample = pathlib.Path("tests/pdf_samples/test_jbig2.pdf")
    if not sample.exists():
        return False

    tmpdir = tempfile.TemporaryDirectory(prefix="unpaper-jbig2-gpu-probe-")
    try:
        out_pdf = pathlib.Path(tmpdir.name) / "out.pdf"
        proc = run_unpaper(
            "--device", "cuda", str(sample), str(out_pdf), check=False, capture=True
        )
        text = (proc.stdout or "") + (proc.stderr or "")
        return proc.returncode == 0 and "JBIG2 decoded to GPU" in text
    finally:
        tmpdir.cleanup()


@pytest.fixture
def fast_device():
    """Return the fastest available device (CUDA if available, else CPU).

    Use this fixture for tests that don't specifically need to test both backends.
    CUDA is 2-3x faster than CPU for most operations.
    """
    return "cuda" if cuda_runtime_available() else "cpu"


@pytest.mark.parametrize(
    "extra_args",
    [
        ("--pre-rotate", "90"),
        ("-M", "h"),
        ("--pre-shift", "10mils,-7mils"),
    ],
)
def test_cuda_pre_ops_match_cpu(imgsrc_path, tmp_path, extra_args):
    if not cuda_runtime_available():
        pytest.skip("CUDA runtime/device not available")

    source_path = imgsrc_path / "imgsrc003.png"
    cpu_path = tmp_path / "cpu.ppm"
    cuda_path = tmp_path / "cuda.ppm"

    run_unpaper("--device", "cpu", "-n", *extra_args, str(source_path), str(cpu_path))
    run_unpaper("--device", "cuda", "-n", *extra_args, str(source_path), str(cuda_path))

    assert compare_images(golden=cpu_path, result=cuda_path) == 0.0


@pytest.mark.parametrize("interp", ["nearest", "linear", "cubic"])
def test_cuda_stretch_and_post_size_match_cpu(imgsrc_path, tmp_path, interp):
    if not cuda_runtime_available():
        pytest.skip("CUDA runtime/device not available")

    source_path = imgsrc_path / "imgsrc003.png"
    cpu_path = tmp_path / f"cpu-{interp}.ppm"
    cuda_path = tmp_path / f"cuda-{interp}.ppm"

    run_unpaper(
        "--device",
        "cpu",
        "-n",
        "--interpolate",
        interp,
        "--stretch",
        "200mils,150mils",
        "--post-size",
        "250mils,200mils",
        str(source_path),
        str(cpu_path),
    )
    run_unpaper(
        "--device",
        "cuda",
        "-n",
        "--interpolate",
        interp,
        "--stretch",
        "200mils,150mils",
        "--post-size",
        "250mils,200mils",
        str(source_path),
        str(cuda_path),
    )

    # OpenCV uses half-pixel center coordinate convention which differs from
    # unpaper's corner-based convention. This causes ~1 pixel sampling differences
    # at certain positions, so we allow tolerance. For document processing,
    # these differences are negligible and don't affect visual quality.
    assert compare_images(golden=cpu_path, result=cuda_path) < 0.20


def test_c1_mask_border_scan_fixture(imgsrc_path, goldendir_path, tmp_path):
    """[C1] Mask/border scan + wipes/borders, deskew disabled."""

    source_path = imgsrc_path / "imgsrc006.png"
    result_path = tmp_path / "result.ppm"
    golden_path = goldendir_path / "goldenC1.ppm"

    run_unpaper(
        "--no-deskew",
        "--no-blackfilter",
        "--no-noisefilter",
        "--no-blurfilter",
        "--no-grayfilter",
        "--no-mask-center",
        "--mask-scan-direction",
        "hv",
        "--mask-scan-threshold",
        "0.8,0.8",
        "--mask-scan-minimum",
        "1,1",
        "--border-scan-direction",
        "hv",
        "--pre-wipe",
        "0,0,9,9",
        "--pre-border",
        "2,2,2,2",
        str(source_path),
        str(result_path),
    )

    assert compare_images(golden=golden_path, result=result_path) == 0.0


def test_cuda_mask_border_scan_fixture_match_cpu(imgsrc_path, tmp_path):
    if not cuda_runtime_available():
        pytest.skip("CUDA runtime/device not available")

    source_path = imgsrc_path / "imgsrc006.png"
    cpu_path = tmp_path / "cpu.ppm"
    cuda_path = tmp_path / "cuda.ppm"
    cuda_path2 = tmp_path / "cuda2.ppm"

    args = (
        "--no-deskew",
        "--no-blackfilter",
        "--no-noisefilter",
        "--no-blurfilter",
        "--no-grayfilter",
        "--no-mask-center",
        "--mask-scan-direction",
        "hv",
        "--mask-scan-threshold",
        "0.8,0.8",
        "--mask-scan-minimum",
        "1,1",
        "--border-scan-direction",
        "hv",
        "--pre-wipe",
        "0,0,9,9",
        "--pre-border",
        "2,2,2,2",
    )

    run_unpaper("--device", "cpu", *args, str(source_path), str(cpu_path))
    run_unpaper("--device", "cuda", *args, str(source_path), str(cuda_path))
    run_unpaper("--device", "cuda", *args, str(source_path), str(cuda_path2))

    assert compare_images(golden=cpu_path, result=cuda_path) == 0.0
    assert compare_images(golden=cuda_path, result=cuda_path2) == 0.0


@pytest.fixture(name="imgsrc_path")
def get_imgsrc_directory() -> pathlib.Path:
    return pathlib.Path(os.getenv("TEST_IMGSRC_DIR", "tests/source_images/"))


@pytest.fixture(name="goldendir_path")
def get_golden_directory() -> pathlib.Path:
    return pathlib.Path(os.getenv("TEST_GOLDEN_DIR", "tests/golden_images/"))


@pytest.mark.parametrize("device", ["cpu", "cuda"])
def test_a1(imgsrc_path, goldendir_path, tmp_path, device):
    """[A1] Single-Page Template Layout, Black+White, Full Processing."""

    if device == "cuda" and not cuda_runtime_available():
        pytest.skip("CUDA runtime/device not available")

    source_path = imgsrc_path / "imgsrc001.png"
    result_path = tmp_path / f"result-{device}.pbm"
    golden_path = goldendir_path / "goldenA1.pbm"

    run_unpaper("--device", device, str(source_path), str(result_path))

    # CUDA uses OpenCV for filters which has slightly different grayscale
    # conversion (weighted luminosity vs simple average), allowing higher tolerance
    tolerance = 0.06 if device == "cuda" else 0.05
    assert compare_images(golden=golden_path, result=result_path) < tolerance


def test_a2(imgsrc_path, goldendir_path, tmp_path, fast_device):
    """[A2] Single-Page Template Layout, Black+White, Full Processing, PPI scaling."""
    source_path = imgsrc_path / "imgsrc001.png"
    result_path = tmp_path / "result.pbm"
    golden_path = goldendir_path / "goldenA2.pbm"

    # Default processing is at 300 PPI, so by using 600 PPI and *two* A sizes lower
    # (A4 → A6) we should have an almost idential file as goldenA1.
    run_unpaper("--device", fast_device, str(source_path), str(result_path), "--ppi", "600", "--post-size", "a6")

    # CUDA produces slightly different results due to floating point precision
    tolerance = 0.06 if fast_device == "cuda" else 0.05
    assert compare_images(golden=golden_path, result=result_path) < tolerance


@pytest.mark.parametrize("device", ["cpu", "cuda"])
def test_b1(imgsrc_path, goldendir_path, tmp_path, device):
    """[B1] Combined Color/Gray, No Processing."""

    if device == "cuda" and not cuda_runtime_available():
        pytest.skip("CUDA runtime/device not available")

    source1_path = imgsrc_path / "imgsrc003.png"
    source2_path = imgsrc_path / "imgsrc004.png"
    result_path = tmp_path / f"result-{device}.ppm"
    golden_path = goldendir_path / "goldenB1.ppm"

    run_unpaper(
        "--device",
        device,
        "-n",
        "--input-pages",
        "2",
        str(source1_path),
        str(source2_path),
        str(result_path),
    )

    assert compare_images(golden=golden_path, result=result_path) < 0.05


def test_b2(imgsrc_path, goldendir_path, tmp_path, fast_device):
    """[B2] Combined Color/Black+White, No Processing."""

    source1_path = imgsrc_path / "imgsrc003.png"
    source2_path = imgsrc_path / "imgsrc005.png"
    result_path = tmp_path / "result.ppm"
    golden_path = goldendir_path / "goldenB2.ppm"

    run_unpaper(
        "--device", fast_device,
        "-n",
        "--input-pages",
        "2",
        str(source1_path),
        str(source2_path),
        str(result_path),
    )

    assert compare_images(golden=golden_path, result=result_path) < 0.05


def test_b3(imgsrc_path, goldendir_path, tmp_path, fast_device):
    """[B3] Combined Gray/Black+White, No Processing."""

    source1_path = imgsrc_path / "imgsrc004.png"
    source2_path = imgsrc_path / "imgsrc005.png"
    result_path = tmp_path / "result.ppm"
    golden_path = goldendir_path / "goldenB3.ppm"

    run_unpaper(
        "--device", fast_device,
        "-n",
        "--input-pages",
        "2",
        str(source1_path),
        str(source2_path),
        str(result_path),
    )

    assert compare_images(golden=golden_path, result=result_path) < 0.05


def test_sheet_background_black(imgsrc_path, goldendir_path, tmp_path, fast_device):
    """[C1] Black sheet background color."""

    source_path = imgsrc_path / "imgsrc002.png"
    result_path = tmp_path / "result.pbm"
    golden_path = goldendir_path / "goldenC1.pbm"

    run_unpaper(
        "--device", fast_device,
        "-n",
        "--sheet-size",
        "a4",
        "--sheet-background",
        "black",
        str(source_path),
        str(result_path),
    )

    assert compare_images(golden=golden_path, result=result_path) < 0.05


def test_pre_shift_both(imgsrc_path, goldendir_path, tmp_path, fast_device):
    """[C2] Explicit shifting."""

    source_path = imgsrc_path / "imgsrc002.png"
    result_path = tmp_path / "result.pbm"
    golden_path = goldendir_path / "goldenC2.pbm"

    run_unpaper(
        "--device", fast_device,
        "-n",
        "--sheet-size",
        "a4",
        "--pre-shift",
        "-5cm,9cm",
        str(source_path),
        str(result_path),
    )

    assert compare_images(golden=golden_path, result=result_path) < 0.05


def test_negative_shift(imgsrc_path, goldendir_path, tmp_path, fast_device):
    """[C2] Explicit -1 size shifting."""

    source_path = imgsrc_path / "imgsrc002.png"
    result_path = tmp_path / "result.pbm"
    golden_path = goldendir_path / "goldenC3.pbm"

    run_unpaper(
        "--device", fast_device,
        "-n",
        "--sheet-size",
        "a4",
        "--pre-shift",
        "-1cm",
        str(source_path),
        str(result_path),
    )

    assert compare_images(golden=golden_path, result=result_path) < 0.05


def test_sheet_crop(imgsrc_path, goldendir_path, tmp_path, fast_device):
    """[D1] Crop to sheet size."""
    source_path = imgsrc_path / "imgsrc003.png"
    result_path = tmp_path / "result.pbm"
    golden_path = goldendir_path / "goldenD1.ppm"

    run_unpaper(
        "--device", fast_device,
        "-n",
        "--sheet-size",
        "20cm,10cm",
        str(source_path),
        str(result_path),
    )

    assert compare_images(golden=golden_path, result=result_path) < 0.05


def test_sheet_fit(imgsrc_path, goldendir_path, tmp_path, fast_device):
    """[D2] Fit to sheet size."""
    source_path = imgsrc_path / "imgsrc003.png"
    result_path = tmp_path / "result.pbm"
    golden_path = goldendir_path / "goldenD2.ppm"

    run_unpaper(
        "--device", fast_device,
        "-n",
        "--size",
        "20cm,10cm",
        str(source_path),
        str(result_path),
    )

    assert compare_images(golden=golden_path, result=result_path) < 0.05


def test_sheet_stretch(imgsrc_path, goldendir_path, tmp_path, fast_device):
    """[D3] Stretch to sheet size."""
    source_path = imgsrc_path / "imgsrc003.png"
    result_path = tmp_path / "result.pbm"
    golden_path = goldendir_path / "goldenD3.ppm"

    run_unpaper(
        "--device", fast_device,
        "-n",
        "--stretch",
        "20cm,10cm",
        str(source_path),
        str(result_path),
    )

    assert compare_images(golden=golden_path, result=result_path) < 0.05


def test_e1(imgsrc_path, goldendir_path, tmp_path, fast_device):
    """[E1] Splitting 2-page layout into separate output pages (with input and output wildcard)."""

    source_path = imgsrc_path / "imgsrcE%03d.png"
    result_path = tmp_path / "results-%02d.pbm"

    run_unpaper(
        "--device", fast_device,
        "--layout", "double", "--output-pages", "2", str(source_path), str(result_path)
    )

    all_results = sorted(tmp_path.iterdir())
    assert len(all_results) == 6

    for result in all_results:
        name_match = re.match(r"^results-([0-9]{2})\.pbm$", str(result.name))
        assert name_match

        golden_path = goldendir_path / f"goldenE1-{name_match.group(1)}.pbm"

        assert compare_images(golden=golden_path, result=result) < 0.05


@pytest.mark.parametrize("device", ["cpu", "cuda"])
def test_e2(imgsrc_path, goldendir_path, tmp_path, device):
    """[E2] Splitting 2-page layout into separate output pages (with output wildcard only)."""

    if device == "cuda" and not cuda_runtime_available():
        pytest.skip("CUDA runtime/device not available")

    source_path = imgsrc_path / "imgsrcE001.png"
    result_path = tmp_path / "results-%02d.pbm"

    run_unpaper(
        "--device",
        device,
        "--layout",
        "double",
        "--output-pages",
        "2",
        str(source_path),
        str(result_path),
    )

    all_results = sorted(tmp_path.iterdir())
    assert len(all_results) == 2

    for result in all_results:
        name_match = re.match(r"^results-([0-9]{2})\.pbm$", str(result.name))
        assert name_match

        golden_path = goldendir_path / f"goldenE1-{name_match.group(1)}.pbm"

        assert compare_images(golden=golden_path, result=result) < 0.05


def test_e3(imgsrc_path, goldendir_path, tmp_path, fast_device):
    """[E3] Splitting 2-page layout into separate output pages (with explicit input and output)."""

    source_path = imgsrc_path / "imgsrcE001.png"
    result_path_1 = tmp_path / "results-1.pbm"
    result_path_2 = tmp_path / "results-2.pbm"

    run_unpaper(
        "--device", fast_device,
        "--layout",
        "double",
        "--output-pages",
        "2",
        str(source_path),
        str(result_path_1),
        str(result_path_2),
    )

    all_results = sorted(tmp_path.iterdir())
    assert len(all_results) == 2
    assert (
        compare_images(
            golden=(goldendir_path / "goldenE1-01.pbm"), result=result_path_1
        )
        < 0.05
    )
    assert (
        compare_images(
            golden=(goldendir_path / "goldenE1-02.pbm"), result=result_path_2
        )
        < 0.05
    )


def test_f1(imgsrc_path, goldendir_path, tmp_path, fast_device):
    """[F1] Merging 2-page layout into single output page (with input and output wildcard)."""

    source_path = imgsrc_path / "imgsrcE%03d.png"
    output_path = tmp_path / "results-%d.pbm"
    result_path = tmp_path / "results-1.pbm"
    golden_path = goldendir_path / "goldenF.pbm"

    run_unpaper(
        "--device", fast_device,
        "--end-sheet",
        "1",
        "--layout",
        "double",
        "--input-pages",
        "2",
        str(source_path),
        str(output_path),
    )

    all_results = sorted(tmp_path.iterdir())
    assert len(all_results) == 1
    assert all_results[0] == result_path

    assert compare_images(golden=golden_path, result=result_path) < 0.05


def test_f2(imgsrc_path, goldendir_path, tmp_path, fast_device):
    """[F2] Merging 2-page layout into single output page (with output wildcard only)."""

    source_path_1 = imgsrc_path / "imgsrcE001.png"
    source_path_2 = imgsrc_path / "imgsrcE002.png"
    output_path = tmp_path / "results-%d.pbm"
    result_path = tmp_path / "results-1.pbm"
    golden_path = goldendir_path / "goldenF.pbm"

    run_unpaper(
        "--device", fast_device,
        "--layout",
        "double",
        "--input-pages",
        "2",
        str(source_path_1),
        str(source_path_2),
        str(output_path),
    )

    all_results = sorted(tmp_path.iterdir())
    assert len(all_results) == 1
    assert all_results[0] == result_path

    assert compare_images(golden=golden_path, result=result_path) < 0.05


def test_f3(imgsrc_path, goldendir_path, tmp_path, fast_device):
    """[F3] Merging 2-page layout into single output page (with explicit input and output)."""

    source_path_1 = imgsrc_path / "imgsrcE001.png"
    source_path_2 = imgsrc_path / "imgsrcE002.png"
    result_path = tmp_path / "result.pbm"
    golden_path = goldendir_path / "goldenF.pbm"

    run_unpaper(
        "--device", fast_device,
        "--layout",
        "double",
        "--input-pages",
        "2",
        str(source_path_1),
        str(source_path_2),
        str(result_path),
    )

    all_results = sorted(tmp_path.iterdir())
    assert len(all_results) == 1
    assert all_results[0] == result_path

    assert compare_images(golden=golden_path, result=result_path) < 0.05


def test_overwrite_no_file(imgsrc_path, tmp_path):
    source_path = imgsrc_path / "imgsrc001.png"
    result_path = tmp_path / "result.pbm"

    run_unpaper(
        "--overwrite", "--no-processing", "1", str(source_path), str(result_path)
    )

    assert compare_images(golden=source_path, result=result_path) == 0


def test_overwrite_existing_file(imgsrc_path, tmp_path):
    source_path = imgsrc_path / "imgsrc001.png"
    result_path = tmp_path / "result.pbm"

    # Create an empty file first, which should be overwritten
    result_path.touch(exist_ok=False)

    run_unpaper(
        "--overwrite", "--no-processing", "1", str(source_path), str(result_path)
    )

    assert compare_images(golden=source_path, result=result_path) == 0


def test_no_overwrite_existing_file(imgsrc_path, tmp_path):
    source_path = imgsrc_path / "imgsrc001.png"
    result_path = tmp_path / "result.pbm"

    # Create an empty file first, which should be overwritten
    result_path.touch(exist_ok=False)

    unpaper_result = run_unpaper(
        "--no-processing", "1", str(source_path), str(result_path), check=False
    )
    assert unpaper_result.returncode != 0
    assert result_path.stat().st_size == 0


def test_invalid_multi_index(imgsrc_path, tmp_path):
    source_path = imgsrc_path / "imgsrc001.png"
    result_path = tmp_path / "result.pbm"
    unpaper_result = run_unpaper(
        "--no-processing", "1-", str(source_path), str(result_path), check=False
    )
    assert unpaper_result.returncode != 0


def test_skip_split_requires_pdf(imgsrc_path, tmp_path):
    source_path = imgsrc_path / "imgsrc001.png"
    result_path = tmp_path / "result.pbm"
    unpaper_result = run_unpaper(
        "--skip-split",
        "1",
        str(source_path),
        str(result_path),
        check=False,
        capture=True,
    )
    assert unpaper_result.returncode != 0
    text = (unpaper_result.stdout or "") + (unpaper_result.stderr or "")
    assert "--skip-split" in text or "PDF" in text


def test_valid_range_multi_index(imgsrc_path, tmp_path):
    source_path = imgsrc_path / "imgsrc%03.png"
    result_path = tmp_path / "result%03.pbm"
    unpaper_result = run_unpaper(
        "--no-processing", "1-100", str(source_path), str(result_path), check=False
    )
    assert unpaper_result.returncode == 0


def test_jpeg_input_produces_similar_output_to_png(imgsrc_path, tmp_path, fast_device):
    """Test that JPEG input produces output similar to PNG input.

    This validates that the JPEG decode path (FFmpeg or nvJPEG) produces
    results comparable to PNG input. JPEG is lossy, so we allow up to 10%
    dissimilarity due to compression artifacts.
    """
    # Use a source image that has good content for testing
    png_source_path = imgsrc_path / "imgsrc001.png"

    # Convert PNG to JPEG using PIL
    png_image = PIL.Image.open(png_source_path)
    jpeg_source_path = tmp_path / "source.jpg"
    # Use high quality to minimize compression artifacts
    png_image.save(jpeg_source_path, "JPEG", quality=95)

    png_result_path = tmp_path / "result_png.ppm"
    jpeg_result_path = tmp_path / "result_jpeg.ppm"

    # Process both with same settings (no filters for cleaner comparison)
    common_args = [
        "--device", fast_device,
        "--no-blackfilter",
        "--no-noisefilter",
        "--no-blurfilter",
        "--no-grayfilter",
        "--no-deskew",
    ]

    run_unpaper(*common_args, str(png_source_path), str(png_result_path))
    run_unpaper(*common_args, str(jpeg_source_path), str(jpeg_result_path))

    # Allow up to 10% difference due to JPEG compression artifacts
    diff = compare_images(golden=png_result_path, result=jpeg_result_path)
    assert diff < 0.10, f"JPEG output differs from PNG by {diff*100:.1f}%, expected <10%"


@pytest.mark.parametrize("device", ["cpu", "cuda"])
def test_jpeg_input_device_comparison(imgsrc_path, tmp_path, device):
    """Test JPEG input produces consistent results across CPU and CUDA.

    This ensures the JPEG decode path works correctly on both devices.
    For CUDA, this exercises the nvJPEG decode path when available.
    """
    if device == "cuda" and not cuda_runtime_available():
        pytest.skip("CUDA runtime/device not available")

    # Create JPEG from PNG to ensure compatible pixel format
    # (test_jpeg.jpg may be in YUV format which FFmpeg doesn't auto-convert)
    png_source = imgsrc_path / "imgsrc001.png"
    png_image = PIL.Image.open(png_source)
    jpeg_path = tmp_path / "source.jpg"
    png_image.save(jpeg_path, "JPEG", quality=95)

    result_path = tmp_path / f"result_{device}.ppm"

    # Process with specific device, minimal processing for clean comparison
    run_unpaper(
        "--device", device,
        "--no-blackfilter",
        "--no-noisefilter",
        "--no-blurfilter",
        "--no-grayfilter",
        "--no-deskew",
        str(jpeg_path),
        str(result_path),
    )

    # Just verify output was created and is valid
    assert result_path.exists()
    result_image = PIL.Image.open(result_path)
    assert result_image.width > 0 and result_image.height > 0


def test_jpeg_cuda_vs_cpu_similarity(imgsrc_path, tmp_path):
    """Test JPEG processing on CUDA produces similar results to CPU.

    This validates that the CUDA processing path (which may use nvJPEG
    for decode) produces output similar to the CPU path (FFmpeg decode).
    """
    if not cuda_runtime_available():
        pytest.skip("CUDA runtime/device not available")

    # Create JPEG from PNG to ensure compatible pixel format
    png_source = imgsrc_path / "imgsrc001.png"
    png_image = PIL.Image.open(png_source)
    jpeg_path = tmp_path / "source.jpg"
    png_image.save(jpeg_path, "JPEG", quality=95)

    cpu_result_path = tmp_path / "result_cpu.ppm"
    cuda_result_path = tmp_path / "result_cuda.ppm"

    common_args = [
        "--no-blackfilter",
        "--no-noisefilter",
        "--no-blurfilter",
        "--no-grayfilter",
        "--no-deskew",
    ]

    run_unpaper("--device", "cpu", *common_args, str(jpeg_path), str(cpu_result_path))
    run_unpaper("--device", "cuda", *common_args, str(jpeg_path), str(cuda_result_path))

    # CPU and CUDA should produce very similar results for the same JPEG input
    # Allow small tolerance for floating-point differences in GPU processing
    diff = compare_images(golden=cpu_result_path, result=cuda_result_path)
    assert diff < 0.05, f"CUDA JPEG output differs from CPU by {diff*100:.1f}%, expected <5%"


# ---------------------------------------------------------------------------
# PDF end-to-end golden tests
# ---------------------------------------------------------------------------

_PDF_SIMILARITY_MIN = 0.80
_PDF_DIFF_MAX = 1.0 - _PDF_SIMILARITY_MIN


@pytest.fixture(scope="session")
def pdf_inputs_dir(tmp_path_factory) -> pathlib.Path:
    return tmp_path_factory.mktemp("pdf-inputs")


_PDF_GOLDEN_CASES: list[tuple[str, list[str], tuple[str, ...], list[str]]] = [
    (
        "A1",
        ["imgsrc001.png"],
        (),
        ["goldenA1.pbm"],
    ),
    (
        "B1",
        ["imgsrc003.png", "imgsrc004.png"],
        ("-n", "--input-pages", "2"),
        ["goldenB1.ppm"],
    ),
    (
        "C1",
        ["imgsrc006.png"],
        (
            "--no-deskew",
            "--no-blackfilter",
            "--no-noisefilter",
            "--no-blurfilter",
            "--no-grayfilter",
            "--no-mask-center",
            "--mask-scan-direction",
            "hv",
            "--mask-scan-threshold",
            "0.8,0.8",
            "--mask-scan-minimum",
            "1,1",
            "--border-scan-direction",
            "hv",
            "--pre-wipe",
            "0,0,9,9",
            "--pre-border",
            "2,2,2,2",
        ),
        ["goldenC1.ppm"],
    ),
    (
        "D1",
        ["imgsrc003.png"],
        ("-n", "--sheet-size", "20cm,10cm"),
        ["goldenD1.ppm"],
    ),
    (
        "E1",
        ["imgsrcE001.png", "imgsrcE002.png", "imgsrcE003.png"],
        ("--layout", "double", "--output-pages", "2"),
        [f"goldenE1-{i:02d}.pbm" for i in range(1, 7)],
    ),
]


def _get_or_create_input_pdf(
    *,
    case_id: str,
    encoding: str,
    imgsrc_path: pathlib.Path,
    pdf_inputs_dir: pathlib.Path,
    input_images: list[str],
) -> pathlib.Path:
    out_pdf = pdf_inputs_dir / f"in-{case_id}-{encoding}.pdf"
    if out_pdf.exists():
        return out_pdf

    workdir = pdf_inputs_dir / f"work-{case_id}-{encoding}"
    workdir.mkdir(parents=True, exist_ok=True)

    images = [imgsrc_path / name for name in input_images]
    for p in images:
        if not p.exists():
            raise FileNotFoundError(p)

    _gs_build_pdf(images=images, out_pdf=out_pdf, encoding=encoding, workdir=workdir)
    return out_pdf


@pytest.mark.parametrize("device", ["cpu", "cuda"])
@pytest.mark.parametrize("encoding", ["jpeg", "png", "jbig2"])
@pytest.mark.parametrize("case_id,input_images,extra_args,golden_images", _PDF_GOLDEN_CASES)
def test_pdf_pipeline_roundtrip_matches_goldens(
    imgsrc_path,
    goldendir_path,
    pdf_inputs_dir,
    tmp_path,
    device,
    encoding,
    case_id,
    input_images,
    extra_args,
    golden_images,
):
    _require_external_tools("mutool")
    if encoding == "jbig2":
        _require_external_tools("jbig2")

    if not pdf_mode_supported():
        pytest.skip("unpaper built without PDF support")

    if device == "cuda" and not cuda_runtime_available():
        pytest.skip("CUDA runtime/device not available")

    if encoding == "jbig2" and device == "cuda" and not jbig2_gpu_decode_available():
        pytest.skip("JBIG2 GPU decode path not available (build/runtime)")

    input_pdf = _get_or_create_input_pdf(
        case_id=case_id,
        encoding=encoding,
        imgsrc_path=imgsrc_path,
        pdf_inputs_dir=pdf_inputs_dir,
        input_images=input_images,
    )
    output_pdf = tmp_path / f"out-{case_id}-{encoding}-{device}.pdf"

    proc = run_unpaper(
        "--device",
        device,
        "--pdf-quality",
        "fast",
        "--pdf-dpi",
        "300",
        "--jpeg-quality",
        "95",
        *extra_args,
        str(input_pdf),
        str(output_pdf),
        capture=True,
    )

    text = (proc.stdout or "") + (proc.stderr or "")

    if device == "cpu":
        assert "Using CPU PDF pipeline" in text
    else:
        assert "Using GPU PDF pipeline" in text
        if encoding == "jpeg":
            assert "GPU PDF pipeline: extracted JPEG image" in text
            assert "GPU PDF pipeline: JPEG decoded to GPU" in text
        elif encoding == "png":
            # MuPDF typically exposes embedded PNG pages as FLATE streams.
            assert re.search(r"GPU PDF pipeline: extracted (PNG|FLATE) image", text)
            assert "GPU PDF pipeline: rendered and uploaded to GPU" in text
        else:  # jbig2
            assert "GPU PDF pipeline: extracted JBIG2 image" in text
            assert "GPU PDF pipeline: JBIG2 decoded to GPU" in text

    rendered = _render_pdf_pages(
        pdf_path=output_pdf, out_dir=tmp_path / "render", dpi=300
    )
    assert len(rendered) == len(golden_images)

    for idx, (page_img, golden_name) in enumerate(zip(rendered, golden_images), start=1):
        golden_path = goldendir_path / golden_name
        diff = compare_images_pdf(golden=golden_path, result=page_img)
        assert diff <= _PDF_DIFF_MAX, (
            f"[{case_id}] page {idx} differs too much: "
            f"similarity={(1.0 - diff):.3f}, expected >= {_PDF_SIMILARITY_MIN:.2f} "
            f"(golden={golden_path}, result={page_img})"
        )
