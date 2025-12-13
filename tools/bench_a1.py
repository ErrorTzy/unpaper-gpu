#!/usr/bin/env python3
# SPDX-FileCopyrightText: 2025 The unpaper authors
#
# SPDX-License-Identifier: GPL-2.0-only

import argparse
import statistics
import subprocess
import sys
import tempfile
import time
from pathlib import Path


def run_once(binary: Path, device: str, input_path: Path, output_path: Path) -> float:
    start = time.perf_counter()
    proc = subprocess.run(
        [
            str(binary),
            "--device",
            device,
            str(input_path),
            str(output_path),
        ],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.PIPE,
        check=False,
    )
    end = time.perf_counter()
    if proc.returncode != 0:
        raise RuntimeError(
            f"{binary} --device {device} failed "
            f"(exit {proc.returncode}): {proc.stderr.decode().strip()}"
        )
    return (end - start) * 1000.0


def bench_device(binary: Path, device: str, input_path: Path, tmp_dir: Path,
                 warmup: int, iterations: int) -> None:
    if not binary.exists():
        print(f"[skip] {device}: binary {binary} is missing", file=sys.stderr)
        return

    output_path = tmp_dir / f"bench_a1_{device}.pgm"

    for _ in range(warmup):
        try:
            run_once(binary, device, input_path, output_path)
        except RuntimeError as exc:
            print(f"[skip] {device} warmup failed: {exc}", file=sys.stderr)
            return

    samples = []
    for _ in range(iterations):
        try:
            samples.append(run_once(binary, device, input_path, output_path))
        except RuntimeError as exc:
            print(f"[skip] {device} iteration failed: {exc}", file=sys.stderr)
            return

    mean = statistics.mean(samples)
    stdev = statistics.pstdev(samples) if len(samples) > 1 else 0.0
    print(f"{device.upper():<4} mean={mean:.2f}ms stdev={stdev:.2f}ms "
          f"runs={iterations} warmup={warmup} output={output_path}")


def main() -> int:
    repo_root = Path(__file__).resolve().parent.parent
    default_input = repo_root / "tests" / "source_images" / "imgsrc001.png"
    default_tmp = Path("/dev/shm") if Path("/dev/shm").is_dir() else Path(
        tempfile.gettempdir())

    parser = argparse.ArgumentParser(
        description="Benchmark unpaper on imgsrc001.png for CPU/CUDA devices.")
    parser.add_argument("--builddir", default=repo_root / "builddir",
                        type=Path, help="Meson builddir for CPU binary")
    parser.add_argument("--builddir-cuda", default=repo_root / "builddir-cuda",
                        type=Path, help="Meson builddir for CUDA binary")
    parser.add_argument("--input", default=default_input, type=Path,
                        help="Input image to process")
    parser.add_argument("--tmpdir", default=default_tmp, type=Path,
                        help="Directory for temporary output (use tmpfs if possible)")
    parser.add_argument("--warmup", default=1, type=int,
                        help="Warmup runs per device")
    parser.add_argument("--iterations", default=5, type=int,
                        help="Measured runs per device")
    parser.add_argument("--devices", default="cpu,cuda",
                        help="Comma-separated list of devices to benchmark")

    args = parser.parse_args()

    if not args.input.exists():
        print(f"input file not found: {args.input}", file=sys.stderr)
        return 1

    tmpdir = args.tmpdir
    tmpdir.mkdir(parents=True, exist_ok=True)

    device_list = [d.strip() for d in args.devices.split(",") if d.strip()]
    for device in device_list:
        if device not in {"cpu", "cuda"}:
            print(f"[skip] unknown device '{device}'", file=sys.stderr)
            continue
        builddir = args.builddir if device == "cpu" else args.builddir_cuda
        binary = builddir / "unpaper"
        bench_device(binary, device, args.input, tmpdir, args.warmup,
                     args.iterations)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
