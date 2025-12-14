<!--
SPDX-FileCopyrightText: 2005 The unpaper authors

SPDX-License-Identifier: GPL-2.0-only
-->

unpaper
=======

Originally written by Jens Gulden — see AUTHORS for more information.
The entire `unpaper` project is licensed under GNU GPL v2.
Some of the individual files are licensed under the MIT or Apache 2.0 licenses.
Each file contains an [SPDX license header](https://reuse.software/)
specifying its license. The text of all three licenses is available under
`LICENSES`.

Overview
--------

`unpaper` is a post-processing tool for scanned sheets of paper,
especially for book pages that have been scanned from previously
created photocopies.  The main purpose is to make scanned book pages
better readable on screen after conversion to PDF. Additionally,
`unpaper` might be useful to enhance the quality of scanned pages
before performing optical character recognition (OCR).

`unpaper` tries to clean scanned images by removing dark edges that
appeared through scanning or copying on areas outside the actual page
content (e.g.  dark areas between the left-hand-side and the
right-hand-side of a double- sided book-page scan).

The program also tries to detect misaligned centering and rotation of
pages and will automatically straighten each page by rotating it to
the correct angle. This process is called "deskewing".

Note that the automatic processing will sometimes fail. It is always a
good idea to manually control the results of unpaper and adjust the
parameter settings according to the requirements of the input. Each
processing step can also be disabled individually for each sheet.

See [further documentation][3] for the supported file formats notes.

Dependencies
------------

The only hard dependency of `unpaper` is [ffmpeg][4], which is used for
file input and output.

### CUDA Backend Dependencies

For GPU-accelerated processing (`--device=cuda`), the following are required:

- **CUDA Toolkit**: Tested with CUDA 12.x and 13.x. The `nvcc` compiler
  and CUDA runtime library (`cudart`) must be available.
- **OpenCV 4.x with CUDA support**: Required for CUDA builds. OpenCV must
  be built with CUDA support enabled, including the `cudaarithm` and
  `cudaimgproc` modules. OpenCV provides GPU-accelerated operations
  including connected component labeling for the noisefilter.

Building instructions
---------------------

`unpaper` uses [the Meson Build system](https://mesonbuild.com), which
can be installed using Python's package manage (`pip3` or `pip`):

    unpaper$ pip3 install --user 'meson >= 0.57' 'sphinx >= 3.4'
    unpaper$ CFLAGS="-march=native" meson setup --buildtype=debugoptimized builddir
    unpaper$ meson compile -C builddir

You can pass required optimization flags when creating the meson build
directory in the `CFLAGS` environment variable. Usage of Link-Time
Optimizations (Meson option `-Db_lto=true`) is recommended if
available.

Further optimizations such as `-ftracer` and `-ftree-vectorize` are
thought to work, but their effect has not been evaluated so your
mileage may vary.

Tests depend on `pytest` and `pillow`, which will be auto-detected by
Meson.

### Building with CUDA Support

To enable GPU-accelerated processing, configure with `-Dcuda=enabled`:

    unpaper$ meson setup builddir-cuda -Dcuda=enabled --buildtype=debugoptimized
    unpaper$ meson compile -C builddir-cuda

The CUDA backend requires:
- NVIDIA CUDA Toolkit (nvcc compiler and cudart library)
- OpenCV 4.x with CUDA support (cudaarithm and cudaimgproc modules)

Use `--device=cuda` at runtime to select GPU processing.

To check which backends are active at runtime, use `--perf`:

    unpaper$ ./builddir-cuda/unpaper --perf --device=cuda input.pgm output.pgm
    # Output includes: perf backends: device=cuda opencv=yes ccl=yes

### Recommended Pixel Formats for CUDA

For best CUDA performance, use these pixel formats:

| Image Type | Recommended Format | Notes |
|------------|-------------------|-------|
| **Grayscale** | **GRAY8** (8-bit grayscale) | Full OpenCV CUDA acceleration |
| **Color** | **RGB24** (24-bit RGB) | Full OpenCV CUDA acceleration |

These formats benefit from optimized OpenCV CUDA primitives including
`cv::cuda::transpose`, `cv::cuda::flip`, and `cv::cuda::warpAffine`.

Other formats like Y400A (grayscale with alpha) and 1-bit mono (MONOWHITE,
MONOBLACK) are supported but use custom CUDA kernels since OpenCV lacks
native support for 2-channel and bit-packed images.

FFmpeg automatically selects pixel format based on input. To ensure optimal
format, you can pre-convert images:

    # Convert to GRAY8 for grayscale scans
    ffmpeg -i input.tiff -pix_fmt gray output.pgm

    # Convert to RGB24 for color scans
    ffmpeg -i input.tiff -pix_fmt rgb24 output.ppm

Development Hints
-----------------

The project includes configuration for [pre-commit](https://pre-commit.com/)
which is integrated with GitHub Actions CI. If you're using git for
devleopment, you can install it with
`pip install pre-commit && pre-commit --install`.

Using [Sapling](https://sapling-scm.com/) with this repository is possible
and diffs can be reviewed as a stack.

Further Information
-------------------

You can find more information on the [basic concepts][1] and the
[image processing][2] in the available documentation.

[1]: doc/basic-concepts.md
[2]: doc/image-processing.md
[3]: doc/file-formats.md
[4]: https://www.ffmpeg.org/
