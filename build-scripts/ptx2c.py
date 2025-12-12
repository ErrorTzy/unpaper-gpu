#!/usr/bin/env python3
# SPDX-FileCopyrightText: 2025 The unpaper authors
#
# SPDX-License-Identifier: GPL-2.0-only

import pathlib
import sys


def c_escape(s: str) -> str:
    return (
        s.replace("\\", "\\\\")
        .replace('"', '\\"')
        .replace("\r\n", "\n")
        .replace("\r", "\n")
    )


def main() -> int:
    if len(sys.argv) != 3:
        print("usage: ptx2c.py <input.ptx> <output.c>", file=sys.stderr)
        return 2

    ptx_path = pathlib.Path(sys.argv[1])
    out_path = pathlib.Path(sys.argv[2])

    ptx = ptx_path.read_text(encoding="utf-8")
    ptx = c_escape(ptx)

    lines = [
        "// SPDX-FileCopyrightText: 2025 The unpaper authors",
        "//",
        "// SPDX-License-Identifier: GPL-2.0-only",
        "",
        '#include <stddef.h>',
        "",
        "const char unpaper_cuda_kernels_ptx[] =",
    ]
    for line in ptx.split("\n"):
        lines.append(f'"{line}\\n"')
    lines.append(";")
    lines.append("")
    lines.append(
        "const size_t unpaper_cuda_kernels_ptx_size = sizeof(unpaper_cuda_kernels_ptx);"
    )
    lines.append("")

    out_path.write_text("\n".join(lines), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

