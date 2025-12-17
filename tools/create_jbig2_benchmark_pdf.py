#!/usr/bin/env python3
# SPDX-FileCopyrightText: 2025 The unpaper authors
# SPDX-License-Identifier: GPL-2.0-only

"""
Create a multi-page JBIG2 PDF for benchmarking.

Usage: python create_jbig2_benchmark_pdf.py [num_pages] [output_path]

Creates realistic B&W document pages with varying content patterns.
"""

import subprocess
import sys
import tempfile
from pathlib import Path

# Try to import PIL, fall back to basic approach if not available
try:
    from PIL import Image, ImageDraw, ImageFont
    HAS_PIL = True
except ImportError:
    HAS_PIL = False
    print("Warning: PIL not available, using basic pattern generation")


def create_document_page_pil(page_num: int, width: int, height: int) -> Image.Image:
    """Create a realistic B&W document page using PIL."""
    # White background
    img = Image.new('1', (width, height), 1)
    draw = ImageDraw.Draw(img)

    # Add page header
    header_y = 50
    draw.rectangle([50, header_y, width - 50, header_y + 40], fill=0)

    # Add "text" lines (simulated as horizontal bars with gaps)
    line_height = 20
    line_spacing = 30
    margin_left = 80
    margin_right = width - 80

    y = 120
    paragraph = 0
    while y < height - 100:
        # Vary line lengths to simulate real text
        if paragraph % 3 == 0:
            # Short paragraph start (indented)
            line_start = margin_left + 40
        else:
            line_start = margin_left

        # Vary line endings
        import random
        random.seed(page_num * 1000 + y)  # Deterministic per page/line
        line_end = margin_right - random.randint(0, 100)

        # Draw "text" line (thin rectangle)
        if y % (line_spacing * 5) != 0:  # Skip some lines for paragraph breaks
            draw.rectangle([line_start, y, line_end, y + 3], fill=0)
        else:
            paragraph += 1
            y += line_spacing  # Extra space for paragraph

        y += line_spacing

    # Add page number at bottom
    page_str = f"- {page_num + 1} -"
    # Center the page number
    text_width = len(page_str) * 8
    draw.rectangle([width // 2 - text_width // 2, height - 50,
                   width // 2 + text_width // 2, height - 40], fill=0)

    # Add some "graphics" elements on certain pages
    if page_num % 5 == 0:
        # Add a box/figure placeholder
        box_y = height // 2
        draw.rectangle([100, box_y, width - 100, box_y + 150], outline=0, width=2)
        # Cross-hatch pattern inside
        for i in range(105, width - 105, 20):
            draw.line([(i, box_y + 5), (i, box_y + 145)], fill=0)

    return img


def create_document_page_basic(page_num: int, width: int, height: int, tmpdir: Path) -> Path:
    """Create a basic B&W PBM image without PIL."""
    # Create a simple pattern as PBM
    pbm_path = tmpdir / f"page_{page_num:03d}.pbm"

    # PBM format: P4 is binary, P1 is ASCII
    # Use P4 (binary) for efficiency
    with open(pbm_path, 'wb') as f:
        # Header
        f.write(f"P4\n{width} {height}\n".encode())

        # Binary data: 1 bit per pixel, packed into bytes
        # 0 = black, 1 = white in PBM
        row_bytes = (width + 7) // 8

        for y in range(height):
            row = bytearray(row_bytes)
            # Fill with white (0xFF)
            for i in range(row_bytes):
                row[i] = 0xFF

            # Add some black content
            # Header bar
            if 50 <= y <= 90:
                for x in range(50, width - 50):
                    byte_idx = x // 8
                    bit_idx = 7 - (x % 8)
                    row[byte_idx] &= ~(1 << bit_idx)  # Set to black

            # Text lines
            if y > 120 and y < height - 100:
                line_in_block = (y - 120) % 30
                if line_in_block < 4 and (y - 120) % 150 < 120:  # Text line
                    for x in range(80, width - 80 - (page_num * 3) % 50):
                        byte_idx = x // 8
                        bit_idx = 7 - (x % 8)
                        row[byte_idx] &= ~(1 << bit_idx)

            # Page number area
            if height - 50 <= y <= height - 40:
                for x in range(width // 2 - 30, width // 2 + 30):
                    byte_idx = x // 8
                    bit_idx = 7 - (x % 8)
                    row[byte_idx] &= ~(1 << bit_idx)

            f.write(bytes(row))

    return pbm_path


def create_jbig2_pdf(num_pages: int, output_path: str, width: int = 2480, height: int = 3508):
    """
    Create a PDF with JBIG2-encoded pages.

    Args:
        num_pages: Number of pages to generate
        output_path: Output PDF path
        width: Page width in pixels (default: A4 @ 300 DPI)
        height: Page height in pixels (default: A4 @ 300 DPI)
    """
    print(f"Creating {num_pages}-page JBIG2 PDF: {output_path}")
    print(f"Page size: {width}x{height} pixels (A4 @ 300 DPI)")

    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir = Path(tmpdir)

        # Generate page images
        print("Generating page images...")
        pbm_files = []

        for i in range(num_pages):
            if HAS_PIL:
                img = create_document_page_pil(i, width, height)
                pbm_path = tmpdir / f"page_{i:03d}.pbm"
                img.save(pbm_path)
            else:
                pbm_path = create_document_page_basic(i, width, height, tmpdir)

            pbm_files.append(pbm_path)
            if (i + 1) % 10 == 0:
                print(f"  Generated {i + 1}/{num_pages} pages")

        # Convert to JBIG2 - each page separately
        # With -p flag, jbig2 outputs to stdout
        print("Converting to JBIG2...")
        pages_data = []

        for i, pbm_path in enumerate(pbm_files):
            jbig2_args = ["jbig2", "-p", str(pbm_path)]
            result = subprocess.run(jbig2_args, capture_output=True)
            if result.returncode != 0:
                print(f"jbig2 error on page {i}: {result.stderr.decode('utf-8', errors='replace')}")
                sys.exit(1)
            pages_data.append(result.stdout)
            if (i + 1) % 10 == 0:
                print(f"  Converted {i + 1}/{num_pages} pages")

        # No globals in simple mode
        globals_data = None
        print(f"Generated {len(pages_data)} JBIG2 streams")

        total_jbig2 = sum(len(p) for p in pages_data)
        if globals_data:
            total_jbig2 += len(globals_data)
        print(f"Total JBIG2 data: {total_jbig2:,} bytes")

        # Build PDF
        print("Building PDF...")
        build_pdf_with_jbig2(output_path, pages_data, globals_data, width, height)

        output_size = Path(output_path).stat().st_size
        print(f"Created: {output_path} ({output_size:,} bytes)")
        print(f"Compression ratio: {(width * height * num_pages / 8) / output_size:.1f}x")


def build_pdf_with_jbig2(output_path: str, pages_data: list, globals_data: bytes | None,
                          width: int, height: int, dpi: int = 300):
    """Build a PDF file with JBIG2 image streams."""

    # Calculate page size in points (72 points per inch)
    page_width_pt = width * 72 / dpi
    page_height_pt = height * 72 / dpi

    # Object tracking
    objects = []
    xref_offsets = []

    def add_object(content: bytes) -> int:
        """Add an object and return its number (1-based)."""
        obj_num = len(objects) + 1
        objects.append(content)
        return obj_num

    # Build objects
    # 1: Catalog
    catalog_content = b"<< /Type /Catalog /Pages 2 0 R >>"
    catalog_obj = add_object(catalog_content)

    # 2: Pages (placeholder, will update later)
    pages_obj = add_object(b"PLACEHOLDER")

    # 3: Global symbols (if present)
    globals_obj = None
    if globals_data:
        globals_stream = (
            f"<< /Length {len(globals_data)} >>\n"
            f"stream\n"
        ).encode() + globals_data + b"\nendstream"
        globals_obj = add_object(globals_stream)

    # Generate page objects
    page_obj_nums = []
    for i, page_data in enumerate(pages_data):
        # Image XObject
        decode_parms = f"/JBIG2Globals {globals_obj} 0 R " if globals_obj else ""
        img_dict = (
            f"<< /Type /XObject /Subtype /Image "
            f"/Width {width} /Height {height} "
            f"/ColorSpace /DeviceGray /BitsPerComponent 1 "
            f"/Filter /JBIG2Decode "
            f"/DecodeParms << {decode_parms}/JBIG2Globals {globals_obj} 0 R >> " if globals_obj else
            f"<< /Type /XObject /Subtype /Image "
            f"/Width {width} /Height {height} "
            f"/ColorSpace /DeviceGray /BitsPerComponent 1 "
            f"/Filter /JBIG2Decode "
        )
        if globals_obj:
            img_dict = (
                f"<< /Type /XObject /Subtype /Image "
                f"/Width {width} /Height {height} "
                f"/ColorSpace /DeviceGray /BitsPerComponent 1 "
                f"/Filter /JBIG2Decode "
                f"/DecodeParms << /JBIG2Globals {globals_obj} 0 R >> "
                f"/Length {len(page_data)} >>"
            )
        else:
            img_dict = (
                f"<< /Type /XObject /Subtype /Image "
                f"/Width {width} /Height {height} "
                f"/ColorSpace /DeviceGray /BitsPerComponent 1 "
                f"/Filter /JBIG2Decode "
                f"/Length {len(page_data)} >>"
            )

        img_stream = img_dict.encode() + b"\nstream\n" + page_data + b"\nendstream"
        img_obj = add_object(img_stream)

        # Content stream (draw image)
        content = f"q {page_width_pt:.2f} 0 0 {page_height_pt:.2f} 0 0 cm /Im0 Do Q".encode()
        content_stream = f"<< /Length {len(content)} >>\nstream\n".encode() + content + b"\nendstream"
        content_obj = add_object(content_stream)

        # Resources
        resources_dict = f"<< /XObject << /Im0 {img_obj} 0 R >> >>".encode()
        resources_obj = add_object(resources_dict)

        # Page
        page_dict = (
            f"<< /Type /Page /Parent 2 0 R "
            f"/MediaBox [0 0 {page_width_pt:.2f} {page_height_pt:.2f}] "
            f"/Contents {content_obj} 0 R "
            f"/Resources {resources_obj} 0 R >>"
        ).encode()
        page_obj = add_object(page_dict)
        page_obj_nums.append(page_obj)

    # Update Pages object
    kids_str = " ".join(f"{n} 0 R" for n in page_obj_nums)
    pages_content = f"<< /Type /Pages /Kids [{kids_str}] /Count {len(page_obj_nums)} >>".encode()
    objects[pages_obj - 1] = pages_content

    # Write PDF
    with open(output_path, 'wb') as f:
        # Header
        f.write(b"%PDF-1.4\n%\xe2\xe3\xcf\xd3\n")

        # Objects
        for i, obj_content in enumerate(objects):
            xref_offsets.append(f.tell())
            f.write(f"{i + 1} 0 obj\n".encode())
            f.write(obj_content)
            f.write(b"\nendobj\n")

        # Xref
        xref_start = f.tell()
        f.write(b"xref\n")
        f.write(f"0 {len(objects) + 1}\n".encode())
        f.write(b"0000000000 65535 f \n")
        for offset in xref_offsets:
            f.write(f"{offset:010d} 00000 n \n".encode())

        # Trailer
        f.write(b"trailer\n")
        f.write(f"<< /Size {len(objects) + 1} /Root {catalog_obj} 0 R >>\n".encode())
        f.write(b"startxref\n")
        f.write(f"{xref_start}\n".encode())
        f.write(b"%%EOF\n")


def main():
    num_pages = 50
    output_path = "tests/pdf_samples/benchmark_jbig2_50page.pdf"

    if len(sys.argv) > 1:
        num_pages = int(sys.argv[1])
    if len(sys.argv) > 2:
        output_path = sys.argv[2]

    create_jbig2_pdf(num_pages, output_path)


if __name__ == "__main__":
    main()
