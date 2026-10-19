"""Minimal zero-dependency PNG writer (8-bit greyscale and RGB)."""
from __future__ import annotations

import struct
import zlib


def _chunk(tag: bytes, data: bytes) -> bytes:
    return (struct.pack(">I", len(data)) + tag + data
            + struct.pack(">I", zlib.crc32(tag + data) & 0xFFFFFFFF))


def _encode(path, width: int, height: int, colour_type: int, raw: bytearray) -> bytes:
    png = b"\x89PNG\r\n\x1a\n"
    png += _chunk(b"IHDR", struct.pack(">IIBBBBB", width, height, 8, colour_type, 0, 0, 0))
    png += _chunk(b"IDAT", zlib.compress(bytes(raw), 6))
    png += _chunk(b"IEND", b"")
    return png


def png_gray(width: int, height: int, pixels) -> bytes:
    """8-bit greyscale PNG from a flat sequence of luminance bytes."""
    stride = width
    raw = bytearray()
    for y in range(height):
        raw.append(0)                                   # filter: None
        raw += pixels[y * stride:(y + 1) * stride]
    return _encode(None, width, height, 0, raw)


def png_rgb(width: int, height: int, rgb) -> bytes:
    """8-bit RGB PNG from a flat sequence of r,g,b bytes (length = w*h*3)."""
    stride = width * 3
    raw = bytearray()
    for y in range(height):
        raw.append(0)
        raw += rgb[y * stride:(y + 1) * stride]
    return _encode(None, width, height, 2, raw)


def png_from_palette(width: int, height: int, index_mask, palette):
    """Render an integer mask through a palette -> RGB bytes."""
    rgb = bytearray(width * height * 3)
    for i, v in enumerate(index_mask):
        r, g, b = palette.get(v, (0, 0, 0))
        rgb[i * 3] = r
        rgb[i * 3 + 1] = g
        rgb[i * 3 + 2] = b
    return png_rgb(width, height, rgb)
