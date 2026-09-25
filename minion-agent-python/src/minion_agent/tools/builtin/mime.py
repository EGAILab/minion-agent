"""Image MIME sniffing (pinned Pi `utils/mime.ts`): magic bytes of the first 4100 bytes only, never
the file extension. The closed result set is JPEG, non-animated PNG, GIF, WebP and BMP."""

from __future__ import annotations

IMAGE_TYPE_SNIFF_BYTES = 4100
_PNG_SIGNATURE = b"\x89PNG\r\n\x1a\n"
_BMP_BITS_PER_PIXEL = (1, 4, 8, 16, 24, 32)


def _byte(buf: bytes, offset: int) -> int:
    """`buffer[offset] ?? 0`: an out-of-range read is 0."""
    return buf[offset] if 0 <= offset < len(buf) else 0


def _uint16_le(buf: bytes, offset: int) -> int:
    return _byte(buf, offset) + (_byte(buf, offset + 1) << 8)


def _uint32_be(buf: bytes, offset: int) -> int:
    return (
        _byte(buf, offset) * 0x1000000
        + (_byte(buf, offset + 1) << 16)
        + (_byte(buf, offset + 2) << 8)
        + _byte(buf, offset + 3)
    )


def _uint32_le(buf: bytes, offset: int) -> int:
    return (
        _byte(buf, offset)
        + (_byte(buf, offset + 1) << 8)
        + (_byte(buf, offset + 2) << 16)
        + _byte(buf, offset + 3) * 0x1000000
    )


def _starts_with_ascii(buf: bytes, offset: int, text: bytes) -> bool:
    return len(buf) >= offset + len(text) and buf[offset : offset + len(text)] == text


def _is_png(buf: bytes) -> bool:
    return (
        len(buf) >= 16
        and _uint32_be(buf, len(_PNG_SIGNATURE)) == 13
        and _starts_with_ascii(buf, 12, b"IHDR")
    )


def _is_animated_png(buf: bytes) -> bool:
    offset = len(_PNG_SIGNATURE)
    while offset + 8 <= len(buf):
        chunk_length = _uint32_be(buf, offset)
        if _starts_with_ascii(buf, offset + 4, b"acTL"):
            return True
        if _starts_with_ascii(buf, offset + 4, b"IDAT"):
            return False
        next_offset = offset + 8 + chunk_length + 4
        if next_offset <= offset or next_offset > len(buf):
            return False
        offset = next_offset
    return False


def _is_bmp(buf: bytes) -> bool:
    if len(buf) < 26:
        return False
    declared_file_size = _uint32_le(buf, 2)
    pixel_data_offset = _uint32_le(buf, 10)
    dib_header_size = _uint32_le(buf, 14)
    if declared_file_size != 0 and declared_file_size < 26:
        return False
    if pixel_data_offset < 14 + dib_header_size:
        return False
    if declared_file_size != 0 and pixel_data_offset >= declared_file_size:
        return False
    if dib_header_size == 12:
        color_planes = _uint16_le(buf, 22)
        bits_per_pixel = _uint16_le(buf, 24)
    elif 40 <= dib_header_size <= 124:
        if len(buf) < 30:
            return False
        color_planes = _uint16_le(buf, 26)
        bits_per_pixel = _uint16_le(buf, 28)
    else:
        return False
    return color_planes == 1 and bits_per_pixel in _BMP_BITS_PER_PIXEL


def detect_supported_image_mime_type(data: bytes) -> str | None:
    """Pi's `detectSupportedImageMimeTypeFromFile`: sniff the first `IMAGE_TYPE_SNIFF_BYTES`."""
    buf = data[:IMAGE_TYPE_SNIFF_BYTES]
    if buf[:3] == b"\xff\xd8\xff":
        return None if len(buf) > 3 and buf[3] == 0xF7 else "image/jpeg"
    if buf[: len(_PNG_SIGNATURE)] == _PNG_SIGNATURE:
        return "image/png" if _is_png(buf) and not _is_animated_png(buf) else None
    if _starts_with_ascii(buf, 0, b"GIF"):
        return "image/gif"
    if _starts_with_ascii(buf, 0, b"RIFF") and _starts_with_ascii(buf, 8, b"WEBP"):
        return "image/webp"
    if _starts_with_ascii(buf, 0, b"BM") and _is_bmp(buf):
        return "image/bmp"
    return None
