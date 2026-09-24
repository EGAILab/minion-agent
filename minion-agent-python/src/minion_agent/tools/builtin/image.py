"""`read`'s image processing (`TOOL-025`, `R005-A`): pinned Pi `utils/image-process.ts`,
`image-convert.ts`, `image-resize.ts`, `image-resize-core.ts` and `exif-orientation.ts`, on the
pinned Photon WASM.

Every Photon call goes to the pinned artifact; the surrounding arithmetic follows ECMAScript
(`_js`). Each conversion and each resize gets its own Photon instance: the R005-A differential
showed instance lifecycle has no observable effect, and a fresh instance never carries state left
behind by an earlier trap.
"""

from __future__ import annotations

import base64
import contextlib
import math
from collections.abc import Callable
from dataclasses import dataclass

from ._js import math_round, to_fixed
from .photon import LANCZOS3, Photon, PhotonTrap

DEFAULT_MAX_WIDTH = 2000
DEFAULT_MAX_HEIGHT = 2000
DEFAULT_MAX_BYTES = 4.5 * 1024 * 1024
"""The base64 payload ceiling (4.5 MiB)."""
DEFAULT_JPEG_QUALITY = 80

CONVERSION_FAILED = "[Image omitted: could not be converted to a supported inline image format.]"
RESIZE_FAILED = "[Image omitted: could not be resized below the inline image size limit.]"


@dataclass(frozen=True, slots=True)
class ResizeOptions:
    max_width: float = DEFAULT_MAX_WIDTH
    max_height: float = DEFAULT_MAX_HEIGHT
    max_bytes: float = DEFAULT_MAX_BYTES
    jpeg_quality: int = DEFAULT_JPEG_QUALITY


@dataclass(frozen=True, slots=True)
class ResizedImage:
    data: str
    """Base64."""
    mime_type: str
    original_width: int
    original_height: int
    width: int
    height: int
    was_resized: bool


@dataclass(frozen=True, slots=True)
class ProcessedImage:
    """`processImage`'s result: `data`/`mime_type`/`hints` on success, `message` on failure."""

    ok: bool
    data: str = ""
    mime_type: str = ""
    hints: tuple[str, ...] = ()
    message: str = ""


def _b64(data: bytes) -> str:
    return base64.b64encode(data).decode("ascii")


# ---- exif-orientation.ts ------------------------------------------------------------------------
def _get(data: bytes, index: int) -> int | None:
    """`Uint8Array` indexing: an out-of-range read is `undefined`."""
    return data[index] if 0 <= index < len(data) else None


def _bit(data: bytes, index: int) -> int:
    """An element as a bitwise operand: `undefined` becomes 0."""
    value = _get(data, index)
    return 0 if value is None else value


def _int32(value: int) -> int:
    value &= 0xFFFFFFFF
    return value - 0x100000000 if value >= 0x80000000 else value


def _read_orientation_from_tiff(data: bytes, tiff_start: int) -> int:
    if tiff_start + 8 > len(data):
        return 1
    little_endian = ((_bit(data, tiff_start) << 8) | _bit(data, tiff_start + 1)) == 0x4949

    def read16(pos: int) -> int:
        if little_endian:
            return _bit(data, pos) | (_bit(data, pos + 1) << 8)
        return (_bit(data, pos) << 8) | _bit(data, pos + 1)

    def read32(pos: int) -> int:
        if little_endian:
            # `a | b << 8 | c << 16 | d << 24` is a signed 32-bit result in JS.
            return _int32(
                _bit(data, pos)
                | (_bit(data, pos + 1) << 8)
                | (_bit(data, pos + 2) << 16)
                | (_bit(data, pos + 3) << 24)
            )
        # `(a << 24 | ...) >>> 0` is unsigned.
        return (
            (_bit(data, pos) << 24)
            | (_bit(data, pos + 1) << 16)
            | (_bit(data, pos + 2) << 8)
            | _bit(data, pos + 3)
        ) & 0xFFFFFFFF

    ifd_start = tiff_start + read32(tiff_start + 4)
    if ifd_start + 2 > len(data):
        return 1
    for index in range(read16(ifd_start)):
        entry = ifd_start + 2 + index * 12
        if entry + 12 > len(data):
            return 1
        if read16(entry) == 0x0112:
            value = read16(entry + 8)
            return value if 1 <= value <= 8 else 1
    return 1


def _has_exif_header(data: bytes, offset: int) -> bool:
    return [_get(data, offset + k) for k in range(6)] == [0x45, 0x78, 0x69, 0x66, 0x00, 0x00]


def _find_jpeg_tiff_offset(data: bytes) -> int:
    offset = 2
    while offset < len(data) - 1:
        if _get(data, offset) != 0xFF:
            return -1
        marker = _get(data, offset + 1)
        if marker == 0xFF:
            offset += 1
            continue
        if marker == 0xE1:
            if offset + 4 >= len(data):
                return -1
            segment = offset + 4
            if segment + 6 > len(data) or not _has_exif_header(data, segment):
                return -1
            return segment + 6
        if offset + 4 > len(data):
            return -1
        offset += 2 + ((_bit(data, offset + 2) << 8) | _bit(data, offset + 3))
    return -1


def _find_webp_tiff_offset(data: bytes) -> int:
    offset = 12
    while offset + 8 <= len(data):
        chunk_id = bytes(_bit(data, offset + k) for k in range(4))
        size = _int32(
            _bit(data, offset + 4)
            | (_bit(data, offset + 5) << 8)
            | (_bit(data, offset + 6) << 16)
            | (_bit(data, offset + 7) << 24)
        )
        data_start = offset + 8
        if chunk_id == b"EXIF":
            if data_start + size > len(data):
                return -1
            if size >= 6 and _has_exif_header(data, data_start):
                return data_start + 6
            return data_start
        offset = data_start + size + int(math.fmod(size, 2))  # JS `%` keeps the dividend's sign
    return -1


def get_exif_orientation(data: bytes) -> int:
    tiff = -1
    if len(data) >= 2 and data[0] == 0xFF and data[1] == 0xD8:
        tiff = _find_jpeg_tiff_offset(data)
    elif len(data) >= 12 and data[0:4] == b"RIFF" and data[8:12] == b"WEBP":
        tiff = _find_webp_tiff_offset(data)
    return 1 if tiff == -1 else _read_orientation_from_tiff(data, tiff)


type _DstIndex = Callable[[int, int, int, int], int]


def _rotate90(photon: Photon, image: int, dst_index: _DstIndex) -> int:
    width, height = photon.get_width(image), photon.get_height(image)
    src = photon.get_raw_pixels(image)
    dst = bytearray(len(src))
    for y in range(height):
        for x in range(width):
            s = (y * width + x) * 4
            d = dst_index(x, y, width, height) * 4
            dst[d : d + 4] = src[s : s + 4]
    return photon.new(bytes(dst), height, width)


def _clockwise(x: int, y: int, _width: int, height: int) -> int:
    return x * height + (height - 1 - y)


def _counter_clockwise(x: int, y: int, width: int, height: int) -> int:
    return (width - 1 - x) * height + y


def apply_exif_orientation(photon: Photon, image: int, original: bytes) -> int:
    orientation = get_exif_orientation(original)
    if orientation == 2:
        photon.fliph(image)
    elif orientation == 3:
        photon.fliph(image)
        photon.flipv(image)
    elif orientation == 4:
        photon.flipv(image)
    elif orientation in (5, 7):
        rotated = _rotate90(photon, image, _clockwise if orientation == 5 else _counter_clockwise)
        photon.fliph(rotated)
        return rotated
    elif orientation in (6, 8):
        return _rotate90(photon, image, _clockwise if orientation == 6 else _counter_clockwise)
    return image


# ---- image-resize-core.ts -----------------------------------------------------------------------
def resize_image(
    data: bytes, mime_type: str, options: ResizeOptions | None = None
) -> ResizedImage | None:
    """`resizeImageInProcess`: EXIF-orient, return the input unchanged when it already fits,
    else Lanczos3-resize and take the FIRST candidate (PNG, then JPEG at each quality) whose base64
    length is strictly below `max_bytes`, shrinking both sides by 0.75 until 1x1. Any Photon
    failure (a decode rejection, the zero-size-target encode trap) yields `None`."""
    opts = options or ResizeOptions()
    input_base64_size = math.ceil(len(data) / 3) * 4
    photon = Photon()
    image: int | None = None
    try:
        raw = photon.new_from_byteslice(data)
        image = apply_exif_orientation(photon, raw, data)
        if image != raw:
            photon.free(raw)
        original_width, original_height = photon.get_width(image), photon.get_height(image)
        if (
            original_width <= opts.max_width
            and original_height <= opts.max_height
            and input_base64_size < opts.max_bytes
        ):
            return ResizedImage(
                _b64(data),
                mime_type,
                original_width,
                original_height,
                original_width,
                original_height,
                False,
            )
        target_width: float = original_width
        target_height: float = original_height
        if target_width > opts.max_width:
            target_height = math_round(target_height * opts.max_width / target_width)
            target_width = opts.max_width
        if target_height > opts.max_height:
            target_width = math_round(target_width * opts.max_height / target_height)
            target_height = opts.max_height
        qualities = list(dict.fromkeys([opts.jpeg_quality, 85, 70, 55, 40]))
        width, height = int(target_width), int(target_height)
        while True:
            resized = photon.resize(image, width, height, LANCZOS3)
            try:
                candidates = [(_b64(photon.get_bytes(resized)), "image/png")]
                candidates += [
                    (_b64(photon.get_bytes_jpeg(resized, q)), "image/jpeg") for q in qualities
                ]
            finally:
                photon.free(resized)
            for encoded, candidate_mime in candidates:
                if len(encoded) < opts.max_bytes:
                    return ResizedImage(
                        encoded,
                        candidate_mime,
                        original_width,
                        original_height,
                        width,
                        height,
                        True,
                    )
            if width == 1 and height == 1:
                return None
            next_width = 1 if width == 1 else max(1, math.floor(width * 0.75))
            next_height = 1 if height == 1 else max(1, math.floor(height * 0.75))
            if next_width == width and next_height == height:  # pragma: no cover
                # Pi's own guard (image-resize-core.ts), kept verbatim. Unreachable for positive
                # integer sides: floor(0.75 * n) < n for every n >= 2, and 1x1 returned above.
                return None
            width, height = next_width, next_height
    except PhotonTrap:
        return None
    finally:
        if image is not None:
            with contextlib.suppress(PhotonTrap):
                photon.free(image)


def format_dimension_note(result: ResizedImage) -> str | None:
    if not result.was_resized:
        return None
    scale = result.original_width / result.width
    return (
        f"[Image: original {result.original_width}x{result.original_height}, displayed at "
        f"{result.width}x{result.height}. Multiply coordinates by {to_fixed(scale, 2)} to map to "
        "original image.]"
    )


# ---- image-convert.ts / image-process.ts --------------------------------------------------------
def convert_image_bytes_to_png(data: bytes) -> bytes | None:
    photon = Photon()
    try:
        raw = photon.new_from_byteslice(data)
        image = apply_exif_orientation(photon, raw, data)
        if image != raw:
            photon.free(raw)
        try:
            return photon.get_bytes(image)
        finally:
            photon.free(image)
    except PhotonTrap:
        return None


_SUPPORTED_MIME_TYPES = {
    "image/png": "image/png",
    "image/jpeg": "image/jpeg",
    "image/jpg": "image/jpeg",
    "image/gif": "image/gif",
    "image/webp": "image/webp",
}


def _base_mime_type(mime_type: str) -> str:
    return mime_type.split(";")[0].strip().lower()


def process_image(data: bytes, mime_type: str, *, auto_resize: bool = True) -> ProcessedImage:
    """Pi's `processImage`: BMP (any unsupported sniffed type) converts to PNG first; then resize
    (unless disabled); hints are the conversion note then the dimension note."""
    normalized = _SUPPORTED_MIME_TYPES.get(_base_mime_type(mime_type))
    converted_from: str | None = None
    if normalized is not None:
        normalized_bytes = data
    else:
        png = convert_image_bytes_to_png(data)
        if png is None:
            return ProcessedImage(ok=False, message=CONVERSION_FAILED)
        normalized_bytes, normalized, converted_from = png, "image/png", _base_mime_type(mime_type)
    if not auto_resize:
        return ProcessedImage(
            ok=True,
            data=_b64(normalized_bytes),
            mime_type=normalized,
            hints=_conversion_hint(converted_from, normalized),
        )
    resized = resize_image(normalized_bytes, normalized)
    if resized is None:
        return ProcessedImage(ok=False, message=RESIZE_FAILED)
    hints = list(_conversion_hint(converted_from, resized.mime_type))
    note = format_dimension_note(resized)
    if note is not None:
        hints.append(note)
    return ProcessedImage(
        ok=True, data=resized.data, mime_type=resized.mime_type, hints=tuple(hints)
    )


def _conversion_hint(converted_from: str | None, to: str) -> tuple[str, ...]:
    if not converted_from or converted_from == to:
        return ()
    return (f"[Image converted from {converted_from} to {to}.]",)
