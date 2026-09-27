"""EXIF orientation and image MIME sniffing against pinned Pi, branch by branch.

Each row's expected values were produced by pinned Pi's own `utils/exif-orientation.ts`
(`getExifOrientation`, exposed by an appended export line only) and `utils/mime.ts`
(`detectSupportedImageMimeType`) on Node 22.15.1 -- not by this implementation.
"""

import pytest

from minion_agent.tools.builtin.image import get_exif_orientation
from minion_agent.tools.builtin.mime import IMAGE_TYPE_SNIFF_BYTES, detect_supported_image_mime_type

PI_CASES = [
    ("jpeg-soi-only", "ffd8", 1, None),
    ("jpeg-3-bytes", "ffd8ff", 1, "image/jpeg"),
    ("jpeg-ls-f7", "ffd8fff70000000000000000", 1, None),
    (
        "jpeg-exif-le-6",
        "ffd8ffe1001e45786966000049492a00080000000100120103000100000006000000",
        6,
        "image/jpeg",
    ),
    (
        "jpeg-exif-be-3",
        "ffd8ffe1001e4578696600004d4d002a000000080001011200030000000100030000",
        3,
        "image/jpeg",
    ),
    (
        "jpeg-fill-bytes-before-app1",
        "ffd8ffffffe1001e45786966000049492a00080000000100120103000100000008000000",
        8,
        "image/jpeg",
    ),
    ("jpeg-non-marker-byte", "ffd800112233", 1, None),
    (
        "jpeg-app1-without-exif-header",
        "ffd8ffe1001e41626364656649492a00080000000100120103000100000006000000",
        1,
        "image/jpeg",
    ),
    ("jpeg-app1-truncated", "ffd8ffe100", 1, "image/jpeg"),
    ("jpeg-app1-short-header", "ffd8ffe10008457869", 1, "image/jpeg"),
    ("jpeg-other-segment-then-eof", "ffd8ffdb00040102", 1, "image/jpeg"),
    ("jpeg-other-segment-truncated-length", "ffd8ffdb00", 1, "image/jpeg"),
    (
        "jpeg-other-segment-then-exif",
        "ffd8ffdb00040102ffe1001e45786966000049492a00080000000100120103000100000005000000",
        5,
        "image/jpeg",
    ),
    ("tiff-too-short", "ffd8ffe1000d45786966000049492a0008", 1, "image/jpeg"),
    ("tiff-ifd-beyond-data", "ffd8ffe1001045786966000049492a00a00f0000", 1, "image/jpeg"),
    (
        "tiff-entries-beyond-data",
        "ffd8ffe1001e45786966000049492a00080000000500000000000000000000000000",
        1,
        "image/jpeg",
    ),
    (
        "tiff-no-orientation-tag",
        "ffd8ffe1001e45786966000049492a00080000000100000103000100000006000000",
        1,
        "image/jpeg",
    ),
    (
        "tiff-orientation-0",
        "ffd8ffe1001e45786966000049492a00080000000100120103000100000000000000",
        1,
        "image/jpeg",
    ),
    (
        "tiff-orientation-9",
        "ffd8ffe1001e45786966000049492a00080000000100120103000100000009000000",
        1,
        "image/jpeg",
    ),
    (
        "tiff-orientation-7",
        "ffd8ffe1001e45786966000049492a00080000000100120103000100000007000000",
        7,
        "image/jpeg",
    ),
    (
        "tiff-le-signed-negative-ifd-offset",
        "ffd8ffe1002845786966000049492a00f8ffffff000000000000000000000000000000000000000000000000",
        1,
        "image/jpeg",
    ),
    (
        "tiff-be-high-bit-ifd-offset",
        "ffd8ffe100284578696600004d4d002afffffff8000000000000000000000000000000000000000000000000",
        1,
        "image/jpeg",
    ),
    (
        "webp-exif-with-header",
        "524946463a00000057454250565038200a00000000000000000000000000455849461c00000045786966000049492a00080000000100120103000100000006000000",
        6,
        "image/webp",
    ),
    (
        "webp-exif-raw-tiff",
        "524946462200000057454250455849461600000049492a00080000000100120103000100000008000000",
        8,
        "image/webp",
    ),
    (
        "webp-odd-chunk-then-exif",
        "524946462e00000057454250494343500300000001020300455849461600000049492a00080000000100120103000100000002000000",
        2,
        "image/webp",
    ),
    (
        "webp-exif-size-beyond-data",
        "524946462200000057454250455849468813000049492a00080000000100120103000100000006000000",
        1,
        "image/webp",
    ),
    (
        "webp-no-exif",
        "524946461600000057454250565038200a00000000000000000000000000",
        1,
        "image/webp",
    ),
    ("webp-tiny-exif-chunk", "524946461000000057454250455849460300000045786900", 1, "image/webp"),
    ("riff-not-webp", "5249464600000000574156450000000000000000", 1, None),
    ("png-signature-only", "89504e470d0a1a0a", 1, None),
    (
        "png-wrong-ihdr-length",
        "89504e470d0a1a0a0000000c4948445200000000000000000000000000000000",
        1,
        None,
    ),
    (
        "png-actl-before-idat",
        "89504e470d0a1a0a0000000d494844520000000000000000000000000000000000000000086163544c00000000000000000000000000000001494441540000000000",
        1,
        None,
    ),
    (
        "png-ihdr-only-walk-runs-out",
        "89504e470d0a1a0a0000000d494844520000000000000000000000000000000000",
        1,
        "image/png",
    ),
    (
        "png-idat-before-actl",
        "89504e470d0a1a0a0000000d49484452000000000000000000000000000000000000000001494441540000000000000000086163544c000000000000000000000000",
        1,
        "image/png",
    ),
    (
        "png-chunk-runs-past-buffer",
        "89504e470d0a1a0a0000000d4948445200000000000000000000000000000000000001869f7445587400000000",
        1,
        "image/png",
    ),
    (
        "png-huge-chunk-length",
        "89504e470d0a1a0a0000000d494844520000000000000000000000000000000000fffffff074455874",
        1,
        "image/png",
    ),
    ("gif", "474946383961000000000000", 1, "image/gif"),
    ("gif-3-bytes", "474946", 1, "image/gif"),
    (
        "bmp-valid-40",
        "424d64000000000000003600000028000000000000000000000001001800000000000000000000000000000000000000000000000000000000000000",
        1,
        "image/bmp",
    ),
    (
        "bmp-valid-core-12",
        "424d64000000000000001a0000000c00000000000000010018000000000000000000000000000000",
        1,
        "image/bmp",
    ),
    (
        "bmp-zero-declared-size",
        "424d00000000000000003600000028000000000000000000000001001800000000000000000000000000000000000000000000000000000000000000",
        1,
        "image/bmp",
    ),
    ("bmp-too-short", "424d0000000000000000000000000000000000000000", 1, None),
    (
        "bmp-declared-size-under-26",
        "424d14000000000000003600000028000000000000000000000001001800000000000000000000000000000000000000000000000000000000000000",
        1,
        None,
    ),
    (
        "bmp-offset-under-header",
        "424d64000000000000001e00000028000000000000000000000001001800000000000000000000000000000000000000000000000000000000000000",
        1,
        None,
    ),
    (
        "bmp-offset-past-declared-size",
        "424d3c000000000000004600000028000000000000000000000001001800000000000000000000000000000000000000000000000000000000000000",
        1,
        None,
    ),
    (
        "bmp-unknown-dib-size",
        "424d64000000000000002800000014000000000000000000000001001800000000000000000000000000000000000000000000000000000000000000",
        1,
        None,
    ),
    ("bmp-40-but-29-bytes", "424d640000000000000036000000280000000000000000000000010018", 1, None),
    (
        "bmp-two-planes",
        "424d64000000000000003600000028000000000000000000000002001800000000000000000000000000000000000000000000000000000000000000",
        1,
        None,
    ),
    (
        "bmp-3-bits",
        "424d64000000000000003600000028000000000000000000000001000300000000000000000000000000000000000000000000000000000000000000",
        1,
        None,
    ),
    (
        "bmp-32-bits",
        "424d64000000000000003600000028000000000000000000000001002000000000000000000000000000000000000000000000000000000000000000",
        1,
        "image/bmp",
    ),
    ("text", "68656c6c6f20776f726c64", 1, None),
    ("empty", "", 1, None),
]


@pytest.mark.parametrize(
    ("name", "hex_bytes", "orientation", "mime"), PI_CASES, ids=[c[0] for c in PI_CASES]
)
def test_matches_pinned_pi(name: str, hex_bytes: str, orientation: int, mime: str | None) -> None:
    data = bytes.fromhex(hex_bytes)
    assert get_exif_orientation(data) == orientation
    assert detect_supported_image_mime_type(data) == mime


def test_sniffing_reads_only_the_first_4100_bytes() -> None:
    """An `acTL` chunk that starts past byte 4100 is never seen: the PNG sniffs as a still image.
    Without the cutoff, the chunk walk would reach it and reject the file as animated."""
    png_head = bytes.fromhex(dict((c[0], c[1]) for c in PI_CASES)["png-idat-before-actl"])[:33]
    text_chunk = (4096).to_bytes(4, "big") + b"tEXt" + bytes(4096) + bytes(4)
    actl_chunk = (8).to_bytes(4, "big") + b"acTL" + bytes(8) + bytes(4)
    data = png_head + text_chunk + actl_chunk
    assert len(png_head + text_chunk) > IMAGE_TYPE_SNIFF_BYTES
    assert detect_supported_image_mime_type(data) == "image/png"
    assert detect_supported_image_mime_type(png_head + (8).to_bytes(4, "big") + b"acTL") is None
