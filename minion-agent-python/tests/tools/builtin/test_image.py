"""`read`'s image pipeline below the tool: the Photon resize core against pinned Pi, the integrity
gate, and the Photon host's wasm-bindgen glue."""

import base64
import hashlib
import json
from pathlib import Path
from typing import Any

import pytest
import wasmtime

from minion_agent.tools.builtin import photon as photon_module
from minion_agent.tools.builtin.image import (
    ResizedImage,
    ResizeOptions,
    convert_image_bytes_to_png,
    format_dimension_note,
    process_image,
    resize_image,
)
from minion_agent.tools.builtin.photon import (
    Photon,
    PhotonIntegrityError,
    PhotonTrap,
    check_integrity,
)

FIXTURES = (
    Path(__file__).resolve().parents[4] / "conformance" / "agent" / "fixtures" / "r005a-photon"
)
CORE = json.loads((Path(__file__).parent / "data" / "r005a_core_cases.json").read_text())["cases"]
_OPTION_NAMES = {
    "maxWidth": "max_width",
    "maxHeight": "max_height",
    "maxBytes": "max_bytes",
    "jpegQuality": "jpeg_quality",
}


def _observe(result: ResizedImage | None) -> dict[str, Any] | None:
    if result is None:
        return None
    data = base64.b64decode(result.data)
    return {
        "mime": result.mime_type,
        "originalWidth": result.original_width,
        "originalHeight": result.original_height,
        "width": result.width,
        "height": result.height,
        "wasResized": result.was_resized,
        "data_sha256": hashlib.sha256(data).hexdigest(),
        "data_bytes": len(data),
        "data_base64_len": len(result.data),
    }


@pytest.mark.parametrize("case", CORE, ids=[c["id"] for c in CORE])
def test_resize_core_matches_pinned_pi(case: dict[str, Any]) -> None:
    """Candidate order (first under the limit, not the smallest), strict `<`, custom quality
    ordering, the 0.75 shrink loop and its exhaustion, custom dimensions, EXIF-then-resize, the
    zero-size-target trap and an undecodable input -- each exactly as pinned Pi returned it."""
    options = ResizeOptions(**{_OPTION_NAMES[k]: v for k, v in case["options"].items()})
    result = resize_image((FIXTURES / case["file"]).read_bytes(), case["mime"], options)
    assert _observe(result) == case["expected"]


def test_conversion_applies_exif_orientation() -> None:
    """Pi's convertImageBytesToPng orients too (image-convert.ts); only BMP reaches it through
    processImage, so exercise the rotation branch directly: 7x3 with orientation 6 -> 3x7."""
    png = convert_image_bytes_to_png((FIXTURES / "jpeg_exif_o6.jpg").read_bytes())
    assert png is not None
    photon = Photon()
    image = photon.new_from_byteslice(png)
    assert (photon.get_width(image), photon.get_height(image)) == (3, 7)


def test_conversion_failure_is_none() -> None:
    assert convert_image_bytes_to_png(b"BM" + b"\x00" * 40) is None


def test_process_image_without_auto_resize_keeps_input() -> None:
    data = (FIXTURES / "png_2001x40.png").read_bytes()
    result = process_image(data, "image/png", auto_resize=False)
    assert (result.ok, result.mime_type, result.hints) == (True, "image/png", ())
    assert base64.b64decode(result.data) == data


def test_process_image_normalizes_parameters_and_jpg_alias() -> None:
    data = (FIXTURES / "jpeg_small.jpg").read_bytes()
    result = process_image(data, " IMAGE/JPG ; charset=x", auto_resize=False)
    assert (result.ok, result.mime_type, result.hints) == (True, "image/jpeg", ())


def test_dimension_note_only_when_resized() -> None:
    kept = ResizedImage("", "image/png", 10, 10, 10, 10, False)
    assert format_dimension_note(kept) is None
    resized = ResizedImage("", "image/png", 2250, 100, 2000, 89, True)
    assert format_dimension_note(resized) == (
        "[Image: original 2250x100, displayed at 2000x89. Multiply coordinates by 1.13 to map to "
        "original image.]"
    )


def test_integrity_gate_refuses_other_bytes() -> None:
    wasm = (Path(photon_module.__file__).parent / "photon_rs_bg.wasm").read_bytes()
    check_integrity(wasm)
    tampered = bytearray(wasm)
    tampered[len(tampered) // 2] ^= 1
    with pytest.raises(PhotonIntegrityError, match="!= pinned"):
        check_integrity(bytes(tampered))


def test_photon_trap_surfaces_as_photon_trap() -> None:
    photon = Photon()
    with pytest.raises(PhotonTrap, match="photonimage_new_from_byteslice"):
        photon.new_from_byteslice(b"not an image")


def test_unexpected_import_traps_with_its_own_name() -> None:
    """Imports Pi's image path never reaches are stubs that trap loudly, never silently succeed."""
    photon = Photon()
    func_type = wasmtime.FuncType([], [])
    stub = photon._make_import(_FakeImport("__wbg_not_on_the_image_path", func_type))
    with pytest.raises(wasmtime.Trap, match="UNEXPECTED_IMPORT __wbg_not_on_the_image_path"):
        stub(photon._store)


def test_error_stack_and_console_error_glue() -> None:
    """wasm-bindgen's panic-reporting imports: `stack` writes a (ptr, len) string pair to the
    return slot, and `console.error` frees the string it was handed."""
    photon = Photon()
    retptr, _ = photon._malloc(b"\x00" * 8)
    photon._error_stack(retptr, None)
    slot = photon._read(retptr, 8)
    ptr, length = int.from_bytes(slot[:4], "little"), int.from_bytes(slot[4:], "little")
    assert photon._read(ptr, length) == b"Error\n    at <minion photon host>"
    photon._console_error(ptr, length)


class _FakeImport:
    def __init__(self, name: str, func_type: wasmtime.FuncType) -> None:
        self.name = name
        self.type = func_type
