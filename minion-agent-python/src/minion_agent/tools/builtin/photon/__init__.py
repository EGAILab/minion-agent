"""Host for the pinned `@silvia-odwyer/photon-node` 0.3.4 `photon_rs_bg.wasm` (`R005-A`).

Executes the SAME WASM bytes pinned Pi runs under Node, through wasmtime, with glue that mirrors
photon-node's own `photon_rs.js` call for call -- the binding the R005-A differential proved
byte-identical to pinned Pi (minion-agent-docs `r005-a-photon-differential/`). Only the imports
Pi's image path can reach are implemented; any other import traps with its own name, so an
unexpected path fails instead of silently diverging.
"""

from __future__ import annotations

import hashlib
from collections.abc import Callable
from functools import cache
from importlib.resources import files
from typing import Any

import wasmtime

PINNED_WASM_SHA256 = "10468181565c56004c867f3a4af96f89a0ef5a63a72f2b5fb12c1f1992a3615c"
LANCZOS3 = 5
"""photon_rs.js `SamplingFilter.Lanczos3`."""


class PhotonTrap(Exception):
    """A trap or thrown error inside Photon -- what Pi's own `catch` blocks see."""


class PhotonIntegrityError(RuntimeError):
    """The vendored WASM is not the pinned artifact. Never falls back to anything else."""


class _JsValue:
    """An opaque JS value held in wasm-bindgen's externref table."""

    __slots__ = ("label",)

    def __init__(self, label: str) -> None:
        self.label = label


def check_integrity(wasm: bytes) -> None:
    digest = hashlib.sha256(wasm).hexdigest()
    if digest != PINNED_WASM_SHA256:
        raise PhotonIntegrityError(
            f"photon_rs_bg.wasm sha256 {digest} != pinned {PINNED_WASM_SHA256}"
        )


@cache
def _compiled() -> tuple[wasmtime.Engine, wasmtime.Module]:
    wasm = files(__package__).joinpath("photon_rs_bg.wasm").read_bytes()
    check_integrity(wasm)
    engine = wasmtime.Engine()
    return engine, wasmtime.Module(engine, wasm)


class Photon:
    """One photon-node instance (a fresh store and instance, like one `require()`)."""

    def __init__(self) -> None:
        self._engine, self._module = _compiled()
        self._store = wasmtime.Store(self._engine)
        imports = [self._make_import(imp) for imp in self._module.imports]
        instance = wasmtime.Instance(self._store, self._module, imports)
        self._exports = instance.exports(self._store)
        memory = self._exports["memory"]
        assert isinstance(memory, wasmtime.Memory)
        self._memory = memory
        self._func("__wbindgen_start")()

    # -- wasm-bindgen glue ------------------------------------------------------------------------
    def _func(self, name: str) -> Any:
        func = self._exports[name]
        assert isinstance(func, wasmtime.Func)
        return lambda *args: func(self._store, *args)

    def _read(self, ptr: int, length: int) -> bytes:
        return bytes(self._memory.read(self._store, ptr, ptr + length))

    def _write(self, ptr: int, data: bytes) -> None:
        self._memory.write(self._store, data, ptr)

    def _malloc(self, data: bytes) -> tuple[int, int]:
        ptr = int(self._func("__wbindgen_malloc")(len(data), 1)) & 0xFFFFFFFF
        self._write(ptr, data)
        return ptr, len(data)

    def _take(self, pair: Any) -> bytes:
        ptr, length = int(pair[0]) & 0xFFFFFFFF, int(pair[1])
        data = self._read(ptr, length)
        self._func("__wbindgen_free")(ptr, length, 1)
        return data

    def _make_import(self, imp: wasmtime.ImportType) -> wasmtime.Func:
        name = imp.name or ""
        ty = imp.type
        assert isinstance(ty, wasmtime.FuncType)
        implementations: dict[str, Callable[..., Any]] = {
            "__wbindgen_init_externref_table": self._init_externref_table,
            "__wbindgen_throw": self._throw,
            "__wbg_new_abda76e883ba8a5f": lambda: _JsValue("Error"),
            "__wbg_stack_658279fe44541cf6": self._error_stack,
            "__wbg_error_f851667af71bcfc6": self._console_error,
            "__wbindgen_memory": lambda: _JsValue("memory"),
        }

        def unexpected(*_args: Any) -> None:
            raise wasmtime.Trap(f"UNEXPECTED_IMPORT {name}")

        return wasmtime.Func(self._store, ty, implementations.get(name, unexpected))

    def _init_externref_table(self) -> None:
        table = self._exports["__wbindgen_export_2"]
        assert isinstance(table, wasmtime.Table)
        offset = table.grow(self._store, 4, None)
        table.set(self._store, 0, _JsValue("undefined"))
        for index, label in enumerate(("undefined", "null", "true", "false")):
            table.set(self._store, offset + index, _JsValue(label))

    def _throw(self, ptr: int, length: int) -> None:
        raise wasmtime.Trap(self._read(ptr & 0xFFFFFFFF, length).decode("utf-8", "replace"))

    def _error_stack(self, retptr: int, _error: Any) -> None:
        # JS writes `new Error().stack`; its text only ever reaches console.error.
        ptr, length = self._malloc(b"Error\n    at <minion photon host>")
        self._write((retptr & 0xFFFFFFFF) + 4, length.to_bytes(4, "little"))
        self._write(retptr & 0xFFFFFFFF, ptr.to_bytes(4, "little"))

    def _console_error(self, ptr: int, length: int) -> None:
        self._func("__wbindgen_free")(ptr & 0xFFFFFFFF, length, 1)

    def _call(self, name: str, *args: Any) -> Any:
        try:
            return self._func(name)(*args)
        except (wasmtime.Trap, wasmtime.WasmtimeError) as exc:
            raise PhotonTrap(f"{name}: {exc}") from exc

    # -- the photon_rs.js surface Pi's image path uses --------------------------------------------
    def new_from_byteslice(self, data: bytes) -> int:
        ptr, length = self._malloc(data)
        return int(self._call("photonimage_new_from_byteslice", ptr, length)) & 0xFFFFFFFF

    def new(self, raw_pixels: bytes, width: int, height: int) -> int:
        ptr, length = self._malloc(raw_pixels)
        return int(self._call("photonimage_new", ptr, length, width, height)) & 0xFFFFFFFF

    def get_width(self, image: int) -> int:
        return int(self._call("photonimage_get_width", image)) & 0xFFFFFFFF

    def get_height(self, image: int) -> int:
        return int(self._call("photonimage_get_height", image)) & 0xFFFFFFFF

    def get_raw_pixels(self, image: int) -> bytes:
        return self._take(self._call("photonimage_get_raw_pixels", image))

    def get_bytes(self, image: int) -> bytes:
        return self._take(self._call("photonimage_get_bytes", image))

    def get_bytes_jpeg(self, image: int, quality: int) -> bytes:
        return self._take(self._call("photonimage_get_bytes_jpeg", image, quality))

    def resize(self, image: int, width: int, height: int, sampling_filter: int) -> int:
        return int(self._call("resize", image, width, height, sampling_filter)) & 0xFFFFFFFF

    def fliph(self, image: int) -> None:
        self._call("fliph", image)

    def flipv(self, image: int) -> None:
        self._call("flipv", image)

    def free(self, image: int) -> None:
        self._call("__wbg_photonimage_free", image, 0)
