"""`L12-D011` (`EXEC-013`; spec/execution.md section 21): a U+0000 in a spawn input fails before any
process starts, with pinned Node v22.15.1's literal `ERR_INVALID_ARG_VALUE` diagnostic.

One validator serves both call sites: `LocalSubprocess.spawn` (every input, section 21.1) and the
built-in tools, which check their UNPROJECTED argv just before the WP-13.3 scalar projection
(`WP133-AUD-R001`) so the diagnostic shows the JavaScript-string value (section 21.2). Lengths and
offsets count UTF-16 code units throughout.
"""

from __future__ import annotations

import re
import struct
from collections.abc import Iterable, Sequence

from .environment import Platform
from .errors import SubprocessError, SubprocessErrorCode

_REASON = "must be a string without null bytes"
_CWD_REASON = "must be a string, Uint8Array, or URL without null bytes"
_NUL = "\x00"
_NAMED = {8: "\\b", 9: "\\t", 10: "\\n", 12: "\\f", 13: "\\r"}
_INDEX = re.compile(r"0|[1-9][0-9]*")
_MAX_INDEX = 4294967294
_MAX_STRING_LENGTH = 10000
_SPLIT_THRESHOLD = 76  # breakLength 80 - indentation 0 - 4 (and > kMinLineLength 16)
_CUT = 128


def units(text: str) -> list[int]:
    """The JavaScript string's UTF-16 code units (an astral character is its surrogate pair)."""
    data = text.encode("utf-16-le", "surrogatepass")
    return list(struct.unpack(f"<{len(data) // 2}H", data))


def from_units(code_units: Sequence[int]) -> str:
    """The string of these UTF-16 code units; a lone surrogate stays a lone surrogate."""
    data = struct.pack(f"<{len(code_units)}H", *code_units)
    return data.decode("utf-16-le", "surrogatepass")


def _quote_piece(piece: list[int]) -> list[int]:
    """Node `strEscape` + `addQuotes` for one piece."""
    quote = 39
    if 39 in piece:
        if 34 not in piece:
            quote = 34
        elif 96 not in piece and "${" not in from_units(piece):
            quote = 96
    out: list[int] = [quote]
    i = 0
    while i < len(piece):
        u = piece[i]
        if u == 39 and quote == 39:
            out += units("\\'")
        elif u == 92:
            out += units("\\\\")
        elif u < 32 or 127 <= u <= 159:
            out += units(_NAMED.get(u) or f"\\x{u:02X}")
        elif 0xD800 <= u <= 0xDBFF and i + 1 < len(piece) and 0xDC00 <= piece[i + 1] <= 0xDFFF:
            out += [u, piece[i + 1]]
            i += 1
        elif 0xD800 <= u <= 0xDFFF:
            out += units(f"\\u{u:x}")
        else:
            out.append(u)
        i += 1
    out.append(quote)
    return out


def inspect_string(value: str) -> list[int]:
    """Node `util.inspect` of a string with default options (section 21.2), as UTF-16 units."""
    kept = units(value)
    trailer: list[int] = []
    if len(kept) > _MAX_STRING_LENGTH:
        rest = len(kept) - _MAX_STRING_LENGTH
        kept = kept[:_MAX_STRING_LENGTH]
        trailer = units(f"... {rest} more character{'s' if rest > 1 else ''}")
    if len(kept) <= _SPLIT_THRESHOLD:
        return _quote_piece(kept) + trailer
    pieces: list[list[int]] = []
    start = 0
    for i, u in enumerate(kept):
        if u == 10:
            pieces.append(kept[start : i + 1])
            start = i + 1
    if start < len(kept) or not pieces:
        pieces.append(kept[start:])
    joined: list[int] = []
    for n, piece in enumerate(pieces):
        if n:
            joined += units(" +\n  ")
        joined += _quote_piece(piece)
    return joined + trailer


def diagnostic(name: str, value: str, reason: str = _REASON) -> str:
    """Node's `ERR_INVALID_ARG_VALUE` message for this input (section 21.2)."""
    received = inspect_string(value)
    if len(received) > _CUT:
        received = received[:_CUT] + units("...")
    kind = "property" if "." in name else "argument"
    return f"The {kind} '{name}' {reason}. Received " + from_units(received)


def _rejection(message: str) -> SubprocessError:
    return SubprocessError(SubprocessErrorCode.SPAWN_ERROR, message, nul_input=True)


def argv_rejection(argv: Sequence[str]) -> SubprocessError | None:
    """`file`, then `args[j-1]` for each later element, in order (section 21.1 rows 1-2)."""
    for j, value in enumerate(argv):
        if _NUL in value:
            return _rejection(diagnostic("file" if j == 0 else f"args[{j - 1}]", value))
    return None


def _utf16_key(name: str) -> tuple[int, ...]:
    return tuple(units(name))


def _is_index(name: str) -> bool:
    return _INDEX.fullmatch(name) is not None and int(name) <= _MAX_INDEX


def env_check_order(
    baseline: Iterable[str], overlay: Iterable[str], platform: Platform
) -> list[str]:
    """The order effective entries are checked in (section 21.3). `baseline` holds the inherited
    names (empty without `inherit_env`), `overlay` the call's names in insertion order."""
    owned = list(dict.fromkeys(overlay))
    names = list(dict.fromkeys([*baseline, *owned]))
    if platform == Platform.WINDOWS:
        return sorted(names, key=_utf16_key)
    owned_set = set(owned)
    index = sorted((n for n in names if _is_index(n)), key=int)
    overlay_rest = [n for n in owned if not _is_index(n)]
    baseline_only = sorted(
        (n for n in names if not _is_index(n) and n not in owned_set), key=_utf16_key
    )
    return [*index, *overlay_rest, *baseline_only]


def spawn_rejection(
    argv: Sequence[str],
    cwd: str,
    environment: dict[str, str],
    baseline_names: Iterable[str],
    overlay_names: Iterable[str],
    platform: Platform,
) -> SubprocessError | None:
    """Every input in section 21.1 order: argv, the effective cwd, then the effective environment
    entries (name before value). `environment` holds the effective values."""
    rejected = argv_rejection(argv)
    if rejected is not None:
        return rejected
    if _NUL in cwd:
        return _rejection(diagnostic("options.cwd", cwd, _CWD_REASON))
    for name in env_check_order(baseline_names, overlay_names, platform):
        value = environment[name]
        if _NUL in name or _NUL in value:
            return _rejection(diagnostic(f"options.env['{name}']", name if _NUL in name else value))
    return None


__all__ = [
    "argv_rejection",
    "diagnostic",
    "env_check_order",
    "from_units",
    "inspect_string",
    "spawn_rejection",
    "units",
]
