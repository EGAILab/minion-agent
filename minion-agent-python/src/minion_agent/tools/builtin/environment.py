"""Pi's spawn environment, rebuilt from an execution world's snapshot (`WP-12.E4`, spec/execution.md
section 15.5) -- the consumer composition every tool that reproduces Pi's `process.env` applies
(WP-13.3 `bash` first). `MINION_ARCHITECTURAL_MAPPING` of pinned Pi's observable `process.env`.

Two boundaries, never conflated (WP-12.E4 audit 2):
  1. native entry -> JavaScript string: Node's view (`node_environment_view`), which decodes, can
     group invalid bytes into one U+FFFD, and can DROP an entry;
  2. JavaScript string -> OS: each unpaired surrogate -> U+FFFD -- the identity here, since the
     view is already scalar and injected values are Minion-generated.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping

from ...execution import EnvSnapshot, Platform
from .collation import pinned_collation


def _native_bytes(text: str, platform: Platform) -> bytes:
    """The native form a snapshot `str` stands for (section 15.3): POSIX bytes recovered from
    surrogateescape; WINDOWS UTF-16 code units in generalized UTF-8 (a lone surrogate as its
    3-byte sequence, `ED A0 80` for U+D800)."""
    if platform is Platform.POSIX:
        return text.encode("utf-8", "surrogateescape")
    return text.encode("utf-8", "surrogatepass")


def node_environment_view(snapshot: EnvSnapshot) -> dict[str, str]:
    """Node's `process.env` for that environment: an entry whose name is not valid UTF-8 is
    DROPPED; a value is WHATWG-decoded with replacement (one U+FFFD per maximal invalid subpart;
    a BOM is kept, not stripped). Pinned Node v22.15.1, characterization section 8."""
    view: dict[str, str] = {}
    for name, value in snapshot.entries():
        try:
            decoded_name = _native_bytes(name, snapshot.platform).decode("utf-8")
        except UnicodeDecodeError:
            continue
        view[decoded_name] = _native_bytes(value, snapshot.platform).decode("utf-8", "replace")
    return view


def _utf16_units(name: str) -> bytes:
    return name.encode("utf-16-be", "surrogatepass")


def windows_spawn_environment(env: Mapping[str, str]) -> dict[str, str]:
    """Node v22.15.1's `normalizeSpawnArguments` arbitration for an explicit `env` on Windows
    (section 15.5, `WP12E4-CON-R002`): names whose ECMAScript `toUpperCase` (Unicode 16.0) is equal
    collide, and only the name that sorts FIRST by UTF-16 code units survives, with its value --
    whatever the insertion order. Applied by the consumer before `spawn`; CPython's own spawn
    keeps the LAST-inserted duplicate instead (characterization P2)."""
    upper = pinned_collation().upper_unicode16
    chosen: dict[str, str] = {}
    for name in sorted(env, key=_utf16_units):
        chosen.setdefault(upper(name), name)
    return {name: env[name] for name in sorted(chosen.values(), key=_utf16_units)}


def compose_spawn_environment(
    snapshot: EnvSnapshot, *, remove: Iterable[str] = (), inject: Mapping[str, str] | None = None
) -> dict[str, str]:
    """The complete environment for a spawn with `inherit_env=False` (Owner F1 section 6): Node's
    view of the snapshot, minus each name in `remove` by EXACT spelling (Pi's `delete`; a case
    variant survives -- C003), plus `inject`; on WINDOWS, then Node's duplicate arbitration."""
    env = node_environment_view(snapshot)
    for name in remove:
        env.pop(name, None)
    if inject:
        env.update(inject)
    if snapshot.platform is Platform.WINDOWS:
        return windows_spawn_environment(env)
    return env
