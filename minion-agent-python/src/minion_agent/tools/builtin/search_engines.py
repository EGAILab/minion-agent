"""The pinned `find`/`grep` match engines (`TOOL-038`; spec/tools.md WP-13.4 "Engines and
provisioning"; `DIV-003`).

`fd 10.4.2` and `ripgrep 15.2.0` are the delegated authority for WHICH paths and lines match. Their
exact identities are fixed below (the same data as `assurance/layers/data/13-wp134/engines.json`);
an engine upgrade is an explicit contract change. Engines are acquired only by the explicit
`provision_search_engines()` -- a tool call never touches the network or `PATH` -- and a tool
verifies the installed binary's SHA-256 before every spawn.
"""

from __future__ import annotations

import asyncio
import contextlib
import hashlib
import io
import os
import platform as host
import sys
import tarfile
import tempfile
import zipfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal

from .paths import BuiltinToolError

type EngineName = Literal["fd", "rg"]


@dataclass(frozen=True, slots=True)
class Artifact:
    url: str
    artifact_sha256: str
    member: str
    binary_sha256: str


@dataclass(frozen=True, slots=True)
class EnginePin:
    label: str
    """Pi's own name for the engine in user-facing text."""
    engine: str
    version: str
    platforms: dict[str, Artifact]


PINS: dict[EngineName, EnginePin] = {
    "fd": EnginePin(
        label="fd",
        engine="fd",
        version="10.4.2",
        platforms={
            "win32-x64": Artifact(
                "https://github.com/sharkdp/fd/releases/download/v10.4.2/fd-v10.4.2-x86_64-pc-windows-msvc.zip",
                "b2816e506390a89941c63c9187d58a3cc10e9a55f2ef0685f9ea0eccaf7c98c8",
                "fd-v10.4.2-x86_64-pc-windows-msvc/fd.exe",
                "4c9d082ee20f0d9e44881ac4e92adf765efc314d82103c53d7f576bd78dc5761",
            ),
            "linux-x64": Artifact(
                "https://github.com/sharkdp/fd/releases/download/v10.4.2/fd-v10.4.2-x86_64-unknown-linux-gnu.tar.gz",
                "def59805cd14b5651b68990855f426ad087f3b96881296d963910431ba3143c8",
                "fd-v10.4.2-x86_64-unknown-linux-gnu/fd",
                "0dff4a420feb3e57fd1d4402d3e29f46115aa38d962467d2f3b72e7439d3ada8",
            ),
        },
    ),
    "rg": EnginePin(
        label="ripgrep (rg)",
        engine="ripgrep",
        version="15.2.0",
        platforms={
            "win32-x64": Artifact(
                "https://github.com/BurntSushi/ripgrep/releases/download/15.2.0/ripgrep-15.2.0-x86_64-pc-windows-msvc.zip",
                "71b2fef860abe467217a538ff31de02f5258807c0129f771846f87bd029aafc5",
                "ripgrep-15.2.0-x86_64-pc-windows-msvc/rg.exe",
                "14231169855ec5205cf5a1b6f1db358ff4aed4247c86b69ce8aae647c77f6680",
            ),
            "linux-x64": Artifact(
                "https://github.com/BurntSushi/ripgrep/releases/download/15.2.0/ripgrep-15.2.0-x86_64-unknown-linux-musl.tar.gz",
                "33e15bcf1624b25cdd2a55813a47a2f95dbe126268203e76aa6a585d1e7b149c",
                "ripgrep-15.2.0-x86_64-unknown-linux-musl/rg",
                "e62198eb19b136b88c330af83647b5a962cb99b6b1f066758568f12de1974849",
            ),
        },
    ),
}


def host_platform() -> str:
    """The local execution world's platform string: `<os>-<arch>` (`win32-x64`, `linux-x64`, and
    for uncertified hosts e.g. `darwin-arm64`)."""
    os_name = {"win32": "win32", "linux": "linux", "darwin": "darwin"}.get(
        sys.platform, sys.platform
    )
    machine = host.machine().lower()
    arch = {"amd64": "x64", "x86_64": "x64", "arm64": "arm64", "aarch64": "arm64"}.get(
        machine, machine
    )
    return f"{os_name}-{arch}"


def default_store_dir() -> Path:
    """The default Minion-managed engine directory."""
    return Path.home() / ".minion" / "search-engines"


def _binary_name(engine: EngineName, platform: str) -> str:
    return f"{engine}.exe" if platform.startswith("win32") else engine


def not_provisioned(engine: EngineName) -> BuiltinToolError:
    pin = PINS[engine]
    return BuiltinToolError(
        f"{pin.label} is not provisioned: the certified {pin.engine} {pin.version} engine "
        "is missing "
        "or failed verification. Run provision_search_engines() to provision it."
    )


def not_available(engine: EngineName, platform: str) -> BuiltinToolError:
    pin = PINS[engine]
    return BuiltinToolError(
        f"{pin.label} is not available on this platform: no certified {pin.engine} engine for "
        f"{platform}."
    )


def _sha256_file(path: Path) -> str | None:
    try:
        return hashlib.sha256(path.read_bytes()).hexdigest()
    except OSError:
        return None


@dataclass(frozen=True, slots=True)
class EngineStore:
    """The certified engine store (TOOL-038): at most one installed binary per engine, under a
    fixed name, verified before every use."""

    root: Path = field(default_factory=default_store_dir)
    platform: str = field(default_factory=host_platform)

    def binary_path(self, engine: EngineName) -> Path:
        return self.root / _binary_name(engine, self.platform)

    def is_verified(self, engine: EngineName) -> bool:
        artifact = PINS[engine].platforms.get(self.platform)
        if artifact is None:
            return False
        return _sha256_file(self.binary_path(engine)) == artifact.binary_sha256

    async def resolve(self, engine: EngineName) -> list[str]:
        """The verified binary as an argv prefix, or the `DIV-003` unavailable-engine error. Never
        consults `PATH`, never downloads."""
        if self.platform not in PINS[engine].platforms:
            raise not_available(engine, self.platform)
        if not await asyncio.to_thread(self.is_verified, engine):
            raise not_provisioned(engine)
        return [str(self.binary_path(engine))]


@dataclass(frozen=True, slots=True)
class EngineOverride:
    """An explicit, UNCERTIFIED engine selection (a Minion extension): each engine is an argv
    prefix used as given (an executable, or an interpreter plus a script) -- not hash-verified, and
    never inheriting `TOOL-038` certification."""

    commands: dict[EngineName, list[str]]

    async def resolve(self, engine: EngineName) -> list[str]:
        command = self.commands.get(engine)
        if not command:
            raise BuiltinToolError(
                f"{PINS[engine].label} is not configured in the engine override."
            )
        return list(command)


type Engines = EngineStore | EngineOverride


class ProvisioningError(Exception):
    """An engine could not be provisioned; nothing was installed for it."""


def _fetch(url: str) -> bytes:  # pragma: no cover - network; exercised by the CI provisioning step
    import httpx

    response = httpx.get(url, follow_redirects=True, timeout=120.0)
    response.raise_for_status()
    return response.content


def _extract(archive: bytes, name: str, member: str) -> bytes:
    if name.endswith(".zip"):
        with zipfile.ZipFile(io.BytesIO(archive)) as zipped:
            return zipped.read(member)
    with tarfile.open(fileobj=io.BytesIO(archive), mode="r:gz") as tarred:
        extracted = tarred.extractfile(member)
        if extracted is None:
            raise ProvisioningError(f"archive member {member} is not a file")
        return extracted.read()


def _install(store: EngineStore, engine: EngineName, binary: bytes) -> None:
    """Atomic: a temporary file in the store, then a rename to the fixed name. An interruption
    leaves at most a temporary file, which nothing resolves."""
    store.root.mkdir(parents=True, exist_ok=True)
    handle, temporary = tempfile.mkstemp(dir=store.root, prefix=f".{engine}-", suffix=".partial")
    try:
        with os.fdopen(handle, "wb") as out:
            out.write(binary)
        if not store.platform.startswith("win32"):
            os.chmod(temporary, 0o755)
        os.replace(temporary, store.binary_path(engine))
    except BaseException:
        with contextlib.suppress(OSError):
            os.unlink(temporary)
        raise


def provision_search_engines(
    store: EngineStore | None = None, *, source: str | Path | None = None
) -> dict[EngineName, str]:
    """Install the pinned engines for the store's (certified) platform. `source` is a local
    directory holding the official artifacts under their official file names; when absent, the
    official upstream URL is used. The source is not authority -- the artifact's SHA-256 is
    verified before extraction and the binary's after. Safe to repeat: an engine whose installed
    binary already verifies is left as it is. Returns, per engine, `"installed"` or `"present"`."""
    store = store or EngineStore()
    outcome: dict[EngineName, str] = {}
    for engine, pin in PINS.items():
        artifact = pin.platforms.get(store.platform)
        if artifact is None:
            raise ProvisioningError(str(not_available(engine, store.platform)))
        if store.is_verified(engine):
            outcome[engine] = "present"
            continue
        name = artifact.url.rsplit("/", 1)[-1]
        data = Path(source, name).read_bytes() if source is not None else _fetch(artifact.url)
        if hashlib.sha256(data).hexdigest() != artifact.artifact_sha256:
            raise ProvisioningError(f"{name}: artifact SHA-256 mismatch; nothing installed")
        binary = _extract(data, name, artifact.member)
        if hashlib.sha256(binary).hexdigest() != artifact.binary_sha256:
            raise ProvisioningError(
                f"{artifact.member}: binary SHA-256 mismatch; nothing installed"
            )
        _install(store, engine, binary)
        outcome[engine] = "installed"
    return outcome
