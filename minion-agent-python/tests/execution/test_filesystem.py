"""`EXEC-002`/`EXEC-003`: `ctx.fs` and the `FsTarget` bridge, spec/execution.md sections 3-4.
Exercises every executable witness in spec section 10 that these two rows cover."""

from __future__ import annotations

import asyncio
import os
import time
from pathlib import Path

import pytest

from minion_agent.execution import filesystem as filesystem_module
from minion_agent.execution.errors import FsError, FsErrorCode
from minion_agent.execution.filesystem import (
    FileKind,
    LocalFileSystem,
    _file_info_sync,
    _file_kind_from_stat,
    _UnsupportedFileType,
    resolve_local_path,
)
from minion_agent.execution.result import Err, Ok
from minion_agent.runtime.signal import RunAbortController

# ---------------------------------------------------------------------------
# resolve_local_path (shared lexical resolution)
# ---------------------------------------------------------------------------


def test_relative_path_resolves_against_cwd(tmp_path: Path) -> None:
    assert resolve_local_path(str(tmp_path), "a.txt") == str(tmp_path / "a.txt")


def test_absolute_path_used_as_is(tmp_path: Path) -> None:
    absolute = str(tmp_path / "a.txt")
    assert resolve_local_path("/somewhere/else", absolute) == os.path.normpath(absolute)


def test_lexical_normalization_removes_dot_dot(tmp_path: Path) -> None:
    resolved = resolve_local_path(str(tmp_path), "sub/../a.txt")
    assert resolved == str(tmp_path / "a.txt")


def test_bare_tilde_resolves_to_home() -> None:
    assert resolve_local_path("/cwd", "~") == os.path.normpath(os.path.expanduser("~"))


def test_tilde_slash_resolves_against_home() -> None:
    resolved = resolve_local_path("/cwd", "~/x.txt")
    assert resolved == os.path.normpath(os.path.join(os.path.expanduser("~"), "x.txt"))


def test_file_url_is_parsed(tmp_path: Path) -> None:
    target = tmp_path / "a.txt"
    resolved = resolve_local_path("/cwd", target.as_uri())
    assert Path(resolved) == target


def test_malformed_file_url_is_kept_as_literal_string() -> None:
    """Never raises -- preserves the never-throw contract even at the lexical stage."""
    resolved = resolve_local_path("/cwd", "file://%zz-not-a-valid-escape")
    assert isinstance(resolved, str)


@pytest.mark.skipif(
    os.name == "nt", reason="a non-empty file:// host is a legitimate UNC path on Windows"
)
def test_file_url_with_non_local_host_is_kept_as_literal_string_on_posix() -> None:
    """`L12-PY-R002` witness: pinned Node's `fileURLToPath` throws `ERR_INVALID_FILE_URL_HOST`
    for a non-empty, non-"localhost" host on POSIX -- `file://nonlocalhost/some/path` is
    malformed there, not a legitimate path. This seam never raises, so it must instead preserve
    the literal input string unchanged, exactly like the other malformed-URL witness above --
    NOT silently drop the host and produce a wrong resolved path."""
    literal = "file://nonlocalhost/some/path"
    resolved = resolve_local_path("/cwd", literal)
    assert resolved == literal


def test_file_url_with_non_local_host_is_kept_as_literal_string_portable(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Portable twin of the witness above, closing coverage on Windows (where `os.name` is
    genuinely `"nt"` and the real POSIX-only branch never executes on its own): monkeypatches
    `os.name` just for this one check, matching this module's `_UnsupportedFileType` /
    monkeypatch-`os.lstat` coverage-closing convention used further below."""
    monkeypatch.setattr(os, "name", "posix")
    literal = "file://nonlocalhost/some/path"
    resolved = resolve_local_path("/cwd", literal)
    assert resolved == literal


# ---------------------------------------------------------------------------
# join_path (`L12-PY-R002`)
# ---------------------------------------------------------------------------


async def test_join_path_empty_parts_returns_dot(tmp_path: Path) -> None:
    """Matches Node's own `path.join()` with zero arguments, which returns `"."`, not `""`."""
    fs = LocalFileSystem(cwd=str(tmp_path))
    assert await fs.join_path([]) == Ok(".")


@pytest.mark.skipif(os.name != "nt", reason="exercises the Windows-only join-reset bug")
async def test_join_path_windows_later_separator_prefixed_segment_does_not_reset(
    tmp_path: Path,
) -> None:
    """`L12-PY-R002` witness: `os.path.join("a", "\\\\b")` resets accumulation and returns just
    `"\\\\b"`, because a leading separator makes a segment look absolute to `os.path.join` --
    Node's own `path.join` has no such special case, it simply concatenates every segment and
    THEN normalizes, giving `"a\\\\b"`. This seam must match Node, not `os.path.join`'s own
    behavior."""
    fs = LocalFileSystem(cwd=str(tmp_path))
    assert await fs.join_path(["a", "\\b"]) == Ok("a\\b")


# ---------------------------------------------------------------------------
# Reads, writes, append, rename
# ---------------------------------------------------------------------------


async def test_write_then_read_text_file(tmp_path: Path) -> None:
    fs = LocalFileSystem(cwd=str(tmp_path))
    assert await fs.write_file("a.txt", "hello") == Ok(None)
    assert await fs.read_text_file("a.txt") == Ok("hello")


async def test_write_creates_missing_parent_directories(tmp_path: Path) -> None:
    fs = LocalFileSystem(cwd=str(tmp_path))
    result = await fs.write_file("a/b/c.txt", "x")
    assert result == Ok(None)
    assert (tmp_path / "a" / "b" / "c.txt").read_text() == "x"


async def test_write_binary_content(tmp_path: Path) -> None:
    fs = LocalFileSystem(cwd=str(tmp_path))
    await fs.write_file("bin.dat", b"\x00\x01\x02")
    assert await fs.read_binary_file("bin.dat") == Ok(b"\x00\x01\x02")


async def test_read_text_file_not_found(tmp_path: Path) -> None:
    fs = LocalFileSystem(cwd=str(tmp_path))
    result = await fs.read_text_file("missing.txt")
    assert isinstance(result, Err)
    assert result.error.code == FsErrorCode.NOT_FOUND


async def test_read_text_file_with_invalid_utf8_replaces_instead_of_raising(
    tmp_path: Path,
) -> None:
    """`L12-PY-R002` witness: matches pinned Node's own UTF-8 decoding, which never throws for
    invalid bytes -- it substitutes the replacement character (U+FFFD). The default
    `errors="strict"` would raise `UnicodeDecodeError`, which is NOT an `OSError` subclass and
    would therefore escape this operation's own `except OSError` handling entirely, violating
    the never-raise `Result` contract outright."""
    fs = LocalFileSystem(cwd=str(tmp_path))
    await fs.write_file("bad.txt", b"before \xff\xfe after")
    result = await fs.read_text_file("bad.txt")
    assert isinstance(result, Ok)
    assert "�" in result.value
    assert result.value.startswith("before ")
    assert result.value.endswith(" after")


async def test_append_file_creates_and_appends(tmp_path: Path) -> None:
    fs = LocalFileSystem(cwd=str(tmp_path))
    await fs.append_file("a.txt", "one")
    await fs.append_file("a.txt", "two")
    assert await fs.read_text_file("a.txt") == Ok("onetwo")


async def test_append_file_creates_missing_parents(tmp_path: Path) -> None:
    fs = LocalFileSystem(cwd=str(tmp_path))
    result = await fs.append_file("a/b.txt", "x")
    assert result == Ok(None)


async def test_append_binary_content(tmp_path: Path) -> None:
    fs = LocalFileSystem(cwd=str(tmp_path))
    await fs.write_file("bin.dat", b"\x01")
    await fs.append_file("bin.dat", b"\x02")
    assert await fs.read_binary_file("bin.dat") == Ok(b"\x01\x02")


async def test_rename_file_is_atomic_and_replaces_destination(tmp_path: Path) -> None:
    fs = LocalFileSystem(cwd=str(tmp_path))
    await fs.write_file("a.txt", "A")
    await fs.write_file("b.txt", "B")
    result = await fs.rename_file("a.txt", "b.txt")
    assert result == Ok(None)
    assert await fs.read_text_file("b.txt") == Ok("A")
    assert not (tmp_path / "a.txt").exists()


async def test_rename_file_source_missing(tmp_path: Path) -> None:
    fs = LocalFileSystem(cwd=str(tmp_path))
    result = await fs.rename_file("missing.txt", "dest.txt")
    assert isinstance(result, Err)
    assert result.error.code == FsErrorCode.NOT_FOUND


# ---------------------------------------------------------------------------
# read_text_lines
# ---------------------------------------------------------------------------


async def test_read_text_lines_all(tmp_path: Path) -> None:
    fs = LocalFileSystem(cwd=str(tmp_path))
    await fs.write_file("a.txt", "one\ntwo\nthree\n")
    assert await fs.read_text_lines("a.txt") == Ok(["one", "two", "three"])


async def test_read_text_lines_max_lines_stops_early(tmp_path: Path) -> None:
    fs = LocalFileSystem(cwd=str(tmp_path))
    await fs.write_file("a.txt", "one\ntwo\nthree\n")
    assert await fs.read_text_lines("a.txt", max_lines=2) == Ok(["one", "two"])


async def test_read_text_lines_max_lines_zero_returns_empty_without_touching_file() -> None:
    fs = LocalFileSystem(cwd="/does/not/matter")
    result = await fs.read_text_lines("nonexistent-but-irrelevant.txt", max_lines=0)
    assert result == Ok([])


async def test_read_text_lines_negative_max_lines_returns_empty() -> None:
    fs = LocalFileSystem(cwd="/does/not/matter")
    assert await fs.read_text_lines("x.txt", max_lines=-1) == Ok([])


async def test_read_text_lines_not_found(tmp_path: Path) -> None:
    fs = LocalFileSystem(cwd=str(tmp_path))
    result = await fs.read_text_lines("missing.txt")
    assert isinstance(result, Err)
    assert result.error.code == FsErrorCode.NOT_FOUND


# ---------------------------------------------------------------------------
# file_info / list_dir / exists / canonical_path
# ---------------------------------------------------------------------------


async def test_file_info_reports_file_kind(tmp_path: Path) -> None:
    fs = LocalFileSystem(cwd=str(tmp_path))
    await fs.write_file("a.txt", "x")
    result = await fs.file_info("a.txt")
    assert isinstance(result, Ok)
    assert result.value.kind == FileKind.FILE
    assert result.value.name == "a.txt"
    assert result.value.size == 1


async def test_file_info_reports_directory_kind(tmp_path: Path) -> None:
    fs = LocalFileSystem(cwd=str(tmp_path))
    await fs.create_dir("sub")
    result = await fs.file_info("sub")
    assert isinstance(result, Ok)
    assert result.value.kind == FileKind.DIRECTORY


async def test_file_info_not_found(tmp_path: Path) -> None:
    fs = LocalFileSystem(cwd=str(tmp_path))
    result = await fs.file_info("missing.txt")
    assert isinstance(result, Err)
    assert result.error.code == FsErrorCode.NOT_FOUND


async def test_list_dir_returns_direct_children_only(tmp_path: Path) -> None:
    fs = LocalFileSystem(cwd=str(tmp_path))
    await fs.write_file("a.txt", "x")
    await fs.create_dir("sub")
    await fs.write_file("sub/nested.txt", "y")
    result = await fs.list_dir(".")
    assert isinstance(result, Ok)
    names = {info.name for info in result.value}
    assert names == {"a.txt", "sub"}


async def test_list_dir_not_found(tmp_path: Path) -> None:
    fs = LocalFileSystem(cwd=str(tmp_path))
    result = await fs.list_dir("missing")
    assert isinstance(result, Err)
    assert result.error.code == FsErrorCode.NOT_FOUND


async def test_exists_true_for_present_path(tmp_path: Path) -> None:
    fs = LocalFileSystem(cwd=str(tmp_path))
    await fs.write_file("a.txt", "x")
    assert await fs.exists("a.txt") == Ok(True)


async def test_exists_false_for_missing_path(tmp_path: Path) -> None:
    fs = LocalFileSystem(cwd=str(tmp_path))
    assert await fs.exists("missing.txt") == Ok(False)


async def test_canonical_path_resolves_symlinks(tmp_path: Path) -> None:
    fs = LocalFileSystem(cwd=str(tmp_path))
    await fs.write_file("target.txt", "X")
    os.symlink(tmp_path / "target.txt", tmp_path / "link.txt")
    result = await fs.canonical_path("link.txt")
    assert isinstance(result, Ok)
    assert os.path.samefile(result.value, tmp_path / "target.txt")


async def test_canonical_path_missing_is_not_found(tmp_path: Path) -> None:
    fs = LocalFileSystem(cwd=str(tmp_path))
    result = await fs.canonical_path("missing.txt")
    assert isinstance(result, Err)
    assert result.error.code == FsErrorCode.NOT_FOUND


# ---------------------------------------------------------------------------
# create_dir / remove
# ---------------------------------------------------------------------------


async def test_create_dir_recursive_default(tmp_path: Path) -> None:
    fs = LocalFileSystem(cwd=str(tmp_path))
    result = await fs.create_dir("a/b/c")
    assert result == Ok(None)
    assert (tmp_path / "a" / "b" / "c").is_dir()


async def test_create_dir_recursive_is_idempotent(tmp_path: Path) -> None:
    fs = LocalFileSystem(cwd=str(tmp_path))
    await fs.create_dir("a")
    assert await fs.create_dir("a") == Ok(None)


async def test_create_dir_non_recursive_fails_on_missing_parent(tmp_path: Path) -> None:
    fs = LocalFileSystem(cwd=str(tmp_path))
    result = await fs.create_dir("a/b", recursive=False)
    assert isinstance(result, Err)


async def test_remove_plain_file(tmp_path: Path) -> None:
    fs = LocalFileSystem(cwd=str(tmp_path))
    await fs.write_file("a.txt", "x")
    assert await fs.remove("a.txt") == Ok(None)
    assert not (tmp_path / "a.txt").exists()


async def test_remove_missing_without_force_propagates_not_found(tmp_path: Path) -> None:
    fs = LocalFileSystem(cwd=str(tmp_path))
    result = await fs.remove("missing.txt")
    assert isinstance(result, Err)
    assert result.error.code == FsErrorCode.NOT_FOUND


async def test_remove_missing_with_force_succeeds(tmp_path: Path) -> None:
    fs = LocalFileSystem(cwd=str(tmp_path))
    assert await fs.remove("missing.txt", force=True) == Ok(None)


async def test_remove_directory_without_recursive_fails(tmp_path: Path) -> None:
    """Matches pinned Pi's own `fs.rm` exactly: ANY directory requires `recursive=True`, even
    an empty one -- narrower than POSIX `rmdir`'s own leniency."""
    fs = LocalFileSystem(cwd=str(tmp_path))
    await fs.create_dir("empty")
    result = await fs.remove("empty")
    assert isinstance(result, Err)
    assert result.error.code == FsErrorCode.IS_DIRECTORY


async def test_remove_directory_recursive_removes_contents(tmp_path: Path) -> None:
    fs = LocalFileSystem(cwd=str(tmp_path))
    await fs.write_file("sub/a.txt", "x")
    result = await fs.remove("sub", recursive=True)
    assert result == Ok(None)
    assert not (tmp_path / "sub").exists()


async def test_remove_symlink_never_follows(tmp_path: Path) -> None:
    """`SYMLINK FOLLOWING` matrix (spec section 3.2): `remove` removes the addressed symlink's
    own directory entry, never recursing into the target."""
    fs = LocalFileSystem(cwd=str(tmp_path))
    (tmp_path / "real").mkdir()
    (tmp_path / "real" / "kept.txt").write_text("kept")
    os.symlink(tmp_path / "real", tmp_path / "link", target_is_directory=True)
    result = await fs.remove("link", recursive=True)
    assert result == Ok(None)
    assert not (tmp_path / "link").exists()
    assert (tmp_path / "real" / "kept.txt").exists()  # target's own contents untouched


async def test_create_temp_dir(tmp_path: Path) -> None:
    fs = LocalFileSystem(cwd=str(tmp_path))
    result = await fs.create_temp_dir(prefix="tmp-")
    assert isinstance(result, Ok)
    assert os.path.isdir(result.value)
    assert os.path.basename(result.value).startswith("tmp-")


async def test_create_temp_file_gets_its_own_private_directory() -> None:
    """`DIRECT_PI_PARITY` creation shape: every temp FILE also gets its own private temp
    DIRECTORY, not merely a unique filename in a shared temp directory."""
    fs = LocalFileSystem()
    result = await fs.create_temp_file(prefix="pre-", suffix=".txt")
    assert isinstance(result, Ok)
    file_path = result.value
    assert os.path.basename(file_path).startswith("pre-")
    assert file_path.endswith(".txt")
    assert os.path.isfile(file_path)
    parent = os.path.dirname(file_path)
    assert set(os.listdir(parent)) == {os.path.basename(file_path)}


async def test_cleanup_is_a_true_no_op() -> None:
    """`L12-R017`: `ctx.fs.cleanup()` makes no child-process claim at all."""
    fs = LocalFileSystem()
    await fs.cleanup()  # must not raise; -> None is the no-op contract itself


# ---------------------------------------------------------------------------
# Symlink semantics witnesses (spec section 3.2)
# ---------------------------------------------------------------------------


async def test_symlink_content_io_follows(tmp_path: Path) -> None:
    """`SYMLINK FOLLOWING, CONTENT I/O` witness."""
    fs = LocalFileSystem(cwd=str(tmp_path))
    await fs.write_file("target.txt", "X")
    os.symlink(tmp_path / "target.txt", tmp_path / "link.txt")
    assert await fs.read_text_file("link.txt") == Ok("X")
    info = await fs.file_info("link.txt")
    assert isinstance(info, Ok)
    assert info.value.kind == FileKind.SYMLINK


async def test_symlink_rename_never_follows(tmp_path: Path) -> None:
    """`SYMLINK FOLLOWING, RENAME` witness: renaming a symlink moves the LINK, leaving the
    target untouched, and the renamed path is itself still a symlink."""
    fs = LocalFileSystem(cwd=str(tmp_path))
    await fs.write_file("target.txt", "X")
    os.symlink(tmp_path / "target.txt", tmp_path / "link.txt")
    result = await fs.rename_file("link.txt", "moved.txt")
    assert result == Ok(None)
    assert await fs.read_text_file("target.txt") == Ok("X")
    moved_info = await fs.file_info("moved.txt")
    assert isinstance(moved_info, Ok)
    assert moved_info.value.kind == FileKind.SYMLINK
    assert await fs.read_text_file("moved.txt") == Ok("X")


# ---------------------------------------------------------------------------
# Cancellation checkpoint table (spec section 3.1, L12-R009)
# ---------------------------------------------------------------------------


async def test_read_text_file_succeeds_with_a_never_aborted_signal(tmp_path: Path) -> None:
    """Closes `_race_signal`'s OTHER branch (the operation winning the race, `op_task.done()`):
    every existing signal-bearing witness for these three ops either pre-aborts or aborts
    mid-block, never exercising the ordinary case of a live-but-never-fired signal alongside a
    normal (fast, non-blocking) operation."""
    fs = LocalFileSystem(cwd=str(tmp_path))
    await fs.write_file("a.txt", "hello")
    controller = RunAbortController()
    result = await fs.read_text_file("a.txt", signal=controller.signal)
    assert result == Ok("hello")


async def test_read_text_file_pre_aborted(tmp_path: Path) -> None:
    fs = LocalFileSystem(cwd=str(tmp_path))
    await fs.write_file("a.txt", "x")
    controller = RunAbortController()
    controller.abort()
    result = await fs.read_text_file("a.txt", signal=controller.signal)
    assert isinstance(result, Err)
    assert result.error.code == FsErrorCode.ABORTED


async def test_read_binary_file_pre_aborted(tmp_path: Path) -> None:
    fs = LocalFileSystem(cwd=str(tmp_path))
    await fs.write_file("a.txt", "x")
    controller = RunAbortController()
    controller.abort()
    result = await fs.read_binary_file("a.txt", signal=controller.signal)
    assert isinstance(result, Err)
    assert result.error.code == FsErrorCode.ABORTED


async def test_write_file_pre_aborted(tmp_path: Path) -> None:
    fs = LocalFileSystem(cwd=str(tmp_path))
    controller = RunAbortController()
    controller.abort()
    result = await fs.write_file("a.txt", "x", signal=controller.signal)
    assert isinstance(result, Err)
    assert result.error.code == FsErrorCode.ABORTED
    assert not (tmp_path / "a.txt").exists()


async def test_write_file_aborted_after_mkdir_before_write(tmp_path: Path) -> None:
    """`write_file` gets a THIRD checkpoint, immediately after its own parent-mkdir, before the
    write begins -- an aborted signal there must prevent the write, even though the parent
    directory was already created."""

    class _AbortAfterMkdirSignal:
        def __init__(self) -> None:
            self.calls = 0

        @property
        def aborted(self) -> bool:
            self.calls += 1
            # First call: pre-check (not yet aborted). Second call: the after-mkdir checkpoint.
            return self.calls >= 2

    fs = LocalFileSystem(cwd=str(tmp_path))
    signal = _AbortAfterMkdirSignal()
    result = await fs.write_file("sub/a.txt", "x", signal=signal)  # type: ignore[arg-type]
    assert isinstance(result, Err)
    assert result.error.code == FsErrorCode.ABORTED
    assert (tmp_path / "sub").is_dir()  # the mkdir itself was NOT checkpointed
    assert not (tmp_path / "sub" / "a.txt").exists()  # but the write never started


async def test_rename_file_pre_aborted(tmp_path: Path) -> None:
    fs = LocalFileSystem(cwd=str(tmp_path))
    await fs.write_file("a.txt", "x")
    controller = RunAbortController()
    controller.abort()
    result = await fs.rename_file("a.txt", "b.txt", signal=controller.signal)
    assert isinstance(result, Err)
    assert result.error.code == FsErrorCode.ABORTED
    assert (tmp_path / "a.txt").exists()  # never renamed


async def test_read_text_lines_post_loop_checkpoint(tmp_path: Path) -> None:
    """`CANCELLATION CHECKPOINT COUNT DIFFERS BY OPERATION` witness: a signal that aborts AFTER
    the last line has been yielded but BEFORE the operation returns is still caught -- the
    fourth, post-loop checkpoint `list_dir` deliberately lacks."""
    fs = LocalFileSystem(cwd=str(tmp_path))
    await fs.write_file("a.txt", "one\ntwo\n")

    class _AbortAtCall:
        def __init__(self, trigger_at: int) -> None:
            self.calls = 0
            self._trigger_at = trigger_at

        @property
        def aborted(self) -> bool:
            self.calls += 1
            return self.calls >= self._trigger_at

    # 2 lines: pre-check (1) + 2 loop iterations (2, 3) + post-loop check (4) -- trigger on #4.
    signal = _AbortAtCall(trigger_at=4)
    result = await fs.read_text_lines("a.txt", signal=signal)  # type: ignore[arg-type]
    assert isinstance(result, Err)
    assert result.error.code == FsErrorCode.ABORTED


async def test_list_dir_has_no_post_loop_checkpoint(tmp_path: Path) -> None:
    """The other half of the same witness: `list_dir`, unlike `read_text_lines`, has NO
    checkpoint after its own loop completes -- a signal that only becomes aborted after the last
    entry was yielded does not stop it from returning success."""
    fs = LocalFileSystem(cwd=str(tmp_path))
    await fs.write_file("a.txt", "x")
    await fs.write_file("b.txt", "y")

    class _AbortAtCall:
        def __init__(self, trigger_at: int) -> None:
            self.calls = 0
            self._trigger_at = trigger_at

        @property
        def aborted(self) -> bool:
            self.calls += 1
            return self.calls >= self._trigger_at

    # 2 entries: pre-check (1) + 2 loop iterations (2, 3) = 3 total calls, NO 4th (post-loop)
    # checkpoint. Trigger at #4 to prove a spurious extra check would be caught if one existed.
    signal = _AbortAtCall(trigger_at=4)
    result = await fs.list_dir(".", signal=signal)  # type: ignore[arg-type]
    assert isinstance(result, Ok)


async def test_list_dir_pre_aborted_on_empty_directory_still_returns_err(tmp_path: Path) -> None:
    """`L12-PY-R001` witness: the pre-loop checkpoint must fire independent of entry count.
    Previously an EMPTY directory never entered the `for entry in scandir(...)` loop body at
    all, so a signal that was already aborted BEFORE the call even started was never actually
    checked -- silently returning `Ok([])` instead of `Err(aborted)`."""
    fs = LocalFileSystem(cwd=str(tmp_path))
    controller = RunAbortController()
    controller.abort()
    result = await fs.list_dir(".", signal=controller.signal)
    assert isinstance(result, Err)
    assert result.error.code == FsErrorCode.ABORTED


class _AbortAfterDelay:
    """A `RunSignal`-shaped stand-in that becomes `aborted` after a fixed real-time delay --
    used to race against a genuinely blocking operation without needing an external trigger."""

    def __init__(self, delay: float) -> None:
        self._deadline = asyncio.get_running_loop().time() + delay

    @property
    def aborted(self) -> bool:
        return asyncio.get_running_loop().time() >= self._deadline


async def test_read_text_file_settles_promptly_when_signal_fires_mid_block(
    tmp_path: Path,
) -> None:
    """`L12-PY-R001` witness: a blocking read that is genuinely stuck (no writer on the other
    end of a FIFO) cannot be forcibly interrupted mid-syscall, but the CALLER-OBSERVABLE result
    must still settle promptly once the signal fires -- via `_race_signal`'s polling race, not
    by waiting for the underlying blocked thread to ever return. POSIX-only (real FIFO);
    skipped on platforms without `os.mkfifo` -- see the portable monkeypatch-based twins below
    for the platform-independent equivalent (e.g. Windows coverage)."""
    if not hasattr(os, "mkfifo"):
        pytest.skip("os.mkfifo is not available on this platform")
    fifo_path = tmp_path / "blocking-read-fifo"
    os.mkfifo(str(fifo_path))
    fs = LocalFileSystem(cwd=str(tmp_path))

    signal = _AbortAfterDelay(0.05)
    try:
        result = await asyncio.wait_for(
            fs.read_text_file("blocking-read-fifo", signal=signal),  # type: ignore[arg-type]
            timeout=2.0,
        )
        assert isinstance(result, Err)
        assert result.error.code == FsErrorCode.ABORTED
    finally:
        # `_race_signal` deliberately ABANDONS the still-blocked underlying thread (Python
        # cannot forcibly interrupt a blocked `open()` syscall) -- unblock it here so it
        # doesn't hang the interpreter's own executor-thread join at process exit.
        writer_fd = os.open(str(fifo_path), os.O_WRONLY)
        os.close(writer_fd)


async def test_write_file_settles_promptly_when_signal_fires_mid_block(tmp_path: Path) -> None:
    """The `write_file` half of the same `L12-PY-R001` witness: opening a FIFO for WRITE blocks
    identically until a reader appears -- the same prompt-settlement guarantee applies.
    POSIX-only (real FIFO); see the portable twin below for platform-independent coverage."""
    if not hasattr(os, "mkfifo"):
        pytest.skip("os.mkfifo is not available on this platform")
    fifo_path = tmp_path / "blocking-write-fifo"
    os.mkfifo(str(fifo_path))
    fs = LocalFileSystem(cwd=str(tmp_path))

    signal = _AbortAfterDelay(0.05)
    try:
        result = await asyncio.wait_for(
            fs.write_file("blocking-write-fifo", "x", signal=signal),  # type: ignore[arg-type]
            timeout=2.0,
        )
        assert isinstance(result, Err)
        assert result.error.code == FsErrorCode.ABORTED
    finally:
        reader_fd = os.open(str(fifo_path), os.O_RDONLY)
        os.close(reader_fd)


async def test_read_text_file_settles_promptly_when_signal_fires_mid_block_portable(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Portable (non-FIFO) twin of the witness above, closing coverage on platforms without
    `os.mkfifo` (e.g. Windows): `_read_text_sync` is monkeypatched to take longer than the
    signal's own abort delay, exercising `_race_signal`'s actual polling race directly rather
    than a real blocked syscall."""
    fs = LocalFileSystem(cwd=str(tmp_path))
    await fs.write_file("a.txt", "x")

    def _slow_read_text_sync(_path: str) -> str:
        time.sleep(0.2)
        return "unreachable"

    monkeypatch.setattr(filesystem_module, "_read_text_sync", _slow_read_text_sync)
    result = await fs.read_text_file("a.txt", signal=_AbortAfterDelay(0.05))  # type: ignore[arg-type]
    assert isinstance(result, Err)
    assert result.error.code == FsErrorCode.ABORTED


async def test_read_binary_file_settles_promptly_when_signal_fires_mid_block_portable(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The `read_binary_file` half of the same portable witness."""
    fs = LocalFileSystem(cwd=str(tmp_path))
    await fs.write_file("a.txt", "x")

    def _slow_read_binary_sync(_path: str) -> bytes:
        time.sleep(0.2)
        return b"unreachable"

    monkeypatch.setattr(filesystem_module, "_read_binary_sync", _slow_read_binary_sync)
    result = await fs.read_binary_file(
        "a.txt",
        signal=_AbortAfterDelay(0.05),  # type: ignore[arg-type]
    )
    assert isinstance(result, Err)
    assert result.error.code == FsErrorCode.ABORTED


async def test_write_file_settles_promptly_when_signal_fires_mid_block_portable(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The `write_file` half of the same portable witness."""
    fs = LocalFileSystem(cwd=str(tmp_path))

    def _slow_write_file_sync(_path: str, _content: str | bytes) -> None:
        time.sleep(0.2)

    monkeypatch.setattr(filesystem_module, "_write_file_sync", _slow_write_file_sync)
    result = await fs.write_file(
        "a.txt",
        "x",
        signal=_AbortAfterDelay(0.05),  # type: ignore[arg-type]
    )
    assert isinstance(result, Err)
    assert result.error.code == FsErrorCode.ABORTED


async def test_ten_operations_accept_but_do_not_inspect_signal(tmp_path: Path) -> None:
    """`CANCELLATION`/`CANCELLATION SIGNATURE UNIFORMITY` witnesses: the ten non-inspecting
    operations ACCEPT a pre-aborted signal in their typed signature but have NO effect from it --
    the call always proceeds normally, matching pinned Pi's own reference implementation."""
    fs = LocalFileSystem(cwd=str(tmp_path))
    await fs.write_file("a.txt", "x")
    controller = RunAbortController()
    controller.abort()
    signal = controller.signal

    assert await fs.absolute_path("a.txt", signal=signal) == Ok(str(tmp_path / "a.txt"))
    assert await fs.join_path(["a", "b"], signal=signal) == Ok(os.path.normpath("a/b"))
    assert await fs.append_file("a.txt", "y", signal=signal) == Ok(None)
    file_info_result = await fs.file_info("a.txt", signal=signal)
    assert isinstance(file_info_result, Ok)
    canonical_result = await fs.canonical_path("a.txt", signal=signal)
    assert isinstance(canonical_result, Ok)
    assert await fs.exists("a.txt", signal=signal) == Ok(True)
    assert await fs.create_dir("newdir", signal=signal) == Ok(None)
    assert await fs.remove("newdir", recursive=True, signal=signal) == Ok(None)
    temp_dir_result = await fs.create_temp_dir(signal=signal)
    assert isinstance(temp_dir_result, Ok)
    temp_file_result = await fs.create_temp_file(signal=signal)
    assert isinstance(temp_file_result, Ok)


# ---------------------------------------------------------------------------
# FsTarget / resolve / process_path (EXEC-003)
# ---------------------------------------------------------------------------


async def test_resolve_existing_path_uses_canonical(tmp_path: Path) -> None:
    fs = LocalFileSystem(cwd=str(tmp_path))
    await fs.write_file("a.txt", "x")
    result = await fs.resolve("a.txt")
    assert isinstance(result, Ok)
    assert os.path.samefile(result.value.target_key, tmp_path / "a.txt")


async def test_resolve_missing_path_falls_back_to_absolute(tmp_path: Path) -> None:
    fs = LocalFileSystem(cwd=str(tmp_path))
    result = await fs.resolve("missing.txt")
    assert isinstance(result, Ok)
    assert result.value.target_key == str(tmp_path / "missing.txt")


async def test_resolve_propagates_non_fallback_errors(tmp_path: Path) -> None:
    """`FSTARGET RESOLVED-LOCATION FRAMING` witness: a canonicalization failure OTHER than
    `not_found`/`not_supported` propagates -- `resolve()` does not silently fall back for it."""

    class _PermissionDeniedFileSystem(LocalFileSystem):
        async def canonical_path(self, path: str, signal: object = None) -> object:  # type: ignore[override]
            return Err(FsError(FsErrorCode.PERMISSION_DENIED, "denied", path))

    fs = _PermissionDeniedFileSystem(cwd=str(tmp_path))
    result = await fs.resolve("anything.txt")
    assert isinstance(result, Err)
    assert result.error.code == FsErrorCode.PERMISSION_DENIED


async def test_resolve_does_not_inspect_signal(tmp_path: Path) -> None:
    """`resolve()` itself is in the accepts-but-does-not-inspect group."""
    fs = LocalFileSystem(cwd=str(tmp_path))
    await fs.write_file("a.txt", "x")
    controller = RunAbortController()
    controller.abort()
    result = await fs.resolve("a.txt", signal=controller.signal)
    assert isinstance(result, Ok)


async def test_target_identity_symlink_and_target_share_one_key(tmp_path: Path) -> None:
    fs = LocalFileSystem(cwd=str(tmp_path))
    await fs.write_file("target.txt", "X")
    os.symlink(tmp_path / "target.txt", tmp_path / "link.txt")
    target_key = (await fs.resolve("target.txt")).value.target_key  # type: ignore[union-attr]
    link_key = (await fs.resolve("link.txt")).value.target_key  # type: ignore[union-attr]
    assert target_key == link_key


async def test_target_identity_survives_content_mutation(tmp_path: Path) -> None:
    fs = LocalFileSystem(cwd=str(tmp_path))
    await fs.write_file("a.txt", "one")
    key1 = (await fs.resolve("a.txt")).value.target_key  # type: ignore[union-attr]
    await fs.write_file("a.txt", "two")
    key2 = (await fs.resolve("a.txt")).value.target_key  # type: ignore[union-attr]
    assert key1 == key2


async def test_target_identity_does_not_collide_across_distinct_resources(tmp_path: Path) -> None:
    fs = LocalFileSystem(cwd=str(tmp_path))
    await fs.write_file("a.txt", "same")
    await fs.write_file("b.txt", "same")
    key_a = (await fs.resolve("a.txt")).value.target_key  # type: ignore[union-attr]
    key_b = (await fs.resolve("b.txt")).value.target_key  # type: ignore[union-attr]
    assert key_a != key_b


async def test_target_identity_changes_across_rename(tmp_path: Path) -> None:
    fs = LocalFileSystem(cwd=str(tmp_path))
    await fs.write_file("a.txt", "x")
    key_before = (await fs.resolve("a.txt")).value.target_key  # type: ignore[union-attr]
    await fs.rename_file("a.txt", "c.txt")
    key_after = (await fs.resolve("c.txt")).value.target_key  # type: ignore[union-attr]
    assert key_before != key_after


async def test_target_identity_reused_after_delete_and_recreate(tmp_path: Path) -> None:
    fs = LocalFileSystem(cwd=str(tmp_path))
    await fs.write_file("a.txt", "x")
    key1 = (await fs.resolve("a.txt")).value.target_key  # type: ignore[union-attr]
    await fs.remove("a.txt")
    await fs.write_file("a.txt", "y")
    key2 = (await fs.resolve("a.txt")).value.target_key  # type: ignore[union-attr]
    assert key1 == key2


async def test_process_path_returns_the_same_string_as_target_key_existing(tmp_path: Path) -> None:
    """`PROCESS_PATH RETURNS THE SAME STRING AS TARGET_KEY'S OWN DERIVATION` witness (case A)."""
    fs = LocalFileSystem(cwd=str(tmp_path))
    await fs.write_file("a.txt", "x")
    target = (await fs.resolve("a.txt")).value  # type: ignore[union-attr]
    result = await fs.process_path(target)
    assert result == Ok(target.target_key)


async def test_process_path_returns_the_same_string_as_target_key_missing(tmp_path: Path) -> None:
    """Case B: a not-yet-existing target's `process_path` is the lexical-absolute string, not a
    freshly (impossible) canonicalized one."""
    fs = LocalFileSystem(cwd=str(tmp_path))
    target = (await fs.resolve("missing.txt")).value  # type: ignore[union-attr]
    result = await fs.process_path(target)
    assert result == Ok(target.target_key)
    assert target.target_key == str(tmp_path / "missing.txt")


async def test_process_path_rejects_a_foreign_provider_target(tmp_path: Path) -> None:
    """`PROCESS_PATH SCOPED TO THE PRODUCING PROVIDER ONLY` witness."""
    fs_a = LocalFileSystem(cwd=str(tmp_path))
    fs_b = LocalFileSystem(cwd=str(tmp_path))
    await fs_a.write_file("a.txt", "x")
    target = (await fs_a.resolve("a.txt")).value  # type: ignore[union-attr]
    result = await fs_b.process_path(target)
    assert isinstance(result, Err)
    assert result.error.code == FsErrorCode.INVALID


async def test_file_info_skips_unsupported_entry_kinds_in_list_dir(tmp_path: Path) -> None:
    """Matches pinned Pi's own `listDir` exactly: an entry whose kind is none of
    file/directory/symlink is silently skipped, not propagated as the whole call's failure. On
    POSIX this is exercised with a real FIFO; skipped on platforms without `os.mkfifo`."""
    if not hasattr(os, "mkfifo"):
        pytest.skip("os.mkfifo is not available on this platform")
    fs = LocalFileSystem(cwd=str(tmp_path))
    await fs.write_file("a.txt", "x")
    os.mkfifo(str(tmp_path / "a-fifo"))
    result = await fs.list_dir(".")
    assert isinstance(result, Ok)
    names = {info.name for info in result.value}
    assert names == {"a.txt"}


# ---------------------------------------------------------------------------
# Coverage-closing: platform-portable unit tests for branches that need either
# a real unsupported file kind (no portable cross-platform way to create one) or a genuine
# OSError from an underlying syscall (unreliable to trigger portably without root/admin).
# ---------------------------------------------------------------------------


class _FakeStat:
    """A stat-like object matching none of S_ISREG/S_ISDIR/S_ISLNK -- the `_file_kind_from_stat`
    branch a real FIFO/socket/device would hit, tested directly since `os.mkfifo` is POSIX-only
    (already covered on POSIX by `test_file_info_skips_unsupported_entry_kinds_in_list_dir`)."""

    st_mode = 0  # no S_IS* bit set
    st_size = 0
    st_mtime = 0.0


def test_file_kind_from_stat_returns_none_for_unsupported_kind() -> None:
    assert _file_kind_from_stat(_FakeStat()) is None  # type: ignore[arg-type]


def test_file_info_sync_raises_for_unsupported_kind(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(os, "lstat", lambda _path: _FakeStat())
    with pytest.raises(_UnsupportedFileType):
        _file_info_sync("irrelevant")


async def test_file_info_maps_unsupported_kind_to_invalid(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    fs = LocalFileSystem(cwd=str(tmp_path))
    monkeypatch.setattr(os, "lstat", lambda _path: _FakeStat())
    result = await fs.file_info("anything")
    assert isinstance(result, Err)
    assert result.error.code == FsErrorCode.INVALID


async def test_list_dir_silently_skips_unsupported_entries(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The portable half of `test_file_info_skips_unsupported_entry_kinds_in_list_dir`'s own
    POSIX-only FIFO witness: forcing every entry's own per-entry `lstat` (inside
    `_file_info_sync`, called from `_list_dir_sync`'s loop) to report an unsupported kind
    exercises the `except _UnsupportedFileType: continue` branch directly -- every entry is
    silently skipped rather than the whole call failing."""
    fs = LocalFileSystem(cwd=str(tmp_path))
    await fs.write_file("a.txt", "x")
    monkeypatch.setattr(os, "lstat", lambda _path: _FakeStat())
    result = await fs.list_dir(".")
    assert result == Ok([])


async def test_read_binary_file_not_found(tmp_path: Path) -> None:
    fs = LocalFileSystem(cwd=str(tmp_path))
    result = await fs.read_binary_file("missing.bin")
    assert isinstance(result, Err)
    assert result.error.code == FsErrorCode.NOT_FOUND


async def test_write_file_mkdir_failure_propagates(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    fs = LocalFileSystem(cwd=str(tmp_path))

    def _raise(*_args: object, **_kwargs: object) -> None:
        raise PermissionError("simulated mkdir failure")

    monkeypatch.setattr(os, "makedirs", _raise)
    result = await fs.write_file("sub/a.txt", "x")
    assert isinstance(result, Err)
    assert result.error.code == FsErrorCode.PERMISSION_DENIED


async def test_write_file_to_a_directory_path_fails(tmp_path: Path) -> None:
    """A genuine, portable OSError trigger: writing to a path that is already a directory."""
    fs = LocalFileSystem(cwd=str(tmp_path))
    await fs.create_dir("adir")
    result = await fs.write_file("adir", "x")
    assert isinstance(result, Err)


async def test_append_file_mkdir_or_write_failure_propagates(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    fs = LocalFileSystem(cwd=str(tmp_path))

    def _raise(*_args: object, **_kwargs: object) -> None:
        raise PermissionError("simulated failure")

    monkeypatch.setattr(os, "makedirs", _raise)
    result = await fs.append_file("sub/a.txt", "x")
    assert isinstance(result, Err)
    assert result.error.code == FsErrorCode.PERMISSION_DENIED


async def test_list_dir_not_found_returns_err(tmp_path: Path) -> None:
    """Distinct from `test_list_dir_not_found` above: confirms the `except OSError` branch (not
    the `_AbortedSignal` one) is what maps a genuine lookup failure."""
    fs = LocalFileSystem(cwd=str(tmp_path))
    result = await fs.list_dir("truly-missing")
    assert isinstance(result, Err)
    assert result.error.code == FsErrorCode.NOT_FOUND


async def test_list_dir_mid_loop_abort(tmp_path: Path) -> None:
    """The `_AbortedSignal` branch inside `list_dir`'s own loop, distinct from the
    already-covered "no post-loop checkpoint" case: aborting DURING the loop (not after it) is
    still caught, at whichever entry the checkpoint next runs."""
    fs = LocalFileSystem(cwd=str(tmp_path))
    for i in range(3):
        await fs.write_file(f"{i}.txt", "x")

    class _AbortAtCall:
        def __init__(self, trigger_at: int) -> None:
            self.calls = 0
            self._trigger_at = trigger_at

        @property
        def aborted(self) -> bool:
            self.calls += 1
            return self.calls >= self._trigger_at

    signal = _AbortAtCall(trigger_at=2)  # pre-check (1) then abort on the first loop iteration
    result = await fs.list_dir(".", signal=signal)  # type: ignore[arg-type]
    assert isinstance(result, Err)
    assert result.error.code == FsErrorCode.ABORTED


async def test_read_text_lines_pre_aborted(tmp_path: Path) -> None:
    fs = LocalFileSystem(cwd=str(tmp_path))
    await fs.write_file("a.txt", "one\ntwo\n")
    controller = RunAbortController()
    controller.abort()
    result = await fs.read_text_lines("a.txt", signal=controller.signal)
    assert isinstance(result, Err)
    assert result.error.code == FsErrorCode.ABORTED


async def test_read_text_lines_mid_loop_abort(tmp_path: Path) -> None:
    fs = LocalFileSystem(cwd=str(tmp_path))
    await fs.write_file("a.txt", "one\ntwo\nthree\n")

    class _AbortAtCall:
        def __init__(self, trigger_at: int) -> None:
            self.calls = 0
            self._trigger_at = trigger_at

        @property
        def aborted(self) -> bool:
            self.calls += 1
            return self.calls >= self._trigger_at

    signal = _AbortAtCall(trigger_at=2)  # pre-check (1) then abort on the first loop iteration
    result = await fs.read_text_lines("a.txt", signal=signal)  # type: ignore[arg-type]
    assert isinstance(result, Err)
    assert result.error.code == FsErrorCode.ABORTED


async def test_exists_propagates_non_not_found_errors(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """`exists()` derives from `file_info()`: a `not_found` maps to `Ok(False)`, but any OTHER
    error propagates as-is, never silently collapsed to `False`."""
    fs = LocalFileSystem(cwd=str(tmp_path))
    monkeypatch.setattr(os, "lstat", lambda _path: _FakeStat())
    result = await fs.exists("anything")
    assert isinstance(result, Err)
    assert result.error.code == FsErrorCode.INVALID


async def test_create_temp_dir_failure_propagates(monkeypatch: pytest.MonkeyPatch) -> None:
    import tempfile as tempfile_module

    def _raise(*_args: object, **_kwargs: object) -> str:
        raise PermissionError("simulated mkdtemp failure")

    monkeypatch.setattr(tempfile_module, "mkdtemp", _raise)
    fs = LocalFileSystem()
    result = await fs.create_temp_dir()
    assert isinstance(result, Err)
    assert result.error.code == FsErrorCode.PERMISSION_DENIED


async def test_create_temp_file_propagates_temp_dir_failure(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import tempfile as tempfile_module

    def _raise(*_args: object, **_kwargs: object) -> str:
        raise PermissionError("simulated mkdtemp failure")

    monkeypatch.setattr(tempfile_module, "mkdtemp", _raise)
    fs = LocalFileSystem()
    result = await fs.create_temp_file()
    assert isinstance(result, Err)
    assert result.error.code == FsErrorCode.PERMISSION_DENIED


async def test_create_temp_file_write_failure_propagates(monkeypatch: pytest.MonkeyPatch) -> None:
    def _raise(*_args: object, **_kwargs: object) -> None:
        raise PermissionError("simulated write failure")

    monkeypatch.setattr(filesystem_module, "_write_file_sync", _raise)
    fs = LocalFileSystem()
    result = await fs.create_temp_file()
    assert isinstance(result, Err)
    assert result.error.code == FsErrorCode.PERMISSION_DENIED


async def test_resolve_propagates_absolute_path_failure_after_not_found(
    tmp_path: Path,
) -> None:
    """The defensive `return absolute` branch in `resolve()`: reachable if a provider's own
    `absolute_path` (normally infallible for `LocalFileSystem`) were ever to fail after
    `canonical_path` already reported `not_found` -- exercised here via a subclass override to
    prove the propagation logic itself is correct, not dead code."""

    class _FailingAbsolutePath(LocalFileSystem):
        async def absolute_path(self, path: str, signal: object = None) -> object:  # type: ignore[override]
            return Err(FsError(FsErrorCode.INVALID, "simulated absolute_path failure", path))

    fs = _FailingAbsolutePath(cwd=str(tmp_path))
    result = await fs.resolve("missing.txt")
    assert isinstance(result, Err)
    assert result.error.code == FsErrorCode.INVALID
