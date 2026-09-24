"""JS number semantics, truncation, the TOOL-026 path pipeline and the abort race.

Expected values in the JS tables were produced by Node 22.15.1 (`String(x)`, `Math.round`,
`x.toFixed(n)`, `Array.prototype.slice`), not by this implementation.
"""

import asyncio
import math

import pytest

from minion_agent.runtime import RunAbortController
from minion_agent.tools.builtin import paths
from minion_agent.tools.builtin._js import (
    js_slice,
    math_max,
    math_min,
    math_round,
    number_to_string,
    to_fixed,
)
from minion_agent.tools.builtin._signal import race_abort
from minion_agent.tools.builtin.paths import BuiltinToolError, preprocess_path
from minion_agent.tools.builtin.truncate import format_size, truncate_head

NODE_TO_STRING = [
    (0.0, "0"),
    (-0.0, "0"),
    (1.5, "1.5"),
    (-2.5, "-2.5"),
    (0.1, "0.1"),
    (0.30000000000000004, "0.30000000000000004"),
    (1e21, "1e+21"),
    (1e20, "100000000000000000000"),
    (123456789012345680000, "123456789012345680000"),
    (1e-6, "0.000001"),
    (1e-7, "1e-7"),
    (1.5e-7, "1.5e-7"),
    (5e-324, "5e-324"),
    (1.7976931348623157e308, "1.7976931348623157e+308"),
    (1e100, "1e+100"),
    (13.5, "13.5"),
    (float("nan"), "NaN"),
    (float("inf"), "Infinity"),
    (float("-inf"), "-Infinity"),
]


@pytest.mark.parametrize(("value", "expected"), NODE_TO_STRING)
def test_number_to_string_matches_node(value: float, expected: str) -> None:
    assert number_to_string(value) == expected


NODE_ROUND_AND_FIXED = [
    # value, Math.round, toFixed(1), toFixed(2)
    (0.49999999999999994, 0, "0.5", "0.50"),
    (-0.5, -0.0, "-0.5", "-0.50"),
    (2.5, 3, "2.5", "2.50"),
    (-2.5, -2, "-2.5", "-2.50"),
    (1.125, 1, "1.1", "1.13"),
    (1.005, 1, "1.0", "1.00"),
    (2.675, 3, "2.7", "2.67"),
    (1048575 / 1024, 1024, "1024.0", "1024.00"),
    (4503599627370495.5, 4503599627370496, "4503599627370495.5", "4503599627370495.50"),
    (-0.001, -0.0, "-0.0", "-0.00"),
    (-0.0, -0.0, "0.0", "0.00"),
]


@pytest.mark.parametrize(("value", "rounded", "fixed1", "fixed2"), NODE_ROUND_AND_FIXED)
def test_round_and_to_fixed_match_node(
    value: float, rounded: float, fixed1: str, fixed2: str
) -> None:
    assert math_round(value) == rounded
    assert (to_fixed(value, 1), to_fixed(value, 2)) == (fixed1, fixed2)


def test_non_finite_edges() -> None:
    assert math.isnan(math_round(float("nan")))
    assert math_round(float("inf")) == float("inf")
    assert to_fixed(float("nan"), 2) == "NaN"
    assert to_fixed(1e21, 2) == "1e+21"
    assert to_fixed(float("-inf"), 2) == "-Infinity"
    assert math.isnan(math_min(1.0, float("nan"))) and math.isnan(math_max(float("nan"), 1.0))
    assert (math_min(1.0, 2.0), math_max(1.0, 2.0)) == (1.0, 2.0)


NODE_SLICE = [
    # Array.from({length: 10}, (_, i) => i).slice(start, end)
    (1.5, 0.5, []),
    (0, -3, [0, 1, 2, 3, 4, 5, 6]),
    (1.5, -3.5, [1, 2, 3, 4, 5, 6]),
    (-2, None, [8, 9]),
    (-100, 2, [0, 1]),
    (2, 1e9, [2, 3, 4, 5, 6, 7, 8, 9]),
    (float("nan"), 2, [0, 1]),
    (float("-inf"), float("inf"), list(range(10))),
    (float("inf"), None, []),
]


@pytest.mark.parametrize(("start", "end", "expected"), NODE_SLICE)
def test_js_slice_matches_node(start: float, end: float | None, expected: list[int]) -> None:
    assert js_slice(list(range(10)), start, end) == expected


# formatSize thresholds executed against pinned Pi truncate.ts (assurance R004 record).
@pytest.mark.parametrize(
    ("size", "text"),
    [
        (0, "0B"),
        (1023, "1023B"),
        (1024, "1.0KB"),
        (51200, "50.0KB"),
        (1048575, "1024.0KB"),
        (1048576, "1.0MB"),
        (4718592, "4.5MB"),
    ],
)
def test_format_size(size: int, text: str) -> None:
    assert format_size(size) == text


def test_truncate_head_counts_a_trailing_newline_as_no_extra_line() -> None:
    result = truncate_head("a\nb\n", max_lines=1)
    assert (result.content, result.truncated_by, result.total_lines, result.output_lines) == (
        "a",
        "lines",
        2,
        1,
    )


def test_path_pipeline_steps_one_to_three() -> None:
    assert preprocess_path("\u00a0a\u3000b ") == " a b "  # closed set only; nothing trimmed
    assert preprocess_path("@@x") == "@x"  # exactly one "@"
    assert preprocess_path("\u200bx") == "\u200bx"  # U+200B is not in the set


@pytest.mark.parametrize(
    ("raw", "converted"),
    [
        ("/c/Users/me", "C:\\Users\\me"),
        ("/mnt/d/x/y", "D:\\x\\y"),
        ("/CYGDRIVE/e/", "E:\\"),
        ("/z", "Z:\\"),
        ("//server/share", "//server/share"),
        ("/c\\x", "/c\\x"),
        ("relative/c", "relative/c"),
        ("/cc/x", "/cc/x"),
        ("/c/x\u2028y", "/c/x\u2028y"),  # JS `.` stops at line terminators
        ("/c/x\n", "/c/x\n"),  # JS `$` is end of input
        ("/\u017f/x", "/\u017f/x"),  # `/i` without `u` folds ASCII only
        ("/\u212a/x", "/\u212a/x"),
    ],
)
def test_windows_shell_path_conversion(raw: str, converted: str) -> None:
    """Pi's normalizeWindowsShellPath, exercised directly so the rule is pinned on every host."""
    assert paths._normalize_windows_shell_path(raw) == converted


def test_windows_conversion_applies_only_on_windows(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(paths.os, "name", "posix")
    assert preprocess_path("/c/x") == "/c/x"
    monkeypatch.setattr(paths.os, "name", "nt")
    assert preprocess_path("/c/x") == "C:\\x"


def test_malformed_file_url_is_rejected_with_the_invalid_phrase() -> None:
    with pytest.raises(BuiltinToolError, match=r"^Cannot access file:///%ZZ: invalid path$"):
        preprocess_path("file:///%ZZ")


async def test_race_abort_without_signal_just_awaits() -> None:
    async def work() -> int:
        return 7

    assert await race_abort(work(), None) == 7


async def test_race_abort_pre_aborted_never_starts_the_work() -> None:
    started: list[bool] = []

    async def work() -> int:
        started.append(True)
        return 1

    controller = RunAbortController()
    controller.abort()
    with pytest.raises(BuiltinToolError, match=r"^Operation aborted$"):
        await race_abort(work(), controller.signal)
    assert started == []


async def test_race_abort_rejects_promptly_and_cancels_the_work() -> None:
    controller = RunAbortController()
    cancelled: list[bool] = []

    async def work() -> int:
        try:
            await asyncio.sleep(30)
        except asyncio.CancelledError:
            cancelled.append(True)
            raise
        return 1

    async def abort_soon() -> None:
        await asyncio.sleep(0.05)
        controller.abort()

    aborter = asyncio.ensure_future(abort_soon())
    with pytest.raises(BuiltinToolError, match=r"^Operation aborted$"):
        await asyncio.wait_for(race_abort(work(), controller.signal), timeout=5)
    await aborter
    await asyncio.sleep(0)
    assert cancelled == [True]


async def test_race_abort_after_completion_keeps_the_result() -> None:
    controller = RunAbortController()

    async def work() -> str:
        return "done"

    assert await race_abort(work(), controller.signal) == "done"
    controller.abort()


async def test_race_abort_propagates_work_errors() -> None:
    async def work() -> int:
        raise BuiltinToolError("boom")

    with pytest.raises(BuiltinToolError, match=r"^boom$"):
        await race_abort(work(), RunAbortController().signal)


def test_aborted_helper_text() -> None:
    assert str(paths.aborted()) == "Operation aborted"
