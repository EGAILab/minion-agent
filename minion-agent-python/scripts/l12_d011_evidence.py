"""L12-D011 control evidence reporter (CE-L12D011-01 revision 1; assurance record section 7).

A pytest plugin loaded ONLY by the negative-control driver (`-p l12_d011_evidence`, with
`scripts/` on the path); the normal suite never loads it. For every failed report it OBSERVES
the exception object before any rendering:

- whether its type IS the builtin AssertionError / ValueError (identity captured at import,
  never a name);
- the first line of str(exception);
- every traceback frame, outermost to innermost, as (resolved file, code qualname, line).

It decides nothing: the driver classifies these observations against each witness's
declaration. Records are written through the terminal reporter after output capture has ended,
framed by START/END lines that carry the per-run token from L12D011_EVIDENCE_TOKEN. An
observation that cannot be made is recorded as such ("error"), never invented. Nothing is
written to disk.
"""

from __future__ import annotations

import builtins
import json
import os
from collections.abc import Generator
from pathlib import Path
from types import TracebackType
from typing import Any

import pytest

TOKEN_ENV = "L12D011_EVIDENCE_TOKEN"
PREFIX = "L12D011-EVIDENCE"
_ASSERTION = builtins.AssertionError
_VALUE = builtins.ValueError
_RECORDS: list[dict[str, Any]] = []


def _frames(tb: TracebackType | None) -> list[dict[str, Any]]:
    frames = []
    while tb is not None:
        code = tb.tb_frame.f_code
        frames.append(
            {
                "file": str(Path(code.co_filename).resolve()),
                "qualname": code.co_qualname,
                "line": tb.tb_lineno,
            }
        )
        tb = tb.tb_next
    return frames


def observe(exc: BaseException) -> dict[str, Any]:
    """The observation of one exception (pure; exercised in-process by the driver's self-test)."""
    return {
        "is_assertion": type(exc) is _ASSERTION,
        "is_value_error": type(exc) is _VALUE,
        "type": f"{type(exc).__module__}.{type(exc).__qualname__}",
        "first_line": str(exc).split("\n", 1)[0],
        "frames": _frames(exc.__traceback__),
    }


@pytest.hookimpl(hookwrapper=True)
def pytest_runtest_makereport(
    item: pytest.Item, call: pytest.CallInfo[None]
) -> Generator[None, Any, None]:
    outcome = yield
    report = outcome.get_result()
    if not report.failed:
        return
    record: dict[str, Any] = {"node": report.nodeid, "when": report.when}
    try:
        if call.excinfo is None:
            record["error"] = "no exception information"
        else:
            record.update(observe(call.excinfo.value))
    except BaseException as error:
        record = {"node": report.nodeid, "when": report.when, "error": repr(error)}
    _RECORDS.append(record)


def pytest_terminal_summary(terminalreporter: Any) -> None:
    token = os.environ.get(TOKEN_ENV, "")
    terminalreporter.write_line(f"{PREFIX} START " + json.dumps({"token": token}))
    for record in _RECORDS:
        terminalreporter.write_line(
            f"{PREFIX} RECORD " + json.dumps({"token": token, **record}, ensure_ascii=True)
        )
    terminalreporter.write_line(
        f"{PREFIX} END " + json.dumps({"token": token, "count": len(_RECORDS)})
    )
