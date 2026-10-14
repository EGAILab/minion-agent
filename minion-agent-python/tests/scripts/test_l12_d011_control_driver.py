"""W4 (CE-L12D011-01 revision 1, record section 7.4): the L12-D011 negative-control credit rules
run in the full suite on every platform -- the credit matrix R1-R13 on real in-process
observations (including the C001/C002 counterfeits and a genuine native NUL fire through the
seam), the declaration table R14 against the current test source, and the driver meta-controls
D1-D10 (each known-bad mutation must fail its named row). In memory only: no file writes, no
pytest subprocess (W3 runs in the driver itself)."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest

DRIVER = Path(__file__).resolve().parents[2] / "scripts" / "l12_d011_negative_controls.py"


def _driver():  # type: ignore[no-untyped-def]
    spec = importlib.util.spec_from_file_location("l12_d011_negative_controls_w4", DRIVER)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


CHECKS = _driver().in_process_checks()


@pytest.mark.parametrize(("label", "ok"), CHECKS, ids=[label for label, _ in CHECKS])
def test_control_credit_rule(label: str, ok: bool) -> None:
    assert ok, label


def test_every_meta_control_is_present() -> None:
    names = {label.split(":", 1)[0] for label, _ in CHECKS if label.startswith("D")}
    ids = {name.split(" ", 1)[0] for name in names}
    assert len(names) == 11 and ids == {f"D{i}" for i in range(1, 11)}
