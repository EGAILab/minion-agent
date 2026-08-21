"""Detects scenario files scaffolded with `TO_BE_*` sentinel content.

A placeholder scenario carries the frozen `2026-08-20-minion-agent-design.md`'s canonical scenario
*name* so its intended coverage is trackable, but no real `given`/`when`/`expect` content yet --
written when implementation reaches the behavior it names. Detecting this from the document itself
(any string value starting with ``TO_BE_``) rather than a hardcoded name list means a scenario stops
being treated as a placeholder the moment its real content replaces the sentinel, with no separate
list to remember to update.
"""

from __future__ import annotations

from typing import Any


def is_placeholder(document: dict[str, Any]) -> bool:
    """True if any string value in `document` is a `TO_BE_*` sentinel."""

    def _walk(value: Any) -> bool:
        if isinstance(value, str):
            return value.startswith("TO_BE_")
        if isinstance(value, dict):
            return any(_walk(v) for v in value.values())
        if isinstance(value, list):
            return any(_walk(v) for v in value)
        return False

    return _walk(document)
