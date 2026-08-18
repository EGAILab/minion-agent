"""The append-only session log.

Sequence-numbered and JSON-validated at append, because the log is the
system's semantic truth: it must replay exactly and port to another language
without carrying Python-only values.
"""

from __future__ import annotations

from typing import Any

from .events import EventKind, SessionEvent, is_surface

_JSON_SCALARS = (str, int, float, bool, type(None))


class NotJsonSafeError(Exception):
    """Event data contained a value JSON cannot represent."""


def _check_json_safe(value: Any, path: str = "data") -> None:
    """Raise unless `value` is representable in JSON."""
    if isinstance(value, _JSON_SCALARS):
        return
    if isinstance(value, dict):
        for key, item in value.items():
            if not isinstance(key, str):
                raise NotJsonSafeError(f"{path}: object keys must be strings, got {key!r}")
            _check_json_safe(item, f"{path}.{key}")
        return
    if isinstance(value, list | tuple):
        for index, item in enumerate(value):
            _check_json_safe(item, f"{path}[{index}]")
        return
    raise NotJsonSafeError(f"{path}: {type(value).__name__} is not JSON-safe")


class SessionLog:
    """An ordered, append-only sequence of events."""

    def __init__(
        self,
        session_id: str,
        ancestor: SessionLog | None = None,
        boundary: int = 0,
    ) -> None:
        self.session_id = session_id
        self.ancestor = ancestor
        """The log this one forked from, or None for a root session."""
        self.boundary = boundary
        """The ancestor sequence number this fork branched at."""
        self._events: list[SessionEvent] = []

    def __len__(self) -> int:
        return len(self._events)

    @property
    def events(self) -> tuple[SessionEvent, ...]:
        """Every event, in append order."""
        return tuple(self._events)

    def append(self, kind: EventKind, data: dict[str, Any]) -> SessionEvent:
        """Append one event, assigning the next sequence number.

        Validates before appending, so a rejected event leaves no trace.
        """
        _check_json_safe(data)
        event = SessionEvent(seq=len(self._events) + 1, kind=kind, data=dict(data))
        self._events.append(event)
        return event

    def surface(self) -> tuple[SessionEvent, ...]:
        """Only the events that project into model history."""
        return tuple(event for event in self._events if is_surface(event))
