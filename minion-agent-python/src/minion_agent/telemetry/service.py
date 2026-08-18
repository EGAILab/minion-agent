"""The `ctx.telemetry` seam.

Sinks are plugins; a deployment may mount none. Emission scrubs before any
sink runs, and a failing sink is contained — telemetry is an observational
projection, so nothing it does may change behavior.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Protocol

from .sanitize import Sanitizer
from .spans import Span


class Sink(Protocol):
    """Somewhere scrubbed spans go."""

    def emit(self, span: Span) -> None: ...


class RecordingSink:
    """Keeps every span it receives. The default, and what tests assert on."""

    def __init__(self) -> None:
        self.spans: list[Span] = []

    def emit(self, span: Span) -> None:
        self.spans.append(span)


class TelemetryService:
    """Scrubs spans and fans them out to sinks."""

    __service_name__ = "telemetry"

    def __init__(self) -> None:
        self.sanitizer = Sanitizer()
        self.recording: RecordingSink | None = None
        """Set by the plugin when it mounts a default recording sink."""
        self._sinks: list[Sink] = []

    def add_sink(self, sink: Sink) -> Callable[[], None]:
        """Register `sink`; returns a handle that removes it."""
        self._sinks.append(sink)

        def remove() -> None:
            if sink in self._sinks:
                self._sinks.remove(sink)

        return remove

    def emit(self, span: Span) -> None:
        """Scrub `span`, then deliver it to every sink.

        A sink that raises is contained: the remaining sinks still receive the
        span, and the caller never learns. An observational projection must not
        be able to fail the thing it observes.
        """
        scrubbed = self.sanitizer.scrub(span)
        for sink in list(self._sinks):
            try:
                sink.emit(scrubbed)
            except Exception:  # noqa: BLE001 - telemetry must never affect behavior
                continue
