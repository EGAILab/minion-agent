"""The service scrubs before sinks and never lets a sink affect behavior."""

from minion_agent.telemetry.sanitize import REDACTED
from minion_agent.telemetry.service import RecordingSink, TelemetryService
from minion_agent.telemetry.spans import Span, SpanKind


def _span(**attributes: object) -> Span:
    return Span(kind=SpanKind.STEP, name="step", attributes=dict(attributes))


def test_emitted_spans_reach_a_sink() -> None:
    service = TelemetryService()
    sink = RecordingSink()
    service.add_sink(sink)

    service.emit(_span(model="mock-1"))

    assert sink.spans[0].attributes["model"] == "mock-1"


def test_sinks_never_see_unscrubbed_spans() -> None:
    """The ordering guarantee: redaction is a boundary, not a listener."""
    service = TelemetryService()
    sink = RecordingSink()
    service.add_sink(sink)
    service.sanitizer.add_secret("sk-abc123")

    service.emit(_span(authorization="Bearer sk-abc123"))

    assert REDACTED in sink.spans[0].attributes["authorization"]


def test_every_sink_receives_every_span() -> None:
    service = TelemetryService()
    first, second = RecordingSink(), RecordingSink()
    service.add_sink(first)
    service.add_sink(second)

    service.emit(_span())

    assert len(first.spans) == len(second.spans) == 1


def test_removing_a_sink_stops_delivery() -> None:
    service = TelemetryService()
    sink = RecordingSink()
    remove = service.add_sink(sink)

    remove()
    service.emit(_span())

    assert sink.spans == []


def test_removing_a_sink_twice_is_harmless() -> None:
    service = TelemetryService()
    remove = service.add_sink(RecordingSink())

    remove()
    remove()


def test_a_failing_sink_does_not_break_the_others() -> None:
    """Telemetry is observational: a broken sink must not change behavior."""

    class Broken:
        def emit(self, span: Span) -> None:
            raise RuntimeError("sink is down")

    service = TelemetryService()
    service.add_sink(Broken())
    healthy = RecordingSink()
    service.add_sink(healthy)

    service.emit(_span())

    assert len(healthy.spans) == 1


def test_emitting_with_no_sinks_is_harmless() -> None:
    TelemetryService().emit(_span())
