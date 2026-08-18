"""The loop reports; telemetry never reports back."""

from minion_agent.agent_loop.driver import AgentLoop
from minion_agent.llm import TextBlock, UserMessage
from minion_agent.llm.adapters.mock import ScriptedResponse
from minion_agent.llm.messages import StopReason
from minion_agent.telemetry import RecordingSink, Span, SpanKind, TelemetryService

from .test_single_turn import _loop


def _say(text: str) -> UserMessage:
    return UserMessage(content=(TextBlock(text=text),), timestamp=1)


def _with_telemetry(loop: AgentLoop) -> tuple[TelemetryService, RecordingSink]:
    service = TelemetryService()
    recording = RecordingSink()
    service.recording = recording
    service.add_sink(recording)
    loop.telemetry = service
    return service, recording


async def test_a_turn_emits_a_turn_span() -> None:
    loop = _loop(ScriptedResponse((), StopReason.STOP))
    _, recording = _with_telemetry(loop)
    loop.instance.inbox.followup(_say("hello"))

    await loop.run_until_idle()

    kinds = [span.kind for span in recording.spans]
    assert SpanKind.TURN in kinds
    assert SpanKind.STEP in kinds


async def test_spans_identify_their_instance() -> None:
    loop = _loop(ScriptedResponse((), StopReason.STOP))
    _, recording = _with_telemetry(loop)
    loop.instance.inbox.followup(_say("hello"))

    await loop.run_until_idle()

    turn = next(s for s in recording.spans if s.kind is SpanKind.TURN)
    assert turn.attributes["instance"] == "room-a"
    assert turn.attributes["reason"] == "completed"


async def test_a_failing_sink_cannot_break_a_turn() -> None:
    """Telemetry is an observational projection: it must not be able to fail
    the thing it observes."""

    class Broken:
        def emit(self, span: Span) -> None:
            raise RuntimeError("sink down")

    loop = _loop(ScriptedResponse((TextBlock(text="ok"),), StopReason.STOP))
    telemetry, _ = _with_telemetry(loop)
    telemetry.add_sink(Broken())
    loop.instance.inbox.followup(_say("hello"))

    await loop.run_until_idle()

    assert len(loop.instance.log) > 0


async def test_no_telemetry_service_is_harmless() -> None:
    loop = _loop(ScriptedResponse((), StopReason.STOP))
    loop.instance.inbox.followup(_say("hello"))

    await loop.run_until_idle()

    assert len(loop.instance.log) > 0
