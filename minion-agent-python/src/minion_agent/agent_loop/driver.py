"""The concrete agent loop.

Imperative and package-internal, following DSH's `ReactLoopAgent`: a stateful
class with an explicit phase. Neither pi nor DSH factors the live loop as a
pure reducer, and this design does not invent one.

Pi vocabulary, adopted exactly (Layer 08): a **run** is one `prompt()`/
`continue()`-equivalent invocation, bracketed by `agent_start`/`agent_end`. A
**turn** is one assistant response plus the tool calls/results it triggers,
bracketed by `turn_start`/`turn_end` -- never more than one provider request.
Pi has no "step" concept; `_run_step` is Minion's own internal helper name for
"run one turn," kept because the observable boundary it now emits (one
`TURN_START`/`TURN_END` pair per call) is what matters, not the helper's own
name. An earlier revision conflated a turn with "zero or more steps" and let
`_run_step` run several times inside a single `TURN_START`/`TURN_END`
bracket -- observably wrong once more than one provider request occurred in
what Minion called one turn (confirmed against pinned Pi's `agent-loop.ts`,
where `turn_start`/`turn_end` bracket exactly one assistant response). What
that prior single bracket actually represented was pi's **run**: `_run_once`
(renamed from `_run_turn`) now owns the `AGENT_START`/`AGENT_END` bracket,
and `_run_step` owns `TURN_START`/`TURN_END` around exactly one provider
request.
"""

from __future__ import annotations

from ..agent.decisions import (
    Enter,
    PreStepDecision,
    PreStepReason,
    Reject,
    TurnStopping,
    resolve_stopping,
)
from ..agent.envelope import ClaimPolicy, InboxTarget, InputEnvelope
from ..agent.events import AGENT_PRE_STEP, AGENT_TURN_STOPPING
from ..agent.identity import AgentStatus
from ..agent.instance import AgentInstance
from ..llm import (
    LlmService,
    Message,
    Request,
    StopReason,
    StreamChunk,
    TextDelta,
    ThinkingDelta,
    ToolCallBlock,
    ToolCallDelta,
    collect,
)
from ..session import (
    ArtifactStore,
    EventKind,
    assemble_system,
    derive_messages,
    encode_message,
    record_header,
)
from ..telemetry import Span, SpanKind, TelemetryService
from ..tools.batch import BatchOutcome, execute_batch, execute_length_stop_batch
from ..tools.registry import ToolRegistry


class AgentLoop:
    """Drives one agent instance through turns and steps."""

    def __init__(
        self,
        *,
        instance: AgentInstance,
        llm: LlmService,
        tools: ToolRegistry,
        artifacts: ArtifactStore,
        telemetry: TelemetryService | None = None,
    ) -> None:
        self.instance = instance
        # Collaborators are public: tests configure them directly.
        self.llm = llm
        self.tools = tools
        self.artifacts = artifacts
        self.telemetry = telemetry
        self.next_turn_policy = ClaimPolicy.ONE_AT_A_TIME
        self.next_step_policy = ClaimPolicy.ONE_AT_A_TIME
        self._cancelled = False

    async def run_until_idle(self) -> None:
        """Open turns while input is pending, then settle idle."""
        inbox = self.instance.inbox
        if not inbox.pending(InboxTarget.NEXT_TURN):
            inbox.take_wake()
            return

        self.instance.set_status(AgentStatus.RUNNING)
        try:
            while inbox.pending(InboxTarget.NEXT_TURN):
                await self._run_once()
        finally:
            inbox.take_wake()
            self.instance.set_status(AgentStatus.IDLE)

    def _span(self, kind: SpanKind, name: str, **attributes: object) -> None:
        """Emit a span if telemetry is mounted. Never affects control flow."""
        if self.telemetry is None:
            return
        self.telemetry.emit(
            Span(
                kind=kind,
                name=name,
                attributes={"instance": self.instance.id, **attributes},
            )
        )

    def cancel(self) -> None:
        """Request that the current turn end at its next boundary.

        Work already in flight -- a running tool, an open request -- is allowed
        to finish, so the transcript stays coherent. Cancellation stops the
        *next* request, not the current one.
        """
        self._cancelled = True

    async def _run_once(self) -> None:
        """One pi-equivalent run: `AGENT_START` to `AGENT_END`, one or more turns."""
        log = self.instance.log
        claimed = self.instance.inbox.claim(InboxTarget.NEXT_TURN, self.next_turn_policy)
        causes = [{"id": envelope.id, "origin": envelope.origin} for envelope in claimed]
        log.append(EventKind.AGENT_START, {"causes": causes})

        # The first step claims step input too, so steering queued before the
        # turn opened enters it rather than waiting for a second step.
        entering = tuple(envelope.message for envelope in claimed) + tuple(
            envelope.message for envelope in self._claim_step_input()
        )
        decision = await self._pre_step(entering, PreStepReason.INITIAL)
        if isinstance(decision, Reject):
            self._cancelled = False
            self._span(SpanKind.TURN, "turn", reason="rejected")
            log.append(
                EventKind.AGENT_END,
                {"reason": "rejected", "causes": causes, "detail": decision.reason},
            )
            return

        reason = PreStepReason.INITIAL
        steps = 0
        end_reason = "completed"

        while True:
            outcome = await self._run_step(decision, reason)
            steps += 1
            if outcome is None:
                break

            # The batch's own verdict, and a loop invariant inherited from pi:
            # evaluated before `agent/turn-stopping` is dispatched, so no
            # listener can override it (design spec section 6).
            if outcome.terminate:
                end_reason = "terminated"
                break

            if steps >= self.instance.definition.max_steps:
                end_reason = "max_steps"
                break

            if self._cancelled:
                end_reason = "cancelled"
                break

            # Only now is there a decision to make. Hard termination has
            # already been resolved above, so no listener can override it.
            if await self._should_stop():
                end_reason = "stopped"
                break

            step_input = self._claim_step_input()
            reason = PreStepReason.STEERING if step_input else PreStepReason.TOOL_RESULTS
            decision = await self._pre_step(
                tuple(envelope.message for envelope in step_input), reason
            )
            if isinstance(decision, Reject):
                end_reason = "rejected"
                break

        # Cleared with the run: a cancelled run must not poison the next.
        self._cancelled = False
        self._span(SpanKind.TURN, "turn", reason=end_reason)
        # Causes repeat at the end so a consumer reading only completions can
        # route a result without replaying the whole run.
        log.append(EventKind.AGENT_END, {"reason": end_reason, "causes": causes})

    async def _should_stop(self) -> bool:
        """Ask listeners whether to stop, folding by first-opinion-wins.

        Serial dispatch returns the last listener's value, so the fold is
        applied to that single opinion here; it earns its keep once Plan 4
        collects several.
        """
        decision = await self.instance.ctx.events.serial(
            AGENT_TURN_STOPPING, self.instance, scope=self.instance.scope.key
        )
        if decision is None:
            return False
        return resolve_stopping([decision]) is TurnStopping.STOP

    def _claim_step_input(self) -> tuple[InputEnvelope, ...]:
        """Take whatever is waiting at the step boundary."""
        return self.instance.inbox.claim(InboxTarget.NEXT_STEP, self.next_step_policy)

    async def _pre_step(
        self, messages: tuple[Message, ...], reason: PreStepReason
    ) -> PreStepDecision:
        """Ask listeners what should enter this step.

        The terminal continuation is `Enter(messages)`, so a chain whose
        listeners all delegate runs the step with exactly what was claimed. A
        listener that owns the decision returns without delegating; one that
        transforms delegates with replacement arguments, which the listeners
        after it receive.
        """
        decision: PreStepDecision = await self.instance.ctx.events.waterfall(
            AGENT_PRE_STEP,
            self.instance,
            reason,
            messages,
            terminal=Enter(messages=messages),
            scope=self.instance.scope.key,
        )
        return decision

    async def _run_step(self, decision: Enter, reason: PreStepReason) -> BatchOutcome | None:
        """Run one model request and its tools.

        Tools execute as a batch -- parallel by default, with pi's contagion
        rule serializing around an exclusive call. Returns the batch outcome,
        or `None` when the model called no tools -- which the caller reads as
        "nothing is owed".
        """
        log = self.instance.log

        # `reason` is Minion-internal bookkeeping (why this turn began -- initial/
        # steering/tool_results), not part of pi's own bare `turn_start` (no fields
        # at all) -- kept on the raw log entry, not projected into the public
        # `TurnStart` event.
        log.append(EventKind.TURN_START, {"reason": reason.value})

        # After `turn/start`, matching pinned Pi exactly (`runAgentLoop`/`runLoop`
        # emit `turn_start` before the entering messages, every turn, not only the
        # first): what entered the turn caused it, and the model cannot see a
        # prompt that is not yet on the surface when the request is derived.
        for message in decision.messages:
            log.append(EventKind.USER_MESSAGE, {"message": encode_message(message)})

        components = {
            "system_base": decision.system_override
            if decision.system_override is not None
            else self.instance.definition.system
        }
        schemas = self.tools.schemas(self.instance.scope)
        record_header(
            log,
            self.artifacts,
            components,
            model=self.instance.definition.model.model,
            tools=schemas,
        )

        history = derive_messages(log)
        if decision.history_window is not None:
            history = history[-decision.history_window :]

        request = Request(
            model=self.instance.definition.model,
            system=assemble_system(components),
            messages=history,
            tools=schemas,
        )

        def log_chunk(chunk: StreamChunk) -> None:
            """Record streaming fidelity. Log-only: a delta that derived would
            duplicate the message it is part of."""
            match chunk:
                case TextDelta():
                    kind, delta = "text", chunk.delta
                case ThinkingDelta():
                    kind, delta = "thinking", chunk.delta
                case ToolCallDelta():
                    kind, delta = "tool_call", chunk.delta
                case _:
                    return
            log.append(
                EventKind.ASSISTANT_CHUNK,
                {"kind": kind, "content_index": chunk.content_index, "delta": delta},
            )

        reply = await collect(self.llm.stream(request), log_chunk)
        log.append(EventKind.ASSISTANT_MESSAGE, {"message": encode_message(reply)})

        calls = [block for block in reply.content if isinstance(block, ToolCallBlock)]
        for call in calls:
            log.append(
                EventKind.TOOL_CALL,
                {"id": call.id, "name": call.name, "arguments": call.arguments},
            )

        if reply.stop_reason is StopReason.LENGTH and calls:
            # A length stop means the output was cut off by the token limit, so every
            # tool call the message carries may itself have truncated arguments. Pinned
            # Pi fails them all instead of executing potentially-truncated calls
            # (`failToolCallsFromTruncatedMessage`, TOOL-017) -- none reach the registry.
            outcome = await execute_length_stop_batch(
                calls, ctx=self.instance.ctx, scope=self.instance.scope
            )
        else:
            outcome = await execute_batch(
                calls,
                registry=self.tools,
                ctx=self.instance.ctx,
                scope=self.instance.scope,
            )
        for result in outcome.results:
            log.append(
                EventKind.TOOL_RESULT,
                {
                    "message": encode_message(result.to_message()),
                    # Pi emits `tool_execution_end` in completion order while
                    # messages follow source order. Results are appended in
                    # source order -- that is what the model reads -- so the
                    # completion order rides along as data for the projection.
                    "completion_index": outcome.completion_index(result.tool_call_id),
                    "added_tool_names": list(result.added_tool_names),
                },
            )

        log.append(EventKind.TURN_END, {})
        self._span(SpanKind.STEP, "step", reason=reason.value)
        return outcome if calls else None
