"""The concrete agent loop.

Imperative and package-internal, following DSH's `ReactLoopAgent`: a stateful
class with an explicit phase. Neither pi nor DSH factors the live loop as a
pure reducer, and this design does not invent one.

Pi vocabulary, adopted exactly (Layer 08): a **run** is one `prompt()`/
`continue()` invocation, bracketed by `agent_start`/`agent_end`. A **turn** is
one assistant response plus the tool calls/results it triggers, bracketed by
`turn_start`/`turn_end` -- never more than one provider request. Pi has no
"step" concept; `_run_step` is Minion's own internal helper name for "run one
turn," kept because the observable boundary it emits (one `TURN_START`/
`TURN_END` pair per call) is what matters, not the helper's own name.

PASS 2 completes the public entry points pi actually has: `prompt()`/
`continue_()` (Python cannot name a method `continue`, a reserved word), each
with pi's own exact caller-rejection text, a run-start config snapshot pi's
own `createContextSnapshot()` takes, `prepareNextTurn` (run-local
system/model/thinking_level override, never persisted back to
`AgentInstance`), mid-run follow-up continuation (the same run keeps going,
not a fresh `AGENT_START`/`AGENT_END` pair per queued batch), and pi's
`handleRunFailure` fallback for an unexpected listener/callback exception.
`run_until_idle()` -- Minion's own pump, with no pi equivalent -- is now a
thin driver over the same `_run_wrapped`/`_execute_run` primitives `prompt`/
`continue_` use, one call per claimed batch.
"""

from __future__ import annotations

from dataclasses import dataclass

from ..agent.decisions import (
    Enter,
    PreStepDecision,
    PreStepReason,
    Reject,
    RunConfigUpdate,
    TurnStopping,
    resolve_stopping,
)
from ..agent.envelope import ClaimPolicy, InboxTarget, InputEnvelope
from ..agent.events import AGENT_PRE_STEP, AGENT_PREPARE_NEXT_TURN, AGENT_TURN_STOPPING
from ..agent.identity import AgentStatus, ThinkingLevel
from ..agent.instance import AgentActiveError, AgentInstance
from ..llm import (
    AssistantMessage,
    LlmService,
    Message,
    ModelId,
    Request,
    StopReason,
    StreamChunk,
    TextBlock,
    TextDelta,
    ThinkingDelta,
    ToolCallBlock,
    ToolCallDelta,
    Usage,
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
from ..tools.events import TOOLS_EXECUTION_END, TOOLS_EXECUTION_START
from ..tools.registry import ToolRegistry


class _LoopCallbackFailure(Exception):
    """Wraps an unexpected exception from `AGENT_PRE_STEP`/`AGENT_TURN_STOPPING`/
    `AGENT_PREPARE_NEXT_TURN` listener dispatch -- pinned Pi's own `handleRunFailure`
    boundary, "unexpected loop callback failure." Deliberately narrow: model/
    provider/tool-execution failures are NOT wrapped here and propagate (or settle
    via their own existing, different mechanisms) normally -- an unresolvable model
    is a caller bug, reported eagerly, never smuggled into a settled failure turn
    (`eager-invalid-model-fails-before-stream.yaml`, section 4's boundary)."""


@dataclass
class _RunSnapshot:
    """Pi's per-run config snapshot (`Agent.createContextSnapshot()`):
    `system_prompt`/`model`/`thinking_level` read ONCE at run start, so a
    caller mutating the certified Layer-07 `AgentInstance` mid-run does not
    retroactively alter a run already in progress -- confirmed directly
    against pi (`Agent.createContextSnapshot()` shallow-copies once; the run
    loop's own local `config`/`currentContext` never re-reads `Agent._state`).
    `prepareNextTurn` may still update this run-locally via `apply()`; those
    updates are never written back to `AgentInstance`."""

    system_prompt: str
    model: ModelId
    thinking_level: ThinkingLevel

    def apply(self, update: RunConfigUpdate) -> None:
        if update.system_prompt is not None:
            self.system_prompt = update.system_prompt
        if update.model is not None:
            self.model = update.model
        if update.thinking_level is not None:
            self.thinking_level = update.thinking_level


class AgentLoop:
    """Drives one agent instance through runs, turns, and their steps."""

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

    async def prompt(self, message: Message | tuple[Message, ...]) -> None:
        """Pinned Pi's `Agent.prompt()`: start a new run with `message` as its
        own entering prompt. Rejects while a run is already active, with pi's
        exact text."""
        if self.instance.status is not AgentStatus.IDLE:
            raise AgentActiveError(
                "Agent is already processing a prompt. Use steer() or "
                "followUp() to queue messages, or wait for completion."
            )
        entering = message if isinstance(message, tuple) else (message,)
        await self._run_wrapped(entering=entering, causes=[])

    async def continue_(self) -> None:
        """Pinned Pi's `Agent.continue()` (renamed: `continue` is a Python
        keyword). Rejects while active or with an empty transcript, with pi's
        exact text for each. When the transcript's last message is
        assistant, drains eligible steering (skipping the run's own initial
        steering poll, so the same batch is never claimed twice) or,
        failing that, eligible follow-up; with neither queued, rejects.
        Otherwise runs a plain continuation: no entering messages, full
        history still sent."""
        if self.instance.status is not AgentStatus.IDLE:
            raise AgentActiveError(
                "Agent is already processing. Wait for completion before continuing."
            )
        messages = self.instance.messages
        if not messages:
            raise AgentActiveError("No messages to continue from")

        if isinstance(messages[-1], AssistantMessage):
            steering = self.instance.inbox.claim(InboxTarget.NEXT_STEP, self.next_step_policy)
            if steering:
                await self._run_wrapped(
                    entering=tuple(envelope.message for envelope in steering),
                    causes=[{"id": e.id, "origin": e.origin} for e in steering],
                    skip_initial_steering_poll=True,
                )
                return
            followups = self.instance.inbox.claim(InboxTarget.NEXT_TURN, self.next_turn_policy)
            if followups:
                await self._run_wrapped(
                    entering=tuple(envelope.message for envelope in followups),
                    causes=[{"id": e.id, "origin": e.origin} for e in followups],
                )
                return
            raise AgentActiveError("Cannot continue from message role: assistant")

        await self._run_wrapped(entering=(), causes=[])

    async def run_until_idle(self) -> None:
        """Minion's own pump (no pi equivalent, AG-019): open a run per
        claimed follow-up batch while input is pending, then settle idle.
        Each claimed batch gets its own `_run_wrapped` call -- its own
        `AGENT_START`/`AGENT_END` bracket and its own `is_streaming`
        true/false toggle, matching pi's own per-invocation lifecycle, not
        one shared bracket spanning every batch this pump happens to drain."""
        inbox = self.instance.inbox
        while inbox.pending(InboxTarget.NEXT_TURN):
            claimed = inbox.claim(InboxTarget.NEXT_TURN, self.next_turn_policy)
            causes: list[dict[str, object]] = [
                {"id": envelope.id, "origin": envelope.origin} for envelope in claimed
            ]
            await self._run_wrapped(
                entering=tuple(envelope.message for envelope in claimed), causes=causes
            )
        inbox.take_wake()

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

    async def _run_wrapped(
        self,
        *,
        entering: tuple[Message, ...],
        causes: list[dict[str, object]],
        skip_initial_steering_poll: bool = False,
    ) -> None:
        """One pi-equivalent run's full lifecycle: status/`error_message`
        reset at entry (pinned Pi's `runWithLifecycle`), guaranteed status
        settlement on exit, regardless of success, an internal defect, or a
        run-loop failure this pass already settles gracefully (see
        `_settle_run_failure`)."""
        if self.instance.status is not AgentStatus.IDLE:
            # Pinned Pi's own `runWithLifecycle` guard -- a third, distinct
            # "already processing" string, defensive and normally
            # unreachable through `prompt()`/`continue_()`'s own prior
            # checks (there is no `await` between those checks and this one
            # under normal single-caller use), kept for the same reason pi
            # keeps it: belt-and-suspenders against a future caller that
            # skips the public guards.
            raise AgentActiveError("Agent is already processing.")
        self.instance.set_status(AgentStatus.RUNNING)
        self.instance.error_message = None
        try:
            await self._execute_run(
                entering=entering,
                causes=causes,
                skip_initial_steering_poll=skip_initial_steering_poll,
            )
        finally:
            self.instance.set_status(AgentStatus.IDLE)

    async def _execute_run(
        self,
        *,
        entering: tuple[Message, ...],
        causes: list[dict[str, object]],
        skip_initial_steering_poll: bool,
    ) -> None:
        """One pi-equivalent run: `AGENT_START` to `AGENT_END`, one or more
        turns, continuing across follow-up-triggered continuations within
        the same bracket (pi's own outer `runLoop` loop -- Minion's own pump
        starts a *new* run per queued batch only when this run has already
        ended, not while it is still open)."""
        log = self.instance.log
        log.append(EventKind.AGENT_START, {"causes": causes})
        snapshot = _RunSnapshot(
            system_prompt=self.instance.system_prompt,
            model=self.instance.model,
            thinking_level=self.instance.thinking_level,
        )
        try:
            end_reason, detail = await self._run_inner(
                entering, causes, skip_initial_steering_poll, snapshot
            )
        except _LoopCallbackFailure as failure:
            end_reason, detail = self._settle_run_failure(failure.__cause__ or failure, snapshot)

        self._cancelled = False
        payload: dict[str, object] = {"reason": end_reason, "causes": causes}
        if detail is not None:
            payload["detail"] = detail
        log.append(EventKind.AGENT_END, payload)

    async def _run_inner(
        self,
        entering: tuple[Message, ...],
        causes: list[dict[str, object]],
        skip_initial_steering_poll: bool,
        snapshot: _RunSnapshot,
    ) -> tuple[str, str | None]:
        if skip_initial_steering_poll:
            initial_step_input: tuple[InputEnvelope, ...] = ()
        else:
            initial_step_input = self._claim_step_input()
        pending = entering + tuple(envelope.message for envelope in initial_step_input)
        decision = await self._pre_step(pending, PreStepReason.INITIAL)
        if isinstance(decision, Reject):
            self._span(SpanKind.TURN, "turn", reason="rejected")
            return "rejected", decision.reason

        reason = PreStepReason.INITIAL
        steps = 0
        last_terminated = False

        while True:  # outer: continues across follow-up-triggered continuations
            while True:  # inner: turns within one continuation
                outcome = await self._run_step(decision, reason, snapshot)
                steps += 1
                last_terminated = outcome is not None and outcome.terminate
                has_more_tool_calls = outcome is not None and not outcome.terminate

                if steps >= self.instance.definition.max_steps:
                    self._span(SpanKind.TURN, "turn", reason="max_steps")
                    return "max_steps", None
                if self._cancelled:
                    self._span(SpanKind.TURN, "turn", reason="cancelled")
                    return "cancelled", None

                # Always dispatched, including after a tool batch's `terminate`
                # verdict (`L08-PASS2`): `terminate` suppresses only the
                # tool-driven inner-loop continuation (`has_more_tool_calls`
                # above), matching pinned Pi exactly -- it must not also skip
                # `prepareNextTurn`/`shouldStopAfterTurn`/the steering poll,
                # which pi's own `runLoop` still runs for that same turn.
                snapshot.apply(await self._prepare_next_turn(outcome))

                if await self._should_stop():
                    self._span(SpanKind.TURN, "turn", reason="stopped")
                    return "stopped", None

                step_input = self._claim_step_input()
                if not has_more_tool_calls and not step_input:
                    break  # inner loop exhausted -> poll follow-up below

                reason = PreStepReason.STEERING if step_input else PreStepReason.TOOL_RESULTS
                decision = await self._pre_step(
                    tuple(envelope.message for envelope in step_input), reason
                )
                if isinstance(decision, Reject):
                    self._span(SpanKind.TURN, "turn", reason="rejected")
                    return "rejected", decision.reason

            followups = self.instance.inbox.claim(InboxTarget.NEXT_TURN, self.next_turn_policy)
            if not followups:
                end_reason = "terminated" if last_terminated else "completed"
                self._span(SpanKind.TURN, "turn", reason=end_reason)
                return end_reason, None

            causes.extend({"id": envelope.id, "origin": envelope.origin} for envelope in followups)
            reason = PreStepReason.NEXT_TURN
            decision = await self._pre_step(
                tuple(envelope.message for envelope in followups), reason
            )
            if isinstance(decision, Reject):
                self._span(SpanKind.TURN, "turn", reason="rejected")
                return "rejected", decision.reason

    def _settle_run_failure(
        self, error: BaseException, snapshot: _RunSnapshot
    ) -> tuple[str, None]:
        """Pinned Pi's `handleRunFailure`: an unexpected exception from
        run-loop callback/listener code (not a normal model/tool failure,
        both of which already settle as ordinary assistant/tool-result
        content) still produces a coherent, terminal turn -- never an
        unhandled exception escaping `prompt()`/`continue_()`/
        `run_until_idle()`. A matched `TURN_START`/`TURN_END` pair brackets
        the synthesized failure message -- a small, disclosed simplification
        of pinned Pi's own bare `turn_end` (no preceding `turn_start` at
        all), chosen so the projection's own turn-scoped accumulator never
        has to special-case an unmatched `TURN_END`."""
        log = self.instance.log
        log.append(EventKind.TURN_START, {"reason": "failure"})
        failure = AssistantMessage(
            content=(TextBlock(text=""),),
            stop_reason=StopReason.ERROR,
            usage=Usage(),
            model=snapshot.model.model,
            provider=snapshot.model.provider,
            timestamp=0,
            error_message=str(error),
        )
        log.append(EventKind.ASSISTANT_MESSAGE, {"message": encode_message(failure)})
        log.append(EventKind.TURN_END, {})
        self.instance.error_message = str(error)
        self.instance.streaming_message = None
        self.instance.pending_tool_calls = frozenset()
        return "failed", None

    async def _should_stop(self) -> bool:
        """Ask listeners whether to stop, folding by first-opinion-wins.

        Serial dispatch returns the last listener's value, so the fold is
        applied to that single opinion here; it earns its keep once Plan 4
        collects several.
        """
        try:
            decision = await self.instance.ctx.events.serial(
                AGENT_TURN_STOPPING, self.instance, scope=self.instance.scope.key
            )
        except Exception as error:
            raise _LoopCallbackFailure() from error
        if decision is None:
            return False
        return resolve_stopping([decision]) is TurnStopping.STOP

    async def _prepare_next_turn(self, outcome: BatchOutcome | None) -> RunConfigUpdate:
        """Pinned Pi's `prepareNextTurn`: a run-local override for the next
        provider request's `system_prompt`/`model`/`thinking_level` only."""
        try:
            update: RunConfigUpdate = await self.instance.ctx.events.waterfall(
                AGENT_PREPARE_NEXT_TURN,
                self.instance,
                outcome,
                terminal=RunConfigUpdate(),
                scope=self.instance.scope.key,
            )
        except Exception as error:
            raise _LoopCallbackFailure() from error
        return update

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
        try:
            decision: PreStepDecision = await self.instance.ctx.events.waterfall(
                AGENT_PRE_STEP,
                self.instance,
                reason,
                messages,
                terminal=Enter(messages=messages),
                scope=self.instance.scope.key,
            )
        except Exception as error:
            raise _LoopCallbackFailure() from error
        return decision

    async def _run_step(
        self, decision: Enter, reason: PreStepReason, snapshot: _RunSnapshot
    ) -> BatchOutcome | None:
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
            else snapshot.system_prompt
        }
        schemas = self.tools.schemas(self.instance.scope)
        record_header(
            log,
            self.artifacts,
            components,
            model=snapshot.model.model,
            tools=schemas,
        )

        history = derive_messages(log)
        if decision.history_window is not None:
            history = history[-decision.history_window :]

        request = Request(
            model=snapshot.model,
            system=assemble_system(components),
            messages=history,
            tools=schemas,
        )

        # Non-`None` for exactly the duration of one provider request, matching
        # pinned Pi's own `streamingMessage` write points (`message_start`/
        # `message_update` -> the partial message; `message_end` -> `undefined`).
        # Content-level fidelity is text-only (accumulated `TextDelta`s): pi's own
        # partial reconstruction is opaque to this layer, and Minion's certified
        # Layer-02/04 `collect()` exposes only raw deltas, not a live partial
        # message object, to build a richer one from -- a disclosed simplification,
        # not a silently dropped requirement (AG-008).
        partial_text: dict[int, str] = {}
        self.instance.streaming_message = AssistantMessage(
            content=(),
            stop_reason=StopReason.PENDING,
            usage=Usage(),
            model=snapshot.model.model,
            provider=snapshot.model.provider,
            timestamp=0,
        )

        def log_chunk(chunk: StreamChunk) -> None:
            """Record streaming fidelity. Log-only: a delta that derived would
            duplicate the message it is part of."""
            match chunk:
                case TextDelta():
                    kind, delta = "text", chunk.delta
                    partial_text[chunk.content_index] = (
                        partial_text.get(chunk.content_index, "") + chunk.delta
                    )
                    self.instance.streaming_message = AssistantMessage(
                        content=tuple(TextBlock(text=text) for text in partial_text.values()),
                        stop_reason=StopReason.PENDING,
                        usage=Usage(),
                        model=snapshot.model.model,
                        provider=snapshot.model.provider,
                        timestamp=0,
                    )
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
        self.instance.streaming_message = None
        log.append(EventKind.ASSISTANT_MESSAGE, {"message": encode_message(reply)})
        if reply.error_message:
            # Pinned Pi's own `processEvents`: `turn_end` sets `errorMessage` from
            # a failed/aborted assistant message's own `errorMessage`, and nothing
            # clears it again until the *next* run starts (`_run_wrapped`) or
            # `reset()` (Layer-07, certified) -- it is not cleared here, and not
            # cleared by this run's own `agent_end` either.
            self.instance.error_message = reply.error_message

        calls = [block for block in reply.content if isinstance(block, ToolCallBlock)]
        for call in calls:
            log.append(
                EventKind.TOOL_CALL,
                {"id": call.id, "name": call.name, "arguments": call.arguments},
            )

        # `pending_tool_calls` tracks the real batch window via the already-
        # certified Layer-06 `tools/execution-start`/`tools/execution-end` events
        # (`AG-008`) -- add on start, remove on end, matching pinned Pi's own
        # `processEvents` reducer exactly, through an existing seam rather than a
        # new one.
        def on_execution_start(call_id: str, name: str, arguments: dict[str, object]) -> None:
            self.instance.pending_tool_calls = self.instance.pending_tool_calls | {call_id}

        def on_execution_end(call_id: str, name: str, result: object) -> None:
            self.instance.pending_tool_calls = self.instance.pending_tool_calls - {call_id}

        dispose_start = self.instance.ctx.events.on(
            TOOLS_EXECUTION_START, on_execution_start, scope=self.instance.scope.key
        )
        dispose_end = self.instance.ctx.events.on(
            TOOLS_EXECUTION_END, on_execution_end, scope=self.instance.scope.key
        )
        try:
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
        finally:
            dispose_start()
            dispose_end()
            self.instance.pending_tool_calls = frozenset()
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
