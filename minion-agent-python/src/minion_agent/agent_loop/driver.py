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

PASS 4 remediates an independent Rust re-review rejection (L08-R002, R004,
R005, R006, R009) against the PASS-3 candidate:

- a live, listener-bearing Agent-event seam (`AGENT_LIFECYCLE_EVENT`)
  matching pinned Pi's own `processEvents`, used identically by ordinary
  turn/run progress and by `handleRunFailure` recovery -- a recovery listener
  failure now genuinely interrupts the remaining recovery sequence and
  propagates, instead of the run settling gracefully via raw log appends
  alone (`L08-R002`);
- the initial prompt's own message lifecycle is now admitted and dispatched
  BEFORE the initial steering queue is ever claimed, not merely after
  `TURN_START` (`L08-R006`);
- `max_steps` removed entirely from the Pi-equivalent `prompt()`/
  `continue_()` run seam -- it no longer exists as a concept at this layer at
  all (`L08-R005`);
- the pre-existing local boundary-stop latch, formerly named `cancel()`,
  renamed to `request_boundary_stop()` and explicitly disposed as a
  Minion-only host extension distinct from pinned Pi's own `Agent.abort()`
  (deferred to Layer 09), never presented as Pi cancellation (`L08-R009`).

`spec/agent.md`'s own Layer-08 section is the normative contract this file
implements; PASS 4 rewrote it as one coherent current statement (`L08-R004`)
rather than leaving it PASS-2 text describing a since-superseded design.
"""

from __future__ import annotations

import time
from dataclasses import dataclass

from ..agent.decisions import (
    Enter,
    PreStepDecision,
    PreStepReason,
    Reject,
    RunConfig,
    RunConfigUpdate,
    RunContext,
    TurnStopping,
    resolve_stopping,
)
from ..agent.envelope import ClaimPolicy, InboxTarget, InputEnvelope
from ..agent.events import (
    AGENT_LIFECYCLE_EVENT,
    AGENT_PRE_STEP,
    AGENT_PREPARE_NEXT_TURN,
    AGENT_TURN_STOPPING,
)
from ..agent.identity import AgentStatus
from ..agent.instance import AgentActiveError, AgentInstance
from ..agent.projection import (
    AgentEnd,
    AgentEvent,
    AgentStart,
    MessageEnd,
    MessageStart,
    TurnEnd,
    TurnStart,
)
from ..llm import (
    AssistantMessage,
    ImageBlock,
    LlmService,
    Message,
    Request,
    StopReason,
    StreamChunk,
    StreamDone,
    StreamError,
    StreamStart,
    TextBlock,
    TextDelta,
    TextEnd,
    TextStart,
    ThinkingDelta,
    ThinkingEnd,
    ThinkingStart,
    ToolCallBlock,
    ToolCallDelta,
    ToolCallEnd,
    ToolCallStart,
    UnknownModelError,
    Usage,
    UserContentBlock,
    UserMessage,
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
from ..tools.definition import ToolDefinition
from ..tools.events import TOOLS_EXECUTION_END, TOOLS_EXECUTION_START
from ..tools.registry import ToolRegistry


def _snapshot_tool_registry(tools: tuple[ToolDefinition, ...]) -> ToolRegistry:
    """A fresh, unscoped `ToolRegistry` holding exactly `tools` -- the
    execution-time counterpart of `RunContext.tools` (`L08-R001`). Pinned
    Pi's own tool-call lookup (`prepareToolCall`) resolves against
    `currentContext.tools`, the snapshot, not a live registry; registering
    each definition with `scope=None` (visible from every scope,
    `ScopedRegistry.visible_from`'s own untagged fallback) reproduces that
    through the existing, unmodified Layer-05/06 `ToolRegistry`/
    `execute_batch` seam -- no lower-layer reopen."""
    registry = ToolRegistry()
    for definition in tools:
        registry.register(definition)
    return registry


@dataclass(frozen=True, slots=True)
class _StepResult:
    """One turn's outcome, threaded back to `_run_inner` for the post-turn
    decision sequence and, when terminal, the immediate-return path
    (`L08-R008`)."""

    message: AssistantMessage
    tool_results: tuple[Message, ...]
    has_more_tool_calls: bool
    hard_terminated: bool
    terminal: bool
    """True for a represented `error`/`aborted` assistant message: pinned Pi
    returns immediately after this turn's own `turn_end`, running none of
    `prepareNextTurn`/`shouldStopAfterTurn`/the steering or follow-up poll."""


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
        self._boundary_stop_requested = False

    async def prompt(
        self,
        message: Message | tuple[Message, ...] | str,
        images: tuple[ImageBlock, ...] = (),
    ) -> None:
        """Pinned Pi's `Agent.prompt()`: start a new run with `message` as its
        own entering prompt. Rejects with pi's exact text while a run is
        already active.

        Two accepted forms, neither narrowing the other (`L08-R007`): the
        typed `Message | tuple[Message, ...]` boundary (unchanged), and pi's
        own convenience overload, a plain `str` (optionally with `images`) --
        normalized into exactly one `UserMessage` whose content is the text
        followed by the supplied images, in that order, matching pinned Pi's
        `[{type:"text",...}, ...images]` construction exactly."""
        if self.instance.status is not AgentStatus.IDLE:
            raise AgentActiveError(
                "Agent is already processing a prompt. Use steer() or "
                "followUp() to queue messages, or wait for completion."
            )
        if isinstance(message, str):
            content: tuple[UserContentBlock, ...] = (TextBlock(text=message), *images)
            entering: tuple[Message, ...] = (
                UserMessage(content=content, timestamp=int(time.time() * 1000)),
            )
        elif isinstance(message, tuple):
            entering = message
        else:
            entering = (message,)
        await self._run_wrapped(entering=entering, causes=[])

    async def continue_(self) -> None:
        """Pinned Pi's `Agent.continue()` (renamed: `continue` is a Python
        keyword). Rejects with its own, different exact text while active or
        with an empty transcript. When the transcript's last message is
        assistant: drains eligible steering (skipping this run's own initial
        steering poll, so the same batch is never claimed twice) or, failing
        that, eligible follow-up; with neither queued, rejects. Otherwise runs
        a plain continuation: no entering messages, full history still sent.
        """
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

    def request_boundary_stop(self) -> None:
        """Request that the current run's next turn boundary end the run.

        This is a Minion-only host/pump extension (`AG-022`; `L08-R009`),
        NOT pinned Pi's `Agent.abort()`: it does not signal the running
        provider, tools, or hooks -- work already in flight (a running tool,
        an open request) is allowed to finish, so the transcript stays
        coherent, and only the *next* request is stopped. Real active
        cancellation -- provider/tool/hook signal propagation -- remains
        Layer 09's own territory (`AG-007`), unimplemented here. Renamed from
        `cancel()` (PASS 3 and earlier) specifically so its name does not
        imply Pi's own cancellation surface, which this is not.
        """
        self._boundary_stop_requested = True

    async def _run_wrapped(
        self,
        *,
        entering: tuple[Message, ...],
        causes: list[dict[str, object]],
        skip_initial_steering_poll: bool = False,
    ) -> None:
        """One pi-equivalent run's full lifecycle: status/`error_message`
        reset at entry (pinned Pi's `runWithLifecycle`), guaranteed status
        settlement on exit, regardless of success or a run-executor failure
        this pass already settles gracefully (see `_settle_run_failure`)."""
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

    async def _dispatch_agent_event(self, event: AgentEvent) -> None:
        """Pinned Pi's own `processEvents`: the single seam every lifecycle
        event -- ordinary turn/run progress and `handleRunFailure` recovery
        alike -- passes through, awaiting every subscribed listener in
        registration order (`L08-R002`). `EventBus.serial` already matches
        pinned Pi's own raw `for (const listener of this.listeners) await
        listener(event, signal)` loop exactly: a listener that throws aborts
        the remaining dispatch for THIS event and propagates -- no new
        dispatch primitive was needed. Callers append the corresponding log
        entry (the durable "reduce" step) before calling this (the "notify
        listeners" step), matching pinned Pi's own reduce-then-dispatch
        order."""
        await self.instance.ctx.events.serial(
            AGENT_LIFECYCLE_EVENT, self.instance, event, scope=self.instance.scope.key
        )

    async def _admit_messages(
        self, messages: tuple[Message, ...], context: RunContext, new_messages: list[Message]
    ) -> None:
        """Log, dispatch, and accumulate one admitted message batch --
        pinned Pi's own unconditional message lifecycle for whatever enters
        a turn (the initial prompt, claimed steering, tool-result-triggered
        continuation input, or a follow-up-triggered continuation's own
        entering batch). Every admission point in `_run_inner` calls this
        the same way, so the initial prompt's own lifecycle is genuinely
        complete -- logged AND dispatched to any live listener -- before the
        very next thing `_run_inner` does (`L08-R006`). Appended to the
        run-local `context`/`new_messages` accumulators as it goes, matching
        pinned Pi's own `currentContext.messages.push(message)`/
        `newMessages.push(message)` timing exactly (`L08-R001`)."""
        log = self.instance.log
        for message in messages:
            log.append(EventKind.USER_MESSAGE, {"message": encode_message(message)})
            await self._dispatch_agent_event(MessageStart(message=message))
            await self._dispatch_agent_event(MessageEnd(message=message))
            context.messages.append(message)
            new_messages.append(message)

    async def _execute_run(
        self,
        *,
        entering: tuple[Message, ...],
        causes: list[dict[str, object]],
        skip_initial_steering_poll: bool,
    ) -> None:
        """One pi-equivalent run: `AGENT_START` to `AGENT_END`, one or more
        turns, continuing across follow-up-triggered continuations within
        the same bracket (pi's own outer `runLoop` loop).

        `context`/`config` are pinned Pi's own `createContextSnapshot()`/
        `createLoopConfig()`: taken once, here, at run start (`L08-R001`) --
        `_run_step` never re-reads `AgentInstance`/Session/`ToolRegistry`
        after this point; only `prepareNextTurn` may replace them, run-locally.

        The success path's own `AGENT_END` append/dispatch lives INSIDE this
        method's `try` block, sharing the same `except Exception` boundary as
        `_run_inner` itself (`L08-R002`): a listener that throws while this
        run's own ordinary `agent_end` is being delivered is, per pinned
        Pi's own architecture, indistinguishable from any other run-executor
        failure -- `runWithLifecycle`'s catch does not know or care WHERE in
        `runAgentLoop` the exception came from -- so it is caught here too
        and converted into a settled failure turn via `_settle_run_failure`,
        never silently swallowed and never left as a half-delivered success.
        """
        log = self.instance.log
        log.append(EventKind.AGENT_START, {"causes": causes})
        await self._dispatch_agent_event(AgentStart(causes=tuple(causes)))
        context = RunContext(
            system_prompt=self.instance.system_prompt,
            messages=list(derive_messages(self.instance.log)),
            tools=self.tools.visible_from(self.instance.scope),
        )
        config = RunConfig(model=self.instance.model, thinking_level=self.instance.thinking_level)
        new_messages: list[Message] = []

        try:
            end_reason, detail = await self._run_inner(
                entering, causes, skip_initial_steering_poll, context, config, new_messages
            )
        except UnknownModelError:
            # Pinned Pi's own boundary: resolving a model happens before a
            # stream is returned, so an unresolvable one is a caller/config
            # bug, reported eagerly -- never smuggled into a settled failure
            # turn (L08-R002, `eager-invalid-model-fails-before-stream.yaml`).
            raise
        except Exception as error:
            self._boundary_stop_requested = False
            # Propagates uncaught if a recovery listener itself throws
            # (`L08-R002`) -- pinned Pi's own `handleRunFailure` has no
            # further catch around it either; `_run_wrapped`'s own `finally`
            # still settles status, matching pinned Pi's `finishRun`.
            await self._settle_run_failure(error, config, causes)
            return

        self._boundary_stop_requested = False
        payload: dict[str, object] = {"reason": end_reason, "causes": causes}
        if detail is not None:
            payload["detail"] = detail
        log.append(EventKind.AGENT_END, payload)
        await self._dispatch_agent_event(
            AgentEnd(reason=end_reason, causes=tuple(causes), messages=tuple(new_messages))
        )

    async def _run_inner(
        self,
        entering: tuple[Message, ...],
        causes: list[dict[str, object]],
        skip_initial_steering_poll: bool,
        context: RunContext,
        config: RunConfig,
        new_messages: list[Message],
    ) -> tuple[str, str | None]:
        log = self.instance.log

        # Pi's runAgentLoop/runAgentLoopContinue: agent_start (already
        # appended by _execute_run), turn_start, THEN the entering (prompt)
        # messages' own complete message lifecycle -- admitted and dispatched
        # here, before anything else -- and ONLY THEN, for this very first
        # turn, the initial steering poll (runLoop's own first statement,
        # called only after runAgentLoop has already emitted turn_start and
        # the prompt messages). Subsequent turns' steering claims already
        # happen correctly, as part of the PRECEDING turn's own post-turn
        # sequence, before the next TURN_START -- only the very first poll
        # needed this staged admission (`L08-R006`; an earlier revision
        # claimed steering before the prompt's own lifecycle was admitted at
        # all, which a listener present at claim time could observe).
        log.append(EventKind.TURN_START, {"reason": PreStepReason.INITIAL.value})
        await self._dispatch_agent_event(TurnStart())

        decision = await self._pre_step(entering, PreStepReason.INITIAL)
        if isinstance(decision, Reject):
            self._span(SpanKind.TURN, "turn", reason="rejected")
            return "rejected", decision.reason
        await self._admit_messages(decision.messages, context, new_messages)

        if skip_initial_steering_poll:
            initial_step_input: tuple[InputEnvelope, ...] = ()
        else:
            initial_step_input = self._claim_step_input()
        if initial_step_input:
            steering_decision = await self._pre_step(
                tuple(envelope.message for envelope in initial_step_input), PreStepReason.STEERING
            )
            if isinstance(steering_decision, Reject):
                self._span(SpanKind.TURN, "turn", reason="rejected")
                return "rejected", steering_decision.reason
            await self._admit_messages(steering_decision.messages, context, new_messages)
            # The request-shaping decision (system_override/history_window)
            # for this first turn: the latest-resolved admission wins, same
            # as every later turn only ever has one decision in play at all.
            decision = steering_decision

        reason = PreStepReason.INITIAL
        steps = 0
        last_terminated = False
        open_turn = False  # this first turn's TURN_START was already appended above

        while True:  # outer: continues across follow-up-triggered continuations
            while True:  # inner: turns within one continuation
                result = await self._run_step(
                    decision, reason, context, config, new_messages, open_turn=open_turn
                )
                open_turn = True
                steps += 1

                if result.terminal:
                    # Represented error/aborted (L08-R008): pinned Pi returns
                    # immediately after this turn's own turn_end, running
                    # none of prepareNextTurn/shouldStopAfterTurn/steering/
                    # follow-up for it.
                    self._span(SpanKind.TURN, "turn", reason=result.message.stop_reason.value)
                    return result.message.stop_reason.value, None

                last_terminated = result.hard_terminated
                has_more_tool_calls = result.has_more_tool_calls

                # Dispatched after every turn, including one a tool batch's
                # `terminate` verdict ended: `terminate` suppresses only the
                # tool-driven inner-loop continuation (`has_more_tool_calls`
                # above), matching pinned Pi exactly.
                update = await self._prepare_next_turn(
                    result.message, result.tool_results, context, tuple(new_messages)
                )
                if update.context is not None:
                    context = update.context
                if update.model is not None:
                    config.model = update.model
                if update.thinking_level is not None:
                    config.thinking_level = update.thinking_level

                if await self._should_stop(
                    result.message, result.tool_results, context, tuple(new_messages)
                ):
                    self._span(SpanKind.TURN, "turn", reason="stopped")
                    return "stopped", None

                step_input = self._claim_step_input()
                if not has_more_tool_calls and not step_input:
                    break  # inner loop exhausted -> poll follow-up below

                if self._boundary_stop_requested:
                    # A Minion-only host extension (`AG-022`, no pi
                    # equivalent -- `request_boundary_stop()`, `L08-R009`),
                    # gated the same way pinned Pi's own post-turn ordering
                    # is never skipped for a turn that already happened: only
                    # after this turn's own prepareNextTurn/
                    # shouldStopAfterTurn/steering poll have already run
                    # unconditionally. Work already in flight finishes; only
                    # the *next* request is stopped.
                    self._span(SpanKind.TURN, "turn", reason="boundary_stop")
                    return "boundary_stop", None

                reason = PreStepReason.STEERING if step_input else PreStepReason.TOOL_RESULTS
                decision = await self._pre_step(
                    tuple(envelope.message for envelope in step_input), reason
                )
                if isinstance(decision, Reject):
                    self._span(SpanKind.TURN, "turn", reason="rejected")
                    return "rejected", decision.reason
                await self._admit_messages(decision.messages, context, new_messages)

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
            await self._admit_messages(decision.messages, context, new_messages)

    async def _settle_run_failure(
        self, error: BaseException, config: RunConfig, causes: list[dict[str, object]]
    ) -> None:
        """Pinned Pi's `handleRunFailure`: an unexpected exception from the
        run executor (any of it -- listener dispatch, an adapter breaking its
        streaming contract, or any other unforeseen failure inside a turn;
        NOT the eager, pre-stream `UnknownModelError` `_execute_run` already
        excludes) still produces a coherent, terminal turn -- UNLESS a
        listener invoked during this very recovery sequence itself throws, in
        which case pinned Pi's own bare sequential `await
        this.processEvents(...)` chain aborts at that exact point and the
        exception propagates uncaught, exactly reproduced here by awaiting
        each dispatch in turn with nothing catching a failure from it
        (`L08-R002`).

        Emits exactly pinned Pi's own bare sequence -- `message_start`/
        `message_end`/`turn_end(failure, [])`/`agent_end(messages=[failure])`
        -- through the SAME live `AGENT_LIFECYCLE_EVENT` seam ordinary
        turn/run progress uses, not a recovery-only special path, and with NO
        preceding `turn_start` at all. `TURN_END` still carries an explicit
        override the offline `project()` honors directly (see
        `projection.py`) for callers reconstructing the log after the fact."""
        log = self.instance.log
        failure = AssistantMessage(
            content=(TextBlock(text=""),),
            stop_reason=StopReason.ERROR,
            usage=Usage(),
            model=config.model.model,
            provider=config.model.provider,
            timestamp=0,
            error_message=str(error),
        )
        log.append(EventKind.ASSISTANT_MESSAGE, {"message": encode_message(failure)})
        await self._dispatch_agent_event(MessageStart(message=failure))
        await self._dispatch_agent_event(MessageEnd(message=failure))
        self.instance.streaming_message = None

        log.append(EventKind.TURN_END, {"message": encode_message(failure)})
        await self._dispatch_agent_event(TurnEnd(message=failure, tool_results=()))
        self.instance.error_message = str(error)
        self.instance.pending_tool_calls = frozenset()

        payload: dict[str, object] = {
            "reason": "failed",
            "causes": causes,
            "messages": [encode_message(failure)],
        }
        log.append(EventKind.AGENT_END, payload)
        await self._dispatch_agent_event(
            AgentEnd(reason="failed", causes=tuple(causes), messages=(failure,))
        )

    async def _should_stop(
        self,
        message: AssistantMessage,
        tool_results: tuple[Message, ...],
        context: RunContext,
        new_messages: tuple[Message, ...],
    ) -> bool:
        """Ask listeners whether to stop, folding by first-opinion-wins.

        Serial dispatch returns the last listener's value, so the fold is
        applied to that single opinion here; it earns its keep once Plan 4
        collects several. `message`/`tool_results`/`context`/`new_messages`
        mirror pinned Pi's own `ShouldStopAfterTurnContext` exactly
        (`L08-R001`) -- an earlier revision gave listeners no context at all.
        """
        decision = await self.instance.ctx.events.serial(
            AGENT_TURN_STOPPING,
            self.instance,
            message,
            tool_results,
            context,
            new_messages,
            scope=self.instance.scope.key,
        )
        if decision is None:
            return False
        return resolve_stopping([decision]) is TurnStopping.STOP

    async def _prepare_next_turn(
        self,
        message: AssistantMessage,
        tool_results: tuple[Message, ...],
        context: RunContext,
        new_messages: tuple[Message, ...],
    ) -> RunConfigUpdate:
        """Pinned Pi's `prepareNextTurn`: a run-local override for the next
        provider request's whole `context`/`model`/`thinking_level`.
        `message`/`tool_results`/`context`/`new_messages` mirror pinned Pi's
        own `PrepareNextTurnContext` exactly (`L08-R001`)."""
        update: RunConfigUpdate = await self.instance.ctx.events.waterfall(
            AGENT_PREPARE_NEXT_TURN,
            self.instance,
            message,
            tool_results,
            context,
            new_messages,
            terminal=RunConfigUpdate(),
            scope=self.instance.scope.key,
        )
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
        decision: PreStepDecision = await self.instance.ctx.events.waterfall(
            AGENT_PRE_STEP,
            self.instance,
            reason,
            messages,
            terminal=Enter(messages=messages),
            scope=self.instance.scope.key,
        )
        return decision

    async def _run_step(
        self,
        decision: Enter,
        reason: PreStepReason,
        context: RunContext,
        config: RunConfig,
        new_messages: list[Message],
        *,
        open_turn: bool = True,
    ) -> _StepResult:
        """Run one model request and its tools.

        Tools execute as a batch -- parallel by default, with pi's contagion
        rule serializing around an exclusive call. `context`/`config` are the
        run's own snapshot (`L08-R001`) -- never re-read from `AgentInstance`/
        Session/`ToolRegistry` here. `decision.messages` are NOT admitted
        here: every call site admits (logs, dispatches, and accumulates) its
        own turn's entering messages via `_admit_messages` before calling
        this method (`L08-R006`) -- only `decision.system_override`/
        `.history_window` are still read here, to shape the request itself.
        """
        log = self.instance.log

        # `reason` is Minion-internal bookkeeping (why this turn began -- initial/
        # steering/tool_results), not part of pi's own bare `turn_start` (no fields
        # at all) -- kept on the raw log entry, not projected into the public
        # `TurnStart` event. `open_turn=False` only for this run's very first
        # turn, whose TURN_START `_run_inner` already appended (`L08-R006`).
        if open_turn:
            log.append(EventKind.TURN_START, {"reason": reason.value})
            await self._dispatch_agent_event(TurnStart())

        components = {
            "system_base": decision.system_override
            if decision.system_override is not None
            else context.system_prompt
        }
        schemas = tuple(definition.schema() for definition in context.tools)
        record_header(
            log,
            self.artifacts,
            components,
            model=config.model.model,
            tools=schemas,
        )

        history: list[Message] | tuple[Message, ...] = context.messages
        if decision.history_window is not None:
            history = history[-decision.history_window :]

        request = Request(
            model=config.model,
            system=assemble_system(components),
            messages=tuple(history),
            tools=schemas,
        )

        # Non-`None` for exactly the duration of one provider request, matching
        # pinned Pi's own `streamingMessage` write points (`message_start`/
        # `message_update` -> the partial message; `message_end` -> `undefined`).
        # Set directly from each chunk's own `partial` (`L08-R003`): the
        # already-certified Layer-02/04 `StreamChunk` carries the complete
        # partial assistant message on every variant, so no independent
        # reconstruction from raw deltas is needed or attempted. Not
        # dispatched through `AGENT_LIFECYCLE_EVENT` (`L08-R002`): `collect()`
        # accepts only a synchronous `on_chunk` callback, so a chunk-level
        # listener has nothing to `await` -- log + offline `project()` remain
        # this reply's own observability surface, unchanged from PASS 3.
        stream_opened = False

        def log_chunk(chunk: StreamChunk) -> None:
            nonlocal stream_opened
            match chunk:
                case StreamStart():
                    stream_opened = True
                    self.instance.streaming_message = chunk.partial
                    log.append(
                        EventKind.ASSISTANT_STREAM_START,
                        {"partial": encode_message(chunk.partial)},
                    )
                    return
                case TextStart():
                    kind = "text_start"
                case TextDelta():
                    kind = "text_delta"
                case TextEnd():
                    kind = "text_end"
                case ThinkingStart():
                    kind = "thinking_start"
                case ThinkingDelta():
                    kind = "thinking_delta"
                case ThinkingEnd():
                    kind = "thinking_end"
                case ToolCallStart():
                    kind = "toolcall_start"
                case ToolCallDelta():
                    kind = "toolcall_delta"
                case ToolCallEnd():
                    kind = "toolcall_end"
                case StreamDone() | StreamError():
                    return
            self.instance.streaming_message = chunk.partial
            log.append(
                EventKind.ASSISTANT_CHUNK,
                {
                    "kind": kind,
                    "content_index": chunk.content_index,
                    "partial": encode_message(chunk.partial),
                },
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
        context.messages.append(reply)
        new_messages.append(reply)

        if reply.stop_reason in (StopReason.ERROR, StopReason.ABORTED):
            # Represented error/aborted (`L08-R008`): pinned Pi emits this
            # turn's own `turn_end` with empty `toolResults` and returns
            # immediately -- no tool calls are ever inspected or executed for
            # a message representing its own failure.
            log.append(EventKind.TURN_END, {})
            await self._dispatch_agent_event(TurnEnd(message=reply, tool_results=()))
            self._span(SpanKind.STEP, "step", reason=reason.value)
            return _StepResult(
                message=reply,
                tool_results=(),
                has_more_tool_calls=False,
                hard_terminated=False,
                terminal=True,
            )

        calls = [block for block in reply.content if isinstance(block, ToolCallBlock)]
        for call in calls:
            log.append(
                EventKind.TOOL_CALL,
                {"id": call.id, "name": call.name, "arguments": call.arguments},
            )

        outcome: BatchOutcome | None = None
        tool_result_messages: tuple[Message, ...] = ()
        if calls:
            execution_registry = _snapshot_tool_registry(context.tools)

            # `pending_tool_calls` tracks the real batch window via the already-
            # certified Layer-06 `tools/execution-start`/`tools/execution-end`
            # events -- add on start, remove on end, matching pinned Pi's own
            # `processEvents` reducer exactly, through an existing seam.
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
                if reply.stop_reason is StopReason.LENGTH:
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
                        registry=execution_registry,
                        ctx=self.instance.ctx,
                        scope=self.instance.scope,
                    )
            finally:
                dispose_start()
                dispose_end()
                self.instance.pending_tool_calls = frozenset()

            results: list[Message] = []
            for result in outcome.results:
                message = result.to_message()
                log.append(
                    EventKind.TOOL_RESULT,
                    {
                        "message": encode_message(message),
                        # Pi emits `tool_execution_end` in completion order while
                        # messages follow source order. Results are appended in
                        # source order -- that is what the model reads -- so the
                        # completion order rides along as data for the projection.
                        "completion_index": outcome.completion_index(result.tool_call_id),
                        "added_tool_names": list(result.added_tool_names),
                    },
                )
                await self._dispatch_agent_event(MessageStart(message=message))
                await self._dispatch_agent_event(MessageEnd(message=message))
                context.messages.append(message)
                new_messages.append(message)
                results.append(message)
                if result.added_tool_names:
                    # Layer 06's own boundary (spec/tools.md): `added_tool_names`
                    # is pass-through evidence Layer 06 itself neither
                    # interprets nor acts on, leaving "available from this
                    # transcript point onward" to a later Agent-loop layer.
                    # Consumed here by extending the run-local `context.tools`
                    # snapshot in place -- the same way `context.messages`
                    # grows from this run's own tool results -- never a live
                    # re-read of `self.tools` as a whole (`L08-R001` still
                    # holds: an unrelated external registration made through
                    # `self.tools` while this run is in flight does not leak
                    # in, only the specific names this run's own result named).
                    known = {definition.name for definition in context.tools}
                    newly_visible = []
                    for added_name in result.added_tool_names:
                        if added_name in known:
                            continue
                        added_definition = self.tools.resolve(added_name, self.instance.scope)
                        if added_definition is not None:
                            newly_visible.append(added_definition)
                            known.add(added_name)
                    if newly_visible:
                        context.tools = context.tools + tuple(newly_visible)
            tool_result_messages = tuple(results)

        log.append(EventKind.TURN_END, {})
        await self._dispatch_agent_event(TurnEnd(message=reply, tool_results=tool_result_messages))
        self._span(SpanKind.STEP, "step", reason=reason.value)
        hard_terminated = outcome is not None and outcome.terminate
        has_more_tool_calls = bool(calls) and not hard_terminated
        return _StepResult(
            message=reply,
            tool_results=tool_result_messages,
            has_more_tool_calls=has_more_tool_calls,
            hard_terminated=hard_terminated,
            terminal=False,
        )
