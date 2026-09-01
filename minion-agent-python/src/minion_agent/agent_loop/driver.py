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

PASS 5 remediated L08-R005, R009, R010 (unchanged since; see git history) and
made a first attempt at R002/R004 that a subsequent independent re-review
rejected: it captured Layer-06's own `tools/execution-start`/`-end` EMIT
events via temporary synchronous listeners and redelivered them through
`AGENT_LIFECYCLE_EVENT` only after the whole batch settled -- reordering
sequential batches from `start A, end A, start B, end B` to
`start A, start B, end A, end B`, letting a listener failure surface after
the tool's own side effects had already happened, and dropping
`tool_execution_update` entirely.

PASS 6 remediates that rejection (L08-R002, R004) against the PASS-5
candidate:

- Layer-06's `tools/execute.py`/`tools/batch.py` gained additive, optional
  `on_execution_start`/`on_execution_end` async hooks (`OnExecutionStart`/
  `OnExecutionEnd`), awaited at the EXACT points the existing, still-certified
  `tools/execution-start`/`-end` EMIT events already fire -- so `_run_step`'s
  own `AgentEvent` dispatch for `ToolExecutionStart`/`ToolExecutionEnd` is now
  genuinely LIVE, at real execution points, not captured-and-replayed: a
  listener failure now genuinely prevents that call from proceeding, and
  sequential-mode ordering is the real `start A, end A, start B, end B`
  (`L08-R002`). `None` (every other caller) preserves Layer 06's own
  certified behavior exactly -- no lower-layer contract reopened;
- `tool_execution_update` now reaches the seam too (previously missing
  entirely): the tool `update()` callback is called SYNCHRONOUSLY by a
  tool's own `execute()`, an established SDK-level calling convention this
  pass does not redesign, so updates are captured via the existing sync
  `tools/update` EMIT listener and redelivered per call, immediately before
  that SAME call's own (now-live) `ToolExecutionEnd` dispatch -- preserving
  real per-call causal order (start, its own updates, end) exactly; only the
  relative interleaving of two DIFFERENT concurrent calls' updates is not
  reproducible without redesigning every tool's own synchronous calling
  convention, a disclosed, genuine SDK constraint distinct from PASS 5's own
  Layer-06-EMIT-timing choice;
- `MessageUpdate`/`ToolExecutionEnd` payloads are complete: `MessageUpdate`
  now carries pinned Pi's own raw `assistantMessageEvent` (the exact
  `StreamChunk` variant), not a normalized `kind`/`content_index` pair;
  `ToolExecutionEnd` now carries `tool_name` and the finalized `result`
  itself, not merely `is_error` derived from it (`L08-R002`);
- the no-start stream fallback now reduces in the right order: `streaming_
  message` is set to the reply and `MessageStart` dispatched FIRST (when the
  stream never opened one of its own), THEN `streaming_message` is cleared,
  the transcript entry appended, and `MessageEnd` dispatched -- an earlier
  revision cleared `streaming_message` and appended the transcript entry
  before the fallback `MessageStart`'s own dispatch (`L08-R002`);
- ordinary `agent_start` and a successful run's own `agent_end` dispatch now
  live INSIDE `_execute_run`'s `try`, sharing `_run_inner`'s own exception
  boundary: a listener failure at either point is, per pinned Pi's own
  architecture, indistinguishable from any other run-executor failure and is
  now caught and settled via `_settle_run_failure` the same way, instead of
  escaping straight to `_run_wrapped`'s `finally` (`L08-R002`).

`max_steps` (`L08-R005`) and the removed boundary-stop latch (`L08-R009`,
`L08-R010`) are unchanged since PASS 5. `spec/agent.md`'s own Layer-08
section is the normative contract this file implements; PASS 6 removed the
narrower-tool-event-seam carve-out language PASS 5 left in place, now false.
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
    MessageUpdate,
    ToolExecutionEnd,
    ToolExecutionStart,
    ToolExecutionUpdate,
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
from ..tools.registry import ToolRegistry
from ..tools.result import ToolResult


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

    async def _run_wrapped(
        self,
        *,
        entering: tuple[Message, ...],
        causes: list[dict[str, object]],
        skip_initial_steering_poll: bool = False,
    ) -> None:
        """One pi-equivalent run's full lifecycle: three unconditional state
        writes at entry, matching pinned Pi's own `runWithLifecycle` exactly
        (`isStreaming = true`, `streamingMessage = undefined`, `errorMessage
        = undefined`), and three more, unconditionally, on exit via `finally`
        -- matching pinned Pi's own `finishRun()` (`isStreaming = false`,
        `streamingMessage = undefined`, `pendingToolCalls = new Set()`) --
        regardless of success or a run-executor failure this pass already
        settles gracefully (see `_settle_run_failure`)."""
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
        self.instance.streaming_message = None
        self.instance.error_message = None
        try:
            await self._execute_run(
                entering=entering,
                causes=causes,
                skip_initial_steering_poll=skip_initial_steering_poll,
            )
        finally:
            self.instance.set_status(AgentStatus.IDLE)
            self.instance.streaming_message = None
            self.instance.pending_tool_calls = frozenset()

    async def _dispatch_agent_event(self, event: AgentEvent) -> None:
        """Pinned Pi's own `processEvents`: the single seam every lifecycle
        event -- ordinary turn/run progress and `handleRunFailure` recovery
        alike -- passes through, awaiting every subscribed listener in
        registration order (`L08-R002`). `EventBus.serial` matches pinned
        Pi's own raw `for (const listener of this.listeners) await
        listener(event, signal)` loop exactly: a listener that throws aborts
        the remaining dispatch for THIS event and propagates. `yield_after_
        each=True` (Layer 08, PASS 9, `L08-R002`, contract-convergence
        remediation) reproduces JS's own unconditional per-`await`
        microtask-turn deferral -- pinned Pi's dispatch suspends after EVERY
        listener, even a fully synchronous one, before advancing to the
        next; Python's own `await` on a listener that never itself performs
        a genuine suspension does not share that property without this
        explicit yield. Callers append the corresponding log entry (the
        durable "reduce" step) before calling this (the "notify listeners"
        step), matching pinned Pi's own reduce-then-dispatch order."""
        await self.instance.ctx.events.serial(
            AGENT_LIFECYCLE_EVENT,
            self.instance,
            event,
            scope=self.instance.scope.key,
            yield_after_each=True,
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
        `newMessages.push(message)` timing exactly (`L08-R001`).

        Reduce-before-listener, per event, matching pinned Pi's own
        `processEvents` exactly (`L08-R002`, PASS 4): pinned Pi's own
        reducer applies UNCONDITIONALLY to every `message_start`/
        `message_end`, not only the assistant's own streamed reply --
        `streamingMessage` is set to the message at `message_start`, and
        only at `message_end` is it cleared AND the message pushed onto the
        transcript. An earlier revision appended the durable log entry (this
        run's own "transcript push") BEFORE dispatching `MessageStart`,
        which let a `message_start` listener observe the message already
        present in derived history one event early, and never set
        `streaming_message` for a plain admitted message at all. `log.append`
        (the transcript-push reduce) now happens between the two dispatches,
        exactly where pinned Pi's own `message_end` reducer runs."""
        log = self.instance.log
        for message in messages:
            self.instance.streaming_message = message
            await self._dispatch_agent_event(MessageStart(message=message))

            self.instance.streaming_message = None
            log.append(EventKind.USER_MESSAGE, {"message": encode_message(message)})
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

        `agent_start`'s own append/dispatch AND the success path's own
        `AGENT_END` append/dispatch both live INSIDE this method's `try`
        block, sharing the same `except Exception` boundary as `_run_inner`
        itself (`L08-R002`, PASS 6): a listener that throws while `agent_start`
        or an ordinary, successful `agent_end` is being delivered is, per
        pinned Pi's own architecture, indistinguishable from any other
        run-executor failure -- `runWithLifecycle`'s catch wraps the WHOLE
        run executor, not just the turns in between -- so it is caught here
        too and converted into a settled failure turn via
        `_settle_run_failure`, never silently swallowed and never left as a
        half-delivered success. `context`/`config` are built before the
        `try` (pure local assembly, no listener dispatch involved) so
        `_settle_run_failure` always has a `config` to build its failure
        message from, even when `agent_start` itself is what failed.
        """
        log = self.instance.log
        context = RunContext(
            system_prompt=self.instance.system_prompt,
            messages=list(derive_messages(self.instance.log)),
            tools=self.tools.visible_from(self.instance.scope),
        )
        config = RunConfig(model=self.instance.model, thinking_level=self.instance.thinking_level)
        new_messages: list[Message] = []

        try:
            log.append(EventKind.AGENT_START, {"causes": causes})
            await self._dispatch_agent_event(AgentStart(causes=tuple(causes)))

            end_reason, detail = await self._run_inner(
                entering, causes, skip_initial_steering_poll, context, config, new_messages
            )

            # agent_end's own reduce, matching pinned Pi's reducer exactly
            # (`case "agent_end": streamingMessage = undefined`) -- always
            # already `None` by this point via each turn's own message_end,
            # but set unconditionally here too, not left to happen to be true.
            self.instance.streaming_message = None
            payload: dict[str, object] = {"reason": end_reason, "causes": causes}
            if detail is not None:
                payload["detail"] = detail
            log.append(EventKind.AGENT_END, payload)
            await self._dispatch_agent_event(
                AgentEnd(reason=end_reason, causes=tuple(causes), messages=tuple(new_messages))
            )
        except UnknownModelError:
            # Pinned Pi's own boundary: resolving a model happens before a
            # stream is returned, so an unresolvable one is a caller/config
            # bug, reported eagerly -- never smuggled into a settled failure
            # turn (L08-R002, `eager-invalid-model-fails-before-stream.yaml`).
            raise
        except Exception as error:
            # Propagates uncaught if a recovery listener itself throws
            # (`L08-R002`) -- pinned Pi's own `handleRunFailure` has no
            # further catch around it either; `_run_wrapped`'s own `finally`
            # still settles status, matching pinned Pi's `finishRun`. If the
            # successful `AGENT_END` above was already appended and only its
            # OWN dispatch is what failed, this produces a second, `failed`
            # `AGENT_END` right after it -- matching pinned Pi exactly:
            # `handleRunFailure` has no awareness of how far the run had
            # already reduced when its own dispatch throws.
            await self._settle_run_failure(error, config, causes)

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
        `projection.py`) for callers reconstructing the log after the fact.

        Reduce-before-listener, per event, matching pinned Pi's own
        `processEvents` exactly (`L08-R002`, PASS 4): `streaming_message` is
        set to the failure at `message_start` and cleared at `message_end`
        (an earlier revision never set it at all); the failure's transcript
        entry is appended at `message_end` time, between the two dispatches,
        not before either; `error_message` is set as part of `turn_end`'s own
        reduce, before that event's own dispatch (an earlier revision set it
        only after `turn_end`'s listeners had already run)."""
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

        self.instance.streaming_message = failure
        await self._dispatch_agent_event(MessageStart(message=failure))

        self.instance.streaming_message = None
        log.append(EventKind.ASSISTANT_MESSAGE, {"message": encode_message(failure)})
        await self._dispatch_agent_event(MessageEnd(message=failure))

        self.instance.error_message = str(error)
        log.append(EventKind.TURN_END, {"message": encode_message(failure)})
        await self._dispatch_agent_event(TurnEnd(message=failure, tool_results=()))

        payload: dict[str, object] = {
            "reason": "failed",
            "causes": causes,
            "messages": [encode_message(failure)],
        }
        log.append(EventKind.AGENT_END, payload)
        await self._dispatch_agent_event(
            AgentEnd(reason="failed", causes=tuple(causes), messages=(failure,))
        )
        # Pinned Pi's `finishRun()` (not `handleRunFailure` itself) resets
        # `pendingToolCalls`; `_run_wrapped`'s own `finally` mirrors that,
        # unconditionally, for every run -- not repeated here.

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

        # Streamed assistant reply lifecycle, fully live (`L08-R002`, PASS 5):
        # iterates `self.llm.stream(request)` directly rather than through the
        # certified Layer-02/04 `collect()` convenience wrapper, whose own
        # `on_chunk` callback is synchronous by design and cannot itself
        # `await` a dispatch. This does not reopen or modify `collect()` --
        # still certified, still used elsewhere -- it reproduces its own
        # trivial drain loop here, the one call site that needs an async
        # per-chunk dispatch. `streaming_message` is set directly from each
        # chunk's own already-complete `partial` (`L08-R003`, unchanged): the
        # certified `StreamChunk` carries it on every variant, so no
        # independent reconstruction from raw deltas is attempted. Reduce
        # (the log append / `streaming_message` write), THEN dispatch, per
        # event -- matching pinned Pi's own `processEvents` order exactly.
        stream_opened = False
        reply: AssistantMessage | None = None
        async for chunk in self.llm.stream(request):
            match chunk:
                case StreamStart():
                    stream_opened = True
                    self.instance.streaming_message = chunk.partial
                    log.append(
                        EventKind.ASSISTANT_STREAM_START,
                        {"partial": encode_message(chunk.partial)},
                    )
                    await self._dispatch_agent_event(MessageStart(message=chunk.partial))
                    continue
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
                    reply = chunk.message
                    break
            self.instance.streaming_message = chunk.partial
            # `chunk` itself IS pinned Pi's own raw `assistantMessageEvent` shape
            # (`TextStart | TextDelta | ... | ToolCallEnd`) -- logged/dispatched
            # verbatim, not normalized (`L08-R002`, PASS 6). `"delta"` is the one
            # field a delta-kind chunk carries that its own already-logged
            # `partial` cannot reconstruct (the incremental string itself, not
            # the accumulated total); the other kinds need nothing extra --
            # `project()` (`agent/projection.py`) derives a start/end chunk's
            # own type-specific fields from `partial.content[content_index]`.
            chunk_data: dict[str, object] = {
                "kind": kind,
                "content_index": chunk.content_index,
                "partial": encode_message(chunk.partial),
            }
            if kind in ("text_delta", "thinking_delta", "toolcall_delta"):
                chunk_data["delta"] = chunk.delta  # type: ignore[union-attr]
            log.append(EventKind.ASSISTANT_CHUNK, chunk_data)
            await self._dispatch_agent_event(MessageUpdate(event=chunk, message=chunk.partial))
        # Never actually `None`: `self.llm` is always a `LlmService`, whose
        # own certified `_settled()` wrapper (`llm/service.py`, Layer 02/04)
        # guarantees every returned stream yields a terminal chunk before
        # ending, even for a raw adapter that breaks its own protocol -- the
        # same guarantee `collect()` enforces defensively for a caller that
        # might hand it an unwrapped stream, which this call site never does.
        assert reply is not None, "LlmService.stream always yields a terminal chunk"

        # `not stream_opened` mirrors pinned Pi's own defensive `!addedPartial`
        # fallback: a stream that never emitted its own "start" still gets a
        # `MessageStart` here -- its own full reduce-then-dispatch, BEFORE
        # message_end's (`L08-R002`, PASS 6): an earlier revision cleared
        # `streaming_message` and appended the transcript entry (message_end's
        # own reduce) before this fallback `MessageStart`'s own dispatch,
        # which let a `message_start` listener observe the reply already
        # pushed to the transcript one event early.
        if not stream_opened:
            self.instance.streaming_message = reply
            await self._dispatch_agent_event(MessageStart(message=reply))

        self.instance.streaming_message = None
        log.append(EventKind.ASSISTANT_MESSAGE, {"message": encode_message(reply)})
        await self._dispatch_agent_event(MessageEnd(message=reply))
        context.messages.append(reply)
        new_messages.append(reply)

        if reply.stop_reason in (StopReason.ERROR, StopReason.ABORTED):
            # Represented error/aborted (`L08-R008`): pinned Pi emits this
            # turn's own `turn_end` with empty `toolResults` and returns
            # immediately -- no tool calls are ever inspected or executed for
            # a message representing its own failure. `error_message` is
            # `turn_end`'s own reduce (pinned Pi: `if role === "assistant" &&
            # errorMessage`), set before that event's own dispatch, not after
            # -- and nothing clears it again until the *next* run starts
            # (`_run_wrapped`) or `reset()` (Layer-07, certified).
            if reply.error_message:
                self.instance.error_message = reply.error_message
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

            # `on_execution_start`/`on_execution_end`/`on_execution_update` are Layer-06's own
            # additive, awaited hooks (`tools/execute.py`): genuinely LIVE dispatch, at the exact
            # points pinned Pi's own `tool_execution_start`/`_update`/`_end` fire, not
            # captured-and-replayed after the batch settles (`L08-R002`). `pending_tool_calls`
            # (the reduce) is updated in the same closures, immediately before each event's own
            # dispatch, matching pinned Pi's own `processEvents` reducer exactly -- it stays set
            # for the whole time a call's own updates are still in flight, since `on_execution_end`
            # (which clears it) only fires once `_execute_and_finalize` has already joined every
            # one of that call's own scheduled update dispatches (`tools/execute.py`,
            # `OnExecutionUpdate`).
            #
            # `on_execution_update` is started SYNCHRONOUSLY (`asyncio.eager_task_factory`, PASS
            # 8 -- `ensure_future`'s own deferred-first-step scheduling, tried in PASS 7, still
            # observably reordered `tool-continued` before `listener-entered`), not awaited
            # inline, by `tools/execute.py` itself the moment a tool's own SYNCHRONOUS
            # `update(partial)` callback fires -- matching pinned Pi's own `emit(...)` at callback
            # time exactly (`agent-loop.ts:670-711`) -- so two different calls' own update
            # dispatches interleave according to real scheduling, not a batch-wide
            # capture-and-replay order (PASS 7, correcting PASS 6's own
            # capture-then-redeliver-per-call approximation, itself an improvement over PASS 5's
            # batch-wide one but still not genuinely live).
            async def on_execution_start(
                call_id: str, name: str, arguments: dict[str, object]
            ) -> None:
                self.instance.pending_tool_calls = self.instance.pending_tool_calls | {call_id}
                await self._dispatch_agent_event(
                    ToolExecutionStart(tool_call_id=call_id, tool_name=name, arguments=arguments)
                )

            async def on_execution_update(
                call_id: str, name: str, arguments: dict[str, object], partial_result: str
            ) -> None:
                await self._dispatch_agent_event(
                    ToolExecutionUpdate(
                        tool_call_id=call_id,
                        tool_name=name,
                        arguments=arguments,
                        partial_result=partial_result,
                    )
                )

            async def on_execution_end(call_id: str, name: str, result: ToolResult) -> None:
                self.instance.pending_tool_calls = self.instance.pending_tool_calls - {call_id}
                await self._dispatch_agent_event(
                    ToolExecutionEnd(
                        tool_call_id=call_id,
                        tool_name=name,
                        result=result,
                        is_error=result.is_error,
                    )
                )

            if reply.stop_reason is StopReason.LENGTH:
                # A length stop means the output was cut off by the token limit, so every
                # tool call the message carries may itself have truncated arguments. Pinned
                # Pi fails them all instead of executing potentially-truncated calls
                # (`failToolCallsFromTruncatedMessage`, TOOL-017) -- none reach the registry, so
                # none ever call `update()` -- no `on_execution_update` to supply.
                outcome = await execute_length_stop_batch(
                    calls,
                    ctx=self.instance.ctx,
                    scope=self.instance.scope,
                    on_execution_start=on_execution_start,
                    on_execution_end=on_execution_end,
                )
            else:
                outcome = await execute_batch(
                    calls,
                    registry=execution_registry,
                    ctx=self.instance.ctx,
                    scope=self.instance.scope,
                    on_execution_start=on_execution_start,
                    on_execution_end=on_execution_end,
                    on_execution_update=on_execution_update,
                )

            results: list[Message] = []
            for result in outcome.results:
                message = result.to_message()
                self.instance.streaming_message = message
                await self._dispatch_agent_event(MessageStart(message=message))

                self.instance.streaming_message = None
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
                        # The one `ToolResult` field `to_message()` never copies onto
                        # `ToolResultMessage` (by design -- it must never reach the
                        # model): `project()` (`agent/projection.py`) needs it to
                        # rebuild an equivalent `ToolResult` for `ToolExecutionEnd.
                        # result` from the log alone (`L08-R002`, PASS 6).
                        "terminate": result.terminate,
                    },
                )
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

        # turn_end's own reduce (pinned Pi: `if role === "assistant" &&
        # errorMessage`), before that event's own dispatch -- a no-op here in
        # the ordinary case (a normal reply carries no error), but applied
        # unconditionally, matching pinned Pi's own reducer exactly.
        if reply.error_message:
            self.instance.error_message = reply.error_message
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
