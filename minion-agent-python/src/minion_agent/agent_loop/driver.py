"""The concrete agent loop.

Imperative and package-internal, following DSH's `ReactLoopAgent`: a stateful
class with an explicit phase. Neither pi nor DSH factors the live loop as a
pure reducer, and this design does not invent one.

A **step** is one model request plus the tools it calls. A **turn** is zero or
more steps.
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
from ..agent.tools import ToolService
from ..llm import LlmService, Request, ToolCallBlock, UserMessage, collect
from ..session import (
    ArtifactStore,
    EventKind,
    assemble_system,
    derive_messages,
    encode_message,
    record_header,
)
from ..telemetry import TelemetryService


class AgentLoop:
    """Drives one agent instance through turns and steps."""

    def __init__(
        self,
        *,
        instance: AgentInstance,
        llm: LlmService,
        tools: ToolService,
        artifacts: ArtifactStore,
        telemetry: TelemetryService | None = None,
    ) -> None:
        self.instance = instance
        # Collaborators are public: tests configure them directly, and Plan 4
        # replaces the tool service outright.
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
                await self._run_turn()
        finally:
            inbox.take_wake()
            self.instance.set_status(AgentStatus.IDLE)

    def cancel(self) -> None:
        """Request that the current turn end at its next boundary.

        Work already in flight -- a running tool, an open request -- is allowed
        to finish, so the transcript stays coherent. Cancellation stops the
        *next* request, not the current one.
        """
        self._cancelled = True

    async def _run_turn(self) -> None:
        log = self.instance.log
        claimed = self.instance.inbox.claim(InboxTarget.NEXT_TURN, self.next_turn_policy)
        causes = [{"id": envelope.id, "origin": envelope.origin} for envelope in claimed]
        log.append(EventKind.TURN_START, {"causes": causes})

        # The first step claims step input too, so steering queued before the
        # turn opened enters it rather than waiting for a second step.
        entering = tuple(envelope.message for envelope in claimed) + tuple(
            envelope.message for envelope in self._claim_step_input()
        )
        decision = await self._pre_step(entering, PreStepReason.INITIAL)
        if isinstance(decision, Reject):
            self._cancelled = False
            log.append(
                EventKind.TURN_END,
                {"reason": "rejected", "causes": causes, "detail": decision.reason},
            )
            return

        reason = PreStepReason.INITIAL
        steps = 0
        end_reason = "completed"

        while True:
            owed = await self._run_step(decision, reason)
            steps += 1
            if not owed:
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

        # Cleared with the turn: a cancelled turn must not poison the next.
        self._cancelled = False
        # Causes repeat at the end so a consumer reading only completions can
        # route a result without replaying the whole turn.
        log.append(EventKind.TURN_END, {"reason": end_reason, "causes": causes})

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
        self, messages: tuple[UserMessage, ...], reason: PreStepReason
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

    async def _run_step(self, decision: Enter, reason: PreStepReason) -> bool:
        """Run one model request and its tools. Returns whether tools ran.

        Tools execute sequentially: parallel batching, execution modes, and the
        `terminate` fold are Plan 4's, and the single-call path is what closes
        the round trip.
        """
        log = self.instance.log

        # Before `step/start`: what entered the step caused it, and the model
        # cannot see a prompt that is not yet on the surface when the request
        # is derived.
        for message in decision.messages:
            log.append(EventKind.USER_MESSAGE, {"message": encode_message(message)})

        log.append(EventKind.STEP_START, {"reason": reason.value})

        components = {
            "system_base": decision.system_override
            if decision.system_override is not None
            else self.instance.definition.system
        }
        record_header(log, self.artifacts, components, model=self.instance.definition.model.model)

        history = derive_messages(log)
        if decision.history_window is not None:
            history = history[-decision.history_window :]

        reply = await collect(
            self.llm.stream(
                Request(
                    model=self.instance.definition.model,
                    system=assemble_system(components),
                    messages=history,
                )
            )
        )
        log.append(EventKind.ASSISTANT_MESSAGE, {"message": encode_message(reply)})

        calls = [block for block in reply.content if isinstance(block, ToolCallBlock)]
        for call in calls:
            log.append(
                EventKind.TOOL_CALL,
                {"id": call.id, "name": call.name, "arguments": call.arguments},
            )
            result = await self.tools.execute(call)
            log.append(EventKind.TOOL_RESULT, {"message": encode_message(result)})

        log.append(EventKind.STEP_END, {})
        return bool(calls)
