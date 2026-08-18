"""The concrete agent loop.

Imperative and package-internal, following DSH's `ReactLoopAgent`: a stateful
class with an explicit phase. Neither pi nor DSH factors the live loop as a
pure reducer, and this design does not invent one.

A **step** is one model request plus the tools it calls. A **turn** is zero or
more steps.
"""

from __future__ import annotations

from ..agent.decisions import Enter, PreStepReason
from ..agent.envelope import ClaimPolicy, InboxTarget
from ..agent.identity import AgentStatus
from ..agent.instance import AgentInstance
from ..agent.tools import ToolService
from ..llm import LlmService, Request, collect
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

    async def _run_turn(self) -> None:
        log = self.instance.log
        claimed = self.instance.inbox.claim(InboxTarget.NEXT_TURN, self.next_turn_policy)
        causes = [{"id": envelope.id, "origin": envelope.origin} for envelope in claimed]
        log.append(EventKind.TURN_START, {"causes": causes})

        decision = Enter(messages=tuple(envelope.message for envelope in claimed))
        await self._run_step(decision, PreStepReason.INITIAL)

        # Repeated at the end so a consumer reading only completions can route
        # a result without replaying the whole turn.
        log.append(EventKind.TURN_END, {"reason": "completed", "causes": causes})

    async def _run_step(self, decision: Enter, reason: PreStepReason) -> None:
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
        log.append(EventKind.STEP_END, {})
