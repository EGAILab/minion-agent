"""The inbox: two queues, three aliases, and a wake signal.

DSH's `send(message, target, wakeup)` generalizes pi's two queues, and the
three aliases are its fixed presets:

    followup  next-turn  wakes
    steer     next-step  wakes
    inject    next-step  silent

Accepted message domain (`AG-011`, `L07-R002`): pinned Pi's `steer`/`followUp`
each accept the whole `AgentMessage` union
(`UserMessage | AssistantMessage | ToolResultMessage | CustomAgentMessages[...]`).
`CustomAgentMessages` is empty in pinned Pi itself, so the actual domain is
exactly `Message` -- the already-certified Layer-02 vocabulary -- adopted here
verbatim. An earlier revision narrowed this to `UserMessage` only, with no
coherent disposition; that narrowing is corrected, not merely re-labeled as
intentional, since no architectural reason for it was ever established.
"""

from __future__ import annotations

import uuid

from ..llm import Message
from .envelope import ClaimPolicy, InboxTarget, InputEnvelope, JsonValue

_JSON_SCALARS = (str, int, float, bool, type(None))


class NotJsonSafeOriginError(TypeError):
    """An origin was supplied that JSON cannot represent."""


def _check_json_safe(value: object, path: str = "origin") -> None:
    if isinstance(value, _JSON_SCALARS):
        return
    if isinstance(value, dict):
        for key, item in value.items():
            if not isinstance(key, str):
                raise NotJsonSafeOriginError(f"{path}: keys must be strings, got {key!r}")
            _check_json_safe(item, f"{path}.{key}")
        return
    if isinstance(value, list | tuple):
        for index, item in enumerate(value):
            _check_json_safe(item, f"{path}[{index}]")
        return
    raise NotJsonSafeOriginError(f"{path}: {type(value).__name__} is not JSON-safe")


class Inbox:
    """Queued input for one agent instance."""

    def __init__(self) -> None:
        self._queues: dict[InboxTarget, list[InputEnvelope]] = {
            InboxTarget.NEXT_TURN: [],
            InboxTarget.NEXT_STEP: [],
        }
        self._wake = False

    @property
    def wake_requested(self) -> bool:
        """Whether input has arrived that should start or continue work."""
        return self._wake

    def take_wake(self) -> bool:
        """Consume the wake signal, returning whether one was pending."""
        pending, self._wake = self._wake, False
        return pending

    def send(
        self,
        message: Message,
        target: InboxTarget,
        wakeup: bool,
        origin: JsonValue = None,
    ) -> InputEnvelope:
        """Queue `message`, validating its origin before anything is stored."""
        _check_json_safe(origin)
        envelope = InputEnvelope(id=str(uuid.uuid4()), message=message, origin=origin)
        self._queues[target].append(envelope)
        if wakeup:
            self._wake = True
        return envelope

    def followup(self, message: Message, origin: JsonValue = None) -> InputEnvelope:
        """Queue a prompt for the next turn and wake the driver."""
        return self.send(message, InboxTarget.NEXT_TURN, wakeup=True, origin=origin)

    def steer(self, message: Message, origin: JsonValue = None) -> InputEnvelope:
        """Queue input for the next step boundary and wake the driver."""
        return self.send(message, InboxTarget.NEXT_STEP, wakeup=True, origin=origin)

    def inject(self, message: Message, origin: JsonValue = None) -> InputEnvelope:
        """Queue context for the next step boundary without waking.

        It rides along with whatever wakes the driver next, which is what makes
        it usable for ambient context that should not itself start work.
        """
        return self.send(message, InboxTarget.NEXT_STEP, wakeup=False, origin=origin)

    def pending(self, target: InboxTarget) -> tuple[InputEnvelope, ...]:
        """What is queued at `target`, unclaimed."""
        return tuple(self._queues[target])

    def has_pending(self) -> bool:
        """Whether either queue still holds unclaimed input (pinned Pi's
        `Agent.hasQueuedMessages()`: true when the steering OR the follow-up
        queue has items)."""
        return any(self._queues.values())

    def claim(self, target: InboxTarget, policy: ClaimPolicy) -> tuple[InputEnvelope, ...]:
        """Remove and return queued input according to `policy`."""
        queue = self._queues[target]
        if not queue:
            return ()
        if policy is ClaimPolicy.ALL:
            claimed, queue[:] = tuple(queue), []
            return claimed
        return (queue.pop(0),)

    def clear(self, target: InboxTarget) -> None:
        """Discard whatever is queued at `target`, unclaimed (pinned Pi's
        `clearSteeringQueue()`/`clearFollowUpQueue()`, one queue at a time).
        The wake signal is untouched -- orthogonal concerns: wake means
        "something happened", clearing only removes queued content."""
        self._queues[target].clear()

    def clear_all(self) -> None:
        """Discard everything queued at every target (pinned Pi's
        `clearAllQueues()`)."""
        for target in self._queues:
            self.clear(target)
