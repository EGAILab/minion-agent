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


class _Reservation:
    """A one-shot, claim-bound run-entry reservation (Layer 08 only, `L09-R012`/`L09-R013`/
    `L09-R014` convergence). The ONLY way to obtain one is `Inbox._reserve()`, which atomically
    `claim()`s the entering batch before constructing it -- `.envelopes` is exactly what that
    SAME `claim()` call returned, never caller-suppliable. Exactly one of `.commit()`/
    `.rollback()` may ever be called, and each accepts NO argument at all -- there is no
    parameter through which a caller could substitute a foreign envelope, and a private
    `_settled` guard rejects any second terminal call (whichever method) with `RuntimeError`.

    An earlier revision (`L09-R007` convergence, PASS 5) claimed eagerly and exposed a PUBLIC
    `restore(target, envelopes)` to reverse a failed claim; an independent Rust review found that
    method callable by any caller with any envelope tuple, including one never claimed, or the
    same envelope repeatedly, manufacturing duplicate queue entries sharing an id (`L09-R010`).
    A later revision (PASS 6) replaced it with `peek()`/`_commit_claim()`, deferring the
    destructive removal until a run-entry attempt was certain to proceed -- but the WINDOW that
    created between selection and removal let a re-entrant `RUNNING`-notification observer either
    delete unrelated, never-selected input (`L09-R011`) or leave part of the entering batch
    unrestored (`L09-R013`) or duplicated across two runs (`L09-R014`), depending on what it did
    in that window. This type removes the window entirely: the batch is claimed -- destructively,
    unconditionally, atomically -- BEFORE the observer ever runs, so the observer cannot see or
    touch it at all, and the ONLY question left is whether the run-entry attempt that claimed it
    ultimately succeeds (`.commit()`, a no-op -- the removal already happened) or fails
    (`.rollback()`, restoring the exact batch, prepended ahead of anything the observer itself
    enqueued in the meantime). Only THIS reservation's own claimed batch is ever rolled back;
    anything else a `RUNNING` observer did to `Inbox` (a genuinely unrelated claim, a clear, new
    enqueued input) is never reversed -- this mechanism protects exactly one thing, not the whole
    `Inbox` as a general transaction."""

    __slots__ = ("_inbox", "_settled", "_target", "envelopes")

    def __init__(
        self, inbox: Inbox, target: InboxTarget, envelopes: tuple[InputEnvelope, ...]
    ) -> None:
        self._inbox = inbox
        self._target = target
        self.envelopes = envelopes
        self._settled = False

    def commit(self) -> None:
        """Leave the claimed batch removed. A no-op beyond marking this reservation settled --
        `claim()` already performed the removal when this reservation was created."""
        if self._settled:
            raise RuntimeError("reservation already settled")
        self._settled = True

    def rollback(self) -> None:
        """Restore the claimed batch, prepended ahead of whatever is queued at `target` now."""
        if self._settled:
            raise RuntimeError("reservation already settled")
        self._settled = True
        if self.envelopes:
            self._inbox._queues[self._target][0:0] = self.envelopes


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

    def _reserve(self, target: InboxTarget, policy: ClaimPolicy) -> _Reservation:
        """Atomically `claim()` entering input for a run-entry attempt and return a fresh,
        single-use `_Reservation` bound to exactly that batch (Layer 08 only,
        `AgentLoop._run_wrapped`, via `continue_()`/`run_until_idle()`). Not part of this class's
        own public API -- see `_Reservation`'s own docstring for the full rationale and the
        authority guarantees this closes (`L09-R010`/`L09-R011`/`L09-R012`/`L09-R013`/`L09-R014`).
        `claim()` itself remains the sole PUBLIC removal operation, unchanged by this method's
        existence."""
        return _Reservation(self, target, self.claim(target, policy))

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
