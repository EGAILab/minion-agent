"""Session append/compaction under concurrent asyncio callers.

Delta finding E: "atomic" (design spec section 5, `spec/session.md`) means
logically indivisible, not necessarily thread-safe -- an implementation
whose supported execution model has no concurrent-caller hazard for a given
operation satisfies the rule without extra synchronization. `SessionLog`
carries no lock because none of its operations ever `await` mid-call: under
Python's actual supported execution model (single-threaded asyncio,
cooperative scheduling), a coroutine can only be preempted at an `await`
point, so `append`/`compact` already run to completion atomically relative
to every other coroutine on the same event loop. These tests prove that
property adversarially rather than asserting it from reasoning alone.
"""

from __future__ import annotations

import asyncio

from minion_agent.llm.content import TextBlock
from minion_agent.llm.messages import UserMessage
from minion_agent.session.derive import encode_message
from minion_agent.session.events import EventKind
from minion_agent.session.log import SessionLog
from minion_agent.session.operations import compact


def _say(log: SessionLog, text: str) -> None:
    message = UserMessage(content=(TextBlock(text=text),), timestamp=1)
    log.append(EventKind.USER_MESSAGE, {"message": encode_message(message)})


async def test_concurrent_appends_never_produce_duplicate_or_gapped_sequence_numbers() -> None:
    log = SessionLog("s1")

    async def append_many(prefix: str, count: int) -> None:
        for i in range(count):
            _say(log, f"{prefix}-{i}")
            await asyncio.sleep(0)  # yield between appends -- the adversarial case

    await asyncio.gather(append_many("a", 30), append_many("b", 30), append_many("c", 30))

    seqs = [event.seq for event in log.events]
    assert len(seqs) == 90
    assert seqs == list(range(1, 91)), (
        "sequence numbers must be unique, gapless, and commit-ordered"
    )


async def test_append_racing_compaction_never_corrupts_either_operation() -> None:
    log = SessionLog("s1")
    for i in range(5):
        _say(log, f"seed-{i}")

    async def appender(count: int) -> None:
        for i in range(count):
            _say(log, f"race-{i}")
            await asyncio.sleep(0)

    async def compactor() -> None:
        await asyncio.sleep(0)
        compact(log, summary="mid-flight compaction", keep=1)

    await asyncio.gather(appender(20), compactor(), appender(20))

    seqs = [event.seq for event in log.events]
    assert seqs == list(range(1, len(seqs) + 1)), "no interleaved append may corrupt sequence order"
    compaction_events = [e for e in log.events if e.kind == EventKind.COMPACTION]
    assert len(compaction_events) == 1, "the compaction event itself must not be duplicated or torn"


async def test_two_concurrent_compactions_each_commit_a_complete_event() -> None:
    log = SessionLog("s1")
    for i in range(4):
        _say(log, f"seed-{i}")

    async def do_compact(summary: str) -> None:
        await asyncio.sleep(0)
        compact(log, summary=summary, keep=0)

    await asyncio.gather(do_compact("first"), do_compact("second"))

    compaction_events = [e for e in log.events if e.kind == EventKind.COMPACTION]
    assert len(compaction_events) == 2
    for event in compaction_events:
        assert set(event.data) == {"summary", "superseded_through", "retained"}, (
            "each compaction's data must be a complete, uninterleaved write"
        )
