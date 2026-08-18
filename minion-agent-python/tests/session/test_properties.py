"""Properties that must hold for any sequence of session operations."""

from hypothesis import given
from hypothesis import strategies as st

from minion_agent.llm.content import TextBlock
from minion_agent.llm.messages import UserMessage, text_of
from minion_agent.session.derive import derive_messages, encode_message
from minion_agent.session.events import EventKind
from minion_agent.session.log import SessionLog
from minion_agent.session.operations import compact, fork, reset

texts = st.lists(st.text(min_size=1, max_size=6), min_size=0, max_size=20)


def _say(log: SessionLog, text: str) -> None:
    message = UserMessage(content=(TextBlock(text=text),), timestamp=1)
    log.append(EventKind.USER_MESSAGE, {"message": encode_message(message)})


@given(texts)
def test_derivation_preserves_order_and_content(items: list[str]) -> None:
    log = SessionLog("s1")
    for text in items:
        _say(log, text)

    assert [text_of(m) for m in derive_messages(log)] == items


@given(texts)
def test_reset_always_yields_an_empty_derivation(items: list[str]) -> None:
    log = SessionLog("s1")
    for text in items:
        _say(log, text)

    reset(log)

    assert derive_messages(log) == ()


@given(texts, st.integers(min_value=0, max_value=5))
def test_compaction_never_exceeds_summary_plus_keep(items: list[str], keep: int) -> None:
    log = SessionLog("s1")
    for text in items:
        _say(log, text)

    compact(log, summary="s", keep=keep)

    assert len(derive_messages(log)) <= 1 + min(keep, len(items))


@given(texts)
def test_appending_never_shrinks_derivation(items: list[str]) -> None:
    log = SessionLog("s1")
    previous = 0
    for text in items:
        _say(log, text)
        current = len(derive_messages(log))
        assert current >= previous
        previous = current


@given(texts)
def test_a_fork_starts_from_its_ancestors_derivation(items: list[str]) -> None:
    source = SessionLog("s1")
    for text in items:
        _say(source, text)

    child = fork(source, "s2")

    assert derive_messages(child) == derive_messages(source)


@given(texts)
def test_the_log_never_shrinks(items: list[str]) -> None:
    """Append-only means append-only: no operation reduces the event count."""
    log = SessionLog("s1")
    sizes = [len(log)]
    for text in items:
        _say(log, text)
        sizes.append(len(log))
    reset(log)
    sizes.append(len(log))
    compact(log, summary="s", keep=0)
    sizes.append(len(log))

    assert sizes == sorted(sizes)


@given(st.integers(min_value=1, max_value=6))
def test_a_chain_of_forks_of_any_depth_derives_every_link(depth: int) -> None:
    log = SessionLog("root")
    _say(log, "level-0")

    for level in range(1, depth):
        log = fork(log, f"s{level}")
        _say(log, f"level-{level}")

    assert [text_of(m) for m in derive_messages(log)] == [
        f"level-{level}" for level in range(depth)
    ]
