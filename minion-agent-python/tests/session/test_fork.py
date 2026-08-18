"""Forks reference their ancestor rather than copying it."""

from minion_agent.llm.content import TextBlock
from minion_agent.llm.messages import UserMessage, text_of
from minion_agent.session.derive import derive_messages, encode_message
from minion_agent.session.events import EventKind
from minion_agent.session.log import SessionLog
from minion_agent.session.operations import compact, fork, reset


def _say(log: SessionLog, text: str) -> None:
    message = UserMessage(content=(TextBlock(text=text),), timestamp=1)
    log.append(EventKind.USER_MESSAGE, {"message": encode_message(message)})


def _texts(log: SessionLog) -> list[str]:
    return [text_of(message) for message in derive_messages(log)]


def test_a_fork_inherits_its_ancestors_history() -> None:
    source = SessionLog("s1")
    _say(source, "shared")

    child = fork(source, "s2")

    assert _texts(child) == ["shared"]


def test_a_fork_has_its_own_identity() -> None:
    source = SessionLog("s1")

    child = fork(source, "s2")

    assert (source.session_id, child.session_id) == ("s1", "s2")


def test_writes_to_a_fork_do_not_reach_its_ancestor() -> None:
    source = SessionLog("s1")
    _say(source, "shared")

    child = fork(source, "s2")
    _say(child, "child only")

    assert _texts(source) == ["shared"]
    assert _texts(child) == ["shared", "child only"]


def test_writes_to_the_ancestor_after_a_fork_do_not_reach_the_child() -> None:
    """The boundary is fixed at fork time, not a live view."""
    source = SessionLog("s1")
    _say(source, "shared")
    child = fork(source, "s2")

    _say(source, "parent only")

    assert _texts(child) == ["shared"]


def test_forking_at_an_explicit_boundary_truncates_history() -> None:
    source = SessionLog("s1")
    _say(source, "first")
    boundary = len(source)
    _say(source, "second")

    child = fork(source, "s2", at=boundary)

    assert _texts(child) == ["first"]


def test_a_fork_records_its_ancestry() -> None:
    source = SessionLog("s1")
    _say(source, "shared")

    child = fork(source, "s2")

    assert child.events[0].kind is EventKind.SESSION_FORKED
    assert child.events[0].data["source"] == "s1"


def test_a_fork_copies_nothing() -> None:
    """One place for one truth: the child holds only its own events."""
    source = SessionLog("s1")
    for text in ("a", "b", "c"):
        _say(source, text)

    child = fork(source, "s2")

    assert len(child) == 1


def test_compaction_inside_a_fork_does_not_affect_the_ancestor() -> None:
    source = SessionLog("s1")
    _say(source, "one")
    _say(source, "two")
    child = fork(source, "s2")

    compact(child, summary="summarised", keep=0)

    assert _texts(source) == ["one", "two"]
    assert _texts(child) == ["summarised"]


def test_reset_inside_a_fork_clears_inherited_history() -> None:
    source = SessionLog("s1")
    _say(source, "inherited")
    child = fork(source, "s2")

    reset(child)
    _say(child, "fresh")

    assert _texts(child) == ["fresh"]
    assert _texts(source) == ["inherited"]


def test_a_fork_of_a_fork_walks_the_whole_chain() -> None:
    root = SessionLog("s1")
    _say(root, "root")
    middle = fork(root, "s2")
    _say(middle, "middle")

    leaf = fork(middle, "s3")
    _say(leaf, "leaf")

    assert _texts(leaf) == ["root", "middle", "leaf"]


def test_a_forks_compaction_retains_its_own_tail_only() -> None:
    """Sequence numbers restart in a fork, so retained seqs must not be
    confused with inherited ones carrying the same numbers."""
    source = SessionLog("s1")
    _say(source, "inherited one")
    _say(source, "inherited two")
    child = fork(source, "s2")
    _say(child, "own one")
    _say(child, "own two")

    compact(child, summary="summary", keep=1)

    assert _texts(child) == ["summary", "own two"]
