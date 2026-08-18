"""The pre-execute decision union: block a call, or let it run."""

from minion_agent.tools.decisions import Block, Proceed


def test_block_carries_a_reason_the_model_will_read() -> None:
    assert Block(reason="not permitted").reason == "not permitted"


def test_block_does_not_terminate_by_default() -> None:
    """Refusing one call is the common case; ending the turn is the exception,
    so it has to be asked for."""
    assert not Block(reason="not permitted").terminate


def test_block_may_also_end_the_turn() -> None:
    assert Block(reason="hard stop", terminate=True).terminate


def test_proceed_carries_the_arguments_the_tool_will_receive() -> None:
    """A listener may narrow or rewrite them -- which is how sandboxing pins a
    path -- so the decision carries them rather than only approving."""
    decision = Proceed(arguments={"path": "/safe/only"})

    assert decision.arguments == {"path": "/safe/only"}
