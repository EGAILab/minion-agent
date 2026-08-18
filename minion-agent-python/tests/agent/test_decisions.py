"""Closed decision unions, with their combination rules stated."""

from minion_agent.agent.decisions import (
    Enter,
    PreStepReason,
    Reject,
    TurnStopping,
    resolve_stopping,
)
from minion_agent.llm import TextBlock, UserMessage


def _message(text: str = "hi") -> UserMessage:
    return UserMessage(content=(TextBlock(text=text),), timestamp=1)


def test_enter_carries_the_messages_entering_the_step() -> None:
    decision = Enter(messages=(_message(),))

    assert len(decision.messages) == 1


def test_enter_may_override_the_system_prompt_for_one_step() -> None:
    assert Enter(messages=(), system_override="different").system_override == "different"


def test_enter_may_limit_the_history_window_for_one_step() -> None:
    assert Enter(messages=(), history_window=6).history_window == 6


def test_reject_carries_a_reason() -> None:
    assert Reject(reason="not now").reason == "not now"


def test_pre_step_reasons_cover_pi_call_sites() -> None:
    """Derived from pi's actual call sites; not open for plugins to extend."""
    assert {reason.value for reason in PreStepReason} == {
        "initial",
        "tool_results",
        "steering",
        "next_turn",
        "continuation",
    }


def test_all_no_opinion_continues() -> None:
    """Matching pi's boolean default of false."""
    assert resolve_stopping([TurnStopping.NO_OPINION] * 3) is TurnStopping.CONTINUE


def test_an_empty_chain_continues() -> None:
    assert resolve_stopping([]) is TurnStopping.CONTINUE


def test_the_first_opinion_wins() -> None:
    decisions = [TurnStopping.NO_OPINION, TurnStopping.STOP, TurnStopping.CONTINUE]

    assert resolve_stopping(decisions) is TurnStopping.STOP


def test_a_later_stop_cannot_override_an_earlier_continue() -> None:
    """The trade-off recorded in section 6: order-dependent, consistent with
    the short-circuit pattern used by every other decision event."""
    decisions = [TurnStopping.CONTINUE, TurnStopping.STOP]

    assert resolve_stopping(decisions) is TurnStopping.CONTINUE
