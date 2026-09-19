"""`Result[T, E]`: `Ok`/`Err`, `is_ok`/`is_err` (`EXEC-001`)."""

from minion_agent.execution.result import Err, Ok, is_err, is_ok


def test_ok_carries_its_value() -> None:
    assert Ok(42).value == 42


def test_err_carries_its_error() -> None:
    assert Err("boom").error == "boom"


def test_is_ok_narrows_ok() -> None:
    result: Ok[int] | Err[str] = Ok(1)
    assert is_ok(result) is True
    assert is_err(result) is False


def test_is_err_narrows_err() -> None:
    result: Ok[int] | Err[str] = Err("x")
    assert is_err(result) is True
    assert is_ok(result) is False


def test_ok_equality() -> None:
    assert Ok(1) == Ok(1)
    assert Ok(1) != Ok(2)


def test_err_equality() -> None:
    assert Err("a") == Err("a")
    assert Err("a") != Err("b")
