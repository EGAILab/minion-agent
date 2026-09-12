"""Harness/schema-integrity tests for the llm-service runner's own reference validation.

`C10-C002`/`C10-C003`: these prove the runner rejects malformed *fixture* references (a
`register.adapter`, `register.as`, or `withdraw` naming something no earlier declaration
introduces, a duplicate `adapters[].id`, a colliding observation id, or a dangling `expect` key)
directly, before any `LlmService`/`Adapter` side effect runs -- not that the real `LlmService` seam
produces some particular registration/resolution outcome. That is what the canonical `conformance/
agent/llm-service-*.yaml` scenarios in `test_llm_service_conformance.py` prove; these are
deliberately not canonical scenarios themselves; they exercise the runner module directly with
hand-built, schema-independent documents.

Two cases the runner deliberately does NOT reject, also proven here: withdrawing an
already-withdrawn handle (idempotent, `AI-030`) and a `steps[].stream` observation id `expect`
never names (a legitimate "setup-only" stream action). A `queries[].id` `expect` never names IS
rejected (`L10-C002`) -- unlike a stream action, a query exists only to be observed, so one
nothing ever asserts on is almost certainly a scenario-authoring mistake.
"""

from typing import Any

import pytest

from minion_agent.llm.adapters.mock import MockAdapter
from minion_agent.llm.service import ModelId, Request

from .llm_service_runner import _owner_from_growth, run_llm_service_scenario


def _adapter(id_: str = "a", models: list[str] | None = None) -> dict[str, Any]:
    return {
        "id": id_,
        "provider": "mock",
        "api": "mock",
        "models": models or ["alpha"],
        "behavior": "ok",
    }


def _document(
    adapters: list[dict[str, Any]],
    steps: list[dict[str, Any]],
    queries: list[dict[str, Any]] | None = None,
    expect: dict[str, Any] | None = None,
) -> dict[str, Any]:
    spec: dict[str, Any] = {"adapters": adapters, "steps": steps}
    if queries is not None:
        spec["queries"] = queries
    return {"llm_service": spec, "expect": expect or {}}


async def test_duplicate_adapter_id_rejected() -> None:
    document = _document(
        adapters=[_adapter("a"), _adapter("a")],
        steps=[{"register": {"adapter": "a", "as": "h"}}],
    )
    with pytest.raises(ValueError, match="duplicate"):
        await run_llm_service_scenario(document)


async def test_unknown_register_adapter_rejected() -> None:
    document = _document(
        adapters=[_adapter("a")],
        steps=[{"register": {"adapter": "missing", "as": "h"}}],
    )
    with pytest.raises(ValueError, match="missing"):
        await run_llm_service_scenario(document)


async def test_reused_register_as_handle_rejected() -> None:
    document = _document(
        adapters=[_adapter("a"), _adapter("b")],
        steps=[
            {"register": {"adapter": "a", "as": "h"}},
            {"register": {"adapter": "b", "as": "h"}},
        ],
    )
    with pytest.raises(ValueError, match="reused"):
        await run_llm_service_scenario(document)


async def test_unknown_withdraw_handle_rejected() -> None:
    document = _document(
        adapters=[_adapter("a")],
        steps=[{"withdraw": "missing"}],
    )
    with pytest.raises(ValueError, match="missing"):
        await run_llm_service_scenario(document)


async def test_duplicate_observation_id_rejected() -> None:
    document = _document(
        adapters=[_adapter("a")],
        steps=[
            {"register": {"adapter": "a", "as": "h"}},
            {
                "stream": {
                    "identity": {"provider": "mock", "model": "alpha", "api": "mock"},
                    "as": "dup",
                }
            },
        ],
        queries=[{"id": "dup", "introspect": "models"}],
    )
    with pytest.raises(ValueError, match="dup"):
        await run_llm_service_scenario(document)


async def test_dangling_expect_reference_rejected() -> None:
    document = _document(
        adapters=[_adapter("a")],
        steps=[{"register": {"adapter": "a", "as": "h"}}],
        queries=[{"id": "q", "introspect": "models"}],
        expect={"nonexistent": {"models": []}},
    )
    with pytest.raises(ValueError, match="nonexistent"):
        await run_llm_service_scenario(document)


async def test_withdrawing_an_already_withdrawn_handle_is_not_rejected() -> None:
    document = _document(
        adapters=[_adapter("a")],
        steps=[
            {"register": {"adapter": "a", "as": "h"}},
            {"withdraw": "h"},
            {"withdraw": "h"},
        ],
        queries=[{"id": "q", "introspect": "models"}],
        expect={"q": {"models": []}},
    )
    observed = await run_llm_service_scenario(document)
    assert observed["q"] == {"models": []}


async def test_duplicate_queries_id_rejected() -> None:
    document = _document(
        adapters=[_adapter("a")],
        steps=[{"register": {"adapter": "a", "as": "h"}}],
        queries=[
            {"id": "dup", "introspect": "models"},
            {"id": "dup", "resolve": {"provider": "mock", "model": "alpha", "api": "mock"}},
        ],
    )
    with pytest.raises(ValueError, match="dup"):
        await run_llm_service_scenario(document)


def test_owner_from_growth_raises_on_zero_growth() -> None:
    """`C10-C004`: a synthetically-constructed case where NO candidate's own request-log grew
    must raise, never silently returning some arbitrary adapter id."""
    a, b = MockAdapter([]), MockAdapter([])
    with pytest.raises(AssertionError, match="exactly one"):
        _owner_from_growth({"a": 0, "b": 0}, {"a": a, "b": b})


def test_owner_from_growth_raises_on_multiple_growth() -> None:
    """`C10-C004`: a naively-shared mock adapter object registered under two different fixture
    ids would make BOTH `adapters.items()` entries show growth for a single call -- the guard
    must raise rather than pick a first match (`next()`'s own old failure mode)."""
    shared = MockAdapter([])
    before = {"a": len(shared.requests), "b": len(shared.requests)}
    shared.requests.append(_dummy_request())
    with pytest.raises(AssertionError, match="exactly one"):
        _owner_from_growth(before, {"a": shared, "b": shared})


def test_owner_from_growth_returns_the_sole_grown_adapter() -> None:
    a, b = MockAdapter([]), MockAdapter([])
    before = {"a": len(a.requests), "b": len(b.requests)}
    a.requests.append(_dummy_request())
    assert _owner_from_growth(before, {"a": a, "b": b}) == "a"


def _dummy_request() -> Request:
    return Request(model=ModelId("mock", "alpha", "mock"), system="", messages=())


async def test_a_declared_observation_id_expect_never_names_is_not_rejected() -> None:
    document = _document(
        adapters=[_adapter("a")],
        steps=[
            {"register": {"adapter": "a", "as": "h"}},
            {
                "stream": {
                    "identity": {"provider": "mock", "model": "alpha", "api": "mock"},
                    "as": "setup_only",
                }
            },
        ],
        queries=[{"id": "q", "introspect": "models"}],
        expect={"q": {"models": [{"provider": "mock", "model": "alpha", "api": "mock"}]}},
    )
    observed = await run_llm_service_scenario(document)
    assert "setup_only" in observed
    assert observed["q"] == {"models": [{"provider": "mock", "model": "alpha", "api": "mock"}]}


async def test_an_unasserted_query_is_rejected() -> None:
    """`L10-C002`: unlike a setup-only stream action, a query exists only to be observed -- one
    `expect` never names is rejected, not silently permitted."""
    document = _document(
        adapters=[_adapter("a")],
        steps=[{"register": {"adapter": "a", "as": "h"}}],
        queries=[{"id": "unchecked", "introspect": "models"}],
        expect={},
    )
    with pytest.raises(ValueError, match="unchecked"):
        await run_llm_service_scenario(document)
