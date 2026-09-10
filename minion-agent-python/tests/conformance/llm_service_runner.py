"""Executes `conformance/agent/*.yaml` llm-service (Layer 10) scenarios.

Drives the real `LlmService`/`Adapter`/`MockAdapter` seam directly -- registers real adapters
through the real `register()` effect, withdraws through the real handle it returns, calls the real
`stream()` and drains it to its own terminal, and queries the real `LlmService._adapters`-backed
`models()`/resolution behavior. This module implements no registration, replacement, withdrawal,
or resolution logic itself; that is `LlmService`'s own job (`AI-029`/`AI-030`,
`spec/llm.md`'s own "Provider abstraction" section).
"""

from __future__ import annotations

from typing import Any

from minion_agent.llm.adapters.mock import MockAdapter, ScriptedResponse
from minion_agent.llm.errors import UnknownModelError
from minion_agent.llm.messages import StopReason, TextBlock
from minion_agent.llm.service import LlmService, ModelId, Request

_NO_TOOLS: tuple[Any, ...] = ()


def _identity(raw: dict[str, str]) -> ModelId:
    return ModelId(provider=raw["provider"], model=raw["model"], api=raw["api"])


def _identity_as_dict(model_id: ModelId) -> dict[str, str]:
    return {"provider": model_id.provider, "model": model_id.model, "api": model_id.api}


def _validate_references(spec_doc: dict[str, Any], expect: dict[str, Any]) -> None:
    """Reject malformed declarative references before any `LlmService`/`Adapter` object is
    constructed (mirrors `tool_registry_runner.py::_validate_references`'s own boundary): this
    validates the scenario description itself (structural fixture integrity), never registration/
    resolution/withdrawal semantics -- the real `LlmService` seam still owns all of that once
    execution begins.

    A `withdraw` naming an ALREADY-withdrawn handle is deliberately NOT rejected here -- repeated
    withdrawal of the same handle is a legitimate, idempotent no-op (`AI-030`), not malformed input.
    An observation id declared by a step/query but never named in `expect` is likewise legitimate
    -- a "setup-only" action exists to affect state (e.g. populate an adapter's own request log)
    without itself being asserted on.
    """
    adapter_ids = [entry["id"] for entry in spec_doc["adapters"]]
    duplicate_adapters = {i for i in adapter_ids if adapter_ids.count(i) > 1}
    if duplicate_adapters:
        raise ValueError(f"adapters[] declares duplicate id(s): {sorted(duplicate_adapters)!r}")
    adapter_id_set = set(adapter_ids)

    declared_handles: set[str] = set()
    observation_ids: list[str] = []

    for step in spec_doc["steps"]:
        if "register" in step:
            register = step["register"]
            if register["adapter"] not in adapter_id_set:
                raise ValueError(
                    f"register.adapter references {register['adapter']!r}, which no "
                    "adapters[] entry declares -- malformed canonical input"
                )
            handle = register["as"]
            if handle in declared_handles:
                raise ValueError(
                    f"register.as {handle!r} is reused by an earlier register step in the "
                    "same scenario -- each registration call must introduce its own handle id"
                )
            declared_handles.add(handle)
        elif "withdraw" in step:
            if step["withdraw"] not in declared_handles:
                raise ValueError(
                    f"withdraw references handle {step['withdraw']!r}, which no earlier "
                    "register step declares -- malformed canonical input"
                )
        elif "stream" in step:
            observation_ids.append(step["stream"]["as"])

    observation_ids.extend(query["id"] for query in spec_doc.get("queries", []))
    duplicate_observations = {i for i in observation_ids if observation_ids.count(i) > 1}
    if duplicate_observations:
        raise ValueError(
            f"queries[].id/steps[].stream.as share one namespace and must be unique within a "
            f"scenario; duplicate id(s): {sorted(duplicate_observations)!r}"
        )

    observation_id_set = set(observation_ids)
    dangling = set(expect) - observation_id_set
    if dangling:
        raise ValueError(
            f"expect references observation id(s) no query/steps[].stream declares: "
            f"{sorted(dangling)!r} -- malformed canonical input"
        )


def _owner_from_growth(before: dict[str, int], adapters: dict[str, MockAdapter]) -> str:
    """Exactly one candidate adapter's own request-log length must have grown by exactly one
    since `before` was snapshotted (`C10-C004`) -- zero or more than one is an error in the
    scenario or the runner, never a plausible outcome to normalize via `next()`'s own
    first-match behavior. Factored out so the guard itself is directly unit-testable against a
    synthetically-constructed zero-growth or multiple-growth case, not only through the full
    scenario-document runner (a naively-shared mock adapter object registered under two
    different fixture ids is the only way multiple-growth could arise in practice; this
    function does not care how `before`/`adapters` came to be shaped that way)."""
    grown = [
        adapter_id
        for adapter_id, candidate in adapters.items()
        if len(candidate.requests) - before[adapter_id] == 1
    ]
    assert len(grown) == 1, (
        f"expected exactly one adapter's request log to grow by one, found {grown!r}"
    )
    return grown[0]


def _max_possible_calls(spec_doc: dict[str, Any]) -> int:
    """An upper bound on how many `stream()` calls ANY single adapter could receive across the
    whole scenario (`L10-R007`): every `steps[].stream` action and every `queries[].resolve`
    query is at most one call to SOME one adapter, so the document's own total count of both is a
    safe bound for EVERY fixture's own script length, without predicting which adapter the
    service will actually resolve to (a fixed, hard-coded cap silently converts a schema-valid
    scenario's own later calls into a fabricated `MockAdapter` exhaustion error that nothing in
    the scenario, schema, or `LlmService` itself declares)."""
    stream_calls = sum(1 for step in spec_doc["steps"] if "stream" in step)
    resolve_calls = sum(1 for query in spec_doc.get("queries", []) if "resolve" in query)
    return stream_calls + resolve_calls


def _build_adapter(spec: dict[str, Any], script_length: int) -> MockAdapter:
    """One `MockAdapter` per scenario adapter entry, scripted to respond identically for
    `script_length` calls -- a safe, non-predictive upper bound on how many requests this
    specific adapter could actually receive (`_max_possible_calls`), not a fixed cap unrelated to
    the scenario's own shape."""
    if spec["behavior"] == "ok":
        response = ScriptedResponse(
            content=(TextBlock(text="ok"),),
            stop_reason=StopReason.STOP,
        )
    else:
        response = ScriptedResponse(
            content=(),
            stop_reason=StopReason.ERROR,
            error_message=spec["reject_message"],
        )
    script = [response] * script_length
    adapter = MockAdapter(script)
    adapter.provider = spec["provider"]  # type: ignore[misc]
    adapter.api = spec["api"]  # type: ignore[misc]
    adapter.models = frozenset(spec["models"])  # type: ignore[misc]
    return adapter


async def run_llm_service_scenario(document: dict[str, Any]) -> dict[str, Any]:
    """Run one `llm_service` scenario and return `{observation_id: {...}}` observations, keyed
    by both `queries[].id` (evaluated once, after every step) and `steps[].stream.as` (recorded
    at the point each `stream()` action actually runs)."""
    spec_doc = document["llm_service"]
    _validate_references(spec_doc, document.get("expect", {}))
    service = LlmService()
    script_length = _max_possible_calls(spec_doc)
    adapters = {entry["id"]: _build_adapter(entry, script_length) for entry in spec_doc["adapters"]}
    handles: dict[str, Any] = {}
    observations: dict[str, Any] = {}

    for step in spec_doc["steps"]:
        if "register" in step:
            register = step["register"]
            handles[register["as"]] = service.register(adapters[register["adapter"]])
        elif "withdraw" in step:
            # Idempotent by design (AI-030): a handle may be withdrawn more than once, and this
            # runner calls the SAME handle object again rather than tracking whether it has
            # already fired -- exercising the real production idempotency, not simulating it.
            handles[step["withdraw"]]()
        elif "stream" in step:
            action = step["stream"]
            model = _identity(action["identity"])
            request = Request(model=model, system="", messages=(), tools=_NO_TOOLS)
            try:
                stream = service.stream(request)
            except UnknownModelError:
                observations[action["as"]] = {"raised": True}
                continue
            terminal = None
            async for chunk in stream:
                terminal = chunk
            assert terminal is not None
            settled = "ok" if terminal.partial.stop_reason == StopReason.STOP else "error"
            observation: dict[str, Any] = {"settled": settled}
            if terminal.partial.error_message is not None:
                observation["error_message"] = terminal.partial.error_message
            observations[action["as"]] = observation

    for query in spec_doc.get("queries", []):
        if "resolve" in query:
            model = _identity(query["resolve"])
            request = Request(model=model, system="", messages=(), tools=_NO_TOOLS)
            # Public-API-only ownership check: MockAdapter's own `.requests` instrumentation
            # (already sanctioned for testability by Layer 02's own audit -- LLM-F009), not a
            # private LlmService attribute. Snapshot every candidate's own request-log length
            # BEFORE the call -- `MockAdapter.stream()` appends synchronously at call time,
            # which `LlmService.stream()` itself invokes eagerly as soon as it resolves an
            # adapter, so the snapshot must precede that call, not merely precede draining the
            # returned stream. Then, after draining, collect every adapter whose own count grew
            # by exactly one -- never a value-equality search over request content (L10-R004:
            # two adapters called through different identities produce VALUE-EQUAL `Request`
            # objects since this runner varies only `model`, so an equality search can silently
            # pick the wrong, registration-order-first adapter). Exactly one candidate must have
            # grown by exactly one; zero or more than one is an error in the scenario or the
            # runner, not a plausible outcome to normalize via `next()`'s own first-match
            # behavior (C10-C004).
            before = {
                adapter_id: len(candidate.requests) for adapter_id, candidate in adapters.items()
            }
            try:
                stream = service.stream(request)
            except UnknownModelError:
                observations[query["id"]] = {"resolve": {"found": False}}
                continue
            async for _ in stream:  # drain to completion; only WHO served it matters here
                pass
            owner = _owner_from_growth(before, adapters)
            observations[query["id"]] = {"resolve": {"found": True, "adapter": owner}}
        elif "introspect" in query:
            current = sorted(
                (_identity_as_dict(model_id) for model_id in service.models()),
                key=lambda entry: (entry["provider"], entry["model"], entry["api"]),
            )
            observations[query["id"]] = {"models": current}

    return observations
