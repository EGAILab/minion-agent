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


def _build_adapter(spec: dict[str, Any]) -> MockAdapter:
    """One `MockAdapter` per scenario adapter entry, scripted to respond identically for
    however many requests a scenario happens to make against it (registration/replacement/
    withdrawal scenarios never call `stream()`; failure-settlement scenarios call it once per
    declared model)."""
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
    # Scripted once per model this adapter serves, repeated generously: a scenario's own
    # `steps` decide how many times `stream()` is actually invoked, not this helper.
    script = [response] * 8
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
    service = LlmService()
    adapters = {entry["id"]: _build_adapter(entry) for entry in spec_doc["adapters"]}
    withdrawals: dict[str, list[Any]] = {}
    observations: dict[str, Any] = {}

    for step in spec_doc["steps"]:
        if "register" in step:
            adapter_id = step["register"]
            withdrawals.setdefault(adapter_id, []).append(service.register(adapters[adapter_id]))
        elif "withdraw" in step:
            for withdraw in withdrawals.get(step["withdraw"], []):
                withdraw()
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
            try:
                stream = service.stream(request)
            except UnknownModelError:
                observations[query["id"]] = {"resolve": {"found": False}}
                continue
            async for _ in stream:  # drain to completion; only WHO served it matters here
                pass
            # Public-API-only ownership check: MockAdapter's own `.requests` instrumentation
            # (already sanctioned for testability by Layer 02's own audit -- LLM-F009), not a
            # private LlmService attribute -- exactly one candidate adapter's own request log
            # grew by this call, since a resolve query never shares an identity across adapters.
            owner = next(
                adapter_id
                for adapter_id, candidate in adapters.items()
                if candidate.requests and candidate.requests[-1] == request
            )
            observations[query["id"]] = {"resolve": {"found": True, "adapter": owner}}
        elif "introspect" in query:
            current = sorted(
                (_identity_as_dict(model_id) for model_id in service.models()),
                key=lambda entry: (entry["provider"], entry["model"], entry["api"]),
            )
            observations[query["id"]] = {"models": current}

    return observations
