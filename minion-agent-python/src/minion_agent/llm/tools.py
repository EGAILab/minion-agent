"""Tool schemas: what the model is told a tool accepts.

Model-facing, so it lives with the rest of the provider-neutral vocabulary
rather than in the tools package. Two consequences follow, and both matter:
an adapter can translate a schema without importing the tool registry, and
`llm` keeps its rule of knowing nothing about layers above it.

`parameters` is JSON Schema. Keeping it a plain mapping rather than a pydantic
model is deliberate: the model-facing contract has to be language-neutral, and
a second implementation must be able to produce byte-identical output.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


def _canonical(value: Any) -> Any:
    """Recursively sort mapping keys so equal content compares equal."""
    if isinstance(value, dict):
        return {key: _canonical(value[key]) for key in sorted(value)}
    if isinstance(value, list | tuple):
        return [_canonical(item) for item in value]
    return value


@dataclass(frozen=True, slots=True)
class ToolSchema:
    """One tool as the model sees it."""

    name: str
    description: str
    parameters: dict[str, Any]

    def as_json(self) -> dict[str, Any]:
        """The canonical JSON-safe form.

        Key order is normalized because this form is content-addressed into
        the session log: a schema built with its keys in a different order is
        the same schema and must hash the same.
        """
        return {
            "name": self.name,
            "description": self.description,
            "parameters": _canonical(self.parameters),
        }
