"""Tool schemas: what the model is told a tool accepts.

Model-facing, so it lives with the rest of the provider-neutral vocabulary
rather than in the tools package. Two consequences follow, and both matter:
an adapter can translate a schema without importing the tool registry, and
`llm` keeps its rule of knowing nothing about layers above it.

`parameters` is an object-valued JSON Schema representation compatible with
pinned Pi's `Tool.parameters` boundary (`packages/ai/src/types.ts::Tool<TParameters
extends TSchema>` -- TypeBox's `TSchema` is always object-shaped at this position;
Pi never produces a bare JSON-Schema-spec boolean value here). "Object-valued,"
not "arbitrary JSON Schema" (`L05-R005`) -- the value domain is a JSON Schema
object with standard keywords, never the boolean-shorthand forms `true`/`false`.
Keeping it a plain mapping rather than a pydantic model is deliberate: the
model-facing contract has to be language-neutral, and a second implementation
must be able to produce byte-identical output.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal


def _canonical(value: Any) -> Any:
    """Recursively sort mapping keys so equal content compares equal."""
    if isinstance(value, dict):
        return {key: _canonical(value[key]) for key in sorted(value)}
    if isinstance(value, list | tuple):
        return [_canonical(item) for item in value]
    return value


@dataclass(frozen=True, slots=True)
class JsonSchemaConstrainedSampling:
    """Pi's `{type: "json_schema", strict}` constrained-sampling variant."""

    strict: Literal["prefer", "require"]


@dataclass(frozen=True, slots=True)
class GrammarConstrainedSampling:
    """Pi's `{type: "grammar", variants}` constrained-sampling variant.

    `variants` mirrors pinned Pi's own **closed** `GrammarVariants = Partial<Record<
    GrammarFormat, string>>` (`packages/ai/src/types.ts`), where
    `GrammarFormat = "openai_lark" | "openai_regex"` -- exactly two independently
    optional named formats, never an open string-keyed map (`L05-R001`: an earlier
    draft of this type falsely attributed an open map to Pi). Unlike `api`/`provider`
    elsewhere in this project, which pinned Pi source itself leaves open, `GrammarFormat`
    is a closed TypeScript union at the pinned commit -- verified directly, not assumed
    by analogy."""

    openai_lark: str | None = None
    openai_regex: str | None = None


type ConstrainedSampling = JsonSchemaConstrainedSampling | GrammarConstrainedSampling
"""Pi's `Tool.constrainedSampling?: false | ConstrainedSamplingConfig`
(`packages/ai/src/types.ts`). Three states, all first-class:
`None` (absent -- no constrained-sampling preference stated), `False`
(explicitly disabled), or a config variant (`ToolSchema.constrained_sampling`).
Provider-specific enforcement/fallback is Real Providers (assurance Layer 11)
territory; Layer 05 only owns preserving the metadata end to end."""


@dataclass(frozen=True, slots=True)
class ToolSchema:
    """One tool as the model sees it."""

    name: str
    description: str
    parameters: dict[str, Any]
    constrained_sampling: ConstrainedSampling | Literal[False] | None = None

    def as_json(self) -> dict[str, Any]:
        """The canonical JSON-safe form.

        Key order is normalized because this form is content-addressed into
        the session log: a schema built with its keys in a different order is
        the same schema and must hash the same. `constrained_sampling` follows
        the project's established optional-field convention (`null` when
        absent, matching e.g. `response_model` in `spec/llm.md`), not key
        omission.
        """
        if self.constrained_sampling is None or self.constrained_sampling is False:
            sampling: Any = self.constrained_sampling
        elif isinstance(self.constrained_sampling, JsonSchemaConstrainedSampling):
            sampling = {"type": "json_schema", "strict": self.constrained_sampling.strict}
        else:
            variants: dict[str, str] = {}
            if self.constrained_sampling.openai_lark is not None:
                variants["openai_lark"] = self.constrained_sampling.openai_lark
            if self.constrained_sampling.openai_regex is not None:
                variants["openai_regex"] = self.constrained_sampling.openai_regex
            sampling = {"type": "grammar", "variants": variants}
        return {
            "name": self.name,
            "description": self.description,
            "parameters": _canonical(self.parameters),
            "constrained_sampling": _canonical(sampling),
        }
