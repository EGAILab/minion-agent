"""Provider-neutral content blocks.

These are what a model sees. An adapter translates them to and from its
provider's wire format; nothing above this layer knows that format exists.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True, slots=True)
class TextBlock:
    """Ordinary model-visible text."""

    text: str
    text_signature: str | None = None
    """Opaque provider replay metadata (design spec section 4). Preserves
    Responses-family message item identity/phase; core code persists this
    string without assigning it provider-independent meaning."""


@dataclass(frozen=True, slots=True)
class ThinkingBlock:
    """Reasoning content, when a provider exposes it separately from text."""

    thinking: str
    thinking_signature: str | None = None
    """Opaque provider replay metadata: the complete provider reasoning item
    needed for Responses-family replay (design spec section 4)."""
    redacted: bool = False
    """Whether the provider redacted this thinking block's visible text."""


@dataclass(frozen=True, slots=True)
class ImageBlock:
    """An image the model can see.

    Carries only what an adapter needs to translate it. Whether the bytes
    travel inline or by reference is an implementation choice; the mime type
    and the model-visible presence of an image are not.

    A logged reference must be immutable (design spec section 4) — a mutable
    path or URL would break request reconstruction silently, because the bytes
    the model saw and the bytes reconstruction fetches could differ with
    nothing detecting it. This type does not enforce immutability; the session
    layer resolves references to content-addressed artifacts before dispatch.
    """

    mime_type: str
    data: bytes | None = None
    reference: str | None = None

    def __post_init__(self) -> None:
        if (self.data is None) == (self.reference is None):
            raise ValueError(
                "ImageBlock requires exactly one of `data` or `reference`; "
                f"got data={self.data is not None}, reference={self.reference is not None}"
            )


@dataclass(frozen=True, slots=True)
class ToolCallBlock:
    """A tool invocation the model requested."""

    id: str
    name: str
    arguments: dict[str, Any]
    thought_signature: str | None = None
    """Opaque provider-specific replay metadata used by APIs that need it
    (design spec section 4); cross-model transformation strips it."""
    namespace: str | None = None
    """The tool namespace this call belongs to, when a provider distinguishes
    tools by more than name alone."""


type ContentBlock = TextBlock | ThinkingBlock | ImageBlock | ToolCallBlock
