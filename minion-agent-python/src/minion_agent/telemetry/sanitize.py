"""The mandatory redaction boundary.

Ordering is the whole point (design spec section 7)::

    core / provider data
       -> sanitize + redact      <- single mandatory boundary
       -> safe structured telemetry
       -> sinks

If redaction were a listener among listeners, a sink registered earlier would
observe raw secrets and the guarantee would depend silently on registration
order.

Redaction is known-value: the runtime scrubs credentials it has been told
about, wherever they appear — including inside prompt content, which may carry
secrets the runtime never issued. It cannot catch an arbitrary string nothing
has declared to be a secret.
"""

from __future__ import annotations

from dataclasses import replace
from typing import Any

from .spans import Span

REDACTED = "[redacted]"


class Sanitizer:
    """Masks configured secret values anywhere they appear in a span."""

    def __init__(self) -> None:
        self._secrets: set[str] = set()

    def add_secret(self, value: str) -> None:
        """Declare a value to be masked. Empty strings are ignored."""
        if value:
            self._secrets.add(value)

    def remove_secret(self, value: str) -> None:
        """Stop masking `value`."""
        self._secrets.discard(value)

    def _scrub_text(self, text: str) -> str:
        for secret in self._secrets:
            text = text.replace(secret, REDACTED)
        return text

    def _scrub_value(self, value: Any) -> Any:
        if isinstance(value, str):
            return self._scrub_text(value)
        if isinstance(value, dict):
            return {key: self._scrub_value(item) for key, item in value.items()}
        if isinstance(value, list):
            return [self._scrub_value(item) for item in value]
        if isinstance(value, tuple):
            return tuple(self._scrub_value(item) for item in value)
        return value

    def scrub(self, span: Span) -> Span:
        """Return `span` with every declared secret masked."""
        if not self._secrets:
            return span
        return replace(
            span,
            attributes=self._scrub_value(span.attributes),
            error=self._scrub_text(span.error) if span.error is not None else None,
        )
