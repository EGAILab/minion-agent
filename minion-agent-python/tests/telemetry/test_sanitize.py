"""Redaction is a boundary, not a listener — it runs before any sink sees a span."""

from minion_agent.telemetry.sanitize import REDACTED, Sanitizer
from minion_agent.telemetry.spans import Span, SpanKind


def _span(**attributes: object) -> Span:
    return Span(
        kind=SpanKind.PROVIDER_REQUEST, name="request", attributes=dict(attributes)
    )


def test_a_configured_secret_is_masked() -> None:
    sanitizer = Sanitizer()
    sanitizer.add_secret("sk-abc123")

    scrubbed = sanitizer.scrub(_span(authorization="Bearer sk-abc123"))

    assert "sk-abc123" not in str(scrubbed.attributes)
    assert REDACTED in scrubbed.attributes["authorization"]


def test_a_secret_is_masked_wherever_it_appears() -> None:
    """Prompt content may carry a secret the runtime never issued."""
    sanitizer = Sanitizer()
    sanitizer.add_secret("hunter2")

    scrubbed = sanitizer.scrub(_span(prompt="the user pasted hunter2 into a note"))

    assert "hunter2" not in scrubbed.attributes["prompt"]


def test_nested_attribute_values_are_scrubbed() -> None:
    sanitizer = Sanitizer()
    sanitizer.add_secret("sk-abc123")

    scrubbed = sanitizer.scrub(_span(headers={"auth": ["Bearer sk-abc123"]}))

    assert "sk-abc123" not in str(scrubbed.attributes)


def test_tuple_values_are_scrubbed() -> None:
    sanitizer = Sanitizer()
    sanitizer.add_secret("sk-abc123")

    scrubbed = sanitizer.scrub(_span(pair=("sk-abc123", 1)))

    assert "sk-abc123" not in str(scrubbed.attributes)


def test_non_secret_content_is_untouched() -> None:
    sanitizer = Sanitizer()
    sanitizer.add_secret("sk-abc123")

    scrubbed = sanitizer.scrub(_span(model="mock-1", tokens=42))

    assert scrubbed.attributes == {"model": "mock-1", "tokens": 42}


def test_removing_a_secret_stops_masking_it() -> None:
    sanitizer = Sanitizer()
    sanitizer.add_secret("sk-abc123")
    sanitizer.remove_secret("sk-abc123")

    scrubbed = sanitizer.scrub(_span(authorization="Bearer sk-abc123"))

    assert scrubbed.attributes["authorization"] == "Bearer sk-abc123"


def test_empty_secrets_are_ignored() -> None:
    """An empty string is a substring of everything; masking it would destroy
    every attribute."""
    sanitizer = Sanitizer()
    sanitizer.add_secret("")

    scrubbed = sanitizer.scrub(_span(model="mock-1"))

    assert scrubbed.attributes["model"] == "mock-1"


def test_the_error_field_is_scrubbed_too() -> None:
    sanitizer = Sanitizer()
    sanitizer.add_secret("sk-abc123")
    span = Span(
        kind=SpanKind.PROVIDER_REQUEST,
        name="request",
        attributes={},
        error="auth failed for sk-abc123",
    )

    assert "sk-abc123" not in (sanitizer.scrub(span).error or "")


def test_a_span_without_an_error_survives_scrubbing() -> None:
    sanitizer = Sanitizer()
    sanitizer.add_secret("sk-abc123")

    assert sanitizer.scrub(_span(model="mock-1")).error is None
