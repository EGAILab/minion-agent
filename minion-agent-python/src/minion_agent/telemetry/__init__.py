"""Observability: a typed span vocabulary behind a mandatory sanitize boundary."""

from .sanitize import REDACTED, Sanitizer
from .service import RecordingSink, Sink, TelemetryService
from .spans import Span, SpanKind

__all__ = [
    "REDACTED",
    "RecordingSink",
    "Sanitizer",
    "Sink",
    "Span",
    "SpanKind",
    "TelemetryService",
]
