"""Eager-side LLM errors.

These raise *before* a stream is returned. Once a stream exists, nothing
escapes iteration — failures ride the stream as a terminal error chunk
(design spec section 4).
"""


class LlmError(Exception):
    """Base for eager-side LLM errors."""


class UnknownModelError(LlmError):
    """A model or provider was requested that no adapter supplies."""


class AdapterProtocolError(LlmError):
    """An adapter violated the stream contract.

    Not an in-band model failure — a bug in the adapter — so it raises rather
    than being encoded as a stopped message.
    """
