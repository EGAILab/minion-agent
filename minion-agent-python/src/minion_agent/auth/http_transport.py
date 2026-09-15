"""Injectable outbound HTTP transport seam for `PROV-012` (owner-approved `httpx` implementation
mechanics, `GOVERNANCE_SOURCE` at
`https://github.com/EGAILab/minion-agent/issues/29#issuecomment-5659001629`).

Per the owner's own six constraints: the language-neutral contract (`spec/auth.md`'s own "Token
exchange and refresh" and "Device-code flow" sections) describes HTTP behavior, never `httpx`
mechanics; provider/auth code depends on this INJECTABLE seam, never a concrete transport directly;
tests use a deterministic fake transport, never live network; cancellation maps through this
project's own `Abortable` contract, not `httpx`-defined semantics; `httpx` types never appear in a
shared, language-neutral type -- `HttpResponse` below is this project's own minimal type.
"""

from __future__ import annotations

import asyncio
import contextlib
from collections.abc import Coroutine, Mapping
from dataclasses import dataclass
from typing import Any, Protocol

from .signal import Abortable

POLL_INTERVAL_SECONDS = 0.05
"""The same short, fixed poll interval `abortable_sleep` (`PROV-010`) already establishes for
checking a poll-based `Abortable` signal "promptly enough" without redesigning it."""


@dataclass(frozen=True, slots=True)
class HttpResponse:
    """This project's own minimal response type -- NOT `httpx.Response` re-exported. The body is
    already fully read by the time a transport returns this (matching `httpx`'s own default
    non-streaming behavior), so `text` below is synchronous, unlike pinned Pi's own async
    `Response.text()` -- a disclosed implementation-mechanics mapping, not an observable behavior
    change: by the time it is called, no further I/O can fail. There is no `.json()` accessor here
    (unlike Pi's own `Response`) -- a caller parses `.text()`/`.body` via `json.loads` itself,
    exactly where `spec/auth.md`'s own contract already treats "parse the body as JSON" as the
    caller's own explicit step, not something baked into the response object."""

    status: int
    body: bytes

    def text(self) -> str:
        return self.body.decode("utf-8", errors="replace")


class HttpTransport(Protocol):
    """The injectable seam. A concrete implementation (`HttpxTransport` below, or a test's own
    fake) performs exactly one outbound POST and returns this project's own `HttpResponse` --
    never raising for a non-2xx status (that is the caller's own concern, matching pinned Pi's own
    `fetch`, which likewise never rejects merely for a non-2xx response), only for a genuine
    request-level failure (the request never completed at all) or cancellation."""

    async def post(
        self, url: str, *, headers: Mapping[str, str], body: bytes, signal: Abortable
    ) -> HttpResponse: ...


class TransportCancelled(Exception):
    """Raised by `run_cancellable` when `signal` aborts before the underlying coroutine
    completes. This project's own established `Abortable`/`RunSignal` is POLL-BASED BY CERTIFIED
    DESIGN (Layer 09) and has no push/event mechanism -- exactly the same already-disclosed
    constraint `abortable_sleep` (`PROV-010`) works within for timers. This is the analogous
    poll-based mapping for a DIFFERENT primitive (an in-flight network request): running the
    request as a cancellable task and polling `signal` at a short, fixed interval, tearing the
    task down (not merely abandoning it) the moment `signal` aborts -- reproducing pinned Pi's own
    `fetch`'s native push-based `AbortSignal` cancellation's OBSERVABLE effect (genuine early
    termination, a caller sees an error promptly, not only after the request would have finished
    on its own), not `Models`-level `raceWithAbortSignal`'s own materially narrower "stop waiting,
    do not cancel" behavior (`PROV-013`, deferred, not reused here)."""


async def run_cancellable[T](coro: Coroutine[Any, Any, T], signal: Abortable | None) -> T:
    """Race `coro` against `signal`, polling at `POLL_INTERVAL_SECONDS`. If `signal` aborts first,
    the underlying task is cancelled (not merely abandoned) and `TransportCancelled` is raised;
    otherwise `coro`'s own result (or exception) is returned/propagated unchanged."""
    task: asyncio.Task[T] = asyncio.ensure_future(coro)
    try:
        while True:
            done, _pending = await asyncio.wait({task}, timeout=POLL_INTERVAL_SECONDS)
            if task in done:
                return task.result()
            if signal is not None and signal.aborted:
                task.cancel()
                with contextlib.suppress(BaseException):
                    await task
                raise TransportCancelled("the operation's own signal aborted")
    finally:
        if not task.done():
            task.cancel()


class HttpxTransport:
    """The `httpx`-backed conforming implementation of `HttpTransport`. `httpx` itself is an
    implementation-mechanics detail confined entirely to this one class -- no `httpx` type crosses
    this seam's own boundary."""

    __slots__ = ("_client",)

    def __init__(self, client: Any | None = None) -> None:
        self._client = client

    async def post(
        self, url: str, *, headers: Mapping[str, str], body: bytes, signal: Abortable
    ) -> HttpResponse:
        import httpx

        client = self._client
        owns_client = client is None
        if client is None:
            client = httpx.AsyncClient()

        async def do_request() -> HttpResponse:
            response = await client.post(url, headers=dict(headers), content=body)
            return HttpResponse(status=response.status_code, body=response.content)

        try:
            return await run_cancellable(do_request(), signal)
        finally:
            if owns_client:
                await client.aclose()


async def fetch_with_login_cancellation(
    transport: HttpTransport,
    url: str,
    *,
    headers: Mapping[str, str],
    body: bytes,
    signal: Abortable,
) -> HttpResponse:
    """Pi's own `fetchWithLoginCancellation` (`openai-codex.ts:115-124`): if the underlying
    request fails WHILE `signal` is (by then) aborted, raise the FIXED message `"Login
    cancelled"`, discarding the underlying error entirely; any OTHER failure (`signal` not
    aborted) propagates unchanged. This is a post-hoc check of `signal.aborted` at the moment of
    failure, exactly matching Pi's own simple `if (init.signal?.aborted)` check -- not a claim
    that `signal` specifically CAUSED the failure. Shared by exchange, device-start, and
    device-poll (`L11-SC-R004`) -- refresh does NOT use this wrapper (see `refresh_fetch`)."""
    try:
        return await transport.post(url, headers=headers, body=body, signal=signal)
    except Exception as error:
        if signal.aborted:
            raise LoginCancelledError("Login cancelled") from error
        raise


class LoginCancelledError(Exception):
    """The fixed `"Login cancelled"` message `fetch_with_login_cancellation` raises."""


async def refresh_fetch(
    transport: HttpTransport,
    url: str,
    *,
    headers: Mapping[str, str],
    body: bytes,
    signal: Abortable,
) -> HttpResponse:
    """Pi's own `refreshAccessToken` (`openai-codex.ts:171-189`): ANY request-level failure --
    not a non-2xx response, which is the caller's own concern -- is wrapped as
    `"OpenAI Codex token refresh error: {message}"`, with NO cancellation-specific translation
    (`L11-SC-R004`'s own corrected rationale: refresh's own caller supplies a `CombinedSignal`
    combining an optional caller signal AND a fixed timeout, not a purely user-driven
    cancellation, so `fetch_with_login_cancellation`'s own "Login cancelled" message would not be
    the right uniform wrapping here)."""
    try:
        return await transport.post(url, headers=headers, body=body, signal=signal)
    except Exception as error:
        raise OAuthRefreshTransportError(f"OpenAI Codex token refresh error: {error}") from error


class OAuthRefreshTransportError(Exception):
    """`refresh_fetch`'s own uniform request-level failure wrapping."""
