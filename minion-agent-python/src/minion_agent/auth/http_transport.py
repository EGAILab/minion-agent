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
from collections.abc import Awaitable, Callable, Coroutine, Mapping
from dataclasses import dataclass, field
from typing import Any, Protocol

from .signal import Abortable

POLL_INTERVAL_SECONDS = 0.05
"""The same short, fixed poll interval `abortable_sleep` (`PROV-010`) already establishes for
checking a poll-based `Abortable` signal "promptly enough" without redesigning it."""


@dataclass(frozen=True, slots=True)
class HttpResponse:
    """This project's own minimal response type -- NOT `httpx.Response` re-exported.

    `status`/`reason_phrase` are available as soon as a transport returns this object, matching
    pinned Pi's own `fetch()` promise, which resolves once status/headers arrive, BEFORE the body
    is consumed. The body itself is consumed ONLY via `text()` (async, unlike a plain attribute)
    -- a caller that never calls `text()` never triggers a body read at all (`L11-SC-R022`,
    mandatory final-complete review): pinned Pi's own device-start `404` and device-poll
    `403`/`404` branches never call `.text()`/`.json()` at all, and a buffered-body design that
    reads the ENTIRE body before a caller can even inspect `status` cannot reproduce that -- a
    slow or never-completing body on one of those branches would hang the whole operation instead
    of returning/rejecting immediately, the way Pi's own code does. There is no `.json()` accessor
    here (unlike Pi's own `Response`) -- a caller parses `text()`'s own result via `json.loads`
    itself, exactly where `spec/auth.md`'s own contract already treats "parse the body as JSON" as
    the caller's own explicit step, not something baked into the response object.

    `HttpResponse` may be constructed two ways: with an ALREADY-KNOWN `body: bytes` (the common,
    synchronous-feeling case every test fixture and fake transport uses, since synthetic test
    data has no real I/O to defer), or with a `read_body` callable a REAL transport supplies to
    defer the actual network read until `text()` is first called (`HttpxTransport.post` below is
    the only production caller of the latter form). Either way, the result is cached after the
    first `text()` call (`object.__setattr__` on this otherwise-frozen dataclass -- the one
    deliberately mutable field, guarding against the underlying stream being consumed twice, which
    a real network response cannot support). `text()` decodes via `"utf-8-sig"`, not `"utf-8"`
    (`L11-SC-R023`, mandatory final-complete review): WHATWG Fetch's own body-text decoding strips
    exactly one LEADING UTF-8 byte-order mark before exposing the text to `Response.text()`/
    `.json()`, which pinned Pi relies on implicitly -- `"utf-8-sig"` reproduces this exactly
    (strips a leading BOM, leaves an interior `U+FEFF` untouched), composing correctly with the
    pre-existing `errors="replace"` fallback for malformed bytes.

    `reason_phrase` is pinned Pi's own `Response.statusText` (`L11-SC-R018`): the real HTTP
    reason phrase the server itself sent alongside `status`, NOT a hand-maintained lookup table
    (a fixed few-entry table, however large, can never cover every status a real server might
    send). A body that fails to read AFTER status/headers already arrived successfully -- a
    scenario pinned Pi's own two-phase `fetch()`-then-`.text()` model can represent -- comes back
    here as an EMPTY body on a non-2xx status (`HttpxTransport.post` below performs this
    translation, matching pinned Pi's own `text().catch(() => "")` non-2xx fallback), while the
    SAME failure on a 2xx status propagates as a raised exception from `text()` itself, matching
    pinned Pi's own uncaught `.json()` rejection on the success path.

    A branch that deliberately never calls `text()` at all -- pinned Pi's own device-start `404`
    and device-poll `403`/`404` branches, the exact scenario `read_body`/laziness above exists
    for -- MUST instead call `discard()` (`L11-SC-R022`, targeted re-review): the convergence
    redesign's own original scope decision assumed ordinary Python object-lifetime/GC cleanup
    would eventually release an abandoned response's underlying resources, but a REAL streamed
    `httpx` response and any client `HttpxTransport` itself constructed require an ASYNC close
    operation (`response.aclose()`/`client.aclose()`) to release their own connection -- confirmed
    live this is NOT something ordinary synchronous garbage collection/`__del__` can perform at
    all (there is no running event loop to await anything inside a GC finalizer), so an abandoned
    response was genuinely, silently left open, not merely un-warned-about; repeated device polling
    could abandon one per pending response. `discard()` is IDEMPOTENT and safe to call whether or
    not `text()` was already called (a no-op then, since `text()`'s own cleanup already ran) --
    exactly one of `text()`/`discard()` should be called per response, never neither."""

    status: int
    reason_phrase: str = ""
    body: bytes | None = None
    read_body: Callable[[], Awaitable[bytes]] | None = field(
        default=None, repr=False, compare=False
    )
    discard_body: Callable[[], Awaitable[None]] | None = field(
        default=None, repr=False, compare=False
    )

    async def text(self) -> str:
        body = self.body
        if body is None:
            assert self.read_body is not None, "HttpResponse has neither body= nor read_body="
            body = await self.read_body()
            object.__setattr__(self, "body", body)
        return body.decode("utf-8-sig", errors="replace")

    async def discard(self) -> None:
        if self.discard_body is not None:
            await self.discard_body()


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
    otherwise `coro`'s own result (or exception) is returned/propagated unchanged.

    `signal` is checked BEFORE `coro` is ever scheduled (`L11-SC-R015` -- confirmed live: a
    PRE-aborted signal combined with a fast-completing operation previously let both the start
    and the success happen, since the first `signal.aborted` check only ran after the first poll
    tick). Pi's own `fetch`, given an already-aborted `AbortSignal`, never issues the request at
    all -- it rejects immediately with `AbortError`; a pre-aborted `signal` here must likewise
    never start `coro`, not merely cancel it quickly after starting."""
    if signal is not None and signal.aborted:
        coro.close()
        raise TransportCancelled("the operation's own signal aborted")
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
            # `L11-SC-R016`/`L11-SC-R017` -- confirmed live: `httpx.AsyncClient()`'s own bare
            # defaults are `timeout=Timeout(timeout=5.0)` and `follow_redirects=False`, neither
            # of which matches pinned Pi's own `fetch`. Timeout enforcement must flow ONLY
            # through this project's own signal-based `run_cancellable`, never an independent
            # `httpx`-owned cap (an implicit 5s cap would truncate refresh's own certified 15s
            # `CombinedSignal` budget); `fetch`'s own default redirect mode follows redirects.
            client = httpx.AsyncClient(timeout=None, follow_redirects=True)

        # Two-phase send (`L11-SC-R018`; body-laziness `L11-SC-R022`, mandatory final-complete
        # review): `stream=True` returns as soon as status/headers arrive, BEFORE the body is
        # read, so a caller can inspect `status`/`reason_phrase` and decide whether to read the
        # body AT ALL -- pinned Pi's own device-start `404` and device-poll `403`/`404` branches
        # never call `.text()`/`.json()`, so this method must not either, and a REAL streamed
        # HTTP body that never completes must not block THIS call from returning (confirmed live
        # against the exact rejected candidate before this fix: a never-completing body on a 404
        # device-start response hung the whole operation). The actual body read is deferred
        # entirely to `HttpResponse.text()` (below), invoked only if/when a caller calls it --
        # matching pinned Pi's own two-phase `fetch()`-then-`.text()`/`.json()` model exactly,
        # where `refreshAccessToken`'s own `readTokenResponse(response, "refresh")` call (OUTSIDE
        # its own request-level `try`/`catch`) is what previously let a 2xx body-read failure get
        # mis-wrapped as `"OpenAI Codex token refresh error: ..."` instead of propagating raw --
        # the eager, single-call buffering below this comment used to collapse both phases into
        # one, before a caller ever had the chance to make that distinction.
        async def send_request() -> httpx.Response:
            request = client.build_request("POST", url, headers=dict(headers), content=body)
            return await client.send(request, stream=True)

        try:
            response = await run_cancellable(send_request(), signal)
        except BaseException:
            # Getting status/headers itself failed (or was cancelled) -- there is no deferred
            # path left that could ever close an owned client, so close it here, matching the
            # pre-existing unconditional-cleanup guarantee for this failure mode.
            if owns_client:
                await client.aclose()
            raise

        is_success_status = 200 <= response.status_code < 300

        # Closing the underlying resources is IDEMPOTENT and shared between `read_body` and
        # `discard_body` below (`L11-SC-R022`, targeted re-review): a real streamed `httpx`
        # response and any client this method itself constructed both require an ASYNC close
        # operation to release their connection -- confirmed live ordinary Python object-lifetime/
        # GC cleanup cannot perform this at all (there is no running event loop to await anything
        # inside a synchronous finalizer), so the response's own resources were genuinely, silently
        # left open whenever a caller deliberately never read the body (device-start's own `404`
        # branch, device-poll's own `403`/`404` branch) -- not merely un-warned-about. `closed`
        # guards against double-closing if both paths somehow ran.
        #
        # BOTH close attempts are individually exception-safe (`L11-SC-R022`, second targeted
        # re-review): pinned Pi exposes no explicit "close" operation at all for device-start's
        # own `404`/device-poll's own `403`/`404` branches -- their fixed status-only outcome
        # (`DeviceCodeNotEnabledError`/`PENDING`) is produced WITHOUT any body-stream cleanup step
        # being observable, let alone one that can fail. Confirmed live against the exact
        # candidate before this fix: a response whose `aclose()` itself raises let that raw
        # exception escape `discard()`, REPLACING the caller's own fixed Pi outcome with an
        # unrelated cleanup error -- and, since `response.aclose()` was awaited with no `finally`
        # of its own, a raising response close also skipped the owned client's own close
        # entirely. Neither ordinary close-attempt failure may raise past this function, and a
        # failure in ONE must not skip the OTHER.
        #
        # The owned-client close is attempted in a genuine `finally` (`L11-SC-R022`, third
        # targeted re-review), not merely a second sequential `suppress` block, so it ALSO runs
        # when `response.aclose()` itself raises `asyncio.CancelledError` -- a `BaseException`,
        # deliberately NOT caught by `contextlib.suppress(Exception)`, so cancellation still
        # propagates out of this function exactly as before -- confirmed live against the exact
        # candidate before this fix: a cancelled response close exited the function before the
        # owned client's own close was ever attempted at all, leaving it open even though the
        # cancellation itself correctly propagated.
        closed = False

        async def close_resources() -> None:
            nonlocal closed
            if closed:
                return
            closed = True
            try:
                with contextlib.suppress(Exception):
                    await response.aclose()
            finally:
                if owns_client:
                    with contextlib.suppress(Exception):
                        await client.aclose()

        async def read_body() -> bytes:
            # Whether a body-read failure is swallowed is STATUS-DEPENDENT, matching pinned Pi
            # exactly (`openai-codex.ts` -- `readTokenResponse`, `startOpenAICodexDeviceAuth`,
            # and `pollOpenAICodexDeviceAuth`'s own `poll()` all follow the identical pattern):
            # every non-2xx (`!response.ok`) branch reads the body via `.text().catch(() => "")`
            # (a body-read failure there becomes an empty string, never an exception); every 2xx
            # branch reads the body via `.json()` with NO catch at all (a body-read failure there
            # is an uncaught promise rejection that propagates through the existing call-site
            # boundary).
            #
            # The non-2xx catch is `Exception`, not `httpx.HTTPError` (targeted convergence
            # review, third round): JS's own `.catch(() => "")` catches ANY promise rejection
            # from the body-read operation, not one library-specific error hierarchy. A genuine
            # `TransportCancelled` (this method's own signal-abort translation, raised by
            # `run_cancellable` below) is explicitly excluded from that catch -- a real
            # cancellation must still propagate as a cancellation, never be swallowed into an
            # empty body, regardless of status. `asyncio.CancelledError` (`BaseException`, not
            # `Exception`, since Python 3.8) is likewise never caught here for the same reason.
            try:
                try:
                    await run_cancellable(response.aread(), signal)
                except TransportCancelled:
                    raise
                except Exception:
                    if is_success_status:
                        raise
                    return b""
                return response.content
            finally:
                await close_resources()

        return HttpResponse(
            status=response.status_code,
            reason_phrase=response.reason_phrase,
            read_body=read_body,
            discard_body=close_resources,
        )


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
