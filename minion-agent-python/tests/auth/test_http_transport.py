"""Injectable HTTP transport seam (`PROV-012`) -- `run_cancellable`,
`fetch_with_login_cancellation`, and `refresh_fetch`. No test performs a live network call;
`HttpxTransport` itself is exercised only through its own `run_cancellable` composition, never
against a real socket."""

from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator

import httpx
import pytest

from minion_agent.auth.http_transport import (
    HttpResponse,
    HttpTransport,
    HttpxTransport,
    LoginCancelledError,
    OAuthRefreshTransportError,
    TransportCancelled,
    fetch_with_login_cancellation,
    refresh_fetch,
    run_cancellable,
)
from minion_agent.runtime.signal import RunAbortController


class _FakeTransport:
    """A deterministic, in-memory `HttpTransport` -- either returns a scripted response or
    raises a scripted exception; never touches a real socket."""

    def __init__(
        self, *, response: HttpResponse | None = None, error: Exception | None = None
    ) -> None:
        self._response = response
        self._error = error
        self.calls = 0

    async def post(self, url: str, *, headers: object, body: bytes, signal: object) -> HttpResponse:
        self.calls += 1
        if self._error is not None:
            raise self._error
        assert self._response is not None
        return self._response


def _transport(
    response: HttpResponse | None = None, error: Exception | None = None
) -> HttpTransport:
    return _FakeTransport(response=response, error=error)


async def test_run_cancellable_returns_result_when_not_aborted() -> None:
    async def quick() -> int:
        return 42

    assert await run_cancellable(quick(), None) == 42


async def test_run_cancellable_propagates_underlying_exception_unchanged() -> None:
    async def failing() -> int:
        raise ValueError("boom")

    with pytest.raises(ValueError, match="boom"):
        await run_cancellable(failing(), None)


async def test_run_cancellable_raises_transport_cancelled_when_signal_aborts_first() -> None:
    controller = RunAbortController()

    async def never_completes() -> int:
        await asyncio.Event().wait()
        return 0  # pragma: no cover -- never reached

    controller.abort()
    with pytest.raises(TransportCancelled):
        await run_cancellable(never_completes(), controller.signal)


async def test_run_cancellable_cancels_inner_task_when_outer_await_is_cancelled() -> None:
    """The `finally` block's own safety-net `task.cancel()` covers a DIFFERENT path from the
    abort-triggered one above: the coroutine `await`ing `run_cancellable` itself is cancelled
    (e.g. an outer task teardown) while the inner task is still running -- the inner task must
    not be left orphaned."""
    inner_task_was_cancelled = False

    async def never_completes() -> int:
        nonlocal inner_task_was_cancelled
        try:
            await asyncio.Event().wait()
            return 0  # pragma: no cover -- never reached
        except asyncio.CancelledError:
            inner_task_was_cancelled = True
            raise

    outer = asyncio.ensure_future(run_cancellable(never_completes(), None))
    await asyncio.sleep(0)  # let `outer` start and create the inner task
    outer.cancel()
    with pytest.raises(asyncio.CancelledError):
        await outer
    assert inner_task_was_cancelled


async def test_run_cancellable_cancels_the_underlying_task_on_abort() -> None:
    """Abort happens AFTER the task has already started (mid-flight) -- distinct from
    `test_run_cancellable_pre_aborted_signal_never_starts_the_operation` below (`L11-SC-R015`),
    where the signal is already aborted BEFORE the operation is ever scheduled."""
    controller = RunAbortController()
    task_was_cancelled = False
    started = asyncio.Event()

    async def observe_cancellation() -> int:
        nonlocal task_was_cancelled
        try:
            started.set()
            await asyncio.Event().wait()
            return 0  # pragma: no cover -- never reached
        except asyncio.CancelledError:
            task_was_cancelled = True
            raise

    outer = asyncio.ensure_future(run_cancellable(observe_cancellation(), controller.signal))
    await started.wait()
    controller.abort()
    with pytest.raises(TransportCancelled):
        await outer
    assert task_was_cancelled


async def test_run_cancellable_pre_aborted_signal_never_starts_the_operation() -> None:
    """`L11-SC-R015` -- confirmed live: an already-aborted signal must prevent the underlying
    operation from ever starting, not merely cancel it quickly after starting. A fast-completing
    operation combined with a pre-aborted signal previously let both the start AND the success
    happen (the original `signal.aborted` check only ran after the first poll tick), unlike
    Pi's own `fetch`, which never issues a request for an already-aborted `AbortSignal`."""
    controller = RunAbortController()
    controller.abort()
    started = False

    async def quick() -> int:
        nonlocal started
        started = True
        return 42

    with pytest.raises(TransportCancelled):
        await run_cancellable(quick(), controller.signal)
    assert not started


async def test_fetch_with_login_cancellation_returns_response_on_success() -> None:
    controller = RunAbortController()
    response = HttpResponse(status=200, body=b"{}")
    transport = _transport(response=response)
    result = await fetch_with_login_cancellation(
        transport, "https://example.test", headers={}, body=b"", signal=controller.signal
    )
    assert result is response


async def test_fetch_with_login_cancellation_translates_to_login_cancelled_when_aborted() -> None:
    """Post-hoc `signal.aborted` check at the moment of failure (`L11-SC-R004`) -- any transport
    failure while the signal happens to already be aborted is normalized, discarding the
    underlying error entirely."""
    controller = RunAbortController()
    controller.abort()
    transport = _transport(error=ConnectionError("unrelated network failure"))
    with pytest.raises(LoginCancelledError, match=r"^Login cancelled$"):
        await fetch_with_login_cancellation(
            transport, "https://example.test", headers={}, body=b"", signal=controller.signal
        )


async def test_fetch_with_login_cancellation_propagates_failure_unchanged_when_not_aborted() -> (
    None
):
    controller = RunAbortController()
    transport = _transport(error=ConnectionError("dns failure"))
    with pytest.raises(ConnectionError, match="dns failure"):
        await fetch_with_login_cancellation(
            transport, "https://example.test", headers={}, body=b"", signal=controller.signal
        )


async def test_refresh_fetch_returns_response_on_success() -> None:
    controller = RunAbortController()
    response = HttpResponse(status=200, body=b"{}")
    transport = _transport(response=response)
    result = await refresh_fetch(
        transport, "https://example.test", headers={}, body=b"", signal=controller.signal
    )
    assert result is response


async def test_refresh_fetch_wraps_any_failure_uniformly_no_cancellation_translation() -> None:
    """Unlike `fetch_with_login_cancellation`, refresh applies NO signal-aborted-specific
    translation -- confirmed here with the signal ALREADY aborted, yet the message is still the
    uniform refresh-error wrapping, never `"Login cancelled"` (`L11-SC-R004`)."""
    controller = RunAbortController()
    controller.abort()
    transport = _transport(error=ConnectionError("boom"))
    with pytest.raises(OAuthRefreshTransportError, match="OpenAI Codex token refresh error: boom"):
        await refresh_fetch(
            transport, "https://example.test", headers={}, body=b"", signal=controller.signal
        )


async def test_http_response_text_decodes_utf8_body() -> None:
    response = HttpResponse(status=200, body="café".encode())
    assert await response.text() == "café"


async def test_http_response_text_strips_leading_utf8_bom_only() -> None:
    """`L11-SC-R023` -- confirmed live: WHATWG Fetch's own UTF-8 body-text decoding strips
    exactly one LEADING byte-order mark; an INTERIOR `U+FEFF` is left untouched -- only the very
    first three bytes are special."""
    bom = chr(0xFEFF)
    leading_bom = bom + '{"a":1}'
    response = HttpResponse(status=200, body=leading_bom.encode())
    assert await response.text() == '{"a":1}'

    interior_bom = "a" + bom + "b"
    response2 = HttpResponse(status=200, body=interior_bom.encode())
    assert await response2.text() == interior_bom


async def test_http_response_text_caches_body_after_first_read() -> None:
    """A `read_body` closure that would raise on a SECOND call proves `text()` only invokes it
    once and reuses the cached result thereafter -- guarding against a real network stream being
    consumed twice."""
    calls = 0

    async def read_body() -> bytes:
        nonlocal calls
        calls += 1
        if calls > 1:
            raise AssertionError("read_body called more than once")
        return b"hello"

    response = HttpResponse(status=200, read_body=read_body)
    assert await response.text() == "hello"
    assert await response.text() == "hello"
    assert calls == 1


def _mock_handler(request: httpx.Request) -> httpx.Response:
    return httpx.Response(200, content=b'{"ok":true}')


async def test_httpx_transport_uses_an_injected_client_without_closing_it() -> None:
    """`HttpxTransport(client=...)` never touches a real socket -- `httpx.MockTransport` is
    httpx's own official no-network testing seam. An INJECTED client is never auto-closed by
    `HttpxTransport` itself (the caller owns its own lifecycle)."""
    client = httpx.AsyncClient(transport=httpx.MockTransport(_mock_handler))
    controller = RunAbortController()
    transport = HttpxTransport(client=client)
    try:
        response = await transport.post(
            "https://example.test/x", headers={}, body=b"", signal=controller.signal
        )
        assert response.status == 200
        assert await response.text() == '{"ok":true}'
        assert not client.is_closed
    finally:
        await client.aclose()


async def test_httpx_transport_creates_and_closes_its_own_client_when_none_injected(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """No client injected -- `HttpxTransport` constructs its own `httpx.AsyncClient` internally
    and closes it once the body is read (`L11-SC-R022`: closing now happens lazily, tied to body
    consumption, not eagerly the moment `post()` itself returns -- confirmed here by checking
    `is_closed` only AFTER `text()`, not immediately after `post()`). `httpx.AsyncClient` is
    monkeypatched to a `MockTransport`-backed factory so this still never touches a real socket."""
    created: list[httpx.AsyncClient] = []
    real_async_client = httpx.AsyncClient

    def fake_async_client(*args: object, **kwargs: object) -> httpx.AsyncClient:
        client = real_async_client(transport=httpx.MockTransport(_mock_handler))
        created.append(client)
        return client

    monkeypatch.setattr(httpx, "AsyncClient", fake_async_client)
    controller = RunAbortController()
    transport = HttpxTransport()
    response = await transport.post(
        "https://example.test/x", headers={}, body=b"", signal=controller.signal
    )
    assert response.status == 200
    assert not created[0].is_closed
    assert await response.text() == '{"ok":true}'
    assert len(created) == 1
    assert created[0].is_closed


async def test_httpx_transport_closes_owned_client_when_getting_the_response_itself_fails(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A pre-aborted signal makes `run_cancellable` raise `TransportCancelled` BEFORE
    `send_request()` is ever scheduled (`L11-SC-R015`) -- there is no later deferred `read_body`
    closure that could ever close an owned client in this failure mode, so `post()` itself must
    still close it, matching the pre-existing unconditional-cleanup guarantee this redesign must
    not lose."""
    created: list[httpx.AsyncClient] = []
    real_async_client = httpx.AsyncClient

    def fake_async_client(*args: object, **kwargs: object) -> httpx.AsyncClient:
        client = real_async_client(transport=httpx.MockTransport(_mock_handler))
        created.append(client)
        return client

    monkeypatch.setattr(httpx, "AsyncClient", fake_async_client)
    controller = RunAbortController()
    controller.abort()
    transport = HttpxTransport()
    with pytest.raises(TransportCancelled):
        await transport.post(
            "https://example.test/x", headers={}, body=b"", signal=controller.signal
        )
    assert len(created) == 1


async def test_httpx_transport_owned_client_close_failure_does_not_replace_request_failure(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """`L11-SC-R025`, second mandatory final-complete review -- confirmed live against the exact
    rejected candidate before this fix: pinned Pi's own `fetchWithLoginCancellation` re-throws the
    ORIGINAL request rejection whenever the signal did not abort; Pi has no observable
    client-close operation whose own failure could replace that result at all. An owned client
    whose `send()` raises `ConnectionError` AND whose `aclose()` ALSO raises `RuntimeError` must
    still surface the ORIGINAL `ConnectionError` -- the cleanup failure must not replace it."""
    real_async_client = httpx.AsyncClient

    def fake_async_client(*args: object, **kwargs: object) -> httpx.AsyncClient:
        client = real_async_client(*args, **kwargs)

        async def raising_send(*a: object, **kw: object) -> httpx.Response:
            raise ConnectionError("send boom")

        async def raising_aclose() -> None:
            raise RuntimeError("close boom")

        client.send = raising_send  # type: ignore[method-assign]
        client.aclose = raising_aclose  # type: ignore[method-assign]
        return client

    monkeypatch.setattr(httpx, "AsyncClient", fake_async_client)
    controller = RunAbortController()
    transport = HttpxTransport()
    with pytest.raises(ConnectionError, match="send boom"):
        await transport.post(
            "https://example.test/x", headers={}, body=b"", signal=controller.signal
        )


async def test_httpx_transport_owned_client_disables_its_own_implicit_timeout_cap(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """`L11-SC-R016` -- confirmed live: `httpx.AsyncClient()`'s own bare default is
    `timeout=Timeout(timeout=5.0)`, which would silently truncate refresh's own certified 15s
    `CombinedSignal` budget. Timeout enforcement must flow ONLY through this project's own
    signal-based `run_cancellable`, never an independent `httpx`-owned cap."""
    captured_kwargs: dict[str, object] = {}
    real_async_client = httpx.AsyncClient

    def fake_async_client(*args: object, **kwargs: object) -> httpx.AsyncClient:
        captured_kwargs.update(kwargs)
        return real_async_client(transport=httpx.MockTransport(_mock_handler), **kwargs)

    monkeypatch.setattr(httpx, "AsyncClient", fake_async_client)
    controller = RunAbortController()
    transport = HttpxTransport()
    await transport.post("https://example.test/x", headers={}, body=b"", signal=controller.signal)
    assert captured_kwargs["timeout"] is None


async def test_httpx_transport_owned_client_follows_redirects(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """`L11-SC-R017` -- confirmed live: `httpx.AsyncClient()`'s own bare default is
    `follow_redirects=False`, while `fetch`'s own default redirect mode follows redirects.
    A scripted 302-then-200 `httpx.MockTransport` response proves actual redirect-following
    behavior end-to-end, not merely an inspected constructor argument."""
    real_async_client = httpx.AsyncClient

    def redirecting_handler(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/start":
            return httpx.Response(302, headers={"Location": "https://example.test/end"})
        return httpx.Response(200, content=b'{"ok":true}')

    def fake_async_client(*args: object, **kwargs: object) -> httpx.AsyncClient:
        return real_async_client(transport=httpx.MockTransport(redirecting_handler), **kwargs)

    monkeypatch.setattr(httpx, "AsyncClient", fake_async_client)
    controller = RunAbortController()
    transport = HttpxTransport()
    response = await transport.post(
        "https://example.test/start", headers={}, body=b"", signal=controller.signal
    )
    assert response.status == 200
    assert await response.text() == '{"ok":true}'


async def test_httpx_transport_response_carries_the_real_reason_phrase() -> None:
    """`L11-SC-R018` -- the real HTTP reason phrase (`response.reason_phrase`) comes through on
    an EMPTY-body error response, not a hand-maintained lookup table -- confirmed here with
    status 401, which the local callback server's own `_STATUS_TEXT` table never covers."""

    def handler_401(request: httpx.Request) -> httpx.Response:
        return httpx.Response(401, content=b"")

    client = httpx.AsyncClient(transport=httpx.MockTransport(handler_401))
    controller = RunAbortController()
    transport = HttpxTransport(client=client)
    try:
        response = await transport.post(
            "https://example.test/x", headers={}, body=b"", signal=controller.signal
        )
        assert response.status == 401
        assert response.reason_phrase == "Unauthorized"
        assert await response.text() == ""
    finally:
        await client.aclose()


class _NeverCompletingStream(httpx.AsyncByteStream):
    """A `httpx` response body stream that never finishes -- simulates a slow/stalled body on a
    real network connection, used to prove `post()` itself never blocks on it."""

    async def __aiter__(self) -> AsyncIterator[bytes]:
        await asyncio.Event().wait()
        yield b""  # pragma: no cover -- unreachable, satisfies the generator protocol

    async def aclose(self) -> None:
        pass


async def test_httpx_transport_post_returns_before_the_body_is_ever_read() -> None:
    """`L11-SC-R022`, mandatory final-complete review -- confirmed live against the exact
    rejected candidate before this fix: `post()` itself must return as soon as status/headers
    arrive, even when the body stream never completes, matching pinned Pi's own `fetch()` promise
    (which resolves on headers, not on the body). A caller that never calls `text()` (the
    device-start-404/device-poll-403/404 abandoned-body shape this witness stands in for) must
    never be blocked by a slow or stalled body at all."""

    def handler_never_completing_body(request: httpx.Request) -> httpx.Response:
        return httpx.Response(404, stream=_NeverCompletingStream())

    client = httpx.AsyncClient(transport=httpx.MockTransport(handler_never_completing_body))
    controller = RunAbortController()
    transport = HttpxTransport(client=client)
    try:
        response = await asyncio.wait_for(
            transport.post(
                "https://example.test/x", headers={}, body=b"", signal=controller.signal
            ),
            timeout=2.0,
        )
        assert response.status == 404
    finally:
        await client.aclose()


async def test_httpx_transport_signal_abort_during_body_read_is_not_swallowed() -> None:
    """A signal that aborts WHILE `text()` is awaiting a slow body (not during the initial
    request) still raises `TransportCancelled`, even on a non-2xx status whose ordinary body-read
    failures are otherwise swallowed to an empty string -- confirmed live: a genuine cancellation
    must always propagate as a cancellation, never be treated as just another body-read failure
    that happens to be swallowed by the non-2xx status rule."""

    def handler_slow_body(request: httpx.Request) -> httpx.Response:
        return httpx.Response(401, stream=_NeverCompletingStream())

    client = httpx.AsyncClient(transport=httpx.MockTransport(handler_slow_body))
    controller = RunAbortController()
    transport = HttpxTransport(client=client)
    try:
        response = await transport.post(
            "https://example.test/x", headers={}, body=b"", signal=controller.signal
        )
        assert response.status == 401
        text_task = asyncio.ensure_future(response.text())
        await asyncio.sleep(0.1)
        controller.abort()
        with pytest.raises(TransportCancelled):
            await asyncio.wait_for(text_task, timeout=2.0)
    finally:
        await client.aclose()


class _TrackedStream(httpx.AsyncByteStream):
    """A body stream that records whether it was ever iterated (`started`) and whether `aclose`
    ran (`closed`) -- used to prove `discard()` releases the underlying resources WITHOUT
    triggering a body read."""

    def __init__(self) -> None:
        self.started = False
        self.closed = False

    async def __aiter__(self) -> AsyncIterator[bytes]:
        self.started = True
        await asyncio.Event().wait()
        yield b""  # pragma: no cover -- unreachable, satisfies the generator protocol

    async def aclose(self) -> None:
        self.closed = True


async def test_http_response_discard_closes_an_injected_clients_response_without_reading() -> None:
    """`L11-SC-R022`, targeted re-review -- confirmed live against the exact rejected candidate
    before this fix: ordinary Python object-lifetime/garbage collection CANNOT perform the ASYNC
    close a real streamed `httpx` response requires, so an abandoned response was genuinely left
    open, not merely un-warned-about. `discard()` must close it explicitly, without ever
    triggering a body read, and without requiring the CALLER to also close the client (an
    INJECTED client is never auto-closed, matching the existing ownership rule)."""
    stream = _TrackedStream()

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(404, stream=stream)

    client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    controller = RunAbortController()
    transport = HttpxTransport(client=client)
    try:
        response = await transport.post(
            "https://example.test/x", headers={}, body=b"", signal=controller.signal
        )
        assert response.status == 404
        await response.discard()
        assert not stream.started
        assert stream.closed
        assert not client.is_closed
    finally:
        await client.aclose()


async def test_http_response_discard_closes_an_owned_client_when_none_injected(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """`L11-SC-R022`, targeted re-review, owned-client companion case -- confirmed live: closing
    the client explicitly afterward does NOT retroactively close an already-abandoned response
    (they are genuinely separate close operations), so `discard()` must close BOTH the response
    AND, when `HttpxTransport` owns the client, the client itself -- the exact device-start-404/
    device-poll-403-or-404 path this fix exists for, where repeated polling could otherwise
    abandon one open response/client per pending status."""
    stream = _TrackedStream()
    created: list[httpx.AsyncClient] = []
    real_async_client = httpx.AsyncClient

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(404, stream=stream)

    def fake_async_client(*args: object, **kwargs: object) -> httpx.AsyncClient:
        client = real_async_client(transport=httpx.MockTransport(handler))
        created.append(client)
        return client

    monkeypatch.setattr(httpx, "AsyncClient", fake_async_client)
    controller = RunAbortController()
    transport = HttpxTransport()
    response = await transport.post(
        "https://example.test/x", headers={}, body=b"", signal=controller.signal
    )
    assert response.status == 404
    await response.discard()
    assert not stream.started
    assert stream.closed
    assert len(created) == 1
    assert created[0].is_closed


async def test_http_response_discard_is_idempotent_after_text_already_read() -> None:
    """`discard()` is a safe no-op if `text()` was already called -- the underlying resources are
    already closed by `text()`'s own cleanup, and `discard()` must not attempt to close them
    again (which would raise on an already-closed `httpx` stream/client if not guarded)."""

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, content=b"hello")

    client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    controller = RunAbortController()
    transport = HttpxTransport(client=client)
    try:
        response = await transport.post(
            "https://example.test/x", headers={}, body=b"", signal=controller.signal
        )
        assert await response.text() == "hello"
        await response.discard()  # must not raise
    finally:
        await client.aclose()


async def test_http_response_discard_with_no_transport_backing_is_a_noop() -> None:
    """A synthetic, already-buffered `HttpResponse` (the common test-fixture shape, with no
    `discard_body` closure at all) makes `discard()` a harmless no-op -- there is no live network
    resource to release."""
    response = HttpResponse(status=200, body=b"hello")
    await response.discard()  # must not raise


class _RaisingCloseStream(httpx.AsyncByteStream):
    """A body stream whose `aclose()` itself raises -- pinned Pi exposes no explicit "close"
    operation at all for a status-only outcome (device-start's own `404`, device-poll's own
    `403`/`404`), so a cleanup failure here must never become a new public failure, and must not
    prevent an owned client's own close from being attempted."""

    async def __aiter__(self) -> AsyncIterator[bytes]:
        return
        yield b""  # pragma: no cover -- unreachable, satisfies the generator protocol

    async def aclose(self) -> None:
        raise RuntimeError("close boom")


async def test_http_response_discard_swallows_a_raising_response_close() -> None:
    """`L11-SC-R022`, second targeted re-review -- confirmed live against the exact rejected
    candidate before this fix: a response whose `aclose()` raises let that raw exception escape
    `discard()` entirely, which -- at the real device-start/device-poll call sites -- REPLACED
    the caller's own fixed Pi outcome with an unrelated cleanup error. `discard()` itself must
    never raise for this reason."""

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(404, stream=_RaisingCloseStream())

    client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    controller = RunAbortController()
    transport = HttpxTransport(client=client)
    try:
        response = await transport.post(
            "https://example.test/x", headers={}, body=b"", signal=controller.signal
        )
        assert response.status == 404
        await response.discard()  # must not raise, despite the stream's own aclose() raising
    finally:
        await client.aclose()


async def test_http_response_discard_still_closes_owned_client_when_response_close_raises(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """`L11-SC-R022`, second targeted re-review, owned-client companion case -- confirmed live:
    the shared close helper previously awaited `response.aclose()` with no `finally` of its own,
    so a raising response close also skipped the owned client's own close entirely. Both close
    attempts must be independently exception-safe -- a failure in one must not skip the other."""
    real_async_client = httpx.AsyncClient
    created: list[httpx.AsyncClient] = []

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(404, stream=_RaisingCloseStream())

    def fake_async_client(*args: object, **kwargs: object) -> httpx.AsyncClient:
        client = real_async_client(transport=httpx.MockTransport(handler))
        created.append(client)
        return client

    monkeypatch.setattr(httpx, "AsyncClient", fake_async_client)
    controller = RunAbortController()
    transport = HttpxTransport()
    response = await transport.post(
        "https://example.test/x", headers={}, body=b"", signal=controller.signal
    )
    assert response.status == 404
    await response.discard()  # must not raise
    assert len(created) == 1
    assert created[0].is_closed


class _CancellingCloseStream(httpx.AsyncByteStream):
    """A body stream whose `aclose()` raises `asyncio.CancelledError` (a `BaseException`, not an
    `Exception`) -- distinct from `_RaisingCloseStream` above, which raises an ORDINARY
    exception. Cancellation must still propagate out of `discard()`, unlike an ordinary close
    failure, but the owned client's own close must still be attempted first."""

    async def __aiter__(self) -> AsyncIterator[bytes]:
        return
        yield b""  # pragma: no cover -- unreachable, satisfies the generator protocol

    async def aclose(self) -> None:
        raise asyncio.CancelledError("cancel during response close")


async def test_http_response_discard_still_closes_owned_client_when_response_close_is_cancelled(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """`L11-SC-R022`, third targeted re-review -- confirmed live against the exact rejected
    candidate before this fix: two SEQUENTIAL `contextlib.suppress(Exception)` blocks correctly
    let `asyncio.CancelledError` propagate (it is a `BaseException`, not caught by
    `suppress(Exception)`), but doing so exited the shared close helper BEFORE the owned client's
    own close was ever attempted, leaving it open even though the cancellation itself correctly
    propagated. The owned-client close must run in a genuine `finally`, not a second sequential
    block, so it is attempted EVEN WHEN the response close is itself cancelled."""
    real_async_client = httpx.AsyncClient
    created: list[httpx.AsyncClient] = []

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(404, stream=_CancellingCloseStream())

    def fake_async_client(*args: object, **kwargs: object) -> httpx.AsyncClient:
        client = real_async_client(transport=httpx.MockTransport(handler))
        created.append(client)
        return client

    monkeypatch.setattr(httpx, "AsyncClient", fake_async_client)
    controller = RunAbortController()
    transport = HttpxTransport()
    response = await transport.post(
        "https://example.test/x", headers={}, body=b"", signal=controller.signal
    )
    assert response.status == 404
    with pytest.raises(asyncio.CancelledError):
        await response.discard()
    assert len(created) == 1
    assert created[0].is_closed


class _RaisingStream(httpx.AsyncByteStream):
    """A `httpx` response body stream that always fails partway through reading -- simulates a
    connection reset AFTER status/headers already arrived successfully."""

    async def __aiter__(self) -> AsyncIterator[bytes]:
        raise httpx.ReadError("simulated body read failure")
        yield b""  # pragma: no cover -- unreachable, satisfies the generator protocol

    async def aclose(self) -> None:
        pass


async def test_httpx_transport_propagates_a_2xx_body_read_failure() -> None:
    """`L11-SC-R018` -- confirmed live against pinned Pi (`readTokenResponse`,
    `startOpenAICodexDeviceAuth`, `pollOpenAICodexDeviceAuth`'s own `poll()`): every 2xx branch
    reads the body via `.json()` with NO catch at all, so a body-read failure there is an
    UNCAUGHT rejection that propagates through the existing call-site boundary. `post()` itself
    now succeeds (status/headers only, `L11-SC-R022`) -- the failure surfaces from `text()`
    instead, the exact caller-equivalent point pinned Pi's own code would reach it at."""

    def handler_200_body_read_failure(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, stream=_RaisingStream())

    client = httpx.AsyncClient(transport=httpx.MockTransport(handler_200_body_read_failure))
    controller = RunAbortController()
    transport = HttpxTransport(client=client)
    try:
        response = await transport.post(
            "https://example.test/x", headers={}, body=b"", signal=controller.signal
        )
        assert response.status == 200
        with pytest.raises(httpx.ReadError, match="simulated body read failure"):
            await response.text()
    finally:
        await client.aclose()


async def test_httpx_transport_translates_a_non_2xx_body_read_failure_into_an_empty_body() -> None:
    """`L11-SC-R018` -- confirmed live: a body-read failure AFTER a non-2xx status already
    arrived successfully (e.g. a connection reset mid-body) comes back as an empty string from
    `text()`, with the real status/reason phrase intact, matching pinned Pi's own non-2xx
    branches, which already collapse this and a genuinely empty body identically via
    `text().catch(() => "")`."""

    def handler_401_body_read_failure(request: httpx.Request) -> httpx.Response:
        return httpx.Response(401, stream=_RaisingStream())

    client = httpx.AsyncClient(transport=httpx.MockTransport(handler_401_body_read_failure))
    controller = RunAbortController()
    transport = HttpxTransport(client=client)
    try:
        response = await transport.post(
            "https://example.test/x", headers={}, body=b"", signal=controller.signal
        )
        assert response.status == 401
        assert response.reason_phrase == "Unauthorized"
        assert await response.text() == ""
    finally:
        await client.aclose()


class _NonHttpErrorRaisingStream(httpx.AsyncByteStream):
    """A body stream that fails with an ORDINARY exception outside `httpx`'s own error
    hierarchy -- a realistic failure an injectable, caller-supplied `AsyncByteStream`
    implementation can raise, since `httpx`'s own public seam does not require a custom stream
    to raise specifically `httpx.HTTPError`."""

    async def __aiter__(self) -> AsyncIterator[bytes]:
        raise RuntimeError("body boom")
        yield b""  # pragma: no cover -- unreachable, satisfies the generator protocol

    async def aclose(self) -> None:
        pass


async def test_httpx_transport_swallows_a_non_httpx_non_2xx_body_read_failure() -> None:
    """`L11-SC-R018`, targeted convergence review, third round -- confirmed live: pinned Pi's own
    `.text().catch(() => "")` catches ANY promise rejection from the body-read operation, not one
    library-specific error hierarchy. Catching only `httpx.HTTPError` left an ordinary
    `RuntimeError` from a custom stream uncaught on the non-2xx path, contradicting Pi's own
    broader rule."""

    def handler_401_runtime_error(request: httpx.Request) -> httpx.Response:
        return httpx.Response(401, stream=_NonHttpErrorRaisingStream())

    client = httpx.AsyncClient(transport=httpx.MockTransport(handler_401_runtime_error))
    controller = RunAbortController()
    transport = HttpxTransport(client=client)
    try:
        response = await transport.post(
            "https://example.test/x", headers={}, body=b"", signal=controller.signal
        )
        assert response.status == 401
        assert response.reason_phrase == "Unauthorized"
        assert await response.text() == ""
    finally:
        await client.aclose()


async def test_httpx_transport_propagates_a_non_httpx_2xx_body_read_failure() -> None:
    """`L11-SC-R018`, targeted convergence review, third round, discriminating companion case --
    the SAME non-`httpx.HTTPError` failure under a 2xx status must still propagate from `text()`,
    preserving the already-correct status-conditioned boundary and ruling out a return to the
    first remediation's own uniform catch-everything behavior."""

    def handler_200_runtime_error(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, stream=_NonHttpErrorRaisingStream())

    client = httpx.AsyncClient(transport=httpx.MockTransport(handler_200_runtime_error))
    controller = RunAbortController()
    transport = HttpxTransport(client=client)
    try:
        response = await transport.post(
            "https://example.test/x", headers={}, body=b"", signal=controller.signal
        )
        assert response.status == 200
        with pytest.raises(RuntimeError, match="body boom"):
            await response.text()
    finally:
        await client.aclose()
