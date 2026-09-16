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


def test_http_response_text_decodes_utf8_body() -> None:
    response = HttpResponse(status=200, body="café".encode())
    assert response.text() == "café"


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
        assert response.body == b'{"ok":true}'
        assert not client.is_closed
    finally:
        await client.aclose()


async def test_httpx_transport_creates_and_closes_its_own_client_when_none_injected(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """No client injected -- `HttpxTransport` constructs its own `httpx.AsyncClient` internally
    and closes it afterward. `httpx.AsyncClient` is monkeypatched to a `MockTransport`-backed
    factory so this still never touches a real socket."""
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
    assert response.body == b'{"ok":true}'
    assert len(created) == 1
    assert created[0].is_closed


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
    assert response.body == b'{"ok":true}'


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
        assert response.body == b""
    finally:
        await client.aclose()


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
    UNCAUGHT rejection that propagates through the existing call-site boundary. The first
    remediation over-corrected by swallowing this into an empty body regardless of status,
    which the second independent review caught as its own refined finding: a status-200
    response whose body then fails to read must still raise, not fabricate an empty-body
    success."""

    def handler_200_body_read_failure(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, stream=_RaisingStream())

    client = httpx.AsyncClient(transport=httpx.MockTransport(handler_200_body_read_failure))
    controller = RunAbortController()
    transport = HttpxTransport(client=client)
    try:
        with pytest.raises(httpx.ReadError, match="simulated body read failure"):
            await transport.post(
                "https://example.test/x", headers={}, body=b"", signal=controller.signal
            )
    finally:
        await client.aclose()


async def test_httpx_transport_translates_a_non_2xx_body_read_failure_into_an_empty_body() -> None:
    """`L11-SC-R018` -- confirmed live: a body-read failure AFTER a non-2xx status already
    arrived successfully (e.g. a connection reset mid-body) comes back as a normal
    `HttpResponse` with an empty body and the real status/reason phrase intact, matching pinned
    Pi's own non-2xx branches, which already collapse this and a genuinely empty body
    identically via `text().catch(() => "")` -- NOT an opaque request-level exception, which a
    single buffered response-construction call cannot avoid raising for both cases alike."""

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
        assert response.body == b""
    finally:
        await client.aclose()
