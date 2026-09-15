"""Codex OAuth NETWORK integration (`PROV-012`, Pass 2 Slice C). No test performs a live network
call or uses a real secret -- outbound HTTP is always a deterministic `_FakeTransport`; the local
callback server binds a real, ephemeral LOOPBACK port (never production port 1455) and is exercised
via real local TCP connections, matching this project's own established discipline for local-only
I/O.
"""

from __future__ import annotations

import asyncio
import json
import socket
from collections.abc import Callable, Coroutine

import pytest

from minion_agent.auth.credential import JsonValue, OAuthCredential
from minion_agent.auth.http_transport import (
    HttpResponse,
    LoginCancelledError,
    OAuthRefreshTransportError,
)
from minion_agent.auth.interaction import (
    AuthEventDeviceCode,
    AuthEventUrl,
    AuthPrompt,
    AuthPromptManualCode,
    AuthPromptSelect,
)
from minion_agent.auth.openai_codex_oauth import (
    CLIENT_ID,
    DEVICE_TOKEN_URL,
    DEVICE_USER_CODE_URL,
    REDIRECT_URI,
    DeviceCodeNotEnabledError,
    InvalidDeviceCodeResponseError,
    MissingAuthorizationCodeError,
    StateMismatchError,
    TokenResponseFailedError,
    TokenResponseMissingFieldsError,
    UnknownLoginMethodError,
    _create_authorization_flow,
    _DeviceAuthInfo,
    _exchange_authorization_code,
    _js_number_coerce,
    _login,
    _login_browser,
    _login_device_code,
    _poll_device_auth,
    _read_token_response,
    _refresh_access_token,
    _refresh_openai_codex_token,
    _start_device_auth,
    openai_codex_oauth,
    parse_authorization_input,
)
from minion_agent.runtime.signal import RunAbortController, RunSignal

# Cross-checked directly against a live Node v22 process (`atob` + `JSON.parse`), matching this
# project's own established `PROV-011` fixture (`test_openai_codex.py`) -- duplicated here (not
# imported cross-test-file, no precedent for that in this codebase) so this file stays
# self-contained. Header `{"alg":"none","typ":"JWT"}`; payload `{"https://api.openai.com/auth":
# {"chatgpt_account_id":"acct_synthetic_test_123"}}`; signature segment is the arbitrary literal
# `sig` (never verified). No real account, token, or secret -- entirely synthetic.
VALID_TOKEN = (
    "eyJhbGciOiJub25lIiwidHlwIjoiSldUIn0=."
    "eyJodHRwczovL2FwaS5vcGVuYWkuY29tL2F1dGgiOnsiY2hhdGdwdF9hY2NvdW50X2lkIjoiYWNjdF9zeW50aGV0aWNf"
    "dGVzdF8xMjMifX0=."
    "sig"
)


# --- test infrastructure --------------------------------------------------------------------------


class _FakeTransport:
    """A deterministic, scripted `HttpTransport`. `responses` is consumed in order per call;
    `error` (if set) is raised instead on the NEXT call. Never touches a real socket."""

    def __init__(self) -> None:
        self._responses: list[HttpResponse] = []
        self._error: Exception | None = None
        self.calls: list[str] = []

    def queue_response(self, status: int, body: bytes) -> None:
        self._responses.append(HttpResponse(status=status, body=body))

    def queue_json(self, status: int, value: JsonValue) -> None:
        self.queue_response(status, json.dumps(value).encode())

    def queue_error(self, error: Exception) -> None:
        self._error = error

    async def post(self, url: str, *, headers: object, body: bytes, signal: object) -> HttpResponse:
        self.calls.append(url)
        if self._error is not None:
            error, self._error = self._error, None
            raise error
        return self._responses.pop(0)


class _FakeInteraction:
    """A deterministic `ProviderAuthInteraction` test double. `select_response` answers a
    `select` prompt; `manual_responses` is a queue of zero-argument async callables, each either
    returning the manual input string or raising, consumed one per `manual_code` prompt."""

    def __init__(
        self,
        *,
        signal: RunSignal,
        select_response: str | None = None,
        manual_responses: list[Callable[[], Coroutine[object, object, str]]] | None = None,
    ) -> None:
        self._signal = signal
        self._select_response = select_response
        self._manual_responses = list(manual_responses or [])
        self.notifications: list[object] = []
        self.prompts: list[AuthPrompt] = []

    @property
    def signal(self) -> RunSignal:
        return self._signal

    async def prompt(self, prompt: AuthPrompt) -> str:
        self.prompts.append(prompt)
        if isinstance(prompt, AuthPromptSelect):
            assert self._select_response is not None
            return self._select_response
        if isinstance(prompt, AuthPromptManualCode):
            action = self._manual_responses.pop(0)
            # A realistic interaction implementation observes its own per-prompt `signal` and
            # cancels a pending prompt when it aborts -- reproduced here so a queued action that
            # would otherwise hang forever (`_never_resolves`) unblocks properly once
            # `_login_browser`'s own cleanup aborts `manual_controller`, instead of leaking an
            # orphaned task.
            action_task: asyncio.Task[str] = asyncio.ensure_future(action())
            prompt_signal = prompt.signal
            try:
                while not action_task.done():
                    if prompt_signal is not None and prompt_signal.aborted:
                        action_task.cancel()
                        raise asyncio.CancelledError("manual prompt cancelled via its own signal")
                    await asyncio.sleep(0.01)
                return action_task.result()
            finally:
                if not action_task.done():
                    action_task.cancel()
        raise NotImplementedError(prompt)

    def notify(self, event: object) -> None:
        self.notifications.append(event)


def _free_loopback_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


async def _http_get(host: str, port: int, target: str) -> tuple[int, dict[str, str]]:
    """A minimal, deterministic local HTTP/1.1 client -- reads only the status line and headers,
    enough to verify the callback server's own observable contract."""
    reader, writer = await asyncio.open_connection(host, port)
    try:
        writer.write(f"GET {target} HTTP/1.1\r\nHost: {host}\r\n\r\n".encode())
        await writer.drain()
        status_line = await reader.readline()
        status = int(status_line.decode().split(" ")[1])
        headers: dict[str, str] = {}
        while True:
            line = await reader.readline()
            if line in (b"\r\n", b"\n", b""):
                break
            name, _, value = line.decode().partition(":")
            headers[name.strip().lower()] = value.strip()
        return status, headers
    finally:
        writer.close()


# --- parse_authorization_input ----------------------------------------------------------------


def test_parse_authorization_input_empty_or_whitespace() -> None:
    assert parse_authorization_input("") == parse_authorization_input("   ")
    result = parse_authorization_input("   ")
    assert result.code is None
    assert result.state is None


def test_parse_authorization_input_absolute_url() -> None:
    result = parse_authorization_input(f"{REDIRECT_URI}?code=abc&state=xyz")
    assert result.code == "abc"
    assert result.state == "xyz"


def test_parse_authorization_input_absolute_url_missing_params() -> None:
    result = parse_authorization_input(REDIRECT_URI)
    assert result.code is None
    assert result.state is None


def test_parse_authorization_input_hash_split_discards_content_after_second_hash() -> None:
    """Pi's own `split("#", 2)` semantics: content after a SECOND `"#"` is discarded, never
    appended to `state` -- a host language's own unlimited split on the first `"#"` would
    incorrectly keep it."""
    result = parse_authorization_input("abc#xyz#discarded")
    assert result.code == "abc"
    assert result.state == "xyz"


def test_parse_authorization_input_hash_split_single_hash() -> None:
    result = parse_authorization_input("abc#xyz")
    assert result.code == "abc"
    assert result.state == "xyz"


def test_parse_authorization_input_bare_query_string() -> None:
    result = parse_authorization_input("code=abc&state=xyz")
    assert result.code == "abc"
    assert result.state == "xyz"


def test_parse_authorization_input_bare_code() -> None:
    result = parse_authorization_input("just-a-code")
    assert result.code == "just-a-code"
    assert result.state is None


# --- _js_number_coerce (JS `Number(string)` coercion, empirically verified live against Node) ----


def test_js_number_coerce_empty_string_is_zero() -> None:
    assert _js_number_coerce("") == 0.0
    assert _js_number_coerce("   ") == 0.0


def test_js_number_coerce_ordinary_decimal() -> None:
    assert _js_number_coerce("5") == 5.0
    assert _js_number_coerce("-5") == -5.0
    assert _js_number_coerce("1e3") == 1000.0


def test_js_number_coerce_hex_octal_binary_prefixes() -> None:
    assert _js_number_coerce("0x1A") == 26.0
    assert _js_number_coerce("0o17") == 15.0
    assert _js_number_coerce("0b101") == 5.0


def test_js_number_coerce_infinity_tokens() -> None:
    assert _js_number_coerce("Infinity") == float("inf")
    assert _js_number_coerce("+Infinity") == float("inf")
    assert _js_number_coerce("-Infinity") == float("-inf")


def test_js_number_coerce_garbage_is_nan() -> None:
    result = _js_number_coerce("5abc")
    assert result != result  # NaN != NaN


# --- authorization URL construction -------------------------------------------------------------


def test_create_authorization_flow_url_has_exact_params_in_order() -> None:
    flow = _create_authorization_flow()
    query = flow.url.split("?", 1)[1]
    pairs = [p.split("=", 1)[0] for p in query.split("&")]
    assert pairs == [
        "response_type",
        "client_id",
        "redirect_uri",
        "scope",
        "code_challenge",
        "code_challenge_method",
        "state",
        "id_token_add_organizations",
        "codex_cli_simplified_flow",
        "originator",
    ]
    assert f"client_id={CLIENT_ID}" in flow.url
    assert "originator=pi" in flow.url


def test_create_authorization_flow_state_is_random_hex() -> None:
    flow_a = _create_authorization_flow()
    flow_b = _create_authorization_flow()
    assert flow_a.state != flow_b.state
    assert len(flow_a.state) == 32
    int(flow_a.state, 16)  # does not raise


# --- local callback server (real ephemeral loopback socket, real local TCP client) --------------


async def _manual_resolves_empty_after_delay() -> str:
    """Simulates a manual prompt that eventually settles with no usable input -- lets these
    server-focused tests observe the server's own HTTP response and then let the flow conclude
    naturally, with no signal abort required."""
    await asyncio.sleep(0.15)
    return ""


async def test_callback_server_wrong_path_is_404() -> None:
    controller = RunAbortController()
    interaction = _FakeInteraction(
        signal=controller.signal, manual_responses=[_manual_resolves_empty_after_delay]
    )
    port = _free_loopback_port()
    task = asyncio.ensure_future(_login_browser(interaction, bind_port=port))
    await asyncio.sleep(0.05)
    status, headers = await _http_get("127.0.0.1", port, "/wrong-path")
    assert status == 404
    assert headers["content-type"] == "text/html; charset=utf-8"
    with pytest.raises(MissingAuthorizationCodeError):
        await task


async def test_callback_server_state_mismatch_is_400() -> None:
    controller = RunAbortController()
    interaction = _FakeInteraction(
        signal=controller.signal, manual_responses=[_manual_resolves_empty_after_delay]
    )
    port = _free_loopback_port()
    task = asyncio.ensure_future(_login_browser(interaction, bind_port=port))
    await asyncio.sleep(0.05)
    status, headers = await _http_get("127.0.0.1", port, "/auth/callback?state=wrong&code=abc")
    assert status == 400
    assert headers["content-type"] == "text/html; charset=utf-8"
    with pytest.raises(MissingAuthorizationCodeError):
        await task


async def test_callback_server_missing_code_is_400(monkeypatch: pytest.MonkeyPatch) -> None:
    controller = RunAbortController()
    interaction = _FakeInteraction(
        signal=controller.signal, manual_responses=[_manual_resolves_empty_after_delay]
    )
    port = _free_loopback_port()
    captured_state: list[str] = []
    _capture_state(monkeypatch, captured_state)
    task = asyncio.ensure_future(_login_browser(interaction, bind_port=port))
    await asyncio.sleep(0.05)
    status, _headers = await _http_get(
        "127.0.0.1", port, f"/auth/callback?state={captured_state[0]}"
    )
    assert status == 400
    with pytest.raises(MissingAuthorizationCodeError):
        await task


async def test_callback_server_empty_code_is_400_not_200(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """`L11-SC-R007`: an empty `code=` is rejected exactly like an absent one, never accepted."""
    controller = RunAbortController()
    interaction = _FakeInteraction(
        signal=controller.signal, manual_responses=[_manual_resolves_empty_after_delay]
    )
    port = _free_loopback_port()
    captured_state: list[str] = []
    _capture_state(monkeypatch, captured_state)
    task = asyncio.ensure_future(_login_browser(interaction, bind_port=port))
    await asyncio.sleep(0.05)
    status, _headers = await _http_get(
        "127.0.0.1", port, f"/auth/callback?state={captured_state[0]}&code="
    )
    assert status == 400
    with pytest.raises(MissingAuthorizationCodeError):
        await task


async def test_callback_server_internal_exception_is_contained_as_500(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """`L11-SC-R008`: any exception raised while handling a callback request is CAUGHT and
    produces a contained `500` response, never escaping or crashing the server."""
    import minion_agent.auth.openai_codex_oauth as module

    def raising_urlsplit(*_args: object, **_kwargs: object) -> object:
        raise RuntimeError("simulated internal failure")

    monkeypatch.setattr(module, "urlsplit", raising_urlsplit)

    controller = RunAbortController()
    interaction = _FakeInteraction(
        signal=controller.signal, manual_responses=[_manual_resolves_empty_after_delay]
    )
    port = _free_loopback_port()
    task = asyncio.ensure_future(_login_browser(interaction, bind_port=port))
    await asyncio.sleep(0.05)
    status, headers = await _http_get("127.0.0.1", port, "/auth/callback?state=x&code=y")
    assert status == 500
    assert headers["content-type"] == "text/html; charset=utf-8"
    with pytest.raises(MissingAuthorizationCodeError):
        await task


def _capture_state(monkeypatch: pytest.MonkeyPatch, out: list[str]) -> None:
    import minion_agent.auth.openai_codex_oauth as module

    original = module._create_authorization_flow

    def spy(originator: str = "pi") -> object:
        flow = original(originator)
        out.append(flow.state)
        return flow

    monkeypatch.setattr(module, "_create_authorization_flow", spy)


async def _never_resolves() -> str:
    await asyncio.Event().wait()
    return ""  # pragma: no cover -- never reached


# --- browser flow race (`L11-SC-R002`/`R009`) ----------------------------------------------------


async def test_browser_flow_server_yields_code_first(monkeypatch: pytest.MonkeyPatch) -> None:
    controller = RunAbortController()
    interaction = _FakeInteraction(signal=controller.signal, manual_responses=[_never_resolves])
    port = _free_loopback_port()
    captured_state: list[str] = []
    _capture_state(monkeypatch, captured_state)
    transport = _FakeTransport()
    transport.queue_json(
        200, {"access_token": VALID_TOKEN, "refresh_token": "rt", "expires_in": 3600}
    )
    transport.queue_json(
        200,
        {
            "https_dummy": True,  # never inspected
        },
    )
    task = asyncio.ensure_future(_login_browser(interaction, bind_port=port, transport=transport))
    await asyncio.sleep(0.05)
    status, _headers = await _http_get(
        "127.0.0.1", port, f"/auth/callback?state={captured_state[0]}&code=servercode"
    )
    assert status == 200
    credential = await task
    assert credential.access == VALID_TOKEN
    assert credential.refresh == "rt"


async def test_browser_flow_manual_non_empty_wins_and_cancels_server() -> None:
    controller = RunAbortController()

    async def manual() -> str:
        return "pastedcode"

    interaction = _FakeInteraction(signal=controller.signal, manual_responses=[manual])
    port = _free_loopback_port()
    transport = _FakeTransport()
    transport.queue_json(
        200, {"access_token": VALID_TOKEN, "refresh_token": "rt2", "expires_in": 3600}
    )
    credential = await _login_browser(interaction, bind_port=port, transport=transport)
    assert credential.access == VALID_TOKEN


async def test_browser_flow_manual_empty_value_still_cancels_server_then_fails() -> None:
    """`L11-SC-R002` point 1: an empty manual resolution unconditionally cancels the server's
    own wait, and the flow then fails rather than falling back to the (now-cancelled) server."""
    controller = RunAbortController()

    async def manual() -> str:
        return ""

    interaction = _FakeInteraction(signal=controller.signal, manual_responses=[manual])
    port = _free_loopback_port()
    with pytest.raises(MissingAuthorizationCodeError):
        await _login_browser(interaction, bind_port=port)


async def test_browser_flow_manual_error_propagates() -> None:
    controller = RunAbortController()

    async def manual() -> str:
        raise ValueError("user cancelled the prompt")

    interaction = _FakeInteraction(signal=controller.signal, manual_responses=[manual])
    port = _free_loopback_port()
    with pytest.raises(ValueError, match="user cancelled the prompt"):
        await _login_browser(interaction, bind_port=port)


async def test_browser_flow_manual_state_mismatch_raises() -> None:
    controller = RunAbortController()

    async def manual() -> str:
        return "code#wrong-state"

    interaction = _FakeInteraction(signal=controller.signal, manual_responses=[manual])
    port = _free_loopback_port()
    with pytest.raises(StateMismatchError):
        await _login_browser(interaction, bind_port=port)


async def test_browser_flow_server_absent_then_manual_raises_on_second_check() -> None:
    """The server's own "wait for code" resolving absent (here via a bind failure) with no
    manual code yet either falls through to waiting on the still-pending manual prompt -- and if
    that eventually raises, the error surfaces from the second check, not the first."""
    occupying = await asyncio.start_server(lambda r, w: None, "127.0.0.1", 0)
    port = occupying.sockets[0].getsockname()[1]
    try:
        controller = RunAbortController()

        async def manual() -> str:
            await asyncio.sleep(0.05)
            raise ValueError("settled late with an error")

        interaction = _FakeInteraction(signal=controller.signal, manual_responses=[manual])
        with pytest.raises(ValueError, match="settled late with an error"):
            await _login_browser(interaction, bind_port=port)
    finally:
        occupying.close()
        await occupying.wait_closed()


async def test_browser_flow_server_absent_then_manual_state_mismatch_on_second_check() -> None:
    occupying = await asyncio.start_server(lambda r, w: None, "127.0.0.1", 0)
    port = occupying.sockets[0].getsockname()[1]
    try:
        controller = RunAbortController()

        async def manual() -> str:
            await asyncio.sleep(0.05)
            return "code#wrong-state"

        interaction = _FakeInteraction(signal=controller.signal, manual_responses=[manual])
        with pytest.raises(StateMismatchError):
            await _login_browser(interaction, bind_port=port)
    finally:
        occupying.close()
        await occupying.wait_closed()


async def test_browser_flow_flow_abort_while_both_pending_waits_for_manual() -> None:
    """`L11-SC-R002` point 2 / `L11-SC-R009`: a flow-level abort while a manual prompt is still
    pending does NOT promptly resolve the operation -- it keeps waiting for the manual prompt's
    own eventual settlement."""
    controller = RunAbortController()
    manual_settled = asyncio.Event()
    manual_result = "manual-after-abort"

    async def manual() -> str:
        await manual_settled.wait()
        return manual_result

    interaction = _FakeInteraction(signal=controller.signal, manual_responses=[manual])
    port = _free_loopback_port()
    transport = _FakeTransport()
    transport.queue_json(
        200, {"access_token": VALID_TOKEN, "refresh_token": "rt3", "expires_in": 3600}
    )
    task = asyncio.ensure_future(_login_browser(interaction, bind_port=port, transport=transport))
    await asyncio.sleep(0.05)
    controller.abort()
    await asyncio.sleep(0.2)
    assert not task.done()  # still waiting on the manual prompt, not resolved by the abort alone
    manual_settled.set()
    credential = await task
    assert credential.access == VALID_TOKEN


async def test_browser_flow_pre_aborted_at_entry_still_notifies_and_prompts() -> None:
    """`L11-SC-R009`: a login flow beginning with `signal` already aborted does NOT short-circuit
    before notifying/prompting -- both still occur unconditionally."""
    controller = RunAbortController()
    controller.abort()

    async def manual() -> str:
        return "manualcode"

    interaction = _FakeInteraction(signal=controller.signal, manual_responses=[manual])
    port = _free_loopback_port()
    transport = _FakeTransport()
    transport.queue_json(
        200, {"access_token": VALID_TOKEN, "refresh_token": "rt4", "expires_in": 3600}
    )
    credential = await _login_browser(interaction, bind_port=port, transport=transport)
    assert credential.access == VALID_TOKEN
    assert any(isinstance(event, AuthEventUrl) for event in interaction.notifications)
    assert any(isinstance(p, AuthPromptManualCode) for p in interaction.prompts)


async def test_browser_flow_notify_failure_propagates_and_skips_cleanup() -> None:
    """`L11-SC-R001`: a `notify()` failure happens BEFORE the cleanup boundary -- the server is
    left listening (bind on the same port fails afterward), proving cleanup did not run."""
    controller = RunAbortController()

    class _RaisingInteraction(_FakeInteraction):
        def notify(self, event: object) -> None:
            raise RuntimeError("notify failed")

    interaction = _RaisingInteraction(signal=controller.signal)
    port = _free_loopback_port()
    with pytest.raises(RuntimeError, match="notify failed"):
        await _login_browser(interaction, bind_port=port)

    # the server was never closed -- binding the SAME port again fails
    with pytest.raises(OSError):
        server = await asyncio.start_server(lambda r, w: None, "127.0.0.1", port)
        server.close()
        await server.wait_closed()


async def test_browser_flow_bind_failure_falls_back_to_manual_only() -> None:
    """A port already in use does not crash the flow -- it falls through entirely to the
    manual-code path."""
    occupying = await asyncio.start_server(lambda r, w: None, "127.0.0.1", 0)
    port = occupying.sockets[0].getsockname()[1]
    try:
        controller = RunAbortController()

        async def manual() -> str:
            return "fallbackcode"

        interaction = _FakeInteraction(signal=controller.signal, manual_responses=[manual])
        transport = _FakeTransport()
        transport.queue_json(
            200, {"access_token": VALID_TOKEN, "refresh_token": "rt5", "expires_in": 3600}
        )
        credential = await _login_browser(interaction, bind_port=port, transport=transport)
        assert credential.access == VALID_TOKEN
    finally:
        occupying.close()
        await occupying.wait_closed()


# --- device-code flow ----------------------------------------------------------------------------


async def test_start_device_auth_success() -> None:
    controller = RunAbortController()
    transport = _FakeTransport()
    transport.queue_json(200, {"device_auth_id": "d1", "user_code": "ABCD-EFGH", "interval": 5})
    info = await _start_device_auth(transport, controller.signal)
    assert info.device_auth_id == "d1"
    assert info.user_code == "ABCD-EFGH"
    assert info.interval_seconds == 5.0
    assert transport.calls == [DEVICE_USER_CODE_URL]


async def test_start_device_auth_interval_as_string_uses_js_coercion() -> None:
    controller = RunAbortController()
    transport = _FakeTransport()
    transport.queue_json(200, {"device_auth_id": "d1", "user_code": "u1", "interval": "0x5"})
    info = await _start_device_auth(transport, controller.signal)
    assert info.interval_seconds == 5.0


async def test_start_device_auth_404_is_not_enabled() -> None:
    controller = RunAbortController()
    transport = _FakeTransport()
    transport.queue_response(404, b"")
    with pytest.raises(DeviceCodeNotEnabledError):
        await _start_device_auth(transport, controller.signal)


async def test_start_device_auth_other_failure_status() -> None:
    controller = RunAbortController()
    transport = _FakeTransport()
    transport.queue_response(500, b"server error")
    with pytest.raises(TokenResponseFailedError, match="status 500: server error"):
        await _start_device_auth(transport, controller.signal)


async def test_start_device_auth_truthy_non_string_field_is_rejected() -> None:
    """`PROV-016`, owner-approved divergence: a truthy but non-string field value is rejected,
    NOT accepted the way Pi's own truthy-any-type runtime check would."""
    controller = RunAbortController()
    transport = _FakeTransport()
    transport.queue_json(200, {"device_auth_id": 12345, "user_code": "u1", "interval": 5})
    with pytest.raises(InvalidDeviceCodeResponseError, match="12345"):
        await _start_device_auth(transport, controller.signal)


@pytest.mark.parametrize("bad_field", ["device_auth_id", "user_code"])
async def test_start_device_auth_each_field_absent_or_truthy_non_string(bad_field: str) -> None:
    """`PROV-016`'s own required per-field witnesses (a)/(b)/(c) for `device_auth_id`/`user_code`:
    a valid string is accepted (covered by every success test above); an absent/falsy value is
    rejected via the existing invalid-response path; a present, truthy, non-string value is
    rejected -- this project's own intentional divergence."""
    controller = RunAbortController()

    absent_transport = _FakeTransport()
    valid: dict[str, JsonValue] = {"device_auth_id": "d", "user_code": "u", "interval": 5}
    del valid[bad_field]
    absent_transport.queue_json(200, valid)
    with pytest.raises(InvalidDeviceCodeResponseError):
        await _start_device_auth(absent_transport, controller.signal)

    truthy_transport = _FakeTransport()
    truthy_non_string: dict[str, JsonValue] = {
        "device_auth_id": "d",
        "user_code": "u",
        "interval": 5,
    }
    truthy_non_string[bad_field] = 999
    truthy_transport.queue_json(200, truthy_non_string)
    with pytest.raises(InvalidDeviceCodeResponseError):
        await _start_device_auth(truthy_transport, controller.signal)


async def test_start_device_auth_interval_boolean_is_rejected() -> None:
    """A JSON boolean satisfies `typeof x === "number"` in neither Pi nor this row's own
    `typeof intervalSeconds !== "number"` check -- confirmed rejected, not silently coerced."""
    controller = RunAbortController()
    transport = _FakeTransport()
    transport.queue_json(200, {"device_auth_id": "d1", "user_code": "u1", "interval": True})
    with pytest.raises(InvalidDeviceCodeResponseError):
        await _start_device_auth(transport, controller.signal)


async def test_start_device_auth_interval_wrong_json_type_is_rejected() -> None:
    controller = RunAbortController()
    transport = _FakeTransport()
    transport.queue_json(200, {"device_auth_id": "d1", "user_code": "u1", "interval": [1, 2]})
    with pytest.raises(InvalidDeviceCodeResponseError):
        await _start_device_auth(transport, controller.signal)


async def test_poll_device_auth_immediate_success() -> None:
    controller = RunAbortController()
    transport = _FakeTransport()
    transport.queue_json(200, {"authorization_code": "ac1", "code_verifier": "cv1"})
    device = await _fake_device_info(interval=0.001)
    result = await _poll_device_auth(transport, device, controller.signal)
    assert result.authorization_code == "ac1"
    assert result.code_verifier == "cv1"
    assert transport.calls == [DEVICE_TOKEN_URL]


async def test_poll_device_auth_pending_then_complete() -> None:
    controller = RunAbortController()
    transport = _FakeTransport()
    transport.queue_response(403, b"")
    transport.queue_json(200, {"authorization_code": "ac2", "code_verifier": "cv2"})
    device = await _fake_device_info(interval=0.001)
    result = await _poll_device_auth(transport, device, controller.signal)
    assert result.authorization_code == "ac2"


async def test_poll_device_auth_slow_down_then_complete() -> None:
    controller = RunAbortController()
    transport = _FakeTransport()
    transport.queue_response(429, json.dumps({"error": "slow_down"}).encode())
    transport.queue_json(200, {"authorization_code": "ac3", "code_verifier": "cv3"})
    device = await _fake_device_info(interval=0.001)
    result = await _poll_device_auth(transport, device, controller.signal)
    assert result.authorization_code == "ac3"


async def test_poll_device_auth_pending_via_object_shaped_error_code() -> None:
    """The `error` field may be a nested `{code: "..."}` object rather than a bare string --
    confirmed both shapes route to the same PENDING outcome."""
    controller = RunAbortController()
    transport = _FakeTransport()
    transport.queue_response(
        400, json.dumps({"error": {"code": "deviceauth_authorization_pending"}}).encode()
    )
    transport.queue_json(200, {"authorization_code": "ac4", "code_verifier": "cv4"})
    device = await _fake_device_info(interval=0.001)
    result = await _poll_device_auth(transport, device, controller.signal)
    assert result.authorization_code == "ac4"


async def test_poll_device_auth_unparseable_error_body_is_failed() -> None:
    controller = RunAbortController()
    transport = _FakeTransport()
    transport.queue_response(500, b"not json at all")
    device = await _fake_device_info(interval=0.001)
    with pytest.raises(TokenResponseFailedError, match="not json at all"):
        await _poll_device_auth(transport, device, controller.signal)


async def test_poll_device_auth_unrecognized_error_code_is_failed() -> None:
    controller = RunAbortController()
    transport = _FakeTransport()
    transport.queue_response(400, json.dumps({"error": "something_else"}).encode())
    device = await _fake_device_info(interval=0.001)
    with pytest.raises(TokenResponseFailedError, match="status 400"):
        await _poll_device_auth(transport, device, controller.signal)


async def test_poll_device_auth_truthy_non_string_response_is_failed() -> None:
    controller = RunAbortController()
    transport = _FakeTransport()
    transport.queue_json(200, {"authorization_code": True, "code_verifier": "cv"})
    device = await _fake_device_info(interval=0.001)
    with pytest.raises(TokenResponseFailedError, match="Invalid OpenAI Codex device auth token"):
        await _poll_device_auth(transport, device, controller.signal)


@pytest.mark.parametrize("bad_field", ["authorization_code", "code_verifier"])
async def test_poll_device_auth_each_field_absent_or_truthy_non_string(bad_field: str) -> None:
    """`PROV-016`'s own required per-field witnesses for `authorization_code`/`code_verifier`."""
    controller = RunAbortController()

    absent_transport = _FakeTransport()
    valid: dict[str, JsonValue] = {"authorization_code": "ac", "code_verifier": "cv"}
    del valid[bad_field]
    absent_transport.queue_json(200, valid)
    device = await _fake_device_info(interval=0.001)
    with pytest.raises(TokenResponseFailedError, match="Invalid OpenAI Codex device auth token"):
        await _poll_device_auth(absent_transport, device, controller.signal)

    truthy_transport = _FakeTransport()
    truthy_non_string: dict[str, JsonValue] = {"authorization_code": "ac", "code_verifier": "cv"}
    truthy_non_string[bad_field] = 999
    truthy_transport.queue_json(200, truthy_non_string)
    with pytest.raises(TokenResponseFailedError, match="Invalid OpenAI Codex device auth token"):
        await _poll_device_auth(truthy_transport, device, controller.signal)


class _FakeClock:
    """A deterministic monotonic-shaped fake clock, advanced only by its own paired `sleep` --
    matching this codebase's own established `test_device_code.py` pattern (duplicated here, not
    imported cross-test-file, no precedent for that). Avoids the real-wall-clock flakiness a tight
    real-time deadline would otherwise have under full-suite load."""

    def __init__(self) -> None:
        self.elapsed = 0.0

    def now(self) -> float:
        return self.elapsed

    async def sleep(self, seconds: float) -> None:
        self.elapsed += seconds


async def test_poll_device_auth_timeout_becomes_token_response_failed() -> None:
    controller = RunAbortController()
    transport = _FakeTransport()
    transport.queue_response(403, b"")
    device = await _fake_device_info(interval=0.001)
    clock = _FakeClock()
    with pytest.raises(TokenResponseFailedError):
        await _poll_device_auth(
            transport,
            device,
            controller.signal,
            expires_in_seconds=0.02,
            sleep=clock.sleep,
            now=clock.now,
        )


async def _fake_device_info(*, interval: float = 5.0) -> _DeviceAuthInfo:
    return _DeviceAuthInfo(device_auth_id="d", user_code="u", interval_seconds=interval)


async def test_login_device_code_end_to_end() -> None:
    controller = RunAbortController()
    interaction = _FakeInteraction(signal=controller.signal)
    transport = _FakeTransport()
    transport.queue_json(200, {"device_auth_id": "d1", "user_code": "u1", "interval": 0.001})
    transport.queue_json(200, {"authorization_code": "ac", "code_verifier": "cv"})
    transport.queue_json(
        200, {"access_token": VALID_TOKEN, "refresh_token": "rtd", "expires_in": 3600}
    )
    credential = await _login_device_code(interaction, transport=transport)
    assert credential.access == VALID_TOKEN
    assert any(isinstance(e, AuthEventDeviceCode) for e in interaction.notifications)


# --- token exchange and refresh -------------------------------------------------------------------


async def test_read_token_response_success() -> None:
    response = HttpResponse(
        status=200,
        body=json.dumps({"access_token": "a", "refresh_token": "r", "expires_in": 100}).encode(),
    )
    token = _read_token_response(response, "exchange")
    assert token.access == "a"
    assert token.refresh == "r"


async def test_read_token_response_non_2xx_with_body() -> None:
    response = HttpResponse(status=400, body=b"bad request")
    with pytest.raises(TokenResponseFailedError, match=r"failed \(400\): bad request"):
        _read_token_response(response, "exchange")


async def test_read_token_response_non_2xx_without_body() -> None:
    response = HttpResponse(status=400, body=b"")
    with pytest.raises(TokenResponseFailedError):
        _read_token_response(response, "refresh")


async def test_read_token_response_missing_fields_renders_exact_json() -> None:
    response = HttpResponse(status=200, body=json.dumps({"access_token": "a"}).encode())
    with pytest.raises(
        TokenResponseMissingFieldsError, match=r'missing fields: \{"access_token":"a"\}'
    ):
        _read_token_response(response, "exchange")


async def test_read_token_response_truthy_non_string_token_is_rejected() -> None:
    """`PROV-016`: `access_token`/`refresh_token` must be actual strings."""
    response = HttpResponse(
        status=200,
        body=json.dumps({"access_token": 123, "refresh_token": "r", "expires_in": 10}).encode(),
    )
    with pytest.raises(TokenResponseMissingFieldsError):
        _read_token_response(response, "exchange")


@pytest.mark.parametrize("bad_field", ["access_token", "refresh_token"])
def test_read_token_response_each_field_absent_or_truthy_non_string(bad_field: str) -> None:
    """`PROV-016`'s own required per-field witnesses for `access_token`/`refresh_token`."""
    valid: dict[str, JsonValue] = {"access_token": "a", "refresh_token": "r", "expires_in": 10}
    del valid[bad_field]
    absent_response = HttpResponse(status=200, body=json.dumps(valid).encode())
    with pytest.raises(TokenResponseMissingFieldsError):
        _read_token_response(absent_response, "exchange")

    truthy_non_string: dict[str, JsonValue] = {
        "access_token": "a",
        "refresh_token": "r",
        "expires_in": 10,
    }
    truthy_non_string[bad_field] = 999
    truthy_response = HttpResponse(status=200, body=json.dumps(truthy_non_string).encode())
    with pytest.raises(TokenResponseMissingFieldsError):
        _read_token_response(truthy_response, "exchange")


async def test_exchange_authorization_code_cancellation_translates_to_login_cancelled() -> None:
    controller = RunAbortController()
    controller.abort()
    transport = _FakeTransport()
    transport.queue_error(ConnectionError("boom"))

    with pytest.raises(LoginCancelledError):
        await _exchange_authorization_code(
            "code", "verifier", REDIRECT_URI, controller.signal, transport=transport
        )


async def test_refresh_access_token_no_cancellation_translation() -> None:
    """`L11-SC-R004`: refresh applies NO cancellation-specific translation, even with the signal
    already aborted."""
    controller = RunAbortController()
    controller.abort()
    transport = _FakeTransport()
    transport.queue_error(ConnectionError("boom"))

    with pytest.raises(OAuthRefreshTransportError, match="OpenAI Codex token refresh error"):
        await _refresh_access_token("refresh-token", controller.signal, transport=transport)


async def test_refresh_openai_codex_token_success(monkeypatch: pytest.MonkeyPatch) -> None:
    import minion_agent.auth.openai_codex_oauth as module

    controller = RunAbortController()
    transport = _FakeTransport()
    transport.queue_json(
        200, {"access_token": VALID_TOKEN, "refresh_token": "new_r", "expires_in": 3600}
    )

    monkeypatch.setattr(module, "_default_transport", lambda: transport)
    credential = await _refresh_openai_codex_token(
        OAuthCredential(access="old_a", refresh="old_r", expires=0.0), controller.signal
    )
    assert credential.access == VALID_TOKEN
    assert credential.refresh == "new_r"


# --- login method dispatch ------------------------------------------------------------------------


async def test_login_dispatches_to_browser(monkeypatch: pytest.MonkeyPatch) -> None:
    import minion_agent.auth.openai_codex_oauth as module

    controller = RunAbortController()

    async def manual() -> str:
        return "browsercode"

    interaction = _FakeInteraction(
        signal=controller.signal, select_response="browser", manual_responses=[manual]
    )
    port = _free_loopback_port()
    transport = _FakeTransport()
    transport.queue_json(
        200, {"access_token": VALID_TOKEN, "refresh_token": "rtb", "expires_in": 3600}
    )

    async def fake_login_browser(interaction_arg: object, **_kwargs: object) -> OAuthCredential:
        assert interaction_arg is interaction
        return await _login_browser(interaction, bind_port=port, transport=transport)

    monkeypatch.setattr(module, "_login_browser", fake_login_browser)
    credential = await _login(interaction)
    assert credential.access == VALID_TOKEN


async def test_login_dispatches_to_device_code(monkeypatch: pytest.MonkeyPatch) -> None:
    import minion_agent.auth.openai_codex_oauth as module

    controller = RunAbortController()
    interaction = _FakeInteraction(signal=controller.signal, select_response="device_code")
    transport = _FakeTransport()
    transport.queue_json(200, {"device_auth_id": "d", "user_code": "u", "interval": 0.001})
    transport.queue_json(200, {"authorization_code": "ac", "code_verifier": "cv"})
    transport.queue_json(
        200, {"access_token": VALID_TOKEN, "refresh_token": "rtc", "expires_in": 3600}
    )

    async def fake_login_device_code(interaction_arg: object, **_kwargs: object) -> OAuthCredential:
        assert interaction_arg is interaction
        return await _login_device_code(interaction, transport=transport)

    monkeypatch.setattr(module, "_login_device_code", fake_login_device_code)
    credential = await _login(interaction)
    assert credential.access == VALID_TOKEN


async def test_login_unknown_method_raises_exact_message() -> None:
    controller = RunAbortController()
    interaction = _FakeInteraction(signal=controller.signal, select_response="carrier-pigeon")
    with pytest.raises(
        UnknownLoginMethodError, match="Unknown OpenAI Codex login method: carrier-pigeon"
    ):
        await _login(interaction)


def test_default_transport_returns_httpx_transport() -> None:
    from minion_agent.auth.http_transport import HttpxTransport
    from minion_agent.auth.openai_codex_oauth import _default_transport

    assert isinstance(_default_transport(), HttpxTransport)


# --- `OAuthAuth` composite -----------------------------------------------------------------------


def test_openai_codex_oauth_composite_shape() -> None:
    assert openai_codex_oauth.name == "OpenAI (ChatGPT Plus/Pro)"
    assert openai_codex_oauth.is_subscription is True
    assert openai_codex_oauth.login_label is None


async def test_openai_codex_oauth_to_auth_delegates_to_prov_011() -> None:
    credential = OAuthCredential(access="tok", refresh="r", expires=0.0)
    result = await openai_codex_oauth.to_auth(credential)
    assert result.api_key == "tok"
