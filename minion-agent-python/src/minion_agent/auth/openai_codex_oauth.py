"""Codex OAuth NETWORK integration (`PROV-012`, Pass 2 Slice C; Pi
`packages/ai/src/auth/oauth/openai-codex.ts`).

Consumes the already-adopted `PROV-011` account-id projection (`openai_codex.py`), the
already-adopted `PROV-014` interaction/auth-method vocabulary (`interaction.py`), and the
already-certified `PROV-009`/`PROV-010` PKCE/device-poll primitives. Implements the browser/PKCE
flow (local callback server, manual-code racing), the device-code flow, and token exchange/refresh,
per `spec/auth.md`'s own `PROV-012`/`PROV-016` sections -- the durable, reviewed, owner-approved
language-neutral contract this module is answerable to.
"""

from __future__ import annotations

import asyncio
import contextlib
import os
import re
import secrets
import time
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from urllib.parse import parse_qs, urlencode, urlsplit

from ..runtime.signal import RunAbortController
from .credential import JsonValue, ModelAuth, OAuthCredential
from .device_code import (
    DeviceFlowCancelled,
    DeviceFlowFailed,
    DeviceFlowTimedOut,
    DevicePollComplete,
    DevicePollFailed,
    DevicePollPending,
    DevicePollResult,
    DevicePollSlowDown,
    poll_device_code_flow,
)
from .http_transport import (
    HttpResponse,
    HttpTransport,
    HttpxTransport,
    fetch_with_login_cancellation,
    refresh_fetch,
)
from .interaction import (
    AuthEventDeviceCode,
    AuthEventUrl,
    AuthPromptManualCode,
    AuthPromptOption,
    AuthPromptSelect,
    OAuthAuth,
    ProviderAuthInteraction,
)
from .js_json import js_json_stringify
from .openai_codex import credentials_from_token
from .openai_codex import to_auth as _codex_to_auth
from .pkce import generate_pkce
from .signal import Abortable

# --- Fixed identity/endpoint constants (`spec/auth.md`'s own literal, observable values) --------

CLIENT_ID = "app_EMoamEEZ73f0CkXaXp7hrann"
AUTH_BASE_URL = "https://auth.openai.com"
AUTHORIZE_URL = f"{AUTH_BASE_URL}/oauth/authorize"
TOKEN_URL = f"{AUTH_BASE_URL}/oauth/token"
REDIRECT_URI = "http://localhost:1455/auth/callback"
DEVICE_USER_CODE_URL = f"{AUTH_BASE_URL}/api/accounts/deviceauth/usercode"
DEVICE_TOKEN_URL = f"{AUTH_BASE_URL}/api/accounts/deviceauth/token"
DEVICE_VERIFICATION_URI = f"{AUTH_BASE_URL}/codex/device"
DEVICE_REDIRECT_URI = f"{AUTH_BASE_URL}/deviceauth/callback"
DEVICE_CODE_TIMEOUT_SECONDS = 15 * 60
SCOPE = "openid profile email offline_access"
LOGIN_METHOD_BROWSER = "browser"
LOGIN_METHOD_DEVICE_CODE = "device_code"
CALLBACK_PATH = "/auth/callback"
CALLBACK_PORT = 1455

_ABORT_POLL_INTERVAL_SECONDS = 0.05
"""The same short, fixed poll interval `abortable_sleep`/`run_cancellable` already establish for
checking a poll-based `Abortable` signal "promptly enough" without redesigning it."""


def _callback_host() -> str:
    """`getCallbackHost()` (`openai-codex.ts:44-46`) -- a JAVASCRIPT-TRUTHINESS check, NOT
    blank-normalization: `getProviderEnvValue` performs no trimming at any point, so a
    whitespace-only value is truthy and used verbatim; only a literally empty/absent value falls
    back to the default. Read directly from the process environment, NOT through the
    `AuthContext` seam `ApiKeyCheck`/`ApiKeyResolve` use -- Pi's own `OAuthLogin`/`OAuthRefresh`/
    `OAuthToAuth` callables never receive an `AuthContext` at all."""
    return os.environ.get("PI_OAUTH_CALLBACK_HOST") or "127.0.0.1"


# --- Errors (each maps 1:1 to one of this row's own exact Pi error messages) ---------------------


class UnknownLoginMethodError(Exception):
    """`"Unknown OpenAI Codex login method: {method}"`."""


class StateMismatchError(Exception):
    """`"State mismatch"`."""


class MissingAuthorizationCodeError(Exception):
    """`"Missing authorization code"`."""


class DeviceCodeNotEnabledError(Exception):
    """`"OpenAI Codex device code login is not enabled for this server. Use browser login or
    verify the server URL."`."""


class InvalidDeviceCodeResponseError(Exception):
    """`"Invalid OpenAI Codex device code response: {rendered json}"`."""


class TokenResponseFailedError(Exception):
    """`"OpenAI Codex token {exchange|refresh} failed ({status}): {body}"`."""


class TokenResponseMissingFieldsError(Exception):
    """`"OpenAI Codex token {exchange|refresh} response missing fields: {rendered json}"`."""


# --- Browser/PKCE flow: authorization URL construction -------------------------------------------


@dataclass(frozen=True, slots=True)
class _AuthorizationFlow:
    verifier: str
    state: str
    url: str


def _create_state() -> str:
    """16 cryptographically random bytes, hex-encoded -- Pi's own `randomBytes(16).toString
    ("hex")`."""
    return secrets.token_bytes(16).hex()


def _create_authorization_flow(originator: str = "pi") -> _AuthorizationFlow:
    """`createAuthorizationFlow` (`openai-codex.ts:293-312`). `originator` is a FIXED literal
    `"pi"` at Pi's own single real call site -- not a configurable per-provider value here
    either."""
    pkce = generate_pkce()
    state = _create_state()
    params = {
        "response_type": "code",
        "client_id": CLIENT_ID,
        "redirect_uri": REDIRECT_URI,
        "scope": SCOPE,
        "code_challenge": pkce.challenge,
        "code_challenge_method": "S256",
        "state": state,
        "id_token_add_organizations": "true",
        "codex_cli_simplified_flow": "true",
        "originator": originator,
    }
    url = f"{AUTHORIZE_URL}?{urlencode(params)}"
    return _AuthorizationFlow(verifier=pkce.verifier, state=state, url=url)


# --- Browser/PKCE flow: local callback server -----------------------------------------------------

_STATUS_TEXT = {200: "OK", 400: "Bad Request", 404: "Not Found", 500: "Internal Server Error"}


def _oauth_page(message: str) -> bytes:
    """The response body content itself is presentation, out of this contract's own observable
    surface (`spec/auth.md`) -- a minimal placeholder page, not Pi's own real HTML template."""
    return f"<!doctype html><html><body><p>{message}</p></body></html>".encode()


def _first_query_value(query: dict[str, list[str]], key: str) -> str | None:
    values = query.get(key)
    return values[0] if values else None


def _http_response_bytes(status: int, body: bytes) -> bytes:
    """Every response shares the SAME `Content-Type: text/html; charset=utf-8` header
    (`L11-SC-R008`), regardless of status."""
    header = (
        f"HTTP/1.1 {status} {_STATUS_TEXT[status]}\r\n"
        f"Content-Type: text/html; charset=utf-8\r\n"
        f"Content-Length: {len(body)}\r\n"
        f"Connection: close\r\n"
        f"\r\n"
    ).encode("ascii")
    return header + body


class _CallbackServer:
    """The local OAuth callback HTTP server (`startLocalOAuthServer`, `openai-codex.ts:320-394`).
    Binding is NOT infallible: `start()` catches a bind failure and makes `wait_for_code()`
    resolve absent immediately and forever, matching Pi's own `.on("error", ...)` handler --
    never crashing the login flow."""

    def __init__(self) -> None:
        self._server: asyncio.base_events.Server | None = None
        self._future: asyncio.Future[str | None] = asyncio.get_event_loop().create_future()
        self._settled = False

    def _settle(self, value: str | None) -> None:
        if self._settled:
            return
        self._settled = True
        if not self._future.done():
            self._future.set_result(value)

    async def start(self, state: str, host: str, port: int) -> None:
        async def handle(reader: asyncio.StreamReader, writer: asyncio.StreamWriter) -> None:
            try:
                request_line = await reader.readline()
                while True:
                    line = await reader.readline()
                    if line in (b"\r\n", b"\n", b""):
                        break
                parts = request_line.decode("iso-8859-1").split(" ")
                target = parts[1] if len(parts) >= 2 else "/"
                parsed = urlsplit(target)

                if parsed.path != CALLBACK_PATH:
                    status, message = 404, "Callback route not found."
                    body = _oauth_page(message)
                    code_to_settle: str | None = None
                else:
                    query = parse_qs(parsed.query, keep_blank_values=True)
                    got_state = _first_query_value(query, "state")
                    if got_state != state:
                        status, message = 400, "State mismatch."
                        body = _oauth_page(message)
                        code_to_settle = None
                    else:
                        code = _first_query_value(query, "code") or ""
                        if not code:
                            status, message = 400, "Missing authorization code."
                            body = _oauth_page(message)
                            code_to_settle = None
                        else:
                            status = 200
                            body = _oauth_page(
                                "OpenAI authentication completed. You can close this window."
                            )
                            code_to_settle = code

                writer.write(_http_response_bytes(status, body))
                await writer.drain()
                if code_to_settle is not None:
                    self._settle(code_to_settle)
            except Exception:
                with contextlib.suppress(Exception):
                    writer.write(
                        _http_response_bytes(
                            500, _oauth_page("Internal error while processing OAuth callback.")
                        )
                    )
                    await writer.drain()
            finally:
                with contextlib.suppress(Exception):
                    writer.close()

        try:
            self._server = await asyncio.start_server(handle, host, port)
        except OSError:
            self._server = None
            self._settle(None)

    def cancel_wait(self) -> None:
        self._settle(None)

    async def wait_for_code(self) -> str | None:
        return await self._future

    async def close(self) -> None:
        if self._server is not None:
            self._server.close()
            await self._server.wait_closed()


# --- Browser/PKCE flow: manual pasted-input parsing ----------------------------------------------


@dataclass(frozen=True, slots=True)
class ParsedAuthorizationInput:
    code: str | None
    state: str | None


def parse_authorization_input(raw_input: str) -> ParsedAuthorizationInput:
    """`parseAuthorizationInput` (`openai-codex.ts:73-101`). Tries, in order: empty/whitespace;
    absolute URL; a `"#"`-containing shape (Pi's own `split("#", 2)` semantics -- content after a
    SECOND `"#"`, if any, is DISCARDED, never appended to `state`, unlike a host language's own
    unlimited split on the first `"#"`); a bare `"code="`-containing query string; otherwise the
    whole trimmed input as a bare code."""
    value = raw_input.strip()
    if not value:
        return ParsedAuthorizationInput(code=None, state=None)

    parsed_url = urlsplit(value)
    if parsed_url.scheme and parsed_url.netloc:
        query = parse_qs(parsed_url.query, keep_blank_values=True)
        return ParsedAuthorizationInput(
            code=_first_query_value(query, "code"),
            state=_first_query_value(query, "state"),
        )

    if "#" in value:
        code, remainder = value.split("#", 1)
        state = remainder.split("#", 1)[0]
        return ParsedAuthorizationInput(code=code, state=state)

    if "code=" in value:
        query = parse_qs(value, keep_blank_values=True)
        return ParsedAuthorizationInput(
            code=_first_query_value(query, "code"),
            state=_first_query_value(query, "state"),
        )

    return ParsedAuthorizationInput(code=value, state=None)


# --- Browser/PKCE flow: orchestration -------------------------------------------------------------


_OnAbort = Callable[[], None]


async def _watch_abort(signal: Abortable, on_abort: _OnAbort) -> None:
    """The poll-based mapping of Pi's own push-based `interaction.signal.addEventListener
    ("abort", onAbort, {once: true})` (`openai-codex.ts:450`): `Abortable` has no push/event
    mechanism (Layer 09, poll-based by certified design), so this polls at a short, fixed
    interval, the same idiom `abortable_sleep`/`run_cancellable` already establish."""
    while not signal.aborted:
        await asyncio.sleep(_ABORT_POLL_INTERVAL_SECONDS)
    on_abort()


async def _login_browser(
    interaction: ProviderAuthInteraction,
    *,
    bind_port: int = CALLBACK_PORT,
    transport: HttpTransport | None = None,
) -> OAuthCredential:
    """`loginOpenAICodex` (`openai-codex.ts:445-506`). `bind_port`/`transport` are TESTABILITY
    seams only -- `bind_port` changes where the local callback server itself binds, never
    `REDIRECT_URI`'s own claimed port (always the fixed `:1455`, matching `spec/auth.md`'s own
    "not configurable" contract for the real authorization URL); a test binds an ephemeral port
    here instead of the real one and simulates the callback by connecting directly to it, never
    touching production port 1455. `transport` defaults to the real `httpx`-backed transport."""
    flow = _create_authorization_flow()
    server = _CallbackServer()
    await server.start(flow.state, _callback_host(), bind_port)

    watcher_task = asyncio.ensure_future(_watch_abort(interaction.signal, server.cancel_wait))
    if interaction.signal.aborted:
        server.cancel_wait()

    interaction.notify(
        AuthEventUrl(
            url=flow.url,
            instructions="A browser window should open. Complete login to finish.",
        )
    )
    # A `notify()` failure above propagates directly -- execution never reaches the `try` below,
    # so `watcher_task`/`server` are deliberately left uncleaned (`L11-SC-R001`).

    manual_controller = RunAbortController()
    manual_code: str | None = None
    manual_error: BaseException | None = None

    async def run_manual_prompt() -> None:
        nonlocal manual_code, manual_error
        try:
            manual_code = await interaction.prompt(
                AuthPromptManualCode(
                    message=(
                        "Complete login in your browser, or paste the authorization code / "
                        "redirect URL here:"
                    ),
                    placeholder=REDIRECT_URI,
                    signal=manual_controller.signal,
                )
            )
            server.cancel_wait()
        except BaseException as error:
            manual_error = error
            server.cancel_wait()

    try:
        manual_task = asyncio.ensure_future(run_manual_prompt())

        code: str | None = None
        result_code = await server.wait_for_code()
        if manual_error is not None:
            raise manual_error
        if result_code:
            code = result_code
        elif manual_code:
            parsed = parse_authorization_input(manual_code)
            if parsed.state and parsed.state != flow.state:
                raise StateMismatchError("State mismatch")
            code = parsed.code

        if not code:
            await manual_task
            if manual_error is not None:
                raise manual_error
            if manual_code:
                parsed = parse_authorization_input(manual_code)
                if parsed.state and parsed.state != flow.state:
                    raise StateMismatchError("State mismatch")
                code = parsed.code

        if not code:
            raise MissingAuthorizationCodeError("Missing authorization code")

        return await _exchange_authorization_code_for_credentials(
            code, flow.verifier, REDIRECT_URI, interaction.signal, transport=transport
        )
    finally:
        watcher_task.cancel()
        manual_controller.abort()
        await server.close()


# --- Device-code flow --------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class _DeviceAuthInfo:
    device_auth_id: str
    user_code: str
    interval_seconds: float


_HEX_LITERAL = re.compile(r"^[+-]?0[xX][0-9a-fA-F]+$")
_OCTAL_LITERAL = re.compile(r"^[+-]?0[oO][0-7]+$")
_BINARY_LITERAL = re.compile(r"^[+-]?0[bB][01]+$")
_DECIMAL_LITERAL = re.compile(r"^[+-]?(\d+\.?\d*|\.\d+)([eE][+-]?\d+)?$")


def _js_number_coerce(raw: str) -> float:
    """JavaScript's own `Number(string)` coercion (ECMA-262 `StringToNumber`) -- empirically
    confirmed live against Node, NOT equivalent to a naive `float(trimmed)` parse: an empty
    (post-trim) string coerces to `0`; hex/octal/binary-prefixed strings parse in that base;
    `"Infinity"`/`"+Infinity"`/`"-Infinity"` coerce to the corresponding infinite value; anything
    else that fails to parse coerces to `NaN`."""
    trimmed = raw.strip()
    if trimmed == "":
        return 0.0
    if trimmed in ("Infinity", "+Infinity"):
        return float("inf")
    if trimmed == "-Infinity":
        return float("-inf")
    for pattern, base in ((_HEX_LITERAL, 16), (_OCTAL_LITERAL, 8), (_BINARY_LITERAL, 2)):
        if pattern.match(trimmed):
            sign = -1 if trimmed[0] == "-" else 1
            digits = trimmed.lstrip("+-")[2:]
            return float(sign * int(digits, base))
    if _DECIMAL_LITERAL.match(trimmed):
        return float(trimmed)
    return float("nan")


def _is_finite_non_negative_number(value: float) -> bool:
    is_nan = value != value
    is_infinite = value in (float("inf"), float("-inf"))
    return not is_nan and not is_infinite and value >= 0


async def _start_device_auth(transport: HttpTransport, signal: Abortable) -> _DeviceAuthInfo:
    """`startOpenAICodexDeviceAuth` (`openai-codex.ts:191-233`)."""
    import json as _json

    response = await fetch_with_login_cancellation(
        transport,
        DEVICE_USER_CODE_URL,
        headers={"Content-Type": "application/json"},
        body=_json.dumps({"client_id": CLIENT_ID}).encode(),
        signal=signal,
    )
    if response.status < 200 or response.status >= 300:
        if response.status == 404:
            raise DeviceCodeNotEnabledError(
                "OpenAI Codex device code login is not enabled for this server. "
                "Use browser login or verify the server URL."
            )
        body_text = response.text()
        suffix = f": {body_text}" if body_text else ""
        raise TokenResponseFailedError(
            f"OpenAI Codex device code request failed with status {response.status}{suffix}"
        )

    parsed: JsonValue = _json.loads(response.text())
    device_auth_id = parsed.get("device_auth_id") if isinstance(parsed, dict) else None
    user_code = parsed.get("user_code") if isinstance(parsed, dict) else None
    interval_raw = parsed.get("interval") if isinstance(parsed, dict) else None

    if isinstance(interval_raw, str):
        interval_seconds = _js_number_coerce(interval_raw)
    elif isinstance(interval_raw, bool):
        interval_seconds = float("nan")
    elif isinstance(interval_raw, int | float):
        interval_seconds = float(interval_raw)
    else:
        interval_seconds = float("nan")

    if (
        not isinstance(device_auth_id, str)
        or not device_auth_id
        or not isinstance(user_code, str)
        or not user_code
        or not _is_finite_non_negative_number(interval_seconds)
    ):
        raise InvalidDeviceCodeResponseError(
            f"Invalid OpenAI Codex device code response: {js_json_stringify(parsed)}"
        )

    return _DeviceAuthInfo(
        device_auth_id=device_auth_id, user_code=user_code, interval_seconds=interval_seconds
    )


@dataclass(frozen=True, slots=True)
class _DeviceTokenSuccess:
    authorization_code: str
    code_verifier: str


async def _poll_device_auth(
    transport: HttpTransport,
    device: _DeviceAuthInfo,
    signal: Abortable,
    *,
    expires_in_seconds: float = DEVICE_CODE_TIMEOUT_SECONDS,
    sleep: Callable[[float], Awaitable[None]] = asyncio.sleep,
    now: Callable[[], float] = time.monotonic,
) -> _DeviceTokenSuccess:
    """`pollOpenAICodexDeviceAuth` (`openai-codex.ts:235-291`), driven through the already-
    certified `PROV-010` poll state machine. `expires_in_seconds`/`sleep`/`now` are testability
    seams (the latter two passed straight through to `poll_device_code_flow`'s own already-
    established injection points), defaulting to the real fixed `DEVICE_CODE_TIMEOUT_SECONDS` and
    real wall-clock timing."""
    import json as _json

    async def poll() -> DevicePollResult[_DeviceTokenSuccess]:
        response = await fetch_with_login_cancellation(
            transport,
            DEVICE_TOKEN_URL,
            headers={"Content-Type": "application/json"},
            body=_json.dumps(
                {"device_auth_id": device.device_auth_id, "user_code": device.user_code}
            ).encode(),
            signal=signal,
        )
        if 200 <= response.status < 300:
            parsed: JsonValue = _json.loads(response.text())
            authorization_code = (
                parsed.get("authorization_code") if isinstance(parsed, dict) else None
            )
            code_verifier = parsed.get("code_verifier") if isinstance(parsed, dict) else None
            if (
                not isinstance(authorization_code, str)
                or not authorization_code
                or not isinstance(code_verifier, str)
                or not code_verifier
            ):
                return DevicePollFailed(
                    message=(
                        "Invalid OpenAI Codex device auth token response: "
                        f"{js_json_stringify(parsed)}"
                    )
                )
            return DevicePollComplete(
                value=_DeviceTokenSuccess(
                    authorization_code=authorization_code, code_verifier=code_verifier
                )
            )

        if response.status in (403, 404):
            return DevicePollPending()

        body_text = response.text()
        error_code: str | None = None
        try:
            error_body: JsonValue = _json.loads(body_text)
        except ValueError:
            error_body = None
        if isinstance(error_body, dict):
            error_field = error_body.get("error")
            if isinstance(error_field, str):
                error_code = error_field
            elif isinstance(error_field, dict):
                code_value = error_field.get("code")
                error_code = code_value if isinstance(code_value, str) else None

        if error_code == "deviceauth_authorization_pending":
            return DevicePollPending()
        if error_code == "slow_down":
            return DevicePollSlowDown()

        suffix = f": {body_text}" if body_text else ""
        return DevicePollFailed(
            message=f"OpenAI Codex device auth failed with status {response.status}{suffix}"
        )

    try:
        return await poll_device_code_flow(
            poll,
            interval_seconds=device.interval_seconds,
            expires_in_seconds=expires_in_seconds,
            signal=signal,
            sleep=sleep,
            now=now,
        )
    except (DeviceFlowCancelled, DeviceFlowTimedOut, DeviceFlowFailed) as error:
        raise TokenResponseFailedError(str(error)) from error


async def _login_device_code(
    interaction: ProviderAuthInteraction, *, transport: HttpTransport | None = None
) -> OAuthCredential:
    """`loginOpenAICodexDeviceCode` (`openai-codex.ts:427-443`). `transport` is a testability
    seam, defaulting to the real `httpx`-backed transport."""
    transport = transport or _default_transport()
    device = await _start_device_auth(transport, interaction.signal)
    interaction.notify(
        AuthEventDeviceCode(
            user_code=device.user_code,
            verification_uri=DEVICE_VERIFICATION_URI,
            interval_seconds=device.interval_seconds,
            expires_in_seconds=DEVICE_CODE_TIMEOUT_SECONDS,
        )
    )
    result = await _poll_device_auth(transport, device, interaction.signal)
    return await _exchange_authorization_code_for_credentials(
        result.authorization_code,
        result.code_verifier,
        DEVICE_REDIRECT_URI,
        interaction.signal,
        transport=transport,
    )


# --- Token exchange and refresh ------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class _OAuthToken:
    access: str
    refresh: str
    expires: float


def _read_token_response(response: HttpResponse, operation: str) -> _OAuthToken:
    """`readTokenResponse` (`openai-codex.ts:126-147`). A JSON-parse failure on a `2xx` response
    propagates raw, UNCHANGED (`L11-SC-R004`) -- this function does not catch it."""
    import json as _json
    import time as _time

    if not (200 <= response.status < 300):
        body_text = response.text()
        suffix = f": {body_text}" if body_text else f": {_STATUS_TEXT.get(response.status, '')}"
        raise TokenResponseFailedError(
            f"OpenAI Codex token {operation} failed ({response.status}){suffix}"
        )

    parsed: JsonValue = _json.loads(response.text())
    access_token = parsed.get("access_token") if isinstance(parsed, dict) else None
    refresh_token = parsed.get("refresh_token") if isinstance(parsed, dict) else None
    expires_in = parsed.get("expires_in") if isinstance(parsed, dict) else None

    valid_expires_in = isinstance(expires_in, int | float) and not isinstance(expires_in, bool)

    if (
        not isinstance(access_token, str)
        or not access_token
        or not isinstance(refresh_token, str)
        or not refresh_token
        or not valid_expires_in
    ):
        raise TokenResponseMissingFieldsError(
            f"OpenAI Codex token {operation} response missing fields: {js_json_stringify(parsed)}"
        )

    assert isinstance(expires_in, int | float)
    return _OAuthToken(
        access=access_token,
        refresh=refresh_token,
        expires=_time.time() * 1000.0 + float(expires_in) * 1000.0,
    )


def _default_transport() -> HttpTransport:
    return HttpxTransport()


async def _exchange_authorization_code(
    code: str,
    verifier: str,
    redirect_uri: str,
    signal: Abortable,
    *,
    transport: HttpTransport | None = None,
) -> _OAuthToken:
    transport = transport or _default_transport()
    response = await fetch_with_login_cancellation(
        transport,
        TOKEN_URL,
        headers={"Content-Type": "application/x-www-form-urlencoded"},
        body=urlencode(
            {
                "grant_type": "authorization_code",
                "client_id": CLIENT_ID,
                "code": code,
                "code_verifier": verifier,
                "redirect_uri": redirect_uri,
            }
        ).encode(),
        signal=signal,
    )
    return _read_token_response(response, "exchange")


async def _exchange_authorization_code_for_credentials(
    code: str,
    verifier: str,
    redirect_uri: str,
    signal: Abortable,
    *,
    transport: HttpTransport | None = None,
) -> OAuthCredential:
    token = await _exchange_authorization_code(
        code, verifier, redirect_uri, signal, transport=transport
    )
    return credentials_from_token(token.access, token.refresh, token.expires)


async def _refresh_access_token(
    refresh_token: str, signal: Abortable, *, transport: HttpTransport | None = None
) -> _OAuthToken:
    transport = transport or _default_transport()
    response = await refresh_fetch(
        transport,
        TOKEN_URL,
        headers={"Content-Type": "application/x-www-form-urlencoded"},
        body=urlencode(
            {
                "grant_type": "refresh_token",
                "refresh_token": refresh_token,
                "client_id": CLIENT_ID,
            }
        ).encode(),
        signal=signal,
    )
    return _read_token_response(response, "refresh")


async def _refresh_openai_codex_token(
    credential: OAuthCredential, signal: Abortable
) -> OAuthCredential:
    token = await _refresh_access_token(credential.refresh, signal)
    return credentials_from_token(token.access, token.refresh, token.expires)


# --- `OAuthAuth` composite -----------------------------------------------------------------------


async def _login(interaction: ProviderAuthInteraction) -> OAuthCredential:
    method = await interaction.prompt(
        AuthPromptSelect(
            message="Select OpenAI Codex login method:",
            options=(
                AuthPromptOption(id=LOGIN_METHOD_BROWSER, label="Browser login (default)"),
                AuthPromptOption(id=LOGIN_METHOD_DEVICE_CODE, label="Device code login (headless)"),
            ),
        )
    )
    if method == LOGIN_METHOD_DEVICE_CODE:
        return await _login_device_code(interaction)
    if method != LOGIN_METHOD_BROWSER:
        raise UnknownLoginMethodError(f"Unknown OpenAI Codex login method: {method}")
    return await _login_browser(interaction)


async def _to_auth(credential: OAuthCredential) -> ModelAuth:
    return _codex_to_auth(credential)


openai_codex_oauth = OAuthAuth(
    name="OpenAI (ChatGPT Plus/Pro)",
    login=_login,
    refresh=_refresh_openai_codex_token,
    to_auth=_to_auth,
    is_subscription=True,
)
