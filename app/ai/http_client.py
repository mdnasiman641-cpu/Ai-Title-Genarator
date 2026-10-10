"""Minimal JSON-over-HTTPS client built on the standard library.

Using ``urllib`` instead of the vendors' SDKs keeps the executable small, avoids
two large dependency trees and gives us one place that maps every failure mode
(bad key, rate limit, offline, timeout ...) to :class:`AIError`.
"""

from __future__ import annotations

import json
import socket
import ssl
import urllib.error
import urllib.request
from typing import Any

from app.ai.errors import AIError, ErrorKind
from app.config.constants import APP_NAME, APP_VERSION
from app.utils.logger import get_logger

log = get_logger("ai.http")

_MAX_BODY = 8_000_000
_USER_AGENT = f"{APP_NAME.replace(' ', '-')}/{APP_VERSION}"


def _error_message(raw: bytes) -> str:
    """Extract the human-readable message from a provider error body."""
    try:
        data = json.loads(raw.decode("utf-8", "replace"))
    except ValueError:
        return raw.decode("utf-8", "replace")[:300]
    err = data.get("error") if isinstance(data, dict) else None
    if isinstance(err, dict):
        return str(err.get("message") or err.get("status") or "")[:400]
    if isinstance(err, str):
        return err[:400]
    return ""


def classify_http_error(provider: str, status: int, message: str) -> AIError:
    """Map an HTTP status (and message) to the right :class:`AIError`."""
    lowered = message.lower()
    if status in (401, 403) or "api key not valid" in lowered or "api_key_invalid" in lowered or "incorrect api key" in lowered:
        return AIError(ErrorKind.AUTH, provider, message, status)
    if status == 429:
        return AIError(ErrorKind.RATE_LIMIT, provider, message, status)
    if status == 404 or ("model" in lowered and ("not found" in lowered or "does not exist" in lowered)):
        return AIError(ErrorKind.MODEL_NOT_FOUND, provider, message, status)
    if status in (408, 504):
        return AIError(ErrorKind.TIMEOUT, provider, message, status)
    if status >= 500:
        return AIError(ErrorKind.SERVER, provider, message, status)
    if status == 400:
        return AIError(ErrorKind.BAD_REQUEST, provider, message, status)
    return AIError(ErrorKind.UNKNOWN, provider, f"HTTP {status}: {message}", status)


def request_json(
    provider: str,
    method: str,
    url: str,
    *,
    headers: dict[str, str],
    payload: dict[str, Any] | None = None,
    timeout: float = 60,
) -> dict[str, Any]:
    """Send a JSON request and return the decoded JSON object, raising :class:`AIError` on failure."""
    body = json.dumps(payload).encode("utf-8") if payload is not None else None
    all_headers = {"Content-Type": "application/json", "Accept": "application/json", "User-Agent": _USER_AGENT, **headers}
    request = urllib.request.Request(url, data=body, headers=all_headers, method=method)  # noqa: S310 - https only
    try:
        context = ssl.create_default_context()
        with urllib.request.urlopen(request, timeout=timeout, context=context) as response:  # noqa: S310
            raw = response.read(_MAX_BODY + 1)
    except urllib.error.HTTPError as exc:
        message = _error_message(exc.read(20_000) if exc.fp else b"")
        log.warning("%s HTTP %s", provider, exc.code)
        raise classify_http_error(provider, exc.code, message) from None
    except (TimeoutError, socket.timeout):
        raise AIError(ErrorKind.TIMEOUT, provider, f"No answer within {timeout:.0f}s") from None
    except urllib.error.URLError as exc:
        if isinstance(exc.reason, (TimeoutError, socket.timeout)):
            raise AIError(ErrorKind.TIMEOUT, provider, f"No answer within {timeout:.0f}s") from None
        raise AIError(ErrorKind.NETWORK, provider, str(exc.reason)) from None
    except ssl.SSLError as exc:
        raise AIError(ErrorKind.NETWORK, provider, f"TLS error: {exc}") from None
    except (ConnectionError, OSError) as exc:
        raise AIError(ErrorKind.NETWORK, provider, str(exc)) from None
    if len(raw) > _MAX_BODY:
        raise AIError(ErrorKind.INVALID_RESPONSE, provider, "Response is unreasonably large")
    try:
        data = json.loads(raw.decode("utf-8"))
    except ValueError:
        raise AIError(ErrorKind.INVALID_RESPONSE, provider, "The server did not return JSON") from None
    if not isinstance(data, dict):
        raise AIError(ErrorKind.INVALID_RESPONSE, provider, "Unexpected response shape")
    return data
