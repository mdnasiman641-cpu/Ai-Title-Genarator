"""Error taxonomy for AI calls, with friendly (translatable) messages."""

from __future__ import annotations

from enum import Enum

from app.i18n import tr
from app.utils.security import redact


class ErrorKind(str, Enum):
    NO_KEY = "no_key"
    AUTH = "auth"
    NETWORK = "network"
    RATE_LIMIT = "rate_limit"
    TIMEOUT = "timeout"
    SERVER = "server"
    BAD_REQUEST = "bad_request"
    MODEL_NOT_FOUND = "model_not_found"
    BLOCKED = "blocked"
    INVALID_RESPONSE = "invalid_response"
    CANCELLED = "cancelled"
    LISTING_UNSUPPORTED = "listing_unsupported"
    NO_MODELS = "no_models"
    UNKNOWN = "unknown"


# Failures worth retrying on the *same* model (transient); everything else moves on to the next model.
RETRYABLE_SAME_MODEL = frozenset({ErrorKind.TIMEOUT, ErrorKind.SERVER, ErrorKind.NETWORK, ErrorKind.INVALID_RESPONSE})


PROVIDER_NAMES = {"gemini": "Gemini", "openai": "OpenAI"}


def provider_name(provider_id: str) -> str:
    """Display name of a provider id (``gemini`` -> ``Gemini``)."""
    return PROVIDER_NAMES.get(provider_id, provider_id)


class AIError(Exception):
    """A failed AI request. ``detail`` is technical text (never contains API keys)."""

    def __init__(self, kind: ErrorKind, provider: str, detail: str = "", status: int | None = None) -> None:
        self.kind = kind
        self.provider = provider
        self.detail = redact(detail)[:600]
        self.status = status
        super().__init__(f"{provider_name(provider)}: {kind.value}: {self.detail}")

    @property
    def headline(self) -> str:
        """e.g. ``Gemini request failed.``"""
        return tr("ai.request_failed", provider=provider_name(self.provider))

    @property
    def reason(self) -> str:
        """The friendly reason for this kind of failure."""
        return tr(f"ai.err.{self.kind.value}")

    @property
    def user_message(self) -> str:
        """Headline plus reason (and the provider's own detail when it adds information)."""
        text = f"{self.headline} {self.reason}"
        if self.detail and self.kind in {ErrorKind.BAD_REQUEST, ErrorKind.UNKNOWN, ErrorKind.MODEL_NOT_FOUND, ErrorKind.BLOCKED}:
            text += f"\n{self.detail}"
        return text

    @property
    def allows_fallback(self) -> bool:
        """Errors for which trying another model is sensible (429, 5xx, timeouts, bad JSON, auth, ...)."""
        return self.kind not in {ErrorKind.CANCELLED}

    @property
    def category(self) -> str:
        """Short label for logs and the UI, e.g. ``HTTP 429`` or ``timeout``."""
        return f"HTTP {self.status}" if self.status else self.kind.value
