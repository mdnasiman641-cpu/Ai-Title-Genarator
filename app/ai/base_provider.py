"""Provider abstraction: every AI backend implements :class:`AIProvider`.

The router only ever talks to this interface. Request formatting, authentication
headers and response parsing differ per vendor and stay inside the adapters.
"""

from __future__ import annotations

import json
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field

from app.ai.errors import AIError, ErrorKind
from app.i18n import tr


@dataclass(frozen=True, slots=True)
class AIRequest:
    """What we ask a model: a fixed system prompt and one user message."""

    system_prompt: str
    user_prompt: str


@dataclass(slots=True)
class AIResponse:
    """Raw model output plus bookkeeping."""

    text: str
    provider: str
    model: str
    truncated: bool = False  # the model hit its output limit
    usage: dict[str, int] = field(default_factory=dict)


@dataclass(slots=True)
class ConnectionResult:
    """Result of *Test Connection*."""

    ok: bool
    message: str
    models: list[str] = field(default_factory=list)
    latency_s: float | None = None


PING = AIRequest(
    system_prompt="You are a health check. Reply with JSON only.",
    user_prompt='Reply with exactly this JSON object: {"ok": true}',
)


class AIProvider(ABC):
    """A configured connection to one AI endpoint + model."""

    id: str = ""  # provider type
    display_name: str = ""
    key_required: bool = True
    supports_listing: bool = True
    default_base_url: str = ""

    def __init__(self, api_key: str | None, model: str, timeout: float = 60, base_url: str = "", label: str = "") -> None:
        if self.key_required and not api_key:
            raise AIError(ErrorKind.NO_KEY, label or self.display_name)
        self._api_key = api_key or ""
        self.model = model
        self.timeout = timeout
        self.base_url = (base_url or self.default_base_url).strip().rstrip("/")
        self.error_name = label or self.display_name  # used in user-facing error messages
        if not self.base_url:
            raise AIError(ErrorKind.BAD_REQUEST, self.error_name, "No API Base URL is configured.")

    @abstractmethod
    def generate(self, request: AIRequest) -> AIResponse:
        """Send ``request`` and return the model's JSON answer (raises :class:`AIError`)."""

    def list_models(self) -> list[str]:
        """Model ids available at this endpoint; raises ``LISTING_UNSUPPORTED`` if there is no list."""
        raise AIError(ErrorKind.LISTING_UNSUPPORTED, self.error_name)

    def test_connection(self) -> ConnectionResult:
        """Validate URL + key, send one tiny request, time it and check the answer is JSON."""
        models: list[str] = []
        if self.supports_listing:
            try:
                models = self.list_models()
            except AIError as exc:
                if exc.kind not in (ErrorKind.LISTING_UNSUPPORTED, ErrorKind.MODEL_NOT_FOUND):
                    return ConnectionResult(False, exc.user_message)
        start = time.perf_counter()
        try:
            response = self.generate(PING)
        except AIError as exc:
            return ConnectionResult(False, exc.user_message, models)
        latency = time.perf_counter() - start
        try:
            json.loads(response.text.strip().strip("`").removeprefix("json").strip())
        except ValueError:
            return ConnectionResult(False, tr("ai.test.not_json", model=self.model, seconds=f"{latency:.2f}"), models, latency)
        return ConnectionResult(
            True, tr("ai.test.success", provider=self.display_name, model=self.model, seconds=f"{latency:.2f}"), models, latency
        )
