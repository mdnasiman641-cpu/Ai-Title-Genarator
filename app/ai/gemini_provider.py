"""Google Gemini adapter (REST ``generateContent``).

The API key travels in the ``x-goog-api-key`` header - never in the URL - so it
cannot leak through logs or proxies that record request lines. The base URL is
configurable (default: the official endpoint).
"""

from __future__ import annotations

from typing import Any
from urllib.parse import quote

from app.ai.base_provider import AIProvider, AIRequest, AIResponse
from app.ai.errors import AIError, ErrorKind
from app.ai.http_client import request_json
from app.config.constants import GEMINI_API_BASE, ProviderType
from app.utils.logger import get_logger

log = get_logger("ai.gemini")


class GeminiProvider(AIProvider):
    """Calls a Gemini endpoint; the model name and base URL come from the configuration."""

    id = ProviderType.GEMINI.value
    display_name = "Gemini"
    default_base_url = GEMINI_API_BASE

    def _headers(self) -> dict[str, str]:
        return {"x-goog-api-key": self._api_key}

    def generate(self, request: AIRequest) -> AIResponse:
        url = f"{self.base_url}/models/{quote(self.model, safe='/')}:generateContent"
        payload: dict[str, Any] = {
            "systemInstruction": {"parts": [{"text": request.system_prompt}]},
            "contents": [{"role": "user", "parts": [{"text": request.user_prompt}]}],
            "generationConfig": {"responseMimeType": "application/json"},
        }
        log.debug("Gemini request (model=%s, %d chars)", self.model, len(request.user_prompt))
        data = request_json(self.error_name, "POST", url, headers=self._headers(), payload=payload, timeout=self.timeout)
        return self._parse(data)

    def _parse(self, data: dict[str, Any]) -> AIResponse:
        candidates = data.get("candidates") or []
        if not candidates:
            block = (data.get("promptFeedback") or {}).get("blockReason")
            if block:
                raise AIError(ErrorKind.BLOCKED, self.error_name, f"Request was blocked by Gemini ({block}).")
            raise AIError(ErrorKind.INVALID_RESPONSE, self.error_name, "Gemini returned no answer.")
        candidate = candidates[0]
        parts = (candidate.get("content") or {}).get("parts") or []
        text = "".join(str(p.get("text", "")) for p in parts if isinstance(p, dict) and not p.get("thought"))
        finish = str(candidate.get("finishReason", ""))
        if not text.strip():
            if finish in {"SAFETY", "PROHIBITED_CONTENT", "BLOCKLIST", "RECITATION"}:
                raise AIError(ErrorKind.BLOCKED, self.error_name, f"Gemini declined to answer ({finish}).")
            raise AIError(ErrorKind.INVALID_RESPONSE, self.error_name, "Gemini returned an empty answer.")
        usage = data.get("usageMetadata") or {}
        return AIResponse(
            text=text,
            provider=self.id,
            model=self.model,
            truncated=finish == "MAX_TOKENS",
            usage={
                "input_tokens": int(usage.get("promptTokenCount", 0) or 0),
                "output_tokens": int(usage.get("candidatesTokenCount", 0) or 0),
            },
        )

    def list_models(self) -> list[str]:
        url = f"{self.base_url}/models?pageSize=200"
        try:
            data = request_json(self.error_name, "GET", url, headers=self._headers(), timeout=min(self.timeout, 30))
        except AIError as exc:
            if exc.status in (404, 405):
                raise AIError(ErrorKind.LISTING_UNSUPPORTED, self.error_name, exc.detail, exc.status) from None
            raise
        names: list[str] = []
        for item in data.get("models") or []:
            if not isinstance(item, dict):
                continue
            methods = item.get("supportedGenerationMethods") or []
            if methods and "generateContent" not in methods:
                continue
            name = str(item.get("name", "")).removeprefix("models/")
            if name:
                names.append(name)
        return sorted(names)
