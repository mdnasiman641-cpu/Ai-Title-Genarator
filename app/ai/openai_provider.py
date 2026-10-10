"""OpenAI and OpenAI-compatible adapters (``/chat/completions`` + ``/models``).

* :class:`OpenAIProvider` - api.openai.com (or a proxy with the same API).
* :class:`OpenAICompatibleProvider` - any server that speaks the same protocol
  (vLLM, LM Studio, Ollama, OpenRouter, Together, Groq ...). Such servers differ in
  details, so the adapter is defensive: JSON mode is retried without
  ``response_format`` if the server rejects it, the key is optional, and a missing
  ``/models`` endpoint is reported as "listing not supported" instead of an error.
* :class:`CustomProvider` - the same chat format with no JSON mode and no model
  listing, for minimal implementations.
"""

from __future__ import annotations

import re
from typing import Any

from app.ai.base_provider import AIProvider, AIRequest, AIResponse
from app.ai.errors import AIError, ErrorKind
from app.ai.http_client import request_json
from app.config.constants import OPENAI_API_BASE, ProviderType
from app.utils.logger import get_logger

log = get_logger("ai.openai")

_CHAT_MODEL = re.compile(r"^(gpt-|chatgpt-|o\d)")
_NOT_CHAT = re.compile(r"(audio|realtime|transcribe|tts|image|embedding|moderation|search|instruct|whisper|dall)")


class OpenAIProvider(AIProvider):
    """Calls the OpenAI Chat Completions API."""

    id = ProviderType.OPENAI.value
    display_name = "OpenAI"
    default_base_url = OPENAI_API_BASE
    json_mode = True  # send response_format={"type": "json_object"}
    filter_chat_models = True  # hide embeddings/audio/... in "Load Models"

    def _headers(self) -> dict[str, str]:
        return {"Authorization": f"Bearer {self._api_key}"} if self._api_key else {}

    def _payload(self, request: AIRequest, json_mode: bool) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": request.system_prompt},
                {"role": "user", "content": request.user_prompt},
            ],
        }
        if json_mode:
            payload["response_format"] = {"type": "json_object"}
        return payload

    def generate(self, request: AIRequest) -> AIResponse:
        log.debug("%s request (model=%s, %d chars)", self.display_name, self.model, len(request.user_prompt))
        url = f"{self.base_url}/chat/completions"
        try:
            data = request_json(
                self.error_name, "POST", url, headers=self._headers(), payload=self._payload(request, self.json_mode),
                timeout=self.timeout,
            )  # fmt: skip
        except AIError as exc:
            if not (self.json_mode and exc.kind == ErrorKind.BAD_REQUEST and "response_format" in exc.detail.lower()):
                raise
            log.info("%s rejected JSON mode; retrying without response_format", self.error_name)
            data = request_json(
                self.error_name, "POST", url, headers=self._headers(), payload=self._payload(request, False),
                timeout=self.timeout,
            )  # fmt: skip
        return self._parse(data)

    def _parse(self, data: dict[str, Any]) -> AIResponse:
        choices = data.get("choices") or []
        if not choices or not isinstance(choices[0], dict):
            raise AIError(ErrorKind.INVALID_RESPONSE, self.error_name, "The server returned no answer.")
        choice = choices[0]
        message = choice.get("message") or {}
        if message.get("refusal"):
            raise AIError(ErrorKind.BLOCKED, self.error_name, f"The model declined: {message['refusal']}")
        content = message.get("content")
        if isinstance(content, list):  # some compatible servers return content parts
            content = "".join(str(part.get("text", "")) for part in content if isinstance(part, dict))
        text = str(content or "")
        if not text.strip():
            raise AIError(ErrorKind.INVALID_RESPONSE, self.error_name, "The server returned an empty answer.")
        usage = data.get("usage") or {}
        return AIResponse(
            text=text,
            provider=self.id,
            model=self.model,
            truncated=choice.get("finish_reason") == "length",
            usage={
                "input_tokens": int(usage.get("prompt_tokens", 0) or 0),
                "output_tokens": int(usage.get("completion_tokens", 0) or 0),
            },
        )

    def list_models(self) -> list[str]:
        try:
            data = request_json(
                self.error_name, "GET", f"{self.base_url}/models", headers=self._headers(), timeout=min(self.timeout, 30)
            )
        except AIError as exc:
            if exc.status in (404, 405, 501):
                raise AIError(ErrorKind.LISTING_UNSUPPORTED, self.error_name, exc.detail, exc.status) from None
            raise
        items = data.get("data")
        if not isinstance(items, list):
            items = data.get("models") if isinstance(data.get("models"), list) else None
        if items is None:
            raise AIError(ErrorKind.LISTING_UNSUPPORTED, self.error_name, "Unexpected model-list format.")
        ids = sorted({str(m.get("id") or m.get("name") or "") for m in items if isinstance(m, dict)} - {""})
        if self.filter_chat_models:
            ids = [i for i in ids if _CHAT_MODEL.match(i) and not _NOT_CHAT.search(i)]
        return ids


class OpenAICompatibleProvider(OpenAIProvider):
    """Any server with an OpenAI-style API at a user-supplied base URL."""

    id = ProviderType.OPENAI_COMPATIBLE.value
    display_name = "OpenAI-compatible"
    default_base_url = ""
    key_required = False
    filter_chat_models = False  # model names are vendor specific - show them all


class CustomProvider(OpenAICompatibleProvider):
    """Minimal OpenAI-style chat endpoint: no JSON mode, no model listing."""

    id = ProviderType.CUSTOM.value
    display_name = "Custom"
    json_mode = False
    supports_listing = False

    def list_models(self) -> list[str]:
        raise AIError(ErrorKind.LISTING_UNSUPPORTED, self.error_name)
