"""Provider-type -> adapter class mapping and construction from a model configuration."""

from __future__ import annotations

from app.ai.base_provider import AIProvider
from app.ai.gemini_provider import GeminiProvider
from app.ai.model_config import ModelConfig
from app.ai.openai_provider import (
    CustomProvider,
    OpenAICompatibleProvider,
    OpenAIProvider,
)
from app.config.constants import ProviderType

ADAPTERS: dict[str, type[AIProvider]] = {
    ProviderType.GEMINI.value: GeminiProvider,
    ProviderType.OPENAI.value: OpenAIProvider,
    ProviderType.OPENAI_COMPATIBLE.value: OpenAICompatibleProvider,
    ProviderType.CUSTOM.value: CustomProvider,
}


def build_provider(config: ModelConfig, api_key: str | None) -> AIProvider:
    """Instantiate the right adapter for ``config`` (raises :class:`AIError` e.g. if a key is missing)."""
    cls = ADAPTERS.get(config.provider_type, CustomProvider)
    return cls(api_key, config.model_name, config.timeout_s, config.base_url, config.label)


def build_for_listing(provider_type: str, base_url: str, api_key: str | None, timeout: float = 30) -> AIProvider:
    """An adapter used only to call *Load Models* (no model chosen yet)."""
    cls = ADAPTERS.get(provider_type, CustomProvider)
    return cls(api_key, "-", timeout, base_url, cls.display_name)
