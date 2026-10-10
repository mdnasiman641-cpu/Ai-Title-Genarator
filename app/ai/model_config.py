"""One configured AI model ("model configuration") and provider-type metadata.

A configuration is identified by a random ``id`` (never by its display name) and
holds everything needed to reach a model *except* the API key, which lives in the
encrypted credential store under ``model.<id>``.
"""

from __future__ import annotations

import ipaddress
import re
import uuid
from dataclasses import asdict, dataclass, field, replace
from typing import Any
from urllib.parse import urlparse

from app.config.constants import (
    DEFAULT_MAX_RETRIES,
    DEFAULT_REQUEST_TIMEOUT_S,
    GEMINI_API_BASE,
    OPENAI_API_BASE,
    Capability,
    ProviderType,
)
from app.utils.helpers import now_iso

DEFAULT_CAPABILITIES = (Capability.TEXT.value, Capability.JSON.value)
_MODEL_NAME = re.compile(r"^[^\s\x00-\x1f]{1,200}$")


@dataclass(frozen=True, slots=True)
class ProviderInfo:
    """Static facts about a provider type."""

    label: str
    default_base_url: str
    key_required: bool  # local OpenAI-compatible servers often need no key
    supports_listing: bool  # has a model-list endpoint we know how to read


PROVIDER_TYPES: dict[str, ProviderInfo] = {
    ProviderType.GEMINI.value: ProviderInfo("Gemini", GEMINI_API_BASE, True, True),
    ProviderType.OPENAI.value: ProviderInfo("OpenAI", OPENAI_API_BASE, True, True),
    ProviderType.OPENAI_COMPATIBLE.value: ProviderInfo("OpenAI-compatible", "", False, True),
    ProviderType.CUSTOM.value: ProviderInfo("Custom", "", False, False),
}


def provider_label(provider_type: str) -> str:
    info = PROVIDER_TYPES.get(provider_type)
    return info.label if info else provider_type


def normalize_base_url(url: str) -> str:
    """Trim whitespace and trailing slashes (``https://x/v1/`` -> ``https://x/v1``)."""
    return url.strip().rstrip("/")


def _is_local(host: str) -> bool:
    if host in {"localhost"}:
        return True
    try:
        address = ipaddress.ip_address(host)
    except ValueError:
        return host.endswith(".local")
    return address.is_loopback or address.is_private


def validate_base_url(url: str) -> str | None:
    """Return an error message for an unusable base URL, or ``None`` if it is fine.

    HTTPS is required, except for local/private addresses (e.g. a model server on this PC
    or the LAN), where plain HTTP is allowed. Credentials inside the URL are refused so a
    key can never end up in logs.
    """
    url = normalize_base_url(url)
    if not url:
        return "The API Base URL is empty."
    parsed = urlparse(url)
    if parsed.scheme not in ("http", "https") or not parsed.hostname:
        return "The API Base URL must start with https:// (or http:// for a local server)."
    if parsed.username or parsed.password:
        return "Do not put credentials in the URL - use the API key field."
    if parsed.query or parsed.fragment:
        return "The API Base URL must not contain ? or # parts."
    if parsed.scheme == "http" and not _is_local(parsed.hostname):
        return "Use https:// for remote servers (plain http:// is only allowed for local servers)."
    return None


@dataclass(frozen=True, slots=True)
class ModelConfig:
    """A model the router may use. Immutable: use :meth:`with_changes` to edit."""

    id: str
    provider_type: str
    display_name: str
    base_url: str
    model_name: str
    enabled: bool = True
    priority: int = 0  # position in the list (1 = first); 0 = append at the end when added
    capabilities: tuple[str, ...] = DEFAULT_CAPABILITIES
    timeout_s: int = DEFAULT_REQUEST_TIMEOUT_S
    max_retries: int = DEFAULT_MAX_RETRIES
    rpm_limit: int | None = None  # requests per minute (user supplied, optional)
    rpd_limit: int | None = None  # requests per day (user supplied, optional)
    cost_input_per_mtok: float | None = None  # price per 1M input tokens - None = unknown (never invented)
    cost_output_per_mtok: float | None = None
    created_at: str = field(default_factory=now_iso)

    # ------------------------------------------------------------- factory
    @classmethod
    def new(
        cls,
        provider_type: str,
        model_name: str,
        *,
        display_name: str = "",
        base_url: str = "",
        **kwargs: Any,
    ) -> ModelConfig:
        info = PROVIDER_TYPES.get(provider_type)
        base = normalize_base_url(base_url) or (info.default_base_url if info else "")
        name = display_name.strip() or f"{provider_label(provider_type)} {model_name.strip()}"
        return cls(
            id=uuid.uuid4().hex, provider_type=provider_type, display_name=name, base_url=base,
            model_name=model_name.strip(), **kwargs,
        )  # fmt: skip

    def with_changes(self, **changes: Any) -> ModelConfig:
        return replace(self, **changes)

    # ------------------------------------------------------------ queries
    @property
    def provider_label(self) -> str:
        return provider_label(self.provider_type)

    @property
    def label(self) -> str:
        """``Gemini / gemini-2.5-flash`` - how a model is shown in status texts."""
        return f"{self.provider_label} / {self.model_name}"

    @property
    def info(self) -> ProviderInfo:
        return PROVIDER_TYPES.get(self.provider_type, PROVIDER_TYPES[ProviderType.CUSTOM.value])

    def has(self, capability: str | Capability) -> bool:
        value = capability.value if isinstance(capability, Capability) else capability
        return value in self.capabilities

    @property
    def known_cost(self) -> float | None:
        """Comparable price (input + output per 1M tokens), or ``None`` if the user gave none."""
        if self.cost_input_per_mtok is None and self.cost_output_per_mtok is None:
            return None
        return (self.cost_input_per_mtok or 0.0) + (self.cost_output_per_mtok or 0.0)

    def validate(self) -> list[str]:
        """Problems that prevent saving this configuration."""
        problems: list[str] = []
        if self.provider_type not in PROVIDER_TYPES:
            problems.append(f"Unknown provider type “{self.provider_type}”.")
        if not self.display_name.strip():
            problems.append("The display name is empty.")
        if not _MODEL_NAME.match(self.model_name or ""):
            problems.append("Enter a model ID without spaces.")
        url_problem = validate_base_url(self.base_url)
        if url_problem:
            problems.append(url_problem)
        if not 1 <= self.timeout_s <= 600:
            problems.append("The timeout must be between 1 and 600 seconds.")
        if not 0 <= self.max_retries <= 5:
            problems.append("Max retries must be between 0 and 5.")
        for limit in (self.rpm_limit, self.rpd_limit):
            if limit is not None and limit < 1:
                problems.append("Rate limits must be positive numbers (or empty).")
        for cost in (self.cost_input_per_mtok, self.cost_output_per_mtok):
            if cost is not None and cost < 0:
                problems.append("Prices cannot be negative.")
        unknown = [c for c in self.capabilities if c not in {x.value for x in Capability}]
        if unknown:
            problems.append(f"Unknown capability: {', '.join(unknown)}")
        return problems

    # -------------------------------------------------------- persistence
    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["capabilities"] = list(self.capabilities)
        return data

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> ModelConfig:
        """Tolerant loader (ignores unknown keys, fills defaults) used for the DB and backups."""
        known = {f for f in cls.__dataclass_fields__}
        values = {k: v for k, v in data.items() if k in known}
        values["capabilities"] = tuple(str(c) for c in values.get("capabilities") or DEFAULT_CAPABILITIES)
        values.setdefault("id", uuid.uuid4().hex)
        values["base_url"] = normalize_base_url(str(values.get("base_url", "")))
        for key in ("timeout_s", "max_retries", "priority"):
            if key in values:
                values[key] = int(values[key])
        for key in ("rpm_limit", "rpd_limit"):
            if values.get(key) in ("", 0):
                values[key] = None
        for key in ("cost_input_per_mtok", "cost_output_per_mtok"):
            if values.get(key) in ("",):
                values[key] = None
            elif values.get(key) is not None:
                values[key] = float(values[key])
        values["enabled"] = bool(values.get("enabled", True))
        return cls(**values)
