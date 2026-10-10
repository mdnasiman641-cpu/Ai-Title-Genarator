"""Persistent registry of configured AI models (any number, any provider).

* Non-secret settings live in the SQLite ``model_configs`` table.
* Every configuration has its *own* API key, stored encrypted (Windows DPAPI) in the
  credential store under ``model.<id>`` - so two models may share a key, or not.
* Priorities are kept as a dense 1..n ranking; lower numbers are tried first.
"""

from __future__ import annotations

import threading
from collections.abc import Iterable

from app.ai.model_config import PROVIDER_TYPES, ModelConfig, provider_label
from app.config.constants import (
    DEFAULT_GEMINI_MODEL,
    DEFAULT_OPENAI_MODEL,
    ProviderType,
)
from app.database.database import Database
from app.utils.logger import get_logger
from app.utils.security import SecretStore, mask_key

log = get_logger("ai.registry")
LEGACY_SECRETS = {ProviderType.GEMINI.value: "gemini_api_key", ProviderType.OPENAI.value: "openai_api_key"}
LEGACY_MODELS = {ProviderType.GEMINI.value: DEFAULT_GEMINI_MODEL, ProviderType.OPENAI.value: DEFAULT_OPENAI_MODEL}
MIGRATION_FLAG = "registry.migrated_v1"


class RegistryError(ValueError):
    """Invalid configuration (message is user-facing)."""


class ModelRegistry:
    """CRUD + ordering for :class:`ModelConfig`, with per-model encrypted keys."""

    def __init__(self, db: Database, secrets: SecretStore) -> None:
        self._db = db
        self._secrets = secrets
        self._lock = threading.RLock()

    # ------------------------------------------------------------- reading
    def list(self, *, enabled_only: bool = False) -> list[ModelConfig]:
        """All configurations, best priority first."""
        models = []
        for row in self._db.list_model_rows():
            try:
                models.append(ModelConfig.from_dict(row))
            except (TypeError, ValueError, KeyError) as exc:  # a damaged row must not break the app
                log.warning("Skipping unreadable model configuration: %s", exc)
        models.sort(key=lambda m: (m.priority, m.created_at))
        return [m for m in models if m.enabled] if enabled_only else models

    def get(self, model_id: str) -> ModelConfig | None:
        return next((m for m in self.list() if m.id == model_id), None)

    def __len__(self) -> int:
        return len(self.list())

    # ------------------------------------------------------------- writing
    def _store(self, models: Iterable[ModelConfig]) -> list[ModelConfig]:
        """Persist ``models`` in the given order, renumbering priorities 1..n."""
        ordered = [m.with_changes(priority=i) for i, m in enumerate(models, start=1)]
        self._db.replace_model_rows([(m.id, m.priority, m.to_dict()) for m in ordered])
        return ordered

    def add(self, config: ModelConfig, api_key: str | None = None) -> ModelConfig:
        """Save a new configuration (and its key). ``config.priority`` decides where it goes."""
        problems = config.validate()
        if problems:
            raise RegistryError(problems[0])
        with self._lock:
            models = self.list()
            if any(m.id == config.id for m in models):
                raise RegistryError("A model with this ID already exists.")
            index = max(0, min(config.priority - 1, len(models))) if config.priority else len(models)
            models.insert(index, config)
            stored = self._store(models)
            if api_key:
                self.set_key(config.id, api_key)
        log.info("Model added: %s (%s)", config.label, config.id[:8])
        return next(m for m in stored if m.id == config.id)

    def update(self, config: ModelConfig, api_key: str | None = None) -> ModelConfig:
        """Save changes. ``api_key=None`` keeps the existing key (never wiped by an empty field)."""
        problems = config.validate()
        if problems:
            raise RegistryError(problems[0])
        with self._lock:
            models = [m for m in self.list() if m.id != config.id]
            if len(models) == len(self.list()):
                raise RegistryError("This model no longer exists.")
            index = max(0, min(config.priority - 1, len(models)))
            models.insert(index, config)
            stored = self._store(models)
            if api_key:
                self.set_key(config.id, api_key)
        return next(m for m in stored if m.id == config.id)

    def remove(self, model_id: str) -> None:
        """Delete a configuration together with its stored key."""
        with self._lock:
            self._store([m for m in self.list() if m.id != model_id])
            self._secrets.delete(self._secret_name(model_id))
            self._db.delete_health_row(model_id)

    def set_enabled(self, model_id: str, enabled: bool) -> None:
        with self._lock:
            self._store([m.with_changes(enabled=enabled) if m.id == model_id else m for m in self.list()])

    def move(self, model_id: str, delta: int) -> None:
        """Move a model up (``delta<0``) or down in priority."""
        with self._lock:
            models = self.list()
            index = next((i for i, m in enumerate(models) if m.id == model_id), None)
            if index is None:
                return
            target = max(0, min(len(models) - 1, index + delta))
            models.insert(target, models.pop(index))
            self._store(models)

    def reorder(self, ids: list[str]) -> None:
        """Set the full priority order (e.g. after drag & drop); unknown ids are ignored."""
        with self._lock:
            by_id = {m.id: m for m in self.list()}
            ordered = [by_id.pop(i) for i in ids if i in by_id]
            self._store(ordered + list(by_id.values()))

    # ----------------------------------------------------------- API keys
    @staticmethod
    def _secret_name(model_id: str) -> str:
        return f"model.{model_id}"

    def set_key(self, model_id: str, api_key: str) -> None:
        self._secrets.set(self._secret_name(model_id), api_key)

    def api_key(self, model_id: str) -> str | None:
        """Decrypted key - call only right before a request."""
        return self._secrets.get(self._secret_name(model_id))

    def has_key(self, model_id: str) -> bool:
        return self._secrets.has(self._secret_name(model_id))

    def masked_key(self, model_id: str) -> str:
        return mask_key(self.api_key(model_id))

    def remove_key(self, model_id: str) -> None:
        """Forget the key only; the model configuration stays."""
        self._secrets.delete(self._secret_name(model_id))

    def is_usable(self, config: ModelConfig) -> bool:
        """Enabled and either has a key or does not need one."""
        return config.enabled and (self.has_key(config.id) or not PROVIDER_TYPES[config.provider_type].key_required)

    # ---------------------------------------------------------- migration
    def migrate_legacy(self) -> list[ModelConfig]:
        """Import the pre-registry single Gemini / OpenAI keys and models (runs once).

        Keys are moved (re-encrypted under the new per-model name), so users never
        have to type them again; the old entries are removed afterwards.
        """
        if self._db.get_raw(MIGRATION_FLAG) == "1":
            return []
        created: list[ModelConfig] = []
        default = self._db.get_raw("pref.default_provider") or ""
        preferred = self._db.get_raw("pref.auto_preferred_provider") or ProviderType.GEMINI.value
        first = default if default in LEGACY_SECRETS else preferred
        order = sorted(LEGACY_SECRETS, key=lambda p: 0 if p == first else 1)
        timeout = self._db.get_raw("pref.request_timeout_s")
        for provider in order:
            key = self._secrets.get(LEGACY_SECRETS[provider])
            if not key:
                continue
            model = (self._db.get_raw(f"pref.{provider}_model") or LEGACY_MODELS[provider]).strip()
            config = ModelConfig.new(
                provider, model, display_name=f"{provider_label(provider)} {model}",
                timeout_s=int(timeout) if timeout and timeout.isdigit() else 60,
                priority=len(self.list()) + 1,
            )  # fmt: skip
            try:
                created.append(self.add(config, key))
            except RegistryError as exc:
                log.warning("Could not migrate the old %s configuration: %s", provider, exc)
                continue
            self._secrets.delete(LEGACY_SECRETS[provider])
        fallback = self._db.get_raw("pref.fallback_mode")
        if fallback == "off" and self._db.get_raw("pref.routing_strategy") is None:
            self._db.set_raw("pref.routing_strategy", "auto")
        self._db.set_raw(MIGRATION_FLAG, "1")
        if created:
            log.info("Migrated %d model configuration(s) from the previous version", len(created))
        return created
