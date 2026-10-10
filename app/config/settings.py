"""Typed application settings persisted in SQLite.

``AppSettings`` is a plain dataclass; :class:`SettingsManager` loads/saves it and
owns the encrypted API keys. The set of valid values for choice fields is kept
here so the UI and the rest of the app share one definition.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, fields, replace
from typing import TYPE_CHECKING, Any

from app.config.constants import (
    DEFAULT_BATCH_SIZE,
    DEFAULT_COOLDOWN_AFTER_FAILURES,
    DEFAULT_COOLDOWN_S,
    DEFAULT_MAX_FALLBACK_ATTEMPTS,
    DEFAULT_MAX_REQUEST_CHARS,
    DEFAULT_MAX_RETRIES,
    DEFAULT_REQUEST_TIMEOUT_S,
    DEFAULT_SAMPLE_SIZE,
    GOOGLE_SEARCH_ENDPOINT,
    MAX_BATCH_SIZE,
    RoutingStrategy,
)

if TYPE_CHECKING:
    from app.database.database import Database
    from app.utils.security import SecretStore

THEMES = ("system", "dark", "light")
LANGUAGES = ("en", "bn")
LOG_LEVELS = ("DEBUG", "INFO", "WARNING", "ERROR")
STRATEGIES = tuple(s.value for s in RoutingStrategy)
DUPLICATE_POLICIES = ("flag", "number")
DELETE_MODES = ("trash", "permanent")



@dataclass(slots=True)
class AppSettings:
    """All user-configurable options with their defaults."""

    # --- AI routing (the models themselves live in the Model Registry)
    routing_strategy: str = RoutingStrategy.AUTO_FALLBACK.value
    preferred_model_id: str = ""  # "" = automatic
    cost_aware: bool = False
    max_fallback_attempts: int = DEFAULT_MAX_FALLBACK_ATTEMPTS
    cooldown_seconds: int = DEFAULT_COOLDOWN_S
    cooldown_after_failures: int = DEFAULT_COOLDOWN_AFTER_FAILURES
    health_tracking: bool = True
    request_timeout_s: int = DEFAULT_REQUEST_TIMEOUT_S  # default for new model configurations
    default_max_retries: int = DEFAULT_MAX_RETRIES  # default for new model configurations
    max_request_chars: int = DEFAULT_MAX_REQUEST_CHARS
    batch_size: int = DEFAULT_BATCH_SIZE
    sample_size: int = DEFAULT_SAMPLE_SIZE
    use_offline_parser: bool = True
    send_metadata_to_ai: bool = False
    # --- Google Search (Title Generator context; the API key is in the encrypted store)
    search_engine_id: str = ""
    search_endpoint: str = GOOGLE_SEARCH_ENDPOINT
    search_cache_hours: int = 24
    # --- File operations
    ask_before_apply: bool = True
    create_history: bool = True
    prevent_overwrite: bool = True
    include_subfolders: bool = False
    auto_apply: bool = False
    duplicate_policy: str = "flag"
    delete_mode: str = "trash"
    # --- Appearance / general
    theme: str = "system"
    language: str = "en"
    reopen_last_folder: bool = False
    last_folder: str = ""
    log_level: str = "INFO"
    read_metadata: bool = True
    ffmpeg_dir: str = ""
    default_file_filter: str = "video"
    # --- UI state
    window_geometry: str = ""

    def normalised(self) -> AppSettings:
        """Return a copy with every field coerced into its valid range."""
        out = replace(self)
        out.routing_strategy = _choice(out.routing_strategy, STRATEGIES, RoutingStrategy.AUTO_FALLBACK.value)
        out.max_fallback_attempts = max(1, min(int(out.max_fallback_attempts), 10))
        out.cooldown_seconds = max(10, min(int(out.cooldown_seconds), 3600))
        out.cooldown_after_failures = max(1, min(int(out.cooldown_after_failures), 10))
        out.default_max_retries = max(0, min(int(out.default_max_retries), 5))
        out.max_request_chars = max(2_000, min(int(out.max_request_chars), 500_000))
        out.duplicate_policy = _choice(out.duplicate_policy, DUPLICATE_POLICIES, "flag")
        out.delete_mode = _choice(out.delete_mode, DELETE_MODES, "trash")
        out.theme = _choice(out.theme, THEMES, "system")
        out.language = _choice(out.language, LANGUAGES, "en")
        out.log_level = _choice(str(out.log_level).upper(), LOG_LEVELS, "INFO")
        out.batch_size = max(1, min(int(out.batch_size), MAX_BATCH_SIZE))
        out.sample_size = max(1, min(int(out.sample_size), MAX_BATCH_SIZE))
        out.request_timeout_s = max(5, min(int(out.request_timeout_s), 600))
        out.search_cache_hours = max(0, min(int(out.search_cache_hours), 720))
        out.search_endpoint = out.search_endpoint.strip() or GOOGLE_SEARCH_ENDPOINT
        out.search_engine_id = out.search_engine_id.strip()[:120]
        return out



def _choice(value: str, allowed: tuple[str, ...] | list[str], default: str) -> str:
    return value if value in allowed else default


def _decode(raw: str, default: Any) -> Any:
    """Convert a stored string back into the type of ``default``."""
    if isinstance(default, bool):
        return raw.lower() in {"1", "true", "yes", "on"}
    if isinstance(default, int):
        try:
            return int(raw)
        except ValueError:
            return default
    return raw


def _encode(value: Any) -> str:
    if isinstance(value, bool):
        return "1" if value else "0"
    return str(value)


class SettingsManager:
    """Loads and saves :class:`AppSettings`; also fronts the encrypted API-key store."""

    def __init__(self, db: Database, secrets: SecretStore) -> None:
        self._db = db
        self._secrets = secrets
        self._cache: AppSettings | None = None

    # ------------------------------------------------------------- settings
    def load(self) -> AppSettings:
        """Return the current settings (cached; call :meth:`reload` to re-read)."""
        if self._cache is None:
            stored = self._db.all_settings()
            defaults = AppSettings()
            values: dict[str, Any] = {}
            for f in fields(AppSettings):
                default = getattr(defaults, f.name)
                key = f"pref.{f.name}"
                values[f.name] = _decode(stored[key], default) if key in stored else default
            self._cache = AppSettings(**values).normalised()
        return replace(self._cache)

    def reload(self) -> AppSettings:
        self._cache = None
        return self.load()

    def save(self, settings: AppSettings) -> AppSettings:
        """Validate and persist ``settings``; returns the normalised copy."""
        clean = settings.normalised()
        with self._db.transaction() as conn:
            for f in fields(AppSettings):
                conn.execute(
                    "INSERT INTO settings(key, value) VALUES(?, ?) "
                    "ON CONFLICT(key) DO UPDATE SET value = excluded.value",
                    (f"pref.{f.name}", _encode(getattr(clean, f.name))),
                )
        self._cache = clean
        return replace(clean)

    def update(self, **changes: Any) -> AppSettings:
        """Convenience: change a few fields and save."""
        return self.save(replace(self.load(), **changes))

    @property
    def secrets(self) -> SecretStore:
        """The encrypted credential store (API keys live in the Model Registry, one per model)."""
        return self._secrets

    # --------------------------------------------------------------- export
    def export_json(self) -> str:
        """Settings (without secrets) as JSON - handy for support requests."""
        data = {f.name: getattr(self.load(), f.name) for f in fields(AppSettings)}
        data.pop("window_geometry", None)
        return json.dumps(data, indent=2, ensure_ascii=False)
