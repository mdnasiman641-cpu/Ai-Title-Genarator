"""Application context: the shared services every window needs."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from app.ai.model_health import HealthTracker
from app.ai.model_registry import ModelRegistry
from app.config.settings import SettingsManager
from app.database.database import Database
from app.i18n import tr
from app.utils.helpers import get_app_data_dir
from app.utils.security import SecretStore, default_backend

DEFAULT_PROMPTS = (
    ("prompt.default.blue_bloods", "prompt.default.blue_bloods.text"),
    ("prompt.default.clean", "prompt.default.clean.text"),
    ("prompt.default.numbering", "prompt.default.numbering.text"),
    ("prompt.default.quality", "prompt.default.quality.text"),
    ("prompt.default.youtube", "prompt.default.youtube.text"),
)


@dataclass(slots=True)
class AppContext:
    """Database + settings + model registry + health, created once at start-up.

    Start-up order: settings -> model registry (with one-time migration of the old
    single Gemini/OpenAI configuration) -> health statistics -> saved prompts.
    """

    data_dir: Path
    db: Database
    settings: SettingsManager
    registry: ModelRegistry
    health: HealthTracker

    @classmethod
    def create(cls, data_dir: Path | None = None) -> AppContext:
        data_dir = data_dir or get_app_data_dir()
        data_dir.mkdir(parents=True, exist_ok=True)
        db = Database(data_dir / "app.db")
        secrets = SecretStore(db, default_backend(data_dir))
        settings = SettingsManager(db, secrets)
        registry = ModelRegistry(db, secrets)
        registry.migrate_legacy()  # old Gemini/OpenAI keys + models -> registry (no re-entry needed)
        current = settings.reload()
        health = HealthTracker(
            db, cooldown_s=current.cooldown_seconds, cooldown_after_failures=current.cooldown_after_failures,
            enabled=current.health_tracking,
        )  # fmt: skip
        ctx = cls(data_dir=data_dir, db=db, settings=settings, registry=registry, health=health)
        ctx.seed_default_prompts()
        interrupted = db.mark_interrupted_operations()
        if interrupted:
            from app.utils.logger import get_logger

            get_logger("app").warning("%d operation(s) were interrupted by a previous crash", interrupted)
        return ctx

    def seed_default_prompts(self) -> None:
        """Create the built-in saved prompts the first time the app runs."""
        if self.db.get_raw("seeded_prompts") == "1":
            return
        if self.db.count_saved_prompts() == 0:
            for name_key, text_key in DEFAULT_PROMPTS:
                self.db.add_saved_prompt(tr(name_key), tr(text_key), "auto")
        self.db.set_raw("seeded_prompts", "1")
