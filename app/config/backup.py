"""Export / import of settings and model configurations - **never** including API keys.

A backup contains preferences, the model registry (provider, base URL, model,
priority, capabilities ...) and saved prompts. After restoring on another PC the
user re-enters API keys; on the same PC existing keys of matching model IDs are kept.
"""

from __future__ import annotations

import json
from dataclasses import asdict, fields
from pathlib import Path
from typing import Any

from app.ai.model_config import ModelConfig
from app.ai.model_registry import RegistryError
from app.config.constants import APP_VERSION
from app.config.settings import AppSettings
from app.utils.helpers import now_iso

FORMAT = "ai-video-file-assistant-settings"
_NOT_EXPORTED = {"window_geometry", "last_folder"}


class BackupError(ValueError):
    """The file is not a valid settings backup (message is user-facing)."""


def export_settings(ctx: Any, path: Path) -> dict[str, int]:
    """Write a JSON backup to ``path``. Returns counts for the confirmation message."""
    prefs = {k: v for k, v in asdict(ctx.settings.load()).items() if k not in _NOT_EXPORTED}
    models = [m.to_dict() for m in ctx.registry.list()]
    prompts = [{"name": p.name, "prompt": p.prompt, "provider": p.provider} for p in ctx.db.list_saved_prompts()]
    data = {
        "format": FORMAT, "version": 1, "app_version": APP_VERSION, "exported_at": now_iso(),
        "contains_api_keys": False, "preferences": prefs, "models": models, "saved_prompts": prompts,
    }  # fmt: skip
    text = json.dumps(data, indent=2, ensure_ascii=False)
    for model in ctx.registry.list():  # defence in depth: refuse to write a file that contains a key
        key = ctx.registry.api_key(model.id)
        if key and key in text:
            raise BackupError("Refusing to export: an API key would have been included.")
    path.write_text(text, encoding="utf-8")
    return {"models": len(models), "prompts": len(prompts)}


def import_settings(ctx: Any, path: Path) -> dict[str, int]:
    """Restore preferences, models and prompts from ``path``. Existing models with the same ID are updated."""
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        raise BackupError(f"The file could not be read: {exc}") from exc
    if not isinstance(data, dict) or data.get("format") != FORMAT:
        raise BackupError("This file is not an AI Video File Assistant settings backup.")
    prefs = data.get("preferences") or {}
    if isinstance(prefs, dict):
        current = ctx.settings.load()
        names = {f.name for f in fields(AppSettings)} - _NOT_EXPORTED
        changes = {k: v for k, v in prefs.items() if k in names and isinstance(v, type(getattr(current, k)))}
        ctx.settings.update(**changes)
    added = updated = skipped = 0
    existing = {m.id for m in ctx.registry.list()}
    for raw in data.get("models") or []:
        try:
            model = ModelConfig.from_dict(raw)
            if model.id in existing:
                ctx.registry.update(model)  # key (if any) is kept
                updated += 1
            else:
                ctx.registry.add(model)
                added += 1
        except (RegistryError, TypeError, ValueError, KeyError):
            skipped += 1
    known_prompts = {(p.name, p.prompt) for p in ctx.db.list_saved_prompts()}
    prompts = 0
    for raw in data.get("saved_prompts") or []:
        if isinstance(raw, dict) and raw.get("name") and raw.get("prompt") and (raw["name"], raw["prompt"]) not in known_prompts:
            ctx.db.add_saved_prompt(str(raw["name"]), str(raw["prompt"]), str(raw.get("provider") or "auto"))
            prompts += 1
    return {"added": added, "updated": updated, "skipped": skipped, "prompts": prompts}
