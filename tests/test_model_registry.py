"""Model registry: CRUD, ordering, per-model encrypted keys, persistence, migration, backup."""

from __future__ import annotations

import json

import pytest

from app.ai.model_config import ModelConfig, validate_base_url
from app.ai.model_registry import RegistryError
from app.config.backup import BackupError, export_settings, import_settings
from app.context import AppContext
from app.database.database import Database
from app.utils.security import SecretStore, default_backend

GEMINI_KEY = "AIzaSyTESTKEY-0000000000-abcdefghWXYZ"
OPENAI_KEY = "sk-test-0000000000000000000000ABCD"


@pytest.fixture
def ctx(tmp_path):
    return AppContext.create(tmp_path / "data")


def local(name: str, **kw) -> ModelConfig:
    return ModelConfig.new("openai_compatible", name, base_url="http://localhost:11434/v1", **kw)


# ------------------------------------------------------------------- CRUD
def test_add_list_update_remove(ctx):
    assert len(ctx.registry) == 0
    gem = ctx.registry.add(ModelConfig.new("gemini", "gemini-2.5-flash"), GEMINI_KEY)
    oai = ctx.registry.add(ModelConfig.new("openai", "gpt-4o-mini"), OPENAI_KEY)
    assert [m.id for m in ctx.registry.list()] == [gem.id, oai.id]
    assert [m.priority for m in ctx.registry.list()] == [1, 2]
    assert gem.display_name == "Gemini gemini-2.5-flash" and gem.label == "Gemini / gemini-2.5-flash"
    assert gem.base_url == "https://generativelanguage.googleapis.com/v1beta"
    assert len(gem.id) == 32 and gem.id != oai.id  # unique UUIDs

    edited = ctx.registry.update(gem.with_changes(display_name="Fast", timeout_s=20))
    assert ctx.registry.get(gem.id).display_name == "Fast" and edited.timeout_s == 20
    ctx.registry.remove(oai.id)
    assert [m.id for m in ctx.registry.list()] == [gem.id]
    assert ctx.registry.api_key(oai.id) is None  # key removed with the model


def test_unlimited_models_and_several_keys_per_provider(ctx):
    ids = [ctx.registry.add(ModelConfig.new("openai", f"gpt-{i}"), f"sk-key-number-{i:04d}-abcdef").id for i in range(25)]
    assert len(ctx.registry) == 25
    assert {ctx.registry.api_key(i) for i in ids} == {f"sk-key-number-{i:04d}-abcdef" for i in range(25)}


def test_validation_rejects_bad_configs(ctx):
    with pytest.raises(RegistryError):
        ctx.registry.add(ModelConfig.new("gemini", "has space"))
    with pytest.raises(RegistryError):
        ctx.registry.add(ModelConfig.new("openai_compatible", "m", base_url=""))  # no URL
    with pytest.raises(RegistryError):
        ctx.registry.add(ModelConfig.new("nonsense", "m", base_url="https://x.example"))
    with pytest.raises(RegistryError):
        ctx.registry.add(local("m", timeout_s=0))
    with pytest.raises(RegistryError):
        ctx.registry.add(local("m", cost_input_per_mtok=-1.0))
    assert len(ctx.registry) == 0


@pytest.mark.parametrize(
    ("url", "ok"),
    [
        ("https://api.openai.com/v1", True),
        ("https://openrouter.ai/api/v1", True),
        ("http://localhost:11434/v1", True),  # local servers may use http
        ("http://127.0.0.1:1234/v1", True),
        ("http://192.168.1.20:8000/v1", True),
        ("http://api.example.com/v1", False),  # plain http to a public host would leak the key
        ("ftp://example.com", False),
        ("https://user:pass@example.com/v1", False),  # no credentials in URLs
        ("https://example.com/v1?key=abc", False),  # no keys in query strings
        ("", False),
        ("not a url", False),
    ],
)
def test_base_url_validation(url, ok):
    assert (validate_base_url(url) is None) == ok


def test_ordering_move_reorder_enable(ctx):
    a, b, c = (ctx.registry.add(local(n)) for n in "abc")
    ctx.registry.move(c.id, -1)
    assert [m.model_name for m in ctx.registry.list()] == ["a", "c", "b"]
    ctx.registry.move(a.id, -10)  # clamped
    assert ctx.registry.list()[0].id == a.id
    ctx.registry.reorder([b.id, a.id])
    assert [m.model_name for m in ctx.registry.list()] == ["b", "a", "c"]
    ctx.registry.set_enabled(b.id, False)
    assert [m.model_name for m in ctx.registry.list(enabled_only=True)] == ["a", "c"]
    first = ctx.registry.add(local("first", priority=1))
    assert ctx.registry.list()[0].id == first.id  # explicit priority inserts at that position
    ctx.registry.update(first.with_changes(priority=3))
    assert [m.model_name for m in ctx.registry.list()] == ["b", "a", "first", "c"]


# -------------------------------------------------------------------- keys
def test_keys_are_encrypted_and_masked(ctx):
    model = ctx.registry.add(ModelConfig.new("gemini", "g"), GEMINI_KEY)
    assert ctx.registry.has_key(model.id) and ctx.registry.api_key(model.id) == GEMINI_KEY
    assert ctx.registry.masked_key(model.id) == "************WXYZ"
    raw = ctx.db.get_raw(f"secret.model.{model.id}")
    assert raw and GEMINI_KEY not in raw and "TESTKEY" not in raw
    row = json.dumps(ctx.db.list_model_rows())
    assert GEMINI_KEY not in row and "TESTKEY" not in row  # never in the config table


def test_update_without_key_keeps_existing_key(ctx):
    model = ctx.registry.add(ModelConfig.new("openai", "gpt-4o-mini"), OPENAI_KEY)
    ctx.registry.update(model.with_changes(model_name="gpt-4.1-mini"))  # api_key=None
    ctx.registry.update(model.with_changes(model_name="gpt-4.1"), "")  # empty field
    assert ctx.registry.api_key(model.id) == OPENAI_KEY
    ctx.registry.update(model, "sk-replacement-key-0000000000NEW1")
    assert ctx.registry.masked_key(model.id).endswith("NEW1")


def test_remove_key_keeps_model(ctx):
    model = ctx.registry.add(ModelConfig.new("gemini", "g"), GEMINI_KEY)
    other = ctx.registry.add(ModelConfig.new("openai", "o"), OPENAI_KEY)
    ctx.registry.remove_key(model.id)
    assert ctx.registry.get(model.id) is not None and not ctx.registry.has_key(model.id)
    assert not ctx.registry.is_usable(ctx.registry.get(model.id))  # Gemini needs a key
    assert ctx.registry.api_key(other.id) == OPENAI_KEY  # unrelated settings untouched


def test_keyless_local_models_are_usable(ctx):
    model = ctx.registry.add(local("llama3"))
    assert ctx.registry.is_usable(model)
    ctx.registry.set_enabled(model.id, False)
    assert not ctx.registry.is_usable(ctx.registry.get(model.id))


# ------------------------------------------------------------ persistence
def test_everything_persists_across_restarts(tmp_path):
    first = AppContext.create(tmp_path / "data")
    model = first.registry.add(
        ModelConfig.new(
            "openai_compatible", "qwen2.5:14b", display_name="Local Qwen", base_url="http://localhost:11434/v1/",
            capabilities=("text", "json", "long_context"), timeout_s=90, max_retries=2, rpm_limit=30, rpd_limit=1000,
            cost_input_per_mtok=0.2, cost_output_per_mtok=0.6,
        ),
        "local-secret-key-123456",
    )  # fmt: skip
    first.registry.set_enabled(model.id, False)
    first.settings.update(routing_strategy="cheapest", preferred_model_id=model.id, cost_aware=True, batch_size=42,
                          max_fallback_attempts=3, theme="dark", language="bn", cooldown_seconds=300)  # fmt: skip

    again = AppContext.create(tmp_path / "data")  # simulate an application restart
    loaded = again.registry.get(model.id)
    assert loaded is not None
    assert (loaded.provider_type, loaded.model_name, loaded.display_name) == ("openai_compatible", "qwen2.5:14b", "Local Qwen")
    assert loaded.base_url == "http://localhost:11434/v1" and loaded.enabled is False and loaded.priority == 1
    assert loaded.capabilities == ("text", "json", "long_context")
    assert (loaded.timeout_s, loaded.max_retries, loaded.rpm_limit, loaded.rpd_limit) == (90, 2, 30, 1000)
    assert (loaded.cost_input_per_mtok, loaded.cost_output_per_mtok) == (0.2, 0.6)
    s = again.settings.load()
    assert (s.routing_strategy, s.preferred_model_id, s.cost_aware, s.batch_size) == ("cheapest", model.id, True, 42)
    assert (s.max_fallback_attempts, s.theme, s.language, s.cooldown_seconds) == (3, "dark", "bn", 300)


def test_credentials_persist_separately(tmp_path):
    first = AppContext.create(tmp_path / "data")
    model = first.registry.add(ModelConfig.new("gemini", "g"), GEMINI_KEY)
    again = AppContext.create(tmp_path / "data")
    assert again.registry.has_key(model.id)
    assert again.registry.api_key(model.id) == GEMINI_KEY  # compared, never printed
    assert again.registry.masked_key(model.id) == "************WXYZ"


# --------------------------------------------------------------- migration
def _legacy_profile(path, *, gemini=True, openai=True, default="auto", preferred="openai", fallback="automatic"):
    """Write a settings database as the previous single-Gemini/OpenAI version did."""
    path.mkdir(parents=True, exist_ok=True)
    db = Database(path / "app.db")
    secrets = SecretStore(db, default_backend(path))
    if gemini:
        secrets.set("gemini_api_key", GEMINI_KEY)
    if openai:
        secrets.set("openai_api_key", OPENAI_KEY)
    for key, value in {"gemini_model": "gemini-2.0-flash", "openai_model": "gpt-4.1-mini", "default_provider": default,
                       "auto_preferred_provider": preferred, "fallback_mode": fallback, "request_timeout_s": "45",
                       "theme": "dark"}.items():  # fmt: skip
        db.set_raw(f"pref.{key}", value)
    db.close()


def test_migration_moves_old_keys_and_models(tmp_path):
    _legacy_profile(tmp_path / "data")
    ctx = AppContext.create(tmp_path / "data")
    models = ctx.registry.list()
    assert [(m.provider_type, m.model_name) for m in models] == [("openai", "gpt-4.1-mini"), ("gemini", "gemini-2.0-flash")]
    assert all(m.timeout_s == 45 for m in models)
    assert ctx.registry.api_key(models[0].id) == OPENAI_KEY and ctx.registry.api_key(models[1].id) == GEMINI_KEY
    assert ctx.settings.secrets.get("gemini_api_key") is None and ctx.settings.secrets.get("openai_api_key") is None
    assert ctx.settings.load().theme == "dark" and ctx.settings.load().routing_strategy == "auto_fallback"
    again = AppContext.create(tmp_path / "data")  # runs only once
    assert len(again.registry) == 2


def test_migration_respects_old_default_and_fallback_off(tmp_path):
    _legacy_profile(tmp_path / "data", openai=False, default="gemini", fallback="off")
    ctx = AppContext.create(tmp_path / "data")
    assert [m.provider_type for m in ctx.registry.list()] == ["gemini"]
    assert ctx.settings.load().routing_strategy == "auto"  # "never fall back" kept its meaning


def test_fresh_install_has_no_models(ctx):
    assert len(ctx.registry) == 0 and ctx.db.get_raw("registry.migrated_v1") == "1"


# ------------------------------------------------------------------ backup
def test_export_never_contains_keys_and_import_restores(tmp_path, ctx):
    gem = ctx.registry.add(ModelConfig.new("gemini", "gemini-2.5-flash", capabilities=("text", "json", "fast")), GEMINI_KEY)
    ctx.registry.add(local("llama3"))
    ctx.settings.update(routing_strategy="fastest", batch_size=33)
    path = tmp_path / "backup.json"
    counts = export_settings(ctx, path)
    text = path.read_text(encoding="utf-8")
    assert counts["models"] == 2 and GEMINI_KEY not in text and "TESTKEY" not in text
    data = json.loads(text)
    assert data["contains_api_keys"] is False and "window_geometry" not in data["preferences"]

    other = AppContext.create(tmp_path / "other")  # another PC
    result = import_settings(other, path)
    assert result["added"] == 2 and result["skipped"] == 0
    restored = other.registry.get(gem.id)
    assert restored is not None and restored.capabilities == ("text", "json", "fast")
    assert not other.registry.has_key(gem.id)  # keys must be re-entered
    assert other.settings.load().routing_strategy == "fastest" and other.settings.load().batch_size == 33

    result = import_settings(ctx, path)  # same PC: models updated, keys kept
    assert result["updated"] == 2 and ctx.registry.api_key(gem.id) == GEMINI_KEY


def test_import_rejects_foreign_files(tmp_path, ctx):
    bad = tmp_path / "bad.json"
    bad.write_text('{"hello": 1}', encoding="utf-8")
    with pytest.raises(BackupError):
        import_settings(ctx, bad)
    bad.write_text("not json", encoding="utf-8")
    with pytest.raises(BackupError):
        import_settings(ctx, bad)


def test_import_ignores_injected_keys_and_bad_values(tmp_path, ctx):
    path = tmp_path / "b.json"
    payload = {
        "format": "ai-video-file-assistant-settings", "version": 1,
        "preferences": {"batch_size": "lots", "theme": "light", "unknown": 1},
        "models": [{"provider_type": "openai", "model_name": "gpt-x", "display_name": "X", "base_url": "https://api.openai.com/v1",
                    "api_key": "sk-injected-000000000000000"},
                   {"provider_type": "openai", "model_name": "bad name", "display_name": "Y"}],
    }  # fmt: skip
    path.write_text(json.dumps(payload), encoding="utf-8")
    result = import_settings(ctx, path)
    assert result["added"] == 1 and result["skipped"] == 1
    model = ctx.registry.list()[0]
    assert not ctx.registry.has_key(model.id)  # a key inside a file is never imported
    assert ctx.settings.load().theme == "light" and ctx.settings.load().batch_size != "lots"
