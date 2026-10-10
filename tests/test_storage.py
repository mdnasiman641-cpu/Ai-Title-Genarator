"""SQLite layer, typed settings, encrypted API keys, helpers and translations."""

from __future__ import annotations

import os
import re
from pathlib import Path

import pytest

from app.context import AppContext
from app.database.database import Database
from app.utils import helpers
from app.utils.logger import RedactingFilter
from app.utils.security import (
    KeyFileBackend,
    SecretStore,
    mask_key,
    redact,
    register_secret,
)


@pytest.fixture
def ctx(tmp_path):
    return AppContext.create(tmp_path / "data")


def test_settings_defaults_are_safe(ctx):
    s = ctx.settings.load()
    assert s.auto_apply is False  # Auto Apply must be OFF by default
    assert s.ask_before_apply and s.prevent_overwrite and s.create_history
    assert s.include_subfolders is False
    assert s.routing_strategy == "auto_fallback" and s.preferred_model_id == "" and s.language == "en"
    assert s.cost_aware is False and s.health_tracking is True and s.use_offline_parser is True


def test_settings_roundtrip_and_normalisation(ctx):
    ctx.settings.update(theme="dark", batch_size=75, include_subfolders=True, language="bn")
    ctx.settings.reload()
    s = ctx.settings.load()
    assert (s.theme, s.batch_size, s.include_subfolders, s.language) == ("dark", 75, True, "bn")
    ctx.settings.update(theme="neon", batch_size=99999, routing_strategy="skynet", max_fallback_attempts=500)
    s = ctx.settings.reload()
    assert s.theme == "system" and s.batch_size == 200
    assert s.routing_strategy == "auto_fallback" and s.max_fallback_attempts == 10


def test_secret_store_encrypts_at_rest(ctx):
    store = ctx.settings.secrets
    store.set("model.abc", "AIzaSyDUMMYKEY1234567890abcdefghijklmn")
    assert store.has("model.abc") and store.get("model.abc") == "AIzaSyDUMMYKEY1234567890abcdefghijklmn"
    raw = ctx.db.get_raw("secret.model.abc")
    assert raw and "AIza" not in raw and "DUMMYKEY" not in raw
    assert "secret.model.abc" not in ctx.db.all_settings()  # secrets never listed with settings
    assert "AIza" not in ctx.settings.export_json()
    store.delete("model.abc")
    assert store.get("model.abc") is None and not store.has("model.abc")


def test_mask_never_reveals_short_keys():
    assert mask_key("") == "" and mask_key(None) == ""
    assert mask_key("abcd") == "************"
    assert mask_key("sk-1234567890WXYZ") == "************WXYZ"


def test_redaction_scrubs_keys_everywhere():
    register_secret("my-very-secret-token-value")
    text = "key=my-very-secret-token-value and AIzaSyA1234567890abcdefghijklmnopqrstuv and sk-proj-abcdefghijklmnop1234"
    out = redact(text)
    assert "my-very-secret" not in out and "AIza" not in out and "sk-proj" not in out
    assert "[REDACTED]" in out


def test_logger_filter_redacts_records():
    import logging

    record = logging.LogRecord("x", logging.INFO, "f", 1, "Authorization: Bearer abcdefghijklmnop1234567", None, None)
    RedactingFilter().filter(record)
    assert "abcdefghijklmnop1234567" not in record.getMessage()


def test_keyfile_backend_roundtrip_and_permissions(tmp_path):
    backend = KeyFileBackend(tmp_path / "k" / "secret.key")
    blob = backend.protect(b"hello secret")
    assert b"hello secret" not in blob and backend.unprotect(blob) == b"hello secret"
    assert backend.protect(b"hello secret") != blob  # random nonce
    if os.name != "nt":  # POSIX permission bits do not apply on Windows (where DPAPI is used instead)
        assert (tmp_path / "k" / "secret.key").stat().st_mode & 0o077 == 0


def test_corrupt_secret_returns_none(tmp_path):
    db = Database(tmp_path / "a.db")
    store = SecretStore(db, KeyFileBackend(tmp_path / "secret.key"))
    db.set_raw("secret.x", "!!!not-base64!!!")
    assert store.get("x") is None


def test_saved_prompts_crud(ctx):
    n = ctx.db.count_saved_prompts()
    pid = ctx.db.add_saved_prompt("My prompt", "Do things", "openai")
    prompts = ctx.db.list_saved_prompts()
    assert len(prompts) == n + 1
    mine = next(p for p in prompts if p.id == pid)
    assert (mine.name, mine.prompt, mine.provider) == ("My prompt", "Do things", "openai") and mine.created_at
    ctx.db.update_saved_prompt(pid, "Renamed", "New text", "gemini")
    assert next(p for p in ctx.db.list_saved_prompts() if p.id == pid).name == "Renamed"
    ctx.db.delete_saved_prompt(pid)
    assert all(p.id != pid for p in ctx.db.list_saved_prompts())


def test_default_prompts_seeded_once(tmp_path):
    first = AppContext.create(tmp_path / "d")
    count = first.db.count_saved_prompts()
    assert count == 5
    for p in first.db.list_saved_prompts():
        first.db.delete_saved_prompt(p.id)
    second = AppContext.create(tmp_path / "d")  # deleting defaults must stick
    assert second.db.count_saved_prompts() == 0


def test_command_history_dedupes_favorites_and_limits(ctx):
    a = ctx.db.add_command("remove 1080p", "gemini")
    ctx.db.add_command("add prefix X", "openai")
    ctx.db.add_command("remove 1080p", "gemini")  # moves to top, no duplicate
    cmds = ctx.db.list_commands()
    assert [c.command for c in cmds].count("remove 1080p") == 1
    assert cmds[0].command == "remove 1080p"
    cid = next(c.id for c in cmds if c.command == "add prefix X")
    ctx.db.set_command_favorite(cid, True)
    assert ctx.db.list_commands()[0].favorite  # favourites sort first
    ctx.db.clear_commands(keep_favorites=True)
    assert [c.command for c in ctx.db.list_commands()] == ["add prefix X"]
    ctx.db.clear_commands(keep_favorites=False)
    assert ctx.db.list_commands() == []
    assert a


def test_database_is_usable_from_worker_threads(tmp_path):
    import threading

    db = Database(tmp_path / "t.db")
    errors: list[Exception] = []

    def work(i: int) -> None:
        try:
            db.add_command(f"cmd {i}", "gemini")
        except Exception as exc:  # noqa: BLE001
            errors.append(exc)

    threads = [threading.Thread(target=work, args=(i,)) for i in range(8)]
    [t.start() for t in threads]
    [t.join() for t in threads]
    assert not errors and len(db.list_commands()) == 8


def test_helpers_formatting():
    assert helpers.format_size(0) == "0 B"
    assert helpers.format_size(1536) == "1.5 KB"
    assert helpers.format_size(5 * 1024**3) == "5.0 GB"
    assert helpers.format_duration(None) == "" and helpers.format_duration(65) == "1:05"
    assert helpers.format_duration(3725) == "1:02:05"
    assert sorted(["ep10", "ep2", "Ep1"], key=helpers.natural_sort_key) == ["Ep1", "ep2", "ep10"]
    assert helpers.truncate("abcdef", 4) == "abc…"


def test_describe_os_error_is_friendly():
    import errno

    assert "full" in helpers.describe_os_error(OSError(errno.ENOSPC, "x")).lower()
    assert "permission" in helpers.describe_os_error(PermissionError()).lower()
    assert "not found" in helpers.describe_os_error(FileNotFoundError()).lower()
    assert "exists" in helpers.describe_os_error(FileExistsError()).lower()


def test_translations_are_complete_and_formattable():
    from app.i18n import _BN, _EN, set_language, tr

    missing = set(_EN) - set(_BN)
    assert not missing, f"Bangla translation missing for: {sorted(missing)}"
    assert not set(_BN) - set(_EN), "Bangla has keys English lacks"
    for key, text in _EN.items():  # placeholders must match between languages
        assert set(re.findall(r"{(\w+)}", text)) == set(re.findall(r"{(\w+)}", _BN[key])), key
    set_language("bn")
    try:
        assert tr("folder.title") == "ফোল্ডার"
        assert tr("no.such.key") == "no.such.key"
        assert "7" in tr("files.found", count=7)
    finally:
        set_language("en")
    assert tr("files.found", count=3) == "3 files found"


def test_every_ui_translation_key_used_in_code_exists():
    from app.i18n import _EN

    root = Path(__file__).resolve().parent.parent / "app"
    used: set[str] = set()
    for path in root.rglob("*.py"):
        if "i18n" in path.parts:
            continue
        text = path.read_text(encoding="utf-8")
        used.update(re.findall(r'\btr\(\s*"([a-z_]+(?:\.[a-z_0-9]+)+)"', text))
        used.update(re.findall(r'(?:text|tooltip|placeholder|title|bind)\([^"\n]*?,\s*"([a-z_]+(?:\.[a-z_0-9]+)+)"', text))
        used.update(re.findall(r'(?:_key|Card)\(\s*"([a-z_]+(?:\.[a-z_0-9]+)+)"', text))
    missing = sorted(k for k in used if k not in _EN)
    assert not missing, f"Missing English strings: {missing}"
