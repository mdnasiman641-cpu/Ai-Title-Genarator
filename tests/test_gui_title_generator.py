"""Title Generator UI: panel, presets, Google Search settings and a full preview → apply → undo run (headless Qt)."""

from __future__ import annotations

import json

import pytest
from PySide6.QtWidgets import QMessageBox

from app.ai.model_config import ModelConfig
from app.ai.title_config import TitleConfig, load_current
from app.ai.web_search import SECRET_NAME
from app.context import AppContext
from app.i18n import set_language
from app.ui import title_panel as title_panel_module
from app.ui.main_window import MainWindow
from app.ui.preview_window import COL_NEW, COL_NOTE
from app.ui.settings_window import SettingsDialog
from app.ui.title_panel import TitlePanel
from tests.fake_ai_server import FakeAIServer, openai_ok
from tests.helpers import wait_until

pytestmark = pytest.mark.gui
TITLES = [
    "A Routine Investigation Takes a Dangerous Turn",
    "One Hidden Clue Changes Everything for the Reagan Family",
    "A Difficult Decision Forces Everyone to Question Loyalty",
]


@pytest.fixture
def ctx(tmp_path):
    return AppContext.create(tmp_path / "ctx")


@pytest.fixture
def panel(qapp, ctx, dispose):
    widget = TitlePanel(ctx.db)
    widget.set_expanded(True)
    widget.show()
    yield widget
    dispose(widget)
    set_language("en")


@pytest.fixture
def window(qapp, ctx, dispose, monkeypatch):
    monkeypatch.setattr(QMessageBox, "warning", lambda *a, **k: QMessageBox.StandardButton.Ok)
    monkeypatch.setattr(MainWindow, "_ask", lambda *a, **k: True)
    win = MainWindow(ctx)
    win.files_panel.set_filter("all")
    win.show()
    yield win
    wait_until(lambda: not win._workers, timeout=3)
    dispose(win)


def pick(combo, value):
    combo.setCurrentIndex(combo.findData(value))


# ======================================================================= panel
def test_defaults_and_dependent_controls(panel):
    cfg = panel.config()
    assert (cfg.mode, cfg.source, cfg.variation, cfg.capacity) == ("independent", "ai_only", "high", "medium")
    assert not cfg.enabled and panel.avoid_filename.isChecked() and not panel.avoid_filename.isEnabled()
    assert not panel.search_box.isVisibleTo(panel) and not panel.custom_category.isEnabled()
    pick(panel.source, "search_ai")
    assert panel.search_box.isVisibleTo(panel)
    pick(panel.category, "custom")
    assert panel.custom_category.isEnabled()
    pick(panel.capacity, "short")
    assert (panel.min_words.value(), panel.max_words.value()) == (5, 8) and not panel.min_words.isEnabled()
    pick(panel.capacity, "custom")
    assert panel.min_words.isEnabled()
    pick(panel.mode, "connected")
    assert panel.avoid_filename.isEnabled()
    panel.set_expanded(False)
    assert not panel.body.isVisibleTo(panel)


def test_config_roundtrip(panel):
    cfg = TitleConfig(
        enabled=True, mode="partial", source="search_ai", category="custom", custom_category="Political Thriller",
        topic="secrets", keywords="Mystery, Justice", style="custom", custom_style="episode guide", tone="custom",
        custom_tone="hopeful", capacity="custom", min_words=9, max_words=15, prefix="Blue Bloods", prefix_counts=True,
        variation="medium", unique=False, avoid_filename=False, prevent_duplicates=True, custom_instructions="No clichés.",
        use_filename_lookup=False, query_mode="custom", max_results=10, custom_query="{series} plot", compare_results=False,
        context_only=True, completely_new=True, search_fallback=False,
    ).normalised()  # fmt: skip
    panel.set_config(cfg)
    assert panel.config() == cfg


def test_search_hints(panel):
    panel.set_search_configured(False)
    pick(panel.source, "search_only")
    assert "Settings → AI → Google Search" in panel.search_hint.text() and "AI model is required" in panel.search_hint.text()


def test_presets_save_load_delete_and_persist(panel, ctx, tmp_path, monkeypatch):
    panel.prefix.setText("Blue Bloods")
    pick(panel.style, "facebook")
    assert panel.save_preset("Blue Bloods Facebook Titles")
    panel.prefix.setText("Other")
    pick(panel.style, "simple")
    panel.enabled.setChecked(True)
    panel.preset_combo.setCurrentIndex(panel.preset_combo.findText("Blue Bloods Facebook Titles"))
    assert panel.load_selected_preset()
    cfg = panel.config()
    assert cfg.prefix == "Blue Bloods" and cfg.style == "facebook" and cfg.enabled  # loading keeps the on/off switch
    fresh = TitlePanel(AppContext.create(tmp_path / "ctx").db)  # after a restart
    try:
        assert [fresh.preset_combo.itemText(i) for i in range(fresh.preset_combo.count())] == ["Blue Bloods Facebook Titles"]
    finally:
        fresh.deleteLater()
    monkeypatch.setattr(title_panel_module, "confirm_destructive", lambda *a: False)
    assert not panel.delete_preset() and panel.preset_combo.count() == 1
    monkeypatch.setattr(title_panel_module, "confirm_destructive", lambda *a: True)
    assert panel.delete_preset() and panel.preset_combo.count() == 0 and ctx.db.list_title_presets() == []


def test_clear_cache_button(panel, ctx):
    ctx.db.put_search_cache("q|5|1", 9e9, {"query": "q"})
    panel.clear_cache_button.click()
    assert ctx.db.count_search_cache() == 0 and "1" in panel.preset_status.text()


def test_bangla_labels(panel):
    set_language("bn")
    panel.retranslate()
    assert panel.save_preset_button.text() == "প্রিসেট সংরক্ষণ" and panel.category.itemText(0) == "অপরাধ"


# =============================================================== main window
def test_settings_are_auto_saved_and_restored(window, ctx, tmp_path, dispose):
    tp = window.title_panel
    tp.enabled.setChecked(True)
    tp.prefix.setText("Blue Bloods")
    assert load_current(ctx.db).prefix == "Blue Bloods" and load_current(ctx.db).enabled
    again = MainWindow(AppContext.create(tmp_path / "ctx"))
    try:
        assert again.title_panel.config().prefix == "Blue Bloods" and again.title_panel.enabled.isChecked()
    finally:
        dispose(again)


def test_generate_preview_apply_and_undo(window, ctx, workspace):
    for name in ("video001.mp4", "video002.MKV", "random_video.mp4"):
        (workspace / name).write_bytes(b"x")
    with FakeAIServer() as srv:
        ctx.registry.add(ModelConfig.new("openai_compatible", "local-model", base_url=srv.base, max_retries=0))
        window.refresh_models_ui()
        window.load_folder(workspace)
        assert wait_until(lambda: window.files_panel.model.rowCount() == 3)
        tp = window.title_panel
        tp.enabled.setChecked(True)
        tp.prefix.setText("Blue Bloods")
        pick(tp.capacity, "custom")
        tp.min_words.setValue(6)
        tp.max_words.setValue(12)
        srv.queue(200, openai_ok(json.dumps({"titles": [{"n": i + 1, "title": t} for i, t in enumerate(TITLES)]})), delay=0.4)
        window.command_panel.set_command("")  # no command needed when the Title Generator is on
        window.generate_preview()
        assert window.command_panel._busy and window.command_panel.cancel_button.isVisible()  # UI not blocked
        assert wait_until(lambda: window._plan_worker is None and not window.command_panel._busy, timeout=8)
        body = json.loads(srv.requests[0]["body"])
        prompt = body["messages"][1]["content"]
        assert "video001" not in prompt and "random_video" not in prompt  # Independent Creative: names never sent
    model = window.preview_panel.model
    news = sorted(model.index(r, COL_NEW).data() for r in range(model.rowCount()))
    assert all(n.startswith("Blue Bloods - ") for n in news) and any(n.endswith(".MKV") for n in news)
    assert "AI: OpenAI-compatible / local-model" in model.index(0, COL_NOTE).data()
    assert "Completed · Model: OpenAI-compatible / local-model" in window.preview_panel.info.text()
    window.apply_changes()
    assert wait_until(lambda: sorted(p.name for p in workspace.iterdir() if p.is_file()) == news and not window._operation_running)
    window.undo_last()
    assert wait_until(lambda: sorted(p.name for p in workspace.iterdir() if p.is_file()) == ["random_video.mp4", "video001.mp4", "video002.MKV"])


# ==================================================================== settings
def test_google_search_settings(qapp, ctx, dispose, monkeypatch):
    dialog = SettingsDialog(ctx)
    try:
        assert dialog.search_key_status.text() == "No key saved yet." and not dialog.search_key_remove.isEnabled()
        dialog.search_key_input.setText("AIzaSySEARCH-KEY-000000000000WXYZ")
        dialog.search_engine_id.setText("engine-123")
        dialog.search_cache_hours.setValue(6)
        dialog.save()
        assert ctx.settings.secrets.get(SECRET_NAME) == "AIzaSySEARCH-KEY-000000000000WXYZ"
        raw = ctx.db.get_raw(f"secret.{SECRET_NAME}") or ""
        assert "SEARCH-KEY" not in raw
        s = ctx.settings.reload()
        assert (s.search_engine_id, s.search_cache_hours) == ("engine-123", 6)
    finally:
        dispose(dialog)
    dialog = SettingsDialog(ctx)
    try:
        assert dialog.search_key_status.text() == "Saved key: ************WXYZ"
        dialog.search_endpoint.setText("http://public.example.com/search")
        dialog.save()  # refused: a key must never travel over plain http to a public host
        assert dialog.result() != SettingsDialog.DialogCode.Accepted and "https" in dialog.model_result.text().lower()
        dialog.search_endpoint.setText("")
        from app.ui import settings_window

        monkeypatch.setattr(settings_window, "confirm_destructive", lambda *a: True)
        dialog.search_key_remove.click()
        assert ctx.settings.secrets.get(SECRET_NAME) is None and ctx.settings.load().search_engine_id == "engine-123"
    finally:
        dispose(dialog)
