"""Saved prompts, command history and operation-history dialogs; logging and error hook."""

from __future__ import annotations

import logging

import pytest
from PySide6.QtWidgets import QInputDialog, QMessageBox

from app.ai.model_config import ModelConfig
from app.config.constants import TRASH_DIR_NAME
from app.context import AppContext
from app.i18n import set_language
from app.ui.history_window import HistoryDialog, format_time
from app.ui.main_window import MainWindow
from app.ui.saved_prompts_window import SavedPromptsDialog
from app.ui.theme import apply_theme
from tests.helpers import wait_until

pytestmark = pytest.mark.gui


@pytest.fixture
def ctx(tmp_path):
    return AppContext.create(tmp_path / "ctx")


@pytest.fixture
def yes(monkeypatch):
    monkeypatch.setattr(QMessageBox, "question", lambda *a, **k: QMessageBox.StandardButton.Yes)
    monkeypatch.setattr(QMessageBox, "warning", lambda *a, **k: QMessageBox.StandardButton.Yes)


# ------------------------------------------------------------ saved prompts
def test_default_prompts_are_present_with_readable_names(ctx):
    names = [p.name for p in ctx.db.list_saved_prompts()]
    assert names == sorted(names, key=str.lower) and len(names) == 5
    assert {"Blue Bloods Rename", "Clean Video Filename", "Episode Numbering", "Remove Quality Tags", "YouTube Title Format"} == set(names)
    assert all(not n.startswith("prompt.default") for n in names)  # no untranslated keys leaked into the data


def add_openai(ctx):
    return ctx.registry.add(ModelConfig.new("openai", "gpt-4o-mini"), "sk-test-1234567890abcdefgh")


def test_prompt_crud(qapp, ctx, dispose, yes):
    model = add_openai(ctx)
    dlg = SavedPromptsDialog(ctx)
    assert [dlg.provider_combo.itemData(i) for i in range(dlg.provider_combo.count())] == ["auto", model.id]
    count = dlg.prompt_list.count()
    dlg.new_prompt()
    dlg.save_prompt()  # empty name/text -> refused
    assert ctx.db.count_saved_prompts() == count
    dlg.name_edit.setText("My rename")
    dlg.prompt_edit.setPlainText("Add prefix X to everything")
    dlg.provider_combo.setCurrentIndex(dlg.provider_combo.findData(model.id))
    dlg.save_prompt()
    saved = next(p for p in ctx.db.list_saved_prompts() if p.name == "My rename")
    assert saved.provider == model.id and saved.prompt == "Add prefix X to everything" and saved.created_at
    assert dlg.created_label.text() == saved.created_at  # the created date is shown
    dlg.prompt_edit.setPlainText("Add prefix Y")
    dlg.save_prompt()
    assert next(p for p in ctx.db.list_saved_prompts() if p.id == saved.id).prompt == "Add prefix Y"
    dlg.delete_prompt()
    assert all(p.id != saved.id for p in ctx.db.list_saved_prompts())
    dispose(dlg)


def test_use_prompt_emits_text_and_provider(qapp, ctx, dispose):
    dlg = SavedPromptsDialog(ctx)
    got = []
    dlg.use_requested.connect(lambda text, provider: got.append((text, provider)))
    dlg.prompt_list.setCurrentRow(0)
    dlg.use_button.click()
    assert got and got[0][0] and got[0][1] in {"gemini", "openai", "auto"}
    dispose(dlg)


def test_prefill_opens_a_new_prompt(qapp, ctx, dispose):
    dlg = SavedPromptsDialog(ctx, prefill=("Remove 1080p from all filenames", "gemini"))
    assert dlg.prompt_edit.toPlainText() == "Remove 1080p from all filenames"
    assert dlg.name_edit.text().startswith("Remove 1080p") and not dlg.delete_button.isEnabled()
    dispose(dlg)


def test_command_history_reuse_favorite_delete_clear(qapp, ctx, dispose, yes):
    ctx.db.add_command("first command", "gemini")
    ctx.db.add_command("second command", "openai")
    dlg = SavedPromptsDialog(ctx)
    assert dlg.history_list.count() == 2 and "second command" in dlg.history_list.item(0).text()
    dlg.history_list.setCurrentRow(1)
    dlg.h_fav.click()  # favourite the older one -> it moves to the top with a star
    assert dlg.history_list.item(0).text().startswith("★") and "first command" in dlg.history_list.item(0).text()
    got = []
    dlg.use_requested.connect(lambda text, provider: got.append((text, provider)))
    dlg.history_list.setCurrentRow(0)
    dlg.h_use.click()
    assert got == [("first command", "gemini")]
    dlg2 = SavedPromptsDialog(ctx)
    dlg2.history_list.setCurrentRow(1)
    dlg2.h_delete.click()
    assert [c.command for c in ctx.db.list_commands()] == ["first command"]
    dlg2.h_clear.click()  # keeps favourites
    assert [c.command for c in ctx.db.list_commands()] == ["first command"]
    dispose(dlg, dlg2)


def test_history_save_as_prompt(qapp, ctx, dispose):
    ctx.db.add_command("reusable thing", "auto")
    dlg = SavedPromptsDialog(ctx)
    dlg.history_list.setCurrentRow(0)
    dlg.h_save.click()
    assert dlg.prompt_edit.toPlainText() == "reusable thing" and dlg.tabs.currentIndex() == 0
    dispose(dlg)


def test_dialog_translates(qapp, ctx, dispose):
    dlg = SavedPromptsDialog(ctx)
    set_language("bn")
    dlg.retranslate()
    assert dlg.tabs.tabText(0) == "সংরক্ষিত প্রম্পট" and dlg.save_button.text() == "সংরক্ষণ"
    set_language("en")
    dispose(dlg)


# ------------------------------------------------------- operation history
@pytest.fixture
def history_ctx(ctx, workspace, make_files):
    """A context with one applied rename and one delete (to undo-trash)."""
    from app.ai.schemas import AddPrefixAction, DeleteAction
    from app.files.operation_manager import OperationManager
    from app.files.planner import Planner
    from app.files.scanner import scan_folder

    make_files("a.mp4", "junk.txt")
    mgr = OperationManager(ctx.db)
    entries = scan_folder(workspace).entries
    mgr.execute(Planner(workspace, [e for e in entries if e.name == "a.mp4"]).build([AddPrefixAction("X ")]), command="prefix X", provider="offline")
    entries = scan_folder(workspace).entries
    mgr.execute(Planner(workspace, [e for e in entries if e.name == "junk.txt"]).build([DeleteAction(source="junk.txt")]), command="delete junk", provider="gemini")
    return ctx


def test_history_lists_operations_newest_first(qapp, history_ctx, dispose):
    dlg = HistoryDialog(history_ctx)
    assert dlg.table.rowCount() == 2
    assert dlg.table.item(0, 0).text() == "#2" and dlg.table.item(0, 2).text() == "delete junk"
    assert dlg.table.item(0, 3).text() == "1 delete operation" and dlg.table.item(0, 4).text() == "Applied"
    assert dlg.table.item(1, 3).text() == "1 rename operation"
    assert "delete (undo-trash)" in dlg.details.toPlainText() and "delete junk" in dlg.details.toPlainText()
    dispose(dlg)


def test_history_undo_button_emits_operation_id(qapp, history_ctx, dispose):
    dlg = HistoryDialog(history_ctx)
    got = []
    dlg.undo_requested.connect(got.append)
    dlg.table.selectRow(1)
    assert dlg.undo_button.isEnabled()
    dlg.undo_button.click()
    assert got == [1]
    dispose(dlg)


def test_history_empty_trash_and_remove(qapp, history_ctx, workspace, dispose, yes):
    dlg = HistoryDialog(history_ctx)
    dlg.table.selectRow(0)
    assert dlg.trash_button.isEnabled() and "1 files" in dlg.trash_button.text()
    dlg.trash_button.click()
    assert not (workspace / TRASH_DIR_NAME).exists() and not dlg.trash_button.isEnabled()
    dlg.remove_button.click()
    assert dlg.table.rowCount() == 1
    dlg.clear_button.click()
    assert dlg.table.rowCount() == 0 and not dlg.undo_button.isEnabled()
    dispose(dlg)


def test_format_time_is_human_readable():
    assert format_time("2026-10-02 00:42:10") == "2026-10-02 12:42 AM"
    assert format_time("garbage") == "garbage"


# ------------------------------------------------------- main window wiring
@pytest.fixture
def window(qapp, ctx, dispose):
    apply_theme(qapp, "dark")
    win = MainWindow(ctx)
    win.show()
    yield win
    wait_until(lambda: not win._workers, timeout=3)
    dispose(win)
    set_language("en")


def test_header_has_prompts_history_undo_settings(window):
    assert list(window.header_buttons) == ["prompts", "history", "undo", "settings"]
    assert not window.header_buttons["undo"].isEnabled()  # nothing to undo on a fresh install


def test_save_current_command_as_prompt(window, ctx, monkeypatch):
    monkeypatch.setattr(QInputDialog, "getText", lambda *a, **k: ("Quick prompt", True))
    window.command_panel.set_command("Add suffix _HD")
    model = add_openai(ctx)
    window.refresh_models_ui()
    window.command_panel.set_model_choice(model.id)
    window.save_current_as_prompt()
    saved = next(p for p in ctx.db.list_saved_prompts() if p.name == "Quick prompt")
    assert (saved.prompt, saved.provider) == ("Add suffix _HD", model.id)
    assert any("Quick prompt" in a.text() for a in window.command_panel.prompts_menu.actions())
    assert "saved" in window.status_label.text()


def test_saved_prompt_menu_fills_command_and_model(window, ctx):
    model = add_openai(ctx)
    window.refresh_models_ui()
    pid = ctx.db.add_saved_prompt("Zed", "Remove x264 from all filenames", "openai")  # value stored by older versions
    window.refresh_prompt_menus()
    action = next(a for a in window.command_panel.prompts_menu.actions() if a.text() == "⭐ Zed")
    action.trigger()
    assert window.command_panel.command() == "Remove x264 from all filenames"
    assert window.command_panel.model_choice() == model.id and pid  # legacy "openai" -> first OpenAI model


def test_history_dialog_undo_goes_through_the_main_window(window, history_ctx, workspace, monkeypatch):
    window.ctx = history_ctx
    window.manager.__init__(history_ctx.db)
    monkeypatch.setattr(MainWindow, "_ask", lambda self, *a, **k: True)
    window.refresh_undo_button()
    assert window.header_buttons["undo"].isEnabled() and "#2" in window.header_buttons["undo"].toolTip()
    window.undo_operation(2)
    assert wait_until(lambda: (workspace / "junk.txt").exists() and not window._operation_running)


def test_blocked_undo_is_explained_not_attempted(window, history_ctx, workspace, monkeypatch):
    window.ctx = history_ctx
    window.manager.__init__(history_ctx.db)
    shown = []
    monkeypatch.setattr(QMessageBox, "information", lambda *a, **k: shown.append(a[2]))
    (workspace / "X a.mp4").unlink()  # the renamed file is gone
    window.undo_operation(1)
    assert shown and "cannot be undone safely" in shown[0] and "missing" in shown[0]
    assert not (workspace / "a.mp4").exists()


# ----------------------------------------------------- logging / error hook
def test_logging_writes_a_file_without_secrets(tmp_path, monkeypatch):
    from app.utils.logger import get_log_dir, get_logger, setup_logging
    from app.utils.security import register_secret

    monkeypatch.setenv("AIVFA_DATA_DIR", str(tmp_path / "logs-data"))
    setup_logging("INFO", to_file=True)
    secret = "AIzaSyLOGSECRET1234567890abcdefghijklm"
    register_secret(secret)
    log = get_logger("test")
    log.info("request with key=%s and header Authorization: Bearer sk-abcdefghijklmnop12345678", secret)
    log.error("another %s", secret)
    for handler in logging.getLogger("aivfa").handlers:
        handler.flush()
    text = (get_log_dir() / "app.log").read_text(encoding="utf-8")
    assert "starting" in text and "request with key=" in text  # startup + normal lines are present
    assert secret not in text and "sk-abcdefghijklmnop" not in text and "[REDACTED]" in text
    setup_logging("INFO", to_file=False)


def test_exception_hook_logs_and_shows_friendly_dialog(qapp, monkeypatch, tmp_path):
    import sys

    from app.ui import error_hook

    shown = []
    monkeypatch.setattr(QMessageBox, "critical", lambda *a, **k: shown.append(a[2]))
    monkeypatch.setattr(error_hook, "_last_dialog", 0.0)
    original = sys.excepthook
    try:
        error_hook.install_exception_hooks()
        try:
            raise ValueError("boom")
        except ValueError:
            sys.excepthook(*sys.exc_info())
        assert shown and "ValueError: boom" in shown[0] and "your files were not touched" in shown[0]
        sys.excepthook(ValueError, ValueError("again"), None)  # rate-limited: no dialog storm
        assert len(shown) == 1
    finally:
        sys.excepthook = original
