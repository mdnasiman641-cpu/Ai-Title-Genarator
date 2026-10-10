"""End-to-end UI workflow: folder -> command -> preview -> apply -> undo (headless Qt)."""

from __future__ import annotations

import json

import pytest
from PySide6.QtWidgets import QLabel, QMessageBox

from app.ai.model_config import ModelConfig
from app.config.constants import ProcessingMode, RoutingStrategy
from app.context import AppContext
from app.i18n import set_language
from app.ui.main_window import MainWindow
from app.ui.theme import apply_theme
from tests.fake_ai_server import FakeAIServer, gemini_ok, openai_ok
from tests.helpers import wait_until

pytestmark = pytest.mark.gui
SAMPLE = [f"VID_00{i}_FINAL_1080P_WEB-DL.mp4" for i in (1, 2, 3)]


class Dialogs:
    """Records and scripts every confirmation / message box the window shows."""

    def __init__(self) -> None:
        self.asked: list[tuple[str, str, bool]] = []
        self.answers: dict[str, bool] = {}
        self.default = True
        self.warnings: list[str] = []
        self.infos: list[str] = []
        self.question_answer = QMessageBox.StandardButton.Yes
        self.questions: list[str] = []


@pytest.fixture
def dialogs(monkeypatch):
    d = Dialogs()

    def fake_ask(self, title, text, ok_text, *, destructive=False):
        d.asked.append((title, text, destructive))
        return d.answers.get(title, d.default)

    monkeypatch.setattr(MainWindow, "_ask", fake_ask)
    monkeypatch.setattr(QMessageBox, "warning", lambda *a, **k: d.warnings.append(a[2]) or QMessageBox.StandardButton.Ok)
    monkeypatch.setattr(QMessageBox, "information", lambda *a, **k: d.infos.append(a[2]) or QMessageBox.StandardButton.Ok)
    monkeypatch.setattr(QMessageBox, "question", lambda *a, **k: d.questions.append(a[2]) or d.question_answer)
    return d


@pytest.fixture
def server():
    with FakeAIServer() as srv:
        yield srv


def add_gemini(ctx, server, window=None, **kw) -> ModelConfig:
    """A Gemini model configuration pointed at the fake server (no real key needed)."""
    model = ctx.registry.add(ModelConfig.new("gemini", "gemini-test", base_url=server.base, max_retries=0, **kw),
                             "AIzaSyKEY-1234567890-abcdefghijklmn")  # fmt: skip
    if window is not None:
        window.refresh_models_ui()
    return model


def add_openai(ctx, server, window=None, **kw) -> ModelConfig:
    model = ctx.registry.add(ModelConfig.new("openai", "gpt-test", base_url=server.base, max_retries=0, **kw), "sk-test-1234567890abcdefgh")
    if window is not None:
        window.refresh_models_ui()
    return model


@pytest.fixture
def app_ctx(tmp_path):
    return AppContext.create(tmp_path / "ctx")


@pytest.fixture
def window(qapp, app_ctx, dispose, dialogs):
    apply_theme(qapp, "dark")
    win = MainWindow(app_ctx)
    win.files_panel.set_filter("all")
    win.show()
    yield win
    wait_until(lambda: not win._workers, timeout=3)
    dispose(win)
    set_language("en")


def open_folder(window, workspace, count):
    window.load_folder(workspace)
    assert wait_until(lambda: window.files_panel.model.rowCount() == count)


def generate(window, command, *, processing=None):
    window.command_panel.set_command(command)
    if processing is not None:
        window.command_panel.mode_combo.setCurrentIndex(window.command_panel.mode_combo.findData(processing.value))
    window.generate_preview()
    assert wait_until(lambda: window._plan_worker is None and not window.command_panel._busy, timeout=8)


def names(workspace):
    return sorted(p.name for p in workspace.iterdir() if p.is_file())


# ------------------------------------------------------------- offline path
def test_offline_command_full_cycle_with_undo(window, workspace, make_files, dialogs):
    make_files(*SAMPLE)
    open_folder(window, workspace, 3)
    generate(window, "Remove 1080p and WEB-DL from all filenames.")
    panel = window.preview_panel
    assert panel.model.rowCount() == 3 and panel.apply_count() == 3
    assert "Offline" in window.provider_label.text()
    assert panel.apply_button.text() == "Apply 3 Changes" and panel.apply_button.isEnabled()
    assert names(workspace) == sorted(SAMPLE)  # nothing changed yet: a preview is only a preview
    assert [window.files_panel.model.entry_at(r).status for r in range(3)] == ["→ Rename"] * 3

    window.apply_changes()
    assert dialogs.asked[0][1] == "Apply 3 changes?"
    assert wait_until(lambda: names(workspace) == ["VID_001_FINAL.mp4", "VID_002_FINAL.mp4", "VID_003_FINAL.mp4"])
    assert wait_until(lambda: [e.status for e in window.files_panel.model.entries] == ["Renamed"] * 3)  # rescan done
    assert not window._operation_running and panel.plan is None
    assert "Applied 3 changes" in window.status_label.text()
    assert len(window.files_panel.selected_entries()) == 3  # the renamed files stay selected

    assert window.header_buttons["undo"].isEnabled()
    window.undo_last()
    assert "Undo operation #1" in dialogs.asked[-1][1]
    assert wait_until(lambda: names(workspace) == sorted(SAMPLE))
    assert wait_until(lambda: not window.header_buttons["undo"].isEnabled())


def test_cancelling_the_confirmation_changes_nothing(window, workspace, make_files, dialogs):
    make_files(*SAMPLE)
    open_folder(window, workspace, 3)
    generate(window, "Add prefix Blue Bloods - ")
    dialogs.default = False
    window.apply_changes()
    assert names(workspace) == sorted(SAMPLE) and window.preview_panel.plan is not None  # preview stays for another try


def test_apply_without_confirmation_when_setting_off(window, app_ctx, workspace, make_files, dialogs):
    app_ctx.settings.update(ask_before_apply=False)
    window.settings = app_ctx.settings.load()
    make_files("a.mp4")
    open_folder(window, workspace, 1)
    generate(window, "Add suffix _HD")
    window.apply_changes()
    assert dialogs.asked == []
    assert wait_until(lambda: names(workspace) == ["a_HD.mp4"])


def test_sort_command_only_reorders_the_table(window, workspace, make_files, dialogs):
    (workspace / "small.mp4").write_bytes(b"1")
    (workspace / "big.mp4").write_bytes(b"1" * 999)
    open_folder(window, workspace, 2)
    generate(window, "Sort by size descending")
    panel = window.files_panel
    assert [panel.proxy.index(r, 1).data() for r in range(2)] == ["big.mp4", "small.mp4"]
    assert window.preview_panel.apply_count() == 0 and "No files were changed" in window.status_label.text()
    assert not window.preview_panel.apply_button.isEnabled()


def test_nothing_to_change(window, workspace, make_files):
    make_files("a.mp4")
    open_folder(window, workspace, 1)
    generate(window, "Remove zzz from all filenames")
    assert "Nothing to change" in window.status_label.text() and not window.preview_panel.apply_button.isEnabled()


def test_validation_messages_before_generating(window, workspace, make_files):
    window.command_panel.set_command("Add prefix X")
    window.generate_preview()
    assert "Choose a folder" in window.status_label.text()
    make_files("a.mp4")
    open_folder(window, workspace, 1)
    window.command_panel.set_command("")
    window.generate_preview()
    assert "type a command" in window.status_label.text()
    window.command_panel.set_command("Add prefix X")
    window.files_panel.select_none()
    window.generate_preview()
    assert "No files are selected" in window.status_label.text()


def test_unticking_a_conflicting_row_updates_the_preview(window, workspace, make_files):
    make_files("a_1.mp4", "a_2.mp4")
    open_folder(window, workspace, 2)
    generate(window, "Remove _1 and _2 from all filenames")
    panel = window.preview_panel
    assert "1 conflicts" in panel.summary.text() and panel.apply_count() == 1
    ok_row = next(r for r in range(panel.model.rowCount()) if panel.model._rows[r].status.value == "ok")
    panel.model.setData(panel.model.index(ok_row, 0), 0, 10)  # Qt.CheckStateRole = 10 -> unchecked
    assert panel.apply_count() == 1 and "conflict" not in panel.summary.text()  # the loser no longer collides


def test_discarding_the_preview_clears_marks(window, workspace, make_files):
    make_files("a_1080p.mp4")
    open_folder(window, workspace, 1)
    generate(window, "Remove 1080p from all filenames")
    assert window.files_panel.model.entry_at(0).status
    window.preview_panel.cancel_button.click()
    assert window.preview_panel.plan is None and window.files_panel.model.entry_at(0).status == ""


# ----------------------------------------------------------------- deletes
def test_delete_always_needs_the_extra_confirmation(window, app_ctx, workspace, make_files, dialogs, server):
    app_ctx.settings.update(ask_before_apply=False)
    window.settings = app_ctx.settings.load()
    add_gemini(app_ctx, server, window)
    make_files("keep.mp4", "junk.mp4")
    open_folder(window, workspace, 2)
    server.queue(200, gemini_ok(json.dumps({"actions": [{"type": "delete", "source": "junk.mp4"}], "summary": "Delete junk"})))
    dialogs.default = False  # the user says "no" to the delete confirmation
    generate(window, "please delete the junk file")
    window.apply_changes()
    assert [t for t, *_ in dialogs.asked] == ["Delete files"]  # asked even though "ask before applying" is off
    assert "This action may not be reversible" in dialogs.asked[0][1] and dialogs.asked[0][2] is True
    assert names(workspace) == ["junk.mp4", "keep.mp4"]
    dialogs.default = True
    window.apply_changes()
    assert wait_until(lambda: names(workspace) == ["keep.mp4"] and not window._operation_running)
    window.undo_last()
    assert wait_until(lambda: names(workspace) == ["junk.mp4", "keep.mp4"])


def test_delete_dialog_wording_in_permanent_mode(window, app_ctx, workspace, make_files, dialogs, server):
    app_ctx.settings.update(delete_mode="permanent")
    add_gemini(app_ctx, server, window)
    make_files("junk.mp4")
    open_folder(window, workspace, 1)
    server.queue(200, gemini_ok(json.dumps({"actions": [{"type": "delete", "source": "junk.mp4"}]})))
    generate(window, "delete the junk")
    window.apply_changes()
    assert "deleted permanently" in dialogs.asked[-1][1]


# ---------------------------------------------------------------- AI path
def test_ai_rule_answer_with_gemini(window, app_ctx, workspace, make_files, dialogs, server):
    add_gemini(app_ctx, server, window)
    make_files(*SAMPLE)
    open_folder(window, workspace, 3)
    answer = {"actions": [{"type": "numbering", "start": 1, "width": 2, "position": "replace", "template": "Blue Bloods - Episode {n}"}], "summary": "Rename to Blue Bloods - Episode NN"}
    server.queue(200, gemini_ok(json.dumps(answer)))
    generate(window, "Clean these filenames and rename them sequentially as Blue Bloods Episode 01, 02 and 03. Keep the .mp4 extension.")
    assert len(server.requests) == 1 and "Gemini / gemini-test ✓" in window.provider_label.text()
    assert "Completed · Model: Gemini / gemini-test · Requests: 1 · Fallbacks: 0 · Files: 3" in window.preview_panel.info.text()
    assert "Rename to Blue Bloods" in window.preview_panel.info.text()
    window.apply_changes()
    assert wait_until(lambda: names(workspace) == [f"Blue Bloods - Episode 0{i}.mp4" for i in (1, 2, 3)])
    assert window.ctx.db.list_commands()[0].command.startswith("Clean these filenames")  # recorded in history


def test_ai_explicit_renames_with_openai(window, app_ctx, workspace, make_files, server):
    openai_model = add_openai(app_ctx, server, window)
    make_files(*SAMPLE)
    open_folder(window, workspace, 3)
    actions = [{"type": "rename", "source": s, "target": f"Blue Bloods - Episode 0{i}.mp4"} for i, s in enumerate(SAMPLE, 1)]
    server.queue(200, openai_ok(json.dumps({"actions": actions})))
    window.command_panel.set_model_choice(openai_model.id)
    generate(window, "give them proper titles")
    assert "OpenAI / gpt-test ✓" in window.provider_label.text() and window.preview_panel.apply_count() == 3
    assert server.requests[0]["path"] == "/chat/completions"


def test_missing_key_gives_actionable_error(window, workspace, make_files, dialogs):
    make_files("a.mp4")
    open_folder(window, workspace, 1)
    generate(window, "make them look professional")
    assert dialogs.warnings and "API key" in dialogs.warnings[-1]
    assert window.preview_panel.plan is None and not window.command_panel._busy


def test_api_failure_message_and_state_reset(window, app_ctx, workspace, make_files, dialogs, server):
    add_gemini(app_ctx, server, window)
    make_files("a.mp4")
    open_folder(window, workspace, 1)
    server.queue(429, {"error": {"message": "quota"}})
    generate(window, "make them look professional")
    assert "Gemini / gemini-test" in dialogs.warnings[-1] and "Rate limit" in dialogs.warnings[-1]
    assert window.command_panel.generate_button.isVisible() and not window.command_panel.cancel_button.isVisible()


def test_unsafe_ai_answer_is_blocked_and_files_untouched(window, app_ctx, workspace, make_files, dialogs, server):
    add_gemini(app_ctx, server, window)
    make_files("a.mp4")
    open_folder(window, workspace, 1)
    server.queue(200, gemini_ok(json.dumps({"actions": [{"type": "powershell", "script": "Remove-Item C:\\ -Recurse -Force"}]})))
    generate(window, "tidy")
    assert "rejected for safety" in dialogs.warnings[-1]
    assert window.preview_panel.plan is None and names(workspace) == ["a.mp4"]


def test_auto_fallback_switches_model_and_shows_it(window, app_ctx, workspace, make_files, server):
    add_gemini(app_ctx, server, window)
    add_openai(app_ctx, server, window)
    make_files("a.mp4")
    open_folder(window, workspace, 1)
    server.queue(429, {"error": {"message": "quota exceeded"}})
    server.queue(200, openai_ok(json.dumps({"actions": [{"type": "rename", "source": "a.mp4", "target": "b.mp4"}]})))
    generate(window, "complex request")
    assert "Gemini / gemini-test (HTTP 429) → OpenAI / gpt-test ✓" in window.provider_label.text()
    assert [r["path"] for r in server.requests] == ["/models/gemini-test:generateContent", "/chat/completions"]
    details = window.command_panel.ai_label.text()
    for line in ("Primary: Gemini / gemini-test", "Failed: Gemini / gemini-test (HTTP 429)", "Fallback: OpenAI / gpt-test", "Success ✓"):
        assert line in details
    assert "Fallbacks: 1" in window.preview_panel.info.text() and window.preview_panel.apply_count() == 1
    assert app_ctx.health.in_cooldown(app_ctx.registry.list()[0].id)  # the rate-limited model rests for a while


def test_manual_mode_never_falls_back(window, app_ctx, workspace, make_files, server, dialogs):
    gem = add_gemini(app_ctx, server, window)
    add_openai(app_ctx, server, window)
    make_files("a.mp4")
    open_folder(window, workspace, 1)
    window.command_panel.set_strategy(RoutingStrategy.MANUAL.value)
    window.command_panel.set_model_choice(gem.id)
    server.queue(500, {"error": {"message": "boom"}})
    generate(window, "complex request")
    assert len(server.requests) == 1 and dialogs.warnings  # never silently switched


def test_ai_label_shows_next_model_and_choice_is_remembered(window, app_ctx, server):
    assert "no usable model" in window.command_panel.ai_label.text()
    gem = add_gemini(app_ctx, server, window)
    oai = add_openai(app_ctx, server, window)
    assert window.command_panel.ai_label.text() == "AI: Gemini / gemini-test"
    assert [window.command_panel.model_combo.itemData(i) for i in range(3)] == ["auto", gem.id, oai.id]
    window.command_panel.model_combo.setCurrentIndex(2)  # user picks OpenAI
    assert window.command_panel.ai_label.text() == "AI: OpenAI / gpt-test"
    window.command_panel.strategy_combo.setCurrentIndex(window.command_panel.strategy_combo.findData("fastest"))
    saved = app_ctx.settings.reload()
    assert (saved.preferred_model_id, saved.routing_strategy) == (oai.id, "fastest")  # auto-saved
    window.command_panel.mode_combo.setCurrentIndex(window.command_panel.mode_combo.findData("offline"))
    assert "offline" in window.command_panel.ai_label.text()


def test_routing_choice_restored_on_restart(qapp, tmp_path, server, dispose):
    ctx = AppContext.create(tmp_path / "restart")
    oai = add_openai(ctx, server)
    ctx.settings.update(routing_strategy="cheapest", preferred_model_id=oai.id)
    win = MainWindow(AppContext.create(tmp_path / "restart"))
    try:
        assert win.command_panel.strategy().value == "cheapest" and win.command_panel.model_choice() == oai.id
        assert not win.welcome.isVisibleTo(win)
    finally:
        dispose(win)


def test_welcome_banner_only_without_models(window, app_ctx, server, monkeypatch):
    assert window.welcome.isVisibleTo(window)
    assert "Welcome to AI Video File Assistant" in [w.text() for w in window.welcome.findChildren(QLabel)]
    from app.ui import main_window as mw

    class FakeDialog:
        DialogCode = mw.QDialog.DialogCode

        def __init__(self, registry, *_a, **_k):
            self.saved = [add_gemini(app_ctx, server)]

        def exec(self):
            return mw.QDialog.DialogCode.Accepted

        def deleteLater(self):
            pass

    monkeypatch.setattr(mw, "ModelDialog", FakeDialog)
    window.welcome_button.click()
    assert not window.welcome.isVisibleTo(window) and window.command_panel.model_combo.count() == 2


def test_cancel_while_waiting_for_the_ai(window, app_ctx, workspace, make_files, server):
    add_gemini(app_ctx, server, window)
    make_files("a.mp4")
    open_folder(window, workspace, 1)
    server.queue(200, gemini_ok('{"actions":[{"type":"rename","source":"a.mp4","target":"late.mp4"}]}'), delay=1.2)
    window.command_panel.set_command("complex request")
    window.generate_preview()
    assert wait_until(lambda: window.command_panel.cancel_button.isVisible() or window._plan_worker is None, timeout=2)
    window.command_panel.cancel_button.click()
    assert "Cancelled" in window.status_label.text() and window.command_panel.generate_button.isVisible()
    wait_until(lambda: not window._workers, timeout=5)
    assert window.preview_panel.plan is None  # the late answer was ignored


def test_offline_only_mode_rejects_complex_command(window, workspace, make_files, dialogs):
    make_files("a.mp4")
    open_folder(window, workspace, 1)
    generate(window, "give them nice titles", processing=ProcessingMode.OFFLINE_ONLY)
    assert "needs the AI" in dialogs.warnings[-1]


# ---------------------------------------------------------------- auto apply
def test_auto_apply_applies_safe_plans_without_asking(window, app_ctx, workspace, make_files, dialogs):
    app_ctx.settings.update(auto_apply=True)
    window.settings = app_ctx.settings.load()
    make_files("a_x.mp4")
    open_folder(window, workspace, 1)
    generate(window, "Remove _x from all filenames")
    assert wait_until(lambda: names(workspace) == ["a.mp4"]) and dialogs.asked == []


def test_auto_apply_never_applies_deletes_or_conflicts(window, app_ctx, workspace, make_files, dialogs, server):
    app_ctx.settings.update(auto_apply=True)
    window.settings = app_ctx.settings.load()
    add_gemini(app_ctx, server, window)
    make_files("a.mp4", "junk.mp4")
    open_folder(window, workspace, 2)
    server.queue(200, gemini_ok(json.dumps({"actions": [{"type": "delete", "source": "junk.mp4"}]})))
    generate(window, "delete junk")
    assert names(workspace) == ["a.mp4", "junk.mp4"] and window.preview_panel.plan is not None
    window.preview_panel.cancel_button.click()
    make_files("c_1.mp4", "c_2.mp4")
    window.refresh()
    assert wait_until(lambda: window.files_panel.model.rowCount() == 4)
    window.files_panel.select_none()
    window.files_panel.check_paths(["c_1.mp4", "c_2.mp4"])
    generate(window, "Remove _1 and _2 from all filenames")
    assert "c_1.mp4" in names(workspace) and "c_2.mp4" in names(workspace)  # conflict: waits for the user


# -------------------------------------------------------------- failure / UI
def test_failed_apply_rolls_back_and_reports(window, workspace, make_files, dialogs, monkeypatch):
    import os

    make_files("a.mp4", "b.mp4")
    open_folder(window, workspace, 2)
    generate(window, "Add prefix X ")
    real = os.rename
    calls = {"n": 0}

    def flaky(src, dst, *a, **k):
        calls["n"] += 1
        if calls["n"] == 2:
            raise PermissionError(13, "locked")
        return real(src, dst, *a, **k)

    monkeypatch.setattr(os, "rename", flaky)
    window.apply_changes()
    assert wait_until(lambda: dialogs.warnings, timeout=5)
    monkeypatch.setattr(os, "rename", real)
    assert "rolled back" in dialogs.warnings[-1] and "Permission denied" in dialogs.warnings[-1]
    assert names(workspace) == ["a.mp4", "b.mp4"]
    assert wait_until(lambda: not window._operation_running)


def test_controls_are_locked_while_applying(window, workspace, make_files):
    make_files("a.mp4")
    open_folder(window, workspace, 1)
    window._set_controls_locked(True)
    assert not window.browse_button.isEnabled() and not window.preview_panel.isEnabled()
    window._set_controls_locked(False)
    assert window.browse_button.isEnabled()


def test_language_switch_translates_new_panels(window):
    set_language("bn")
    window.retranslate()
    assert window.command_panel.generate_button.text() == "প্রিভিউ তৈরি করুন"
    assert window.preview_panel.apply_button.text() == "পরিবর্তন প্রয়োগ করুন"
    set_language("en")
    window.retranslate()
    assert window.command_panel.generate_button.text() == "Generate Preview"


def test_saved_prompt_and_recent_menus(window, app_ctx):
    assert len(window.command_panel.prompts_menu.actions()) >= 5  # the five built-in prompts + Manage
    app_ctx.db.add_command("Remove 1080p from all filenames", "gemini")
    window.refresh_prompt_menus()
    recent = [a for a in window.command_panel.recent_menu.actions() if "1080p" in a.text()]
    assert recent
    recent[0].trigger()
    assert window.command_panel.command() == "Remove 1080p from all filenames"
