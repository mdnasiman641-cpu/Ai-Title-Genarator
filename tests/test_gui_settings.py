"""Settings dialog, AI Models table and the model form (headless Qt, fake local AI server)."""

from __future__ import annotations

import pytest
from PySide6.QtCore import Qt
from PySide6.QtWidgets import QLabel, QLineEdit, QMessageBox

from app.ai.model_config import ModelConfig
from app.context import AppContext
from app.i18n import set_language
from app.ui import model_dialog, settings_window
from app.ui.model_dialog import ModelDialog, ModelPickerDialog
from app.ui.settings_window import COL_ENABLED, SettingsDialog
from app.ui.theme import apply_theme
from tests.fake_ai_server import FakeAIServer, openai_ok
from tests.helpers import wait_until

pytestmark = pytest.mark.gui
KEY = "sk-test-ABCDEFGHIJKLMNOP1234"
GEMINI_KEY = "AIzaSyKEY-1234567890-abcdefghWXYZ"


@pytest.fixture
def ctx(tmp_path):
    return AppContext.create(tmp_path / "ctx")


@pytest.fixture
def dialog(qapp, ctx, dispose):
    apply_theme(qapp, "dark")
    dlg = SettingsDialog(ctx)
    yield dlg
    dispose(dlg)
    set_language("en")


@pytest.fixture
def form(qapp, ctx, dispose):
    made = []

    def _make(config=None):
        dlg = ModelDialog(ctx.registry, config)
        made.append(dlg)
        return dlg

    yield _make
    dispose(*made)


def visible_texts(widget) -> list[str]:
    return [w.text() for w in widget.findChildren(QLabel)] + [w.text() for w in widget.findChildren(QLineEdit) if w.echoMode() == QLineEdit.EchoMode.Normal]


def status_of(dialog, row: int) -> str:
    return dialog.table.item(row, 3).text()


# ================================================================ model form
def test_new_model_form_defaults(form):
    dlg = form()
    assert dlg.provider_type.currentData() == "gemini"
    assert dlg.base_url.text() == "https://generativelanguage.googleapis.com/v1beta"
    assert dlg.key_input.echoMode() == QLineEdit.EchoMode.Password
    assert dlg.capability_boxes["text"].isChecked() and dlg.capability_boxes["json"].isChecked()
    assert dlg.enabled.isChecked() and dlg.priority.value() == 1
    dlg.show_key.setChecked(True)
    assert dlg.key_input.echoMode() == QLineEdit.EchoMode.Normal


def test_provider_type_switch_sets_default_url_and_listing(form):
    dlg = form()
    dlg.provider_type.setCurrentIndex(dlg.provider_type.findData("openai"))
    assert dlg.base_url.text() == "https://api.openai.com/v1"
    dlg.provider_type.setCurrentIndex(dlg.provider_type.findData("custom"))
    assert dlg.base_url.text() == "" and not dlg.load_button.isEnabled()
    assert dlg.load_status.text() == "Model listing is not available for this provider. Enter the model ID manually."
    dlg.base_url.setText("http://localhost:8080/v1")
    dlg.provider_type.setCurrentIndex(dlg.provider_type.findData("openai_compatible"))
    assert dlg.base_url.text() == "http://localhost:8080/v1"  # a URL the user typed is never overwritten


def test_saving_a_model_stores_key_encrypted(form, ctx):
    dlg = form()
    dlg.provider_type.setCurrentIndex(dlg.provider_type.findData("openai"))
    dlg.model_name.setCurrentText("gpt-4o-mini")
    dlg.display_name.setText("My OpenAI")
    dlg.capability_boxes["fast"].setChecked(True)
    dlg.timeout.setValue(25)
    dlg.key_input.setText(KEY)
    dlg.save()
    assert dlg.result() == ModelDialog.DialogCode.Accepted
    model = ctx.registry.list()[0]
    assert (model.provider_type, model.model_name, model.display_name, model.timeout_s) == ("openai", "gpt-4o-mini", "My OpenAI", 25)
    assert model.capabilities == ("text", "json", "fast")
    assert ctx.registry.api_key(model.id) == KEY
    assert "ABCDEFGH" not in (ctx.db.get_raw(f"secret.model.{model.id}") or "")
    assert dlg.key_input.text() == ""  # typed key cleared after saving


def test_validation_blocks_bad_input(form, ctx):
    dlg = form()
    dlg.model_name.setCurrentText("gemini-2.5-flash")
    dlg.save()  # Gemini needs a key
    assert len(ctx.registry) == 0 and dlg.result_label.text().startswith("✗")
    dlg.provider_type.setCurrentIndex(dlg.provider_type.findData("openai_compatible"))
    dlg.base_url.setText("http://public.example.com/v1")
    dlg.model_name.setCurrentText("m")
    dlg.save()
    assert len(ctx.registry) == 0 and "https" in dlg.result_label.text().lower()


def test_edit_keeps_existing_key_unless_replaced(form, ctx):
    model = ctx.registry.add(ModelConfig.new("openai", "gpt-4o-mini"), KEY)
    dlg = form(model)
    assert dlg.keep_key_button.isChecked() and not dlg.key_input.isVisibleTo(dlg)
    assert dlg.key_status.text() == "Saved key: ************1234"
    assert not any(KEY in t for t in visible_texts(dlg))
    dlg.model_name.setCurrentText("gpt-4.1-mini")
    dlg.save()
    assert ctx.registry.get(model.id).model_name == "gpt-4.1-mini" and ctx.registry.api_key(model.id) == KEY

    dlg2 = form(ctx.registry.get(model.id))
    dlg2.replace_key_button.click()
    assert dlg2.key_input.isVisibleTo(dlg2)
    dlg2.save()  # Replace Key with an empty field must not wipe the key
    assert ctx.registry.api_key(model.id) == KEY

    dlg3 = form(ctx.registry.get(model.id))
    dlg3.replace_key_button.click()
    dlg3.key_input.setText("sk-brand-new-key-000000000NEW9")
    dlg3.save()
    assert ctx.registry.masked_key(model.id) == "************NEW9"


def test_remove_key_needs_confirmation_and_keeps_model(form, ctx, monkeypatch):
    model = ctx.registry.add(ModelConfig.new("gemini", "gemini-2.5-flash"), GEMINI_KEY)
    other = ctx.registry.add(ModelConfig.new("openai", "gpt-4o-mini"), KEY)
    dlg = form(model)
    asked = []
    monkeypatch.setattr(model_dialog, "confirm_destructive", lambda _p, text, ok: asked.append((text, ok)) or False)
    dlg.remove_key_button.click()
    assert asked == [("Remove this saved API key from this computer?", "Remove")] and ctx.registry.has_key(model.id)
    monkeypatch.setattr(model_dialog, "confirm_destructive", lambda *a: True)
    dlg.remove_key_button.click()
    assert not ctx.registry.has_key(model.id) and ctx.registry.get(model.id) is not None
    assert ctx.registry.api_key(other.id) == KEY
    assert dlg.key_status.text() == "This provider needs an API key." and dlg.key_input.isVisibleTo(dlg)


def test_load_models_and_multi_select_creates_configs(form, ctx):
    with FakeAIServer() as srv:
        dlg = form()
        dlg.provider_type.setCurrentIndex(dlg.provider_type.findData("openai_compatible"))
        dlg.base_url.setText(srv.base + "/v1")
        srv.queue(200, {"data": [{"id": "llama3.1:8b"}, {"id": "qwen2.5:14b"}, {"id": "mistral:7b"}]})
        dlg.load_button.click()
        assert wait_until(lambda: dlg.available.count() == 3)
        assert srv.requests[0]["path"] == "/v1/models"
        assert dlg.model_name.currentText() == "llama3.1:8b"
        for row in range(3):
            if dlg.available.item(row).text() in ("llama3.1:8b", "qwen2.5:14b"):
                dlg.available.item(row).setCheckState(Qt.CheckState.Checked)
        dlg.save()
    assert sorted(m.model_name for m in ctx.registry.list()) == ["llama3.1:8b", "qwen2.5:14b"]
    assert {m.base_url for m in ctx.registry.list()} == {srv.base + "/v1"}


def test_load_models_unsupported_message(form):
    with FakeAIServer() as srv:
        dlg = form()
        dlg.provider_type.setCurrentIndex(dlg.provider_type.findData("openai_compatible"))
        dlg.base_url.setText(srv.base)
        srv.queue(404, {"error": {"message": "not found"}})
        dlg.load_button.click()
        assert wait_until(lambda: "manually" in dlg.load_status.text())
        assert dlg.load_status.text() == "Model listing is not available for this provider. Enter the model ID manually."


def test_test_connection_in_form(form):
    with FakeAIServer() as srv:
        dlg = form()
        dlg.provider_type.setCurrentIndex(dlg.provider_type.findData("openai"))
        dlg.base_url.setText(srv.base)
        dlg.model_name.setCurrentText("gpt-4o-mini")
        dlg.key_input.setText(KEY)
        srv.queue(200, {"data": [{"id": "gpt-4o-mini"}]})
        srv.queue(200, openai_ok('{"ok": true}'))
        dlg.test_button.click()
        assert wait_until(lambda: "Connection successful" in dlg.result_label.text())
        assert "gpt-4o-mini" in dlg.result_label.text() and "seconds" in dlg.result_label.text()
        assert srv.requests[0]["headers"]["authorization"] == f"Bearer {KEY}"
        srv.queue(401, {"error": {"message": "Incorrect API key provided"}})
        dlg.test_button.click()
        assert wait_until(lambda: dlg.result_label.text().startswith("✗"))
        assert "Authentication" in dlg.result_label.text() and KEY not in dlg.result_label.text()


# ============================================================== models table
def test_empty_state_and_table_columns(dialog):
    assert dialog.table.rowCount() == 0 and dialog.empty_hint.isVisibleTo(dialog)
    headers = [dialog.table.horizontalHeaderItem(i).text() for i in range(dialog.table.columnCount())]
    assert headers == ["Provider", "Model", "Base URL", "Status", "Enabled", "Priority", "Capabilities"]
    assert not dialog.edit_button.isEnabled() and dialog.add_button.isEnabled()


def test_table_shows_models_and_status(dialog, ctx):
    gem = ctx.registry.add(ModelConfig.new("gemini", "gemini-2.5-flash"))  # no key
    oai = ctx.registry.add(ModelConfig.new("openai", "gpt-4o-mini"), KEY)
    off = ctx.registry.add(ModelConfig.new("openai_compatible", "llama3", base_url="http://localhost:11434/v1", enabled=False))
    ctx.health.record_failure(oai.id, "rate_limit", "HTTP 429")
    dialog.refresh_models()
    assert dialog.table.rowCount() == 3 and not dialog.empty_hint.isVisibleTo(dialog)
    assert [dialog.table.item(r, 1).text() for r in range(3)] == ["gemini-2.5-flash", "gpt-4o-mini", "llama3"]
    assert status_of(dialog, 0) == "No key"
    assert status_of(dialog, 1).startswith("Cooldown")
    assert status_of(dialog, 2) == "Disabled"
    assert dialog.table.item(2, COL_ENABLED).checkState() == Qt.CheckState.Unchecked
    assert dialog.table.item(0, 6).text() == "Text, JSON"
    ctx.health.record_success(gem.id, 1.8)
    ctx.registry.set_key(gem.id, GEMINI_KEY)
    dialog.refresh_models()
    assert status_of(dialog, 0) == "Online"
    dialog.table.selectRow(0)
    assert "✓ 100% success" in dialog.model_details.text() and "avg 1.8s" in dialog.model_details.text()
    assert "************WXYZ" in dialog.model_details.text() and GEMINI_KEY not in dialog.model_details.text()
    assert off.id  # keep reference


def test_enable_checkbox_move_and_remove(dialog, ctx, monkeypatch):
    a = ctx.registry.add(ModelConfig.new("openai", "a"), KEY)
    b = ctx.registry.add(ModelConfig.new("openai", "b"), KEY)
    changed = []
    dialog.models_changed.connect(lambda: changed.append(1))
    dialog.refresh_models()
    dialog.table.item(0, COL_ENABLED).setCheckState(Qt.CheckState.Unchecked)
    assert not ctx.registry.get(a.id).enabled and changed
    dialog.table.selectRow(1)
    dialog.up_button.click()
    assert [m.id for m in ctx.registry.list()] == [b.id, a.id]
    assert dialog.selected_model().id == b.id  # selection follows the moved model
    dialog.down_button.click()
    assert [m.id for m in ctx.registry.list()] == [a.id, b.id]
    dialog.table.selectRow(0)
    dialog.toggle_button.click()
    assert ctx.registry.get(a.id).enabled
    monkeypatch.setattr(settings_window, "confirm_destructive", lambda *a: False)
    dialog.remove_button.click()
    assert len(ctx.registry) == 2
    monkeypatch.setattr(settings_window, "confirm_destructive", lambda *a: True)
    dialog.table.selectRow(0)
    dialog.remove_button.click()
    assert [m.id for m in ctx.registry.list()] == [b.id] and ctx.registry.api_key(a.id) is None


def test_test_button_updates_status(dialog, ctx):
    with FakeAIServer() as srv:
        model = ctx.registry.add(ModelConfig.new("openai_compatible", "m", base_url=srv.base))
        dialog.refresh_models()
        dialog.table.selectRow(0)
        assert status_of(dialog, 0) == "Untested"
        srv.queue(200, {"data": [{"id": "m"}]})
        srv.queue(200, openai_ok('{"ok": true}'))
        dialog.test_button.click()
        assert wait_until(lambda: status_of(dialog, 0) == "Online")
        srv.queue(500, {"error": {"message": "boom"}})
        dialog.test_button.click()
        assert wait_until(lambda: status_of(dialog, 0) == "Error")
    assert model.id


def test_model_picker_adds_selected_models(qapp, ctx, dispose):
    with FakeAIServer() as srv:
        source = ctx.registry.add(ModelConfig.new("openai", "gpt-4o-mini", base_url=srv.base), KEY)
        srv.queue(200, {"data": [{"id": "gpt-4o-mini"}, {"id": "gpt-4.1"}, {"id": "o3-mini"}]})
        picker = ModelPickerDialog(ctx.registry, source)
        assert wait_until(lambda: picker.list.count() == 3)
        assert srv.requests[0]["headers"]["authorization"] == f"Bearer {KEY}"
        existing = next(picker.list.item(r) for r in range(3) if picker.list.item(r).text().startswith("gpt-4o-mini"))
        assert not existing.flags() & Qt.ItemFlag.ItemIsEnabled  # already configured
        for row in range(3):
            item = picker.list.item(row)
            if item.flags() & Qt.ItemFlag.ItemIsUserCheckable:
                item.setCheckState(Qt.CheckState.Checked)
        picker.filter.setText("o3")
        assert sum(not picker.list.item(r).isHidden() for r in range(3)) == 1
        picker.add_selected()
        dispose(picker)
    names = [m.model_name for m in ctx.registry.list()]
    assert names == ["gpt-4o-mini", "gpt-4.1", "o3-mini"]
    assert all(ctx.registry.api_key(m.id) == KEY for m in ctx.registry.list())  # key copied, still encrypted


# ======================================================== routing / advanced
def test_save_roundtrips_all_options(dialog, ctx):
    model = ctx.registry.add(ModelConfig.new("openai", "gpt-4o-mini"), KEY)
    dialog.refresh_models()
    dialog.strategy.setCurrentIndex(dialog.strategy.findData("cheapest"))
    dialog.preferred_model.setCurrentIndex(dialog.preferred_model.findData(model.id))
    dialog.cost_aware.setChecked(True)
    dialog.batch_size.setValue(25)
    dialog.advanced_toggle.setChecked(True)
    assert dialog.advanced.isVisibleTo(dialog)
    dialog.timeout.setValue(45)
    dialog.default_retries.setValue(2)
    dialog.max_fallback.setValue(3)
    dialog.cooldown.setValue(300)
    dialog.cooldown_after.setValue(4)
    dialog.max_chars.setValue(50_000)
    dialog.health_tracking.setChecked(False)
    dialog.theme.setCurrentIndex(dialog.theme.findData("light"))
    dialog.language.setCurrentIndex(dialog.language.findData("bn"))
    dialog.delete_mode.setCurrentIndex(dialog.delete_mode.findData("permanent"))
    dialog.duplicate_policy.setCurrentIndex(dialog.duplicate_policy.findData("number"))
    dialog.prevent_overwrite.setChecked(False)
    dialog.save()
    s = ctx.settings.reload()
    assert (s.routing_strategy, s.preferred_model_id, s.cost_aware, s.batch_size) == ("cheapest", model.id, True, 25)
    assert (s.request_timeout_s, s.default_max_retries, s.max_fallback_attempts) == (45, 2, 3)
    assert (s.cooldown_seconds, s.cooldown_after_failures, s.max_request_chars, s.health_tracking) == (300, 4, 50_000, False)
    assert (s.theme, s.language) == ("light", "bn")
    assert (s.delete_mode, s.duplicate_policy, s.prevent_overwrite) == ("permanent", "number", False)
    assert s.auto_apply is False and ctx.health.enabled is False


def test_strategy_hint_explains_the_mode(dialog):
    dialog.strategy.setCurrentIndex(dialog.strategy.findData("auto_fallback"))
    assert "HTTP 429" in dialog.strategy_hint.text()
    dialog.strategy.setCurrentIndex(dialog.strategy.findData("manual"))
    assert "only" in dialog.strategy_hint.text()


def test_reset_statistics(dialog, ctx, monkeypatch):
    model = ctx.registry.add(ModelConfig.new("openai", "m"), KEY)
    ctx.health.record_success(model.id, 1.0)
    monkeypatch.setattr(QMessageBox, "question", lambda *a, **k: QMessageBox.StandardButton.Yes)
    dialog.reset_statistics()
    assert ctx.health.stats(model.id).requests == 0


def test_export_and_import_from_settings(dialog, ctx, tmp_path):
    ctx.registry.add(ModelConfig.new("openai", "gpt-4o-mini"), KEY)
    path = tmp_path / "backup.json"
    assert dialog.export_backup(str(path))
    assert KEY not in path.read_text(encoding="utf-8") and "no API keys" in dialog.model_result.text()
    other = AppContext.create(tmp_path / "other")
    second = SettingsDialog(other)
    try:
        assert second.import_backup(str(path))
        assert second.table.rowCount() == 1 and not other.registry.has_key(other.registry.list()[0].id)
    finally:
        second.deleteLater()


def test_auto_apply_requires_confirmation(dialog, monkeypatch):
    assert not dialog.auto_apply.isChecked()
    monkeypatch.setattr(QMessageBox, "warning", lambda *a, **k: QMessageBox.StandardButton.No)
    dialog.auto_apply.setChecked(True)
    assert not dialog.auto_apply.isChecked()
    monkeypatch.setattr(QMessageBox, "warning", lambda *a, **k: QMessageBox.StandardButton.Yes)
    dialog.auto_apply.setChecked(True)
    assert dialog.auto_apply.isChecked()


def test_dialog_translates_to_bangla(dialog):
    set_language("bn")
    dialog.retranslate()
    assert dialog.tabs.tabText(0) == "AI" and dialog.tabs.tabText(1) == "ফাইল অপারেশন"
    assert dialog.test_button.text() == "সংযোগ পরীক্ষা" and dialog.add_button.text() == "মডেল যোগ করুন"
    assert dialog.table.horizontalHeaderItem(0).text() == "প্রোভাইডার"
