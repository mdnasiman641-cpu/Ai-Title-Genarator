"""Video Content Analysis UI: settings, presets, Analyze / Cancel / Clear cache and transcript-based previews (headless Qt)."""

from __future__ import annotations

import json
import time

import pytest
from PySide6.QtWidgets import QMessageBox

from app.ai.model_config import ModelConfig
from app.ai.title_config import TitleConfig
from app.ai.title_generator import ContentReport
from app.ai.transcription import (
    AUTO_SERVICE,
    LOCAL_SERVICE,
    LocalWhisperTranscriber,
    TranscriptionError,
    TranscriptResult,
    TranscriptService,
)
from app.context import AppContext
from app.i18n import set_language
from app.ui import main_window as main_window_module
from app.ui.content_analysis import COL_TITLE, COL_TRANSCRIPT
from app.ui.main_window import MainWindow
from app.ui.preview_window import COL_NEW
from app.ui.title_panel import TitlePanel
from tests.fake_ai_server import FakeAIServer, openai_ok
from tests.helpers import wait_until
from tests.test_transcription import FakeExtractor, FakeTranscriber

pytestmark = pytest.mark.gui


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
    wait_until(lambda: not win._workers, timeout=5)
    dispose(win)


def pick(combo, value):
    combo.setCurrentIndex(combo.findData(value))


class SlowTranscriber(FakeTranscriber):
    def transcribe(self, audio, language):
        time.sleep(0.25)
        return super().transcribe(audio, language)


def fake_service(monkeypatch, ctx, transcriber=None):
    service = TranscriptService(FakeExtractor(), transcriber or FakeTranscriber(), ctx.db)
    monkeypatch.setattr(main_window_module, "make_transcript_service", lambda _ctx: service)
    return service


def load(window, workspace, *names):
    for name in names:
        (workspace / name).write_bytes(b"video-bytes-" + name.encode())
    window.load_folder(workspace)
    assert wait_until(lambda: window.files_panel.model.rowCount() == len(names))


# ====================================================================== panel
def test_settings_feed_the_config_and_presets(panel):
    content = panel.content
    pick(content.duration, "custom")
    content.custom_seconds.setValue(95)
    pick(content.language, "bn")
    content.include_filename.setChecked(True)
    content.fallback.setChecked(True)
    cfg = panel.config()
    assert (cfg.audio_seconds, cfg.transcript_language, cfg.include_filename_context, cfg.transcript_fallback) == (95, "bn", True, True)
    assert panel.save_preset("Dialogue titles")
    panel.set_config(TitleConfig())
    assert panel.config().audio_seconds == 180 and not panel.config().include_filename_context
    panel.preset_combo.setCurrentIndex(panel.preset_combo.findText("Dialogue titles"))
    assert panel.load_selected_preset()
    cfg = panel.config()
    assert (cfg.audio_seconds, cfg.transcript_language, cfg.include_filename_context, cfg.transcript_fallback) == (95, "bn", True, True)
    assert content.duration.currentData() == "custom" and content.custom_seconds.isEnabled()
    pick(content.duration, 60)
    assert panel.config().audio_seconds == 60 and not content.custom_seconds.isEnabled()


def test_content_source_mirrors_information_source(panel):
    pick(panel.content.source, "transcript")
    assert panel.source.currentData() == "transcript" and panel.config().source == "transcript"
    pick(panel.source, "search_ai")
    assert panel.content.source.currentData() == "search_ai"


def test_report_rows_label_fallback_titles(panel):
    content = panel.content
    ok = TranscriptResult("ok", "Detective: Where were you?", "en", 180, "Fake (API)")
    failed = TranscriptResult("failed", error="Rate limit or quota reached.", error_kind="rate_limit")
    content.show_reports([
        ContentReport("a.mp4", ok, "Supported: a witness was questioned", "A Witness Is Questioned About One Night", True),
        ContentReport("b.mp4", failed, "", "A Quiet Evening Turns Tense", False),
    ])  # fmt: skip
    rows = {content.results.topLevelItem(i).text(0): content.results.topLevelItem(i) for i in range(2)}
    assert rows["a.mp4"].text(COL_TITLE) == "A Witness Is Questioned About One Night"
    assert "not transcript-based" in rows["b.mp4"].text(COL_TITLE)
    children = [rows["a.mp4"].child(i).text(0) for i in range(rows["a.mp4"].childCount())]
    assert any(c.startswith("Transcript:") for c in children) and any(c.startswith("Content analysis:") for c in children)
    assert content.counts() == (1, 1, 0) and content.summary.text() == "Successful: 1 · Failed: 1 · Skipped: 0"


# ================================================================ main window
def test_service_options_never_hide_the_selection(window, ctx):
    combo = window.title_panel.content.service
    values = [combo.itemData(i) for i in range(combo.count())]
    assert values[0] == AUTO_SERVICE and (LOCAL_SERVICE in values) == LocalWhisperTranscriber.available()
    assert "OpenAI" in window.title_panel.content.backend_label.text()  # explains what is missing
    config = ctx.registry.add(ModelConfig.new("openai", "gpt-4o-mini"), "sk-test-000000000000000000")
    window.refresh_models_ui()
    assert combo.findData(config.id) >= 0
    window._on_transcription_service_changed("missing-model", "whisper-1")
    assert ctx.settings.load().transcription_service == "missing-model" and combo.currentData() == "missing-model"


def test_analyze_selected_videos_in_background(window, ctx, workspace, monkeypatch):
    service = fake_service(monkeypatch, ctx)
    load(window, workspace, "case.mp4", "clip_silent.mp4", "family.mp4", "x_broken.mp4")
    content = window.title_panel.content
    pick(content.duration, 60)
    window.analyze_videos()
    assert wait_until(lambda: window._analysis_worker is None, timeout=8)
    assert content.counts() == (2, 1, 1) and content.summary.text() == "Successful: 2 · Failed: 1 · Skipped: 1"
    assert {seconds for _name, seconds in service.extractor.calls} == {60}
    assert content.analyze_button.text() == "Analyze Selected Videos" and content.clear_button.isEnabled()
    window.analyze_videos()  # second run: transcripts come from the cache
    assert wait_until(lambda: window._analysis_worker is None, timeout=8)
    row = next(content.results.topLevelItem(i) for i in range(4) if content.results.topLevelItem(i).text(0) == "case.mp4")
    assert "cached" in row.text(COL_TRANSCRIPT)
    assert ctx.db.count_transcript_cache() == 3  # ok + ok + no audio; the unreadable file is retried next time
    window.clear_transcript_cache()
    assert ctx.db.count_transcript_cache() == 0 and "3" in window.status_label.text()


class RejectedKey(FakeTranscriber):
    def transcribe(self, audio, language):
        raise TranscriptionError("auth", "API authentication failed (HTTP 401). The service said: Incorrect API key provided. Check the API key.")


def test_failures_show_the_actual_reason(window, ctx, workspace, monkeypatch):
    fake_service(monkeypatch, ctx, RejectedKey())
    load(window, workspace, "a_broken.mp4", "case.mp4", "family.mp4")
    content = window.title_panel.content
    window.analyze_videos()
    assert wait_until(lambda: window._analysis_worker is None, timeout=8)
    rows = {content.results.topLevelItem(i).text(0): content.results.topLevelItem(i) for i in range(3)}
    assert rows["case.mp4"].text(COL_TRANSCRIPT) == "Failed — API authentication failed (HTTP 401)"
    assert rows["a_broken.mp4"].text(COL_TRANSCRIPT).startswith("Failed — Audio extraction failed")
    assert rows["case.mp4"].isExpanded() and "Incorrect API key provided" in rows["case.mp4"].child(0).text(0)
    assert content.counts() == (0, 3, 0)
    # the status bar names the reason shared by most videos, not the first file's
    assert window.status_label.text() == "Transcription finished: 0 of 3 videos. — API authentication failed (HTTP 401)"
    assert window.status_label.objectName() == "Danger"


def test_cancel_analysis(window, ctx, workspace, monkeypatch):
    fake_service(monkeypatch, ctx, SlowTranscriber())
    load(window, workspace, *(f"video{i}.mp4" for i in range(8)))
    content = window.title_panel.content
    window.analyze_videos()
    assert content.analyze_button.text() == "Cancel Analysis" and not content.clear_button.isEnabled()  # UI stays responsive
    assert wait_until(lambda: content.results.topLevelItemCount() >= 1, timeout=5)
    content.analyze_button.click()  # the same button cancels
    assert wait_until(lambda: window._analysis_worker is None, timeout=8)
    assert content.results.topLevelItemCount() < 8 and "cancel" in window.status_label.text().lower()
    assert content.analyze_button.text() == "Analyze Selected Videos"


def test_setup_problem_is_shown_not_hidden(window, workspace):
    load(window, workspace, "case.mp4")
    window.analyze_videos()  # no OpenAI model configured
    assert window._analysis_worker is None and window.status_label.objectName() == "Danger"


def test_transcript_based_preview(window, ctx, workspace, monkeypatch):
    fake_service(monkeypatch, ctx)
    load(window, workspace, "case.mp4", "clip_silent.mp4")
    answer = {"titles": [{"n": 1, "title": "A Witness's Alibi Falls Apart Over One Late-Night Detail",
                          "analysis": {"supported": ["her car was seen at the docks"], "interpretation": "", "angle": "alibi"}}]}  # fmt: skip
    with FakeAIServer() as srv:
        ctx.registry.add(ModelConfig.new("openai_compatible", "local-model", base_url=srv.base, max_retries=0))
        window.refresh_models_ui()
        tp = window.title_panel
        tp.enabled.setChecked(True)
        pick(tp.source, "transcript")
        assert tp.content.toggle.isChecked()  # choosing the transcript source opens the section
        pick(tp.capacity, "custom")
        tp.min_words.setValue(5)
        tp.max_words.setValue(14)
        srv.queue(200, openai_ok(json.dumps(answer)))
        window.command_panel.set_command("")
        window.generate_preview()
        assert wait_until(lambda: window._plan_worker is None and not window.command_panel._busy, timeout=8)
        prompt = json.loads(srv.requests[0]["body"])["messages"][1]["content"]
        assert "docks" in prompt and "case.mp4" not in prompt  # dialogue sent, filename not
    model = window.preview_panel.model
    news = [model.index(r, COL_NEW).data() for r in range(model.rowCount())]
    assert "A Witness's Alibi Falls Apart Over One Late-Night Detail.mp4" in news
    row = next(tp.content.results.topLevelItem(i) for i in range(tp.content.results.topLevelItemCount())
               if tp.content.results.topLevelItem(i).text(0) == "case.mp4")  # fmt: skip
    assert row.text(COL_TITLE).startswith("A Witness's Alibi")
