"""The collapsible "Video Content Analysis" part of the Title Generator (audio → transcript → analysis)."""

from __future__ import annotations

from typing import Any

from PySide6.QtCore import Qt, Signal
from PySide6.QtWidgets import (
    QAbstractItemView,
    QCheckBox,
    QComboBox,
    QFrame,
    QGridLayout,
    QHBoxLayout,
    QHeaderView,
    QLabel,
    QLineEdit,
    QPushButton,
    QSpinBox,
    QToolButton,
    QTreeWidget,
    QTreeWidgetItem,
    QVBoxLayout,
    QWidget,
)

from app.ai.title_config import AUDIO_SECONDS_RANGE, TRANSCRIPT_LANGUAGES, TitleConfig
from app.ai.transcription import DURATIONS, WHISPER_MODELS, TranscriptResult
from app.i18n import tr
from app.ui.widgets import Retranslator, install_wheel_guard

LANGUAGE_NAMES = {
    "en": "English", "bn": "বাংলা (Bangla)", "hi": "हिन्दी (Hindi)", "ur": "اردو (Urdu)", "ar": "العربية (Arabic)",
    "es": "Español", "fr": "Français", "de": "Deutsch", "it": "Italiano", "pt": "Português", "ru": "Русский",
    "tr": "Türkçe", "ja": "日本語", "ko": "한국어", "zh": "中文",
}  # fmt: skip
COL_FILE, COL_AUDIO, COL_TRANSCRIPT, COL_RANGE, COL_TITLE = range(5)
STAGES = ("waiting", "download", "load", "extract", "transcribe", "analyze", "done")
ROLE_SUGGESTION = Qt.ItemDataRole.UserRole + 1  # (rel_path, title) on a suggestion row


class ContentAnalysisPanel(QFrame):
    """Duration, language, transcription service and per-video transcript results."""

    changed = Signal()  # a Title Generator value changed (duration, language, options)
    source_chosen = Signal(str)  # the user picked a content source here (mirrors Information source)
    service_changed = Signal(str, str)  # transcription service id, model name (application settings)
    whisper_model_changed = Signal(str)  # local Whisper model (tiny / base / small / ...)
    suggestion_chosen = Signal(str, str)  # rel_path, title -> becomes that file's new name in the preview
    analyze_requested = Signal()
    cancel_requested = Signal()
    clear_cache_requested = Signal()

    def __init__(self) -> None:
        super().__init__()
        self.setObjectName("Card")
        self._tr = Retranslator()
        self._loading = False
        self._running = False
        self._rows: dict[str, QTreeWidgetItem] = {}
        outer = QVBoxLayout(self)
        outer.setContentsMargins(12, 6, 12, 10)
        outer.setSpacing(8)

        self.toggle = QToolButton()
        self.toggle.setObjectName("SectionToggle")
        self.toggle.setCheckable(True)
        self.toggle.setToolButtonStyle(Qt.ToolButtonStyle.ToolButtonTextBesideIcon)
        self.toggle.setArrowType(Qt.ArrowType.RightArrow)
        self.toggle.setCursor(Qt.CursorShape.PointingHandCursor)
        self._tr.text(self.toggle, "ca.title")
        self.toggle.toggled.connect(self.set_expanded)
        outer.addWidget(self.toggle)

        self.body = QWidget()
        body = QVBoxLayout(self.body)
        body.setContentsMargins(4, 0, 4, 4)
        body.setSpacing(12)
        body.addLayout(self._build_settings())
        body.addLayout(self._build_actions())
        self.summary = QLabel()
        self.summary.setObjectName("Muted")
        self.summary.setWordWrap(True)
        body.addWidget(self.summary)
        body.addWidget(self._build_results())
        self.body.hide()
        outer.addWidget(self.body)
        for kind in (QComboBox, QSpinBox):
            for widget in self.body.findChildren(kind):
                install_wheel_guard(widget)
        self.retranslate()

    # ================================================================ build
    def _label(self, key: str) -> QLabel:
        label = QLabel()
        label.setObjectName("FieldLabel")
        self._tr.text(label, key)
        return label

    def _build_settings(self) -> QGridLayout:
        grid = QGridLayout()
        grid.setHorizontalSpacing(16)
        grid.setVerticalSpacing(8)
        self.duration = QComboBox()
        for seconds in DURATIONS:
            self.duration.addItem("", seconds)
        self.duration.addItem("", "custom")
        self.custom_seconds = QSpinBox()
        self.custom_seconds.setRange(*AUDIO_SECONDS_RANGE)
        self.custom_seconds.setSuffix(" s")
        self.language = QComboBox()
        for code in TRANSCRIPT_LANGUAGES:
            self.language.addItem(LANGUAGE_NAMES.get(code, code), code)
        self.source = QComboBox()  # mirrors the Title Generator's Information source
        self.service = QComboBox()
        self.whisper_model = QComboBox()  # local Whisper: replaces the API model field
        self.whisper_model.setEditable(True)  # a model folder path also works
        for name, size in WHISPER_MODELS.items():
            self.whisper_model.addItem(f"{name}  (≈{size} MB)", name)
        self._tr.tooltip(self.whisper_model, "ca.whisper_model_tip")
        self.whisper_model.hide()
        self.model = QLineEdit()
        self.model.setMaxLength(120)
        self._tr.placeholder(self.model, "ca.model_ph")
        self._tr.tooltip(self.model, "ca.model_tip")
        self.include_filename = QCheckBox()
        self._tr.text(self.include_filename, "ca.include_filename")
        self.fallback = QCheckBox()
        self._tr.text(self.fallback, "ca.fallback")
        self._tr.tooltip(self.fallback, "ca.fallback_tip")

        duration_row = QHBoxLayout()
        duration_row.addWidget(self.duration, 1)
        duration_row.addWidget(self.custom_seconds)
        service_row = QHBoxLayout()
        service_row.addWidget(self.service, 3)
        service_row.addWidget(self.model, 2)
        service_row.addWidget(self.whisper_model, 2)
        cells = [
            ("ca.duration", duration_row), ("ca.language", self.language),
            ("ca.source", self.source), ("ca.service", service_row),
        ]  # fmt: skip
        for i, (key, field) in enumerate(cells):
            box = QVBoxLayout()
            box.setSpacing(6)
            box.addWidget(self._label(key))
            if isinstance(field, QWidget):
                box.addWidget(field)
            else:
                box.addLayout(field)
            grid.addLayout(box, i // 2, i % 2)
        grid.addWidget(self.include_filename, 2, 0)
        grid.addWidget(self.fallback, 2, 1)
        grid.setColumnStretch(0, 1)
        grid.setColumnStretch(1, 1)

        self.duration.currentIndexChanged.connect(self._on_changed)
        self.custom_seconds.valueChanged.connect(self._on_changed)
        self.language.currentIndexChanged.connect(self._on_changed)
        self.include_filename.toggled.connect(self._on_changed)
        self.fallback.toggled.connect(self._on_changed)
        self.source.currentIndexChanged.connect(self._on_source)
        self.service.currentIndexChanged.connect(self._on_service)
        self.model.editingFinished.connect(self._on_service)
        self.whisper_model.activated.connect(self._on_whisper_model)
        self.whisper_model.lineEdit().editingFinished.connect(self._on_whisper_model)
        return grid

    def _build_actions(self) -> QHBoxLayout:
        row = QHBoxLayout()
        row.setSpacing(10)
        self.analyze_button = QPushButton()
        self.analyze_button.setObjectName("Primary")
        self._tr.text(self.analyze_button, "ca.analyze")
        self._tr.tooltip(self.analyze_button, "ca.analyze_tip")
        self.analyze_button.clicked.connect(self._on_analyze)
        self.clear_button = QPushButton()
        self._tr.text(self.clear_button, "ca.clear_cache")
        self.clear_button.clicked.connect(self.clear_cache_requested)
        self.backend_label = QLabel()
        self.backend_label.setObjectName("Hint")
        self.backend_label.setWordWrap(True)
        self.use_button = QPushButton()  # put the selected suggestion into the preview
        self._tr.text(self.use_button, "ca.use_title")
        self._tr.tooltip(self.use_button, "ca.use_title_tip")
        self.use_button.setEnabled(False)
        self.use_button.clicked.connect(self._use_selected)
        row.addWidget(self.analyze_button)
        row.addWidget(self.clear_button)
        row.addWidget(self.use_button)
        row.addWidget(self.backend_label, 1)
        return row

    def _build_results(self) -> QTreeWidget:
        self.results = QTreeWidget()
        self.results.setColumnCount(5)
        self.results.setWordWrap(True)
        self.results.setUniformRowHeights(False)
        self.results.setSelectionMode(QAbstractItemView.SelectionMode.SingleSelection)
        self.results.setMinimumHeight(220)
        self.results.setRootIsDecorated(True)
        header = self.results.header()
        header.setStretchLastSection(True)
        header.setSectionResizeMode(COL_FILE, QHeaderView.ResizeMode.Interactive)
        self.results.setColumnWidth(COL_FILE, 220)
        for col in (COL_AUDIO, COL_TRANSCRIPT, COL_RANGE):
            header.setSectionResizeMode(col, QHeaderView.ResizeMode.ResizeToContents)
        self.results.itemSelectionChanged.connect(lambda: self.use_button.setEnabled(self._selected_suggestion() is not None))
        self.results.itemDoubleClicked.connect(lambda *_: self._use_selected())
        self.results.hide()
        return self.results

    # ================================================================ state
    def set_expanded(self, expanded: bool) -> None:
        self.toggle.blockSignals(True)
        self.toggle.setChecked(expanded)
        self.toggle.blockSignals(False)
        self.toggle.setArrowType(Qt.ArrowType.DownArrow if expanded else Qt.ArrowType.RightArrow)
        self.body.setVisible(expanded)

    def values(self) -> dict[str, Any]:
        """The Title Generator fields owned by this section."""
        choice = self.duration.currentData()
        seconds = self.custom_seconds.value() if choice == "custom" else int(choice)
        return {
            "audio_seconds": seconds,
            "transcript_language": str(self.language.currentData() or "auto"),
            "include_filename_context": self.include_filename.isChecked(),
            "transcript_fallback": self.fallback.isChecked(),
        }

    def set_values(self, cfg: TitleConfig) -> None:
        self._loading = True
        try:
            index = self.duration.findData(cfg.audio_seconds)
            self.duration.setCurrentIndex(index if index >= 0 else self.duration.findData("custom"))
            self.custom_seconds.setValue(cfg.audio_seconds)
            self.language.setCurrentIndex(max(0, self.language.findData(cfg.transcript_language)))
            self.include_filename.setChecked(cfg.include_filename_context)
            self.fallback.setChecked(cfg.transcript_fallback)
            self.set_source(cfg.source)
        finally:
            self._loading = False
        self._update()

    def set_source_options(self, items: list[tuple[str, str]], current: str) -> None:
        self.source.blockSignals(True)
        self.source.clear()
        for label, value in items:
            self.source.addItem(label, value)
        self.source.blockSignals(False)
        self.set_source(current)

    def set_source(self, value: str) -> None:
        self.source.blockSignals(True)
        self.source.setCurrentIndex(max(0, self.source.findData(value)))
        self.source.blockSignals(False)
        self._update()

    def set_services(self, options: list[tuple[str, str]], current: str, model: str, fixed_model: set[str] | None = None,
                     *, local: set[str] | None = None, whisper_model: str = "small") -> None:  # fmt: skip
        """``options``: (label, value) pairs - "auto", configured model ids, and "local" when installed.

        ``fixed_model``: services that bring their own model (Gemini), so the model field is not used.
        ``local``: services that run local Whisper (they show the Whisper model choice instead).
        """
        self._fixed_model = set(fixed_model or ())
        self._local = set(local or ())
        index = self.whisper_model.findData(whisper_model)
        if index >= 0:
            self.whisper_model.setCurrentIndex(index)
        else:
            self.whisper_model.setEditText(whisper_model)
        self.service.blockSignals(True)
        self.service.clear()
        for label, value in options:
            self.service.addItem(label, value)
        index = self.service.findData(current)
        self.service.setCurrentIndex(max(0, index))
        self.service.blockSignals(False)
        self.model.setText(model)
        self._update()

    def set_backend_text(self, text: str) -> None:
        self.backend_label.setText(text)

    def set_running(self, running: bool) -> None:
        self._running = running
        self.analyze_button.setText(tr("ca.cancel") if running else tr("ca.analyze"))
        self.analyze_button.setObjectName("Danger" if running else "Primary")
        self.analyze_button.style().unpolish(self.analyze_button)
        self.analyze_button.style().polish(self.analyze_button)
        for widget in (self.duration, self.custom_seconds, self.language, self.service, self.model, self.whisper_model, self.clear_button):
            widget.setEnabled(not running)

    def _update(self) -> None:
        self.custom_seconds.setEnabled(self.duration.currentData() == "custom" and not self._running)
        local = self.service.currentData() in getattr(self, "_local", ()) or self.service.currentData() == "local"
        own_model = self.service.currentData() in getattr(self, "_fixed_model", ())
        self.model.setVisible(not local)
        self.whisper_model.setVisible(local)
        self.model.setEnabled(not own_model and not self._running)

    # ============================================================== results
    def clear_results(self) -> None:
        self.results.clear()
        self._rows.clear()
        self.results.hide()
        self.summary.setText("")

    def _row(self, rel_path: str) -> QTreeWidgetItem:
        item = self._rows.get(rel_path)
        if item is None:
            item = QTreeWidgetItem([rel_path, "", "", "", ""])
            item.setToolTip(COL_FILE, rel_path)
            self.results.addTopLevelItem(item)
            self._rows[rel_path] = item
            self.results.show()
        return item

    def set_stage(self, rel_path: str, stage: str) -> None:
        """Live status of one video: waiting, download, load, extract, transcribe, analyze (transcript + titles)."""
        item = self._row(rel_path)
        item.setText(COL_TRANSCRIPT, tr(f"ca.stage_{stage}"))

    def mark_unfinished(self, stage: str) -> None:
        """Rows still waiting / running when the work stopped (e.g. ``cancelled``)."""
        running = {tr(f"ca.stage_{name}") for name in STAGES if name != "done"}
        for item in self._rows.values():
            if item.text(COL_TRANSCRIPT) in running:
                item.setText(COL_TRANSCRIPT, tr(f"ca.stage_{stage}"))
        self._update_summary()

    def add_result(self, rel_path: str, result: TranscriptResult) -> None:
        """Show (or update) one video's transcript outcome."""
        item = self._row(rel_path)
        audio = tr("ca.audio_none") if result.status == "no_audio" else (
            tr("ca.audio_ok") if result.status in ("ok", "empty", "insufficient") else "—")  # fmt: skip
        status_key = {"ok": "ca.status_ok", "empty": "ca.status_empty", "insufficient": "ca.status_insufficient",
                      "no_audio": "ca.status_skipped"}.get(result.status, "ca.status_failed")  # fmt: skip
        status = tr(status_key) + (f" · {tr('ca.cached')}" if result.from_cache else "")
        if result.status == "failed" and result.error:
            stage = tr("ca.failed_extract") if result.stage == "extract" else tr("ca.failed_transcribe")
            status += f" ({stage}) — {error_headline(result.error)}"  # the actual stage and reason, not just "Failed"
        item.setText(COL_AUDIO, audio)
        item.setText(COL_TRANSCRIPT, status)
        item.setToolTip(COL_TRANSCRIPT, result.error or status)
        item.setText(COL_RANGE, result.analyzed_range if result.analyzed_seconds else "—")
        item.takeChildren()
        if result.ok or result.status == "insufficient":
            header = tr("ca.transcript_header", language=result.language or "?", words=result.words)
            parent = self._child(item, tr("ca.transcript"), f"{header} {result.text}")
            for line in result.timestamped()[:200]:  # timestamps when the engine gives them
                self._child(parent, "", line)
        if result.error:
            self._child(item, tr("ca.error"), result.error)
        item.setExpanded(result.status in ("failed", "insufficient"))  # show the full explanation of failures straight away
        self._update_summary()

    def show_reports(self, reports: list[Any]) -> None:
        """After Generate Preview: analysis, validated title suggestions and the title used, per video."""
        for report in reports:
            self.add_result(report.rel_path, report.transcript)
            item = self._rows[report.rel_path]
            if report.title:
                label = report.title if report.transcript_based else f"{report.title}  ({tr('ca.not_transcript_based')})"
                item.setText(COL_TITLE, label)
                item.setToolTip(COL_TITLE, label)
            if report.analysis:
                self._child(item, tr("ca.analysis"), report.analysis)
            if getattr(report, "insufficient", ""):
                self._child(item, tr("ca.insufficient"), report.insufficient)
                item.setExpanded(True)
            suggestions = getattr(report, "suggestions", [])
            if suggestions:
                parent = self._child(item, tr("ca.suggestions"), tr("ca.suggestions_hint", count=len(suggestions)))
                for title in suggestions:
                    child = QTreeWidgetItem([title])
                    child.setData(0, ROLE_SUGGESTION, (report.rel_path, title))
                    parent.addChild(child)
                    child.setFirstColumnSpanned(True)
                parent.setExpanded(True)
                item.setExpanded(True)
                self.mark_chosen(report.rel_path, report.title)
            rejected = getattr(report, "rejected", [])
            if rejected:
                parent = self._child(item, tr("ca.rejected"), tr("ca.rejected_hint", count=len(rejected)))
                for title, reason in rejected:
                    self._child(parent, "", f"{title} — {reason}")

    def mark_chosen(self, rel_path: str, title: str) -> None:
        """Tick the suggestion that is the file's new name in the preview."""
        item = self._rows.get(rel_path)
        if item is None:
            return
        for child in self._suggestion_rows(item):
            path, text = child.data(0, ROLE_SUGGESTION)
            child.setText(0, f"✓ {text}" if text == title else text)
        if title:
            item.setText(COL_TITLE, title)

    @staticmethod
    def _suggestion_rows(item: QTreeWidgetItem) -> list[QTreeWidgetItem]:
        rows = []
        for i in range(item.childCount()):
            group = item.child(i)
            rows += [group.child(j) for j in range(group.childCount()) if group.child(j).data(0, ROLE_SUGGESTION)]
        return rows

    def suggestions_for(self, rel_path: str) -> list[str]:
        item = self._rows.get(rel_path)
        return [child.data(0, ROLE_SUGGESTION)[1] for child in self._suggestion_rows(item)] if item else []

    def _selected_suggestion(self) -> tuple[str, str] | None:
        items = self.results.selectedItems()
        data = items[0].data(0, ROLE_SUGGESTION) if items else None
        return (str(data[0]), str(data[1])) if data else None

    def _use_selected(self) -> None:
        chosen = self._selected_suggestion()
        if chosen is not None:
            self.suggestion_chosen.emit(*chosen)

    @staticmethod
    def _child(parent: QTreeWidgetItem, label: str, text: str) -> QTreeWidgetItem:
        child = QTreeWidgetItem([f"{label}: {text}" if label else text])
        child.setToolTip(0, text)
        parent.addChild(child)
        child.setFirstColumnSpanned(True)
        return child

    def counts(self) -> tuple[int, int, int]:
        """(successful, failed, skipped) among the shown rows."""
        ok = failed = skipped = 0
        for item in self._rows.values():
            text = item.text(COL_TRANSCRIPT)
            if text.startswith(tr("ca.status_ok")):
                ok += 1
            elif text.startswith(tr("ca.status_failed")):
                failed += 1
            elif any(text.startswith(tr(f"ca.stage_{stage}")) for stage in STAGES):
                continue  # still running
            else:
                skipped += 1
        return ok, failed, skipped

    def _update_summary(self) -> None:
        ok, failed, skipped = self.counts()
        self.summary.setText(tr("ca.summary", ok=ok, failed=failed, skipped=skipped))

    # ============================================================== signals
    def _on_changed(self, *_args: object) -> None:
        self._update()
        if not self._loading:
            self.changed.emit()

    def _on_source(self, *_args: object) -> None:
        if not self._loading:
            self.source_chosen.emit(str(self.source.currentData()))

    def _on_service(self, *_args: object) -> None:
        self._update()
        if not self._loading:
            self.service_changed.emit(str(self.service.currentData() or "auto"), self.model.text().strip())

    def _on_whisper_model(self, *_args: object) -> None:
        if self._loading:
            return
        index = self.whisper_model.currentIndex()
        text = self.whisper_model.currentText().strip()
        data = self.whisper_model.itemData(index) if index >= 0 and self.whisper_model.itemText(index) == text else None
        self.whisper_model_changed.emit(str(data or text.split("  (")[0].strip() or "small"))

    def _on_analyze(self) -> None:
        if self._running:
            self.cancel_requested.emit()
        else:
            self.analyze_requested.emit()

    # ========================================================== translation
    def retranslate(self) -> None:
        self._tr.retranslate()
        for i in range(self.duration.count()):
            value = self.duration.itemData(i)
            self.duration.setItemText(i, tr("ca.duration_custom") if value == "custom" else tr("ca.duration_first", minutes=_minutes(value)))
        self.language.setItemText(0, tr("ca.auto_detect"))
        self.results.setHeaderLabels([tr("ca.col_file"), tr("ca.col_audio"), tr("ca.col_transcript"), tr("ca.col_range"),
                                      tr("ca.col_title")])  # fmt: skip
        self.set_running(self._running)


def error_headline(message: str) -> str:
    """First sentence of an error, e.g. ``API authentication failed (HTTP 401)``."""
    first = message.strip().splitlines()[0] if message.strip() else ""
    return first.split(". ")[0].rstrip(".")[:100]


def _minutes(seconds: Any) -> str:
    seconds = int(seconds)
    return f"{seconds // 60}" if seconds % 60 == 0 else f"{seconds / 60:.1f}"
