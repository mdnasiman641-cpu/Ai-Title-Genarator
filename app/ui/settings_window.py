"""Settings dialog: AI models & routing, file operations, appearance and general options."""

from __future__ import annotations

from dataclasses import replace
from datetime import datetime
from pathlib import Path

from PySide6.QtCore import Qt, QThreadPool, QUrl, Signal
from PySide6.QtGui import QColor, QDesktopServices
from PySide6.QtWidgets import (
    QAbstractItemView,
    QCheckBox,
    QComboBox,
    QDialog,
    QDialogButtonBox,
    QFileDialog,
    QFormLayout,
    QGridLayout,
    QGroupBox,
    QHBoxLayout,
    QHeaderView,
    QLabel,
    QLineEdit,
    QMessageBox,
    QPushButton,
    QScrollArea,
    QSpinBox,
    QTableWidget,
    QTableWidgetItem,
    QTabWidget,
    QToolButton,
    QVBoxLayout,
    QWidget,
)

from app.ai.base_provider import ConnectionResult
from app.ai.model_config import PROVIDER_TYPES, ModelConfig, validate_base_url
from app.ai.web_search import SECRET_NAME as SEARCH_SECRET
from app.config.backup import BackupError, export_settings, import_settings
from app.config.constants import MAX_BATCH_SIZE
from app.config.settings import AppSettings
from app.context import AppContext
from app.i18n import LANGUAGE_NAMES, tr
from app.ui.model_choice import AUTO, fill_model_combo, fill_strategy_combo
from app.ui.model_dialog import CAPABILITY_KEYS, ModelDialog, ModelPickerDialog
from app.ui.widgets import Retranslator, confirm_destructive, fit_to_screen
from app.utils.logger import get_log_dir
from app.utils.security import mask_key
from app.workers.ai_worker import ConnectionTestWorker
from app.workers.base_worker import BaseWorker

COLUMNS = ("models.col_provider", "models.col_model", "models.col_base_url", "models.col_status",
           "models.col_enabled", "models.col_priority", "models.col_capabilities")  # fmt: skip
COL_ENABLED = 4


def _when(timestamp: float | None) -> str:
    return datetime.fromtimestamp(timestamp).strftime("%Y-%m-%d %H:%M") if timestamp else "—"


class SettingsDialog(QDialog):
    """Edits :class:`AppSettings` and the Model Registry.

    Model configurations (and their API keys) are saved by the model form's own *Save*
    button straight into the registry / encrypted store; everything else is saved with
    this dialog's *Save* button.
    """

    saved = Signal()
    models_changed = Signal()

    def __init__(self, ctx: AppContext, parent: QWidget | None = None, icon_color: str = "#98a0b3") -> None:
        super().__init__(parent)
        self.ctx = ctx
        self._icon_color = icon_color
        self._tr = Retranslator()
        self._workers: set[BaseWorker] = set()
        self._test_results: dict[str, ConnectionResult] = {}
        self._tr.title(self, "settings.title")
        self.setMinimumSize(760, 560)
        self.setModal(True)
        self.original = ctx.settings.load()

        root = QVBoxLayout(self)
        root.setContentsMargins(16, 16, 16, 12)
        self.tabs = QTabWidget()
        root.addWidget(self.tabs, 1)
        self.tabs.addTab(self._scroll(self._build_ai_tab()), "")
        self.tabs.addTab(self._build_files_tab(), "")
        self.tabs.addTab(self._build_appearance_tab(), "")
        self.tabs.addTab(self._build_general_tab(), "")
        self.buttons = QDialogButtonBox(QDialogButtonBox.StandardButton.Save | QDialogButtonBox.StandardButton.Cancel)
        self.buttons.accepted.connect(self.save)
        self.buttons.rejected.connect(self.reject)
        root.addWidget(self.buttons)
        self._load(self.original)
        self.refresh_models()
        self.retranslate()
        fit_to_screen(self, 980, 760)

    # ------------------------------------------------------------------ helpers
    @staticmethod
    def _scroll(page: QWidget) -> QScrollArea:
        area = QScrollArea()
        area.setWidgetResizable(True)
        area.setFrameShape(QScrollArea.Shape.NoFrame)
        area.setWidget(page)
        return area

    def _label(self, key: str) -> QLabel:
        label = QLabel()
        self._tr.text(label, key)
        return label

    def _hint(self, key: str) -> QLabel:
        label = QLabel()
        label.setObjectName("Hint")
        label.setWordWrap(True)
        self._tr.text(label, key)
        return label

    def _check(self, key: str, tip_key: str | None = None) -> QCheckBox:
        box = QCheckBox()
        self._tr.text(box, key)
        if tip_key:
            self._tr.tooltip(box, tip_key)
        return box

    def _button(self, key: str, handler: object, tip_key: str | None = None) -> QPushButton:
        button = QPushButton()
        self._tr.text(button, key)
        if tip_key:
            self._tr.tooltip(button, tip_key)
        button.clicked.connect(handler)
        return button

    def _group(self, key: str) -> tuple[QGroupBox, QFormLayout]:
        group = QGroupBox()
        self._tr.bind(group.setTitle, key)
        form = QFormLayout(group)
        form.setSpacing(10)
        return group, form

    @staticmethod
    def _spin(low: int, high: int, suffix: str = "", step: int = 1) -> QSpinBox:
        spin = QSpinBox()
        spin.setRange(low, high)
        spin.setSingleStep(step)
        if suffix:
            spin.setSuffix(suffix)
        return spin

    # =================================================================== AI tab
    def _build_ai_tab(self) -> QWidget:
        page = QWidget()
        layout = QVBoxLayout(page)
        layout.setSpacing(12)
        layout.addWidget(self._build_models_group())

        routing, form = self._group("settings.routing")
        self.strategy = QComboBox()
        self.strategy.currentIndexChanged.connect(self._update_strategy_hint)
        self.preferred_model = QComboBox()
        self.preferred_model.setMinimumWidth(260)
        self.cost_aware = self._check("settings.cost_aware", "settings.cost_aware_tip")
        self.strategy_hint = QLabel()
        self.strategy_hint.setObjectName("Hint")
        self.strategy_hint.setWordWrap(True)
        form.addRow(self._label("settings.strategy"), self.strategy)
        form.addRow("", self.strategy_hint)
        form.addRow(self._label("settings.preferred_model"), self.preferred_model)
        form.addRow("", self.cost_aware)
        layout.addWidget(routing)

        requests, form = self._group("settings.request_settings")
        self.batch_size = self._spin(1, MAX_BATCH_SIZE)
        self._tr.tooltip(self.batch_size, "settings.batch_size_tip")
        self.use_offline = self._check("settings.use_offline", "settings.use_offline_tip")
        self.send_metadata = self._check("settings.send_metadata", "settings.send_metadata_tip")
        form.addRow(self._label("settings.batch_size"), self.batch_size)
        form.addRow("", self.use_offline)
        form.addRow("", self.send_metadata)
        layout.addWidget(requests)
        layout.addWidget(self._build_search_group())

        self.advanced_toggle = QToolButton()
        self.advanced_toggle.setCheckable(True)
        self.advanced_toggle.setToolButtonStyle(Qt.ToolButtonStyle.ToolButtonTextBesideIcon)
        self.advanced_toggle.setArrowType(Qt.ArrowType.RightArrow)
        self._tr.text(self.advanced_toggle, "settings.advanced")
        self.advanced_toggle.toggled.connect(self._toggle_advanced)
        layout.addWidget(self.advanced_toggle)
        self.advanced = QWidget()
        adv_layout = QVBoxLayout(self.advanced)
        adv_layout.setContentsMargins(0, 0, 0, 0)

        defaults, form = self._group("settings.provider_defaults")
        self.timeout = self._spin(5, 600, " s")
        self.default_retries = self._spin(0, 5)
        form.addRow(self._label("settings.timeout"), self.timeout)
        form.addRow(self._label("settings.default_retries"), self.default_retries)
        urls = "\n".join(f"{info.label}: {info.default_base_url}" for info in PROVIDER_TYPES.values() if info.default_base_url)
        url_label = QLabel(urls)
        url_label.setObjectName("Hint")
        url_label.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
        form.addRow(self._label("settings.default_urls"), url_label)
        form.addRow("", self._hint("settings.provider_defaults_hint"))
        adv_layout.addWidget(defaults)

        health, form = self._group("settings.health")
        self.max_fallback = self._spin(1, 10)
        self.cooldown = self._spin(10, 3600, " s", 10)
        self.cooldown_after = self._spin(1, 10)
        self.max_chars = self._spin(2_000, 500_000, "", 1_000)
        self.health_tracking = self._check("settings.health_tracking", "settings.health_tracking_tip")
        form.addRow(self._label("settings.max_fallback"), self.max_fallback)
        form.addRow(self._label("settings.cooldown"), self.cooldown)
        form.addRow(self._label("settings.cooldown_after"), self.cooldown_after)
        form.addRow(self._label("settings.max_chars"), self.max_chars)
        form.addRow("", self.health_tracking)
        form.addRow("", self._button("settings.reset_stats", self.reset_statistics))
        adv_layout.addWidget(health)
        self.advanced.hide()
        layout.addWidget(self.advanced)
        layout.addStretch(1)
        return page

    def _build_search_group(self) -> QGroupBox:
        """Google Programmable Search for the Title Generator (key encrypted like the model keys)."""
        group, form = self._group("settings.google_search")
        self.search_key_status = QLabel()
        self.search_key_input = QLineEdit()
        self.search_key_input.setEchoMode(QLineEdit.EchoMode.Password)
        self.search_key_input.setAcceptDrops(False)
        self._tr.placeholder(self.search_key_input, "settings.search_key_placeholder")
        self.search_key_remove = self._button("settings.remove_key", self.remove_search_key)
        self.search_key_remove.setObjectName("Danger")
        key_row = QHBoxLayout()
        key_row.addWidget(self.search_key_input, 1)
        key_row.addWidget(self.search_key_remove)
        self.search_engine_id = QLineEdit()
        self.search_engine_id.setMaxLength(120)
        self._tr.placeholder(self.search_engine_id, "settings.search_engine_placeholder")
        self.search_endpoint = QLineEdit()
        self.search_cache_hours = self._spin(0, 720, " h")
        clear = self._button("tg.clear_cache", self.clear_search_cache)
        cache_row = QHBoxLayout()
        cache_row.addWidget(self.search_cache_hours)
        cache_row.addWidget(clear)
        cache_row.addStretch(1)
        form.addRow(self._label("settings.api_key"), self.search_key_status)
        form.addRow("", key_row)
        form.addRow(self._label("settings.search_engine_id"), self.search_engine_id)
        form.addRow(self._label("settings.search_endpoint"), self.search_endpoint)
        form.addRow(self._label("settings.search_cache"), cache_row)
        form.addRow("", self._hint("settings.search_note"))
        self._refresh_search_key()
        return group

    def _refresh_search_key(self) -> None:
        masked = mask_key(self.ctx.settings.secrets.get(SEARCH_SECRET))
        self.search_key_status.setText(tr("settings.key_saved", masked=masked) if masked else tr("settings.key_none"))
        self.search_key_remove.setEnabled(bool(masked))

    def remove_search_key(self) -> None:
        if confirm_destructive(self, tr("models.remove_key_confirm"), tr("models.remove")):
            self.ctx.settings.secrets.delete(SEARCH_SECRET)
            self.search_key_input.clear()
            self._refresh_search_key()

    def clear_search_cache(self) -> int:
        removed = self.ctx.db.purge_search_cache(None)
        self._set_result(tr("tg.cache_cleared", count=removed), "Success")
        return removed

    def _toggle_advanced(self, shown: bool) -> None:
        self.advanced.setVisible(shown)
        self.advanced_toggle.setArrowType(Qt.ArrowType.DownArrow if shown else Qt.ArrowType.RightArrow)

    def _build_models_group(self) -> QGroupBox:
        group = QGroupBox()
        self._tr.bind(group.setTitle, "settings.ai_models")
        layout = QVBoxLayout(group)
        self.empty_hint = self._hint("models.empty")
        layout.addWidget(self.empty_hint)
        self.table = QTableWidget(0, len(COLUMNS))
        self.table.setSelectionBehavior(QAbstractItemView.SelectionBehavior.SelectRows)
        self.table.setSelectionMode(QAbstractItemView.SelectionMode.SingleSelection)
        self.table.setEditTriggers(QAbstractItemView.EditTrigger.NoEditTriggers)
        self.table.verticalHeader().setVisible(False)
        self.table.setMinimumHeight(190)
        header = self.table.horizontalHeader()
        header.setSectionResizeMode(QHeaderView.ResizeMode.ResizeToContents)
        header.setSectionResizeMode(2, QHeaderView.ResizeMode.Stretch)
        self.table.itemSelectionChanged.connect(self._on_model_selected)
        self.table.itemChanged.connect(self._on_item_changed)
        self.table.doubleClicked.connect(lambda _i: self.edit_model())
        layout.addWidget(self.table)
        self.model_details = QLabel()
        self.model_details.setObjectName("Muted")
        self.model_details.setWordWrap(True)
        self.model_details.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
        layout.addWidget(self.model_details)

        row = QHBoxLayout()
        self.add_button = self._button("models.add", self.add_model)
        self.add_button.setObjectName("Primary")
        self.edit_button = self._button("models.edit", self.edit_model)
        self.remove_button = self._button("models.remove", self.remove_model)
        self.remove_button.setObjectName("Danger")
        self.test_button = self._button("settings.test_connection", self.test_model)
        self.load_button = self._button("models.load_models", self.load_models, "models.load_tip")
        self.toggle_button = self._button("models.enable_disable", self.toggle_model)
        self.up_button = self._button("models.move_up", lambda: self.move_model(-1))
        self.down_button = self._button("models.move_down", lambda: self.move_model(1))
        buttons = QGridLayout()  # two rows of four so the dialog never needs sideways scrolling
        buttons.setHorizontalSpacing(8)
        buttons.setVerticalSpacing(8)
        for i, button in enumerate((self.add_button, self.edit_button, self.remove_button, self.test_button,
                                    self.load_button, self.toggle_button, self.up_button, self.down_button)):  # fmt: skip
            buttons.addWidget(button, i // 4, i % 4)
        buttons.setColumnStretch(4, 1)
        row.addLayout(buttons)
        layout.addLayout(row)
        self.model_result = QLabel()
        self.model_result.setWordWrap(True)
        layout.addWidget(self.model_result)
        return group

    # ============================================================ model table
    def selected_model(self) -> ModelConfig | None:
        rows = self.table.selectionModel().selectedRows() if self.table.selectionModel() else []
        if not rows:
            return None
        item = self.table.item(rows[0].row(), 0)
        return self.ctx.registry.get(str(item.data(Qt.ItemDataRole.UserRole))) if item else None

    def model_status(self, model: ModelConfig) -> tuple[str, str]:
        """``(text, style)`` for the Status column."""
        registry, health = self.ctx.registry, self.ctx.health
        if not model.enabled:
            return tr("models.status_disabled"), "Muted"
        if model.info.key_required and not registry.has_key(model.id):
            return tr("models.status_no_key"), "Danger"
        remaining = health.cooldown_remaining(model.id)
        if remaining > 0:
            return tr("models.status_cooldown", seconds=int(remaining) + 1), "Warning"
        tested = self._test_results.get(model.id)
        if tested is not None and not tested.ok:
            return tr("models.status_error"), "Danger"
        stats = health.stats(model.id)
        if tested is not None or stats.last_success:
            if stats.consecutive_failures and tested is None:
                return tr("models.status_error"), "Danger"
            return tr("models.status_online"), "Success"
        if stats.consecutive_failures:
            return tr("models.status_error"), "Danger"
        return tr("models.status_untested"), "Muted"

    def refresh_models(self, select_id: str | None = None) -> None:
        """Rebuild the table (and the preferred-model combo) from the registry."""
        current = self.selected_model()
        select_id = select_id or (current.id if current else None)
        models = self.ctx.registry.list()
        self.table.blockSignals(True)
        self.table.setRowCount(len(models))
        colors = {"Success": "#2ea043", "Danger": "#e5534b", "Warning": "#d29922"}
        for row, model in enumerate(models):
            status, style = self.model_status(model)
            caps = ", ".join(tr(CAPABILITY_KEYS[c]) for c in model.capabilities if c in CAPABILITY_KEYS)
            values = (model.provider_label, model.model_name, model.base_url or "—", status, "", str(model.priority), caps)
            for col, text in enumerate(values):
                item = QTableWidgetItem(text)
                if col == 0:
                    item.setData(Qt.ItemDataRole.UserRole, model.id)
                    item.setToolTip(model.display_name)
                if col == 3:
                    item.setToolTip(self.ctx.health.summary(model.id))
                    if style in colors:
                        item.setForeground(QColor(colors[style]))
                if col == COL_ENABLED:
                    item.setFlags(Qt.ItemFlag.ItemIsEnabled | Qt.ItemFlag.ItemIsSelectable | Qt.ItemFlag.ItemIsUserCheckable)
                    item.setCheckState(Qt.CheckState.Checked if model.enabled else Qt.CheckState.Unchecked)
                self.table.setItem(row, col, item)
        self.table.blockSignals(False)
        self.empty_hint.setVisible(not models)
        if select_id:
            for row, model in enumerate(models):
                if model.id == select_id:
                    self.table.selectRow(row)
                    break
        fill_model_combo(self.preferred_model, self.ctx.registry)
        self._on_model_selected()

    def _on_model_selected(self) -> None:
        model = self.selected_model()
        for button in (self.edit_button, self.remove_button, self.test_button, self.toggle_button,
                       self.up_button, self.down_button):  # fmt: skip
            button.setEnabled(model is not None)
        self.load_button.setEnabled(model is not None and model.info.supports_listing)
        if model is None:
            self.model_details.setText("")
            return
        stats = self.ctx.health.stats(model.id)
        lines = [
            f"{model.display_name} — {model.label}",
            self.ctx.health.summary(model.id) if stats.requests else tr("models.not_used"),
            tr("models.health_line", success=_when(stats.last_success), failure=_when(stats.last_failure),
               failures=stats.failures, consecutive=stats.consecutive_failures),
        ]  # fmt: skip
        if stats.last_latency_s is not None:
            lines.append(tr("models.last_latency", seconds=f"{stats.last_latency_s:.2f}"))
        if stats.last_error:
            lines.append(tr("models.last_error", error=stats.last_error))
        key = self.ctx.registry.masked_key(model.id)
        lines.append(tr("settings.key_saved", masked=key) if key else tr("settings.key_none"))
        self.model_details.setText("\n".join(lines))

    def _on_item_changed(self, item: QTableWidgetItem) -> None:
        if item.column() != COL_ENABLED:
            return
        model_id = self.table.item(item.row(), 0).data(Qt.ItemDataRole.UserRole)
        self.ctx.registry.set_enabled(str(model_id), item.checkState() == Qt.CheckState.Checked)
        self._models_changed(str(model_id))

    def _models_changed(self, select_id: str | None = None) -> None:
        self.refresh_models(select_id)
        self.models_changed.emit()

    def _dialog_defaults(self) -> dict[str, int]:
        return {"default_timeout": self.timeout.value(), "default_retries": self.default_retries.value()}

    def add_model(self) -> ModelDialog:
        dialog = ModelDialog(self.ctx.registry, None, self, **self._dialog_defaults())
        if dialog.exec() == QDialog.DialogCode.Accepted and dialog.saved:
            self._models_changed(dialog.saved[0].id)
        return dialog

    def edit_model(self) -> None:
        model = self.selected_model()
        if model is None:
            return
        dialog = ModelDialog(self.ctx.registry, model, self, **self._dialog_defaults())
        dialog.exec()
        if dialog.saved:
            self._test_results.pop(model.id, None)
        self._models_changed(model.id)  # also after Remove Key inside the form

    def remove_model(self) -> None:
        model = self.selected_model()
        if model is None:
            return
        if confirm_destructive(self, tr("models.remove_confirm", name=model.display_name), tr("models.remove")):
            self.ctx.registry.remove(model.id)
            self._test_results.pop(model.id, None)
            if self.preferred_model.currentData() == model.id:
                self.preferred_model.setCurrentIndex(0)
            self._models_changed()

    def toggle_model(self) -> None:
        model = self.selected_model()
        if model is not None:
            self.ctx.registry.set_enabled(model.id, not model.enabled)
            self._models_changed(model.id)

    def move_model(self, delta: int) -> None:
        model = self.selected_model()
        if model is not None:
            self.ctx.registry.move(model.id, delta)
            self._models_changed(model.id)

    def load_models(self) -> ModelPickerDialog | None:
        model = self.selected_model()
        if model is None:
            return None
        dialog = ModelPickerDialog(self.ctx.registry, model, self)
        if dialog.exec() == QDialog.DialogCode.Accepted and dialog.added:
            self._models_changed(dialog.added[0].id)
        return dialog

    def test_model(self) -> None:
        model = self.selected_model()
        if model is None:
            return
        worker = ConnectionTestWorker(model, self.ctx.registry.api_key(model.id))
        self.test_button.setEnabled(False)
        self._set_result(tr("settings.testing"), "Muted")
        worker.signals.result.connect(lambda result, mid=model.id: self._on_test_result(mid, result))
        worker.signals.error.connect(lambda msg, _e: self._set_result("✗ " + msg, "Danger"))
        worker.signals.finished.connect(lambda: self._test_finished(worker))
        self._workers.add(worker)
        QThreadPool.globalInstance().start(worker)

    def _on_test_result(self, model_id: str, result: object) -> None:
        assert isinstance(result, ConnectionResult)
        self._test_results[model_id] = result
        if result.ok:
            self.ctx.health.clear_cooldown(model_id)
        self._set_result(result.message if result.ok else "✗ " + result.message, "Success" if result.ok else "Danger")
        self.refresh_models(model_id)

    def _test_finished(self, worker: BaseWorker) -> None:
        self._workers.discard(worker)
        self.test_button.setEnabled(self.selected_model() is not None)

    def _set_result(self, text: str, style: str) -> None:
        self.model_result.setObjectName(style)
        self.model_result.setText(text)
        self.model_result.style().unpolish(self.model_result)
        self.model_result.style().polish(self.model_result)

    def reset_statistics(self) -> None:
        answer = QMessageBox.question(self, tr("app.title"), tr("settings.reset_stats_confirm"))
        if answer == QMessageBox.StandardButton.Yes:
            self.ctx.health.reset()
            self._test_results.clear()
            self.refresh_models()

    def _update_strategy_hint(self, *_args: object) -> None:
        value = self.strategy.currentData()
        self.strategy_hint.setText(tr(f"strategy.{value}_tip") if value else "")

    # ============================================================ other tabs
    def _build_files_tab(self) -> QWidget:
        page = QWidget()
        layout = QVBoxLayout(page)
        group, form = self._group("settings.file_ops")
        self.ask_before = self._check("settings.ask_before", "settings.ask_before_tip")
        self.create_history = self._check("settings.create_history", "settings.create_history_tip")
        self.prevent_overwrite = self._check("settings.prevent_overwrite", "settings.prevent_overwrite_tip")
        self.include_subfolders = self._check("settings.include_subfolders", "settings.include_subfolders_tip")
        self.auto_apply = self._check("settings.auto_apply", "settings.auto_apply_tip")
        self.auto_apply.toggled.connect(self._on_auto_apply_toggled)
        self.duplicate_policy = QComboBox()
        for policy in ("flag", "number"):
            self.duplicate_policy.addItem("", policy)
        self.delete_mode = QComboBox()
        for mode in ("trash", "permanent"):
            self.delete_mode.addItem("", mode)
        form.addRow("", self.ask_before)
        form.addRow("", self.create_history)
        form.addRow("", self.prevent_overwrite)
        form.addRow("", self.include_subfolders)
        form.addRow("", self.auto_apply)
        form.addRow(self._label("settings.duplicate_policy"), self.duplicate_policy)
        form.addRow(self._label("settings.delete_mode"), self.delete_mode)
        layout.addWidget(group)
        layout.addWidget(self._hint("settings.file_ops_note"))
        layout.addStretch(1)
        return page

    def _build_appearance_tab(self) -> QWidget:
        page = QWidget()
        layout = QVBoxLayout(page)
        group, form = self._group("settings.appearance")
        self.theme = QComboBox()
        for theme in ("system", "dark", "light"):
            self.theme.addItem("", theme)
        form.addRow(self._label("settings.theme"), self.theme)
        layout.addWidget(group)
        layout.addStretch(1)
        return page

    def _build_general_tab(self) -> QWidget:
        page = QWidget()
        layout = QVBoxLayout(page)
        group, form = self._group("settings.general")
        self.reopen_last = self._check("settings.reopen_last")
        self.language = QComboBox()
        for code, name in LANGUAGE_NAMES.items():
            self.language.addItem(name, code)
        self.default_filter = QComboBox()
        for value in ("video", "all"):
            self.default_filter.addItem("", value)
        self.read_metadata = self._check("settings.read_metadata", "settings.read_metadata_tip")
        self.ffmpeg_dir = QLineEdit()
        self.ffmpeg_dir.setPlaceholderText(tr("settings.ffmpeg_auto"))
        self.ffmpeg_dir.setToolTip(tr("settings.ffmpeg_dir_tip"))
        browse = self._button("settings.browse", self._browse_ffmpeg)
        ffmpeg_row = QHBoxLayout()
        ffmpeg_row.addWidget(self.ffmpeg_dir, 1)
        ffmpeg_row.addWidget(browse)
        form.addRow("", self.reopen_last)
        form.addRow(self._label("settings.language"), self.language)
        form.addRow(self._label("settings.default_filter"), self.default_filter)
        form.addRow("", self.read_metadata)
        form.addRow(self._label("settings.ffmpeg_dir"), ffmpeg_row)
        layout.addWidget(group)

        backup, backup_form = self._group("settings.backup")
        row = QHBoxLayout()
        row.addWidget(self._button("settings.export", self.export_backup))
        row.addWidget(self._button("settings.import", self.import_backup))
        row.addStretch(1)
        backup_form.addRow(row)
        backup_form.addRow(self._hint("settings.backup_note"))
        layout.addWidget(backup)

        logs, log_form = self._group("settings.logging")
        self.log_level = QComboBox()
        for level in ("DEBUG", "INFO", "WARNING", "ERROR"):
            self.log_level.addItem(level, level)
        open_logs = self._button(
            "settings.open_logs", lambda: QDesktopServices.openUrl(QUrl.fromLocalFile(str(get_log_dir())))
        )
        log_form.addRow(self._label("settings.log_level"), self.log_level)
        log_form.addRow("", open_logs)
        log_form.addRow("", self._hint("settings.log_note"))
        layout.addWidget(logs)
        layout.addStretch(1)
        return page

    # ------------------------------------------------------------- load / save
    def _load(self, s: AppSettings) -> None:
        self.search_engine_id.setText(s.search_engine_id)
        self.search_endpoint.setText(s.search_endpoint)
        self.search_cache_hours.setValue(s.search_cache_hours)
        fill_strategy_combo(self.strategy, s.routing_strategy)
        fill_model_combo(self.preferred_model, self.ctx.registry, s.preferred_model_id or AUTO)
        self._update_strategy_hint()
        self.cost_aware.setChecked(s.cost_aware)
        self.use_offline.setChecked(s.use_offline_parser)
        self.batch_size.setValue(s.batch_size)
        self.send_metadata.setChecked(s.send_metadata_to_ai)
        self.timeout.setValue(s.request_timeout_s)
        self.default_retries.setValue(s.default_max_retries)
        self.max_fallback.setValue(s.max_fallback_attempts)
        self.cooldown.setValue(s.cooldown_seconds)
        self.cooldown_after.setValue(s.cooldown_after_failures)
        self.max_chars.setValue(s.max_request_chars)
        self.health_tracking.setChecked(s.health_tracking)
        self.ask_before.setChecked(s.ask_before_apply)
        self.create_history.setChecked(s.create_history)
        self.prevent_overwrite.setChecked(s.prevent_overwrite)
        self.include_subfolders.setChecked(s.include_subfolders)
        self.auto_apply.blockSignals(True)
        self.auto_apply.setChecked(s.auto_apply)
        self.auto_apply.blockSignals(False)
        _select(self.duplicate_policy, s.duplicate_policy)
        _select(self.delete_mode, s.delete_mode)
        _select(self.theme, s.theme)
        self.reopen_last.setChecked(s.reopen_last_folder)
        _select(self.language, s.language)
        _select(self.default_filter, s.default_file_filter)
        self.read_metadata.setChecked(s.read_metadata)
        self.ffmpeg_dir.setText(s.ffmpeg_dir)
        _select(self.log_level, s.log_level)

    def collect(self) -> AppSettings:
        """Read the form into a new :class:`AppSettings` (not yet saved)."""
        preferred = str(self.preferred_model.currentData() or AUTO)
        return replace(
            self.original,
            routing_strategy=str(self.strategy.currentData()),
            preferred_model_id="" if preferred == AUTO else preferred,
            cost_aware=self.cost_aware.isChecked(),
            use_offline_parser=self.use_offline.isChecked(),
            batch_size=self.batch_size.value(),
            send_metadata_to_ai=self.send_metadata.isChecked(),
            request_timeout_s=self.timeout.value(),
            default_max_retries=self.default_retries.value(),
            max_fallback_attempts=self.max_fallback.value(),
            cooldown_seconds=self.cooldown.value(),
            cooldown_after_failures=self.cooldown_after.value(),
            max_request_chars=self.max_chars.value(),
            health_tracking=self.health_tracking.isChecked(),
            search_engine_id=self.search_engine_id.text().strip(),
            search_endpoint=self.search_endpoint.text().strip(),
            search_cache_hours=self.search_cache_hours.value(),
            ask_before_apply=self.ask_before.isChecked(),
            create_history=self.create_history.isChecked(),
            prevent_overwrite=self.prevent_overwrite.isChecked(),
            include_subfolders=self.include_subfolders.isChecked(),
            auto_apply=self.auto_apply.isChecked(),
            duplicate_policy=self.duplicate_policy.currentData(),
            delete_mode=self.delete_mode.currentData(),
            theme=self.theme.currentData(),
            reopen_last_folder=self.reopen_last.isChecked(),
            language=self.language.currentData(),
            default_file_filter=self.default_filter.currentData(),
            read_metadata=self.read_metadata.isChecked(),
            ffmpeg_dir=self.ffmpeg_dir.text().strip(),
            log_level=self.log_level.currentData(),
        )

    def save(self) -> None:
        """Persist the settings (models were already saved by the model form), then close."""
        problem = validate_base_url(self.search_endpoint.text().strip()) if self.search_endpoint.text().strip() else None
        if problem:
            self._set_result("✗ " + tr("settings.search_endpoint") + ": " + problem, "Danger")
            self.tabs.setCurrentIndex(0)
            return
        key = self.search_key_input.text().strip()
        if key:  # explicit Save for the sensitive value; an empty field never removes the saved key
            self.ctx.settings.secrets.set(SEARCH_SECRET, key)
            self.search_key_input.clear()
        saved = self.ctx.settings.save(self.collect())
        self.ctx.health.configure(
            cooldown_s=saved.cooldown_seconds, cooldown_after_failures=saved.cooldown_after_failures,
            enabled=saved.health_tracking,
        )  # fmt: skip
        self.saved.emit()
        self.accept()

    # ---------------------------------------------------------- backup/restore
    def export_backup(self, path: str | None = None) -> bool:
        if path is None:
            path, _ = QFileDialog.getSaveFileName(
                self, tr("settings.export"), str(Path.home() / "ai-video-file-assistant-settings.json"), "JSON (*.json)"
            )
        if not path:
            return False
        try:
            counts = export_settings(self.ctx, Path(path))
        except (OSError, BackupError) as exc:
            QMessageBox.warning(self, tr("app.title"), tr("settings.export_failed", error=str(exc)))
            return False
        self._set_result(tr("settings.exported", models=counts["models"], prompts=counts["prompts"]), "Success")
        return True

    def import_backup(self, path: str | None = None) -> bool:
        if path is None:
            path, _ = QFileDialog.getOpenFileName(self, tr("settings.import"), str(Path.home()), "JSON (*.json)")
        if not path:
            return False
        try:
            counts = import_settings(self.ctx, Path(path))
        except (OSError, BackupError) as exc:
            QMessageBox.warning(self, tr("app.title"), tr("settings.import_failed", error=str(exc)))
            return False
        self.original = self.ctx.settings.load()
        self._load(self.original)
        self._models_changed()
        self.saved.emit()
        models = counts["added"] + counts["updated"]
        self._set_result(tr("settings.imported", models=models, prompts=counts["prompts"]), "Success")
        return True

    def _browse_ffmpeg(self) -> None:
        chosen = QFileDialog.getExistingDirectory(self, tr("settings.ffmpeg_dir"), self.ffmpeg_dir.text() or "")
        if chosen:
            self.ffmpeg_dir.setText(chosen)

    def _on_auto_apply_toggled(self, checked: bool) -> None:
        if not checked:
            return
        answer = QMessageBox.warning(
            self, tr("app.title"), tr("settings.auto_apply_warning"),
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No, QMessageBox.StandardButton.No,
        )  # fmt: skip
        if answer != QMessageBox.StandardButton.Yes:
            self.auto_apply.blockSignals(True)
            self.auto_apply.setChecked(False)
            self.auto_apply.blockSignals(False)

    # ------------------------------------------------------------ translations
    def retranslate(self) -> None:
        self._tr.retranslate()
        titles = ("settings.tab_ai", "settings.tab_files", "settings.tab_appearance", "settings.tab_general")
        for index, key in enumerate(titles):
            self.tabs.setTabText(index, tr(key))
        self.table.setHorizontalHeaderLabels([tr(key) for key in COLUMNS])
        fill_strategy_combo(self.strategy)
        self._update_strategy_hint()
        _relabel(self.duplicate_policy, {p: tr(f"settings.duplicates_{p}") for p in ("flag", "number")})
        _relabel(self.delete_mode, {m: tr(f"settings.delete_{m}") for m in ("trash", "permanent")})
        _relabel(self.theme, {t: tr(f"settings.theme_{t}") for t in ("system", "dark", "light")})
        _relabel(self.default_filter, {"video": tr("filter.video"), "all": tr("filter.all")})
        self.ffmpeg_dir.setPlaceholderText(tr("settings.ffmpeg_auto"))
        self.ffmpeg_dir.setToolTip(tr("settings.ffmpeg_dir_tip"))
        self.buttons.button(QDialogButtonBox.StandardButton.Save).setText(tr("settings.save"))
        self.buttons.button(QDialogButtonBox.StandardButton.Cancel).setText(tr("common.cancel"))


def _select(combo: QComboBox, data: object) -> None:
    index = combo.findData(data)
    if index >= 0:
        combo.setCurrentIndex(index)


def _relabel(combo: QComboBox, labels: dict[str, str]) -> None:
    for index in range(combo.count()):
        combo.setItemText(index, labels.get(str(combo.itemData(index)), combo.itemText(index)))
