"""Saved Prompts and Command History dialog."""

from __future__ import annotations

from PySide6.QtCore import Qt, Signal
from PySide6.QtWidgets import (
    QComboBox,
    QDialog,
    QFormLayout,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QListWidget,
    QListWidgetItem,
    QMessageBox,
    QPlainTextEdit,
    QPushButton,
    QTabWidget,
    QVBoxLayout,
    QWidget,
)

from app.config.constants import MAX_PROMPT_CHARS
from app.context import AppContext
from app.database.models import CommandRecord, SavedPrompt
from app.i18n import tr
from app.ui.model_choice import (
    fill_model_combo,
    model_choice_label,
    resolve_model_choice,
)
from app.ui.widgets import Retranslator
from app.utils.helpers import truncate


class SavedPromptsDialog(QDialog):
    """Manage reusable prompts (name, text, preferred provider) and the recent-command history."""

    use_requested = Signal(str, str)  # command text, provider
    changed = Signal()

    def __init__(self, ctx: AppContext, parent: QWidget | None = None, *, prefill: tuple[str, str] | None = None) -> None:
        super().__init__(parent)
        self.ctx = ctx
        self._tr = Retranslator()
        self._tr.title(self, "prompts.title")
        self.resize(820, 520)
        self._current_prompt: int | None = None
        self._prompts: list[SavedPrompt] = []
        self._commands: list[CommandRecord] = []

        layout = QVBoxLayout(self)
        self.tabs = QTabWidget()
        self.tabs.addTab(self._build_prompts_tab(), "")
        self.tabs.addTab(self._build_history_tab(), "")
        layout.addWidget(self.tabs)
        close = QHBoxLayout()
        close.addStretch(1)
        self.close_button = QPushButton()
        self.close_button.clicked.connect(self.accept)
        close.addWidget(self.close_button)
        layout.addLayout(close)

        self.reload()
        if prefill is not None:
            self.new_prompt(*prefill)
        self.retranslate()

    # ================================================================== prompts
    def _build_prompts_tab(self) -> QWidget:
        page = QWidget()
        row = QHBoxLayout(page)
        left = QVBoxLayout()
        self.prompt_list = QListWidget()
        self.prompt_list.setMinimumWidth(240)
        self.prompt_list.currentRowChanged.connect(self._on_prompt_selected)
        left.addWidget(self.prompt_list, 1)
        row.addLayout(left)

        right = QVBoxLayout()
        form = QFormLayout()
        self.name_edit = QLineEdit()
        self.name_edit.setMaxLength(80)
        self.provider_combo = QComboBox()  # "Automatic" + the configured models
        fill_model_combo(self.provider_combo, self.ctx.registry, "auto")
        self.prompt_edit = QPlainTextEdit()
        self.created_label = QLabel()
        self.created_label.setObjectName("Muted")
        self._name_label, self._provider_label, self._prompt_label, self._created_label = (QLabel() for _ in range(4))
        form.addRow(self._name_label, self.name_edit)
        form.addRow(self._provider_label, self.provider_combo)
        form.addRow(self._created_label, self.created_label)
        right.addLayout(form)
        right.addWidget(self._prompt_label)
        right.addWidget(self.prompt_edit, 1)
        buttons = QHBoxLayout()
        self.new_button = QPushButton()
        self.new_button.clicked.connect(lambda: self.new_prompt())
        self.save_button = QPushButton()
        self.save_button.setObjectName("Primary")
        self.save_button.clicked.connect(self.save_prompt)
        self.delete_button = QPushButton()
        self.delete_button.setObjectName("Danger")
        self.delete_button.clicked.connect(self.delete_prompt)
        self.use_button = QPushButton()
        self.use_button.clicked.connect(self._use_prompt)
        for button in (self.new_button, self.delete_button):
            buttons.addWidget(button)
        buttons.addStretch(1)
        buttons.addWidget(self.use_button)
        buttons.addWidget(self.save_button)
        right.addLayout(buttons)
        row.addLayout(right, 1)
        return page

    def reload(self) -> None:
        """Re-read prompts and history from the database."""
        self._prompts = self.ctx.db.list_saved_prompts()
        self.prompt_list.blockSignals(True)
        self.prompt_list.clear()
        for prompt in self._prompts:
            self.prompt_list.addItem(f"⭐ {prompt.name}")
        self.prompt_list.blockSignals(False)
        self._reload_history()
        if self._current_prompt is not None:
            index = next((i for i, p in enumerate(self._prompts) if p.id == self._current_prompt), -1)
            self.prompt_list.setCurrentRow(index)
        elif self._prompts:
            self.prompt_list.setCurrentRow(0)
        else:
            self.new_prompt()

    def _on_prompt_selected(self, row: int) -> None:
        if not 0 <= row < len(self._prompts):
            return
        prompt = self._prompts[row]
        self._current_prompt = prompt.id
        self.name_edit.setText(prompt.name)
        self.prompt_edit.setPlainText(prompt.prompt)
        index = self.provider_combo.findData(resolve_model_choice(self.ctx.registry, prompt.provider))
        self.provider_combo.setCurrentIndex(max(index, 0))
        self.created_label.setText(prompt.created_at)
        self.delete_button.setEnabled(True)

    def new_prompt(self, text: str = "", provider: str = "auto") -> None:
        """Start a blank (or pre-filled) prompt in the editor."""
        self._current_prompt = None
        self.prompt_list.blockSignals(True)
        self.prompt_list.setCurrentRow(-1)
        self.prompt_list.blockSignals(False)
        self.name_edit.setText(truncate(text.strip().splitlines()[0], 40) if text.strip() else "")
        self.prompt_edit.setPlainText(text)
        self.provider_combo.setCurrentIndex(max(self.provider_combo.findData(resolve_model_choice(self.ctx.registry, provider)), 0))
        self.created_label.setText("—")
        self.delete_button.setEnabled(False)
        self.tabs.setCurrentIndex(0)
        self.name_edit.setFocus()
        self.name_edit.selectAll()

    def save_prompt(self) -> None:
        name = self.name_edit.text().strip()
        text = self.prompt_edit.toPlainText().strip()[:MAX_PROMPT_CHARS]
        if not name or not text:
            QMessageBox.warning(self, tr("prompts.title"), tr("prompts.need_name_and_text"))
            return
        provider = str(self.provider_combo.currentData() or "auto")
        if self._current_prompt is None:
            self._current_prompt = self.ctx.db.add_saved_prompt(name, text, provider)
        else:
            self.ctx.db.update_saved_prompt(self._current_prompt, name, text, provider)
        self.reload()
        self.changed.emit()

    def delete_prompt(self) -> None:
        if self._current_prompt is None:
            return
        if QMessageBox.question(self, tr("prompts.title"), tr("prompts.delete_confirm")) != QMessageBox.StandardButton.Yes:
            return
        self.ctx.db.delete_saved_prompt(self._current_prompt)
        self._current_prompt = None
        self.reload()
        self.changed.emit()

    def _use_prompt(self) -> None:
        text = self.prompt_edit.toPlainText().strip()
        if text:
            self.use_requested.emit(text, str(self.provider_combo.currentData() or "auto"))
            self.accept()

    # ================================================================== history
    def _build_history_tab(self) -> QWidget:
        page = QWidget()
        layout = QVBoxLayout(page)
        self.history_list = QListWidget()
        self.history_list.itemDoubleClicked.connect(lambda _i: self._use_command())
        layout.addWidget(self.history_list, 1)
        buttons = QHBoxLayout()
        self.h_use = QPushButton()
        self.h_use.setObjectName("Primary")
        self.h_use.clicked.connect(self._use_command)
        self.h_fav = QPushButton()
        self.h_fav.clicked.connect(self._toggle_favorite)
        self.h_save = QPushButton()
        self.h_save.clicked.connect(self._save_command_as_prompt)
        self.h_delete = QPushButton()
        self.h_delete.clicked.connect(self._delete_command)
        self.h_clear = QPushButton()
        self.h_clear.setObjectName("Danger")
        self.h_clear.clicked.connect(self._clear_history)
        buttons.addWidget(self.h_use)
        buttons.addWidget(self.h_fav)
        buttons.addWidget(self.h_save)
        buttons.addStretch(1)
        buttons.addWidget(self.h_delete)
        buttons.addWidget(self.h_clear)
        layout.addLayout(buttons)
        return page

    def _reload_history(self) -> None:
        self._commands = self.ctx.db.list_commands(500)
        self.history_list.clear()
        for record in self._commands:
            item = QListWidgetItem(("★ " if record.favorite else "") + truncate(record.command.replace("\n", " "), 110))
            item.setToolTip(f"{record.command}\n\n{record.created_at} · {model_choice_label(self.ctx.registry, record.provider)}")
            self.history_list.addItem(item)
        if self._commands:
            self.history_list.setCurrentRow(0)

    def _selected_command(self) -> CommandRecord | None:
        row = self.history_list.currentRow()
        return self._commands[row] if 0 <= row < len(self._commands) else None

    def _use_command(self) -> None:
        record = self._selected_command()
        if record is not None:
            self.use_requested.emit(record.command, record.provider)
            self.accept()

    def _toggle_favorite(self) -> None:
        record = self._selected_command()
        if record is not None:
            self.ctx.db.set_command_favorite(record.id, not record.favorite)
            self._reload_history()
            self.changed.emit()

    def _save_command_as_prompt(self) -> None:
        record = self._selected_command()
        if record is not None:
            self.new_prompt(record.command, record.provider)

    def _delete_command(self) -> None:
        record = self._selected_command()
        if record is not None:
            self.ctx.db.delete_command(record.id)
            self._reload_history()
            self.changed.emit()

    def _clear_history(self) -> None:
        if QMessageBox.question(self, tr("prompts.title"), tr("prompts.clear_confirm")) == QMessageBox.StandardButton.Yes:
            self.ctx.db.clear_commands(keep_favorites=True)
            self._reload_history()
            self.changed.emit()

    # ============================================================== translations
    def retranslate(self) -> None:
        self._tr.retranslate()
        self.tabs.setTabText(0, tr("prompts.tab_saved"))
        self.tabs.setTabText(1, tr("prompts.tab_history"))
        self._name_label.setText(tr("prompts.name"))
        self._provider_label.setText(tr("prompts.provider"))
        self._created_label.setText(tr("prompts.created"))
        self._prompt_label.setText(tr("prompts.prompt"))
        fill_model_combo(self.provider_combo, self.ctx.registry)
        for button, key in (
            (self.new_button, "prompts.new"), (self.save_button, "prompts.save"), (self.delete_button, "prompts.delete"),
            (self.use_button, "prompts.use"), (self.h_use, "prompts.use"), (self.h_fav, "prompts.favorite"),
            (self.h_save, "prompts.save_as_prompt"), (self.h_delete, "prompts.delete"), (self.h_clear, "prompts.clear"),
            (self.close_button, "common.close"),
        ):  # fmt: skip
            button.setText(tr(key))
        self.setWindowFlag(Qt.WindowType.WindowContextHelpButtonHint, False)
