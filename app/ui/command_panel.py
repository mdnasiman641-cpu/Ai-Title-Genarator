"""The "AI Command" card: AI mode (routing strategy), model, processing mode, command text and Generate Preview."""

from __future__ import annotations

from typing import Any

from PySide6.QtCore import Qt, Signal
from PySide6.QtGui import QKeyEvent, QResizeEvent
from PySide6.QtWidgets import (
    QComboBox,
    QGridLayout,
    QHBoxLayout,
    QLabel,
    QMenu,
    QPlainTextEdit,
    QToolButton,
)

from app.ai.model_registry import ModelRegistry
from app.config.constants import MAX_PROMPT_CHARS, ProcessingMode, RoutingStrategy
from app.database.models import CommandRecord, SavedPrompt
from app.i18n import tr
from app.ui.model_choice import (
    AUTO,
    fill_model_combo,
    fill_strategy_combo,
    resolve_model_choice,
)
from app.ui.title_panel import TitlePanel
from app.ui.widgets import Card, IconBinder, install_wheel_guard, make_button
from app.utils.helpers import truncate


class CommandEdit(QPlainTextEdit):
    """Multi-line command box; Ctrl+Enter submits."""

    submitted = Signal()

    def keyPressEvent(self, event: QKeyEvent) -> None:  # noqa: N802
        if event.key() in (Qt.Key.Key_Return, Qt.Key.Key_Enter) and event.modifiers() & Qt.KeyboardModifier.ControlModifier:
            self.submitted.emit()
            return
        super().keyPressEvent(event)


NARROW_BELOW = 1000  # page width below which Saved Prompts / Recent move under the selectors


class CommandPanel(Card):
    """Where the user writes a natural-language command (English or Bangla)."""

    generate_requested = Signal()
    cancel_requested = Signal()
    saved_prompt_chosen = Signal(object)  # SavedPrompt
    recent_command_chosen = Signal(object)  # CommandRecord
    manage_prompts_requested = Signal()
    save_prompt_requested = Signal()
    routing_changed = Signal()  # AI mode or model changed by the user (auto-saved by the main window)

    def __init__(self, icons: IconBinder, registry: ModelRegistry | None = None, db: Any | None = None) -> None:
        super().__init__("command.title")
        self._icons = icons
        self._busy = False
        self._registry = registry

        self.body.setSpacing(14)
        row = QHBoxLayout()  # AI mode · model · processing
        row.setSpacing(10)
        tools = QHBoxLayout()  # Saved Prompts · Recent (moves below the selectors on narrow windows)
        tools.setSpacing(6)
        self.strategy_combo = QComboBox()
        fill_strategy_combo(self.strategy_combo, RoutingStrategy.AUTO_FALLBACK.value)
        self._tr.tooltip(self.strategy_combo, "command.strategy_tip")
        self.strategy_combo.currentIndexChanged.connect(self._on_routing_changed)
        row.addWidget(self.strategy_combo, 2)

        self.model_combo = QComboBox()
        self.model_combo.addItem(tr("models.automatic"), AUTO)
        self._tr.tooltip(self.model_combo, "command.model_tip")
        self.model_combo.currentIndexChanged.connect(self._on_routing_changed)
        row.addWidget(self.model_combo, 3)

        self.mode_combo = QComboBox()
        for mode in ProcessingMode:
            self.mode_combo.addItem("", mode.value)
        self._tr.tooltip(self.mode_combo, "command.mode_tip")
        row.addWidget(self.mode_combo, 3)
        for combo in (self.strategy_combo, self.model_combo, self.mode_combo):
            install_wheel_guard(combo)

        self.prompts_button = QToolButton()
        self.prompts_button.setObjectName("HeaderButton")
        self.prompts_button.setPopupMode(QToolButton.ToolButtonPopupMode.InstantPopup)
        self.prompts_button.setToolButtonStyle(Qt.ToolButtonStyle.ToolButtonTextBesideIcon)
        self._icons.bind(self.prompts_button.setIcon, "star", 16)
        self._tr.text(self.prompts_button, "command.saved_prompts")
        self.prompts_menu = QMenu(self)
        self.prompts_button.setMenu(self.prompts_menu)
        tools.addWidget(self.prompts_button)

        self.recent_button = QToolButton()
        self.recent_button.setObjectName("HeaderButton")
        self.recent_button.setPopupMode(QToolButton.ToolButtonPopupMode.InstantPopup)
        self.recent_button.setToolButtonStyle(Qt.ToolButtonStyle.ToolButtonTextBesideIcon)
        self._icons.bind(self.recent_button.setIcon, "history", 16)
        self._tr.text(self.recent_button, "command.recent")
        self.recent_menu = QMenu(self)
        self.recent_button.setMenu(self.recent_menu)
        tools.addWidget(self.recent_button)
        self.top_grid = QGridLayout()
        self.top_grid.setHorizontalSpacing(12)
        self.top_grid.setVerticalSpacing(8)
        self._routing_row, self._tools_row = row, tools
        self._narrow: bool | None = None
        self._place_top(narrow=False)
        self.body.addLayout(self.top_grid)

        self.title_panel: TitlePanel | None = None
        if db is not None:  # the collapsible Title Generator section
            self.title_panel = TitlePanel(db)
            self.body.addWidget(self.title_panel)

        self.edit = CommandEdit()
        self.edit.setObjectName("CommandEdit")
        self.edit.setMinimumHeight(96)
        self.edit.setMaximumHeight(200)
        self.edit.setTabChangesFocus(True)
        self.edit.submitted.connect(self.generate_requested)
        self.body.addWidget(self.edit)

        actions = QHBoxLayout()
        self.counter = QLabel()
        self.counter.setObjectName("Count")
        actions.addWidget(self.counter)
        actions.addStretch(1)
        self.save_button = make_button("command.save_prompt", self._tr, icon="star", binder=icons, tooltip_key="command.save_prompt_tip")
        self.save_button.clicked.connect(self.save_prompt_requested)
        actions.addWidget(self.save_button)
        self.cancel_button = make_button("common.cancel", self._tr, icon="x", binder=icons)
        self.cancel_button.clicked.connect(self.cancel_requested)
        self.cancel_button.hide()
        actions.addWidget(self.cancel_button)
        self.generate_button = make_button(
            "command.generate", self._tr, icon="zap", binder=icons, icon_color="#ffffff", name="Primary",
            tooltip_key="command.generate_tip",
        )  # fmt: skip
        self.generate_button.clicked.connect(self.generate_requested)
        actions.addWidget(self.generate_button)
        self.body.addLayout(actions)

        self.ai_label = QLabel()
        self.ai_label.setObjectName("Muted")
        self.ai_label.setWordWrap(True)
        self.body.addWidget(self.ai_label)

        self.edit.textChanged.connect(self._update_counter)
        self.retranslate()

    # ----------------------------------------------------------------- layout
    def _place_top(self, *, narrow: bool) -> None:
        """Selectors and the Saved Prompts / Recent buttons share one row, or two rows on narrow windows."""
        if narrow == self._narrow:
            return
        self._narrow = narrow
        for layout in (self._routing_row, self._tools_row):
            self.top_grid.removeItem(layout)
        self.top_grid.addLayout(self._routing_row, 0, 0)
        if narrow:
            self.top_grid.addLayout(self._tools_row, 1, 0, Qt.AlignmentFlag.AlignRight)
        else:
            self.top_grid.addLayout(self._tools_row, 0, 1)

    def set_available_width(self, width: int) -> None:
        """Called by the main window with the page width (our own width can be held up by the one-row layout)."""
        self._available = width
        self._place_top(narrow=width < NARROW_BELOW)

    def resizeEvent(self, event: QResizeEvent) -> None:  # noqa: N802
        super().resizeEvent(event)
        if self._available is None:  # stand-alone use: follow our own width
            self._place_top(narrow=self.width() < NARROW_BELOW)

    _available: int | None = None

    # --------------------------------------------------------------- accessors
    def command(self) -> str:
        return self.edit.toPlainText().strip()[:MAX_PROMPT_CHARS]

    def set_command(self, text: str) -> None:
        self.edit.setPlainText(text)
        self.edit.setFocus()

    def strategy(self) -> RoutingStrategy:
        return RoutingStrategy(self.strategy_combo.currentData())

    def set_strategy(self, value: str) -> None:
        index = self.strategy_combo.findData(value)
        if index >= 0:
            self.strategy_combo.blockSignals(True)
            self.strategy_combo.setCurrentIndex(index)
            self.strategy_combo.blockSignals(False)

    def model_choice(self) -> str:
        """``auto`` or the id of the chosen model configuration."""
        return str(self.model_combo.currentData() or AUTO)

    def set_model_choice(self, value: str | None) -> None:
        """Select a model by id (legacy ``gemini`` / ``openai`` values map to the first such model)."""
        choice = resolve_model_choice(self._registry, value) if self._registry is not None else AUTO
        index = self.model_combo.findData(choice)
        self.model_combo.blockSignals(True)
        self.model_combo.setCurrentIndex(max(index, 0))
        self.model_combo.blockSignals(False)

    def refresh_models(self) -> None:
        """Rebuild the Model combo from the registry (keeps the selection when it still exists)."""
        if self._registry is not None:
            fill_model_combo(self.model_combo, self._registry)

    def set_ai_text(self, text: str) -> None:
        """Show which model will be / was used (``AI: Gemini / gemini-2.5-flash``)."""
        self.ai_label.setText(text)
        self.ai_label.setVisible(bool(text))

    def _on_routing_changed(self, *_args: object) -> None:
        self.routing_changed.emit()

    def processing(self) -> ProcessingMode:
        return ProcessingMode(self.mode_combo.currentData())

    def set_busy(self, busy: bool) -> None:
        """While planning: swap Generate for Cancel and lock the inputs."""
        self._busy = busy
        self.generate_button.setVisible(not busy)
        self.cancel_button.setVisible(busy)
        for widget in (self.strategy_combo, self.model_combo, self.mode_combo, self.edit, self.title_panel):
            if widget is not None:
                widget.setEnabled(not busy)

    # ------------------------------------------------------------------ menus
    def set_prompts(self, prompts: list[SavedPrompt], recent: list[CommandRecord]) -> None:
        """Rebuild the Saved Prompts and Recent Commands drop-down menus."""
        self.prompts_menu.clear()
        for prompt in prompts:
            action = self.prompts_menu.addAction(f"⭐ {prompt.name}")
            action.setToolTip(truncate(prompt.prompt, 200))
            action.triggered.connect(lambda _c=False, p=prompt: self.saved_prompt_chosen.emit(p))
        if prompts:
            self.prompts_menu.addSeparator()
        self.prompts_menu.addAction(tr("command.manage")).triggered.connect(self.manage_prompts_requested)

        self.recent_menu.clear()
        for record in recent[:12]:
            label = ("★ " if record.favorite else "") + truncate(record.command.replace("\n", " "), 70)
            action = self.recent_menu.addAction(label)
            action.setToolTip(record.command)
            action.triggered.connect(lambda _c=False, r=record: self.recent_command_chosen.emit(r))
        if not recent:
            empty = self.recent_menu.addAction(tr("command.no_recent"))
            empty.setEnabled(False)
        else:
            self.recent_menu.addSeparator()
        self.recent_menu.addAction(tr("command.manage")).triggered.connect(self.manage_prompts_requested)

    # --------------------------------------------------------------- bookkeeping
    def _update_counter(self) -> None:
        self.counter.setText(f"{len(self.edit.toPlainText())} / {MAX_PROMPT_CHARS}")

    def retranslate(self) -> None:
        super().retranslate()
        fill_strategy_combo(self.strategy_combo)
        if self._registry is not None:
            fill_model_combo(self.model_combo, self._registry)
        else:
            self.model_combo.setItemText(0, tr("models.automatic"))
        modes = {"auto": tr("command.mode_auto"), "ai": tr("command.mode_ai"), "offline": tr("command.mode_offline")}
        for index in range(self.mode_combo.count()):
            self.mode_combo.setItemText(index, modes[str(self.mode_combo.itemData(index))])
        self.edit.setPlaceholderText(tr("command.placeholder") + "\n\n" + tr("command.example"))
        if self.title_panel is not None:
            self.title_panel.retranslate()
        self._update_counter()

