"""Add / edit one AI model configuration (provider, base URL, key, model, priority, capabilities ...)."""

from __future__ import annotations

from PySide6.QtCore import Qt, QThreadPool
from PySide6.QtWidgets import (
    QAbstractItemView,
    QCheckBox,
    QComboBox,
    QDialog,
    QDoubleSpinBox,
    QFormLayout,
    QGridLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QListWidget,
    QListWidgetItem,
    QPushButton,
    QSpinBox,
    QVBoxLayout,
    QWidget,
)

from app.ai.base_provider import ConnectionResult
from app.ai.model_config import PROVIDER_TYPES, ModelConfig, normalize_base_url
from app.ai.model_registry import ModelRegistry, RegistryError
from app.config.constants import SUGGESTED_MODELS, Capability, ProviderType
from app.i18n import tr
from app.ui.widgets import confirm_destructive, fit_to_screen
from app.workers.ai_worker import ConnectionTestWorker, LoadModelsWorker
from app.workers.base_worker import BaseWorker

CAPABILITY_KEYS = {
    Capability.TEXT.value: "models.cap_text",
    Capability.JSON.value: "models.cap_json",
    Capability.LONG_CONTEXT.value: "models.cap_long",
    Capability.FAST.value: "models.cap_fast",
    Capability.VISION.value: "models.cap_vision",
}


def _optional_int(spin: QSpinBox) -> int | None:
    return spin.value() or None


def _optional_float(spin: QDoubleSpinBox) -> float | None:
    return spin.value() if spin.value() > 0 else None


class ModelDialog(QDialog):
    """Form for one :class:`ModelConfig`. Saves to the registry on *Save* (immediately persisted)."""

    def __init__(
        self,
        registry: ModelRegistry,
        config: ModelConfig | None = None,
        parent: QWidget | None = None,
        *,
        default_timeout: int = 60,
        default_retries: int = 1,
    ) -> None:
        super().__init__(parent)
        self.registry = registry
        self.original = config
        self.saved: list[ModelConfig] = []
        self._workers: set[BaseWorker] = set()
        self._replacing_key = config is None or not registry.has_key(config.id)
        self.setWindowTitle(tr("models.edit_title") if config else tr("models.add_title"))
        self.setModal(True)

        root = QVBoxLayout(self)
        form = QFormLayout()
        form.setSpacing(9)
        self.provider_type = QComboBox()
        for value, info in PROVIDER_TYPES.items():
            self.provider_type.addItem(info.label, value)
        self.provider_type.currentIndexChanged.connect(self._on_type_changed)
        form.addRow(tr("models.provider_type"), self.provider_type)

        self.display_name = QLineEdit()
        self.display_name.setPlaceholderText(tr("models.display_placeholder"))
        form.addRow(tr("models.display_name"), self.display_name)

        self.base_url = QLineEdit()
        self.base_url.setPlaceholderText("https://api.example.com/v1")
        form.addRow(tr("models.base_url"), self.base_url)

        form.addRow(tr("settings.api_key"), self._build_key_area())

        model_row = QHBoxLayout()
        self.model_name = QComboBox()
        self.model_name.setEditable(True)
        self.model_name.setMinimumWidth(260)
        self.model_name.lineEdit().setPlaceholderText(tr("models.model_placeholder"))
        self.load_button = QPushButton(tr("models.load_models"))
        self.load_button.clicked.connect(self.load_models)
        model_row.addWidget(self.model_name, 1)
        model_row.addWidget(self.load_button)
        form.addRow(tr("models.model"), model_row)

        self.available = QListWidget()
        self.available.setSelectionMode(QAbstractItemView.SelectionMode.NoSelection)
        self.available.setMaximumHeight(150)
        self.available.hide()
        self.available.itemChanged.connect(self._on_available_changed)
        self.available_label = QLabel(tr("models.available"))
        self.available_label.hide()
        form.addRow(self.available_label, self.available)
        self.load_status = QLabel()
        self.load_status.setObjectName("Hint")
        self.load_status.setWordWrap(True)
        form.addRow("", self.load_status)

        self.priority = QSpinBox()
        self.priority.setRange(1, 999)
        self.timeout = QSpinBox()
        self.timeout.setRange(1, 600)
        self.timeout.setSuffix(" s")
        self.retries = QSpinBox()
        self.retries.setRange(0, 5)
        self.enabled = QCheckBox(tr("models.enabled"))
        numbers = QHBoxLayout()
        for label, widget in ((tr("models.priority"), self.priority), (tr("models.timeout"), self.timeout),
                              (tr("models.retries"), self.retries)):  # fmt: skip
            numbers.addWidget(QLabel(label))
            numbers.addWidget(widget)
        numbers.addStretch(1)
        numbers.addWidget(self.enabled)
        form.addRow("", numbers)

        caps = QGroupBox(tr("models.capabilities"))
        caps_layout = QHBoxLayout(caps)
        self.capability_boxes: dict[str, QCheckBox] = {}
        for value, key in CAPABILITY_KEYS.items():
            box = QCheckBox(tr(key))
            self.capability_boxes[value] = box
            caps_layout.addWidget(box)
        caps_layout.addStretch(1)
        root.addLayout(form)
        root.addWidget(caps)
        root.addWidget(self._build_limits())

        self.result_label = QLabel()
        self.result_label.setWordWrap(True)
        self.result_label.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
        root.addWidget(self.result_label)

        buttons = QHBoxLayout()
        self.test_button = QPushButton(tr("settings.test_connection"))
        self.test_button.clicked.connect(self.test_connection)
        buttons.addWidget(self.test_button)
        buttons.addStretch(1)
        cancel = QPushButton(tr("common.cancel"))
        cancel.clicked.connect(self.reject)
        self.save_button = QPushButton(tr("settings.save"))
        self.save_button.setObjectName("Primary")
        self.save_button.setDefault(True)
        self.save_button.clicked.connect(self.save)
        buttons.addWidget(cancel)
        buttons.addWidget(self.save_button)
        root.addLayout(buttons)

        self._load(config, default_timeout, default_retries)
        fit_to_screen(self, 720, self.sizeHint().height())  # never narrower than its layout needs

    # =================================================================== build
    def _build_key_area(self) -> QWidget:
        area = QWidget()
        layout = QVBoxLayout(area)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(6)
        self.key_status = QLabel()
        self.key_status.setObjectName("Success")
        layout.addWidget(self.key_status)
        row = QHBoxLayout()
        self.keep_key_button = QPushButton(tr("models.keep_key"))
        self.keep_key_button.setCheckable(True)
        self.replace_key_button = QPushButton(tr("models.replace_key"))
        self.replace_key_button.setCheckable(True)
        self.remove_key_button = QPushButton(tr("settings.remove_key"))
        self.remove_key_button.setObjectName("Danger")
        self.keep_key_button.clicked.connect(lambda: self._set_replacing(False))
        self.replace_key_button.clicked.connect(lambda: self._set_replacing(True))
        self.remove_key_button.clicked.connect(self.remove_key)
        for button in (self.keep_key_button, self.replace_key_button, self.remove_key_button):
            row.addWidget(button)
        row.addStretch(1)
        layout.addLayout(row)
        input_row = QHBoxLayout()
        self.key_input = QLineEdit()
        self.key_input.setEchoMode(QLineEdit.EchoMode.Password)
        self.key_input.setPlaceholderText(tr("settings.key_placeholder"))
        self.key_input.setAcceptDrops(False)
        self.show_key = QPushButton(tr("models.show"))
        self.show_key.setCheckable(True)
        self.show_key.toggled.connect(
            lambda on: self.key_input.setEchoMode(QLineEdit.EchoMode.Normal if on else QLineEdit.EchoMode.Password)
        )
        input_row.addWidget(self.key_input, 1)
        input_row.addWidget(self.show_key)
        layout.addLayout(input_row)
        return area

    def _build_limits(self) -> QGroupBox:
        box = QGroupBox(tr("models.limits"))
        grid = QGridLayout(box)
        self.rpm = QSpinBox()
        self.rpd = QSpinBox()
        for spin in (self.rpm, self.rpd):
            spin.setRange(0, 1_000_000)
            spin.setSpecialValueText(tr("models.unlimited"))
        self.cost_in = QDoubleSpinBox()
        self.cost_out = QDoubleSpinBox()
        for spin in (self.cost_in, self.cost_out):
            spin.setRange(0, 10_000)
            spin.setDecimals(4)
            spin.setSpecialValueText(tr("models.unknown"))
            spin.setPrefix("$ ")
        grid.addWidget(QLabel(tr("models.rpm")), 0, 0)
        grid.addWidget(self.rpm, 0, 1)
        grid.addWidget(QLabel(tr("models.rpd")), 0, 2)
        grid.addWidget(self.rpd, 0, 3)
        grid.addWidget(QLabel(tr("models.cost_in")), 1, 0)
        grid.addWidget(self.cost_in, 1, 1)
        grid.addWidget(QLabel(tr("models.cost_out")), 1, 2)
        grid.addWidget(self.cost_out, 1, 3)
        hint = QLabel(tr("models.cost_hint"))
        hint.setObjectName("Hint")
        hint.setWordWrap(True)
        grid.addWidget(hint, 2, 0, 1, 4)
        return box

    # ================================================================== state
    def _load(self, config: ModelConfig | None, default_timeout: int, default_retries: int) -> None:
        if config is None:
            self.provider_type.setCurrentIndex(0)
            self._on_type_changed()
            self.priority.setValue(len(self.registry.list()) + 1)
            self.timeout.setValue(default_timeout)
            self.retries.setValue(default_retries)
            self.enabled.setChecked(True)
            for value in (Capability.TEXT.value, Capability.JSON.value):
                self.capability_boxes[value].setChecked(True)
        else:
            self.provider_type.blockSignals(True)
            self.provider_type.setCurrentIndex(max(0, self.provider_type.findData(config.provider_type)))
            self.provider_type.blockSignals(False)
            self._fill_suggestions(config.provider_type)
            self.display_name.setText(config.display_name)
            self.base_url.setText(config.base_url)
            self.model_name.setCurrentText(config.model_name)
            self.priority.setValue(config.priority)
            self.timeout.setValue(config.timeout_s)
            self.retries.setValue(config.max_retries)
            self.enabled.setChecked(config.enabled)
            for value, box in self.capability_boxes.items():
                box.setChecked(config.has(value))
            self.rpm.setValue(config.rpm_limit or 0)
            self.rpd.setValue(config.rpd_limit or 0)
            self.cost_in.setValue(config.cost_input_per_mtok or 0)
            self.cost_out.setValue(config.cost_output_per_mtok or 0)
        self._refresh_key_area()

    def _on_type_changed(self, *_args: object) -> None:
        value = str(self.provider_type.currentData())
        info = PROVIDER_TYPES[value]
        current = normalize_base_url(self.base_url.text())
        defaults = {i.default_base_url for i in PROVIDER_TYPES.values() if i.default_base_url}
        if not current or current in defaults:
            self.base_url.setText(info.default_base_url)
        self._fill_suggestions(value)
        self.load_button.setEnabled(info.supports_listing)
        self.load_status.setText("" if info.supports_listing else tr("models.listing_unsupported"))
        self._refresh_key_area()

    def _fill_suggestions(self, provider_type: str) -> None:
        text = self.model_name.currentText()
        self.model_name.clear()
        self.model_name.addItems(list(SUGGESTED_MODELS.get(provider_type, ())))
        self.model_name.setCurrentText(text)

    def _has_saved_key(self) -> bool:
        return self.original is not None and self.registry.has_key(self.original.id)

    def _refresh_key_area(self) -> None:
        saved = self._has_saved_key()
        required = PROVIDER_TYPES[str(self.provider_type.currentData())].key_required
        if saved:
            self.key_status.setText(tr("models.key_saved", masked=self.registry.masked_key(self.original.id)))  # type: ignore[union-attr]
        else:
            self.key_status.setText(tr("models.key_required") if required else tr("models.key_optional"))
        self.key_status.setObjectName("Success" if saved else "Muted")
        self.key_status.style().unpolish(self.key_status)
        self.key_status.style().polish(self.key_status)
        for button in (self.keep_key_button, self.replace_key_button, self.remove_key_button):
            button.setVisible(saved)
        self.keep_key_button.setChecked(saved and not self._replacing_key)
        self.replace_key_button.setChecked(saved and self._replacing_key)
        replacing = self._replacing_key or not saved
        self.key_input.setVisible(replacing)
        self.show_key.setVisible(replacing)

    def _set_replacing(self, replacing: bool) -> None:
        self._replacing_key = replacing
        if not replacing:
            self.key_input.clear()
        self._refresh_key_area()

    def remove_key(self) -> None:
        if self.original is None:
            return
        if confirm_destructive(self, tr("models.remove_key_confirm"), tr("models.remove")):
            self.registry.remove_key(self.original.id)
            self._replacing_key = True
            self._refresh_key_area()
            self._set_result(tr("models.key_removed"), "Muted")

    def typed_key(self) -> str:
        return self.key_input.text().strip() if self.key_input.isVisible() or self._replacing_key else ""

    def effective_key(self) -> str | None:
        """Key for Test / Load Models: the newly typed one, else the saved one."""
        typed = self.typed_key()
        if typed:
            return typed
        if self.original is not None and not self._replacing_key:
            return self.registry.api_key(self.original.id)
        return None

    def collect(self) -> ModelConfig:
        """Build a :class:`ModelConfig` from the form (not saved yet)."""
        caps = tuple(value for value, box in self.capability_boxes.items() if box.isChecked())
        base: dict[str, object] = {
            "provider_type": str(self.provider_type.currentData()),
            "display_name": self.display_name.text().strip(),
            "base_url": normalize_base_url(self.base_url.text()),
            "model_name": self.model_name.currentText().strip(),
            "enabled": self.enabled.isChecked(),
            "priority": self.priority.value(),
            "capabilities": caps,
            "timeout_s": self.timeout.value(),
            "max_retries": self.retries.value(),
            "rpm_limit": _optional_int(self.rpm),
            "rpd_limit": _optional_int(self.rpd),
            "cost_input_per_mtok": _optional_float(self.cost_in),
            "cost_output_per_mtok": _optional_float(self.cost_out),
        }
        if self.original is not None:
            return self.original.with_changes(**{**base, "display_name": base["display_name"] or self.original.display_name})
        return ModelConfig.new(
            base.pop("provider_type"), base.pop("model_name"), display_name=base.pop("display_name"),
            base_url=base.pop("base_url"), **base,
        )  # fmt: skip

    # ================================================================ actions
    def save(self) -> None:
        """Validate and persist (the key only if a new one was typed - never wiped by an empty field)."""
        config = self.collect()
        info = PROVIDER_TYPES.get(config.provider_type)
        key = self.typed_key() or None
        problems = config.validate()
        if info and info.key_required and not key and not self._has_saved_key():
            problems.append(tr("models.key_required"))
        if problems:
            self._set_result("✗ " + problems[0], "Danger")
            return
        try:
            if self.original is None:
                self.saved = [self.registry.add(config, key)]
            else:
                self.saved = [self.registry.update(config, key)]
            for extra in self._extra_selected(config.model_name):  # several models picked in "Load Models"
                clone = ModelConfig.new(
                    config.provider_type, extra, display_name=f"{config.provider_label} {extra}", base_url=config.base_url,
                    priority=len(self.registry.list()) + 1, capabilities=config.capabilities, timeout_s=config.timeout_s,
                    max_retries=config.max_retries, enabled=config.enabled,
                )  # fmt: skip
                self.saved.append(self.registry.add(clone, key or self.effective_key()))
        except RegistryError as exc:
            self._set_result("✗ " + str(exc), "Danger")
            return
        self.key_input.clear()
        self.accept()

    def _extra_selected(self, primary: str) -> list[str]:
        names = []
        for row in range(self.available.count()):
            item = self.available.item(row)
            if item.checkState() == Qt.CheckState.Checked and item.text() != primary:
                names.append(item.text())
        return names

    def load_models(self) -> None:
        worker = LoadModelsWorker(str(self.provider_type.currentData()), self.base_url.text(), self.effective_key())
        self.load_button.setEnabled(False)
        self.load_status.setText(tr("models.loading"))
        worker.signals.result.connect(self._on_models_loaded)
        worker.signals.error.connect(lambda msg, _e: self.load_status.setText(msg))
        worker.signals.finished.connect(lambda: self._finished(worker, self.load_button))
        self._workers.add(worker)
        QThreadPool.globalInstance().start(worker)

    def _on_models_loaded(self, models: object) -> None:
        names = [str(m) for m in models] if isinstance(models, list) else []
        current = self.model_name.currentText()
        self.available.blockSignals(True)
        self.available.clear()
        for name in names:
            item = QListWidgetItem(name)
            item.setFlags(item.flags() | Qt.ItemFlag.ItemIsUserCheckable)
            item.setCheckState(Qt.CheckState.Checked if name == current else Qt.CheckState.Unchecked)
            self.available.addItem(item)
        self.available.blockSignals(False)
        self.available.setVisible(bool(names))
        self.available_label.setVisible(bool(names))
        self.model_name.clear()
        self.model_name.addItems(names)
        self.model_name.setCurrentText(current or (names[0] if names else ""))
        self.load_status.setText(tr("models.loaded", count=len(names)))

    def _on_available_changed(self, item: QListWidgetItem) -> None:
        if item.checkState() == Qt.CheckState.Checked and not self.model_name.currentText().strip():
            self.model_name.setCurrentText(item.text())

    def checked_models(self) -> list[str]:
        return [self.available.item(r).text() for r in range(self.available.count())
                if self.available.item(r).checkState() == Qt.CheckState.Checked]  # fmt: skip

    def test_connection(self) -> None:
        config = self.collect()
        worker = ConnectionTestWorker(config, self.effective_key())
        self.test_button.setEnabled(False)
        self._set_result(tr("settings.testing"), "Muted")
        worker.signals.result.connect(self._on_test_result)
        worker.signals.error.connect(lambda msg, _e: self._set_result("✗ " + msg, "Danger"))
        worker.signals.finished.connect(lambda: self._finished(worker, self.test_button))
        self._workers.add(worker)
        QThreadPool.globalInstance().start(worker)

    def _on_test_result(self, result: object) -> None:
        assert isinstance(result, ConnectionResult)
        self._set_result(result.message if result.ok else "✗ " + result.message, "Success" if result.ok else "Danger")

    def _finished(self, worker: BaseWorker, button: QPushButton) -> None:
        self._workers.discard(worker)
        if button is self.load_button:
            button.setEnabled(PROVIDER_TYPES[str(self.provider_type.currentData())].supports_listing)
        else:
            button.setEnabled(True)

    def _set_result(self, text: str, style: str) -> None:
        self.result_label.setObjectName(style)
        self.result_label.setText(text)
        self.result_label.style().unpolish(self.result_label)
        self.result_label.style().polish(self.result_label)


class ModelPickerDialog(QDialog):
    """Lists the models a configured endpoint offers; the checked ones become new configurations.

    The new configurations copy the source's provider type, base URL, settings and API key.
    """

    def __init__(self, registry: ModelRegistry, source: ModelConfig, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.registry = registry
        self.source = source
        self.added: list[ModelConfig] = []
        self._workers: set[BaseWorker] = set()
        self.setWindowTitle(tr("models.load_title", name=source.display_name))
        self.setMinimumSize(460, 420)
        self.setModal(True)
        layout = QVBoxLayout(self)
        self.status = QLabel(tr("models.loading"))
        self.status.setWordWrap(True)
        layout.addWidget(self.status)
        self.filter = QLineEdit()
        self.filter.setPlaceholderText(tr("models.filter"))
        self.filter.textChanged.connect(self._apply_filter)
        layout.addWidget(self.filter)
        self.list = QListWidget()
        layout.addWidget(self.list, 1)
        buttons = QHBoxLayout()
        buttons.addStretch(1)
        cancel = QPushButton(tr("common.cancel"))
        cancel.clicked.connect(self.reject)
        self.add_button = QPushButton(tr("models.add_selected"))
        self.add_button.setObjectName("Primary")
        self.add_button.setEnabled(False)
        self.add_button.clicked.connect(self.add_selected)
        buttons.addWidget(cancel)
        buttons.addWidget(self.add_button)
        layout.addLayout(buttons)
        self.start()

    def start(self) -> None:
        worker = LoadModelsWorker(self.source.provider_type, self.source.base_url, self.registry.api_key(self.source.id))
        worker.signals.result.connect(self.show_models)
        worker.signals.error.connect(lambda msg, _e: self.status.setText(msg))
        worker.signals.finished.connect(lambda: self._workers.discard(worker))
        self._workers.add(worker)
        QThreadPool.globalInstance().start(worker)

    def show_models(self, models: object) -> None:
        names = [str(m) for m in models] if isinstance(models, list) else []
        existing = {m.model_name for m in self.registry.list() if m.base_url == self.source.base_url}
        self.list.clear()
        for name in names:
            item = QListWidgetItem(name)
            if name in existing:
                item.setFlags(item.flags() & ~Qt.ItemFlag.ItemIsEnabled)
                item.setText(f"{name}  {tr('models.already_added')}")
                item.setData(Qt.ItemDataRole.UserRole, None)
            else:
                item.setFlags(item.flags() | Qt.ItemFlag.ItemIsUserCheckable)
                item.setCheckState(Qt.CheckState.Unchecked)
                item.setData(Qt.ItemDataRole.UserRole, name)
            self.list.addItem(item)
        self.status.setText(tr("models.loaded", count=len(names)))
        self.add_button.setEnabled(bool(names))

    def _apply_filter(self, text: str) -> None:
        needle = text.strip().lower()
        for row in range(self.list.count()):
            item = self.list.item(row)
            item.setHidden(bool(needle) and needle not in item.text().lower())

    def checked(self) -> list[str]:
        out = []
        for row in range(self.list.count()):
            item = self.list.item(row)
            name = item.data(Qt.ItemDataRole.UserRole)
            if name and item.checkState() == Qt.CheckState.Checked:
                out.append(str(name))
        return out

    def add_selected(self) -> None:
        key = self.registry.api_key(self.source.id)
        src = self.source
        for name in self.checked():
            config = ModelConfig.new(
                src.provider_type, name, display_name=f"{src.provider_label} {name}", base_url=src.base_url,
                priority=len(self.registry.list()) + 1, capabilities=src.capabilities, timeout_s=src.timeout_s,
                max_retries=src.max_retries,
            )  # fmt: skip
            try:
                self.added.append(self.registry.add(config, key))
            except RegistryError as exc:
                self.status.setText("✗ " + str(exc))
                return
        self.accept()


__all__ = ["ModelDialog", "ModelPickerDialog", "ProviderType"]
