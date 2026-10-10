"""Shared helpers for "Automatic + configured models" pickers and the routing-strategy combo."""

from __future__ import annotations

from PySide6.QtWidgets import QComboBox

from app.ai.model_registry import ModelRegistry
from app.config.constants import RoutingStrategy
from app.i18n import tr

AUTO = "auto"
STRATEGY_KEYS = {
    RoutingStrategy.AUTO_FALLBACK.value: "strategy.auto_fallback",
    RoutingStrategy.AUTO.value: "strategy.auto",
    RoutingStrategy.MANUAL.value: "strategy.manual",
    RoutingStrategy.CHEAPEST.value: "strategy.cheapest",
    RoutingStrategy.FASTEST.value: "strategy.fastest",
    RoutingStrategy.HIGHEST_PRIORITY.value: "strategy.highest_priority",
}


def fill_strategy_combo(combo: QComboBox, current: str | None = None) -> None:
    """(Re)fill ``combo`` with the routing strategies, keeping the selection."""
    selected = current if current is not None else combo.currentData()
    combo.blockSignals(True)
    combo.clear()
    for value, key in STRATEGY_KEYS.items():
        combo.addItem(tr(key), value)
        combo.setItemData(combo.count() - 1, tr(key + "_tip"), 3)  # Qt.ToolTipRole
    index = combo.findData(selected)
    combo.setCurrentIndex(max(index, 0))
    combo.blockSignals(False)


def resolve_model_choice(registry: ModelRegistry, value: str | None) -> str:
    """Map a stored choice to a current one: a model id, or ``auto``.

    Saved prompts and history from older versions stored ``gemini`` / ``openai``; those map
    to the first configured model of that provider type (or Automatic).
    """
    if not value or value == AUTO:
        return AUTO
    if registry.get(value) is not None:
        return value
    match = next((m for m in registry.list() if m.provider_type == value), None)
    return match.id if match else AUTO


def model_choice_label(registry: ModelRegistry, value: str | None) -> str:
    choice = resolve_model_choice(registry, value)
    model = registry.get(choice) if choice != AUTO else None
    return model.label if model else tr("models.automatic")


def fill_model_combo(combo: QComboBox, registry: ModelRegistry, current: str | None = None) -> None:
    """(Re)fill ``combo`` with *Automatic* plus every configured model (disabled ones marked)."""
    selected = resolve_model_choice(registry, current if current is not None else combo.currentData())
    combo.blockSignals(True)
    combo.clear()
    combo.addItem(tr("models.automatic"), AUTO)
    for model in registry.list():
        text = model.display_name if model.display_name else model.label
        if not model.enabled:
            text += "  " + tr("models.disabled_suffix")
        combo.addItem(text, model.id)
        combo.setItemData(combo.count() - 1, model.label, 3)
    index = combo.findData(selected)
    combo.setCurrentIndex(max(index, 0))
    combo.blockSignals(False)
