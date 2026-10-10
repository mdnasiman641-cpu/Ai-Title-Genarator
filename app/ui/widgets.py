"""Reusable UI building blocks: cards, translatable text binding, helpers."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

from PySide6.QtCore import QEvent, QObject, QSize, Qt
from PySide6.QtGui import QIcon
from PySide6.QtWidgets import (
    QApplication,
    QComboBox,
    QFrame,
    QHBoxLayout,
    QLabel,
    QLayout,
    QMessageBox,
    QPushButton,
    QVBoxLayout,
    QWidget,
)

from app.i18n import tr
from app.ui.icons import get_icon


class IconBinder:
    """Keeps icons tinted correctly across theme changes.

    ``bind(setter, name)`` paints the icon in the current muted colour (or a fixed
    ``color``); :meth:`refresh` repaints every bound icon after the theme changes.
    """

    def __init__(self, color: str) -> None:
        self.color = color
        self._items: list[tuple[Callable[[QIcon], None], str, int, str | None]] = []

    def bind(self, setter: Callable[[QIcon], None], name: str, size: int = 16, color: str | None = None) -> None:
        self._items.append((setter, name, size, color))
        setter(get_icon(name, color or self.color, size))

    def refresh(self, color: str) -> None:
        self.color = color
        for setter, name, size, fixed in self._items:
            setter(get_icon(name, fixed or color, size))


class Retranslator:
    """Remembers ``(widget, setter, key)`` bindings and re-applies them on language change."""

    def __init__(self) -> None:
        self._bindings: list[tuple[Callable[[str], None], str, dict[str, Any]]] = []

    def bind(self, setter: Callable[[str], None], key: str, **kwargs: Any) -> None:
        self._bindings.append((setter, key, kwargs))
        setter(tr(key, **kwargs))

    def text(self, widget: Any, key: str, **kwargs: Any) -> None:
        self.bind(widget.setText, key, **kwargs)

    def tooltip(self, widget: QWidget, key: str, **kwargs: Any) -> None:
        self.bind(widget.setToolTip, key, **kwargs)

    def placeholder(self, widget: Any, key: str, **kwargs: Any) -> None:
        self.bind(widget.setPlaceholderText, key, **kwargs)

    def title(self, widget: QWidget, key: str, **kwargs: Any) -> None:
        self.bind(widget.setWindowTitle, key, **kwargs)

    def retranslate(self) -> None:
        for setter, key, kwargs in self._bindings:
            setter(tr(key, **kwargs))


class Card(QFrame):
    """A rounded surface with an optional small-caps title and a right-aligned header area."""

    def __init__(self, title_key: str | None = None, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setObjectName("Card")
        self._tr = Retranslator()
        outer = QVBoxLayout(self)
        outer.setContentsMargins(18, 16, 18, 18)
        outer.setSpacing(12)
        self.header = QHBoxLayout()
        self.header.setSpacing(8)
        self.title_label = QLabel()
        self.title_label.setObjectName("CardTitle")
        self.header.addWidget(self.title_label)
        self.header.addStretch(1)
        self.body = QVBoxLayout()
        self.body.setSpacing(12)
        outer.addLayout(self.header)
        outer.addLayout(self.body, 1)
        if title_key:
            self._tr.bind(lambda t: self.title_label.setText(t.upper()), title_key)
        else:
            self.title_label.hide()

    def retranslate(self) -> None:
        self._tr.retranslate()


def make_button(
    text_key: str | None,
    retranslator: Retranslator,
    *,
    icon: str | None = None,
    binder: IconBinder | None = None,
    icon_color: str | None = None,
    name: str | None = None,
    tooltip_key: str | None = None,
    icon_size: int = 16,
) -> QPushButton:
    """Create a translatable button; ``name`` selects a QSS style (``Primary``, ``Danger``...).

    ``icon`` is an icon name from ``resources/icons``; with a ``binder`` it re-tints on theme change.
    """
    button = QPushButton()
    if name:
        button.setObjectName(name)
    if icon is not None and binder is not None:
        button.setIconSize(QSize(icon_size, icon_size))
        binder.bind(button.setIcon, icon, icon_size, icon_color)
    if text_key:
        retranslator.text(button, text_key)
    if tooltip_key:
        retranslator.tooltip(button, tooltip_key)
    button.setCursor(Qt.CursorShape.PointingHandCursor)
    return button


def clear_layout(layout: QLayout) -> None:
    """Remove and delete every widget in ``layout``."""
    while layout.count():
        item = layout.takeAt(0)
        widget = item.widget()
        if widget is not None:
            widget.deleteLater()


def confirm_destructive(parent: QWidget | None, text: str, ok_text: str) -> bool:
    """``text`` with [Cancel] / [ok_text]; Cancel is the default. True if the user confirmed."""
    box = QMessageBox(parent)
    box.setWindowTitle(tr("app.title"))
    box.setIcon(QMessageBox.Icon.Warning)
    box.setText(text)
    ok = box.addButton(ok_text, QMessageBox.ButtonRole.DestructiveRole)
    cancel = box.addButton(tr("common.cancel"), QMessageBox.ButtonRole.RejectRole)
    box.setDefaultButton(cancel)
    box.exec()
    return box.clickedButton() is ok


class _WheelGuard(QObject):
    """Lets the mouse wheel scroll the page instead of changing an unfocused combo box / spin box."""

    def eventFilter(self, obj: QObject, event: QEvent) -> bool:  # noqa: N802
        if event.type() == QEvent.Type.Wheel and isinstance(obj, QWidget) and not obj.hasFocus():
            event.ignore()
            return True
        return False


_WHEEL_GUARD: _WheelGuard | None = None


def install_wheel_guard(widget: QWidget) -> None:
    """Apply :class:`_WheelGuard` to ``widget`` (used for controls inside the scrolling page)."""
    global _WHEEL_GUARD
    if _WHEEL_GUARD is None:
        _WHEEL_GUARD = _WheelGuard()
    widget.setFocusPolicy(Qt.FocusPolicy.StrongFocus)
    widget.installEventFilter(_WHEEL_GUARD)


def fit_to_screen(widget: QWidget, width: int, height: int) -> None:
    """Open ``widget`` at a comfortable size: at least its layout's size hint, at most 90 % of the screen."""
    hint = widget.sizeHint()
    w, h = max(width, hint.width()), max(height, hint.height())
    screen = widget.screen() or QApplication.primaryScreen()
    if screen is not None:
        area = screen.availableGeometry()
        w, h = min(w, int(area.width() * 0.9)), min(h, int(area.height() * 0.9))
    widget.resize(w, h)


def make_compact(combo: QComboBox, chars: int = 6) -> None:
    """Let a combo box shrink below its longest entry on small windows (the popup still shows full text)."""
    combo.setSizeAdjustPolicy(QComboBox.SizeAdjustPolicy.AdjustToMinimumContentsLengthWithIcon)
    combo.setMinimumContentsLength(chars)
    combo.setMinimumWidth(0)

