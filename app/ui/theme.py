"""Dark/light themes built from a colour palette and one QSS template."""

from __future__ import annotations

from PySide6.QtCore import Qt
from PySide6.QtGui import QColor, QPalette
from PySide6.QtWidgets import QApplication

from app.ui.icons import clear_icon_cache
from app.utils.helpers import resource_path

PALETTES: dict[str, dict[str, str]] = {
    "dark": {
        "bg": "#13151a", "surface": "#1b1e25", "surface_alt": "#232733", "input_bg": "#161920",
        "border": "#2b303d", "border_strong": "#3a4152", "text": "#e8eaf0", "muted": "#9aa3b6",
        "disabled": "#5d6579", "accent": "#4c8dff", "accent_hover": "#6ba0ff", "accent_text": "#ffffff",
        "accent_disabled": "#2b3f66", "accent_disabled_text": "#7f8da8", "hover": "#2a2f3d",
        "selection": "#26375a", "row_alt": "#191c23", "danger": "#ff6b7a", "danger_bg": "#3a1f25",
        "warning": "#f2b84b", "success": "#3ecf8e", "scroll": "#3a4152", "scroll_hover": "#4b5469",
        "row_ok": "#173326", "row_warn": "#3a2f14", "row_bad": "#3a1f25", "row_delete": "#3a1f2c",
    },
    "light": {
        "bg": "#f3f5f9", "surface": "#ffffff", "surface_alt": "#f0f2f7", "input_bg": "#ffffff",
        "border": "#dfe3ec", "border_strong": "#c3cad8", "text": "#1c2333", "muted": "#667085",
        "disabled": "#a3abbb", "accent": "#2f6df6", "accent_hover": "#1f5ce0", "accent_text": "#ffffff",
        "accent_disabled": "#b9ccf7", "accent_disabled_text": "#ffffff", "hover": "#e8ecf4",
        "selection": "#dbe6ff", "row_alt": "#f8f9fc", "danger": "#d92d3f", "danger_bg": "#fdecee",
        "warning": "#b7791f", "success": "#16955e", "scroll": "#c3cad8", "scroll_hover": "#a9b2c4",
        "row_ok": "#e6f6ee", "row_warn": "#fdf3dc", "row_bad": "#fdecee", "row_delete": "#fbe6ef",
    },
}


def resolve_theme(name: str, app: QApplication | None = None) -> str:
    """Turn ``system``/``dark``/``light`` into ``dark`` or ``light``."""
    if name in PALETTES:
        return name
    app = app or QApplication.instance()  # type: ignore[assignment]
    if app is not None:
        try:
            scheme = app.styleHints().colorScheme()
            return "light" if scheme == Qt.ColorScheme.Light else "dark"
        except AttributeError:  # very old Qt
            pass
    return "dark"


def build_stylesheet(palette: dict[str, str]) -> str:
    """Fill the QSS template with palette colours."""
    template = resource_path("styles", "app.qss").read_text(encoding="utf-8")
    values = dict(palette)
    values["icons"] = resource_path("icons").as_posix()
    for key, value in values.items():
        template = template.replace(f"@{key}@", value)
    return template


def apply_theme(app: QApplication, name: str) -> dict[str, str]:
    """Apply a theme application-wide and return the palette in use."""
    resolved = resolve_theme(name, app)
    palette = PALETTES[resolved]
    app.setStyle("Fusion")  # consistent look; QSS on top of the native style is unreliable
    qpalette = QPalette()
    qpalette.setColor(QPalette.ColorRole.Window, QColor(palette["bg"]))
    qpalette.setColor(QPalette.ColorRole.WindowText, QColor(palette["text"]))
    qpalette.setColor(QPalette.ColorRole.Base, QColor(palette["input_bg"]))
    qpalette.setColor(QPalette.ColorRole.AlternateBase, QColor(palette["row_alt"]))
    qpalette.setColor(QPalette.ColorRole.Text, QColor(palette["text"]))
    qpalette.setColor(QPalette.ColorRole.Button, QColor(palette["surface_alt"]))
    qpalette.setColor(QPalette.ColorRole.ButtonText, QColor(palette["text"]))
    qpalette.setColor(QPalette.ColorRole.Highlight, QColor(palette["accent"]))
    qpalette.setColor(QPalette.ColorRole.HighlightedText, QColor(palette["accent_text"]))
    qpalette.setColor(QPalette.ColorRole.ToolTipBase, QColor(palette["surface_alt"]))
    qpalette.setColor(QPalette.ColorRole.ToolTipText, QColor(palette["text"]))
    qpalette.setColor(QPalette.ColorRole.PlaceholderText, QColor(palette["disabled"]))
    app.setPalette(qpalette)
    app.setStyleSheet(build_stylesheet(palette))
    clear_icon_cache()
    return palette
