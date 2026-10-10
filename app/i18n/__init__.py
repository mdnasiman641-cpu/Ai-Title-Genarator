"""Tiny translation system (English + Bangla).

Usage::

    from app.i18n import tr
    label.setText(tr("main.refresh"))
    status.setText(tr("status.files_found", count=126))

Missing keys fall back to English and then to the key itself, so a gap in a
translation can never crash the UI. Listeners registered with
:func:`on_language_changed` are called after :func:`set_language`.
"""

from __future__ import annotations

from collections.abc import Callable

from app.i18n.bn import STRINGS as _BN
from app.i18n.en import STRINGS as _EN

LANGUAGE_NAMES = {"en": "English", "bn": "বাংলা (Bangla)"}
_TABLES: dict[str, dict[str, str]] = {"en": _EN, "bn": _BN}
_current = "en"
_listeners: list[Callable[[], None]] = []


def get_language() -> str:
    return _current


def set_language(code: str) -> None:
    """Switch the active language and notify listeners."""
    global _current
    code = code if code in _TABLES else "en"
    if code != _current:
        _current = code
        for listener in list(_listeners):
            listener()


def on_language_changed(callback: Callable[[], None]) -> None:
    """Register ``callback`` to run after the language changes."""
    _listeners.append(callback)


def tr(key: str, **kwargs: object) -> str:
    """Translate ``key``; ``{placeholders}`` are filled from ``kwargs``."""
    text = _TABLES[_current].get(key) or _EN.get(key) or key
    if kwargs:
        try:
            return text.format(**kwargs)
        except (KeyError, IndexError, ValueError):
            return text
    return text


def all_keys() -> set[str]:
    """Every English key (used by the translation-consistency test)."""
    return set(_EN)
