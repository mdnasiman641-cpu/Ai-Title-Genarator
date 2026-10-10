"""Test helpers shared by GUI tests."""

from __future__ import annotations

import time

from PySide6.QtWidgets import QApplication


def wait_until(predicate, timeout: float = 5.0, interval: float = 0.01) -> bool:
    """Spin the Qt event loop until ``predicate()`` is true (or ``timeout`` seconds pass)."""
    app = QApplication.instance()
    end = time.monotonic() + timeout
    while time.monotonic() < end:
        app.processEvents()
        if predicate():
            return True
        time.sleep(interval)
    app.processEvents()
    return bool(predicate())
