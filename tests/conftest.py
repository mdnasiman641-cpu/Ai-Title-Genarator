"""Shared fixtures. Qt runs headless (offscreen) so the suite works on CI and servers."""

from __future__ import annotations

import os
import sys
from pathlib import Path

import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


@pytest.fixture(autouse=True)
def _isolated_data_dir(tmp_path, monkeypatch):
    """Every test gets its own app-data directory so the real profile is never touched."""
    data = tmp_path / "appdata"
    data.mkdir()
    monkeypatch.setenv("AIVFA_DATA_DIR", str(data))
    return data


@pytest.fixture
def workspace(tmp_path) -> Path:
    """An empty folder to act as the user's selected workspace."""
    root = tmp_path / "Videos"
    root.mkdir()
    return root


@pytest.fixture
def make_files(workspace):
    """Create files (with tiny content) inside the workspace and return their paths."""

    def _make(*names: str, content: str = "x") -> list[Path]:
        created = []
        for name in names:
            path = workspace.joinpath(*name.split("/"))
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(content + name, encoding="utf-8")
            created.append(path)
        return created

    return _make


@pytest.fixture(scope="session")
def qapp():
    from PySide6.QtWidgets import QApplication

    from app.ui.gc_guard import MainThreadGC

    app = QApplication.instance() or QApplication(sys.argv[:1])
    guard = MainThreadGC(app)  # see app/ui/gc_guard.py: no GC-driven Qt deletion on worker threads
    yield app
    guard.stop()


@pytest.fixture
def dispose(qapp):
    """Return a function that closes and *deletes* widgets on the GUI thread."""
    import gc

    from PySide6.QtCore import QEvent

    def _dispose(*widgets) -> None:
        for widget in widgets:
            widget.close()
            widget.deleteLater()
        qapp.sendPostedEvents(None, QEvent.Type.DeferredDelete)
        qapp.processEvents()
        gc.collect()

    return _dispose
