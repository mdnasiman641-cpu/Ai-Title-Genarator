"""Main-window smoke tests (headless Qt)."""

from __future__ import annotations

import pytest
from PySide6.QtCore import QUrl

from app.config.constants import FileKind
from app.context import AppContext
from app.i18n import set_language, tr
from app.ui.main_window import MainWindow
from app.ui.theme import apply_theme
from tests.helpers import wait_until

pytestmark = pytest.mark.gui


@pytest.fixture
def window(qapp, tmp_path, dispose):
    apply_theme(qapp, "dark")
    ctx = AppContext.create(tmp_path / "ctx")
    win = MainWindow(ctx)
    win.files_panel.set_filter("all")  # tests assume every file is visible unless they say otherwise
    win.show()
    yield win
    wait_until(lambda: not win._workers, timeout=3)  # let in-flight workers finish first
    dispose(win)
    set_language("en")


def test_default_filter_shows_videos_only(qapp, tmp_path, dispose, make_files, workspace):
    make_files("a.mp4", "notes.txt")
    win = MainWindow(AppContext.create(tmp_path / "ctx2"))
    win.show()
    win.load_folder(workspace)
    assert wait_until(lambda: win.files_panel.model.rowCount() == 2)
    assert win.files_panel.proxy.rowCount() == 1  # only the video is listed by default
    assert "1 of 2" in win.files_panel.count_label.text()
    wait_until(lambda: not win._workers, timeout=3)
    dispose(win)


def test_window_title_and_defaults(window):
    assert window.windowTitle() == "AI Video File Assistant"
    assert not window.subfolders_check.isChecked()
    assert window.status_label.text() == "Ready"


def test_scan_populates_table_and_preselects_videos(window, make_files, workspace):
    make_files("a.mp4", "b.mkv", "notes.txt", "বাংলা.mp4")
    window.load_folder(workspace)
    assert wait_until(lambda: window.files_panel.model.rowCount() == 4)
    model = window.files_panel.model
    assert {e.name for e in model.checked_entries()} == {"a.mp4", "b.mkv", "বাংলা.mp4"}
    assert "4 files found" in window.files_panel.count_label.text()


def test_select_all_none_video_only(window, make_files, workspace):
    make_files("a.mp4", "b.txt")
    window.load_folder(workspace)
    assert wait_until(lambda: window.files_panel.model.rowCount() == 2)
    panel = window.files_panel
    panel.select_none()
    assert panel.model.checked_count() == 0
    panel.select_all_visible()
    assert panel.model.checked_count() == 2
    panel.select_videos_only()
    assert [e.name for e in panel.model.checked_entries()] == ["a.mp4"]


def test_type_filter_and_search(window, make_files, workspace):
    make_files("a.mp4", "b.mp3", "c.jpg", "episode 5.mkv")
    window.load_folder(workspace)
    assert wait_until(lambda: window.files_panel.model.rowCount() == 4)
    panel = window.files_panel
    panel.set_filter("video")
    assert panel.proxy.rowCount() == 2
    panel.set_filter("audio")
    assert panel.proxy.rowCount() == 1
    panel.set_filter("all")
    panel.search.setText("EPISODE")
    assert panel.proxy.rowCount() == 1
    panel.search.setText("")
    assert panel.proxy.rowCount() == 4


def test_select_all_only_touches_visible_rows(window, make_files, workspace):
    make_files("a.mp4", "b.mp3")
    window.load_folder(workspace)
    assert wait_until(lambda: window.files_panel.model.rowCount() == 2)
    panel = window.files_panel
    panel.select_none()
    panel.set_filter("audio")
    panel.select_all_visible()
    assert [e.name for e in panel.model.checked_entries()] == ["b.mp3"]


def test_sorting_by_size_and_direction(window, workspace):
    (workspace / "small.mp4").write_bytes(b"1")
    (workspace / "big.mp4").write_bytes(b"1" * 5000)
    window.load_folder(workspace)
    panel = window.files_panel
    assert wait_until(lambda: panel.model.rowCount() == 2)
    panel.sort_combo.setCurrentIndex(1)  # size
    names = [panel.proxy.index(r, 1).data() for r in range(2)]
    assert names == ["small.mp4", "big.mp4"]
    panel.order_button.setChecked(True)  # descending
    names = [panel.proxy.index(r, 1).data() for r in range(2)]
    assert names == ["big.mp4", "small.mp4"]


def test_natural_sort_in_table(window, make_files, workspace):
    make_files("ep10.mp4", "ep2.mp4", "ep1.mp4")
    window.load_folder(workspace)
    panel = window.files_panel
    assert wait_until(lambda: panel.model.rowCount() == 3)
    assert [panel.proxy.index(r, 1).data() for r in range(3)] == ["ep1.mp4", "ep2.mp4", "ep10.mp4"]
    assert [e.name for e in panel.selected_entries()] == ["ep1.mp4", "ep2.mp4", "ep10.mp4"]


def test_include_subfolders_is_opt_in(window, make_files, workspace):
    make_files("a.mp4", "sub/b.mp4")
    window.load_folder(workspace)
    assert wait_until(lambda: window.files_panel.model.rowCount() == 1)
    window.subfolders_check.setChecked(True)
    assert wait_until(lambda: window.files_panel.model.rowCount() == 2)
    assert window.ctx.settings.load().include_subfolders is True


def test_missing_folder_is_reported_not_crashing(window, tmp_path, monkeypatch):
    from PySide6.QtWidgets import QMessageBox

    shown = []
    monkeypatch.setattr(QMessageBox, "warning", lambda *a, **k: shown.append(a[2]))
    window.load_folder(tmp_path / "does-not-exist")
    assert shown and "not found" in shown[0]


def test_drop_folder_and_files(window, make_files, workspace):
    make_files("a.mp4", "b.mp4", "c.txt")
    window.handle_drop([QUrl.fromLocalFile(str(workspace))])
    assert wait_until(lambda: window.files_panel.model.rowCount() == 3)
    window.handle_drop([QUrl.fromLocalFile(str(workspace / "b.mp4"))])
    assert wait_until(lambda: [e.name for e in window.files_panel.model.checked_entries()] == ["b.mp4"])
    window.handle_drop([QUrl.fromLocalFile(str(workspace / "c.txt"))])  # non-video forces "all" filter
    assert wait_until(lambda: [e.name for e in window.files_panel.model.checked_entries()] == ["c.txt"])
    assert window.files_panel.filter_combo.currentIndex() == 0


def test_language_switch_updates_ui(window):
    set_language("bn")
    window.retranslate()
    assert window.browse_button.text() == tr("folder.browse") == "ব্রাউজ"
    assert window.files_panel.btn_all.text() == "সব নির্বাচন"
    set_language("en")
    window.retranslate()
    assert window.browse_button.text() == "Browse"


def test_light_theme_applies(qapp):
    from app.ui.theme import PALETTES, build_stylesheet

    apply_theme(qapp, "light")
    css = build_stylesheet(PALETTES["light"])
    import re

    assert not re.findall(r"@[a-z_]+@", css)  # every palette token was substituted
    apply_theme(qapp, "dark")


def test_filekind_enum_values():
    assert FileKind.VIDEO.value == "video"


def test_header_click_toggles_sort_and_stays_in_sync(window, workspace):
    from app.ui.file_table_model import COL_SIZE

    (workspace / "small.mp4").write_bytes(b"1")
    (workspace / "big.mp4").write_bytes(b"1" * 500)
    window.load_folder(workspace)
    panel = window.files_panel
    assert wait_until(lambda: panel.model.rowCount() == 2)
    header = panel.table.horizontalHeader()
    header.sectionClicked.emit(COL_SIZE)  # first click: ascending
    assert [panel.proxy.index(r, 1).data() for r in range(2)] == ["small.mp4", "big.mp4"]
    assert panel.sort_combo.currentIndex() == 1 and not panel.order_button.isChecked()
    header.sectionClicked.emit(COL_SIZE)  # second click: descending
    assert [panel.proxy.index(r, 1).data() for r in range(2)] == ["big.mp4", "small.mp4"]
    assert panel.order_button.isChecked() and panel.order_button.text() == "▼"


def test_sort_survives_a_rescan(window, workspace):
    (workspace / "a.mp4").write_bytes(b"1")
    (workspace / "b.mp4").write_bytes(b"1" * 9)
    window.load_folder(workspace)
    panel = window.files_panel
    assert wait_until(lambda: panel.model.rowCount() == 2)
    panel.apply_sort("size", True)
    window.refresh()
    assert wait_until(lambda: [panel.proxy.index(r, 1).data() for r in range(2)] == ["b.mp4", "a.mp4"])
