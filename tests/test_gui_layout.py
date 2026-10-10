"""Layout behaviour: one page scroll, responsive sections, readable preview rows (headless Qt)."""

from __future__ import annotations

import pytest
from PySide6.QtCore import QPoint, QPointF, Qt
from PySide6.QtGui import QWheelEvent
from PySide6.QtWidgets import QApplication, QScrollArea

from app.ai.schemas import RenameAction
from app.context import AppContext
from app.files.planner import Planner, PlannerOptions
from app.files.scanner import scan_folder
from app.ui.main_window import MainWindow
from app.ui.preview_window import COL_NEW, COL_OLD, MAX_FITTED_ROWS, breakable
from app.ui.theme import apply_theme
from tests.helpers import wait_until

pytestmark = pytest.mark.gui
LONG = "Blue Bloods - A Long-Buried Secret Threatens to Tear the Reagan Family Apart After an Unexpected Discovery"


@pytest.fixture
def window(qapp, tmp_path, dispose):
    apply_theme(qapp, "dark")
    win = MainWindow(AppContext.create(tmp_path / "ctx"))
    win.files_panel.set_filter("all")
    win.resize(1280, 900)
    win.show()
    settle()
    yield win
    wait_until(lambda: not win._workers, timeout=3)
    dispose(win)


def settle(rounds: int = 20) -> None:
    for _ in range(rounds):
        QApplication.processEvents()


def show_plan(window, workspace, count: int, title: str = LONG) -> None:
    for i in range(count):
        (workspace / f"Blue.Bloods.S05E{i:03d}.1080p.WEB-DL.x264.mp4").write_bytes(b"")
    entries = scan_folder(workspace).entries
    actions = [RenameAction(source=e.rel_path, target=f"{title} {i}.mp4") for i, e in enumerate(entries)]
    plan = Planner(workspace, entries, PlannerOptions()).build(actions, summary="s")
    window.preview_panel.set_plan(plan, "Completed · Model: Gemini / gemini-2.5-flash · Requests: 2 · Fallbacks: 1 · Files: 3")
    window.preview_panel._fit_rows()
    settle()


def test_one_page_scroll_reaches_every_section(window):
    window.title_panel.set_expanded(True)
    window.resize(780, 560)  # laptop-sized / small window
    settle()
    bar = window.scroll.verticalScrollBar()
    assert bar.maximum() > 0  # the page scrolls instead of squeezing sections
    widths = {name: getattr(window, name).minimumSizeHint().width()
              for name in ("folder_card", "files_panel", "command_panel", "title_panel", "preview_panel")}  # fmt: skip
    assert window.scroll.horizontalScrollBar().maximum() == 0, widths  # no sideways page scrolling
    for widget in (window.folder_card, window.files_panel, window.command_panel.generate_button, window.title_panel.instructions,
                   window.preview_panel.apply_button):  # fmt: skip
        window.scroll.ensureWidgetVisible(widget)
        settle()
        top = widget.mapTo(window.scroll.viewport(), QPoint(0, 0)).y()
        viewport = window.scroll.viewport().height()
        assert 0 <= top < viewport, widget  # reachable by scrolling
        if widget.height() < viewport:
            assert top + widget.height() <= viewport, widget  # and completely visible when it fits


def test_title_generator_has_no_inner_scroll_and_grows_the_page(window):
    tp = window.title_panel
    assert not tp.findChildren(QScrollArea)  # no nested scroll container inside the section
    collapsed = window.page.sizeHint().height()
    tp.set_expanded(True)
    settle()
    expanded = window.page.sizeHint().height()
    assert tp.body.isVisible() and expanded > collapsed + 400  # the page simply becomes taller
    tp.toggle.click()
    settle()
    assert not tp.body.isVisible() and window.page.sizeHint().height() < expanded


def test_title_generator_switches_between_two_and_one_column(window):
    tp = window.title_panel
    tp.set_expanded(True)
    window.resize(1400, 900)
    settle()
    assert tp._columns == 2
    assert tp.mode.mapTo(tp, QPoint(0, 0)).y() == tp.source.mapTo(tp, QPoint(0, 0)).y()  # side by side
    window.resize(780, 700)
    settle()
    assert tp._columns == 1
    assert tp.source.mapTo(tp, QPoint(0, 0)).y() > tp.mode.mapTo(tp, QPoint(0, 0)).y()  # stacked
    window.resize(1400, 900)
    settle()
    assert tp._columns == 2


def test_controls_have_comfortable_sizes(window):
    tp = window.title_panel
    tp.set_expanded(True)
    settle()
    for widget in (tp.mode, tp.topic, window.command_panel.generate_button, window.command_panel.strategy_combo,
                   window.preview_panel.apply_button):  # fmt: skip
        assert widget.height() >= 36, widget
    assert tp.instructions.height() >= 100 and tp.header_bar.height() >= 40


def test_long_custom_instructions_stay_usable(window):
    tp = window.title_panel
    tp.set_expanded(True)
    tp.instructions.setPlainText("Avoid clichés. " * 150)
    settle()
    assert 100 <= tp.instructions.height() <= 200  # its own scroll bar inside the text box, page layout unchanged
    assert tp.config().custom_instructions.startswith("Avoid clichés.")


def test_files_and_preview_grow_with_the_window(window):
    window.resize(1280, 700)
    settle()
    small = window.files_panel.height()
    window.resize(1280, 1300)
    settle()
    assert window.files_panel.height() > small  # more rows visible on big screens
    assert window.preview_panel.minimumHeight() >= 380


def test_long_titles_are_shown_completely(window, workspace):
    show_plan(window, workspace, 3)
    table, model = window.preview_panel.table, window.preview_panel.model
    assert model.index(0, COL_NEW).data() == f"{LONG} 0.mp4"  # never truncated in the model
    assert table.rowHeight(0) > 40  # wrapped over several lines
    assert table.textElideMode() == Qt.TextElideMode.ElideNone and table.wordWrap()
    assert window.preview_panel.info.isVisible() and window.preview_panel.info.wordWrap()
    assert "Fallbacks: 1" in window.preview_panel.info.text()
    assert breakable("Blue.Bloods.S05E01.mp4").replace("​", "") == "Blue.Bloods.S05E01.mp4"  # display-only
    old = model.index(0, COL_OLD).data()
    assert "​" not in old


def test_many_preview_rows_use_fast_fixed_height(window, workspace):
    show_plan(window, workspace, MAX_FITTED_ROWS + 5, title="Short title")
    table = window.preview_panel.table
    assert table.model().rowCount() == MAX_FITTED_ROWS + 5
    assert table.verticalHeader().defaultSectionSize() == 48
    assert window.preview_panel.apply_button.isEnabled()


def test_files_table_scrolls_inside_a_long_list(window, workspace):
    for i in range(300):
        (workspace / f"v{i:03d}.mp4").write_bytes(b"")
    window.load_folder(workspace)
    assert wait_until(lambda: window.files_panel.model.rowCount() == 300)
    settle()
    assert window.files_panel.table.verticalScrollBar().maximum() > 0  # the table keeps its own scroll for long lists


def test_wheel_over_unfocused_combo_scrolls_the_page_not_the_value(window):
    tp = window.title_panel
    tp.set_expanded(True)
    window.resize(1000, 600)
    settle()
    before = tp.category.currentIndex()
    pos = QPointF(5, 5)
    event = QWheelEvent(pos, tp.category.mapToGlobal(pos), QPoint(), QPoint(0, -120), Qt.MouseButton.NoButton,
                        Qt.KeyboardModifier.NoModifier, Qt.ScrollPhase.NoScrollPhase, False)  # fmt: skip
    QApplication.sendEvent(tp.category, event)
    assert tp.category.currentIndex() == before


def test_command_area_buttons_wrap_on_narrow_windows(window):
    window.resize(1400, 900)
    settle()
    cp = window.command_panel
    assert cp._narrow is False
    window.resize(780, 700)
    settle()
    assert cp._narrow is True
    assert cp.prompts_button.mapTo(cp, QPoint(0, 0)).y() > cp.strategy_combo.mapTo(cp, QPoint(0, 0)).y()
