"""The "Files" card: filter bar, selection shortcuts and the file table."""

from __future__ import annotations

from collections.abc import Iterable

from PySide6.QtCore import QModelIndex, QPoint, Qt, QTimer, Signal
from PySide6.QtGui import QIcon
from PySide6.QtWidgets import (
    QAbstractItemView,
    QComboBox,
    QHBoxLayout,
    QHeaderView,
    QLabel,
    QLineEdit,
    QMenu,
    QTableView,
    QToolButton,
    QWidget,
)

from app.config.constants import FileKind
from app.files.scanner import FileEntry
from app.i18n import tr
from app.ui.file_table_model import (
    COL_DURATION,
    COL_EXT,
    COL_MODIFIED,
    COL_NAME,
    COL_SELECT,
    COL_SIZE,
    COL_STATUS,
    ENTRY_ROLE,
    FileFilterProxy,
    FileTableModel,
)
from app.ui.details_panel import DetailsPanel
from app.ui.widgets import Card, IconBinder, make_button

# (translation key, kinds shown or None for all)
FILTERS: tuple[tuple[str, tuple[FileKind, ...] | None], ...] = (
    ("filter.all", None),
    ("filter.video", (FileKind.VIDEO,)),
    ("filter.audio", (FileKind.AUDIO,)),
    ("filter.image", (FileKind.IMAGE,)),
    ("filter.document", (FileKind.DOCUMENT,)),
)
FILTER_VALUES = ("all", "video", "audio", "image", "document")
SORT_COLUMNS = (
    ("sort.name", COL_NAME),
    ("sort.size", COL_SIZE),
    ("sort.date", COL_MODIFIED),
    ("sort.type", COL_EXT),
    ("sort.duration", COL_DURATION),
)


class FilesPanel(Card):
    """File list with search, type filter, sorting and select shortcuts."""

    selection_changed = Signal(int)  # number of checked files
    current_entry_changed = Signal(object)  # FileEntry | None
    visible_rows_changed = Signal()  # debounced: scrolled / filtered / resized

    def __init__(self, icons: IconBinder) -> None:
        super().__init__("files.title")
        self._icons = icons
        self.model = FileTableModel()
        self.proxy = FileFilterProxy()
        self.proxy.setSourceModel(self.model)

        self.count_label = QLabel()
        self.count_label.setObjectName("Count")
        self.header.insertWidget(1, self.count_label)

        self._build_toolbar()
        self._build_table()
        self.model.dataChanged.connect(self._on_data_changed)
        self.model.modelReset.connect(self._emit_selection)
        self.proxy.layoutChanged.connect(self._schedule_visible)
        self.proxy.modelReset.connect(self._schedule_visible)
        self._visible_timer = QTimer(self)
        self._visible_timer.setSingleShot(True)
        self._visible_timer.setInterval(200)
        self._visible_timer.timeout.connect(self.visible_rows_changed)
        self.retranslate()

    # ------------------------------------------------------------------ build
    def _build_toolbar(self) -> None:
        row = QHBoxLayout()
        row.setSpacing(8)
        self.search = QLineEdit()
        self.search.setClearButtonEnabled(True)
        search_action = self.search.addAction(QIcon(), QLineEdit.ActionPosition.LeadingPosition)
        self._icons.bind(search_action.setIcon, "search", 16)
        self.search.textChanged.connect(self._on_search)
        self.search.setMinimumWidth(100)
        row.addWidget(self.search, 1)

        self.filter_combo = QComboBox()
        for key, _ in FILTERS:
            self.filter_combo.addItem(tr(key))
        self.filter_combo.currentIndexChanged.connect(self._on_filter_changed)
        row.addWidget(self.filter_combo)

        self.sort_combo = QComboBox()
        for key, _ in SORT_COLUMNS:
            self.sort_combo.addItem(tr(key))
        self.sort_combo.currentIndexChanged.connect(self._on_sort_combo)
        row.addWidget(self.sort_combo)

        self.order_button = QToolButton()
        self.order_button.setCheckable(True)
        self.order_button.setObjectName("HeaderButton")
        self.order_button.toggled.connect(self._on_sort_combo)
        row.addWidget(self.order_button)
        self.body.addLayout(row)

        row2 = QHBoxLayout()
        row2.setSpacing(8)
        self.btn_all = make_button("files.select_all", self._tr, name="Chip", tooltip_key="files.select_all_tip")
        self.btn_none = make_button("files.select_none", self._tr, name="Chip", tooltip_key="files.select_none_tip")
        self.btn_video = make_button("files.video_only", self._tr, name="Chip", tooltip_key="files.video_only_tip")
        self.btn_all.clicked.connect(self.select_all_visible)
        self.btn_none.clicked.connect(self.select_none)
        self.btn_video.clicked.connect(self.select_videos_only)
        for button in (self.btn_all, self.btn_none, self.btn_video):
            row2.addWidget(button)
        row2.addStretch(1)
        self.selected_label = QLabel()
        self.selected_label.setObjectName("Count")
        row2.addWidget(self.selected_label)
        self.body.addLayout(row2)

    def _build_table(self) -> None:
        self.table = QTableView()
        self.table.setModel(self.proxy)
        self.table.setAlternatingRowColors(True)
        self.table.setSelectionBehavior(QAbstractItemView.SelectionBehavior.SelectRows)
        self.table.setSelectionMode(QAbstractItemView.SelectionMode.ExtendedSelection)
        self.table.setEditTriggers(QAbstractItemView.EditTrigger.NoEditTriggers)
        self.table.setShowGrid(False)
        self.table.setWordWrap(False)
        self.table.setTextElideMode(Qt.TextElideMode.ElideMiddle)
        self.table.verticalHeader().hide()
        self.table.verticalHeader().setDefaultSectionSize(34)
        self.table.setSortingEnabled(False)  # sorting is done by the model (see FileTableModel.sort_by)
        self.table.setContextMenuPolicy(Qt.ContextMenuPolicy.CustomContextMenu)
        self.table.customContextMenuRequested.connect(self._context_menu)
        self.table.setMinimumHeight(220)  # grows with the window (see MainWindow._fit_sections)
        header = self.table.horizontalHeader()
        header.setStretchLastSection(False)
        header.setDefaultAlignment(Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter)
        header.setSectionResizeMode(QHeaderView.ResizeMode.Interactive)
        header.setSectionResizeMode(COL_NAME, QHeaderView.ResizeMode.Stretch)
        header.setSectionResizeMode(COL_SELECT, QHeaderView.ResizeMode.Fixed)
        self.table.setColumnWidth(COL_SELECT, 44)
        self.table.setColumnWidth(COL_EXT, 80)
        self.table.setColumnWidth(COL_SIZE, 90)
        self.table.setColumnWidth(COL_DURATION, 90)
        self.table.setColumnWidth(COL_MODIFIED, 140)
        self.table.setColumnWidth(COL_STATUS, 130)
        header.setSectionsClickable(True)
        header.setSortIndicatorShown(True)
        header.sectionClicked.connect(self._on_header_clicked)
        self.table.verticalScrollBar().valueChanged.connect(self._schedule_visible)
        self.table.selectionModel().currentRowChanged.connect(self._on_current_row)
        self._sort_column, self._sort_desc = COL_NAME, False
        header.setSortIndicator(COL_NAME, Qt.SortOrder.AscendingOrder)
        self.details = DetailsPanel(self._icons)
        row = QHBoxLayout()
        row.setSpacing(10)
        row.addWidget(self.table, 1)
        row.addWidget(self.details)
        self.body.addLayout(row, 1)

    # ------------------------------------------------------------------- data
    def set_entries(self, entries: list[FileEntry], *, check_videos: bool = True) -> None:
        """Load a fresh scan. Videos are pre-selected (other file types need explicit opt-in)."""
        if check_videos:
            for entry in entries:
                entry.checked = entry.kind == FileKind.VIDEO
        self.model.set_entries(entries)
        self.model.sort_by(self._sort_column, self._sort_desc)
        self._update_count_label()
        self._emit_selection()

    def selected_entries(self) -> list[FileEntry]:
        """Checked files, in the order shown on screen."""
        checked = [self.model.entries[r] for r in self.proxy.source_rows() if self.model.entries[r].checked]
        seen = {e.rel_path for e in checked}
        checked.extend(e for e in self.model.entries if e.checked and e.rel_path not in seen)  # hidden by filter
        return checked

    def check_paths(self, rel_paths: Iterable[str]) -> None:
        """Check exactly these files (used after drag & drop of files)."""
        wanted = set(rel_paths)
        self.model.set_checked_where(lambda e: e.rel_path in wanted)

    def current_entry(self) -> FileEntry | None:
        index = self.table.currentIndex()
        return index.data(ENTRY_ROLE) if index.isValid() else None

    def visible_entries(self) -> list[FileEntry]:
        return [self.model.entries[r] for r in self.proxy.source_rows()]

    def visible_viewport_entries(self) -> list[FileEntry]:
        """Entries whose rows are currently within the table viewport (for lazy loading)."""
        total = self.proxy.rowCount()
        if total == 0:
            return []
        first = self.table.rowAt(0)
        last = self.table.rowAt(self.table.viewport().height() - 1)
        first = max(first, 0)
        last = total - 1 if last < 0 else last
        out: list[FileEntry] = []
        for row in range(first, min(last + 1, total)):
            entry = self.proxy.index(row, 0).data(ENTRY_ROLE)
            if entry is not None:
                out.append(entry)
        return out

    # -------------------------------------------------------------- selection
    def select_all_visible(self) -> None:
        self.model.set_checked_rows(self.proxy.source_rows(), True)

    def select_none(self) -> None:
        self.model.set_checked_where(lambda e: False)

    def select_videos_only(self) -> None:
        self.model.set_checked_where(lambda e: e.kind == FileKind.VIDEO)

    def _selected_source_rows(self) -> list[int]:
        return sorted({self.proxy.mapToSource(i).row() for i in self.table.selectionModel().selectedRows()})

    def _context_menu(self, pos: QPoint) -> None:
        rows = self._selected_source_rows()
        if not rows:
            return
        menu = QMenu(self)
        menu.addAction(tr("files.ctx_check")).triggered.connect(lambda: self.model.set_checked_rows(rows, True))
        menu.addAction(tr("files.ctx_uncheck")).triggered.connect(lambda: self.model.set_checked_rows(rows, False))
        menu.addAction(tr("files.ctx_invert")).triggered.connect(lambda: self.model.invert_rows(rows))
        menu.exec(self.table.viewport().mapToGlobal(pos))

    # ----------------------------------------------------------- filter / sort
    def set_filter(self, value: str) -> None:
        """Select a type filter by its stable id (``all``, ``video`` ...)."""
        if value in FILTER_VALUES:
            self.filter_combo.setCurrentIndex(FILTER_VALUES.index(value))

    def _on_filter_changed(self, index: int) -> None:
        self.proxy.set_kinds(FILTERS[index][1])
        self._update_count_label()
        self._schedule_visible()

    def _on_search(self, text: str) -> None:
        self.proxy.set_search(text)
        self._update_count_label()
        self._schedule_visible()

    def _on_sort_combo(self, *_args: object) -> None:
        column = SORT_COLUMNS[self.sort_combo.currentIndex()][1]
        self._sort(column, self.order_button.isChecked())

    def apply_sort(self, by: str, descending: bool) -> None:
        """Sort the table by ``by`` (``name``/``size``/``date``/``type``/``duration``), e.g. for a "sort" command."""
        columns = {"name": COL_NAME, "size": COL_SIZE, "date": COL_MODIFIED, "type": COL_EXT, "duration": COL_DURATION}
        self._sort(columns.get(by, COL_NAME), descending)

    def _on_header_clicked(self, column: int) -> None:
        if column == self._sort_column:
            self._sort(column, not self._sort_desc)
        else:
            self._sort(column, False)

    def _sort(self, column: int, descending: bool) -> None:
        """Single entry point: sort the model, then bring the combo, arrow and header in line."""
        self._sort_column, self._sort_desc = column, descending
        self.model.sort_by(column, descending)
        header = self.table.horizontalHeader()
        header.setSortIndicator(column, Qt.SortOrder.DescendingOrder if descending else Qt.SortOrder.AscendingOrder)
        index = next((i for i, (_, col) in enumerate(SORT_COLUMNS) if col == column), None)
        with _blocked(self.sort_combo, self.order_button):
            if index is not None:
                self.sort_combo.setCurrentIndex(index)
            self.order_button.setChecked(descending)
        self._update_order_button()
        self._schedule_visible()

    def _update_order_button(self) -> None:
        descending = self.order_button.isChecked()
        self.order_button.setText("▼" if descending else "▲")
        self.order_button.setToolTip(tr("sort.desc" if descending else "sort.asc"))

    # ------------------------------------------------------------- bookkeeping
    def _on_data_changed(self, *_args: object) -> None:
        self._emit_selection()

    def _emit_selection(self) -> None:
        count = self.model.checked_count()
        self.selected_label.setText(tr("files.selected_count", count=count))
        self.selection_changed.emit(count)

    def _on_current_row(self, current: QModelIndex, _previous: QModelIndex) -> None:
        self.current_entry_changed.emit(current.data(ENTRY_ROLE) if current.isValid() else None)

    def _schedule_visible(self, *_args: object) -> None:
        if hasattr(self, "_visible_timer"):
            self._visible_timer.start()

    def _update_count_label(self) -> None:
        total = len(self.model.entries)
        shown = self.proxy.rowCount()
        if shown != total:
            self.count_label.setText(tr("files.found_filtered", total=total, shown=shown))
        else:
            self.count_label.setText(tr("files.found", count=total))

    def retranslate(self) -> None:
        super().retranslate()
        for i, (key, _) in enumerate(FILTERS):
            self.filter_combo.setItemText(i, tr(key))
        for i, (key, _) in enumerate(SORT_COLUMNS):
            self.sort_combo.setItemText(i, tr(key))
        self.search.setPlaceholderText(tr("files.search_placeholder"))
        self.order_button.setToolTip(tr("sort.desc" if self.order_button.isChecked() else "sort.asc"))
        self._update_order_button()
        self._update_count_label()
        self.selected_label.setText(tr("files.selected_count", count=self.model.checked_count()))
        self.model.retranslate()
        self.details.retranslate()


class _blocked:  # noqa: N801 - tiny context manager
    """Temporarily block signals of several QObjects."""

    def __init__(self, *objects: QWidget) -> None:
        self._objects = objects

    def __enter__(self) -> None:
        self._previous = [o.blockSignals(True) for o in self._objects]

    def __exit__(self, *_exc: object) -> None:
        for obj, prev in zip(self._objects, self._previous, strict=True):
            obj.blockSignals(prev)


__all__ = ["FILTER_VALUES", "FILTERS", "FilesPanel"]
