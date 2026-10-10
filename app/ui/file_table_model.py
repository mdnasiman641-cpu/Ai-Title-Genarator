"""Table model and filter/sort proxy for the scanned file list.

The model holds plain :class:`FileEntry` objects (no per-row widgets), so it
stays responsive with tens of thousands of files. Bulk operations emit a single
``dataChanged`` for the affected range instead of one per row.
"""

from __future__ import annotations

from collections.abc import Callable, Iterable

from PySide6.QtCore import (
    QAbstractTableModel,
    QModelIndex,
    QPersistentModelIndex,
    QSortFilterProxyModel,
    Qt,
)

from app.config.constants import FileKind
from app.files.scanner import FileEntry
from app.i18n import tr
from app.utils.helpers import format_duration, format_size, format_timestamp

COL_SELECT, COL_NAME, COL_EXT, COL_SIZE, COL_DURATION, COL_MODIFIED, COL_STATUS = range(7)
COLUMN_KEYS = ("col.select", "col.name", "col.ext", "col.size", "col.duration", "col.modified", "col.status")
ENTRY_ROLE = Qt.ItemDataRole.UserRole + 1

_Index = QModelIndex | QPersistentModelIndex


class FileTableModel(QAbstractTableModel):
    """Source model: one row per scanned file."""

    def __init__(self) -> None:
        super().__init__()
        self._entries: list[FileEntry] = []
        self._row_by_path: dict[str, int] = {}

    # ----------------------------------------------------------- bulk setup
    def set_entries(self, entries: list[FileEntry]) -> None:
        self.beginResetModel()
        self._entries = entries
        self._row_by_path = {e.rel_path: i for i, e in enumerate(entries)}
        self.endResetModel()

    @property
    def entries(self) -> list[FileEntry]:
        return self._entries

    def entry_at(self, row: int) -> FileEntry | None:
        return self._entries[row] if 0 <= row < len(self._entries) else None

    def entry_for_path(self, rel_path: str) -> FileEntry | None:
        row = self._row_by_path.get(rel_path)
        return None if row is None else self._entries[row]

    def checked_entries(self) -> list[FileEntry]:
        return [e for e in self._entries if e.checked]

    def checked_count(self) -> int:
        return sum(1 for e in self._entries if e.checked)

    # ---------------------------------------------------------------- sorting
    def sort_by(self, column: int, descending: bool = False) -> None:
        """Sort the rows in place with Python's native sort (fast even for tens of thousands of files).

        ``QSortFilterProxyModel`` sorting calls back into Python for every comparison, which is ~1 s
        per sort at 20 000 rows; sorting the source list directly takes a few milliseconds.
        """
        keys = {
            COL_SELECT: lambda e: (not e.checked, e.natural_key),
            COL_EXT: lambda e: (e.ext, e.natural_key),
            COL_SIZE: lambda e: (e.size, e.natural_key),
            COL_DURATION: lambda e: (e.duration or -1.0, e.natural_key),
            COL_MODIFIED: lambda e: (e.mtime, e.natural_key),
            COL_STATUS: lambda e: (e.status, e.natural_key),
        }
        key = keys.get(column, lambda e: e.natural_key)
        self.beginResetModel()
        self._entries.sort(key=key, reverse=descending)
        self._row_by_path = {e.rel_path: i for i, e in enumerate(self._entries)}
        self.endResetModel()

    # ----------------------------------------------------- check-state tools
    def set_checked_rows(self, rows: Iterable[int], checked: bool) -> None:
        """Set the check state of the given source rows (one signal for the whole span)."""
        touched = [r for r in rows if 0 <= r < len(self._entries) and self._entries[r].checked != checked]
        for row in touched:
            self._entries[row].checked = checked
        if touched:
            self._emit_range(min(touched), max(touched), [Qt.ItemDataRole.CheckStateRole])

    def set_checked_where(self, predicate: Callable[[FileEntry], bool]) -> None:
        """Check exactly the entries for which ``predicate`` is true; uncheck all others."""
        if not self._entries:
            return
        for entry in self._entries:
            entry.checked = predicate(entry)
        self._emit_range(0, len(self._entries) - 1, [Qt.ItemDataRole.CheckStateRole])

    def invert_rows(self, rows: Iterable[int]) -> None:
        rows = [r for r in rows if 0 <= r < len(self._entries)]
        for row in rows:
            self._entries[row].checked = not self._entries[row].checked
        if rows:
            self._emit_range(min(rows), max(rows), [Qt.ItemDataRole.CheckStateRole])

    # -------------------------------------------------------- row metadata
    def set_status_map(self, statuses: dict[str, str]) -> None:
        """Replace the Status column text (entries not in ``statuses`` are cleared)."""
        changed = False
        for entry in self._entries:
            new = statuses.get(entry.rel_path, "")
            if entry.status != new:
                entry.status = new
                changed = True
        if changed and self._entries:
            self._emit_range(0, len(self._entries) - 1, [Qt.ItemDataRole.DisplayRole], COL_STATUS, COL_STATUS)

    def set_duration(self, rel_path: str, seconds: float | None) -> None:
        row = self._row_by_path.get(rel_path)
        if row is None:
            return
        self._entries[row].duration = seconds if seconds is not None else -1.0  # -1 = probed, unknown
        self._emit_range(row, row, [Qt.ItemDataRole.DisplayRole], COL_DURATION, COL_DURATION)

    def _emit_range(
        self, first: int, last: int, roles: list[Qt.ItemDataRole], col_a: int = 0, col_b: int | None = None
    ) -> None:
        col_b = self.columnCount() - 1 if col_b is None else col_b
        self.dataChanged.emit(self.index(first, col_a), self.index(last, col_b), roles)

    # ------------------------------------------------------------ Qt model
    def rowCount(self, parent: _Index = QModelIndex()) -> int:  # noqa: B008
        return 0 if parent.isValid() else len(self._entries)

    def columnCount(self, parent: _Index = QModelIndex()) -> int:  # noqa: B008
        return 0 if parent.isValid() else len(COLUMN_KEYS)

    def headerData(
        self, section: int, orientation: Qt.Orientation, role: int = Qt.ItemDataRole.DisplayRole
    ) -> object:
        if orientation == Qt.Orientation.Horizontal and role == Qt.ItemDataRole.DisplayRole:
            return "" if section == COL_SELECT else tr(COLUMN_KEYS[section])
        return None

    def flags(self, index: _Index) -> Qt.ItemFlag:
        if not index.isValid():
            return Qt.ItemFlag.NoItemFlags
        flags = Qt.ItemFlag.ItemIsEnabled | Qt.ItemFlag.ItemIsSelectable
        if index.column() == COL_SELECT:
            flags |= Qt.ItemFlag.ItemIsUserCheckable
        return flags

    def data(self, index: _Index, role: int = Qt.ItemDataRole.DisplayRole) -> object:
        if not index.isValid():
            return None
        entry = self._entries[index.row()]
        col = index.column()
        if role == Qt.ItemDataRole.CheckStateRole and col == COL_SELECT:
            return Qt.CheckState.Checked if entry.checked else Qt.CheckState.Unchecked
        if role == ENTRY_ROLE:
            return entry
        if role == Qt.ItemDataRole.DisplayRole:
            return self._display(entry, col)
        if role == Qt.ItemDataRole.ToolTipRole and col == COL_NAME:
            return entry.rel_path
        if role == Qt.ItemDataRole.TextAlignmentRole and col in (COL_SIZE, COL_DURATION):
            return int(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)
        return None

    @staticmethod
    def _display(entry: FileEntry, col: int) -> str:
        if col == COL_NAME:
            return entry.rel_path
        if col == COL_EXT:
            return entry.ext.lstrip(".").upper()
        if col == COL_SIZE:
            return format_size(entry.size)
        if col == COL_DURATION:
            return format_duration(entry.duration) if entry.duration and entry.duration > 0 else ""
        if col == COL_MODIFIED:
            return format_timestamp(entry.mtime)
        if col == COL_STATUS:
            return entry.status
        return ""

    def setData(self, index: _Index, value: object, role: int = Qt.ItemDataRole.EditRole) -> bool:
        if index.isValid() and index.column() == COL_SELECT and role == Qt.ItemDataRole.CheckStateRole:
            entry = self._entries[index.row()]
            entry.checked = Qt.CheckState(value) == Qt.CheckState.Checked  # type: ignore[arg-type]
            self.dataChanged.emit(index, index, [Qt.ItemDataRole.CheckStateRole])
            return True
        return False

    def retranslate(self) -> None:
        self.headerDataChanged.emit(Qt.Orientation.Horizontal, 0, len(COLUMN_KEYS) - 1)


class FileFilterProxy(QSortFilterProxyModel):
    """Filters by file kind and search text. (Sorting is done by :meth:`FileTableModel.sort_by`.)"""

    def __init__(self) -> None:
        super().__init__()
        self._kinds: frozenset[FileKind] | None = None
        self._needle = ""
        self.setDynamicSortFilter(False)  # never re-sort/re-filter implicitly during bulk edits

    def set_kinds(self, kinds: Iterable[FileKind] | None) -> None:
        self._kinds = None if kinds is None else frozenset(kinds)
        self.invalidateFilter()

    def set_search(self, text: str) -> None:
        self._needle = text.strip().casefold()
        self.invalidateFilter()

    def filterAcceptsRow(self, source_row: int, source_parent: _Index) -> bool:  # noqa: N802
        model = self.sourceModel()
        if not isinstance(model, FileTableModel):
            return True
        entry = model.entry_at(source_row)
        if entry is None:
            return False
        if self._kinds is not None and entry.kind not in self._kinds:
            return False
        return not self._needle or self._needle in entry.rel_path.casefold()

    def source_rows(self) -> list[int]:
        """Source-model rows currently visible, in display order."""
        return [self.mapToSource(self.index(r, 0)).row() for r in range(self.rowCount())]
