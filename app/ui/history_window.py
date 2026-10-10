"""Operation history: every applied batch, with Undo."""

from __future__ import annotations

from datetime import datetime
from pathlib import Path

from PySide6.QtCore import Qt, Signal
from PySide6.QtWidgets import (
    QAbstractItemView,
    QDialog,
    QHBoxLayout,
    QHeaderView,
    QLabel,
    QMessageBox,
    QPlainTextEdit,
    QPushButton,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
    QWidget,
)

from app.context import AppContext
from app.database.models import OperationRecord
from app.files.operation_manager import OperationManager
from app.i18n import tr
from app.ui.widgets import Retranslator
from app.utils.helpers import truncate

MAX_DETAIL_ITEMS = 400
STATUS_KEYS = {
    "applied": "history.status.applied", "undone": "history.status.undone", "rolled_back": "history.status.rolled_back",
    "partial": "history.status.partial", "interrupted": "history.status.interrupted", "running": "history.status.running",
    "failed": "history.status.failed",
}  # fmt: skip


def format_time(created_at: str) -> str:
    """``2026-10-02 00:42:10`` -> ``2026-10-02 12:42 AM``."""
    try:
        return datetime.fromisoformat(created_at).strftime("%Y-%m-%d %I:%M %p")
    except ValueError:
        return created_at


class HistoryDialog(QDialog):
    """Lists operations (newest first) and lets the user undo one or manage the undo-trash."""

    undo_requested = Signal(int)

    def __init__(self, ctx: AppContext, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.ctx = ctx
        self._tr = Retranslator()
        self._tr.title(self, "history.title")
        self.resize(940, 560)
        self._records: list[OperationRecord] = []

        layout = QVBoxLayout(self)
        self.table = QTableWidget(0, 6)
        self.table.setSelectionBehavior(QAbstractItemView.SelectionBehavior.SelectRows)
        self.table.setSelectionMode(QAbstractItemView.SelectionMode.SingleSelection)
        self.table.setEditTriggers(QAbstractItemView.EditTrigger.NoEditTriggers)
        self.table.verticalHeader().hide()
        self.table.setShowGrid(False)
        self.table.setAlternatingRowColors(True)
        header = self.table.horizontalHeader()
        header.setSectionResizeMode(QHeaderView.ResizeMode.Interactive)
        header.setSectionResizeMode(2, QHeaderView.ResizeMode.Stretch)
        header.setDefaultAlignment(Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter)
        for column, width in enumerate((70, 160, 0, 220, 130, 0)):
            if width:
                self.table.setColumnWidth(column, width)
        self.table.itemSelectionChanged.connect(self._on_selection)
        layout.addWidget(self.table, 3)

        self.detail_label = QLabel()
        self.detail_label.setObjectName("CardTitle")
        layout.addWidget(self.detail_label)
        self.details = QPlainTextEdit()
        self.details.setReadOnly(True)
        layout.addWidget(self.details, 2)

        buttons = QHBoxLayout()
        self.undo_button = QPushButton()
        self.undo_button.setObjectName("Primary")
        self.undo_button.clicked.connect(self._undo_selected)
        self.trash_button = QPushButton()
        self.trash_button.clicked.connect(self._empty_trash)
        self.remove_button = QPushButton()
        self.remove_button.clicked.connect(self._remove_selected)
        self.clear_button = QPushButton()
        self.clear_button.setObjectName("Danger")
        self.clear_button.clicked.connect(self._clear_all)
        self.close_button = QPushButton()
        self.close_button.clicked.connect(self.accept)
        for button in (self.undo_button, self.trash_button):
            buttons.addWidget(button)
        buttons.addStretch(1)
        for button in (self.remove_button, self.clear_button, self.close_button):
            buttons.addWidget(button)
        layout.addLayout(buttons)
        self.reload()
        self.retranslate()

    # -------------------------------------------------------------------- data
    def reload(self) -> None:
        self._records = self.ctx.db.list_operations(300)
        self.table.setRowCount(len(self._records))
        for row, record in enumerate(self._records):
            values = (
                f"#{record.id}", format_time(record.created_at), truncate(record.command or "—", 90), record.summary,
                tr(STATUS_KEYS.get(record.status, "history.status.failed")), Path(record.workspace).name or record.workspace,
            )  # fmt: skip
            for column, value in enumerate(values):
                item = QTableWidgetItem(value)
                item.setToolTip(record.command if column == 2 else record.workspace if column == 5 else "")
                self.table.setItem(row, column, item)
        if self._records:
            self.table.selectRow(0)
        self._on_selection()

    def selected_record(self) -> OperationRecord | None:
        rows = self.table.selectionModel().selectedRows() if self.table.selectionModel() else []
        return self._records[rows[0].row()] if rows else None

    def _on_selection(self) -> None:
        record = self.selected_record()
        self.undo_button.setEnabled(record is not None and record.can_undo)
        self.remove_button.setEnabled(record is not None)
        self.clear_button.setEnabled(bool(self._records))
        self.trash_button.setEnabled(False)
        self.trash_button.setText(tr("history.empty_trash"))
        if record is None:
            self.details.clear()
            return
        trash_files = OperationManager.trash_size(Path(record.workspace))
        self.trash_button.setEnabled(trash_files > 0)
        if trash_files:
            self.trash_button.setText(tr("history.empty_trash_n", count=trash_files))
        items = self.ctx.db.get_items(record.id)
        lines = [f"{tr('history.cmd')}: {record.command or '—'}", f"{tr('history.provider')}: {record.provider or '—'}", ""]
        for item in items[:MAX_DETAIL_ITEMS]:
            if item.type == "create_folder":
                lines.append(f"[{item.status}] create folder  {item.target}")
            elif item.type == "delete":
                where = "undo-trash" if item.extra.get("mode") == "trash" else "permanently"
                lines.append(f"[{item.status}] delete ({where})  {item.source}")
            else:
                lines.append(f"[{item.status}] {item.type}  {item.source}  →  {item.target}")
        if len(items) > MAX_DETAIL_ITEMS:
            lines.append(tr("undo.more", count=len(items) - MAX_DETAIL_ITEMS))
        self.details.setPlainText("\n".join(lines))
        self.detail_label.setText(tr("history.details", id=record.id).upper())

    # ----------------------------------------------------------------- actions
    def _undo_selected(self) -> None:
        record = self.selected_record()
        if record is not None and record.can_undo:
            self.undo_requested.emit(record.id)
            self.accept()

    def _empty_trash(self) -> None:
        record = self.selected_record()
        if record is None:
            return
        workspace = Path(record.workspace)
        count = OperationManager.trash_size(workspace)
        if QMessageBox.warning(
            self, tr("history.title"), tr("history.empty_trash_confirm", count=count),
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No, QMessageBox.StandardButton.No,
        ) == QMessageBox.StandardButton.Yes:  # fmt: skip
            OperationManager.empty_trash(workspace)
            self._on_selection()

    def _remove_selected(self) -> None:
        record = self.selected_record()
        if record is not None and QMessageBox.question(
            self, tr("history.title"), tr("history.remove_confirm")
        ) == QMessageBox.StandardButton.Yes:
            self.ctx.db.delete_operation(record.id)
            self.reload()

    def _clear_all(self) -> None:
        if QMessageBox.question(self, tr("history.title"), tr("history.clear_confirm")) == QMessageBox.StandardButton.Yes:
            self.ctx.db.clear_operations()
            self.reload()

    # ------------------------------------------------------------ translations
    def retranslate(self) -> None:
        self._tr.retranslate()
        self.table.setHorizontalHeaderLabels(
            [tr("history.col_id"), tr("history.col_time"), tr("history.col_command"), tr("history.col_summary"),
             tr("history.col_status"), tr("history.col_folder")]
        )  # fmt: skip
        self.undo_button.setText(tr("history.undo"))
        self.remove_button.setText(tr("history.remove"))
        self.clear_button.setText(tr("history.clear"))
        self.close_button.setText(tr("common.close"))
        self._on_selection()
