"""Main application window: folder → files → AI command → preview → apply → undo."""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path

from PySide6.QtCore import QByteArray, QEvent, QObject, Qt, QThreadPool, QTimer, QUrl
from PySide6.QtGui import QCloseEvent, QDragEnterEvent, QDropEvent, QIcon
from PySide6.QtWidgets import (
    QApplication,
    QCheckBox,
    QComboBox,
    QDialog,
    QFileDialog,
    QFrame,
    QGridLayout,
    QHBoxLayout,
    QInputDialog,
    QLabel,
    QLineEdit,
    QMainWindow,
    QMessageBox,
    QProgressBar,
    QPushButton,
    QScrollArea,
    QVBoxLayout,
    QWidget,
)

from app.ai.ai_router import ModelRouter, make_router
from app.ai.pipeline import PipelineRequest, PipelineResult
from app.ai.task_analyzer import analyze_task
from app.ai.title_config import load_current, save_current
from app.ai.web_search import make_search_service
from app.config.constants import APP_NAME, FileKind, ProcessingMode
from app.context import AppContext
from app.database.models import CommandRecord, SavedPrompt
from app.files.metadata import MetadataProbe, VideoMetadata
from app.files.operation_manager import ExecutionResult, OperationManager, UndoResult
from app.files.plan import OpKind, Plan
from app.files.scanner import FileEntry, ScanResult
from app.files.thumbnails import ThumbnailCache
from app.i18n import set_language, tr
from app.ui.command_panel import CommandPanel
from app.ui.files_panel import FilesPanel
from app.ui.history_window import HistoryDialog
from app.ui.model_choice import AUTO
from app.ui.model_dialog import ModelDialog
from app.ui.preview_window import PreviewPanel
from app.ui.saved_prompts_window import SavedPromptsDialog
from app.ui.settings_window import SettingsDialog
from app.ui.theme import PALETTES, apply_theme, resolve_theme
from app.ui.widgets import Card, IconBinder, Retranslator, make_button, make_compact
from app.utils.helpers import classify_extension, resource_path
from app.utils.logger import get_logger, setup_logging
from app.workers.ai_worker import PlanWorker
from app.workers.base_worker import BaseWorker
from app.workers.metadata_worker import MetadataWorker, ThumbnailWorker, media_pool
from app.workers.operation_worker import OperationWorker, UndoWorker
from app.workers.scan_worker import ScanWorker

log = get_logger("ui.main")


class MainWindow(QMainWindow):
    """Dashboard that drives the whole workflow."""

    def __init__(self, ctx: AppContext) -> None:
        super().__init__()
        self.ctx = ctx
        self.settings = ctx.settings.load()
        self.manager = OperationManager(ctx.db)
        self.workspace: Path | None = None
        self._tr = Retranslator()
        self._pool = QThreadPool.globalInstance()
        self._workers: set[BaseWorker] = set()
        self._scan_generation = 0
        self._plan_generation = 0
        self._busy = 0
        self._operation_running = False
        self._pending_paths: set[str] = set()  # relative paths to tick after the next scan
        self._pending_statuses: dict[str, str] = {}
        self._probe = MetadataProbe(self.settings.ffmpeg_dir)
        self._thumbs = ThumbnailCache(self.settings.ffmpeg_dir)
        self._meta: dict[str, VideoMetadata | None] = {}
        self._meta_requested: set[str] = set()
        self._meta_generation = 0
        self._thumb_timer = QTimer(self)
        self._thumb_timer.setSingleShot(True)
        self._thumb_timer.setInterval(250)
        self._thumb_timer.timeout.connect(self._request_thumbnail)
        self._sticky_status: tuple[str, str] | None = None  # shown after the next scan instead of the file count
        self._plan_worker: PlanWorker | None = None
        self._active_command = ""
        self._active_provider = ""
        self._palette = PALETTES[resolve_theme(self.settings.theme)]
        self._icon_color = self._palette["muted"]
        self._icons = IconBinder(self._icon_color)

        self._tr.title(self, "app.title")
        self.setMinimumSize(760, 520)  # smaller windows simply scroll
        screen = QApplication.primaryScreen().availableGeometry() if QApplication.primaryScreen() else None
        self.resize(
            min(1280, int(screen.width() * 0.94)) if screen else 1240,
            min(1000, int(screen.height() * 0.94)) if screen else 900,
        )
        icon_path = resource_path("icons", "app.png")
        if icon_path.exists():
            self.setWindowIcon(QIcon(str(icon_path)))
        self.setAcceptDrops(True)

        self._build_ui()
        self._restore_geometry()
        QApplication.instance().installEventFilter(self)  # type: ignore[union-attr]
        self.files_panel.set_filter(self.settings.default_file_filter)
        self.command_panel.set_strategy(self.settings.routing_strategy)
        self.command_panel.refresh_models()
        self.command_panel.set_model_choice(self.settings.preferred_model_id or AUTO)
        self.command_panel.routing_changed.connect(self._on_routing_changed)
        self.command_panel.mode_combo.currentIndexChanged.connect(lambda _i: self._update_ai_label())
        self.refresh_models_ui()
        self.refresh_search_ui()
        self.refresh_prompt_menus()
        self.refresh_undo_button()
        self.show_status(tr("status.ready"))
        if self.settings.reopen_last_folder and self.settings.last_folder and Path(self.settings.last_folder).is_dir():
            self.load_folder(self.settings.last_folder)

    # ================================================================== build
    def _build_ui(self) -> None:
        root = QWidget()
        root.setObjectName("Root")
        self.setCentralWidget(root)
        layout = QVBoxLayout(root)
        layout.setContentsMargins(18, 14, 18, 12)
        layout.setSpacing(12)

        layout.addLayout(self._build_header())

        # One natural vertical page scroll for every section (the header and status bar stay fixed).
        self.page = QWidget()
        self.page.setObjectName("Root")
        page_layout = QVBoxLayout(self.page)
        page_layout.setContentsMargins(0, 0, 6, 4)
        page_layout.setSpacing(14)
        self.scroll = QScrollArea()
        self.scroll.setObjectName("PageScroll")
        self.scroll.setWidgetResizable(True)
        self.scroll.setFrameShape(QFrame.Shape.NoFrame)
        self.scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAsNeeded)
        self.scroll.setVerticalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAsNeeded)
        self.scroll.setWidget(self.page)
        self.scroll.viewport().installEventFilter(self)
        layout.addWidget(self.scroll, 1)

        self.welcome = self._build_welcome()
        page_layout.addWidget(self.welcome)
        self.folder_card = self._build_folder_card()
        page_layout.addWidget(self.folder_card)
        self.files_panel = FilesPanel(self._icons)
        self.files_panel.selection_changed.connect(self._on_selection_changed)
        self.files_panel.visible_rows_changed.connect(self._request_visible_metadata)
        self.files_panel.current_entry_changed.connect(self._on_current_entry)
        self.files_panel.details.set_capabilities(probe=self._probe.available, ffmpeg=self._thumbs.available)
        self.command_panel = CommandPanel(self._icons, self.ctx.registry, self.ctx.db)
        self.title_panel = self.command_panel.title_panel
        assert self.title_panel is not None
        self.title_panel.set_config(load_current(self.ctx.db))
        self.title_panel.changed.connect(self._on_title_settings_changed)
        self.command_panel.generate_requested.connect(self.generate_preview)
        self.command_panel.cancel_requested.connect(self.cancel_planning)
        self.command_panel.saved_prompt_chosen.connect(self._use_saved_prompt)
        self.command_panel.manage_prompts_requested.connect(self.open_prompts)
        self.command_panel.save_prompt_requested.connect(self.save_current_as_prompt)
        self.command_panel.recent_command_chosen.connect(self._use_recent_command)
        self.preview_panel = PreviewPanel(self._icons)
        self.preview_panel.set_palette(self._palette)
        self.preview_panel.apply_requested.connect(self.apply_changes)
        self.preview_panel.cancel_requested.connect(self.clear_preview)
        page_layout.addWidget(self.files_panel, 3)
        page_layout.addWidget(self.command_panel, 0)
        page_layout.addWidget(self.preview_panel, 3)
        for combo in self.page.findChildren(QComboBox):  # dropdowns may shrink on small windows
            make_compact(combo)
        make_compact(self.files_panel.filter_combo, 7)  # these two have no stretch: keep their labels readable
        make_compact(self.files_panel.sort_combo, 10)
        self._fit_sections()

        self._build_status_bar()
        self.add_header_button("prompts", "star", "header.prompts", "header.prompts_tip", self.open_prompts)
        self.add_header_button("history", "history", "header.history", "header.history_tip", self.open_history)
        self.add_header_button("undo", "undo", "header.undo", "header.undo_tip", self.undo_last)
        self.add_header_button("settings", "settings", "header.settings", "header.settings_tip", self.open_settings)

    def reveal_preview(self) -> None:
        """Scroll the page so the top of the Preview card is visible."""
        self.scroll.ensureWidgetVisible(self.preview_panel.title_label, 0, 24)

    def _fit_sections(self) -> None:
        """Give the Files and Preview cards a height that follows the window (more rows on big screens).

        The page itself scrolls, so nothing is ever squeezed out of reach on small windows; the
        tables keep their own scroll bar only for long file lists.
        """
        available = self.scroll.viewport().height() if hasattr(self, "scroll") else 700
        if hasattr(self, "scroll"):
            self.title_panel.set_available_width(self.scroll.viewport().width())
            self.command_panel.set_available_width(self.scroll.viewport().width())
            self._place_folder_row(narrow=self.scroll.viewport().width() < 900)
        self.files_panel.setMinimumHeight(max(360, int(available * 0.62)))
        self.preview_panel.setMinimumHeight(max(380, int(available * 0.62)))

    def _build_welcome(self) -> QFrame:
        """First-start banner, shown only while no AI model is configured."""
        frame = QFrame()
        frame.setObjectName("Card")
        row = QHBoxLayout(frame)
        row.setContentsMargins(16, 12, 16, 12)
        texts = QVBoxLayout()
        title = QLabel()
        title.setObjectName("CardTitle")
        self._tr.text(title, "welcome.title")
        body = QLabel()
        body.setObjectName("Muted")
        body.setWordWrap(True)
        self._tr.text(body, "welcome.body")
        texts.addWidget(title)
        texts.addWidget(body)
        row.addLayout(texts, 1)
        self.welcome_button = make_button("models.add", self._tr, icon="settings", binder=self._icons, name="Primary")
        self.welcome_button.clicked.connect(self.add_first_model)
        row.addWidget(self.welcome_button)
        return frame

    def _build_header(self) -> QHBoxLayout:
        row = QHBoxLayout()
        row.setSpacing(6)
        title = QLabel()
        title.setObjectName("AppTitle")
        self._tr.bind(lambda t: title.setText(t.upper()), "app.title")
        row.addWidget(title)
        row.addStretch(1)
        self.header_layout = row
        self.header_buttons: dict[str, QPushButton] = {}
        return row

    def add_header_button(self, key: str, icon_name: str, text_key: str, tip_key: str, handler: Callable[[], None]) -> QPushButton:
        """Add a button to the top-right of the header (Prompts / History / Undo / Settings)."""
        button = make_button(
            text_key, self._tr, icon=icon_name, binder=self._icons, name="HeaderButton", tooltip_key=tip_key, icon_size=18
        )
        button.clicked.connect(handler)
        self.header_buttons[key] = button
        self.header_layout.addWidget(button)
        return button

    def _build_folder_card(self) -> Card:
        card = Card("folder.title")
        row = QHBoxLayout()
        row.setSpacing(8)
        self._folder_grid = QGridLayout()
        self._folder_grid.setHorizontalSpacing(12)
        self._folder_grid.setVerticalSpacing(8)
        self._folder_row = row
        self._folder_narrow: bool | None = None
        self.folder_edit = QLineEdit()
        self.folder_edit.setClearButtonEnabled(True)
        self.folder_edit.setAcceptDrops(False)  # drops are handled by the window
        self._tr.placeholder(self.folder_edit, "folder.placeholder")
        self.folder_edit.returnPressed.connect(self._on_folder_entered)
        row.addWidget(self.folder_edit, 1)
        self.browse_button = make_button(
            "folder.browse", self._tr, icon="folder", binder=self._icons, icon_color="#ffffff", name="Primary",
            tooltip_key="folder.browse_tip",
        )  # fmt: skip
        self.browse_button.clicked.connect(self.browse_folder)
        row.addWidget(self.browse_button)
        self.refresh_button = make_button("folder.refresh", self._tr, icon="refresh", binder=self._icons, tooltip_key="folder.refresh_tip")
        self.refresh_button.clicked.connect(self.refresh)
        row.addWidget(self.refresh_button)
        self.subfolders_check = QCheckBox()
        self._tr.text(self.subfolders_check, "folder.subfolders")
        self._tr.tooltip(self.subfolders_check, "folder.subfolders_tip")
        self.subfolders_check.setChecked(self.settings.include_subfolders)
        self.subfolders_check.toggled.connect(self._on_subfolders_toggled)
        self._place_folder_row(narrow=False)
        card.body.addLayout(self._folder_grid)
        return card

    def _place_folder_row(self, *, narrow: bool) -> None:
        """'Include subfolders' sits beside the buttons, or below the path on narrow windows."""
        if narrow == self._folder_narrow:
            return
        self._folder_narrow = narrow
        self._folder_grid.removeItem(self._folder_row)
        self._folder_grid.removeWidget(self.subfolders_check)
        self._folder_grid.addLayout(self._folder_row, 0, 0)
        if narrow:
            self._folder_grid.addWidget(self.subfolders_check, 1, 0)
        else:
            self._folder_grid.addWidget(self.subfolders_check, 0, 1)

    def _build_status_bar(self) -> None:
        bar = self.statusBar()
        bar.setSizeGripEnabled(False)
        self.status_label = QLabel()
        self.status_label.setContentsMargins(8, 0, 0, 0)
        bar.addWidget(self.status_label, 1)
        self.progress = QProgressBar()
        self.progress.setFixedWidth(220)
        self.progress.setTextVisible(False)
        self.progress.hide()
        bar.addPermanentWidget(self.progress)
        self.provider_label = QLabel()
        self.provider_label.setContentsMargins(12, 0, 8, 0)
        bar.addPermanentWidget(self.provider_label)

    # ================================================================== status
    def show_status(self, text: str, kind: str = "info") -> None:
        """Show a message in the status bar (``kind``: info / success / warning / error)."""
        self.status_label.setText(text)
        names = {"success": "Success", "warning": "Warning", "error": "Danger"}
        self.status_label.setObjectName(names.get(kind, ""))
        self.status_label.style().unpolish(self.status_label)
        self.status_label.style().polish(self.status_label)
        log.debug("status[%s]: %s", kind, text)

    def set_provider_text(self, text: str) -> None:
        self.provider_label.setText(text)

    def _set_busy(self, busy: bool, determinate: tuple[int, int] | None = None) -> None:
        if busy:
            self.progress.show()
            if determinate and determinate[1] > 0:
                self.progress.setRange(0, determinate[1])
                self.progress.setValue(determinate[0])
            else:
                self.progress.setRange(0, 0)
        else:
            self.progress.hide()

    # ================================================================= workers
    def run_worker(
        self,
        worker: BaseWorker,
        *,
        on_result: Callable[[object], None] | None = None,
        on_error: Callable[[str, object], None] | None = None,
        on_cancelled: Callable[[], None] | None = None,
        on_progress: Callable[[int, int, str], None] | None = None,
        busy: bool = True,
    ) -> BaseWorker:
        """Start ``worker`` on the thread pool; the window keeps a reference until it finishes."""
        self._workers.add(worker)
        if busy:
            self._busy += 1
            self._set_busy(True)
        signals = worker.signals
        if on_result:
            signals.result.connect(on_result)
        signals.error.connect(on_error or (lambda msg, _exc: self._show_error(msg)))
        if on_cancelled:
            signals.cancelled.connect(on_cancelled)
        signals.progress.connect(on_progress or self._default_progress)
        signals.finished.connect(lambda: self._worker_finished(worker, busy))
        self._pool.start(worker)
        return worker

    def _default_progress(self, done: int, total: int, message: str) -> None:
        if message:
            self.show_status(message)
        if total > 0:
            self._set_busy(True, (done, total))

    def _worker_finished(self, worker: BaseWorker, counted: bool) -> None:
        self._workers.discard(worker)
        if counted:
            self._busy = max(0, self._busy - 1)
            if self._busy == 0:
                self._set_busy(False)

    def _show_error(self, message: str) -> None:
        self.show_status(message.splitlines()[0] if message else "", "error")
        QMessageBox.warning(self, APP_NAME, message)

    # =================================================================== folder
    def browse_folder(self) -> None:
        start = str(self.workspace) if self.workspace else self.settings.last_folder or str(Path.home())
        chosen = QFileDialog.getExistingDirectory(self, tr("folder.dialog_title"), start)
        if chosen:
            self.load_folder(chosen)

    def _on_folder_entered(self) -> None:
        text = self.folder_edit.text().strip().strip('"')
        if text:
            self.load_folder(text)

    def load_folder(self, path: str | Path) -> None:
        """Select ``path`` as the workspace and scan it."""
        folder = Path(path)
        if not folder.is_dir():
            message = tr("error.folder_missing", path=str(folder))
            self.show_status(message, "error")
            QMessageBox.warning(self, APP_NAME, message)
            return
        if self.workspace is not None and folder != self.workspace:
            self.clear_preview()
        self.workspace = folder
        self.folder_edit.setText(str(folder))
        self.ctx.settings.update(last_folder=str(folder))
        self.settings = self.ctx.settings.load()
        self.refresh()

    def refresh(self) -> None:
        """Rescan the current folder."""
        if self.workspace is None:
            self.show_status(tr("status.choose_folder"), "warning")
            return
        self._scan_generation += 1
        generation = self._scan_generation
        worker = ScanWorker(self.workspace, self.subfolders_check.isChecked())
        if self._sticky_status is None:
            self.show_status(tr("status.scanning"))
        self.run_worker(
            worker,
            on_result=lambda res: self._on_scan_done(generation, res),  # type: ignore[arg-type]
            on_error=lambda msg, _e: self._on_scan_failed(generation, msg),
        )

    def _on_scan_done(self, generation: int, result: ScanResult) -> None:
        if generation != self._scan_generation:
            return  # a newer scan superseded this one
        self._reset_metadata()
        self.files_panel.set_entries(result.entries)
        if self._pending_paths:
            wanted, self._pending_paths = self._pending_paths, set()
            self.files_panel.check_paths(wanted)
        if self._pending_statuses:
            statuses, self._pending_statuses = self._pending_statuses, {}
            self.files_panel.model.set_status_map(statuses)
        if self._sticky_status is not None:
            text, kind = self._sticky_status
            self._sticky_status = None
            self.show_status(text, kind)
        else:
            self.show_status(tr("status.files_found_in", count=len(result.entries), folder=result.root.name or str(result.root)))
        log.info("Folder loaded: %d files", len(result.entries))

    def _on_scan_failed(self, generation: int, message: str) -> None:
        if generation != self._scan_generation:
            return
        self.files_panel.set_entries([])
        self._show_error(message)

    def _on_subfolders_toggled(self, checked: bool) -> None:
        self.ctx.settings.update(include_subfolders=checked)
        self.settings = self.ctx.settings.load()
        if self.workspace is not None:
            self.refresh()

    def _on_selection_changed(self, count: int) -> None:
        """Hook: the selection changed (the preview is only re-generated on request)."""

    # ===================================================== metadata / thumbnails
    def _reset_metadata(self) -> None:
        """A new scan invalidates everything learned about the previous file list."""
        self._meta_generation += 1
        self._meta.clear()
        self._meta_requested.clear()
        self.files_panel.details.show_entry(None, None)

    def _request_visible_metadata(self) -> None:
        """Lazily probe only the rows that are on screen (ffprobe reads headers, never full videos)."""
        if not self.settings.read_metadata or not self._probe.available or self.workspace is None:
            return
        wanted: list[FileEntry] = [
            e for e in self.files_panel.visible_viewport_entries()
            if e.kind in (FileKind.VIDEO, FileKind.AUDIO) and e.rel_path not in self._meta_requested
        ][:60]  # fmt: skip
        if not wanted:
            return
        self._meta_requested.update(e.rel_path for e in wanted)
        generation = self._meta_generation
        worker = MetadataWorker(self.workspace, wanted, self._probe, self.ctx.db)
        worker.signals.item.connect(lambda rel, meta: self._on_metadata_item(generation, rel, meta))  # type: ignore[attr-defined]
        self._workers.add(worker)
        worker.signals.finished.connect(lambda: self._workers.discard(worker))
        media_pool().start(worker)

    def _on_metadata_item(self, generation: int, rel_path: str, meta: object) -> None:
        if generation != self._meta_generation:
            return
        value = meta if isinstance(meta, VideoMetadata) else None
        self._meta[rel_path] = value
        self.files_panel.model.set_duration(rel_path, value.duration if value else None)
        details = self.files_panel.details
        if details.current_rel_path == rel_path:
            entry = self.files_panel.model.entry_for_path(rel_path)
            details.show_entry(entry, value)
            self._thumb_timer.start()

    def _on_current_entry(self, entry: object) -> None:
        details = self.files_panel.details
        if not isinstance(entry, FileEntry):
            details.show_entry(None, None)
            return
        details.show_entry(entry, self._meta.get(entry.rel_path))
        if entry.kind == FileKind.VIDEO:
            self._thumb_timer.start()  # debounced: arrowing through rows must not spawn ffmpeg per row
        if (
            self.settings.read_metadata and self._probe.available and entry.rel_path not in self._meta_requested
            and entry.kind in (FileKind.VIDEO, FileKind.AUDIO)
        ):  # fmt: skip
            self._request_visible_metadata()

    def _request_thumbnail(self) -> None:
        details = self.files_panel.details
        rel = details.current_rel_path
        entry = self.files_panel.model.entry_for_path(rel) if rel else None
        if entry is None or self.workspace is None or entry.kind != FileKind.VIDEO or not self._thumbs.available:
            return
        cached = self._thumbs.cached(entry.absolute(self.workspace))
        if cached is not None:
            details.set_thumbnail(str(cached))
            return
        generation = self._meta_generation
        worker = ThumbnailWorker(self.workspace, entry, self._thumbs)
        worker.signals.thumbnail.connect(lambda r, path: self._on_thumbnail(generation, r, path))  # type: ignore[attr-defined]
        self._workers.add(worker)
        worker.signals.finished.connect(lambda: self._workers.discard(worker))
        media_pool().start(worker)

    def _on_thumbnail(self, generation: int, rel_path: str, image_path: str) -> None:
        details = self.files_panel.details
        if generation == self._meta_generation and details.current_rel_path == rel_path and image_path:
            details.set_thumbnail(image_path)

    # ============================================================ prompt/history
    def refresh_prompt_menus(self) -> None:
        """Rebuild the Saved Prompts / Recent Commands menus from the database."""
        self.command_panel.set_prompts(self.ctx.db.list_saved_prompts(), self.ctx.db.list_commands(30))

    def _use_saved_prompt(self, prompt: SavedPrompt) -> None:
        self.command_panel.set_command(prompt.prompt)
        self._choose_model(prompt.provider)

    def _use_recent_command(self, record: CommandRecord) -> None:
        self.command_panel.set_command(record.command)
        self._choose_model(record.provider)

    def _choose_model(self, value: str) -> None:
        """Select the model a saved prompt / recent command asks for (``auto`` keeps the current choice)."""
        if value and value != AUTO:
            self.command_panel.set_model_choice(value)
            self._on_routing_changed()

    def open_prompts(self, prefill: tuple[str, str] | None = None) -> None:
        """Saved Prompts / Command History dialog (``Use`` copies a command into the box)."""
        dialog = SavedPromptsDialog(self.ctx, self, prefill=prefill if isinstance(prefill, tuple) else None)
        dialog.use_requested.connect(self._on_prompt_used)
        dialog.changed.connect(self.refresh_prompt_menus)
        dialog.exec()
        dialog.deleteLater()
        self.refresh_prompt_menus()

    def _on_prompt_used(self, text: str, provider: str) -> None:
        self.command_panel.set_command(text)
        self._choose_model(provider)

    def save_current_as_prompt(self) -> None:
        """Quick-save the command in the box as a Saved Prompt (asks only for a name)."""
        command = self.command_panel.command()
        if not command:
            self.show_status(tr("pipeline.no_command"), "warning")
            return
        default_name = command.splitlines()[0][:40]
        name, ok = QInputDialog.getText(self, tr("prompts.title"), tr("command.prompt_name"), text=default_name)
        if ok and name.strip():
            self.ctx.db.add_saved_prompt(name.strip(), command, self.command_panel.model_choice())
            self.refresh_prompt_menus()
            self.show_status(tr("status.prompt_saved", name=name.strip()), "success")

    def open_history(self) -> None:
        """Operation history with Undo."""
        dialog = HistoryDialog(self.ctx, self)
        dialog.undo_requested.connect(self.undo_operation)
        dialog.exec()
        dialog.deleteLater()
        self.refresh_undo_button()

    # ================================================================== preview
    def generate_preview(self) -> None:
        """Ask for a plan (offline parse or AI) for the typed command and show it in the preview."""
        if self._operation_running or self._plan_worker is not None:
            return
        if self.workspace is None:
            self.show_status(tr("status.choose_folder"), "warning")
            return
        command = self.command_panel.command()
        title_config = self.title_panel.config()
        if not command and not title_config.enabled:
            self.show_status(tr("pipeline.no_command"), "warning")
            self.command_panel.edit.setFocus()
            return
        files = self.files_panel.selected_entries()
        if not files:
            self.show_status(tr("pipeline.no_files"), "warning")
            return
        settings = self.ctx.settings.load()
        self.settings = settings
        request = PipelineRequest(
            command=command, files=files, workspace=self.workspace, processing=self.command_panel.processing(),
            settings=settings, title=title_config if title_config.enabled else None,
            search=make_search_service(self.ctx) if title_config.enabled and title_config.uses_search else None,
        )  # fmt: skip
        self._plan_generation += 1
        generation = self._plan_generation
        self._active_command = command or tr("tg.history_command", count=len(files))
        router = self.make_router()
        self._update_ai_label(router, command, len(files))
        worker = PlanWorker(request, router)
        self._plan_worker = worker
        self.command_panel.set_busy(True)
        self.preview_panel.clear()
        self.show_status(tr("status.planning"))
        log.info(
            "Generate preview: %d files, strategy=%s, processing=%s", len(files), router.options.strategy.value,
            request.processing.value,
        )  # fmt: skip
        self.run_worker(
            worker,
            on_result=lambda res: self._on_plan_ready(generation, res),  # type: ignore[arg-type]
            on_error=lambda msg, _e: self._on_plan_failed(generation, msg),
            on_cancelled=lambda: self._on_plan_cancelled(generation),
        )

    def cancel_planning(self) -> None:
        if self._plan_worker is not None:
            self._plan_worker.cancel()
            self._plan_generation += 1  # ignore whatever the worker still delivers
            self._plan_worker = None
            self.command_panel.set_busy(False)
            self.show_status(tr("status.planning_cancelled"))

    # ================================================================== models
    def make_router(self) -> ModelRouter:
        """Router for the AI mode / model chosen in the command area."""
        choice = self.command_panel.model_choice()
        return make_router(
            self.ctx.registry, self.ctx.health, self.ctx.settings.load(), strategy=self.command_panel.strategy(),
            preferred_model_id="" if choice == AUTO else choice,
        )  # fmt: skip

    def _update_ai_label(self, router: ModelRouter | None = None, command: str = "", file_count: int = 0) -> None:
        """``AI: Gemini / gemini-2.5-flash`` - the model the next request would start with."""
        if self.command_panel.processing() == ProcessingMode.OFFLINE_ONLY:
            self.command_panel.set_ai_text(tr("command.ai_offline"))
            return
        router = router or self.make_router()
        profile = analyze_task(command, file_count) if command else None
        model = router.preview(profile)
        self.command_panel.set_ai_text(tr("command.ai_next", model=model.label) if model else tr("command.ai_none"))

    def _on_routing_changed(self) -> None:
        """AI mode / model changed in the command area: auto-save it (non-sensitive preference)."""
        choice = self.command_panel.model_choice()
        self.settings = self.ctx.settings.update(
            routing_strategy=self.command_panel.strategy().value, preferred_model_id="" if choice == AUTO else choice
        )
        self._update_ai_label()

    def _on_title_settings_changed(self) -> None:
        """Title Generator settings are non-sensitive preferences: auto-saved on every change."""
        save_current(self.ctx.db, self.title_panel.config())

    def refresh_search_ui(self) -> None:
        self.title_panel.set_search_configured(make_search_service(self.ctx).configured)

    def refresh_models_ui(self) -> None:
        """Models were added / edited / removed: update the combos, the welcome banner and the AI label."""
        self.command_panel.refresh_models()
        self.welcome.setVisible(len(self.ctx.registry) == 0)
        self._update_ai_label()

    def add_first_model(self) -> None:
        """The welcome banner's *Add Model* button."""
        settings = self.ctx.settings.load()
        dialog = ModelDialog(
            self.ctx.registry, None, self, default_timeout=settings.request_timeout_s,
            default_retries=settings.default_max_retries,
        )  # fmt: skip
        if dialog.exec() == QDialog.DialogCode.Accepted:
            self.refresh_models_ui()
            self.show_status(tr("models.saved", name=dialog.saved[0].display_name), "success")
        dialog.deleteLater()

    def _on_plan_failed(self, generation: int, message: str) -> None:
        if generation != self._plan_generation:
            return
        self._plan_worker = None
        self.command_panel.set_busy(False)
        self.preview_panel.clear()
        self.set_provider_text("")
        self._update_ai_label()
        self._show_error(message)

    def _on_plan_cancelled(self, generation: int) -> None:
        if generation == self._plan_generation:
            self._plan_worker = None
            self.command_panel.set_busy(False)
            self.show_status(tr("status.planning_cancelled"))

    def _on_plan_ready(self, generation: int, result: PipelineResult) -> None:
        if generation != self._plan_generation:
            return
        self._plan_worker = None
        self.command_panel.set_busy(False)
        plan = result.plan
        self._active_provider = result.provider_used
        if self.command_panel.command():  # (an empty command with the Title Generator is not history-worthy)
            self.ctx.db.add_command(self._active_command, self.command_panel.model_choice())
        self.refresh_prompt_menus()
        self.set_provider_text(tr("status.provider", text=result.provider_text))
        if result.source == "ai":
            self.command_panel.set_ai_text("   ".join([tr("command.ai_used", text=result.provider_text), *(
                result.route_details if result.fallbacks else [])]))  # fmt: skip
        else:
            self.command_panel.set_ai_text(result.provider_text)
        info = " — ".join(filter(None, [result.completion_text, plan.summary]))
        self.preview_panel.set_plan(plan, info)
        QTimer.singleShot(0, self.reveal_preview)  # bring the new preview into view (the page scrolls)
        self._mark_file_statuses(plan)
        if plan.sort is not None:
            self.files_panel.apply_sort(plan.sort.by, plan.sort.descending)
        log.info("Plan ready via %s: %s", result.provider_used, dict(plan.counts()))

        count = self.preview_panel.apply_count()
        if count == 0 and plan.sort is not None:
            self.show_status(tr("status.sorted_only", text=plan.summary or ""), "success")
            return
        if count == 0:
            self.show_status(tr("status.nothing_to_change"), "warning")
            return
        self.show_status(tr("status.plan_ready", count=count), "success")
        if self._can_auto_apply(plan):
            self._start_apply(plan)

    def _can_auto_apply(self, plan: Plan) -> bool:
        """Auto Apply never fires for plans with deletes, conflicts or invalid items."""
        return self.settings.auto_apply and not plan.has_deletes and not plan.has_blockers and not plan.is_empty

    def _mark_file_statuses(self, plan: Plan) -> None:
        labels = {
            OpKind.RENAME: "preview.rename", OpKind.MOVE: "preview.move",
            OpKind.COPY: "preview.copy", OpKind.DELETE: "preview.delete",
        }  # fmt: skip
        statuses = {
            op.source: "→ " + tr(labels[op.kind])
            for op in plan.applicable_ops()
            if op.source and op.kind in labels
        }
        self.files_panel.model.set_status_map(statuses)

    def clear_preview(self) -> None:
        """Discard the current preview (nothing was changed on disk)."""
        self.preview_panel.clear()
        self.files_panel.model.set_status_map({})
        self.set_provider_text("")
        if self.workspace is not None:
            self.show_status(tr("status.ready"))

    # =================================================================== apply
    def apply_changes(self) -> None:
        """Confirm, then execute the previewed plan."""
        plan = self.preview_panel.plan
        if plan is None or self.preview_panel.apply_count() == 0 or self._operation_running:
            return
        if not self.confirm_apply(plan):
            return
        self._start_apply(plan)

    def confirm_apply(self, plan: Plan) -> bool:
        """The confirmation dialogs: always for deletes, otherwise per the *Ask before applying* setting."""
        count = self.preview_panel.apply_count()
        if self.settings.ask_before_apply:
            if not self._ask(tr("confirm.apply_title"), tr("confirm.apply", count=count), tr("confirm.apply_button")):
                return False
        deletes = len(plan.delete_ops)
        if deletes:
            key = "confirm.delete_permanent" if plan.delete_mode == "permanent" else "confirm.delete_trash"
            if not self._ask(tr("confirm.delete_title"), tr(key, count=deletes), tr("confirm.delete_button"), destructive=True):
                return False
        return True

    def _ask(self, title: str, text: str, ok_text: str, *, destructive: bool = False) -> bool:
        box = QMessageBox(self)
        box.setWindowTitle(title)
        box.setIcon(QMessageBox.Icon.Warning if destructive else QMessageBox.Icon.Question)
        box.setText(text)
        ok = box.addButton(ok_text, QMessageBox.ButtonRole.DestructiveRole if destructive else QMessageBox.ButtonRole.AcceptRole)
        cancel = box.addButton(tr("common.cancel"), QMessageBox.ButtonRole.RejectRole)
        box.setDefaultButton(cancel if destructive else ok)
        box.exec()
        return box.clickedButton() is ok

    def _start_apply(self, plan: Plan) -> None:
        self._operation_running = True
        self._set_controls_locked(True)
        selected_before = {e.rel_path for e in self.files_panel.selected_entries()}
        for op in plan.applicable_ops():
            if op.kind in (OpKind.RENAME, OpKind.MOVE, OpKind.DELETE) and op.source:
                selected_before.discard(op.source)
                if op.kind != OpKind.DELETE and op.target:
                    selected_before.add(op.target)
        self._pending_paths = selected_before
        settings = self.ctx.settings.load()
        worker = OperationWorker(
            self.manager, plan, command=self._active_command, provider=self._active_provider, keep_history=settings.create_history
        )
        self.show_status(tr("status.applying"))
        self.run_worker(
            worker,
            on_result=lambda res: self._on_apply_done(res),  # type: ignore[arg-type]
            on_error=lambda msg, _e: self._on_apply_crashed(msg),
        )

    def _on_apply_done(self, result: ExecutionResult) -> None:
        self._operation_running = False
        self._set_controls_locked(False)
        self.refresh_undo_button()
        if result.ok:
            self._pending_statuses = dict(result.changes)
            self.preview_panel.clear()
            note = f" {tr('status.irreversible', count=result.irreversible)}" if result.irreversible else ""
            message = tr("status.applied", count=result.applied, id=result.operation_id or "-") + note
            self._sticky_status = (message, "success")
            self.show_status(message, "success")
            self.refresh()
            return
        self._pending_paths = set()
        key = "error.apply_failed_rolled_back" if result.rolled_back else "error.apply_failed"
        message = tr(key, reason=result.message)
        self.show_status(message.splitlines()[0], "error")
        QMessageBox.warning(self, APP_NAME, message)
        self.preview_panel.clear()
        self.refresh()

    def _on_apply_crashed(self, message: str) -> None:
        self._operation_running = False
        self._set_controls_locked(False)
        self._pending_paths = set()
        self._show_error(tr("error.apply_failed", reason=message))
        self.refresh()

    def _set_controls_locked(self, locked: bool) -> None:
        """Disable inputs that could change the folder while a batch is running."""
        for widget in (
            self.browse_button, self.refresh_button, self.folder_edit, self.subfolders_check,
            self.command_panel, self.preview_panel, self.files_panel,
        ):  # fmt: skip
            widget.setEnabled(not locked)
        for key in ("undo", "history", "settings", "prompts"):
            if key in self.header_buttons:
                self.header_buttons[key].setEnabled(not locked)

    # ==================================================================== undo
    def refresh_undo_button(self) -> None:
        button = self.header_buttons.get("undo")
        if button is None:
            return
        record = self.ctx.db.last_undoable_operation()
        button.setEnabled(record is not None and not self._operation_running)
        if record is not None:
            button.setToolTip(tr("header.undo_tip_op", id=record.id, summary=record.summary))
        else:
            button.setToolTip(tr("header.undo_tip"))

    def undo_last(self) -> None:
        record = self.ctx.db.last_undoable_operation()
        if record is None:
            self.show_status(tr("status.nothing_to_undo"), "warning")
            return
        self.undo_operation(record.id)

    def undo_operation(self, operation_id: int) -> None:
        """Ask for confirmation (with a pre-check explanation if it cannot be done) and undo."""
        record = self.ctx.db.get_operation(operation_id)
        if record is None or self._operation_running:
            return
        check = self.manager.check_undo(operation_id)
        if not check.possible:
            self._explain_undo_blocked(check.blockers)
            return
        text = tr("confirm.undo", id=record.id, summary=record.summary)
        if check.notes:
            text += "\n\n" + "\n".join(f"• {n}" for n in check.notes[:6])
        if not self._ask(tr("confirm.undo_title"), text, tr("confirm.undo_button")):
            return
        self._operation_running = True
        self._set_controls_locked(True)
        self.show_status(tr("status.undoing"))
        self.run_worker(
            UndoWorker(self.manager, operation_id),
            on_result=lambda res: self._on_undo_done(res, record.workspace),  # type: ignore[arg-type]
            on_error=lambda msg, _e: self._on_undo_crashed(msg),
        )

    def _on_undo_done(self, result: UndoResult, workspace: str) -> None:
        self._operation_running = False
        self._set_controls_locked(False)
        self.refresh_undo_button()
        if result.ok:
            message = tr("status.undone", count=result.restored)
            self.show_status(message, "success")
            if self.workspace is not None and Path(workspace) == self.workspace:
                self._sticky_status = (message, "success")
                self.preview_panel.clear()
                self.refresh()
            return
        self._explain_undo_blocked(result.blockers, fallback=result.message)

    def _on_undo_crashed(self, message: str) -> None:
        self._operation_running = False
        self._set_controls_locked(False)
        self.refresh_undo_button()
        self._show_error(message)

    def _explain_undo_blocked(self, blockers: list[str], fallback: str = "") -> None:
        lines = blockers or ([fallback] if fallback else [])
        text = tr("undo.cannot") + "\n\n" + "\n".join(f"• {b}" for b in lines[:8])
        if len(lines) > 8:
            text += "\n" + tr("undo.more", count=len(lines) - 8)
        self.show_status(tr("undo.cannot"), "warning")
        QMessageBox.information(self, APP_NAME, text)

    # ================================================================ settings
    def open_settings(self) -> None:
        """Show the Settings dialog and apply whatever the user changed."""
        dialog = SettingsDialog(self.ctx, self, self._icon_color)
        dialog.saved.connect(self._on_settings_saved)
        dialog.models_changed.connect(self.refresh_models_ui)
        dialog.exec()
        dialog.deleteLater()
        self.refresh_models_ui()

    def _on_settings_saved(self) -> None:
        previous = self.settings
        self.settings = self.ctx.settings.load()
        if self.settings.language != previous.language:
            set_language(self.settings.language)
            self.retranslate()
        if self.settings.theme != previous.theme:
            apply_theme(QApplication.instance(), self.settings.theme)  # type: ignore[arg-type]
            self._palette = PALETTES[resolve_theme(self.settings.theme)]
            self._icon_color = self._palette["muted"]
            self._icons.refresh(self._icon_color)
            self.preview_panel.set_palette(self._palette)
        if self.settings.log_level != previous.log_level:
            setup_logging(self.settings.log_level)
        if self.settings.include_subfolders != previous.include_subfolders:
            self.subfolders_check.setChecked(self.settings.include_subfolders)  # triggers a rescan
        if (self.settings.routing_strategy, self.settings.preferred_model_id) != (
            previous.routing_strategy, previous.preferred_model_id
        ):  # fmt: skip
            self.command_panel.set_strategy(self.settings.routing_strategy)
            self.command_panel.set_model_choice(self.settings.preferred_model_id or AUTO)
        self.refresh_models_ui()
        self.refresh_search_ui()
        if (self.settings.ffmpeg_dir, self.settings.read_metadata) != (previous.ffmpeg_dir, previous.read_metadata):
            self._probe = MetadataProbe(self.settings.ffmpeg_dir)
            self._thumbs = ThumbnailCache(self.settings.ffmpeg_dir)
            self.files_panel.details.set_capabilities(probe=self._probe.available, ffmpeg=self._thumbs.available)
            self._meta_requested.clear()
            self._request_visible_metadata()
        self.show_status(tr("status.settings_saved"), "success")

    # ============================================================== drag & drop
    def eventFilter(self, obj: QObject, event: QEvent) -> bool:  # noqa: N802
        """Catch folder/file drops anywhere in this window (child widgets would swallow them)."""
        kind = event.type()
        if kind == QEvent.Type.Resize and hasattr(self, "scroll") and obj is self.scroll.viewport():
            self._fit_sections()
        if kind in (QEvent.Type.DragEnter, QEvent.Type.DragMove, QEvent.Type.Drop):
            if isinstance(obj, QWidget) and obj.window() is self and event.mimeData().hasUrls():  # type: ignore[attr-defined]
                if kind == QEvent.Type.Drop:
                    assert isinstance(event, QDropEvent)
                    self.handle_drop(event.mimeData().urls())
                event.accept()
                return True
        return super().eventFilter(obj, event)

    def dragEnterEvent(self, event: QDragEnterEvent) -> None:  # noqa: N802
        if event.mimeData().hasUrls():
            event.acceptProposedAction()

    def dropEvent(self, event: QDropEvent) -> None:  # noqa: N802
        self.handle_drop(event.mimeData().urls())
        event.acceptProposedAction()

    def handle_drop(self, urls: list[QUrl]) -> None:
        """A dropped folder becomes the workspace; dropped files select themselves in their folder."""
        if self._operation_running:
            return
        paths = [Path(u.toLocalFile()) for u in urls if u.isLocalFile()]
        if not paths:
            return
        first = paths[0]
        if first.is_dir():
            self.load_folder(first)
            return
        folder = first.parent
        same_folder = [p for p in paths if p.parent == folder and p.is_file()]
        skipped = len(paths) - len(same_folder)
        self._pending_paths = {p.name for p in same_folder}
        if any(classify_extension(p.suffix) != FileKind.VIDEO for p in same_folder):
            self.files_panel.set_filter("all")
        self.load_folder(folder)
        if skipped:
            self.show_status(tr("status.drop_skipped", count=skipped), "warning")

    # ===================================================================== misc
    def retranslate(self) -> None:
        self._tr.retranslate()
        self.folder_card.retranslate()
        self.files_panel.retranslate()
        self.command_panel.retranslate()
        self._update_ai_label()
        self.preview_panel.retranslate()
        self.refresh_prompt_menus()
        self.refresh_undo_button()
        if not self.status_label.text():
            self.show_status(tr("status.ready"))

    def _restore_geometry(self) -> None:
        if self.settings.window_geometry:
            self.restoreGeometry(QByteArray.fromBase64(self.settings.window_geometry.encode("ascii")))

    def closeEvent(self, event: QCloseEvent) -> None:  # noqa: N802
        if self._operation_running:
            answer = QMessageBox.question(self, APP_NAME, tr("confirm.close_busy"))
            if answer != QMessageBox.StandardButton.Yes:
                event.ignore()
                return
        for worker in list(self._workers):
            worker.cancel()
        self._pool.waitForDone(5000)
        media_pool().waitForDone(3000)
        try:
            geometry = bytes(self.saveGeometry().toBase64().data()).decode("ascii")
            self.ctx.settings.update(window_geometry=geometry)
        except Exception:  # noqa: BLE001 - never block closing
            log.debug("could not persist window geometry", exc_info=True)
        QApplication.instance().removeEventFilter(self)  # type: ignore[union-attr]
        super().closeEvent(event)
