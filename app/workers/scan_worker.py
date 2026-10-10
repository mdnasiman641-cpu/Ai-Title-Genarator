"""Background folder scan."""

from __future__ import annotations

from pathlib import Path

from app.files.scanner import ScanResult, scan_folder
from app.i18n import tr
from app.utils.helpers import describe_os_error
from app.workers.base_worker import BaseWorker, CancelledError


class ScanWorker(BaseWorker):
    """Scans a folder without blocking the UI; emits a :class:`ScanResult`."""

    def __init__(self, folder: Path | str, recursive: bool) -> None:
        super().__init__()
        self.folder = Path(folder)
        self.recursive = recursive

    def execute(self) -> ScanResult:
        def progress(count: int) -> None:
            self.emit_progress(count, 0, tr("status.scanning_count", count=count))

        result = scan_folder(self.folder, recursive=self.recursive, cancel=self.cancel_event, progress=progress)
        if result.cancelled:
            raise CancelledError
        return result

    def friendly_error(self, exc: BaseException) -> str:
        if isinstance(exc, FileNotFoundError):
            return tr("error.folder_missing", path=str(self.folder))
        if isinstance(exc, OSError):
            return tr("error.folder_unreadable", path=str(self.folder), reason=describe_os_error(exc))
        return super().friendly_error(exc)
