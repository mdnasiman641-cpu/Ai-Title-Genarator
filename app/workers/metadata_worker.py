"""Background ffprobe / thumbnail jobs (on their own small thread pool, never blocking the UI)."""

from __future__ import annotations

from pathlib import Path

from PySide6.QtCore import QThreadPool, Signal

from app.database.database import Database
from app.files.metadata import MetadataProbe, VideoMetadata
from app.files.scanner import FileEntry
from app.files.thumbnails import ThumbnailCache
from app.workers.base_worker import BaseWorker, WorkerSignals

_pool: QThreadPool | None = None


def media_pool() -> QThreadPool:
    """A separate pool (2 threads) so metadata work never starves scans, AI calls or file operations."""
    global _pool
    if _pool is None:
        _pool = QThreadPool()
        _pool.setMaxThreadCount(2)
    return _pool


class MetadataSignals(WorkerSignals):
    item = Signal(str, object)  # relative path, VideoMetadata | None


class MetadataWorker(BaseWorker):
    """Probes the given files one by one and emits each result as soon as it is known."""

    signals_class = MetadataSignals

    def __init__(self, root: Path, entries: list[FileEntry], probe: MetadataProbe, db: Database | None = None) -> None:
        super().__init__()
        self.root = root
        self.entries = entries
        self.probe = probe
        self.db = db

    def execute(self) -> int:
        done = 0
        for entry in self.entries:
            self.check_cancelled()
            meta = self._lookup(entry)
            self.signals.item.emit(entry.rel_path, meta)  # type: ignore[attr-defined]
            done += 1
        return done

    def _lookup(self, entry: FileEntry) -> VideoMetadata | None:
        path = entry.absolute(self.root)
        if self.db is not None:
            cached = self.db.get_cached_metadata(str(path), entry.size, entry.mtime)
            if cached is not None:
                return VideoMetadata.from_dict(cached) if cached else None
        meta = self.probe.probe(path)
        if self.db is not None:
            self.db.put_cached_metadata(str(path), entry.size, entry.mtime, meta.to_dict() if meta else {})
        return meta


class ThumbnailSignals(WorkerSignals):
    thumbnail = Signal(str, str)  # relative path, image path ('' if none)


class ThumbnailWorker(BaseWorker):
    """Creates (or fetches from cache) the thumbnail of one video."""

    signals_class = ThumbnailSignals

    def __init__(self, root: Path, entry: FileEntry, cache: ThumbnailCache) -> None:
        super().__init__()
        self.root = root
        self.entry = entry
        self.cache = cache

    def execute(self) -> str:
        path = self.cache.get(self.entry.absolute(self.root), self.entry.duration)
        text = str(path) if path else ""
        self.signals.thumbnail.emit(self.entry.rel_path, text)  # type: ignore[attr-defined]
        return text
