"""Fast, cancellable folder scanning.

``os.scandir`` is used (it returns size/mtime without extra system calls on
Windows) and the scan never follows symlinks or junctions, so it cannot escape
the chosen workspace.
"""

from __future__ import annotations

import os
import stat
import threading
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path

from app.config.constants import INTERNAL_DIR_PREFIX, FileKind
from app.utils.helpers import classify_extension, natural_sort_key
from app.utils.logger import get_logger

log = get_logger("scanner")

FILE_ATTRIBUTE_HIDDEN = 0x2
FILE_ATTRIBUTE_SYSTEM = 0x4
ProgressCallback = Callable[[int], None]


@dataclass(slots=True)
class FileEntry:
    """One file found by the scanner. Paths are relative to the workspace, ``/``-separated."""

    rel_path: str
    name: str
    ext: str
    size: int
    mtime: float
    kind: FileKind
    duration: float | None = None
    checked: bool = False
    status: str = ""
    _sort_key: list[object] | None = field(default=None, repr=False, compare=False)

    @property
    def stem(self) -> str:
        return self.name[: len(self.name) - len(self.ext)] if self.ext else self.name

    @property
    def directory(self) -> str:
        """Relative directory ('' for the workspace root)."""
        head, _, _ = self.rel_path.rpartition("/")
        return head

    @property
    def natural_key(self) -> list[object]:
        if self._sort_key is None:
            self._sort_key = natural_sort_key(self.rel_path)
        return self._sort_key

    def absolute(self, root: Path) -> Path:
        return root.joinpath(*self.rel_path.split("/"))


@dataclass(slots=True)
class ScanResult:
    """Outcome of one scan."""

    root: Path
    entries: list[FileEntry]
    recursive: bool
    cancelled: bool = False
    skipped: int = 0  # unreadable / hidden entries


def _is_hidden_or_system(entry: os.DirEntry[str], st: os.stat_result) -> bool:
    attrs = getattr(st, "st_file_attributes", 0)
    return bool(attrs & (FILE_ATTRIBUTE_HIDDEN | FILE_ATTRIBUTE_SYSTEM))


def _is_junction_or_link(entry: os.DirEntry[str]) -> bool:
    if entry.is_symlink():
        return True
    is_junction = getattr(entry, "is_junction", None)
    return bool(is_junction()) if is_junction else False


def scan_folder(
    root: Path | str,
    *,
    recursive: bool = False,
    cancel: threading.Event | None = None,
    progress: ProgressCallback | None = None,
) -> ScanResult:
    """Scan ``root`` and return its files.

    Only the selected folder is scanned unless ``recursive`` is true. Hidden/system
    files and the application's own trash folder are skipped.
    """
    root_path = Path(root)
    if not root_path.is_dir():
        raise FileNotFoundError(f"Folder not found: {root_path}")
    cancel = cancel or threading.Event()
    entries: list[FileEntry] = []
    skipped = 0
    stack: list[tuple[str, str]] = [(str(root_path), "")]  # (absolute dir, relative prefix)
    while stack:
        if cancel.is_set():
            return ScanResult(root_path, entries, recursive, cancelled=True, skipped=skipped)
        directory, prefix = stack.pop()
        try:
            with os.scandir(directory) as it:
                for entry in it:
                    try:
                        if entry.is_dir(follow_symlinks=False):
                            if (
                                recursive
                                and not entry.name.startswith(INTERNAL_DIR_PREFIX)
                                and not _is_junction_or_link(entry)
                            ):
                                stack.append((entry.path, f"{prefix}{entry.name}/"))
                            continue
                        if not entry.is_file(follow_symlinks=False):
                            skipped += 1  # symlinks, devices, ...
                            continue
                        st = entry.stat(follow_symlinks=False)
                        if _is_hidden_or_system(entry, st):
                            skipped += 1
                            continue
                        ext = os.path.splitext(entry.name)[1]
                        entries.append(
                            FileEntry(
                                rel_path=f"{prefix}{entry.name}",
                                name=entry.name,
                                ext=ext.lower(),
                                size=st.st_size,
                                mtime=st.st_mtime,
                                kind=classify_extension(ext),
                            )
                        )
                    except OSError as exc:  # entry vanished / unreadable
                        skipped += 1
                        log.debug("Skipping %s: %s", entry.name, exc)
        except OSError as exc:
            skipped += 1
            log.warning("Cannot read folder %s: %s", directory, exc)
            if directory == str(root_path):
                raise
        if progress is not None:
            progress(len(entries))
    entries.sort(key=lambda e: e.natural_key)
    log.info("Scanned %s: %d files (recursive=%s, skipped=%d)", root_path.name, len(entries), recursive, skipped)
    return ScanResult(root_path, entries, recursive, skipped=skipped)


def is_regular_file(path: Path) -> bool:
    """True if ``path`` is a regular (non-symlink) file."""
    try:
        mode = path.lstat().st_mode
    except OSError:
        return False
    return stat.S_ISREG(mode)
