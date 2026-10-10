"""Small, dependency-free helpers shared across the application."""

from __future__ import annotations

import errno
import os
import re
import sys
from datetime import datetime
from pathlib import Path

from app.config.constants import (
    APP_NAME,
    AUDIO_EXTENSIONS,
    DATA_DIR_ENV,
    DOCUMENT_EXTENSIONS,
    IMAGE_EXTENSIONS,
    VIDEO_EXTENSIONS,
    FileKind,
)

_DIGITS = re.compile(r"(\d+)")


# ----------------------------------------------------------------- locations
def get_app_data_dir() -> Path:
    """Return (and create) the per-user directory for the database, logs and caches."""
    override = os.environ.get(DATA_DIR_ENV)
    if override:
        base = Path(override)
    elif sys.platform == "win32":
        base = Path(os.environ.get("APPDATA") or Path.home() / "AppData" / "Roaming") / APP_NAME
    else:
        xdg = os.environ.get("XDG_DATA_HOME")
        base = (Path(xdg) if xdg else Path.home() / ".local" / "share") / APP_NAME.replace(" ", "")
    base.mkdir(parents=True, exist_ok=True)
    return base


def resource_path(*parts: str) -> Path:
    """Locate a bundled resource both from source and from a PyInstaller build."""
    if getattr(sys, "frozen", False):
        root = Path(getattr(sys, "_MEIPASS", Path(sys.executable).parent))
    else:
        root = Path(__file__).resolve().parent.parent.parent
    return root.joinpath("app", "resources", *parts)


def bundled_binary(name: str) -> Path | None:
    """Return the path of a binary shipped next to the app (e.g. ``ffprobe.exe``)."""
    candidates = [resource_path("bin", name)]
    if getattr(sys, "frozen", False):
        candidates.append(Path(sys.executable).parent / "bin" / name)
    for candidate in candidates:
        if candidate.is_file():
            return candidate
    return None


# --------------------------------------------------------------- formatting
def natural_sort_key(text: str) -> list[object]:
    """Key that orders ``ep2`` before ``ep10`` and ignores case."""
    parts = _DIGITS.split(text)
    return [int(p) if i % 2 else p.casefold() for i, p in enumerate(parts)]


def format_size(num_bytes: int | float) -> str:
    """Human readable file size, e.g. ``1.4 GB``."""
    size = float(num_bytes)
    for unit in ("B", "KB", "MB", "GB", "TB"):
        if size < 1024 or unit == "TB":
            return f"{int(size)} {unit}" if unit == "B" else f"{size:.1f} {unit}"
        size /= 1024
    return f"{size:.1f} TB"  # pragma: no cover - loop always returns


def format_duration(seconds: float | None) -> str:
    """``H:MM:SS`` (or ``M:SS``) for a duration; empty string when unknown."""
    if seconds is None or seconds < 0:
        return ""
    total = int(round(seconds))
    hours, rest = divmod(total, 3600)
    minutes, secs = divmod(rest, 60)
    return f"{hours}:{minutes:02d}:{secs:02d}" if hours else f"{minutes}:{secs:02d}"


def format_timestamp(timestamp: float) -> str:
    """Local ``YYYY-MM-DD HH:MM`` for a POSIX timestamp."""
    try:
        return datetime.fromtimestamp(timestamp).strftime("%Y-%m-%d %H:%M")
    except (OverflowError, OSError, ValueError):
        return ""


def now_iso() -> str:
    """Current local time as an ISO-8601 string without microseconds."""
    return datetime.now().replace(microsecond=0).isoformat(sep=" ")


def classify_extension(ext: str) -> FileKind:
    """Map a lower-case extension (with dot) to a :class:`FileKind`."""
    ext = ext.lower()
    if ext in VIDEO_EXTENSIONS:
        return FileKind.VIDEO
    if ext in AUDIO_EXTENSIONS:
        return FileKind.AUDIO
    if ext in IMAGE_EXTENSIONS:
        return FileKind.IMAGE
    if ext in DOCUMENT_EXTENSIONS:
        return FileKind.DOCUMENT
    return FileKind.OTHER


def split_name(name: str) -> tuple[str, str]:
    """Split ``name`` into ``(stem, extension)``; dot-files keep their whole name as stem."""
    stem, ext = os.path.splitext(name)
    return stem, ext


def truncate(text: str, limit: int) -> str:
    """Shorten ``text`` to ``limit`` characters, adding an ellipsis when cut."""
    return text if len(text) <= limit else text[: max(limit - 1, 0)] + "…"


# ------------------------------------------------------------------- errors
_WINERROR_MESSAGES = {
    2: "The file was not found.",
    3: "The folder was not found.",
    5: "Permission denied.",
    32: "The file is in use by another program (locked).",
    80: "A file with that name already exists.",
    123: "The file name or path is not valid on Windows.",
    183: "A file with that name already exists.",
    206: "The file path is too long for Windows.",
}


def describe_os_error(exc: BaseException) -> str:
    """Translate an :class:`OSError` into a short, user-friendly sentence."""
    winerror = getattr(exc, "winerror", None)
    if winerror in _WINERROR_MESSAGES:
        return _WINERROR_MESSAGES[winerror]
    if isinstance(exc, FileNotFoundError):
        return "The file or folder was not found (it may have been moved or deleted)."
    if isinstance(exc, FileExistsError):
        return "A file with that name already exists."
    if isinstance(exc, PermissionError):
        return "Permission denied, or the file is in use by another program."
    if isinstance(exc, IsADirectoryError):
        return "Expected a file but found a folder."
    if isinstance(exc, NotADirectoryError):
        return "Expected a folder but found a file."
    if isinstance(exc, OSError):
        code = exc.errno
        if code == errno.ENOSPC:
            return "The disk is full."
        if code == errno.ENAMETOOLONG:
            return "The file name or path is too long."
        if code == errno.EROFS:
            return "The location is read-only."
        if code == errno.EXDEV:
            return "Cannot move between different drives in a single step."
        return exc.strerror or str(exc)
    return str(exc)
