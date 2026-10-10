"""Low-level, safe file-system primitives (move, copy, trash, folders).

Nothing here decides *what* to do - the planner has already validated that. These
helpers only refuse to clobber anything, translate OS errors into friendly
messages, and report every step so a failed batch can be rolled back exactly.
"""

from __future__ import annotations

import errno
import os
import shutil
from dataclasses import dataclass, field
from pathlib import Path

from app.config.constants import TRASH_DIR_NAME
from app.utils.helpers import describe_os_error
from app.utils.logger import get_logger

log = get_logger("fs")


class OperationCancelled(Exception):
    """The user cancelled while a batch was running (the caller rolls the batch back)."""


class FsError(Exception):
    """A file-system step failed. ``path`` is the file concerned; ``message`` is user-friendly."""

    def __init__(self, path: Path | str, message: str) -> None:
        self.path = str(path)
        self.message = message
        super().__init__(f"{Path(path).name}: {message}")


@dataclass(slots=True)
class StepLog:
    """Everything done so far, in order, so it can be undone in reverse."""

    moves: list[tuple[Path, Path]] = field(default_factory=list)  # (src, dst) actually performed
    copies: list[Path] = field(default_factory=list)  # files created by copy
    folders: list[Path] = field(default_factory=list)  # directories created (parents first)

    def __len__(self) -> int:
        return len(self.moves) + len(self.copies) + len(self.folders)


def _lexists(path: Path) -> bool:
    return os.path.lexists(path)


def safe_move(src: Path, dst: Path, steps: StepLog | None = None, *, create_parents: bool = False) -> None:
    """Move ``src`` to ``dst``; refuses to overwrite and records the step."""
    if not _lexists(src):
        raise FsError(src, "The file no longer exists (it may have been moved or deleted).")
    if _lexists(dst) and os.path.normcase(os.path.abspath(src)) != os.path.normcase(os.path.abspath(dst)):
        raise FsError(dst, "A file with that name already exists.")
    if create_parents:
        make_folder(dst.parent, steps)
    try:
        os.rename(src, dst)
    except OSError as exc:
        if exc.errno == errno.EXDEV:  # across volumes: fall back to copy + delete
            try:
                shutil.move(str(src), str(dst))
            except OSError as inner:
                raise FsError(src, describe_os_error(inner)) from inner
        else:
            raise FsError(src, describe_os_error(exc)) from exc
    if steps is not None:
        steps.moves.append((src, dst))


def safe_copy(src: Path, dst: Path, steps: StepLog | None = None) -> None:
    """Copy a file (metadata included); never overwrites; a partial copy is removed on failure."""
    if not _lexists(src):
        raise FsError(src, "The file no longer exists (it may have been moved or deleted).")
    if _lexists(dst):
        raise FsError(dst, "A file with that name already exists.")
    try:
        shutil.copy2(src, dst)
    except OSError as exc:
        try:
            if _lexists(dst):
                os.remove(dst)
        except OSError:
            log.warning("could not remove partial copy %s", dst.name)
        raise FsError(src, describe_os_error(exc)) from exc
    if steps is not None:
        steps.copies.append(dst)


def make_folder(path: Path, steps: StepLog | None = None) -> None:
    """Create ``path`` (and missing parents), remembering which directories are new."""
    missing: list[Path] = []
    probe = path
    while not probe.exists() and probe != probe.parent:
        missing.append(probe)
        probe = probe.parent
    if _lexists(path) and not path.is_dir():
        raise FsError(path, "A file with that name already exists.")
    try:
        path.mkdir(parents=True, exist_ok=True)
    except OSError as exc:
        raise FsError(path, describe_os_error(exc)) from exc
    if steps is not None:
        steps.folders.extend(reversed(missing))


def remove_empty_dir(path: Path) -> bool:
    """Remove ``path`` only if it is an empty directory. Returns whether it was removed."""
    try:
        path.rmdir()
    except OSError:
        return False
    return True


def delete_file(path: Path) -> None:
    """Permanently delete a file (the user was warned this cannot be undone)."""
    try:
        os.remove(path)
    except OSError as exc:
        raise FsError(path, describe_os_error(exc)) from exc


def trash_relative(token: str, rel: str) -> str:
    """Workspace-relative location inside the undo-trash for ``rel``."""
    return f"{TRASH_DIR_NAME}/{token}/{rel}"


def rollback(steps: StepLog) -> list[str]:
    """Undo ``steps`` in reverse order. Returns descriptions of anything that could not be undone."""
    failures: list[str] = []
    for src, dst in reversed(steps.moves):
        try:
            if _lexists(dst) and not _lexists(src):
                make_folder(src.parent)
                os.rename(dst, src)
            elif not _lexists(dst) and _lexists(src):
                continue  # already back
            else:
                failures.append(f"{src.name}: could not be restored")
        except (OSError, FsError) as exc:
            failures.append(f"{src.name}: {describe_os_error(exc) if isinstance(exc, OSError) else exc}")
    for copy in reversed(steps.copies):
        try:
            if _lexists(copy):
                os.remove(copy)
        except OSError as exc:
            failures.append(f"{copy.name}: {describe_os_error(exc)}")
    for folder in reversed(steps.folders):
        remove_empty_dir(folder)
    return failures
