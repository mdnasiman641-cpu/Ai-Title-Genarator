"""Workspace path safety.

Every path that comes from the AI is untrusted. :class:`WorkspacePaths`
converts it into a clean *relative* path (``/``-separated) or raises
:class:`PathError`. It rejects absolute paths, drive letters, UNC paths,
``..`` traversal, illegal Windows characters and reserved names, and it verifies
(including through symlinks / junctions) that the result stays inside the
selected workspace.
"""

from __future__ import annotations

import os
import re
import unicodedata
from pathlib import Path

from app.config.constants import CASE_INSENSITIVE_FS, INTERNAL_DIR_PREFIX, TEMP_PREFIX
from app.utils.validators import validate_component

MAX_DEPTH = 32
_DRIVE = re.compile(r"^[A-Za-z]:")


class PathError(ValueError):
    """A path is unsafe or invalid; the message is shown in the preview."""


class WorkspacePaths:
    """Path rules for one workspace folder."""

    def __init__(self, root: Path | str, *, case_insensitive: bool | None = None) -> None:
        self.root = Path(os.path.abspath(root))
        self.real_root = Path(os.path.realpath(root))
        self.case_insensitive = CASE_INSENSITIVE_FS if case_insensitive is None else case_insensitive

    # ------------------------------------------------------------- comparison
    def key(self, rel: str) -> str:
        """Comparison key: NFC-normalised and (on Windows) case-folded."""
        text = unicodedata.normalize("NFC", rel)
        return text.casefold() if self.case_insensitive else text

    # --------------------------------------------------------- AI path -> rel
    def normalize_relative(self, raw: str, *, what: str = "path", allow_empty: bool = False) -> str:
        """Validate an AI-supplied relative path and return it normalised (``a/b/c``)."""
        if not isinstance(raw, str):
            raise PathError(f"The {what} must be text.")
        if "\x00" in raw:
            raise PathError(f"The {what} contains an invalid character.")
        text = raw.strip().replace("\\", "/")
        if text.startswith("//") or raw.strip().startswith("\\\\"):
            raise PathError(f"Network (UNC) paths are not allowed: {raw!r}")
        if text.startswith("/") or _DRIVE.match(text):
            raise PathError(f"Absolute paths are not allowed: {raw!r}")
        parts = [p for p in text.split("/") if p not in ("", ".")]
        if not parts:
            if allow_empty:
                return ""
            raise PathError(f"The {what} is empty.")
        if len(parts) > MAX_DEPTH:
            raise PathError(f"The {what} is nested too deeply.")
        for part in parts:
            if part == "..":
                raise PathError(f"Path traversal (“..”) is not allowed: {raw!r}")
            problems = validate_component(part)
            if problems:
                raise PathError(f"“{part}”: {problems[0]}")
            if part.startswith(INTERNAL_DIR_PREFIX) or part.startswith(TEMP_PREFIX):
                raise PathError(f"“{part}” is reserved for internal use.")
        return "/".join(parts)

    # ------------------------------------------------------------- resolution
    def absolute(self, rel: str) -> Path:
        """Absolute path of ``rel`` inside the workspace (no file-system access)."""
        return self.root.joinpath(*[p for p in rel.split("/") if p])

    def _within(self, path: Path, base: Path) -> bool:
        norm = os.path.normcase
        try:
            return os.path.commonpath([norm(str(path)), norm(str(base))]) == norm(str(base))
        except ValueError:  # different drives
            return False

    def ensure_contained(self, rel: str) -> Path:
        """Return the absolute path of ``rel``, verifying it cannot escape the workspace.

        Besides the lexical check, the deepest *existing* ancestor is resolved with
        ``realpath`` so a symlink or junction pointing outside the workspace is caught.
        """
        absolute = self.absolute(rel)
        if not self._within(Path(os.path.abspath(absolute)), self.root):
            raise PathError(f"The path is outside the selected folder: {rel!r}")
        probe = absolute
        while probe != self.root and not os.path.lexists(probe):
            probe = probe.parent
        real = Path(os.path.realpath(probe))
        if not self._within(real, self.real_root):
            raise PathError(f"The path leaves the selected folder (link/junction): {rel!r}")
        return absolute

    @staticmethod
    def split(rel: str) -> tuple[str, str]:
        """``"a/b/c.mp4"`` -> ``("a/b", "c.mp4")``."""
        head, _, tail = rel.rpartition("/")
        return head, tail

    @staticmethod
    def join(directory: str, name: str) -> str:
        return f"{directory}/{name}" if directory else name
