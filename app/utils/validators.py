"""Windows file-name validation and repair helpers.

The rules follow the NTFS/Win32 naming conventions: forbidden characters,
reserved device names, trailing dots/spaces and the 255 character limit.
Bangla and other Unicode names are fully supported - only the characters
Windows itself forbids are rejected.
"""

from __future__ import annotations

import re
import sys
from dataclasses import dataclass

from app.config.constants import (
    ILLEGAL_FILENAME_CHARS,
    MAX_NAME_BYTES_POSIX,
    MAX_NAME_UTF16,
    MAX_WIN_PATH,
    WINDOWS_RESERVED_NAMES,
)

_CONTROL_CHARS = {chr(i) for i in range(32)}
_ILLEGAL_SET = set(ILLEGAL_FILENAME_CHARS) | _CONTROL_CHARS
_MULTI_SPACE = re.compile(r"\s{2,}")
# Zero-width / bidirectional controls can be used to disguise names.
_INVISIBLE = re.compile("[\u200b\u200e\u200f\u202a-\u202e\u2066-\u2069\ufeff]")


def utf16_length(text: str) -> int:
    """Length of ``text`` in UTF-16 code units (how NTFS measures names)."""
    return len(text.encode("utf-16-le")) // 2


def find_illegal_chars(name: str) -> list[str]:
    """Return the distinct characters in ``name`` that Windows forbids."""
    seen: list[str] = []
    for char in name:
        if char in _ILLEGAL_SET and char not in seen:
            seen.append(char)
    return seen


def _printable(chars: list[str]) -> str:
    return " ".join(repr(c)[1:-1] if c in _CONTROL_CHARS else c for c in chars)


def is_reserved_name(name: str) -> bool:
    """True for ``CON``, ``NUL``, ``COM1`` ... with or without an extension."""
    base = name.split(".", 1)[0].rstrip(" ").upper()
    return base in WINDOWS_RESERVED_NAMES


def validate_component(name: str, *, max_length: int | None = None) -> list[str]:
    """Return a list of problems with a single path component (empty list = valid)."""
    problems: list[str] = []
    if not name or not name.strip():
        return ["Name is empty."]
    if name in {".", ".."}:
        return ["Name cannot be '.' or '..'."]
    bad = find_illegal_chars(name)
    if bad:
        problems.append(f"Illegal character(s): {_printable(bad)}")
    if _INVISIBLE.search(name):
        problems.append("Name contains invisible control characters.")
    if name != name.rstrip(" ."):
        problems.append("Name cannot end with a space or a dot.")
    if name != name.lstrip(" "):
        problems.append("Name cannot start with a space.")
    if is_reserved_name(name):
        problems.append("Reserved Windows name.")
    limit = max_length or MAX_NAME_UTF16
    if utf16_length(name) > limit:
        problems.append(f"Name is longer than {limit} characters.")
    if sys.platform != "win32" and len(name.encode("utf-8")) > MAX_NAME_BYTES_POSIX:
        problems.append("Name is too long for this file system.")
    return problems


def sanitize_component(name: str, replacement: str = "_") -> str:
    """Produce a legal version of ``name`` (used only for *suggestions* shown to users)."""
    cleaned = "".join(replacement if c in _ILLEGAL_SET else c for c in name)
    cleaned = _INVISIBLE.sub("", cleaned)
    cleaned = _MULTI_SPACE.sub(" ", cleaned).strip(" .")
    if is_reserved_name(cleaned):
        cleaned = f"_{cleaned}"
    return cleaned or "_"


@dataclass(frozen=True, slots=True)
class FitResult:
    """Outcome of :func:`fit_stem` - the (possibly shortened) stem and whether it was cut."""

    stem: str
    truncated: bool
    possible: bool  # False when even a one-character stem would not fit


def fit_stem(stem: str, ext: str, *, directory_len: int = 0, enforce_path_limit: bool = False) -> FitResult:
    """Shorten ``stem`` so ``stem + ext`` obeys the component (and optionally path) limit.

    ``directory_len`` is the length of the containing directory path including
    the trailing separator; it is only used when ``enforce_path_limit`` is set
    (Windows MAX_PATH handling).
    """
    budget = MAX_NAME_UTF16 - utf16_length(ext)
    if enforce_path_limit:
        budget = min(budget, MAX_WIN_PATH - directory_len - utf16_length(ext))
    if budget < 1:
        return FitResult(stem, False, False)
    if utf16_length(stem) <= budget and (
        sys.platform == "win32" or len((stem + ext).encode("utf-8")) <= MAX_NAME_BYTES_POSIX
    ):
        return FitResult(stem, False, True)
    shortened = stem
    while shortened and (
        utf16_length(shortened) > budget
        or (sys.platform != "win32" and len((shortened + ext).encode("utf-8")) > MAX_NAME_BYTES_POSIX)
    ):
        shortened = shortened[:-1]
    shortened = shortened.rstrip(" .")
    return FitResult(shortened, True, bool(shortened))
