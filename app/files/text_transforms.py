"""Pure string transformations used by the rename engine (no file-system access).

All functions operate on a file's *stem* (the name without its extension), so
extensions are preserved automatically. They are Unicode-safe: Bangla and other
scripts pass through untouched.
"""

from __future__ import annotations

import re

_EMPTY_BRACKETS = re.compile(r"\(\s*\)|\[\s*\]|\{\s*\}")
_REPEATED_SEPARATOR = re.compile(r"([ ._\-])\1+")
_WHITESPACE = re.compile(r"\s+")
_WORD = re.compile(r"[^\W_]+(?:['’][^\W_]+)*", re.UNICODE)
_EDGE_SEPARATORS = " ._-"
_SEPARATOR_AFTER = re.compile(r"\s[-–—:|]$")
_SEPARATOR_BEFORE = re.compile(r"^[-–—|]\s")


def remove_texts(stem: str, texts: tuple[str, ...] | list[str], *, case_sensitive: bool = False) -> str:
    """Delete every occurrence of each text from ``stem`` and tidy the leftovers."""
    result = stem
    for text in texts:
        if not text:
            continue
        pattern = re.compile(re.escape(text), 0 if case_sensitive else re.IGNORECASE)
        result = pattern.sub("", result)
    return tidy(result) if result != stem else stem


def replace_text(stem: str, find: str, replacement: str, *, case_sensitive: bool = False) -> str:
    """Replace ``find`` with ``replacement`` (literal match; no regex)."""
    if not find:
        return stem
    pattern = re.compile(re.escape(find), 0 if case_sensitive else re.IGNORECASE)
    result = pattern.sub(lambda _m: replacement, stem)
    return tidy(result) if result != stem else stem


def tidy(stem: str) -> str:
    """Clean up after text removal: empty brackets, doubled separators, edge separators."""
    previous = None
    result = stem
    while previous != result:  # nested empty brackets collapse over several passes
        previous = result
        result = _EMPTY_BRACKETS.sub("", result)
    result = _REPEATED_SEPARATOR.sub(r"\1", result)
    result = _WHITESPACE.sub(" ", result)
    return result.strip(_EDGE_SEPARATORS)


def change_case(stem: str, mode: str) -> str:
    """``lower`` / ``upper`` / ``title`` / ``sentence`` capitalisation."""
    if mode == "lower":
        return stem.lower()
    if mode == "upper":
        return stem.upper()
    if mode == "title":
        return _WORD.sub(lambda m: m.group(0)[:1].upper() + m.group(0)[1:].lower(), stem)
    if mode == "sentence":
        lowered = stem.lower()
        for index, char in enumerate(lowered):
            if char.isalpha():
                return lowered[:index] + char.upper() + lowered[index + 1 :]
        return lowered
    return stem


def default_prefix_separator(text: str) -> str:
    """Separator inserted between a prefix and the name when the user did not specify one."""
    if not text or text.endswith((" ", "\t")):
        return ""
    if _SEPARATOR_AFTER.search(text):
        return " "  # "Show -" + "Name" -> "Show - Name"
    if text[-1] in "_-.([{":
        return ""
    return " "


def default_suffix_separator(text: str) -> str:
    """Separator inserted between the name and a suffix when none was specified."""
    if not text or text.startswith((" ", "\t")):
        return ""
    if _SEPARATOR_BEFORE.match(text):
        return " "  # "- HD" -> "Name - HD"
    if text[0] in "_-.":
        return ""
    return " "


def add_prefix(stem: str, text: str, separator: str | None = None) -> str:
    sep = default_prefix_separator(text) if separator is None else separator
    return f"{text}{sep}{stem}"


def add_suffix(stem: str, text: str, separator: str | None = None) -> str:
    sep = default_suffix_separator(text) if separator is None else separator
    return f"{stem}{sep}{text}"


def format_number(number: int, width: int) -> str:
    """Zero-padded number (``7, 2`` -> ``07``)."""
    return str(number).zfill(width)


def apply_template(template: str, number_text: str) -> str:
    """Fill ``{n}`` in ``template``. Plain replacement - never ``str.format`` - so templates can't run code."""
    return template.replace("{n}", number_text)


def normalise_stem(stem: str) -> str:
    """Final light clean-up for rule output: trim surrounding whitespace only.

    Internal spacing is left exactly as the user (or the file) had it; removal/replacement
    rules tidy up after themselves via :func:`tidy`.
    """
    return stem.strip()
