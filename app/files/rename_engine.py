"""Rename engine: applies rule actions to files and relocates files safely.

Two responsibilities:

1. :class:`RuleEngine` - applies deterministic rules (remove/replace text, prefix,
   suffix, numbering, case, sort, move/copy/delete by pattern) to the selected
   files *locally*. The AI describes the rule once; this class does the per-file
   work, which keeps API usage tiny even for thousands of files.
2. :func:`relocate_batch` - performs a batch of renames/moves with a **two-phase**
   temp-name strategy whenever targets collide with sources in the same batch
   (``A→B`` + ``B→C``, swaps, case-only renames), so nothing is overwritten.
"""

from __future__ import annotations

import os
import threading
import uuid
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path

from app.ai import schemas as s
from app.config.constants import (
    AUDIO_EXTENSIONS,
    DOCUMENT_EXTENSIONS,
    IMAGE_EXTENSIONS,
    IS_WINDOWS,
    TEMP_PREFIX,
    VIDEO_EXTENSIONS,
)
from app.files import text_transforms as tt
from app.files.move_engine import OperationCancelled, StepLog, safe_move
from app.files.path_validator import PathError, WorkspacePaths
from app.files.scanner import FileEntry
from app.utils.helpers import split_name
from app.utils.validators import fit_stem, sanitize_component, validate_component

KNOWN_EXTENSIONS = VIDEO_EXTENSIONS | AUDIO_EXTENSIONS | IMAGE_EXTENSIONS | DOCUMENT_EXTENSIONS


# ===================================================================== names
@dataclass(slots=True)
class NameResult:
    """A finished file name or the reasons it is unusable."""

    name: str = ""
    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)


def finalize_name(stem: str, ext: str, *, directory_abs: str = "", enforce_path_limit: bool | None = None) -> NameResult:
    """Turn a transformed stem + the original extension into a validated file name."""
    result = NameResult()
    stem = tt.normalise_stem(stem)
    if not stem or not stem.strip(" ."):
        result.errors.append("The new name would be empty.")
        return result
    enforce = IS_WINDOWS if enforce_path_limit is None else enforce_path_limit
    fit = fit_stem(stem, ext, directory_len=len(directory_abs) + 1, enforce_path_limit=enforce)
    if not fit.possible:
        result.errors.append("The path would be too long for Windows.")
        return result
    if fit.truncated:
        result.warnings.append("Name was shortened to fit Windows length limits.")
    name = fit.stem + ext
    problems = validate_component(name)
    if problems:
        result.errors.append(f"{problems[0]} (suggestion: {sanitize_component(name)})")
        return result
    result.name = name
    return result


def resolve_explicit_target(source_name: str, target: str, change_extension: bool) -> NameResult:
    """Validate an AI-supplied ``rename`` target, preserving the extension unless told otherwise."""
    result = NameResult()
    _stem, src_ext = split_name(source_name)
    target = target.strip()
    if not target:
        result.errors.append("The new name is empty.")
        return result
    if "/" in target or "\\" in target:
        result.errors.append("A rename target must be a file name, not a path (use a move to change folders).")
        return result
    if change_extension or not src_ext:
        result.name = target
    elif target.lower().endswith(src_ext.lower()):
        result.name = target
    else:
        tgt_ext = os.path.splitext(target)[1].lower()
        if tgt_ext in KNOWN_EXTENSIONS and tgt_ext != src_ext.lower():
            result.errors.append(
                f"Extension change {src_ext} → {tgt_ext} was not requested. Extensions are kept unless you explicitly ask."
            )
            return result
        result.name = target + src_ext
        result.warnings.append(f"Extension “{src_ext}” was kept.")
    if change_extension and os.path.splitext(result.name)[1].lower() != src_ext.lower():
        result.warnings.append(f"Extension changes from {src_ext or '(none)'} to {os.path.splitext(result.name)[1] or '(none)'}.")
    problems = validate_component(result.name)
    if problems:
        result.errors.append(f"{problems[0]} (suggestion: {sanitize_component(result.name)})")
    return result


# ============================================================ rule application
@dataclass(slots=True)
class FileState:
    """Working state of one selected file while rules are applied."""

    entry: FileEntry
    stem: str
    folder: str  # workspace-relative destination folder ('' = root)
    delete: bool = False
    copy_folders: list[str] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)  # e.g. invalid move target
    origins: list[str] = field(default_factory=list)  # action types that touched it

    @property
    def renamed(self) -> bool:
        return self.stem != self.entry.stem

    @property
    def moved(self) -> bool:
        return self.folder != self.entry.directory


@dataclass(slots=True)
class RuleOutcome:
    """Result of applying rule actions."""

    states: list[FileState]
    sort: s.SortAction | None = None
    notes: list[str] = field(default_factory=list)


def _sort_key(entry: FileEntry, by: str) -> tuple[object, ...]:
    if by == "size":
        return (entry.size, entry.natural_key)
    if by == "date":
        return (entry.mtime, entry.natural_key)
    if by == "type":
        return (entry.ext, entry.natural_key)
    if by == "duration":
        return (entry.duration or -1.0, entry.natural_key)
    return (entry.natural_key,)


def sort_entries(entries: list[FileEntry], by: str, descending: bool) -> list[FileEntry]:
    """Return ``entries`` sorted by ``by`` (``name``/``date``/``size``/``type``/``duration``)."""
    return sorted(entries, key=lambda e: _sort_key(e, by), reverse=descending)


class RuleEngine:
    """Applies rule actions, in order, to the selected files."""

    def __init__(self, files: list[FileEntry], paths: WorkspacePaths) -> None:
        self._paths = paths
        self.states = [FileState(entry=f, stem=f.stem, folder=f.directory) for f in files]
        self._sort: s.SortAction | None = None
        self._notes: list[str] = []

    # ---------------------------------------------------------------- driver
    def apply(self, actions: list[s.Action]) -> RuleOutcome:
        for action in actions:
            self._apply_one(action)
        for state in self.states:
            if state.renamed:
                state.stem = tt.normalise_stem(state.stem)
        return RuleOutcome(self.states, self._sort, self._notes)

    def _apply_one(self, action: s.Action) -> None:
        if isinstance(action, s.SortAction):
            self._sort = action
            order = sort_entries([st.entry for st in self.states], action.by, action.descending)
            by_path = {st.entry.rel_path: st for st in self.states}
            self.states = [by_path[e.rel_path] for e in order]
        elif isinstance(action, s.AddPrefixAction):
            self._each(action, action.match, lambda st: tt.add_prefix(st.stem, action.text, action.separator))
        elif isinstance(action, s.AddSuffixAction):
            self._each(action, action.match, lambda st: tt.add_suffix(st.stem, action.text, action.separator))
        elif isinstance(action, s.RemoveTextAction):
            self._each(
                action, action.match, lambda st: tt.remove_texts(st.stem, action.texts, case_sensitive=action.case_sensitive)
            )
        elif isinstance(action, s.ReplaceTextAction):
            self._each(
                action, action.match,
                lambda st: tt.replace_text(st.stem, action.find, action.replace, case_sensitive=action.case_sensitive),
            )  # fmt: skip
        elif isinstance(action, s.ChangeCaseAction):
            self._each(action, action.match, lambda st: tt.change_case(st.stem, action.mode))
        elif isinstance(action, s.NumberingAction):
            self._number(action)
        elif isinstance(action, s.MoveAction) and action.is_rule:
            self._move_rule(action)
        elif isinstance(action, s.CopyAction) and action.is_rule:
            self._copy_rule(action)
        elif isinstance(action, s.DeleteAction) and action.is_rule:
            for st in self._matching(action.match):
                st.delete = True
                st.origins.append(action.type)

    # --------------------------------------------------------------- helpers
    def _matching(self, match: tuple[str, ...]) -> list[FileState]:
        if not match:
            return list(self.states)
        needles = [m.casefold() for m in match]
        return [st for st in self.states if any(n in st.entry.name.casefold() for n in needles)]

    def _each(self, action: s.Action, match: tuple[str, ...], transform: Callable[[FileState], str]) -> None:
        for st in self._matching(match):
            new = transform(st)
            if new != st.stem:
                st.stem = new
                st.origins.append(action.type)

    def _number(self, action: s.NumberingAction) -> None:
        targets = self._matching(action.match)
        if not targets:
            return
        width = max(action.width, len(str(action.start + len(targets) - 1)))
        for offset, st in enumerate(targets):
            text = tt.format_number(action.start + offset, width)
            if action.template:
                text = tt.apply_template(action.template, text)
            if action.position == "replace":
                st.stem = text
            elif action.position == "prefix":
                st.stem = f"{text}{action.separator}{st.stem}"
            else:
                st.stem = f"{st.stem}{action.separator}{text}"
            st.origins.append(action.type)

    def _folder_for(self, state: FileState, raw: str) -> str | None:
        try:
            return self._paths.normalize_relative(raw, what="folder")
        except PathError as exc:
            state.errors.append(str(exc))
            return None

    def _move_rule(self, action: s.MoveAction) -> None:
        for st in self._matching(action.match):
            folder = self._folder_for(st, action.target_folder)
            if folder is not None:
                st.folder = folder
                st.origins.append(action.type)
        if action.new_name:
            self._notes.append("“new_name” is ignored for rule-based moves.")

    def _copy_rule(self, action: s.CopyAction) -> None:
        for st in self._matching(action.match):
            folder = self._folder_for(st, action.target_folder)
            if folder is not None:
                st.copy_folders.append(folder)
                st.origins.append(action.type)
        if action.new_name:
            self._notes.append("“new_name” is ignored for rule-based copies.")


# ================================================================ relocation
ProgressFn = Callable[[int, int], None]


def _key(path: Path) -> str:
    return os.path.normcase(os.path.abspath(path))


def needs_two_phase(pairs: list[tuple[Path, Path]]) -> bool:
    """True if any destination is also a source in the same batch (chains, swaps, case-only renames)."""
    sources = {_key(src) for src, _ in pairs}
    return any(_key(dst) in sources for _, dst in pairs)


def relocate_batch(
    pairs: list[tuple[Path, Path]],
    steps: StepLog,
    *,
    token: str | None = None,
    cancel: threading.Event | None = None,
    progress: ProgressFn | None = None,
    temp_names: list[str] | None = None,
) -> bool:
    """Move/rename every ``(src, dst)`` pair; returns whether the two-phase strategy was used.

    Direct mode renames each file in turn. Two-phase mode first renames every source
    to a unique ``.tmp_ai_<token>_NNN`` name in its own folder, then renames each
    temp file to its final name, so ``A→B`` and ``B→C`` can never overwrite each
    other. Every primitive step is appended to ``steps`` so the caller can roll back.
    ``temp_names`` (optional) receives the temp file names for the journal.
    """
    if not pairs:
        return False
    cancel = cancel or threading.Event()
    two_phase = needs_two_phase(pairs)
    total = len(pairs) * (2 if two_phase else 1)
    done = 0

    def tick() -> None:
        nonlocal done
        done += 1
        if progress is not None:
            progress(done, total)
        if cancel.is_set():
            raise OperationCancelled

    if not two_phase:
        for src, dst in pairs:
            safe_move(src, dst, steps, create_parents=True)
            tick()
        return False

    token = token or uuid.uuid4().hex[:8]
    temps: list[Path] = []
    for index, (src, _dst) in enumerate(pairs, start=1):
        temp = src.with_name(f"{TEMP_PREFIX}{token}_{index:03d}")
        suffix = 0
        while os.path.lexists(temp):  # astronomically unlikely, but never overwrite anything
            suffix += 1
            temp = src.with_name(f"{TEMP_PREFIX}{token}_{index:03d}_{suffix}")
        safe_move(src, temp, steps)
        temps.append(temp)
        if temp_names is not None:
            temp_names.append(temp.name)
        tick()
    for temp, (_src, dst) in zip(temps, pairs, strict=True):
        safe_move(temp, dst, steps, create_parents=True)
        tick()
    return True
