"""Batch validation of a plan: conflicts, overwrites, missing folders.

Validation looks at the *final state* of the whole batch rather than at each
operation in isolation, which is what makes chains (``A→B``, ``B→C``), swaps and
case-only renames legal while still catching real collisions.
"""

from __future__ import annotations

import os
from pathlib import Path

from app.files.path_validator import WorkspacePaths
from app.files.plan import OpKind, OpStatus, Plan, PlannedOp
from app.utils.helpers import split_name
from app.utils.logger import get_logger

log = get_logger("plan")


class DirIndex:
    """Cached directory listings (one ``scandir`` per folder) for fast existence checks."""

    def __init__(self, paths: WorkspacePaths) -> None:
        self._paths = paths
        self._listings: dict[str, dict[str, str] | None] = {}

    def _names(self, directory: str) -> dict[str, str] | None:
        if directory not in self._listings:
            absolute = self._paths.absolute(directory)
            try:
                with os.scandir(absolute) as it:
                    self._listings[directory] = {self._paths.key(e.name): e.name for e in it}
            except OSError:
                self._listings[directory] = None
        return self._listings[directory]

    def dir_exists(self, directory: str) -> bool:
        """True if ``directory`` (``''`` = workspace root) exists as a folder."""
        return directory == "" or self._names(directory) is not None

    def exists(self, rel: str) -> bool:
        head, tail = WorkspacePaths.split(rel)
        names = self._names(head)
        return names is not None and self._paths.key(tail) in names

    def is_dir(self, rel: str) -> bool:
        return self.exists(rel) and os.path.isdir(self._paths.absolute(rel))

    def is_file(self, rel: str) -> bool:
        return self.exists(rel) and os.path.isfile(self._paths.absolute(rel))


class BatchValidator:
    """Recomputes status/messages of every operation in a :class:`Plan`."""

    def __init__(
        self,
        paths: WorkspacePaths,
        *,
        prevent_overwrite: bool = True,
        duplicate_policy: str = "flag",
        delete_mode: str = "trash",
    ) -> None:
        self.paths = paths
        self.prevent_overwrite = prevent_overwrite
        self.duplicate_policy = duplicate_policy
        self.delete_mode = delete_mode

    # ----------------------------------------------------------------- entry
    def validate(self, plan: Plan, *, fresh: bool = False) -> None:
        """Update ``op.status``/``messages``/``overwrite``/``noop`` for the whole plan.

        ``fresh`` is accepted for symmetry with the executor's pre-flight; a new
        :class:`DirIndex` is always built, so results reflect the disk *now*.
        """
        index = DirIndex(self.paths)
        key = self.paths.key
        active = [op for op in plan.ops if op.included]
        for op in plan.ops:
            if op.draft_target is not None:
                op.target = op.draft_target  # undo any duplicate-numbering from a previous run
            op.messages = list(op.static_messages)
            op.overwrite = False
            op.noop = False
            if op.intent_error:
                op.status = OpStatus.INVALID
                op.messages.insert(0, op.intent_error)
            else:
                op.status = OpStatus.WARNING if op.static_messages else OpStatus.OK
        live = [op for op in active if op.intent_error is None]

        vacated = {
            key(op.source)
            for op in live
            if op.source and (op.kind in (OpKind.RENAME, OpKind.MOVE) or (op.kind == OpKind.DELETE and self.delete_mode == "trash"))
        }
        created = [key(op.target) for op in live if op.kind == OpKind.CREATE_FOLDER and op.target]
        needed_dirs = {
            key(WorkspacePaths.split(op.target)[0])
            for op in live
            if op.kind in (OpKind.RENAME, OpKind.MOVE, OpKind.COPY) and op.target and WorkspacePaths.split(op.target)[0]
        }
        taken_targets: dict[str, PlannedOp] = {}

        # 1. sources must (still) exist
        for op in live:
            if op.source and op.kind != OpKind.CREATE_FOLDER and not index.is_file(op.source):
                self._mark(op, OpStatus.INVALID, "The file no longer exists (it may have been moved or deleted).")
        live = [op for op in live if op.status != OpStatus.INVALID]

        # 2. folder creation
        for op in live:
            if op.kind == OpKind.CREATE_FOLDER and op.target:
                self._check_folder(op, index, needed_dirs, key)

        # 3. deletes
        for op in live:
            if op.kind == OpKind.DELETE:
                if self.delete_mode == "trash":
                    self._mark(op, OpStatus.WARNING, "Moved to the undo-trash; Undo can restore it.")
                else:
                    self._mark(op, OpStatus.WARNING, "Permanently deleted - this cannot be undone.")

        # 4. relocations and copies: targets, collisions, parent folders
        for op in live:
            if op.kind not in (OpKind.RENAME, OpKind.MOVE, OpKind.COPY) or not op.target or op.status == OpStatus.INVALID:
                continue
            self._check_target(op, index, vacated, taken_targets, created, key)

    # ---------------------------------------------------------------- pieces
    @staticmethod
    def _mark(op: PlannedOp, status: OpStatus, message: str) -> None:
        order = {OpStatus.OK: 0, OpStatus.WARNING: 1, OpStatus.CONFLICT: 2, OpStatus.INVALID: 3}
        if order[status] >= order[op.status]:
            op.status = status
        op.messages.append(message)

    def _check_folder(self, op: PlannedOp, index: DirIndex, needed_dirs: set[str], key) -> None:  # type: ignore[no-untyped-def]
        assert op.target is not None
        if index.is_dir(op.target):
            op.noop = True
            op.messages.append("Folder already exists.")
        elif index.exists(op.target):
            self._mark(op, OpStatus.CONFLICT, "A file with this name already exists, so the folder cannot be created.")
        elif op.implicit and not any(d == key(op.target) or d.startswith(key(op.target) + "/") for d in needed_dirs):
            op.noop = True  # nothing included needs this automatically-created folder
        else:
            op.messages.append("Folder will be created.")  # informational only; status stays OK

    def _check_target(self, op, index, vacated, taken, created, key) -> None:  # type: ignore[no-untyped-def]
        assert op.target is not None
        tkey = key(op.target)
        own_key = key(op.source) if op.source else None
        case_only = op.kind in (OpKind.RENAME, OpKind.MOVE) and own_key == tkey

        parent = WorkspacePaths.split(op.target)[0]
        if parent and not index.dir_exists(parent):
            pkey = key(parent)
            if not any(c == pkey or c.startswith(pkey + "/") for c in created):
                self._mark(op, OpStatus.INVALID, f"The folder “{parent}” does not exist and will not be created.")
                return

        existing_blocks = index.exists(op.target) and tkey not in vacated and not case_only
        duplicate = taken.get(tkey)
        if duplicate is not None or existing_blocks:
            if existing_blocks and duplicate is None and index.is_dir(op.target):
                self._mark(op, OpStatus.CONFLICT, f"“{WorkspacePaths.split(op.target)[1]}” already exists as a folder.")
                return
            if duplicate is None and not self.prevent_overwrite and op.kind != OpKind.COPY:
                op.overwrite = True
                self._mark(op, OpStatus.WARNING, "An existing file with this name will be replaced (it moves to the undo-trash first).")
                taken[tkey] = op
                return
            if self.duplicate_policy == "number":
                self._renumber(op, index, vacated, taken, key)
                return
            reason = (
                f"“{WorkspacePaths.split(op.target)[1]}” is also the new name of “{duplicate.source}”."
                if duplicate is not None and duplicate.source
                else f"“{WorkspacePaths.split(op.target)[1]}” already exists."
            )
            self._mark(op, OpStatus.CONFLICT, reason)
            return
        taken[tkey] = op

    def _renumber(self, op, index, vacated, taken, key) -> None:  # type: ignore[no-untyped-def]
        assert op.target is not None
        directory, name = WorkspacePaths.split(op.target)
        stem, ext = split_name(name)
        counter = 2
        while True:
            candidate = f"{stem} ({counter}){ext}"
            full = WorkspacePaths.join(directory, candidate)
            ckey = key(full)
            if ckey not in taken and (not index.exists(full) or ckey in vacated):
                break
            counter += 1
        op.target = full
        taken[ckey] = op
        self._mark(op, OpStatus.WARNING, f"The name was already taken; using “{candidate}” instead.")


def make_validator(root: Path, **kwargs) -> BatchValidator:  # type: ignore[no-untyped-def]
    """Convenience constructor used by tests and the executor's pre-flight."""
    case_insensitive = kwargs.pop("case_insensitive", None)
    return BatchValidator(WorkspacePaths(root, case_insensitive=case_insensitive), **kwargs)
