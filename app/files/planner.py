"""Turns AI (or offline) actions into a validated :class:`Plan`.

Pipeline: rule actions are expanded over the selected files by the
:class:`RuleEngine`; explicit actions are resolved against the selection; every
path is validated against the workspace; the resulting *drafts* are then checked
as a batch by :class:`BatchValidator`. Nothing here touches the disk beyond
reading directory listings.
"""

from __future__ import annotations

import os
import unicodedata
from dataclasses import dataclass
from pathlib import Path

from app.ai import schemas as s
from app.files.path_validator import PathError, WorkspacePaths
from app.files.plan import OpKind, OpStatus, Plan, PlannedOp
from app.files.plan_validator import BatchValidator, DirIndex
from app.files.rename_engine import (
    FileState,
    RuleEngine,
    finalize_name,
    resolve_explicit_target,
)
from app.files.scanner import FileEntry
from app.utils.logger import get_logger

log = get_logger("planner")


@dataclass(slots=True)
class PlannerOptions:
    """Behaviour switches (mirrors the File Operations settings)."""

    prevent_overwrite: bool = True
    duplicate_policy: str = "flag"
    delete_mode: str = "trash"
    case_insensitive: bool | None = None  # None = platform default
    enforce_path_limit: bool | None = None  # None = Windows only


class _Selection:
    """Lookup of the selected files by the names the AI may use."""

    def __init__(self, files: list[FileEntry], paths: WorkspacePaths) -> None:
        self._exact = {f.rel_path: f for f in files}
        self._nfc: dict[str, list[FileEntry]] = {}
        self._base: dict[str, list[FileEntry]] = {}
        for f in files:
            self._nfc.setdefault(unicodedata.normalize("NFC", f.rel_path), []).append(f)
            self._base.setdefault(unicodedata.normalize("NFC", f.name), []).append(f)

    def find(self, raw: str) -> FileEntry | str:
        """The selected file named ``raw``, or an error message."""
        text = raw.replace("\\", "/")
        if text in self._exact:
            return self._exact[text]
        normalised = unicodedata.normalize("NFC", text)
        for table in (self._nfc, self._base):
            hits = table.get(normalised, [])
            if len(hits) == 1:
                return hits[0]
            if len(hits) > 1:
                return f"“{raw}” matches several files; use the full relative path."
        return f"“{raw}” is not one of the selected files."


class Planner:
    """Builds :class:`Plan` objects for one workspace and selection."""

    def __init__(self, root: Path | str, files: list[FileEntry], options: PlannerOptions | None = None) -> None:
        self.options = options or PlannerOptions()
        self.paths = WorkspacePaths(root, case_insensitive=self.options.case_insensitive)
        self.files = files
        self._selection = _Selection(files, self.paths)
        self._next_id = 1
        self._index = DirIndex(self.paths)

    # ------------------------------------------------------------------ build
    def build(
        self,
        actions: list[s.Action],
        rejected: list[s.RejectedAction] | None = None,
        *,
        summary: str = "",
    ) -> Plan:
        """Create a validated plan for ``actions`` (rule and explicit actions may be mixed)."""
        rules = [a for a in actions if a.is_rule]
        explicit = [a for a in actions if not a.is_rule]
        outcome = RuleEngine(self.files, self.paths).apply(rules)
        drafts: list[PlannedOp] = []
        unchanged = 0
        touched: set[str] = set()
        for state in outcome.states:
            made = self._drafts_for_state(state)
            drafts.extend(made)
            if made:
                touched.add(state.entry.rel_path)
            else:
                unchanged += 1
        for action in explicit:
            made = self._drafts_for_explicit(action)
            drafts.extend(made)
        for item in rejected or []:
            drafts.append(self._invalid(OpKind.RENAME, item.raw, None, "ai", f"Rejected AI action: {item.reason}"))

        self._flag_reused_sources(drafts)
        drafts = self._with_implicit_folders(drafts)
        drafts.sort(key=lambda op: 0 if op.kind == OpKind.CREATE_FOLDER else 1)  # stable: folders first
        for op in drafts:
            op.id = self._next_id
            self._next_id += 1
            op.draft_target = op.target

        validator = BatchValidator(
            self.paths,
            prevent_overwrite=self.options.prevent_overwrite,
            duplicate_policy=self.options.duplicate_policy,
            delete_mode=self.options.delete_mode,
        )
        plan = Plan(
            root=self.paths.root,
            ops=drafts,
            summary=summary,
            sort=outcome.sort,
            unchanged=unchanged,
            notes=list(outcome.notes),
            delete_mode=self.options.delete_mode,
            _revalidate=validator.validate,
        )
        validator.validate(plan)
        log.info(
            "Plan built: %d operation(s), %d unchanged, counts=%s",
            len(plan.ops), unchanged, dict(plan.counts()),
        )  # fmt: skip
        return plan

    # ------------------------------------------------------- draft factories
    def _invalid(self, kind: OpKind, source: str | None, target: str | None, origin: str, message: str) -> PlannedOp:
        return PlannedOp(id=0, kind=kind, source=source, target=target, origin=origin, intent_error=message)

    def _contained(self, rel: str) -> str | None:
        """Error message if ``rel`` would leave the workspace (symlink/junction), else ``None``."""
        try:
            self.paths.ensure_contained(rel)
        except PathError as exc:
            return str(exc)
        return None

    def _drafts_for_state(self, state: FileState) -> list[PlannedOp]:
        entry = state.entry
        ops: list[PlannedOp] = []
        origin = state.origins[-1] if state.origins else "rule"
        for error in state.errors:
            ops.append(self._invalid(OpKind.MOVE, entry.rel_path, None, origin, error))
        if state.errors:
            return ops
        if state.delete:
            op = PlannedOp(id=0, kind=OpKind.DELETE, source=entry.rel_path, origin="delete")
            if state.renamed or state.moved:
                op.static_messages.append("Rename/move was ignored because the file is being deleted.")
            return [op]

        changed = state.renamed or state.moved
        if changed:
            op = self._relocation_for_state(state, origin)
            if op is not None:
                ops.append(op)
        for folder in state.copy_folders:
            copy = PlannedOp(
                id=0, kind=OpKind.COPY, source=entry.rel_path, target=WorkspacePaths.join(folder, entry.name), origin="copy"
            )
            if changed:
                copy.intent_error = "A file cannot be copied and renamed/moved in the same step. Do it in two steps."
            elif copy.target and (err := self._contained(copy.target)):
                copy.intent_error = err
            ops.append(copy)
        return ops

    def _relocation_for_state(self, state: FileState, origin: str) -> PlannedOp | None:
        entry = state.entry
        directory_abs = str(self.paths.absolute(state.folder))
        result = finalize_name(
            state.stem, entry.ext, directory_abs=directory_abs, enforce_path_limit=self.options.enforce_path_limit
        )
        kind = OpKind.MOVE if state.moved else OpKind.RENAME
        if result.errors:
            attempted = WorkspacePaths.join(state.folder, state.stem + entry.ext)
            return self._invalid(kind, entry.rel_path, attempted, origin, result.errors[0])
        target = WorkspacePaths.join(state.folder, result.name)
        if target == entry.rel_path:
            return None
        op = PlannedOp(id=0, kind=kind, source=entry.rel_path, target=target, origin=origin, static_messages=list(result.warnings))
        if err := self._contained(target):
            op.intent_error = err
        return op

    def _drafts_for_explicit(self, action: s.Action) -> list[PlannedOp]:
        if isinstance(action, s.CreateFolderAction):
            return [self._create_folder(action)]
        if isinstance(action, s.RenameAction):
            return self._explicit_rename(action)
        if isinstance(action, (s.MoveAction, s.CopyAction)):
            return self._explicit_move_copy(action)
        if isinstance(action, s.DeleteAction) and action.source is not None:
            found = self._selection.find(action.source)
            if isinstance(found, str):
                return [self._invalid(OpKind.DELETE, action.source, None, action.type, found)]
            return [PlannedOp(id=0, kind=OpKind.DELETE, source=found.rel_path, origin=action.type)]
        return []

    def _create_folder(self, action: s.CreateFolderAction) -> PlannedOp:
        try:
            rel = self.paths.normalize_relative(action.path, what="folder")
        except PathError as exc:
            return self._invalid(OpKind.CREATE_FOLDER, None, action.path, action.type, str(exc))
        op = PlannedOp(id=0, kind=OpKind.CREATE_FOLDER, target=rel, origin=action.type)
        if err := self._contained(rel):
            op.intent_error = err
        return op

    def _explicit_rename(self, action: s.RenameAction) -> list[PlannedOp]:
        found = self._selection.find(action.source)
        if isinstance(found, str):
            return [self._invalid(OpKind.RENAME, action.source, action.target, action.type, found)]
        result = resolve_explicit_target(found.name, action.target, action.change_extension)
        if result.errors:
            return [self._invalid(OpKind.RENAME, found.rel_path, action.target, action.type, result.errors[0])]
        stem, ext = _split(result.name)
        final = finalize_name(
            stem, ext, directory_abs=str(self.paths.absolute(found.directory)),
            enforce_path_limit=self.options.enforce_path_limit,
        )  # fmt: skip
        if final.errors:
            return [self._invalid(OpKind.RENAME, found.rel_path, action.target, action.type, final.errors[0])]
        target = WorkspacePaths.join(found.directory, final.name)
        if target == found.rel_path:
            return []  # already has that name
        op = PlannedOp(
            id=0, kind=OpKind.RENAME, source=found.rel_path, target=target, origin=action.type,
            static_messages=result.warnings + final.warnings,
        )  # fmt: skip
        if err := self._contained(target):
            op.intent_error = err
        return [op]

    def _explicit_move_copy(self, action: s.MoveAction | s.CopyAction) -> list[PlannedOp]:
        kind = OpKind.MOVE if isinstance(action, s.MoveAction) else OpKind.COPY
        assert action.source is not None
        found = self._selection.find(action.source)
        if isinstance(found, str):
            return [self._invalid(kind, action.source, None, action.type, found)]
        try:
            folder = self.paths.normalize_relative(action.target_folder, what="folder")
        except PathError as exc:
            return [self._invalid(kind, found.rel_path, action.target_folder, action.type, str(exc))]
        warnings: list[str] = []
        name = found.name
        if action.new_name:
            renamed = resolve_explicit_target(found.name, action.new_name, False)
            if renamed.errors:
                return [self._invalid(kind, found.rel_path, action.target_folder, action.type, renamed.errors[0])]
            name, warnings = renamed.name, renamed.warnings
        target = WorkspacePaths.join(folder, name)
        if target == found.rel_path:
            if kind == OpKind.COPY:
                return [self._invalid(kind, found.rel_path, target, action.type, "The copy would overwrite the original.")]
            return []
        op = PlannedOp(id=0, kind=kind, source=found.rel_path, target=target, origin=action.type, static_messages=warnings)
        if err := self._contained(target):
            op.intent_error = err
        return [op]

    # ------------------------------------------------------------ cross-checks
    @staticmethod
    def _flag_reused_sources(drafts: list[PlannedOp]) -> None:
        """A file may be renamed/moved/deleted once, and not both copied and modified."""
        modifier: dict[str, PlannedOp] = {}
        for op in drafts:
            if op.intent_error or not op.source or op.kind not in (OpKind.RENAME, OpKind.MOVE, OpKind.DELETE):
                continue
            if op.source in modifier:
                op.intent_error = "This file is already used by another action in this plan."
            else:
                modifier[op.source] = op
        for op in drafts:
            if op.kind == OpKind.COPY and not op.intent_error and op.source in modifier:
                op.intent_error = "A file cannot be copied and renamed/moved/deleted in the same step."

    def _with_implicit_folders(self, drafts: list[PlannedOp]) -> list[PlannedOp]:
        """Add a folder-creation step for move/copy targets whose folder is neither present nor requested."""
        declared = {self.paths.key(op.target) for op in drafts if op.kind == OpKind.CREATE_FOLDER and op.target}
        extra: list[PlannedOp] = []
        for op in drafts:
            if op.intent_error or op.kind not in (OpKind.MOVE, OpKind.COPY, OpKind.RENAME) or not op.target:
                continue
            folder = WorkspacePaths.split(op.target)[0]
            if not folder:
                continue
            fkey = self.paths.key(folder)
            if fkey in declared or self._index.dir_exists(folder):
                continue
            declared.add(fkey)
            extra.append(
                PlannedOp(id=0, kind=OpKind.CREATE_FOLDER, target=folder, origin="auto", implicit=True)
            )
        return drafts + extra


def _split(name: str) -> tuple[str, str]:
    return os.path.splitext(name)


__all__ = ["OpStatus", "Planner", "PlannerOptions"]
