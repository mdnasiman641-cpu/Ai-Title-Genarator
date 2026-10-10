"""Executes plans safely and undoes them later.

Safety properties:

* **Journal first** - every item is written to SQLite as ``pending`` *before* the
  first file is touched, so a crash leaves a record that Undo can use.
* **All-or-nothing** - if any step fails (locked file, disk full, cancel ...) every
  completed step is rolled back in reverse order.
* **Two-phase renames** - collisions inside the batch are avoided with temp names.
* **Reversible deletes** - by default files are moved into an undo-trash inside
  the workspace (same volume, so instant) rather than being destroyed.
* **Undo pre-flight** - before reversing anything, the current state of the disk is
  checked; if it cannot be reversed safely the reasons are explained and nothing
  is changed.
"""

from __future__ import annotations

import os
import shutil
import threading
import uuid
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path

from app.config.constants import TEMP_PREFIX, TRASH_DIR_NAME
from app.database.database import Database
from app.database.models import OperationItemRecord, OperationRecord
from app.files.move_engine import (
    FsError,
    OperationCancelled,
    StepLog,
    delete_file,
    make_folder,
    remove_empty_dir,
    rollback,
    safe_copy,
    safe_move,
    trash_relative,
)
from app.files.path_validator import PathError, WorkspacePaths
from app.files.plan import OpKind, Plan, PlannedOp
from app.files.rename_engine import needs_two_phase, relocate_batch
from app.utils.logger import get_logger

log = get_logger("ops")
ProgressFn = Callable[[int, int, str], None]

STATUS_LABELS = {"rename": "Renamed", "move": "Moved", "copy": "Copied"}


@dataclass(slots=True)
class ExecutionResult:
    """Outcome of applying a plan."""

    ok: bool
    operation_id: int | None = None
    applied: int = 0
    message: str = ""
    rolled_back: bool = False
    rollback_failures: list[str] = field(default_factory=list)
    changes: dict[str, str] = field(default_factory=dict)  # new rel path -> status label
    summary: str = ""
    irreversible: int = 0  # permanently deleted files that Undo cannot restore


@dataclass(slots=True)
class UndoCheck:
    """Whether an operation can be undone right now, and why not."""

    blockers: list[str] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)

    @property
    def possible(self) -> bool:
        return not self.blockers


@dataclass(slots=True)
class UndoResult:
    ok: bool
    message: str = ""
    restored: int = 0
    notes: list[str] = field(default_factory=list)
    blockers: list[str] = field(default_factory=list)


def summarize_kinds(kinds: dict[str, int]) -> str:
    """``{"rename": 126}`` -> ``"126 rename operations"``; mixed -> ``"3 move, 2 delete operations"``."""
    parts = [f"{count} {kind.replace('_', ' ')}" for kind, count in kinds.items() if count]
    if not parts:
        return "no operations"
    return ", ".join(parts) + (" operation" if sum(kinds.values()) == 1 else " operations")


class OperationManager:
    """Runs plans and undoes recorded operations."""

    def __init__(self, db: Database | None = None) -> None:
        self._db = db

    # ============================================================= execution
    def execute(
        self,
        plan: Plan,
        *,
        command: str = "",
        provider: str = "",
        keep_history: bool = True,
        cancel: threading.Event | None = None,
        progress: ProgressFn | None = None,
    ) -> ExecutionResult:
        """Apply every applicable operation of ``plan`` (all-or-nothing, journaled)."""
        cancel = cancel or threading.Event()
        before = {op.id for op in plan.applicable_ops()}
        plan.revalidate()  # pre-flight: the disk may have changed since the preview
        after = {op.id for op in plan.applicable_ops()}
        if before - after:
            lost = [op for op in plan.ops if op.id in before - after]
            detail = "; ".join(f"{op.source or op.target}: {op.messages[-1] if op.messages else 'no longer valid'}" for op in lost[:3])
            return ExecutionResult(False, message=f"The folder changed since the preview. {detail}")
        ops = plan.applicable_ops()
        if not ops:
            return ExecutionResult(True, message="Nothing to do.")

        paths = WorkspacePaths(plan.root)
        token = uuid.uuid4().hex[:8]
        trash_mode = plan.delete_mode == "trash"
        items = self._build_items(ops, token, trash_mode, paths)
        kinds = plan.kind_summary()
        summary = summarize_kinds(kinds)
        op_id: int | None = None
        item_ids: dict[int, int] = {}
        if self._db is not None:
            op_id = self._db.create_operation(str(plan.root), command, provider, summary, items)
            item_ids = {it.seq: it.id for it in self._db.get_items(op_id)}

        steps = StepLog()
        total = len(items) + sum(1 for it in items if it.type in ("rename", "move") and it.temp_path)
        done = 0

        def tick(message: str = "") -> None:
            nonlocal done
            done += 1
            if progress is not None:
                progress(min(done, total), total, message or f"Processing {min(done, total)} / {total}")
            if cancel.is_set():
                raise OperationCancelled

        permanent_failures: list[str] = []
        irreversible = 0
        try:
            self._run_forward(items, paths, steps, token, cancel, tick, plan.root)
            if not trash_mode:
                irreversible, permanent_failures = self._permanent_deletes(items, paths, cancel, tick)
        except OperationCancelled:
            return self._fail(op_id, steps, "Cancelled. No files were changed.", cancel_ok=True)
        except FsError as exc:
            return self._fail(op_id, steps, f"{exc.message} ({Path(exc.path).name})" if exc.path else exc.message)
        except OSError as exc:  # defensive: anything unexpected is rolled back too
            return self._fail(op_id, steps, str(exc))

        changes = {
            op.target: STATUS_LABELS[op.kind.value]
            for op in ops
            if op.target and op.kind.value in STATUS_LABELS
        }
        status = "partial" if permanent_failures else "applied"
        if self._db is not None and op_id is not None:
            self._db.update_item_statuses((item_ids[it.seq], "done") for it in items if it.seq in item_ids)
            self._db.set_operation_status(op_id, status)
            has_trash = any(it.type == "delete" and it.extra.get("mode") == "trash" for it in items)
            if not keep_history and not has_trash and not permanent_failures:
                self._db.delete_operation(op_id)
                op_id = None
        message = ""
        if permanent_failures:
            message = "Some files could not be deleted: " + "; ".join(permanent_failures[:3])
        applied = sum(1 for op in ops if not (op.implicit and op.kind == OpKind.CREATE_FOLDER))
        log.info("Applied operation %s: %s (%d items)", op_id, summary, len(items))
        return ExecutionResult(
            ok=not permanent_failures, operation_id=op_id, applied=applied, message=message, changes=changes,
            summary=summary, irreversible=irreversible,
        )  # fmt: skip

    # ------------------------------------------------------- building items
    def _build_items(
        self, ops: list[PlannedOp], token: str, trash_mode: bool, paths: WorkspacePaths
    ) -> list[OperationItemRecord]:
        items: list[OperationItemRecord] = []

        def add(type_: str, source: str | None, target: str | None, extra: dict[str, object] | None = None, temp: str | None = None) -> None:
            items.append(
                OperationItemRecord(seq=len(items) + 1, type=type_, source=source, target=target, temp_path=temp, extra=extra or {})
            )

        for op in ops:
            if op.kind == OpKind.CREATE_FOLDER and op.target:
                parts = op.target.split("/")
                prefixes = ["/".join(parts[: i + 1]) for i in range(len(parts))]
                created = [p for p in prefixes if not os.path.lexists(paths.absolute(p))]  # only what is really new
                add("create_folder", None, op.target, {"created": created})
        for op in ops:
            if op.overwrite and op.target:
                add("delete", op.target, trash_relative(token, op.target), {"mode": "trash", "reason": "overwrite", "token": token})
        for op in ops:
            if op.kind == OpKind.DELETE and op.source:
                if trash_mode:
                    add("delete", op.source, trash_relative(token, op.source), {"mode": "trash", "token": token})
                else:
                    add("delete", op.source, None, {"mode": "permanent"})
        for op in ops:
            if op.kind == OpKind.COPY and op.source and op.target:
                size = self._size(paths, op.source)
                add("copy", op.source, op.target, {"size": size})
        relocations = [op for op in ops if op.kind in (OpKind.RENAME, OpKind.MOVE) and op.source and op.target]
        pairs = [(paths.absolute(op.source or ""), paths.absolute(op.target or "")) for op in relocations]
        two_phase = needs_two_phase(pairs)
        for index, op in enumerate(relocations, start=1):
            temp = None
            if two_phase:
                head, _ = WorkspacePaths.split(op.source or "")
                temp = WorkspacePaths.join(head, f"{TEMP_PREFIX}{token}_{index:03d}")
            add(op.kind.value, op.source, op.target, {"token": token}, temp)
        return items

    @staticmethod
    def _size(paths: WorkspacePaths, rel: str) -> int:
        try:
            return os.stat(paths.absolute(rel)).st_size
        except OSError:
            return -1

    # ------------------------------------------------------------- forward
    def _run_forward(
        self,
        items: list[OperationItemRecord],
        paths: WorkspacePaths,
        steps: StepLog,
        token: str,
        cancel: threading.Event,
        tick: Callable[[str], None],
        root: Path,
    ) -> None:
        def absolute(rel: str) -> Path:
            return paths.ensure_contained(rel)

        for item in items:
            if item.type == "create_folder":
                make_folder(absolute(item.target or ""), steps)
                tick("")
        for item in items:
            if item.type == "delete" and item.extra.get("mode") == "trash":
                safe_move(absolute(item.source or ""), root.joinpath(*(item.target or "").split("/")), steps, create_parents=True)
                tick("")
        for item in items:
            if item.type == "copy":
                safe_copy(absolute(item.source or ""), absolute(item.target or ""), steps)
                tick("")
        relocations = [it for it in items if it.type in ("rename", "move")]
        if relocations:
            pairs = [(absolute(it.source or ""), absolute(it.target or "")) for it in relocations]
            relocate_batch(pairs, steps, token=token, cancel=cancel, progress=lambda d, t: tick(""))

    def _permanent_deletes(
        self, items: list[OperationItemRecord], paths: WorkspacePaths, cancel: threading.Event, tick: Callable[[str], None]
    ) -> tuple[int, list[str]]:
        deleted, failures = 0, []
        for item in items:
            if item.type == "delete" and item.extra.get("mode") == "permanent":
                if cancel.is_set():
                    failures.append("Cancelled before all files were deleted.")
                    break
                try:
                    delete_file(paths.ensure_contained(item.source or ""))
                    deleted += 1
                except (FsError, PathError) as exc:
                    failures.append(str(exc))
                tick("")
        return deleted, failures

    def _fail(self, op_id: int | None, steps: StepLog, message: str, *, cancel_ok: bool = False) -> ExecutionResult:
        failures = rollback(steps)
        status = "rolled_back" if not failures else "partial"
        if self._db is not None and op_id is not None:
            self._db.set_operation_status(op_id, status)
        if failures:
            message += " Some changes could not be reverted: " + "; ".join(failures[:3])
        log.warning("Operation %s failed (%s): %s", op_id, status, message)
        return ExecutionResult(
            False, operation_id=op_id, message=message, rolled_back=not failures, rollback_failures=failures
        )

    # ================================================================== undo
    def check_undo(self, op_id: int) -> UndoCheck:
        """Inspect the disk to see whether ``op_id`` can be reversed without losing anything."""
        record, items = self._load(op_id)
        check = UndoCheck()
        root = Path(record.workspace)
        if not root.is_dir():
            check.blockers.append(f"The folder “{root}” no longer exists.")
            return check
        paths = WorkspacePaths(root)
        reloc = [it for it in items if it.type in ("rename", "move")]
        vacating: set[str] = set()
        for it in reloc:
            cur = self._current_location(paths, it)
            if cur is not None:
                vacating.add(paths.key(cur))
        for it in items:
            try:
                self._check_item(it, paths, vacating, check)
            except PathError as exc:
                check.blockers.append(str(exc))
        return check

    def _current_location(self, paths: WorkspacePaths, it: OperationItemRecord) -> str | None:
        for rel in (it.target, it.temp_path):
            if rel and os.path.lexists(paths.absolute(rel)):
                return rel
        return None

    def _check_item(self, it: OperationItemRecord, paths: WorkspacePaths, vacating: set[str], check: UndoCheck) -> None:
        exists = lambda rel: bool(rel) and os.path.lexists(paths.ensure_contained(rel))  # noqa: E731
        if it.type in ("rename", "move"):
            cur = self._current_location(paths, it)
            if cur is None:
                if exists(it.source):
                    check.notes.append(f"“{it.source}” is already back at its original name.")
                else:
                    check.blockers.append(f"“{it.target}” is missing (it was moved, renamed or deleted after this operation).")
            elif exists(it.source) and paths.key(it.source or "") not in vacating and paths.key(it.source or "") != paths.key(cur):
                check.blockers.append(f"A file named “{it.source}” already exists, so “{it.target}” cannot be restored to it.")
        elif it.type == "copy":
            if exists(it.target):
                size = it.extra.get("size", -1)
                if size not in (-1, None) and os.stat(paths.absolute(it.target or "")).st_size != size:
                    check.blockers.append(f"The copy “{it.target}” was modified after it was created, so it will not be deleted.")
        elif it.type == "create_folder":
            target = paths.absolute(it.target or "")
            if target.is_dir() and any(target.iterdir()):
                check.notes.append(f"Folder “{it.target}” is not empty and will be kept.")
        elif it.type == "delete":
            if it.extra.get("mode") == "permanent":
                check.notes.append(f"“{it.source}” was permanently deleted and cannot be restored.")
            elif not exists(it.target):
                if exists(it.source):
                    check.notes.append(f"“{it.source}” has already been restored.")
                else:
                    check.blockers.append(f"“{it.source}” is no longer in the undo-trash (was the trash emptied?).")
            elif exists(it.source) and paths.key(it.source or "") not in vacating:
                check.blockers.append(f"A file already exists at “{it.source}”, so the deleted file cannot be restored there.")

    def undo(self, op_id: int, *, cancel: threading.Event | None = None, progress: ProgressFn | None = None) -> UndoResult:
        """Reverse operation ``op_id`` (after a safety pre-check)."""
        cancel = cancel or threading.Event()
        record, items = self._load(op_id)
        check = self.check_undo(op_id)
        if not check.possible:
            return UndoResult(False, "Undo is not possible right now.", notes=check.notes, blockers=check.blockers)
        root = Path(record.workspace)
        paths = WorkspacePaths(root)
        steps = StepLog()
        total = len(items)
        done = 0

        def tick() -> None:
            nonlocal done
            done += 1
            if progress is not None:
                progress(min(done, total), total, f"Undoing {min(done, total)} / {total}")
            if cancel.is_set():
                raise OperationCancelled

        restored = 0
        try:
            reloc = [it for it in items if it.type in ("rename", "move")]
            pairs = []
            for it in reloc:
                cur = self._current_location(paths, it)
                if cur is not None:
                    pairs.append((paths.absolute(cur), paths.absolute(it.source or "")))
            if pairs:
                relocate_batch(pairs, steps, token=uuid.uuid4().hex[:8], cancel=cancel, progress=lambda d, t: None)
                restored += len(pairs)
            for it in items:
                if it.type == "delete" and it.extra.get("mode") == "trash":
                    trash_abs = paths.absolute(it.target or "")
                    if os.path.lexists(trash_abs):
                        safe_move(trash_abs, paths.absolute(it.source or ""), steps, create_parents=True)
                        restored += 1
                tick()
        except OperationCancelled:
            rollback(steps)
            return UndoResult(False, "Undo cancelled. Nothing was changed.")
        except (FsError, PathError) as exc:
            failures = rollback(steps)
            message = f"Undo failed: {exc}. Nothing was changed."
            if failures:
                message = f"Undo failed: {exc}. Some files could not be put back: {'; '.join(failures[:3])}"
            return UndoResult(False, message)
        notes = list(check.notes)
        for it in reversed(items):  # copies and folders last: these cannot fail the undo
            if it.type == "copy":
                target = paths.absolute(it.target or "")
                if os.path.lexists(target):
                    try:
                        os.remove(target)
                        restored += 1
                    except OSError as exc:
                        notes.append(f"Could not delete copy “{it.target}”: {exc}")
            elif it.type == "create_folder":
                created = it.extra.get("created") or [it.target]
                for rel in reversed([str(c) for c in created]):  # deepest first; only folders we created
                    folder = paths.absolute(rel)
                    if folder.is_dir():
                        remove_empty_dir(folder)
        self._cleanup_trash(paths.root)
        if self._db is not None:
            self._db.update_item_statuses((it.id, "undone") for it in items)
            self._db.set_operation_status(op_id, "undone", undone=True)
        log.info("Undid operation %s (%d restored)", op_id, restored)
        return UndoResult(True, "Undone.", restored=restored, notes=notes)

    @staticmethod
    def _cleanup_trash(root: Path) -> None:
        """Remove empty directories left in the undo-trash."""
        trash = root / TRASH_DIR_NAME
        if not trash.is_dir():
            return
        for current, _dirs, _files in os.walk(trash, topdown=False):
            remove_empty_dir(Path(current))

    def _load(self, op_id: int) -> tuple[OperationRecord, list[OperationItemRecord]]:
        if self._db is None:
            raise RuntimeError("No history database configured.")
        record = self._db.get_operation(op_id)
        if record is None:
            raise KeyError(f"Operation {op_id} not found")
        return record, self._db.get_items(op_id)

    # ================================================================= trash
    @staticmethod
    def trash_size(workspace: Path) -> int:
        """Number of files currently in the undo-trash of ``workspace``."""
        trash = workspace / TRASH_DIR_NAME
        return sum(len(files) for _d, _s, files in os.walk(trash)) if trash.is_dir() else 0

    @staticmethod
    def empty_trash(workspace: Path) -> int:
        """Permanently delete the undo-trash (operations that deleted files can no longer be undone)."""
        trash = workspace / TRASH_DIR_NAME
        if not trash.is_dir():
            return 0
        count = sum(len(files) for _d, _s, files in os.walk(trash))
        shutil.rmtree(trash, ignore_errors=True)
        return count
