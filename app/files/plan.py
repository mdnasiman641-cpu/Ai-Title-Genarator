"""Data model of a *plan*: the validated list of operations shown in the preview."""

from __future__ import annotations

from collections import Counter
from collections.abc import Callable
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path

from app.ai.schemas import SortAction


class OpKind(str, Enum):
    RENAME = "rename"
    MOVE = "move"
    COPY = "copy"
    CREATE_FOLDER = "create_folder"
    DELETE = "delete"


class OpStatus(str, Enum):
    OK = "ok"  # valid, will be applied
    WARNING = "warning"  # valid but worth a look (applied)
    CONFLICT = "conflict"  # blocked by another file / duplicate target
    INVALID = "invalid"  # illegal name, unsafe path, missing source ...

    @property
    def applicable(self) -> bool:
        return self in (OpStatus.OK, OpStatus.WARNING)


@dataclass(slots=True)
class PlannedOp:
    """One operation. ``source``/``target`` are workspace-relative, ``/``-separated."""

    id: int
    kind: OpKind
    source: str | None = None
    target: str | None = None
    draft_target: str | None = None  # target as planned; validation may adjust ``target`` (duplicate numbering)
    origin: str = ""  # action type that produced it
    status: OpStatus = OpStatus.OK
    messages: list[str] = field(default_factory=list)
    static_messages: list[str] = field(default_factory=list)  # known at draft time, never change
    intent_error: str | None = None  # draft-time failure (bad source, illegal name ...)
    included: bool = True
    overwrite: bool = False  # an existing target will be moved to the undo-trash first
    implicit: bool = False  # created automatically (missing target folder)
    noop: bool = False  # nothing to do (e.g. folder already exists)
    info: str = ""  # display-only note (e.g. Title Generator: search status and AI model); never affects status

    @property
    def counts_as_change(self) -> bool:
        return self.included and self.status.applicable and not self.noop


@dataclass(slots=True)
class Plan:
    """A set of operations plus everything the UI needs to present them."""

    root: Path
    ops: list[PlannedOp] = field(default_factory=list)
    summary: str = ""
    sort: SortAction | None = None
    unchanged: int = 0  # selected files the command left as they were
    notes: list[str] = field(default_factory=list)
    delete_mode: str = "trash"
    _revalidate: Callable[[Plan], None] | None = field(default=None, repr=False)

    # ----------------------------------------------------------- queries
    def applicable_ops(self) -> list[PlannedOp]:
        """Operations that will run: included, valid and not no-ops."""
        return [op for op in self.ops if op.counts_as_change]

    def counts(self) -> Counter[str]:
        """Counts by status value among included operations, plus ``"total"``/``"applicable"``."""
        counter: Counter[str] = Counter()
        for op in self.ops:
            if not op.included or op.noop:
                continue
            counter[op.status.value] += 1
            counter["total"] += 1
            if op.status.applicable:
                counter["applicable"] += 1
        return counter

    @property
    def delete_ops(self) -> list[PlannedOp]:
        return [op for op in self.applicable_ops() if op.kind == OpKind.DELETE]

    @property
    def has_deletes(self) -> bool:
        return bool(self.delete_ops)

    @property
    def has_blockers(self) -> bool:
        """True if any included operation is a conflict or invalid."""
        return any(op.included and not op.noop and not op.status.applicable for op in self.ops)

    @property
    def is_empty(self) -> bool:
        return not self.applicable_ops()

    def revalidate(self) -> None:
        """Re-run conflict detection after the user (un)ticks operations."""
        if self._revalidate is not None:
            self._revalidate(self)

    def kind_summary(self) -> dict[str, int]:
        """Applicable operations per kind (user-facing change count excludes implicit folders)."""
        counter: Counter[str] = Counter()
        for op in self.applicable_ops():
            if op.implicit and op.kind == OpKind.CREATE_FOLDER:
                continue
            counter[op.kind.value] += 1
        return dict(counter)
