"""Plain data records mirrored by the SQLite tables."""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(slots=True)
class SavedPrompt:
    """A user-defined reusable command."""

    id: int
    name: str
    prompt: str
    provider: str
    created_at: str


@dataclass(slots=True)
class CommandRecord:
    """One entry of the command history."""

    id: int
    command: str
    provider: str
    favorite: bool
    created_at: str


@dataclass(slots=True)
class OperationRecord:
    """A batch of file operations that was applied (and can be undone)."""

    id: int
    created_at: str
    workspace: str
    command: str
    provider: str
    status: str  # running | applied | interrupted | undone | failed | rolled_back | partial
    summary: str
    undone_at: str | None = None

    @property
    def can_undo(self) -> bool:
        return self.status in {"applied", "partial", "interrupted"}


@dataclass(slots=True)
class OperationItemRecord:
    """One logical file operation inside a batch (rename, move, copy, ...)."""

    seq: int
    type: str
    source: str | None
    target: str | None
    status: str = "pending"  # pending | done | undone | skipped
    temp_path: str | None = None
    extra: dict[str, object] = field(default_factory=dict)
    id: int = 0
