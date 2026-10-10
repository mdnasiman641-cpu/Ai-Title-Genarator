"""The closed vocabulary of actions the AI may return.

Two families exist:

* **Explicit actions** name concrete files: ``rename``, ``move``, ``copy``,
  ``create_folder``, ``delete``.
* **Rule actions** describe a deterministic transformation that the *local*
  rename engine applies to every selected file: ``add_prefix``, ``add_suffix``,
  ``remove_text``, ``replace_text``, ``change_case``, ``numbering`` and ``sort``
  (``move``/``copy``/``delete`` may also act as rules via ``match``).

Keeping rules separate is what makes large folders cheap: the model answers
once and the app does the per-file work offline.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import ClassVar

from app.config.constants import (
    ACTION_ADD_PREFIX,
    ACTION_ADD_SUFFIX,
    ACTION_CHANGE_CASE,
    ACTION_COPY,
    ACTION_CREATE_FOLDER,
    ACTION_DELETE,
    ACTION_MOVE,
    ACTION_NUMBERING,
    ACTION_REMOVE_TEXT,
    ACTION_RENAME,
    ACTION_REPLACE_TEXT,
    ACTION_SORT,
)

CASE_MODES = ("lower", "upper", "title", "sentence")
SORT_KEYS = ("name", "date", "size", "type", "duration")
NUMBER_POSITIONS = ("prefix", "suffix", "replace")


class Action:
    """Base class: every action has a fixed ``type`` string."""

    type: ClassVar[str] = ""

    @property
    def is_rule(self) -> bool:
        """True if the action is applied across files by the local engine."""
        return False


@dataclass(frozen=True, slots=True)
class RenameAction(Action):
    type: ClassVar[str] = ACTION_RENAME
    source: str
    target: str
    change_extension: bool = False


@dataclass(frozen=True, slots=True)
class MoveAction(Action):
    """Move one file (``source``) or every file whose name contains any ``match`` text."""

    type: ClassVar[str] = ACTION_MOVE
    target_folder: str
    source: str | None = None
    match: tuple[str, ...] = ()
    new_name: str | None = None

    @property
    def is_rule(self) -> bool:
        return self.source is None


@dataclass(frozen=True, slots=True)
class CopyAction(Action):
    type: ClassVar[str] = ACTION_COPY
    target_folder: str
    source: str | None = None
    match: tuple[str, ...] = ()
    new_name: str | None = None

    @property
    def is_rule(self) -> bool:
        return self.source is None


@dataclass(frozen=True, slots=True)
class CreateFolderAction(Action):
    type: ClassVar[str] = ACTION_CREATE_FOLDER
    path: str


@dataclass(frozen=True, slots=True)
class DeleteAction(Action):
    type: ClassVar[str] = ACTION_DELETE
    source: str | None = None
    match: tuple[str, ...] = ()

    @property
    def is_rule(self) -> bool:
        return self.source is None


@dataclass(frozen=True, slots=True)
class AddPrefixAction(Action):
    type: ClassVar[str] = ACTION_ADD_PREFIX
    text: str
    separator: str | None = None
    match: tuple[str, ...] = ()

    @property
    def is_rule(self) -> bool:
        return True


@dataclass(frozen=True, slots=True)
class AddSuffixAction(Action):
    type: ClassVar[str] = ACTION_ADD_SUFFIX
    text: str
    separator: str | None = None
    match: tuple[str, ...] = ()

    @property
    def is_rule(self) -> bool:
        return True


@dataclass(frozen=True, slots=True)
class RemoveTextAction(Action):
    type: ClassVar[str] = ACTION_REMOVE_TEXT
    texts: tuple[str, ...]
    case_sensitive: bool = False
    match: tuple[str, ...] = ()

    @property
    def is_rule(self) -> bool:
        return True


@dataclass(frozen=True, slots=True)
class ReplaceTextAction(Action):
    type: ClassVar[str] = ACTION_REPLACE_TEXT
    find: str
    replace: str
    case_sensitive: bool = False
    match: tuple[str, ...] = ()

    @property
    def is_rule(self) -> bool:
        return True


@dataclass(frozen=True, slots=True)
class NumberingAction(Action):
    """Sequential numbers. ``replace`` builds the whole name from ``template`` (must contain ``{n}``)."""

    type: ClassVar[str] = ACTION_NUMBERING
    start: int = 1
    width: int = 2
    position: str = "suffix"
    template: str | None = None
    separator: str = " "
    match: tuple[str, ...] = ()

    @property
    def is_rule(self) -> bool:
        return True


@dataclass(frozen=True, slots=True)
class SortAction(Action):
    """Orders the working list (affects numbering order and the table); touches no files."""

    type: ClassVar[str] = ACTION_SORT
    by: str = "name"
    descending: bool = False

    @property
    def is_rule(self) -> bool:
        return True


@dataclass(frozen=True, slots=True)
class ChangeCaseAction(Action):
    type: ClassVar[str] = ACTION_CHANGE_CASE
    mode: str = "title"
    match: tuple[str, ...] = ()

    @property
    def is_rule(self) -> bool:
        return True


ANY_ACTION = (
    RenameAction | MoveAction | CopyAction | CreateFolderAction | DeleteAction | AddPrefixAction
    | AddSuffixAction | RemoveTextAction | ReplaceTextAction | NumberingAction | SortAction | ChangeCaseAction
)  # fmt: skip


@dataclass(frozen=True, slots=True)
class RejectedAction:
    """An action from the AI that failed schema validation (shown as invalid in the preview)."""

    index: int
    raw: str
    reason: str


@dataclass(slots=True)
class ParsedResponse:
    """A syntactically valid AI answer."""

    actions: list[Action]
    summary: str = ""
    rejected: list[RejectedAction] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
