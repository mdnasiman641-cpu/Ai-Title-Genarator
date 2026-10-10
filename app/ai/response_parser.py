"""Turns raw model text into validated :class:`Action` objects.

AI output is **untrusted data**. This module is the first gate:

* the text must contain one JSON object (code fences / chatter are tolerated);
* keys that smell like command execution anywhere in the JSON reject the whole
  answer;
* every action must be one of the allowed types with correctly typed fields -
  otherwise that single action is rejected and reported;
* nothing in here touches the file system (path safety is checked later by the
  planner against the real workspace).
"""

from __future__ import annotations

import json
import re
from typing import Any

from app.ai.schemas import (
    CASE_MODES,
    NUMBER_POSITIONS,
    SORT_KEYS,
    Action,
    AddPrefixAction,
    AddSuffixAction,
    ChangeCaseAction,
    CopyAction,
    CreateFolderAction,
    DeleteAction,
    MoveAction,
    NumberingAction,
    ParsedResponse,
    RejectedAction,
    RemoveTextAction,
    RenameAction,
    ReplaceTextAction,
    SortAction,
)
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
    ALLOWED_ACTION_TYPES,
    FORBIDDEN_RESPONSE_KEYS,
    MAX_ACTIONS_PER_RESPONSE,
    MAX_AI_RESPONSE_CHARS,
)
from app.utils.helpers import truncate

MAX_TEXT_LEN = 300
MAX_LIST_LEN = 500
_BENIGN_KEYS = {"reason", "note", "notes", "comment", "description", "explanation"}
_SHELL_TOKENS = frozenset(
    {"shell", "powershell", "pwsh", "cmd", "sh", "bash", "exec", "execute", "script", "registry", "system",
     "process", "subprocess", "run", "invoke", "eval", "command", "terminal"}
)  # fmt: skip
_FENCE = re.compile(r"^```[a-zA-Z0-9_-]*\s*|\s*```$")


class ResponseError(ValueError):
    """The whole AI answer is unusable or unsafe. The message is shown to the user."""

    def __init__(self, message: str, *, unsafe: bool = False) -> None:
        super().__init__(message)
        self.unsafe = unsafe


# ------------------------------------------------------------------ JSON
def extract_json(text: str) -> Any:
    """Decode the first JSON value in ``text``, ignoring Markdown fences and surrounding prose."""
    if len(text) > MAX_AI_RESPONSE_CHARS:
        raise ResponseError("The AI answer is unreasonably large.")
    cleaned = text.strip().lstrip("\ufeff")
    cleaned = _FENCE.sub("", cleaned).strip()
    try:
        return json.loads(cleaned)
    except ValueError:
        pass
    decoder = json.JSONDecoder()
    for match in re.finditer(r"[{\[]", cleaned):
        try:
            value, _end = decoder.raw_decode(cleaned[match.start() :])
        except ValueError:
            continue
        return value
    raise ResponseError("The AI did not return valid JSON.")


def _scan_forbidden(value: Any, depth: int = 0) -> None:
    """Reject the whole answer if any object key looks like a command-execution attempt."""
    if depth > 12:
        raise ResponseError("The AI answer is nested too deeply.", unsafe=True)
    if isinstance(value, dict):
        for key, inner in value.items():
            if str(key).strip().lower() in FORBIDDEN_RESPONSE_KEYS:
                raise ResponseError(f"Unsafe AI answer rejected (contains “{key}”).", unsafe=True)
            _scan_forbidden(inner, depth + 1)
    elif isinstance(value, list):
        for inner in value:
            _scan_forbidden(inner, depth + 1)


# --------------------------------------------------------- field helpers
class _Invalid(ValueError):
    """Raised by field readers; becomes a RejectedAction."""


def _str(raw: dict[str, Any], key: str, *, required: bool = True, allow_empty: bool = False) -> str | None:
    value = raw.get(key)
    if value is None:
        if required:
            raise _Invalid(f"missing “{key}”")
        return None
    if not isinstance(value, str):
        raise _Invalid(f"“{key}” must be text")
    if len(value) > MAX_TEXT_LEN:
        raise _Invalid(f"“{key}” is too long")
    if not allow_empty and not value.strip():
        if required:
            raise _Invalid(f"“{key}” is empty")
        return None
    if "\x00" in value:
        raise _Invalid(f"“{key}” contains a NUL character")
    return value


def _bool(raw: dict[str, Any], key: str, default: bool = False) -> bool:
    value = raw.get(key, default)
    if not isinstance(value, bool):
        raise _Invalid(f"“{key}” must be true or false")
    return value


def _int(raw: dict[str, Any], key: str, default: int, low: int, high: int) -> int:
    value = raw.get(key, default)
    if isinstance(value, bool) or not isinstance(value, int):
        if isinstance(value, str) and value.strip().isdigit():
            value = int(value.strip())
        else:
            raise _Invalid(f"“{key}” must be a whole number")
    if not low <= value <= high:
        raise _Invalid(f"“{key}” must be between {low} and {high}")
    return value


def _match(raw: dict[str, Any]) -> tuple[str, ...]:
    value = raw.get("match")
    if value is None:
        return ()
    items = [value] if isinstance(value, str) else value
    if not isinstance(items, list) or len(items) > MAX_LIST_LEN:
        raise _Invalid("“match” must be text or a short list of text")
    out: list[str] = []
    for item in items:
        if not isinstance(item, str) or not item.strip() or len(item) > MAX_TEXT_LEN:
            raise _Invalid("“match” entries must be non-empty text")
        out.append(item)
    return tuple(out)


def _texts(raw: dict[str, Any]) -> tuple[str, ...]:
    value = raw.get("texts", raw.get("text"))
    items = [value] if isinstance(value, str) else value
    if not isinstance(items, list) or not items or len(items) > MAX_LIST_LEN:
        raise _Invalid("“texts” must be a list of text to remove")
    out: list[str] = []
    for item in items:
        if not isinstance(item, str) or not item or len(item) > MAX_TEXT_LEN:
            raise _Invalid("“texts” entries must be non-empty text")
        out.append(item)
    return tuple(out)


# --------------------------------------------------------- action builders
def _build_rename(raw: dict[str, Any]) -> Action:
    return RenameAction(
        source=_str(raw, "source") or "",
        target=_str(raw, "target") or "",
        change_extension=_bool(raw, "change_extension"),
    )


def _build_move_copy(cls: type[MoveAction] | type[CopyAction], raw: dict[str, Any]) -> Action:
    source = _str(raw, "source", required=False)
    match = _match(raw)
    if source is None and not match:
        raise _Invalid("needs “source” or “match”")
    if source is not None and match:
        raise _Invalid("use either “source” or “match”, not both")
    folder = _str(raw, "target_folder", required=False) or _str(raw, "target", required=False)
    if folder is None:
        raise _Invalid("missing “target_folder”")
    return cls(target_folder=folder, source=source, match=match, new_name=_str(raw, "new_name", required=False))


def _build_delete(raw: dict[str, Any]) -> Action:
    source = _str(raw, "source", required=False)
    match = _match(raw)
    if (source is None) == (not match):
        raise _Invalid("needs exactly one of “source” or “match”")
    return DeleteAction(source=source, match=match)


def _build_create_folder(raw: dict[str, Any]) -> Action:
    path = _str(raw, "path", required=False) or _str(raw, "target", required=False)
    if path is None:
        raise _Invalid("missing “path”")
    return CreateFolderAction(path=path)


def _build_separator(raw: dict[str, Any]) -> str | None:
    return _str(raw, "separator", required=False, allow_empty=True)


def _build_numbering(raw: dict[str, Any]) -> Action:
    position = str(raw.get("position", "suffix")).lower()
    if position not in NUMBER_POSITIONS:
        raise _Invalid(f"“position” must be one of {', '.join(NUMBER_POSITIONS)}")
    template = _str(raw, "template", required=False)
    if position == "replace" and (template is None or "{n}" not in template):
        raise _Invalid("“template” containing {n} is required when position is “replace”")
    if template is not None and "{n}" not in template:
        raise _Invalid("“template” must contain {n}")
    separator = _str(raw, "separator", required=False, allow_empty=True)
    return NumberingAction(
        start=_int(raw, "start", 1, 0, 1_000_000),
        width=_int(raw, "width", 2, 1, 8),
        position=position,
        template=template,
        separator=" " if separator is None else separator,
        match=_match(raw),
    )


def _build_sort(raw: dict[str, Any]) -> Action:
    by = str(raw.get("by", "name")).lower()
    if by not in SORT_KEYS:
        raise _Invalid(f"“by” must be one of {', '.join(SORT_KEYS)}")
    return SortAction(by=by, descending=_bool(raw, "descending"))


def _build_case(raw: dict[str, Any]) -> Action:
    mode = str(raw.get("mode", "title")).lower()
    if mode not in CASE_MODES:
        raise _Invalid(f"“mode” must be one of {', '.join(CASE_MODES)}")
    return ChangeCaseAction(mode=mode, match=_match(raw))


_BUILDERS = {
    ACTION_RENAME: _build_rename,
    ACTION_MOVE: lambda raw: _build_move_copy(MoveAction, raw),
    ACTION_COPY: lambda raw: _build_move_copy(CopyAction, raw),
    ACTION_CREATE_FOLDER: _build_create_folder,
    ACTION_DELETE: _build_delete,
    ACTION_ADD_PREFIX: lambda raw: AddPrefixAction(
        text=_str(raw, "text") or "", separator=_build_separator(raw), match=_match(raw)
    ),
    ACTION_ADD_SUFFIX: lambda raw: AddSuffixAction(
        text=_str(raw, "text") or "", separator=_build_separator(raw), match=_match(raw)
    ),
    ACTION_REMOVE_TEXT: lambda raw: RemoveTextAction(
        texts=_texts(raw), case_sensitive=_bool(raw, "case_sensitive"), match=_match(raw)
    ),
    ACTION_REPLACE_TEXT: lambda raw: ReplaceTextAction(
        find=_str(raw, "find") or "",
        replace=_str(raw, "replace", required=False, allow_empty=True) or "",
        case_sensitive=_bool(raw, "case_sensitive"),
        match=_match(raw),
    ),
    ACTION_NUMBERING: _build_numbering,
    ACTION_SORT: _build_sort,
    ACTION_CHANGE_CASE: _build_case,
}

_KNOWN_FIELDS = {
    "type", "source", "target", "target_folder", "new_name", "path", "match", "text", "texts", "separator",
    "find", "replace", "case_sensitive", "start", "width", "position", "template", "by", "descending",
    "mode", "change_extension",
}  # fmt: skip


def build_action(raw: dict[str, Any]) -> tuple[Action, list[str]]:
    """Build one action from its JSON object; returns the action and any warnings."""
    action_type = raw.get("type")
    if not isinstance(action_type, str) or not action_type.strip():
        raise _Invalid("missing action type")
    action_type = action_type.strip().lower()
    builder = _BUILDERS.get(action_type)
    if builder is None:
        raise _Invalid(f"unsupported action type “{truncate(action_type, 40)}”")
    warnings = [
        f"Ignored unknown field “{key}” in {action_type}"
        for key in raw
        if key not in _KNOWN_FIELDS and key not in _BENIGN_KEYS
    ]
    return builder(raw), warnings


def _is_shell_like(action_type: object) -> bool:
    if not isinstance(action_type, str):
        return False
    lowered = action_type.strip().lower()
    if lowered in ALLOWED_ACTION_TYPES:
        return False
    return any(token in _SHELL_TOKENS for token in re.split(r"[^a-z]+", lowered))


# ------------------------------------------------------------------ entry
def parse_response(text: str) -> ParsedResponse:
    """Parse and validate a model answer. Raises :class:`ResponseError` if it is unusable/unsafe."""
    data = extract_json(text)
    _scan_forbidden(data)
    if isinstance(data, list):
        data = {"actions": data}
    if not isinstance(data, dict):
        raise ResponseError("The AI answer must be a JSON object with an “actions” list.")
    raw_actions = data.get("actions")
    if raw_actions is None:
        raise ResponseError("The AI answer has no “actions” list.")
    if not isinstance(raw_actions, list):
        raise ResponseError("“actions” must be a list.")
    if len(raw_actions) > MAX_ACTIONS_PER_RESPONSE:
        raise ResponseError("The AI returned too many actions.")

    summary = data.get("summary")
    parsed = ParsedResponse(actions=[], summary=truncate(summary.strip(), 300) if isinstance(summary, str) else "")
    for index, raw in enumerate(raw_actions):
        if not isinstance(raw, dict):
            parsed.rejected.append(RejectedAction(index, truncate(repr(raw), 80), "action must be an object"))
            continue
        if _is_shell_like(raw.get("type")):
            raise ResponseError(f"Unsafe AI answer rejected (action type “{truncate(str(raw.get('type')), 40)}”).", unsafe=True)
        try:
            action, warnings = build_action(raw)
        except _Invalid as exc:
            parsed.rejected.append(RejectedAction(index, truncate(json.dumps(raw, ensure_ascii=False), 120), str(exc)))
            continue
        parsed.actions.append(action)
        parsed.warnings.extend(warnings)
    return parsed
