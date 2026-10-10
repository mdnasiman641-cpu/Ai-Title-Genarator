"""AI output is untrusted: JSON parsing, schema validation and unsafe-answer rejection."""

from __future__ import annotations

import json

import pytest

from app.ai.response_parser import ResponseError, extract_json, parse_response
from app.ai.schemas import (
    AddPrefixAction,
    ChangeCaseAction,
    CopyAction,
    CreateFolderAction,
    DeleteAction,
    MoveAction,
    NumberingAction,
    RemoveTextAction,
    RenameAction,
    ReplaceTextAction,
    SortAction,
)


def parse(obj) -> object:
    return parse_response(obj if isinstance(obj, str) else json.dumps(obj, ensure_ascii=False))


# ------------------------------------------------------------ JSON syntax
def test_spec_example_response():
    result = parse({"actions": [
        {"type": "rename", "source": "VID_001.mp4", "target": "Blue Bloods - Episode 01.mp4"},
        {"type": "rename", "source": "VID_002.mp4", "target": "Blue Bloods - Episode 02.mp4"},
    ]})
    assert result.actions == [
        RenameAction("VID_001.mp4", "Blue Bloods - Episode 01.mp4"),
        RenameAction("VID_002.mp4", "Blue Bloods - Episode 02.mp4"),
    ]
    assert not result.rejected


def test_markdown_fences_and_chatter_are_tolerated():
    body = '{"actions":[{"type":"sort","by":"size"}],"summary":"ok"}'
    assert parse_response(f"```json\n{body}\n```").actions == [SortAction("size", False)]
    assert parse_response(f"Sure! Here is the plan:\n{body}\nHope that helps.").summary == "ok"
    assert parse_response("\ufeff" + body).actions


def test_top_level_list_is_accepted_as_actions():
    assert parse([{"type": "sort", "by": "date", "descending": True}]).actions == [SortAction("date", True)]


@pytest.mark.parametrize("text", ["", "not json at all", "{broken json", "[1, 2", "null", "42", '"string"'])
def test_invalid_json_is_rejected(text):
    with pytest.raises(ResponseError):
        parse_response(text)


@pytest.mark.parametrize("obj", [{}, {"actions": "rename"}, {"actions": None}, {"result": []}])
def test_missing_or_wrong_actions_list(obj):
    with pytest.raises(ResponseError):
        parse(obj)


def test_unreasonably_large_answer_rejected():
    with pytest.raises(ResponseError):
        extract_json("x" * 5_000_000)


# -------------------------------------------------------- action schemas
def test_all_action_types_parse():
    result = parse({"actions": [
        {"type": "rename", "source": "a.mp4", "target": "b.mp4", "change_extension": False},
        {"type": "move", "source": "a.mp4", "target_folder": "Season 2"},
        {"type": "move", "match": ["S02"], "target_folder": "Season 2", "new_name": None},
        {"type": "copy", "source": "a.mp4", "target_folder": "Backup"},
        {"type": "create_folder", "path": "Season 2"},
        {"type": "delete", "source": "junk.txt"},
        {"type": "delete", "match": "sample"},
        {"type": "add_prefix", "text": "Blue Bloods - "},
        {"type": "add_suffix", "text": " HD", "match": ["x"]},
        {"type": "remove_text", "texts": ["1080p", "WEB-DL"]},
        {"type": "replace_text", "find": "WEB-DL", "replace": "WEB"},
        {"type": "numbering", "start": 1, "width": 2, "position": "replace", "template": "Ep {n}"},
        {"type": "sort", "by": "name"},
        {"type": "change_case", "mode": "title"},
    ]})
    assert not result.rejected and len(result.actions) == 14
    kinds = {type(a) for a in result.actions}
    assert {RenameAction, MoveAction, CopyAction, CreateFolderAction, DeleteAction, AddPrefixAction,
            RemoveTextAction, ReplaceTextAction, NumberingAction, SortAction, ChangeCaseAction} <= kinds
    assert result.actions[1].is_rule is False and result.actions[2].is_rule is True
    assert result.actions[6] == DeleteAction(source=None, match=("sample",))


@pytest.mark.parametrize(
    "bad",
    [
        {"type": "rename", "source": "a.mp4"},  # no target
        {"type": "rename", "source": 5, "target": "x"},
        {"type": "rename", "source": "", "target": "x"},
        {"type": "move", "source": "a", "match": ["b"], "target_folder": "x"},  # both source and match
        {"type": "move", "target_folder": "x"},  # neither
        {"type": "delete"},
        {"type": "delete", "source": "a", "match": ["b"]},
        {"type": "remove_text", "texts": []},
        {"type": "remove_text", "texts": [1]},
        {"type": "replace_text", "replace": "x"},
        {"type": "numbering", "position": "middle"},
        {"type": "numbering", "position": "replace"},  # replace needs a template
        {"type": "numbering", "position": "replace", "template": "no placeholder"},
        {"type": "numbering", "width": 99},
        {"type": "sort", "by": "color"},
        {"type": "change_case", "mode": "wavy"},
        {"type": "create_folder"},
        {"type": "rename", "source": "a", "target": "b" * 400},
        {"type": "rename", "source": "a\x00", "target": "b"},
        {"type": "rename", "source": "a", "target": "b", "change_extension": "yes"},
        {"type": "organize", "source": "a"},  # unsupported but harmless type
        {"source": "a"},
        "just a string",
    ],
)
def test_malformed_actions_are_rejected_individually(bad):
    good = {"type": "sort", "by": "name"}
    result = parse({"actions": [bad, good]})
    assert len(result.rejected) == 1 and result.rejected[0].index == 0 and result.rejected[0].reason
    assert result.actions == [SortAction("name", False)]  # the valid neighbour survives


def test_unknown_benign_fields_are_ignored_with_warning():
    result = parse({"actions": [{"type": "sort", "by": "name", "reason": "r", "colour": "red"}]})
    assert result.actions and any("colour" in w for w in result.warnings)
    assert not any("reason" in w for w in result.warnings)


def test_numeric_strings_for_numbers_are_coerced():
    result = parse({"actions": [{"type": "numbering", "start": "5", "width": "3"}]})
    assert result.actions == [NumberingAction(start=5, width=3)]


# ------------------------------------------------- unsafe answers (whole)
@pytest.mark.parametrize("key", ["command", "cmd", "shell", "powershell", "script", "exec", "execute", "run", "bash", "COMMAND"])
def test_command_keys_reject_the_entire_answer(key):
    with pytest.raises(ResponseError) as err:
        parse({"actions": [{"type": "rename", "source": "a", "target": "b", key: "del /f /q *.*"}]})
    assert err.value.unsafe


def test_command_key_hidden_deep_in_json_is_found():
    with pytest.raises(ResponseError) as err:
        parse({"actions": [], "meta": {"nested": [{"powershell": "Remove-Item C:\\ -Recurse"}]}})
    assert err.value.unsafe


@pytest.mark.parametrize("action_type", ["shell", "powershell", "run_command", "exec", "execute_script", "cmd", "registry_edit", "system", "subprocess"])
def test_shell_like_action_types_reject_the_entire_answer(action_type):
    with pytest.raises(ResponseError) as err:
        parse({"actions": [{"type": "sort", "by": "name"}, {"type": action_type, "args": "whatever"}]})
    assert err.value.unsafe


def test_deeply_nested_json_is_rejected():
    nested: object = {}
    for _ in range(40):
        nested = {"a": nested}
    with pytest.raises(ResponseError):
        parse({"actions": [], "x": nested})


def test_too_many_actions_rejected():
    with pytest.raises(ResponseError):
        parse({"actions": [{"type": "sort"}] * 20_001})


def test_bangla_text_roundtrips():
    result = parse({"actions": [{"type": "rename", "source": "ভিডিও ০১.mp4", "target": "পর্ব ০১.mp4"}]})
    assert result.actions == [RenameAction("ভিডিও ০১.mp4", "পর্ব ০১.mp4")]
