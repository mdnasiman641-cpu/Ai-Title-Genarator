"""Offline command recognition (English + Bangla): no AI needed for simple commands."""

from __future__ import annotations

import pytest

from app.ai.schemas import (
    AddPrefixAction,
    AddSuffixAction,
    ChangeCaseAction,
    NumberingAction,
    RemoveTextAction,
    ReplaceTextAction,
    SortAction,
)
from app.files.local_commands import parse_local_command


def actions(command: str):
    plan = parse_local_command(command)
    assert plan is not None, f"expected an offline plan for: {command!r}"
    assert plan.description
    return plan.actions


@pytest.mark.parametrize(
    ("command", "texts"),
    [
        ("Remove 1080p from all filenames.", ("1080p",)),
        ("Remove 1080p and WEB-DL from all filenames.", ("1080p", "WEB-DL")),
        ("remove 1080p, WEB-DL, x264 from every file name", ("1080p", "WEB-DL", "x264")),
        ("Please strip \"[YTS.MX]\" from the names", ("[YTS.MX]",)),
        ("delete 720p from all filenames", ("720p",)),
        ("Remove HDRip", ("HDRip",)),
        ("১০৮০p এবং WEB-DL বাদ দাও।", ("১০৮০p", "WEB-DL")),
        ("1080p এবং WEB-DL বাদ দাও।", ("1080p", "WEB-DL")),
        ("সব ভিডিওর নাম থেকে 1080p বাদ দাও", ("1080p",)),
    ],
)
def test_remove_text(command, texts):
    assert actions(command) == [RemoveTextAction(texts=texts)]


@pytest.mark.parametrize(
    ("command", "expected"),
    [
        ("Replace WEB-DL with WEB.", [ReplaceTextAction("WEB-DL", "WEB")]),
        ("replace 'x264' with 'x265' in all filenames", [ReplaceTextAction("x264", "x265")]),
        ("Replace underscores with spaces", [ReplaceTextAction("_", " ")]),
        ("replace dots and underscores with spaces", [ReplaceTextAction(".", " "), ReplaceTextAction("_", " ")]),
        ("Replace \"WEB-DL\" with nothing", [ReplaceTextAction("WEB-DL", "")]),
        ("WEB-DL কে WEB দিয়ে replace করো", [ReplaceTextAction("WEB-DL", "WEB")]),
    ],
)
def test_replace_text(command, expected):
    assert actions(command) == expected


@pytest.mark.parametrize(
    ("command", "expected"),
    [
        ("Add prefix Blue Bloods - ", AddPrefixAction("Blue Bloods -")),
        ('Add prefix "S02 - " to all files', AddPrefixAction("S02 - ")),
        ("Add Blue Bloods to the beginning of all filenames", AddPrefixAction("Blue Bloods")),
        ("Blue Bloods নামটা শুরুতে যোগ করো।", AddPrefixAction("Blue Bloods")),
        ("সব ভিডিওর নামের শুরুতে \"Blue Bloods - \" যোগ করো", AddPrefixAction("Blue Bloods - ")),
        ("Add suffix _HD", AddSuffixAction("_HD")),
        ("Append \" (2020)\" to every name", AddSuffixAction(" (2020)")),
        ("add 2024 to the end of all filenames", AddSuffixAction("2024")),
        ("সব ভিডিওর নামের শেষে 2024 যোগ করো", AddSuffixAction("2024")),
    ],
)
def test_prefix_and_suffix(command, expected):
    assert actions(command) == [expected]


@pytest.mark.parametrize(
    ("command", "expected"),
    [
        ("Number all files from 01", NumberingAction(start=1, width=2, position="suffix")),
        ("Add sequential numbering to the end of filenames starting from 05", NumberingAction(start=5, width=2, position="suffix")),
        ("add numbering at the start starting from 100", NumberingAction(start=100, width=3, position="prefix")),
        ("Number the files sequentially", NumberingAction(start=1, width=2, position="suffix")),
        ("সব ভিডিওর নামের শেষে 01 থেকে numbering দাও।", NumberingAction(start=1, width=2, position="suffix")),
        ("সব ভিডিওর নামের শুরুতে 10 থেকে numbering দাও", NumberingAction(start=10, width=2, position="prefix")),
    ],
)
def test_numbering(command, expected):
    assert actions(command) == [expected]


@pytest.mark.parametrize(
    ("command", "mode"),
    [
        ("Convert all filenames to lowercase", "lower"),
        ("make the names UPPERCASE", "upper"),
        ("Change filenames to title case", "title"),
        ("lowercase all files", "lower"),
        ("সব ফাইলের নাম ছোট হাতের অক্ষরে করো", "lower"),
    ],
)
def test_change_case(command, mode):
    assert actions(command) == [ChangeCaseAction(mode=mode)]


@pytest.mark.parametrize(
    ("command", "expected"),
    [
        ("Sort by name", SortAction("name", False)),
        ("sort files by size descending", SortAction("size", True)),
        ("Sort by date newest first", SortAction("date", True)),
        ("নাম অনুযায়ী সাজাও", SortAction("name", False)),
    ],
)
def test_sort(command, expected):
    assert actions(command) == [expected]


@pytest.mark.parametrize(
    "command",
    [
        "",
        "   ",
        "Rename all videos using clean Blue Bloods episode titles. Keep episode numbering sequential.",
        "Clean these filenames and rename them sequentially as Blue Bloods Episode 01, 02 and 03. Keep the .mp4 extension.",
        "delete all files",
        "delete the duplicate files",
        "remove all duplicate files",
        "remove everything",
        "Season 2-এর সব ভিডিও আলাদা folder-এ রাখো",
        "সব ভিডিওর নাম clean করে দাও। Blue Bloods নামটা শুরুতে রাখবে এবং episode number 01 থেকে sequential করবে। 1080p এবং WEB-DL বাদ দাও।",
        "Add numbering with episode titles",
        "Move all S02 files into a Season 2 folder",
        "What is the weather like?",
        "replace all files with videos",
        "add prefix",
        "line one\nline two: remove 1080p from all filenames",
    ],
)
def test_ambiguous_commands_are_left_to_the_ai(command):
    assert parse_local_command(command) is None
