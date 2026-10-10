"""Windows file-name validation: illegal characters, reserved names, Unicode, length."""

from __future__ import annotations

import pytest

from app.utils.validators import (
    find_illegal_chars,
    fit_stem,
    is_reserved_name,
    sanitize_component,
    utf16_length,
    validate_component,
)


@pytest.mark.parametrize(
    "name",
    [
        "Blue Bloods - Episode 01.mp4",
        "বাংলা ভিডিও ০১.mkv",  # Bangla
        "Café ñandú 日本語.mp4",  # accents + CJK
        "Movie (2020) [1080p] {tag}.mkv",  # brackets / parentheses
        "a b  c.mp4",  # spaces
        "it's #1 & 100% done!.mp4",
        "épisode_01-final.v2.mp4",
        ".hidden",
    ],
)
def test_valid_names(name):
    assert validate_component(name) == []


@pytest.mark.parametrize("char", list('<>:"/\\|?*'))
def test_illegal_characters_are_rejected(char):
    problems = validate_component(f"bad{char}name.mp4")
    assert problems and "Illegal" in problems[0]
    assert find_illegal_chars(f"bad{char}name.mp4") == [char]


def test_control_characters_rejected():
    assert validate_component("bad\x00name.mp4")
    assert validate_component("tab\tname.mp4")


@pytest.mark.parametrize("name", ["CON", "con.txt", "NUL.mp4", "COM1", "lpt9.log", "AUX.tar.gz"])
def test_reserved_names(name):
    assert is_reserved_name(name)
    assert any("Reserved" in p for p in validate_component(name))


def test_not_reserved():
    assert not is_reserved_name("CONSOLE.mp4")
    assert not is_reserved_name("COM10.mp4")


@pytest.mark.parametrize("name", ["name.", "name ", " name", "", "   ", ".", ".."])
def test_trailing_leading_and_empty(name):
    assert validate_component(name)


def test_invisible_characters_rejected():
    assert validate_component("evil\u202egnp.mp4")
    assert validate_component("zero\u200bwidth.mp4")


def test_length_limit_counts_utf16_units():
    assert validate_component("a" * 255) == []
    assert validate_component("a" * 256)
    emoji = "😀" * 128  # 2 UTF-16 units each => 256
    assert utf16_length(emoji) == 256
    assert validate_component(emoji)


def test_sanitize_produces_valid_name():
    cleaned = sanitize_component('Ep 1: "The Start"?.mp4')
    assert validate_component(cleaned) == []
    assert sanitize_component("CON") != "CON"
    assert validate_component(sanitize_component("???")) == []


def test_fit_stem_truncates_long_names_and_keeps_extension_room():
    result = fit_stem("x" * 400, ".mp4")
    assert result.truncated and result.possible
    assert utf16_length(result.stem + ".mp4") <= 255


def test_fit_stem_short_name_untouched():
    result = fit_stem("short", ".mp4")
    assert result.stem == "short" and not result.truncated


def test_fit_stem_path_limit():
    result = fit_stem("y" * 300, ".mkv", directory_len=200, enforce_path_limit=True)
    assert result.possible and len(result.stem) + 4 <= 259 - 200
    impossible = fit_stem("y", ".mkv", directory_len=300, enforce_path_limit=True)
    assert not impossible.possible
