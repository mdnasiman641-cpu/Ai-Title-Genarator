"""Pure name transformations."""

from __future__ import annotations

import pytest

from app.files import text_transforms as tt


@pytest.mark.parametrize(
    ("stem", "texts", "expected"),
    [
        ("VID_001_FINAL_1080P_WEB-DL", ("1080p", "WEB-DL"), "VID_001_FINAL"),  # case-insensitive, leftovers tidied
        ("Movie [1080p] 2020", ("1080p",), "Movie 2020"),  # empty brackets vanish
        ("Movie (1080p)", ("1080p",), "Movie"),
        ("Show - S01E01 - 720p", ("720p",), "Show - S01E01"),
        ("a.b.1080p.c", ("1080p",), "a.b.c"),
        ("keep", ("zzz",), "keep"),
        ("বাংলা ভিডিও 1080p", ("1080p",), "বাংলা ভিডিও"),
        ("x264 and X264", ("x264",), "and"),
    ],
)
def test_remove_texts(stem, texts, expected):
    assert tt.remove_texts(stem, texts) == expected


def test_remove_case_sensitive():
    assert tt.remove_texts("A a A", ("a",), case_sensitive=True) == "A A"


def test_remove_is_literal_not_regex():
    assert tt.remove_texts("file (1) v2", ("(1)",)) == "file v2"
    assert tt.remove_texts("a.b", (".",)) == "ab"  # '.' is a literal dot, not "any character"
    assert tt.remove_texts("a+b", ("+",)) == "ab"
    assert tt.remove_texts("abc", (".*",)) == "abc"


def test_replace_text():
    assert tt.replace_text("Show.WEB-DL.x264", "WEB-DL", "WEB") == "Show.WEB.x264"
    assert tt.replace_text("a_b_c", "_", " ") == "a b c"
    assert tt.replace_text("Hello", "hello", "Bye") == "Bye"
    assert tt.replace_text("Hello", "hello", "Bye", case_sensitive=True) == "Hello"
    assert tt.replace_text("a", "", "x") == "a"
    assert tt.replace_text("price $5", "$5", "\\1") == "price \\1"  # replacement is literal


def test_tidy():
    assert tt.tidy("a  b") == "a b"
    assert tt.tidy("a__b") == "a_b"
    assert tt.tidy("--a--") == "a"
    assert tt.tidy("a [] b ()") == "a b"
    assert tt.tidy("a [ ( ) ] b") == "a b"
    assert tt.tidy("a - b") == "a - b"  # intentional separators are kept


@pytest.mark.parametrize(
    ("mode", "expected"),
    [
        ("lower", "blue bloods - episode 01"),
        ("upper", "BLUE BLOODS - EPISODE 01"),
        ("title", "Blue Bloods - Episode 01"),
        ("sentence", "Blue bloods - episode 01"),
    ],
)
def test_change_case(mode, expected):
    assert tt.change_case("bLUE bloods - EPISODE 01", mode) == expected


def test_title_case_handles_apostrophes_and_unicode():
    assert tt.change_case("don't stop", "title") == "Don't Stop"
    assert tt.change_case("été CAFÉ", "title") == "Été Café"
    assert tt.change_case("বাংলা ভিডিও", "title") == "বাংলা ভিডিও"  # no case in Bangla


def test_prefix_suffix_separators():
    assert tt.add_prefix("VID", "Blue Bloods - ") == "Blue Bloods - VID"
    assert tt.add_prefix("VID", "Blue Bloods -") == "Blue Bloods - VID"
    assert tt.add_prefix("VID", "S02_") == "S02_VID"
    assert tt.add_prefix("VID", "Blue Bloods") == "Blue Bloods VID"
    assert tt.add_prefix("VID", "[Group]") == "[Group] VID"
    assert tt.add_prefix("VID", "X", separator="") == "XVID"
    assert tt.add_suffix("VID", "_HD") == "VID_HD"
    assert tt.add_suffix("VID", "HD") == "VID HD"
    assert tt.add_suffix("VID", "- HD") == "VID - HD"
    assert tt.add_suffix("VID", "(2020)") == "VID (2020)"
    assert tt.add_suffix("VID", " HD") == "VID HD"


def test_numbers_and_templates():
    assert tt.format_number(7, 2) == "07" and tt.format_number(123, 2) == "123"
    assert tt.apply_template("Ep {n}", "05") == "Ep 05"
    assert tt.apply_template("{n}{n}", "1") == "11"
    assert tt.apply_template("{0.__class__} {n}", "1") == "{0.__class__} 1"  # never str.format


def test_normalise_stem():
    assert tt.normalise_stem("  a   b  ") == "a   b"  # trims the edges, leaves inner spacing alone
