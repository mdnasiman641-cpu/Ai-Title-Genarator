"""Workspace path safety: traversal, absolute/UNC paths, links, illegal names."""

from __future__ import annotations

import pytest

from app.files.path_validator import PathError, WorkspacePaths


@pytest.fixture
def paths(workspace):
    return WorkspacePaths(workspace)


@pytest.mark.parametrize(
    "bad",
    [
        "../secret.mp4",
        "a/../../b",
        "..",
        "a/..",
        "..\\evil",
        "/etc/passwd",
        "\\Windows\\System32",
        "C:\\Windows\\x.dll",
        "c:evil",
        "D:/data",
        "\\\\server\\share\\x.mp4",
        "//server/share/x.mp4",
        "a/b:c",
        "a/b*c",
        'a/"quote"',
        "a/CON",
        "a/nul.txt",
        "a/trailing.",
        "a /b",  # a component ending in a space
        "",
        "   ",
        "./",
        "a\x00b",
        ".ai_video_assistant_trash/1/x.mp4",
        ".tmp_ai_deadbeef_001",
        "/".join(["d"] * 40),
    ],
)
def test_dangerous_paths_are_rejected(paths, bad):
    with pytest.raises(PathError):
        paths.normalize_relative(bad)


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ("Season 2", "Season 2"),
        ("Season 2/", "Season 2"),
        ("a\\b\\c", "a/b/c"),
        ("./a//b/./c", "a/b/c"),
        ("  Season 2  ", "Season 2"),
        ("বাংলা/ফোল্ডার", "বাংলা/ফোল্ডার"),
        ("Movies (2020) [HD]/Ep 1", "Movies (2020) [HD]/Ep 1"),
    ],
)
def test_valid_relative_paths_are_normalised(paths, raw, expected):
    assert paths.normalize_relative(raw) == expected


def test_empty_allowed_when_requested(paths):
    assert paths.normalize_relative("", allow_empty=True) == ""
    with pytest.raises(PathError):
        paths.normalize_relative(123)  # type: ignore[arg-type]


def test_absolute_resolution_stays_inside(paths, workspace):
    assert paths.ensure_contained("a/b.mp4") == workspace / "a" / "b.mp4"


def test_symlink_escape_is_detected(paths, workspace, tmp_path):
    outside = tmp_path / "outside"
    outside.mkdir()
    try:
        (workspace / "link").symlink_to(outside, target_is_directory=True)
    except OSError:
        pytest.skip("symlinks unavailable")
    with pytest.raises(PathError):
        paths.ensure_contained("link/new.mp4")
    (workspace / "real").mkdir()
    assert paths.ensure_contained("real/new.mp4")  # ordinary folders are fine


def test_symlink_pointing_inside_is_allowed(paths, workspace):
    (workspace / "real").mkdir()
    try:
        (workspace / "alias").symlink_to(workspace / "real", target_is_directory=True)
    except OSError:
        pytest.skip("symlinks unavailable")
    assert paths.ensure_contained("alias/x.mp4")


def test_comparison_keys_respect_case_sensitivity(workspace):
    ci = WorkspacePaths(workspace, case_insensitive=True)
    cs = WorkspacePaths(workspace, case_insensitive=False)
    assert ci.key("A.MP4") == ci.key("a.mp4") and cs.key("A.MP4") != cs.key("a.mp4")
    assert ci.key("e\u0301.mp4") == ci.key("\u00e9.mp4")  # NFC/NFD forms are the same name


def test_split_and_join():
    assert WorkspacePaths.split("a/b/c.mp4") == ("a/b", "c.mp4")
    assert WorkspacePaths.split("c.mp4") == ("", "c.mp4")
    assert WorkspacePaths.join("", "c.mp4") == "c.mp4" and WorkspacePaths.join("a", "c.mp4") == "a/c.mp4"
