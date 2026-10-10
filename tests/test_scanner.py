"""Folder scanning: non-recursive default, recursion opt-in, hidden/internal skipping."""

from __future__ import annotations

import threading

import pytest

from app.config.constants import TRASH_DIR_NAME, FileKind
from app.files.scanner import scan_folder


def test_scans_only_selected_folder_by_default(workspace, make_files):
    make_files("a.mp4", "b.mkv", "sub/c.mp4")
    result = scan_folder(workspace)
    assert [e.name for e in result.entries] == ["a.mp4", "b.mkv"]
    assert not result.recursive


def test_recursive_scan_includes_subfolders_with_relative_paths(workspace, make_files):
    make_files("a.mp4", "sub/c.mp4", "sub/deeper/d.mkv")
    result = scan_folder(workspace, recursive=True)
    assert {e.rel_path for e in result.entries} == {"a.mp4", "sub/c.mp4", "sub/deeper/d.mkv"}
    deep = next(e for e in result.entries if e.name == "d.mkv")
    assert deep.directory == "sub/deeper" and deep.stem == "d"


def test_kinds_and_extensions(workspace, make_files):
    make_files("v.MP4", "s.mp3", "p.jpg", "d.pdf", "x.bin", "noext")
    kinds = {e.name: e.kind for e in scan_folder(workspace).entries}
    assert kinds["v.MP4"] == FileKind.VIDEO
    assert kinds["s.mp3"] == FileKind.AUDIO
    assert kinds["p.jpg"] == FileKind.IMAGE
    assert kinds["d.pdf"] == FileKind.DOCUMENT
    assert kinds["x.bin"] == FileKind.OTHER
    assert kinds["noext"] == FileKind.OTHER


def test_all_supported_video_extensions(workspace, make_files):
    exts = [".mp4", ".mkv", ".avi", ".mov", ".wmv", ".flv", ".webm", ".m4v", ".ts"]
    make_files(*[f"v{i}{ext}" for i, ext in enumerate(exts)])
    assert all(e.kind == FileKind.VIDEO for e in scan_folder(workspace).entries)


def test_natural_ordering(workspace, make_files):
    make_files("ep10.mp4", "ep2.mp4", "ep1.mp4")
    assert [e.name for e in scan_folder(workspace).entries] == ["ep1.mp4", "ep2.mp4", "ep10.mp4"]


def test_internal_trash_folder_is_skipped(workspace, make_files):
    make_files("a.mp4", f"{TRASH_DIR_NAME}/1/old.mp4")
    assert [e.name for e in scan_folder(workspace, recursive=True).entries] == ["a.mp4"]


def test_unicode_names(workspace, make_files):
    make_files("বাংলা ভিডিও ০১.mkv", "日本語.mp4")
    names = {e.name for e in scan_folder(workspace).entries}
    assert names == {"বাংলা ভিডিও ০১.mkv", "日本語.mp4"}


def test_missing_folder_raises(tmp_path):
    with pytest.raises(FileNotFoundError):
        scan_folder(tmp_path / "nope")


def test_cancel_returns_partial_flagged_result(workspace, make_files):
    make_files("a.mp4")
    event = threading.Event()
    event.set()
    assert scan_folder(workspace, cancel=event).cancelled


def test_symlinks_are_not_followed(workspace, make_files, tmp_path):
    outside = tmp_path / "outside"
    outside.mkdir()
    (outside / "secret.mp4").write_text("s")
    try:
        (workspace / "link").symlink_to(outside, target_is_directory=True)
    except OSError:
        pytest.skip("symlinks unavailable")
    make_files("a.mp4")
    assert [e.name for e in scan_folder(workspace, recursive=True).entries] == ["a.mp4"]


def test_scales_to_thousands_of_files(workspace):
    for i in range(3000):
        (workspace / f"clip_{i:05d}.mp4").write_bytes(b"")
    result = scan_folder(workspace)
    assert len(result.entries) == 3000
