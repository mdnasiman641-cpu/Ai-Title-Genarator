"""ffprobe metadata, thumbnails and lazy loading (uses the real ffmpeg when installed)."""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import time
from pathlib import Path

import pytest

from app.files import metadata as md
from app.files.metadata import MetadataProbe, VideoMetadata, parse_probe_output
from app.files.scanner import scan_folder
from app.files.thumbnails import MAX_CACHE_FILES, ThumbnailCache
from tests.helpers import wait_until

HAVE_FFMPEG = bool(shutil.which("ffmpeg") and shutil.which("ffprobe"))
needs_ffmpeg = pytest.mark.skipif(not HAVE_FFMPEG, reason="ffmpeg/ffprobe not installed")


def make_video(path: Path, seconds: int = 3) -> Path:
    for codec in ("mpeg4", "libx264", "mjpeg"):
        done = subprocess.run(
            ["ffmpeg", "-v", "error", "-y", "-f", "lavfi", "-i", f"testsrc=duration={seconds}:size=320x240:rate=10",
             "-f", "lavfi", "-i", f"sine=frequency=440:duration={seconds}", "-c:v", codec, "-c:a", "aac", "-shortest", str(path)],
            capture_output=True,
        )  # fmt: skip
        if done.returncode == 0 and path.exists():
            return path
    pytest.skip("this ffmpeg build cannot encode a test video")


SAMPLE_JSON = json.dumps({
    "streams": [
        {"codec_type": "video", "codec_name": "h264", "width": 1920, "height": 1080, "avg_frame_rate": "30000/1001", "disposition": {"attached_pic": 0}},
        {"codec_type": "audio", "codec_name": "aac"},
    ],
    "format": {"duration": "2535.250000", "bit_rate": "4500000"},
})


# --------------------------------------------------------------- parsing
def test_parse_typical_output():
    meta = parse_probe_output(SAMPLE_JSON)
    assert meta.duration == pytest.approx(2535.25) and meta.resolution == "1920×1080"
    assert meta.fps == pytest.approx(29.97, abs=0.001) and meta.video_codec == "h264" and meta.audio_codec == "aac"
    assert meta.bitrate == 4_500_000


def test_parse_handles_missing_and_na_values():
    meta = parse_probe_output(json.dumps({"streams": [{"codec_type": "video", "codec_name": "vp9", "avg_frame_rate": "0/0"}], "format": {"duration": "N/A"}}))
    assert meta.duration is None and meta.fps is None and meta.resolution == "" and meta.video_codec == "vp9"
    assert parse_probe_output("{}") == VideoMetadata()


def test_parse_uses_stream_duration_and_skips_cover_art():
    raw = json.dumps({"streams": [
        {"codec_type": "video", "codec_name": "mjpeg", "disposition": {"attached_pic": 1}},
        {"codec_type": "video", "codec_name": "av1", "width": 640, "height": 360, "duration": "12.5", "r_frame_rate": "25/1"},
    ], "format": {}})
    meta = parse_probe_output(raw)
    assert meta.video_codec == "av1" and meta.duration == 12.5 and meta.fps == 25.0


def test_parse_rejects_garbage():
    with pytest.raises(ValueError):
        parse_probe_output("not json")


def test_metadata_dict_roundtrip():
    meta = parse_probe_output(SAMPLE_JSON)
    assert VideoMetadata.from_dict(json.loads(json.dumps(meta.to_dict()))) == meta
    assert VideoMetadata.from_dict({"unknown_field": 1, "duration": 5}).duration == 5


def test_probe_is_unavailable_without_binary(monkeypatch):
    monkeypatch.setattr(md, "find_binary", lambda name, extra_dir="": None)
    probe = MetadataProbe()
    assert not probe.available and probe.probe(Path("x.mp4")) is None
    assert not ThumbnailCache().available


def test_find_binary_prefers_configured_folder(tmp_path):
    fake = tmp_path / ("ffprobe.exe" if os.name == "nt" else "ffprobe")  # Windows looks for the .exe
    fake.write_text("#!/bin/sh\n")
    assert md.find_binary("ffprobe", str(tmp_path)) == fake


# ----------------------------------------------------------- real ffprobe
@needs_ffmpeg
def test_real_probe_reads_duration_resolution_fps_codec(workspace):
    video = make_video(workspace / "clip.mp4")
    meta = MetadataProbe().probe(video)
    assert meta is not None and meta.duration == pytest.approx(3.0, abs=0.5)
    assert meta.resolution == "320×240" and meta.fps == pytest.approx(10, abs=0.1) and meta.video_codec and meta.audio_codec


@needs_ffmpeg
def test_probe_non_video_and_missing_files_return_none(workspace):
    junk = workspace / "notes.mp4"
    junk.write_text("this is not a video")
    assert MetadataProbe().probe(junk) is None
    assert MetadataProbe().probe(workspace / "missing.mp4") is None


@needs_ffmpeg
def test_filename_starting_with_dash_is_not_treated_as_an_option(workspace):
    video = make_video(workspace / "-version.mp4")
    meta = MetadataProbe().probe(video)
    assert meta is not None and meta.duration


@needs_ffmpeg
def test_unicode_and_bangla_filenames_probe_fine(workspace):
    video = make_video(workspace / "ভিডিও ০১ (2020) [HD].mp4")
    assert MetadataProbe().probe(video).resolution == "320×240"


# ------------------------------------------------------------- thumbnails
@needs_ffmpeg
def test_thumbnail_generation_cache_and_reuse(workspace, tmp_path, monkeypatch):
    video = make_video(workspace / "clip.mp4")
    cache = ThumbnailCache(cache_dir=tmp_path / "thumbs")
    assert cache.cached(video) is None  # nothing is generated until asked
    thumb = cache.get(video, duration=3)
    assert thumb is not None and thumb.suffix == ".jpg" and thumb.stat().st_size > 500
    from PySide6.QtGui import QImage

    image = QImage(str(thumb))
    assert not image.isNull() and image.width() == 320
    monkeypatch.setattr(cache, "binary", None)  # a second request must be served from the cache, not ffmpeg
    assert cache.get(video, duration=3) == thumb and cache.cached(video) == thumb


@needs_ffmpeg
def test_thumbnail_cache_key_changes_when_file_changes(workspace, tmp_path):
    video = make_video(workspace / "clip.mp4")
    cache = ThumbnailCache(cache_dir=tmp_path / "t")
    first = cache.get(video)
    time.sleep(0.01)
    make_video(video, seconds=2)  # rewritten file -> different size/mtime -> new thumbnail
    second = cache.get(video)
    assert first is not None and second is not None and first != second


@needs_ffmpeg
def test_thumbnail_of_non_video_is_none_and_leaves_no_junk(workspace, tmp_path):
    junk = workspace / "x.mp4"
    junk.write_text("not a video")
    cache = ThumbnailCache(cache_dir=tmp_path / "t")
    assert cache.get(junk) is None
    assert not list((tmp_path / "t").glob("*.jpg"))


def test_thumbnail_cache_is_bounded(tmp_path):
    cache = ThumbnailCache(cache_dir=tmp_path / "t")
    cache.dir.mkdir()
    for i in range(MAX_CACHE_FILES + 40):
        (cache.dir / f"{i:05d}.jpg").write_bytes(b"x")
    cache._prune()
    assert len(list(cache.dir.glob("*.jpg"))) == MAX_CACHE_FILES


# ----------------------------------------------------------- worker + cache
class CountingProbe:
    available = True

    def __init__(self):
        self.calls = []

    def probe(self, path):
        self.calls.append(path.name)
        return VideoMetadata(duration=42.0, width=1280, height=720)


def test_metadata_worker_uses_sqlite_cache(workspace, make_files, tmp_path):
    from app.database.database import Database
    from app.workers.metadata_worker import MetadataWorker

    make_files("a.mp4", "b.mp4")
    db = Database(tmp_path / "m.db")
    entries = scan_folder(workspace).entries
    probe = CountingProbe()
    seen = []
    worker = MetadataWorker(workspace, entries, probe, db)
    worker.signals.item.connect(lambda rel, meta: seen.append((rel, meta)))
    assert worker.execute() == 2 and probe.calls == ["a.mp4", "b.mp4"]
    again = MetadataWorker(workspace, entries, probe, db)
    again.execute()
    assert probe.calls == ["a.mp4", "b.mp4"]  # second run: all served from the cache
    (workspace / "a.mp4").write_text("changed content with different size")
    changed = MetadataWorker(workspace, scan_folder(workspace).entries, probe, db)
    changed.execute()
    assert probe.calls == ["a.mp4", "b.mp4", "a.mp4"]  # only the modified file is probed again


def test_failed_probe_is_cached_too(workspace, make_files, tmp_path):
    from app.database.database import Database
    from app.workers.metadata_worker import MetadataWorker

    make_files("broken.mp4")
    db = Database(tmp_path / "m.db")

    class FailingProbe(CountingProbe):
        def probe(self, path):
            self.calls.append(path.name)
            return None

    probe = FailingProbe()
    entries = scan_folder(workspace).entries
    MetadataWorker(workspace, entries, probe, db).execute()
    MetadataWorker(workspace, entries, probe, db).execute()
    assert probe.calls == ["broken.mp4"]  # not retried on every scan


# -------------------------------------------------------------------- GUI
@pytest.mark.gui
@needs_ffmpeg
def test_gui_lazy_durations_details_and_thumbnail(qapp, tmp_path, workspace, dispose):
    from app.context import AppContext
    from app.ui.main_window import MainWindow
    from app.ui.theme import apply_theme

    make_video(workspace / "clip.mp4")
    (workspace / "notes.txt").write_text("hi")
    apply_theme(qapp, "dark")
    win = MainWindow(AppContext.create(tmp_path / "ctx"))
    win.files_panel.set_filter("all")
    win.show()
    win.load_folder(workspace)
    assert wait_until(lambda: win.files_panel.model.rowCount() == 2)
    video_row = next(r for r, e in enumerate(win.files_panel.model.entries) if e.name == "clip.mp4")
    assert wait_until(lambda: win.files_panel.model.entries[video_row].duration not in (None,), timeout=8)  # lazily filled
    index = win.files_panel.proxy.mapFromSource(win.files_panel.model.index(video_row, 4))
    assert wait_until(lambda: win.files_panel.proxy.data(index).startswith("0:0"), timeout=5)  # 0:03
    assert win.files_panel.model.entry_for_path("notes.txt").duration is None  # non-media files are never probed

    win.files_panel.table.selectRow(win.files_panel.proxy.mapFromSource(win.files_panel.model.index(video_row, 0)).row())
    details = win.files_panel.details
    assert wait_until(lambda: details._values["resolution"].text() == "320×240", timeout=5)
    assert wait_until(lambda: details.thumb.pixmap() is not None and not details.thumb.pixmap().isNull(), timeout=10)
    assert details.hint.text() == ""  # tools present: no "install ffmpeg" nag
    wait_until(lambda: not win._workers, timeout=5)
    dispose(win)


@pytest.mark.gui
def test_gui_works_without_ffmpeg(qapp, tmp_path, workspace, make_files, dispose, monkeypatch):
    from app.context import AppContext
    from app.ui.main_window import MainWindow

    monkeypatch.setattr(md, "find_binary", lambda name, extra_dir="": None)
    make_files("a.mp4")
    win = MainWindow(AppContext.create(tmp_path / "ctx"))
    win.show()
    win.load_folder(workspace)
    assert wait_until(lambda: win.files_panel.model.rowCount() == 1)
    win.files_panel.table.selectRow(0)
    assert "ffmpeg" in win.files_panel.details.hint.text()
    assert win.files_panel.details.name.text() == "a.mp4"
    assert win.files_panel.model.entries[0].duration is None  # nothing probed, nothing crashed
    dispose(win)
