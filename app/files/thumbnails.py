"""Lazy, cached video thumbnails (one ffmpeg call per *viewed* file, never in bulk)."""

from __future__ import annotations

import hashlib
import os
import subprocess
from pathlib import Path

from app.files import metadata
from app.files.metadata import subprocess_flags
from app.utils.helpers import get_app_data_dir
from app.utils.logger import get_logger

log = get_logger("thumbs")
THUMB_WIDTH = 320
FFMPEG_TIMEOUT_S = 25
MAX_CACHE_FILES = 1500


class ThumbnailCache:
    """JPEG thumbnails stored in the application data folder, keyed by path + size + mtime."""

    def __init__(self, ffmpeg_dir: str = "", cache_dir: Path | None = None) -> None:
        self.binary = metadata.find_binary("ffmpeg", ffmpeg_dir)
        self.dir = cache_dir or (get_app_data_dir() / "thumbnails")

    @property
    def available(self) -> bool:
        return self.binary is not None

    def path_for(self, source: Path) -> Path:
        stat = source.stat()
        digest = hashlib.sha1(f"{source.resolve()}|{stat.st_size}|{stat.st_mtime_ns}".encode()).hexdigest()
        return self.dir / f"{digest}.jpg"

    def cached(self, source: Path) -> Path | None:
        """The cached thumbnail if one exists already (no ffmpeg involved)."""
        try:
            path = self.path_for(source)
        except OSError:
            return None
        return path if path.is_file() and path.stat().st_size > 0 else None

    def get(self, source: Path, duration: float | None = None) -> Path | None:
        """Return the thumbnail for ``source``, generating it on first use."""
        hit = self.cached(source)
        if hit is not None:
            os.utime(hit)  # LRU-ish: recently viewed thumbnails survive pruning
            return hit
        if self.binary is None:
            return None
        try:
            target = self.path_for(source)
        except OSError:
            return None
        self.dir.mkdir(parents=True, exist_ok=True)
        seek = 1.0 if not duration or duration < 4 else min(duration * 0.1, 60.0)
        command = [
            str(self.binary), "-v", "error", "-y", "-ss", f"{seek:.2f}", "-i", f"file:{source.resolve()}",
            "-frames:v", "1", "-vf", f"scale={THUMB_WIDTH}:-2", "-q:v", "4", str(target),
        ]  # fmt: skip
        try:
            done = subprocess.run(  # noqa: S603 - fixed argv list, no shell
                command, capture_output=True, timeout=FFMPEG_TIMEOUT_S, creationflags=subprocess_flags(), check=False
            )  # fmt: skip
        except (OSError, subprocess.SubprocessError) as exc:
            log.warning("thumbnail failed for %s: %s", source.name, exc)
            return None
        if done.returncode != 0 or not target.is_file() or target.stat().st_size == 0:
            target.unlink(missing_ok=True)
            log.debug("no thumbnail for %s", source.name)
            return None
        self._prune()
        return target

    def _prune(self) -> None:
        """Keep the cache bounded: drop the least recently used files beyond the limit."""
        try:
            files = sorted(self.dir.glob("*.jpg"), key=lambda p: p.stat().st_mtime)
        except OSError:
            return
        for old in files[: max(0, len(files) - MAX_CACHE_FILES)]:
            old.unlink(missing_ok=True)
