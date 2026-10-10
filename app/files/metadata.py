"""Optional video metadata through ``ffprobe``.

ffprobe only reads the container headers - it never decodes the video - so a
probe takes a few milliseconds per file. Results are cached in SQLite (keyed by
path, size and modification time), and probing is requested lazily for the rows
the user can actually see. If ffprobe is not installed the application simply
works without durations.
"""

from __future__ import annotations

import json
import shutil
import subprocess
from dataclasses import asdict, dataclass
from pathlib import Path

from app.config.constants import IS_WINDOWS
from app.utils.helpers import bundled_binary
from app.utils.logger import get_logger

log = get_logger("metadata")
PROBE_TIMEOUT_S = 20
_CREATE_NO_WINDOW = 0x08000000  # keep ffprobe/ffmpeg from flashing a console window on Windows


def subprocess_flags() -> int:
    """``creationflags`` for child processes (hidden console on Windows)."""
    return _CREATE_NO_WINDOW if IS_WINDOWS else 0


def find_binary(name: str, extra_dir: str = "") -> Path | None:
    """Locate ``ffprobe``/``ffmpeg``: user-configured folder, bundled copy, then PATH."""
    exe = f"{name}.exe" if IS_WINDOWS else name
    if extra_dir:
        candidate = Path(extra_dir) / exe
        if candidate.is_file():
            return candidate
    bundled = bundled_binary(exe)
    if bundled is not None:
        return bundled
    found = shutil.which(name)
    return Path(found) if found else None


@dataclass(slots=True)
class VideoMetadata:
    """The few facts shown in the UI (and optionally sent to the AI)."""

    duration: float | None = None
    width: int | None = None
    height: int | None = None
    fps: float | None = None
    video_codec: str = ""
    audio_codec: str = ""
    bitrate: int | None = None

    @property
    def resolution(self) -> str:
        return f"{self.width}×{self.height}" if self.width and self.height else ""

    def to_dict(self) -> dict[str, object]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, object]) -> VideoMetadata:
        known = {k: data.get(k) for k in cls.__dataclass_fields__ if k in data}
        return cls(**known)  # type: ignore[arg-type]


def _parse_fps(value: str | None) -> float | None:
    if not value or "/" not in value:
        return None
    num, _, den = value.partition("/")
    try:
        fraction = float(num) / float(den)
    except (ValueError, ZeroDivisionError):
        return None
    return round(fraction, 3) if fraction > 0 else None


def parse_probe_output(raw: str) -> VideoMetadata:
    """Convert ffprobe's JSON (``-show_format -show_streams``) into :class:`VideoMetadata`."""
    data = json.loads(raw)
    meta = VideoMetadata()
    fmt = data.get("format") or {}
    try:
        meta.duration = float(fmt["duration"]) if fmt.get("duration") not in (None, "N/A") else None
    except (TypeError, ValueError):
        meta.duration = None
    try:
        meta.bitrate = int(fmt["bit_rate"]) if fmt.get("bit_rate") not in (None, "N/A") else None
    except (TypeError, ValueError):
        meta.bitrate = None
    for stream in data.get("streams") or []:
        kind = stream.get("codec_type")
        if kind == "video" and not meta.video_codec and stream.get("disposition", {}).get("attached_pic") != 1:
            meta.video_codec = str(stream.get("codec_name", ""))
            meta.width = int(stream["width"]) if stream.get("width") else None
            meta.height = int(stream["height"]) if stream.get("height") else None
            meta.fps = _parse_fps(stream.get("avg_frame_rate")) or _parse_fps(stream.get("r_frame_rate"))
            if meta.duration is None and stream.get("duration") not in (None, "N/A"):
                try:
                    meta.duration = float(stream["duration"])
                except (TypeError, ValueError):
                    pass
        elif kind == "audio" and not meta.audio_codec:
            meta.audio_codec = str(stream.get("codec_name", ""))
    return meta


class MetadataProbe:
    """Runs ffprobe for one file at a time (no shell, absolute paths only)."""

    def __init__(self, ffmpeg_dir: str = "") -> None:
        self.binary = find_binary("ffprobe", ffmpeg_dir)

    @property
    def available(self) -> bool:
        return self.binary is not None

    def probe(self, path: Path) -> VideoMetadata | None:
        """Return metadata for ``path`` or ``None`` if ffprobe is missing / the file is unreadable."""
        if self.binary is None:
            return None
        absolute = str(path.resolve())  # absolute => can never be mistaken for an option
        command = [
            str(self.binary), "-v", "error", "-print_format", "json", "-show_format", "-show_streams", f"file:{absolute}",
        ]  # fmt: skip
        try:
            done = subprocess.run(  # noqa: S603 - fixed argv list, no shell
                command, capture_output=True, text=True, timeout=PROBE_TIMEOUT_S, creationflags=subprocess_flags(),
                encoding="utf-8", errors="replace", check=False,
            )  # fmt: skip
        except (OSError, subprocess.SubprocessError) as exc:
            log.warning("ffprobe failed for %s: %s", path.name, exc)
            return None
        if done.returncode != 0 or not done.stdout.strip():
            log.debug("ffprobe could not read %s (code %s)", path.name, done.returncode)
            return None
        try:
            return parse_probe_output(done.stdout)
        except (ValueError, KeyError, TypeError) as exc:
            log.warning("unexpected ffprobe output for %s: %s", path.name, exc)
            return None
