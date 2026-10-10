"""Extract the first N seconds of a video's audio for transcription (the video is never modified).

ffmpeg reads only the beginning of the file (``-t``), writes a small mono 16 kHz WAV into
a private temporary folder, and the caller deletes it as soon as it has been transcribed
(:func:`extracted_audio` is a context manager that always cleans up).
"""

from __future__ import annotations

import shutil
import subprocess
import tempfile
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path

from app.files import metadata
from app.files.metadata import subprocess_flags
from app.utils.logger import get_logger

log = get_logger("audio")
SAMPLE_RATE = 16_000  # what speech recognisers expect; ~1.9 MB per minute as 16-bit mono WAV
MAX_SECONDS = 600  # 10 minutes ≈ 19 MB - well under the usual 25 MB upload limit of transcription APIs
EXTRACT_TIMEOUT_S = 180
_NO_AUDIO_HINTS = ("does not contain any stream", "matches no streams", "output file is empty", "no audio")


class AudioError(Exception):
    """Audio could not be extracted. ``kind``: no_ffmpeg, no_audio, unreadable, timeout."""

    def __init__(self, kind: str, message: str) -> None:
        super().__init__(message)
        self.kind = kind
        self.message = message


@dataclass(frozen=True, slots=True)
class AudioClip:
    path: Path
    seconds: float  # length actually extracted (shorter videos give shorter clips)


class AudioExtractor:
    """Wraps the ffmpeg binary found the same way as for thumbnails (settings folder, bundled copy, PATH)."""

    def __init__(self, ffmpeg_dir: str = "") -> None:
        self.binary = metadata.find_binary("ffmpeg", ffmpeg_dir)

    @property
    def available(self) -> bool:
        return self.binary is not None

    @contextmanager
    def extracted_audio(self, source: Path, seconds: float) -> Iterator[AudioClip]:
        """Yield a temporary WAV of the first ``seconds`` of ``source``; it is deleted afterwards."""
        if self.binary is None:
            raise AudioError("no_ffmpeg", "FFmpeg not found. Install ffmpeg or set its folder in Settings → General → ffmpeg folder.")
        seconds = max(1.0, min(float(seconds), MAX_SECONDS))
        workdir = Path(tempfile.mkdtemp(prefix="aivfa-audio-"))
        target = workdir / "clip.wav"
        try:
            command = [
                str(self.binary), "-v", "error", "-nostdin", "-y", "-ss", "0", "-t", f"{seconds:.2f}",
                "-i", f"file:{source.resolve()}", "-map", "0:a:0?", "-vn", "-sn", "-dn",
                "-ac", "1", "-ar", str(SAMPLE_RATE), "-c:a", "pcm_s16le", "-f", "wav", str(target),
            ]  # fmt: skip
            try:
                done = subprocess.run(  # noqa: S603 - fixed argv list, no shell
                    command, capture_output=True, timeout=EXTRACT_TIMEOUT_S, creationflags=subprocess_flags(), check=False
                )  # fmt: skip
            except subprocess.TimeoutExpired:
                raise AudioError("timeout", f"Audio extraction failed: it took longer than {EXTRACT_TIMEOUT_S}s.") from None
            except OSError as exc:
                raise AudioError("no_ffmpeg", f"FFmpeg not found or could not be started: {exc}") from None
            stderr = done.stderr.decode("utf-8", "replace").strip()
            size = target.stat().st_size if target.is_file() else 0
            if done.returncode != 0 or size <= 44:  # 44 bytes = a WAV header without samples
                lowered = stderr.lower()
                if size <= 44 and (done.returncode == 0 or any(h in lowered for h in _NO_AUDIO_HINTS)):
                    raise AudioError("no_audio", "Video contains no audio stream.")
                log.info("audio extraction failed for %s: %s", source.name, stderr[-300:])
                reason = stderr.splitlines()[-1][:200] if stderr else f"ffmpeg exit code {done.returncode}"
                raise AudioError("unreadable", f"Audio extraction failed: the video could not be read (damaged or unsupported file). ffmpeg: {reason}")
            clip_seconds = (size - 44) / (SAMPLE_RATE * 2)
            yield AudioClip(target, round(clip_seconds, 2))
        finally:
            shutil.rmtree(workdir, ignore_errors=True)  # never keep audio files around
