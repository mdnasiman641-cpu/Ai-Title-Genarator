"""Speech-to-text for the Title Generator's *Video Audio / Transcript* source.

* **API** (default): the OpenAI-style ``POST {base_url}/audio/transcriptions`` endpoint of a model
  configuration that already exists in *Settings → AI → AI Models* - its base URL and its
  encrypted API key are reused, so there is no second key store and no second provider manager.
  Works with OpenAI (``gpt-4o-mini-transcribe``, ``gpt-4o-transcribe``, ``whisper-1``) and with
  OpenAI-compatible services that offer the same endpoint (e.g. Groq ``whisper-large-v3``).
  Gemini's text-generation endpoint is *not* used for audio.
* **Local** (optional): `faster-whisper` when it is installed; nothing leaves the PC.

Only a short audio clip (the first N seconds, see :mod:`app.files.audio`) is uploaded - never the
video. Transcripts are cached per file identity (path, size, modification time) + duration +
language + backend/model, so unchanged files are never transcribed twice.
"""

from __future__ import annotations

import hashlib
import importlib.util
import threading
import time
import uuid
from abc import ABC, abstractmethod
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Any

from app.ai.errors import AIError, ErrorKind
from app.ai.http_client import request_json
from app.ai.model_config import ModelConfig
from app.files.audio import AudioError, AudioExtractor
from app.utils.logger import get_logger

log = get_logger("ai.transcribe")
DEFAULT_TRANSCRIPTION_MODEL = "gpt-4o-mini-transcribe"
LOCAL_SERVICE = "local"
AUTO_SERVICE = "auto"
DURATIONS = (60, 120, 180, 300)  # seconds; "custom" uses TitleConfig.audio_seconds directly
MAX_TRANSCRIPT_CHARS = 200_000


class TranscriptionError(Exception):
    """``kind``: not_configured, auth, rate_limit, timeout, network, server, invalid, unsupported, failed."""

    def __init__(self, kind: str, message: str) -> None:
        super().__init__(message)
        self.kind = kind
        self.message = message


# =================================================================== backends
class Transcriber(ABC):
    """Turns a WAV clip into text."""

    local: bool = False

    @property
    @abstractmethod
    def label(self) -> str:
        """Shown to the user, e.g. ``OpenAI / gpt-4o-mini-transcribe (API)``."""

    @property
    @abstractmethod
    def cache_id(self) -> str:
        """Identifies backend + model in the transcript cache key."""

    @abstractmethod
    def transcribe(self, audio: Path, language: str) -> tuple[str, str]:
        """Return ``(text, detected_language)``; raises :class:`TranscriptionError`."""


def _multipart(fields: dict[str, str], file_field: str, file_path: Path) -> tuple[bytes, str]:
    boundary = f"----aivfa{uuid.uuid4().hex}"
    parts: list[bytes] = []
    for name, value in fields.items():
        parts.append(f'--{boundary}\r\nContent-Disposition: form-data; name="{name}"\r\n\r\n{value}\r\n'.encode())
    parts.append(
        f'--{boundary}\r\nContent-Disposition: form-data; name="{file_field}"; filename="{file_path.name}"\r\n'
        "Content-Type: audio/wav\r\n\r\n".encode()
    )
    parts.append(file_path.read_bytes())
    parts.append(f"\r\n--{boundary}--\r\n".encode())
    return b"".join(parts), f"multipart/form-data; boundary={boundary}"


_ERROR_KINDS = {
    ErrorKind.AUTH: "auth", ErrorKind.RATE_LIMIT: "rate_limit", ErrorKind.TIMEOUT: "timeout", ErrorKind.NETWORK: "network",
    ErrorKind.SERVER: "server", ErrorKind.INVALID_RESPONSE: "invalid", ErrorKind.MODEL_NOT_FOUND: "unsupported",
    ErrorKind.BAD_REQUEST: "unsupported",
}  # fmt: skip


class ApiTranscriber(Transcriber):
    """``/audio/transcriptions`` of an existing (OpenAI or OpenAI-compatible) model configuration."""

    def __init__(self, config: ModelConfig, api_key: str | None, model: str, timeout: float = 120) -> None:
        self.config = config
        self.api_key = api_key or ""
        self.model = model.strip() or DEFAULT_TRANSCRIPTION_MODEL
        self.timeout = timeout

    @property
    def label(self) -> str:
        return f"{self.config.provider_label} / {self.model} (API)"

    @property
    def cache_id(self) -> str:
        return f"api|{self.config.base_url}|{self.model}"

    def transcribe(self, audio: Path, language: str) -> tuple[str, str]:
        fields = {"model": self.model, "response_format": "json"}
        if language and language != "auto":
            fields["language"] = language
        body, content_type = _multipart(fields, "file", audio)
        headers = {"Content-Type": content_type}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        try:
            data = request_json(self.label, "POST", f"{self.config.base_url}/audio/transcriptions", headers=headers,
                                body=body, timeout=self.timeout)  # fmt: skip
        except AIError as exc:
            raise TranscriptionError(_ERROR_KINDS.get(exc.kind, "failed"), exc.user_message) from None
        text = data.get("text")
        if not isinstance(text, str):
            raise TranscriptionError("invalid", "The transcription service returned no text field.")
        detected = str(data.get("language") or (language if language != "auto" else ""))
        return text.strip()[:MAX_TRANSCRIPT_CHARS], detected


class LocalWhisperTranscriber(Transcriber):
    """Optional on-device transcription with `faster-whisper` (only offered when installed)."""

    local = True

    def __init__(self, model: str = "small") -> None:
        self.model_name = model if model and "transcribe" not in model and not model.startswith("gpt") else "small"
        self._model: Any = None
        self._lock = threading.Lock()

    @staticmethod
    def available() -> bool:
        return importlib.util.find_spec("faster_whisper") is not None

    @property
    def label(self) -> str:
        return f"faster-whisper / {self.model_name} (local)"

    @property
    def cache_id(self) -> str:
        return f"local|{self.model_name}"

    def transcribe(self, audio: Path, language: str) -> tuple[str, str]:
        if not self.available():
            raise TranscriptionError("not_configured", "Local transcription needs the optional 'faster-whisper' package.")
        try:
            with self._lock:
                if self._model is None:
                    from faster_whisper import WhisperModel  # type: ignore[import-not-found]

                    self._model = WhisperModel(self.model_name, device="cpu", compute_type="int8")
                segments, info = self._model.transcribe(str(audio), language=None if language == "auto" else language)
                text = " ".join(segment.text.strip() for segment in segments)
        except Exception as exc:  # noqa: BLE001 - third-party engine: report, never crash
            raise TranscriptionError("failed", f"Local transcription failed: {exc}") from None
        return text.strip()[:MAX_TRANSCRIPT_CHARS], str(getattr(info, "language", "") or "")


def transcription_candidates(registry: Any) -> list[ModelConfig]:
    """Configured models whose endpoint may offer ``/audio/transcriptions`` (OpenAI and OpenAI-compatible)."""
    return [m for m in registry.list() if m.provider_type in ("openai", "openai_compatible") and registry.is_usable(m)]


def build_transcriber(registry: Any, settings: Any) -> Transcriber:
    """The backend chosen in the Title Generator - never silently another one."""
    service = settings.transcription_service or AUTO_SERVICE
    if service == LOCAL_SERVICE:
        if not LocalWhisperTranscriber.available():
            raise TranscriptionError("not_configured", "Local transcription is selected but 'faster-whisper' is not installed.")
        return LocalWhisperTranscriber(settings.transcription_model)
    if service == AUTO_SERVICE:
        config = next((m for m in transcription_candidates(registry) if m.provider_type == "openai"), None)
        if config is None:
            raise TranscriptionError(
                "not_configured",
                "No transcription service is configured. Add an OpenAI model with its API key (Settings → AI → AI Models), "
                "or choose an OpenAI-compatible model that offers speech-to-text in Video Content Analysis.",
            )
    else:
        config = registry.get(service)
        if config is None or not registry.is_usable(config):
            raise TranscriptionError("not_configured", "The selected transcription service is no longer available or has no API key.")
    return ApiTranscriber(config, registry.api_key(config.id), settings.transcription_model, settings.transcription_timeout_s)


# ============================================================= service + cache
@dataclass(slots=True)
class TranscriptResult:
    """What happened for one video (``status``: ok, empty, no_audio, failed)."""

    status: str
    text: str = ""
    language: str = ""
    analyzed_seconds: float = 0.0
    backend: str = ""
    error: str = ""
    error_kind: str = ""
    from_cache: bool = False

    @property
    def ok(self) -> bool:
        return self.status == "ok"

    @property
    def analyzed_range(self) -> str:
        total = int(round(self.analyzed_seconds))
        return f"00:00–{total // 60:02d}:{total % 60:02d}"

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data.pop("from_cache", None)
        return data


class TranscriptService:
    """Audio extraction + transcription + cache. One instance per batch (deduplicates within it)."""

    def __init__(self, extractor: AudioExtractor, transcriber: Transcriber | None, db: Any | None = None,
                 *, setup_error: TranscriptionError | None = None) -> None:  # fmt: skip
        self.extractor = extractor
        self.transcriber = transcriber
        self.db = db
        self.setup_error = setup_error
        self.requests = 0  # real transcription calls (for status/tests)
        self._memo: dict[str, TranscriptResult] = {}

    @property
    def label(self) -> str:
        return self.transcriber.label if self.transcriber else ""

    def cache_key(self, source: Path, seconds: float, language: str) -> str:
        stat = source.stat()
        backend = self.transcriber.cache_id if self.transcriber else "-"
        raw = f"{source.resolve()}|{stat.st_size}|{stat.st_mtime_ns}|{int(seconds)}|{language}|{backend}"
        return hashlib.sha1(raw.encode("utf-8")).hexdigest()

    def cached(self, source: Path, seconds: float, language: str) -> TranscriptResult | None:
        try:
            key = self.cache_key(source, seconds, language)
        except OSError:
            return None
        if key in self._memo:
            return replace(self._memo[key], from_cache=True)
        if self.db is None:
            return None
        data = self.db.get_transcript_cache(key)
        if not data:
            return None
        result = TranscriptResult(**{k: v for k, v in data.items() if k in TranscriptResult.__dataclass_fields__})
        result.from_cache = True
        self._memo[key] = result
        return result

    def get(self, source: Path, seconds: float, language: str = "auto", *, step: Any = None) -> TranscriptResult:
        """Transcript of the first ``seconds`` of ``source`` (cached when the file is unchanged)."""
        if self.setup_error is not None:
            return TranscriptResult("failed", error=self.setup_error.message, error_kind=self.setup_error.kind)
        hit = self.cached(source, seconds, language)
        if hit is not None:
            return hit
        assert self.transcriber is not None
        try:
            key = self.cache_key(source, seconds, language)
        except OSError as exc:
            return TranscriptResult("failed", error=f"The file cannot be read: {exc.strerror or exc}", error_kind="unreadable")
        result = self._transcribe(source, seconds, language, step)
        if result.status in ("ok", "empty", "no_audio"):  # deterministic outcomes are worth remembering
            self._memo[key] = result
            if self.db is not None:
                try:
                    self.db.put_transcript_cache(key, time.time(), result.to_dict())
                except Exception:  # noqa: BLE001 - a cache problem must never break the batch
                    log.debug("transcript cache write failed", exc_info=True)
        return result

    def _transcribe(self, source: Path, seconds: float, language: str, step: Any) -> TranscriptResult:
        assert self.transcriber is not None
        backend = self.transcriber.label
        try:
            if step:
                step("extract")
            with self.extractor.extracted_audio(source, seconds) as clip:
                if step:
                    step("transcribe")
                self.requests += 1
                text, detected = self.transcriber.transcribe(clip.path, language)
                analyzed = clip.seconds
        except AudioError as exc:
            status = "no_audio" if exc.kind == "no_audio" else "failed"
            return TranscriptResult(status, error=exc.message, error_kind=exc.kind, backend=backend)
        except TranscriptionError as exc:
            log.warning("transcription failed (%s) for %s", exc.kind, source.name)
            return TranscriptResult("failed", error=exc.message, error_kind=exc.kind, backend=backend)
        if not text.strip():
            return TranscriptResult("empty", language=detected, analyzed_seconds=analyzed, backend=backend,
                                    error="No speech was recognised in the analysed part.", error_kind="empty")  # fmt: skip
        return TranscriptResult("ok", text=text, language=detected, analyzed_seconds=analyzed, backend=backend)

    def clear_cache(self) -> int:
        return self.db.purge_transcript_cache() if self.db is not None else 0


def make_transcript_service(ctx: Any) -> TranscriptService:
    """Service built from the saved settings; a configuration problem is reported per file, never hidden."""
    settings = ctx.settings.load()
    extractor = AudioExtractor(settings.ffmpeg_dir)
    try:
        transcriber = build_transcriber(ctx.registry, settings)
    except TranscriptionError as exc:
        return TranscriptService(extractor, None, ctx.db, setup_error=exc)
    if not extractor.available:
        return TranscriptService(extractor, transcriber, ctx.db,
                                 setup_error=TranscriptionError("no_ffmpeg", "ffmpeg was not found (Settings → General → ffmpeg folder)."))  # fmt: skip
    return TranscriptService(extractor, transcriber, ctx.db)

