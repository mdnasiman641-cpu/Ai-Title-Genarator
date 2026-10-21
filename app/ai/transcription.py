"""Speech-to-text (Whisper) for the Title Generator's *Video Audio / Transcript* source.

* **Local Whisper** (default when installed - it is bundled with the Windows app): `faster-whisper`
  runs OpenAI's Whisper model on this PC (CPU, or an NVIDIA GPU when CUDA really works). The model
  (default ``small``) is downloaded once into the app's data folder and reused; nothing is uploaded.
* **Whisper API**: the OpenAI-style ``POST {base_url}/audio/transcriptions`` endpoint of a model
  configuration that already exists in *Settings → AI → AI Models* (OpenAI ``whisper-1``, Groq
  ``whisper-large-v3`` ...) - its base URL and encrypted API key are reused. Providers whose
  OpenAI-compatible API has no speech-to-text endpoint are never sent audio.
* **Gemini audio understanding** (explicit choice only): audio input to an existing Gemini model.

Only a short audio clip (the first N seconds, see :mod:`app.files.audio`) is transcribed - never the
whole video. Transcripts are cleaned (silence/music "hallucinations" removed, meaning unchanged) and
cached per file identity (path, size, modification time) + duration + language + engine/model.
"""

from __future__ import annotations

import base64
import hashlib
import importlib.util
import json
import os
import re
import threading
import time
import unicodedata
import uuid
from abc import ABC, abstractmethod
from collections.abc import Callable
from dataclasses import asdict, dataclass, field, replace
from pathlib import Path
from typing import Any
from urllib.parse import quote, urlparse

from app.ai.errors import AIError, ErrorKind
from app.ai.http_client import request_json
from app.ai.model_config import PROVIDER_TYPES, ModelConfig
from app.files.audio import AudioError, AudioExtractor
from app.utils.logger import get_logger

log = get_logger("ai.transcribe")
DEFAULT_TRANSCRIPTION_MODEL = "whisper-1"  # Whisper over the OpenAI API (verbose_json gives timestamps)
DEFAULT_WHISPER_MODEL = "small"
WHISPER_MODELS = {  # local model -> approximate one-time download (MB)
    "tiny": 75, "base": 145, "small": 484, "medium": 1530, "large-v3-turbo": 1620, "large-v3": 3090,
}  # fmt: skip
MIN_MEANINGFUL_WORDS = 4  # fewer recognised words cannot describe a video
LOCAL_SERVICE = "local"
AUTO_SERVICE = "auto"
DURATIONS = (60, 120, 180, 300)  # seconds; "custom" uses TitleConfig.audio_seconds directly
MAX_TRANSCRIPT_CHARS = 200_000


class TranscriptionError(Exception):
    """``kind``: not_configured, auth, quota, rate_limit, timeout, network, server, invalid, unsupported, too_large, failed."""

    def __init__(self, kind: str, message: str) -> None:
        super().__init__(message)
        self.kind = kind
        self.message = message


# =================================================================== speech
@dataclass(frozen=True, slots=True)
class Segment:
    start: float
    end: float
    text: str


@dataclass(slots=True)
class Speech:
    """What an engine recognised: the text, the (detected) language and timestamped segments when available."""

    text: str
    language: str = ""
    segments: list[Segment] = field(default_factory=list)


_NON_SPEECH = re.compile(r"[\[(（【][^\])）】]{0,40}(?:music|applause|laughter|silence|noise|blank_audio|inaudible|"
                         r"সঙ্গীত|গান|বাজনা)[^\])）】]{0,40}[\])）】]|[♪♫]+", re.IGNORECASE)  # fmt: skip
_SILENCE_HALLUCINATIONS = {  # what Whisper typically "hears" in silence or music - never a transcript on their own
    "you", "thank you", "thanks for watching", "thank you for watching", "thank you so much for watching",
    "please subscribe", "subtitles by the amara.org community", "ご視聴ありがとうございました", "ধন্যবাদ",
}  # fmt: skip
def meaningful_words(text: str) -> int:
    """Words of two or more characters that are not just digits (any script, incl. Bangla)."""
    from app.ai.title_quality import words

    return sum(1 for w in words(text) if len(w) >= 2 and not w.isdigit())


def _norm_phrase(text: str) -> str:
    return re.sub(r"[\s.,!?।'\"…-]+", " ", unicodedata.normalize("NFKC", text).casefold()).strip()


def clean_speech(speech: Speech) -> Speech:
    """Tidy the formatting without changing the meaning: drop non-speech markers, repeated loops and pure silence phrases."""
    segments = speech.segments or ([Segment(0.0, 0.0, speech.text)] if speech.text.strip() else [])
    kept: list[Segment] = []
    for seg in segments:
        text = re.sub(r"\s+", " ", _NON_SPEECH.sub(" ", seg.text)).strip()
        if not text or (kept and _norm_phrase(text) == _norm_phrase(kept[-1].text)):
            continue  # empty after cleaning, or Whisper repeating itself
        kept.append(Segment(seg.start, seg.end, text))
    if kept and all(_norm_phrase(seg.text) in _SILENCE_HALLUCINATIONS for seg in kept):
        kept = []  # only "Thank you." / "Thanks for watching" etc.: silence or music, not speech
    text = " ".join(seg.text for seg in kept)
    return Speech(text[:MAX_TRANSCRIPT_CHARS], speech.language, kept if speech.segments else [])


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
    def transcribe(self, audio: Path, language: str) -> Speech | tuple[str, str]:
        """Return the recognised :class:`Speech` (or ``(text, language)``); raises :class:`TranscriptionError`."""

    def prepare(self, step: Callable[[str], None] | None = None) -> None:  # noqa: B027 - optional hook
        """Load what the engine needs before the first clip (e.g. download a local model)."""


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


# Services verified NOT to offer ``/audio/transcriptions`` (their OpenAI-compatible layer is chat-only).
# Gemini's was checked live: chat/completions answers, audio/transcriptions is HTTP 404 like an unknown path.
NO_SPEECH_TO_TEXT = {
    "generativelanguage.googleapis.com": "Google Gemini's OpenAI-compatible API",
    "openrouter.ai": "OpenRouter",
    "api.deepseek.com": "DeepSeek",
    "api.anthropic.com": "Anthropic",
}
MAX_GEMINI_AUDIO_BYTES = 14_000_000  # inline requests are limited to 20 MB; base64 adds a third (~7 minutes of WAV)


def host_of(config: ModelConfig) -> str:
    return (urlparse(config.base_url).hostname or "").lower()


def speech_to_text_problem(config: ModelConfig) -> str | None:
    """Why an OpenAI-style model configuration cannot transcribe (``None`` = it may)."""
    service = NO_SPEECH_TO_TEXT.get(host_of(config))
    if service is None:
        return None
    return (f"Provider does not support transcription: '{config.display_name}' uses {service} ({host_of(config)}), "
            "which has no /audio/transcriptions endpoint.")  # fmt: skip


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
        default = PROVIDER_TYPES[self.config.provider_type].default_base_url if self.config.provider_type in PROVIDER_TYPES else ""
        where = "" if self.config.base_url == default else f" @ {host_of(self.config)}"  # show where the audio really goes
        return f"{self.config.provider_label}{where} / {self.model} (API)"

    @property
    def cache_id(self) -> str:
        return f"api|{self.config.base_url}|{self.model}"

    @property
    def whisper(self) -> bool:
        return "whisper" in self.model.lower()

    def transcribe(self, audio: Path, language: str) -> Speech:
        fields = {"model": self.model, "response_format": "verbose_json" if self.whisper else "json"}
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
            error = _explain(exc, self.config, self.model, self.timeout, endpoint="/audio/transcriptions")
            log.warning("transcription request failed: %s (HTTP %s) %s", error.kind, exc.status or "-", exc.detail)  # detail is redacted
            raise error from None
        text = data.get("text")
        if not isinstance(text, str):
            raise TranscriptionError("invalid", "Invalid API response: the transcription service returned no 'text' field.")
        detected = _language_code(str(data.get("language") or (language if language != "auto" else "")))
        segments = [Segment(float(seg.get("start") or 0), float(seg.get("end") or 0), str(seg.get("text") or "").strip())
                    for seg in data.get("segments") or [] if isinstance(seg, dict) and _is_speech(seg)]  # fmt: skip
        if data.get("segments") and not segments:
            text = ""  # every segment was silence / noise
        return Speech(text.strip()[:MAX_TRANSCRIPT_CHARS], detected, segments)


def _explain(exc: AIError, config: ModelConfig, model: str, timeout: float, *, endpoint: str) -> TranscriptionError:
    """Actionable message with the HTTP status and the service's own explanation (never the key)."""
    status = f" (HTTP {exc.status})" if exc.status else ""
    said = f" The service said: {exc.detail}" if exc.detail else ""
    where = f"'{config.display_name}' ({config.base_url})"
    lowered = exc.detail.lower()
    if exc.kind == ErrorKind.AUTH:
        return TranscriptionError("auth", f"API authentication failed{status}.{said} Check the API key of {where} in Settings → AI → AI Models"
                                  " (a restricted key also needs permission for audio).")  # fmt: skip
    if exc.kind == ErrorKind.RATE_LIMIT:
        if "quota" in lowered or "billing" in lowered or "credit" in lowered:
            return TranscriptionError("quota", f"API quota exhausted{status}.{said} Add credit or a payment method to the account of {where}.")
        return TranscriptionError("rate_limit", f"Rate limit exceeded{status}.{said} Wait a minute and analyse again.")
    if exc.status == 404 and "model" not in lowered and endpoint == "/audio/transcriptions":
        return TranscriptionError("unsupported_provider", f"Provider does not support transcription{status}: {where} has no {endpoint} endpoint.{said}"
                                  " Choose an OpenAI model with an OpenAI API key, or a Gemini model (audio understanding), in Transcription service.")  # fmt: skip
    if exc.kind == ErrorKind.MODEL_NOT_FOUND or (exc.kind == ErrorKind.BAD_REQUEST and "model" in lowered):
        return TranscriptionError("unsupported", f"Unsupported model or endpoint{status}: '{model}' is not available for speech-to-text at {where}.{said}"
                                  " With OpenAI use gpt-4o-mini-transcribe, gpt-4o-transcribe or whisper-1.")  # fmt: skip
    if exc.status == 413 or "too large" in lowered or "maximum content size" in lowered or "exceeds the maximum" in lowered:
        return TranscriptionError("too_large", f"Audio upload failed{status}: the audio clip is too large for the service. Choose a shorter duration.")
    if exc.kind == ErrorKind.BAD_REQUEST:
        return TranscriptionError("unsupported", f"The transcription request was rejected{status}.{said}")
    if exc.kind == ErrorKind.TIMEOUT:
        return TranscriptionError("timeout", f"Network timeout{status}: no answer from {where} within {timeout:.0f}s."
                                  " Try a shorter duration or a longer timeout.")  # fmt: skip
    if exc.kind == ErrorKind.NETWORK:
        return TranscriptionError("network", f"Audio upload failed: {where} could not be reached ({exc.detail or 'network error'}).")
    if exc.kind == ErrorKind.SERVER:
        return TranscriptionError("server", f"The transcription service had a server error{status}.{said} Try again shortly.")
    if exc.kind == ErrorKind.INVALID_RESPONSE:
        return TranscriptionError("invalid", f"Invalid API response from {where}: {exc.detail}")
    if exc.kind == ErrorKind.BLOCKED:
        return TranscriptionError("blocked", f"The service declined to transcribe this audio.{said}")
    return TranscriptionError(_ERROR_KINDS.get(exc.kind, "failed"), f"Transcription failed{status}.{said}")


_LANGUAGE_NAMES = {"english": "en", "bengali": "bn", "bangla": "bn", "hindi": "hi", "urdu": "ur", "arabic": "ar", "spanish": "es",
                   "french": "fr", "german": "de", "italian": "it", "portuguese": "pt", "russian": "ru", "turkish": "tr",
                   "japanese": "ja", "korean": "ko", "chinese": "zh"}  # fmt: skip


def _language_code(value: str) -> str:
    """The Whisper API reports full names ("english"); the app uses ISO codes ("en")."""
    value = value.strip().lower()
    return _LANGUAGE_NAMES.get(value, value)


def _is_speech(segment: Any) -> bool:
    """Whisper's own rule for silence plus a guard against repetition loops."""
    get = segment.get if isinstance(segment, dict) else lambda key, default=None: getattr(segment, key, default)
    no_speech, logprob, ratio = float(get("no_speech_prob", 0) or 0), float(get("avg_logprob", 0) or 0), float(get("compression_ratio", 0) or 0)
    return not ((no_speech > 0.6 and logprob < -1.0) or ratio > 2.4)


_GEMINI_PROMPT = (
    "Transcribe the speech in this audio clip verbatim, in the language that is spoken. Do not translate, summarise, "
    "describe sounds or music, or add anything that is not said. Answer only with JSON: "
    '{"speech": true or false, "language": "<ISO 639-1 code or empty>", "transcript": "<the spoken words>"}. '
    "If there is no intelligible speech, answer {\"speech\": false, \"language\": \"\", \"transcript\": \"\"}."
)


class GeminiAudioTranscriber(Transcriber):
    """Gemini audio understanding: the clip is sent as audio input to ``generateContent`` of an existing Gemini model.

    Google documents transcription as a supported use of audio input for Gemini models. Only chosen
    explicitly in *Transcription service* - never as a silent substitute for a speech-to-text API.
    """

    def __init__(self, config: ModelConfig, api_key: str | None, timeout: float = 120) -> None:
        self.config = config
        self.api_key = api_key or ""
        self.model = config.model_name
        self.timeout = timeout

    @property
    def label(self) -> str:
        return f"Gemini / {self.model} — audio understanding (API)"

    @property
    def cache_id(self) -> str:
        return f"gemini|{self.config.base_url}|{self.model}"

    def transcribe(self, audio: Path, language: str) -> Speech:
        data = audio.read_bytes()
        if len(data) > MAX_GEMINI_AUDIO_BYTES:
            raise TranscriptionError("too_large", "Audio upload failed: the clip is too large for Gemini (20 MB request limit). Choose 5 minutes or less.")
        prompt = _GEMINI_PROMPT + (f" The speech is in the language with code '{language}'." if language and language != "auto" else "")
        payload = {
            "contents": [{"role": "user", "parts": [{"text": prompt}, {"inline_data": {"mime_type": "audio/wav", "data": base64.b64encode(data).decode("ascii")}}]}],
            "generationConfig": {"temperature": 0, "responseMimeType": "application/json"},
        }  # fmt: skip
        url = f"{self.config.base_url}/models/{quote(self.model, safe='/')}:generateContent"
        try:
            answer = request_json(self.label, "POST", url, headers={"x-goog-api-key": self.api_key}, payload=payload, timeout=self.timeout)
        except AIError as exc:
            error = _explain(exc, self.config, self.model, self.timeout, endpoint=":generateContent")
            log.warning("gemini transcription failed: %s (HTTP %s) %s", error.kind, exc.status or "-", exc.detail)
            raise error from None
        return self._parse(answer, language)

    def _parse(self, answer: dict[str, Any], language: str) -> Speech:
        candidates = answer.get("candidates") or []
        if not candidates:
            block = (answer.get("promptFeedback") or {}).get("blockReason")
            raise TranscriptionError("blocked" if block else "invalid",
                                     f"Gemini declined to transcribe this audio ({block})." if block else "Invalid API response: Gemini returned no answer.")  # fmt: skip
        parts = (candidates[0].get("content") or {}).get("parts") or []
        raw = "".join(str(p.get("text", "")) for p in parts if isinstance(p, dict) and not p.get("thought")).strip()
        if not raw:
            finish = str(candidates[0].get("finishReason", ""))
            raise TranscriptionError("invalid", f"Invalid API response: Gemini returned an empty answer ({finish or 'no reason given'}).")
        try:
            result = json.loads(raw.removeprefix("```json").removesuffix("```").strip())
        except ValueError:
            raise TranscriptionError("invalid", "Invalid API response: Gemini did not return the requested transcript format.") from None
        if not isinstance(result, dict) or not isinstance(result.get("transcript", ""), str):
            raise TranscriptionError("invalid", "Invalid API response: Gemini did not return the requested transcript format.")
        text = "" if result.get("speech") is False else str(result.get("transcript") or "")
        detected = str(result.get("language") or (language if language != "auto" else ""))
        return Speech(text.strip()[:MAX_TRANSCRIPT_CHARS], detected)


_LOADED_MODELS: dict[tuple[str, str], Any] = {}  # loaded once per app run, shared by every batch
_LOAD_LOCK = threading.Lock()


def _wav_samples(path: Path) -> Any:
    """16-bit mono 16 kHz WAV -> float32 samples in [-1, 1] (what Whisper expects).

    Passing samples instead of a file name keeps faster-whisper away from its own PyAV-based decoder,
    which breaks with newer PyAV releases ("open() got an unexpected keyword argument 'metadata_errors'").
    """
    import wave

    import numpy as np

    with wave.open(str(path), "rb") as clip:
        if clip.getnchannels() != 1 or clip.getsampwidth() != 2 or clip.getframerate() != 16_000:
            raise TranscriptionError("failed", "Internal error: the extracted audio is not 16 kHz mono 16-bit WAV.")
        frames = clip.readframes(clip.getnframes())
    return np.frombuffer(frames, dtype="<i2").astype(np.float32) / 32768.0


def whisper_models_dir() -> Path:
    from app.utils.helpers import get_app_data_dir

    return get_app_data_dir() / "whisper-models"


class LocalWhisperTranscriber(Transcriber):
    """Whisper on this PC with `faster-whisper` (bundled with the Windows app; offered only when importable)."""

    local = True

    def __init__(self, model: str = DEFAULT_WHISPER_MODEL, models_dir: Path | None = None, device: str = "auto") -> None:
        model = (model or "").strip()
        self.model_name = model if model and "transcribe" not in model and not model.startswith(("gpt", "whisper-1")) else DEFAULT_WHISPER_MODEL
        self.models_dir = models_dir or whisper_models_dir()
        self.device = device if device in ("auto", "cpu", "cuda") else "auto"
        self.device_used = ""

    @staticmethod
    def available() -> bool:
        return importlib.util.find_spec("faster_whisper") is not None

    @property
    def label(self) -> str:
        return f"Whisper {self.model_name} (local, faster-whisper)"

    @property
    def cache_id(self) -> str:
        return f"local|whisper|{self.model_name}"

    def model_downloaded(self) -> bool:
        if os.path.isdir(self.model_name):
            return True  # a model folder chosen by path
        try:
            from faster_whisper.utils import download_model  # type: ignore[import-not-found]

            download_model(self.model_name, local_files_only=True, cache_dir=str(self.models_dir))
            return True
        except Exception:  # noqa: BLE001 - not downloaded yet (or unknown name: reported when loading)
            return False

    def prepare(self, step: Callable[[str], None] | None = None) -> None:
        self._model(step)

    def _model(self, step: Callable[[str], None] | None = None) -> Any:
        if not self.available():
            raise TranscriptionError("not_configured", "Whisper is unavailable: the local Whisper engine (faster-whisper) is not installed.")
        key = (self.model_name, self.device)
        with _LOAD_LOCK:
            if key in _LOADED_MODELS:
                return _LOADED_MODELS[key]
            os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")  # no console in the GUI app
            os.environ.setdefault("HF_HUB_DISABLE_TELEMETRY", "1")
            self.models_dir.mkdir(parents=True, exist_ok=True)
            if step and not self.model_downloaded():
                step("download")
            elif step:
                step("load")
            try:
                import ctranslate2  # type: ignore[import-not-found]
                from faster_whisper import WhisperModel  # type: ignore[import-not-found]
            except Exception as exc:  # noqa: BLE001
                raise TranscriptionError("not_configured", f"Whisper is unavailable: the local engine could not start ({exc}).") from None
            attempts = [("cpu", "int8")]
            if self.device != "cpu" and ctranslate2.get_cuda_device_count() > 0:
                attempts.insert(0, ("cuda", "float16"))  # only when CUDA really works; otherwise the CPU is used
            last: Exception | None = None
            for device, compute in attempts:
                try:
                    model = WhisperModel(self.model_name, device=device, compute_type=compute, download_root=str(self.models_dir))
                except Exception as exc:  # noqa: BLE001
                    last = exc
                    log.info("Whisper %s could not load on %s: %s", self.model_name, device, exc)
                    continue
                self.device_used = device
                _LOADED_MODELS[key] = model
                log.info("Whisper %s loaded on %s (%s)", self.model_name, device, compute)
                return model
            text = str(last or "")
            if not self.model_downloaded():
                raise TranscriptionError("model_download", f"The Whisper model '{self.model_name}' could not be downloaded "
                                         f"(first use needs internet, about {WHISPER_MODELS.get(self.model_name, '?')} MB): {text}")  # fmt: skip
            raise TranscriptionError("model_load", f"The local Whisper model '{self.model_name}' could not be loaded: {text}")

    def transcribe(self, audio: Path, language: str) -> Speech:
        model = self._model()
        try:
            samples = _wav_samples(audio)  # our clip is already 16 kHz mono - faster-whisper's own decoder is not needed
            segments, info = model.transcribe(samples, language=None if language in ("", "auto") else language, beam_size=5,
                                              vad_filter=True, condition_on_previous_text=False)  # fmt: skip
            kept = [Segment(round(seg.start, 2), round(seg.end, 2), seg.text.strip()) for seg in segments if _is_speech(seg)]
        except Exception as exc:  # noqa: BLE001 - third-party engine: report, never crash
            raise TranscriptionError("failed", f"Local Whisper transcription failed: {exc}") from None
        text = " ".join(seg.text for seg in kept)
        return Speech(text.strip()[:MAX_TRANSCRIPT_CHARS], str(getattr(info, "language", "") or ""), kept)


def transcription_candidates(registry: Any) -> list[ModelConfig]:
    """Usable OpenAI / OpenAI-compatible models whose endpoint may offer ``/audio/transcriptions``."""
    return [m for m in registry.list() if m.provider_type in ("openai", "openai_compatible") and registry.is_usable(m)
            and speech_to_text_problem(m) is None]  # fmt: skip


def gemini_candidates(registry: Any) -> list[ModelConfig]:
    """Usable Gemini models (audio understanding through ``generateContent``)."""
    return [m for m in registry.list() if m.provider_type == "gemini" and registry.is_usable(m)]


def build_transcriber(registry: Any, settings: Any) -> Transcriber:
    """The backend chosen in the Title Generator - never silently another one."""
    service = settings.transcription_service or AUTO_SERVICE
    timeout = settings.transcription_timeout_s
    local_model = getattr(settings, "whisper_model", DEFAULT_WHISPER_MODEL)
    device = getattr(settings, "whisper_device", "auto")
    if service == LOCAL_SERVICE:
        if not LocalWhisperTranscriber.available():
            raise TranscriptionError("not_configured", "Whisper is unavailable: local Whisper is selected but 'faster-whisper' is not installed.")
        return LocalWhisperTranscriber(local_model, device=device)
    if service == AUTO_SERVICE:
        if LocalWhisperTranscriber.available():  # free and private: no API key, nothing uploaded
            return LocalWhisperTranscriber(local_model, device=device)
        config = next((m for m in transcription_candidates(registry) if m.provider_type == "openai"), None)
        if config is None:
            skipped = [speech_to_text_problem(m) for m in registry.list()
                       if m.provider_type in ("openai", "openai_compatible") and speech_to_text_problem(m)]  # fmt: skip
            hint = " Or choose a Gemini model (audio understanding) in Transcription service." if gemini_candidates(registry) else ""
            raise TranscriptionError(
                "not_configured",
                "Whisper is unavailable: the local Whisper engine is not installed and no Whisper API is configured. "
                "Add an OpenAI model with its API key (Settings → AI → AI Models), "
                "or choose an OpenAI-compatible model that offers speech-to-text in Video Content Analysis." + hint
                + (" " + " ".join(p for p in skipped if p) if skipped else ""),
            )
    else:
        config = registry.get(service)
        if config is None or not registry.is_usable(config):
            raise TranscriptionError("not_configured", "The selected transcription service is no longer available or has no API key.")
        if config.provider_type == "gemini":
            return GeminiAudioTranscriber(config, registry.api_key(config.id), timeout)
        if config.provider_type == "custom":
            raise TranscriptionError("unsupported_provider", f"Provider does not support transcription: '{config.display_name}' is a custom provider.")
        problem = speech_to_text_problem(config)
        if problem:
            raise TranscriptionError("unsupported_provider", problem)  # refused before any audio is uploaded
    return ApiTranscriber(config, registry.api_key(config.id), settings.transcription_model, timeout)


# ============================================================= service + cache
@dataclass(slots=True)
class TranscriptResult:
    """What happened for one video (``status``: ok, empty, insufficient, no_audio, failed).

    ``stage`` names where a failure happened: ``extract`` (audio) or ``transcribe`` (Whisper / API).
    ``segments``: ``{"start", "end", "text"}`` dicts when the engine gives timestamps.
    """

    status: str
    text: str = ""
    language: str = ""
    analyzed_seconds: float = 0.0
    backend: str = ""
    error: str = ""
    error_kind: str = ""
    from_cache: bool = False
    stage: str = ""
    segments: list[dict[str, Any]] = field(default_factory=list)

    @property
    def words(self) -> int:
        return meaningful_words(self.text)

    def timestamped(self, limit: int = 200) -> list[str]:
        """``[00:12] text`` lines (or the plain text when there are no timestamps)."""
        if not self.segments:
            return [self.text] if self.text else []
        return [f"[{_clock(seg.get('start', 0))}–{_clock(seg.get('end', 0))}] {seg.get('text', '')}" for seg in self.segments[:limit]]

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


def _clock(seconds: Any) -> str:
    total = int(float(seconds or 0))
    return f"{total // 60:02d}:{total % 60:02d}"


_PER_FILE_KINDS = ("no_audio", "unreadable", "empty", "insufficient")


def main_failure(results: list[TranscriptResult]) -> TranscriptResult | None:
    """The failure that explains most videos - service problems (key, quota, model, network) before per-file ones."""
    failed = [r for r in results if r.status == "failed" and r.error]
    if not failed:
        return None
    counts: dict[str, int] = {}
    for result in failed:
        counts[result.error_kind] = counts.get(result.error_kind, 0) + 1
    return max(failed, key=lambda r: (r.error_kind not in _PER_FILE_KINDS, counts[r.error_kind]))


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
        self._prepare_error: TranscriptionError | None = None  # e.g. model download failed: not retried for every video

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
        if result.status in ("ok", "empty", "insufficient", "no_audio"):  # deterministic outcomes are worth remembering
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
        stage = "transcribe"
        try:
            if self._prepare_error is not None:
                raise self._prepare_error
            try:
                self.transcriber.prepare(step)  # e.g. download / load the local Whisper model once
            except TranscriptionError as exc:
                self._prepare_error = exc
                raise
            stage = "extract"
            if step:
                step("extract")
            with self.extractor.extracted_audio(source, seconds) as clip:
                stage = "transcribe"
                if step:
                    step("transcribe")
                self.requests += 1
                raw = self.transcriber.transcribe(clip.path, language)
                analyzed = clip.seconds
        except AudioError as exc:
            status = "no_audio" if exc.kind == "no_audio" else "failed"
            return TranscriptResult(status, error=exc.message, error_kind=exc.kind, backend=backend, stage="extract")
        except TranscriptionError as exc:
            log.warning("transcription failed (%s) for %s", exc.kind, source.name)
            return TranscriptResult("failed", error=exc.message, error_kind=exc.kind, backend=backend, stage=stage)
        except Exception as exc:  # noqa: BLE001 - one video's unexpected problem must not end the whole batch
            log.exception("unexpected transcription error for %s", source.name)
            return TranscriptResult("failed", error=f"Transcription failed: {type(exc).__name__}: {exc}", error_kind="failed",
                                    backend=backend, stage=stage)  # fmt: skip
        speech = clean_speech(raw if isinstance(raw, Speech) else Speech(str(raw[0]), str(raw[1])))
        segments = [{"start": seg.start, "end": seg.end, "text": seg.text} for seg in speech.segments]
        common = {"language": speech.language, "analyzed_seconds": analyzed, "backend": backend, "segments": segments}
        if not speech.text.strip():
            return TranscriptResult("empty", error="No speech detected in the analysed part (silence, music or noise only).",
                                    error_kind="empty", **common)  # fmt: skip
        words = meaningful_words(speech.text)
        if words < MIN_MEANINGFUL_WORDS:
            return TranscriptResult("insufficient", text=speech.text, error=f"Too little speech to describe the video "
                                    f"(only {words} word(s) recognised).", error_kind="insufficient", **common)  # fmt: skip
        return TranscriptResult("ok", text=speech.text, **common)

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
                                 setup_error=TranscriptionError("no_ffmpeg", "FFmpeg not found. Install ffmpeg or set its folder in Settings → General → ffmpeg folder."))  # fmt: skip
    return TranscriptService(extractor, transcriber, ctx.db)

