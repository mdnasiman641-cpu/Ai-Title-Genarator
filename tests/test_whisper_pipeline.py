"""Whisper workflow: audio extraction (ffmpeg + built-in decoder), local/API Whisper, transcript cleanup,
grounded title suggestions, Bangla handling and the GUI pieces. External engines and APIs are mocked here;
tools/verify_whisper.py runs the real Whisper model on real speech (CI)."""

from __future__ import annotations

import hashlib
import shutil
import types
from pathlib import Path

import pytest

from app.ai.model_config import ModelConfig
from app.ai.pipeline import run_pipeline
from app.ai.title_quality import count_words
from app.ai.transcript_check import Suggestion, TranscriptIndex, problem, validate_suggestions
from app.ai.transcription import (
    ApiTranscriber,
    LocalWhisperTranscriber,
    Segment,
    Speech,
    TranscriptionError,
    TranscriptService,
    build_transcriber,
    clean_speech,
    meaningful_words,
)
from app.config.settings import AppSettings
from app.context import AppContext
from app.files import audio as audio_module
from app.files.audio import AudioError, AudioExtractor
from tests.fake_ai_server import FakeAIServer
from tests.test_transcription import FakeExtractor, FakeTranscriber, TranscriptModel, files, request, transcript_cfg

DATA = Path(__file__).parent / "data"
JFK = DATA / "jfk_speech.mp4"
JFK_TEXT = "And so, my fellow Americans, ask not what your country can do for you, ask what you can do for your country."
BANGLA = "আজকে আমরা ঢাকার নতুন মেট্রোরেল নিয়ে কথা বলব। এই ট্রেন প্রতিদিন হাজার হাজার মানুষকে সময়মতো অফিসে পৌঁছে দেয়।"
FFMPEG = shutil.which("ffmpeg")
needs_ffmpeg = pytest.mark.skipif(FFMPEG is None, reason="ffmpeg not installed")
needs_pyav = pytest.mark.skipif(not audio_module.pyav_available(), reason="PyAV not installed")


@pytest.fixture
def ctx(tmp_path):
    return AppContext.create(tmp_path / "ctx")


@pytest.fixture
def router_for(ctx):
    from app.ai.ai_router import ModelRouter, RouterOptions
    from app.ai.model_health import HealthTracker

    def _make(*models):
        table = {}
        for model in models:
            config = ctx.registry.add(ModelConfig.new("openai_compatible", model.model, base_url="http://localhost:9", max_retries=0))
            table[config.id] = model
        return ModelRouter(ctx.registry, HealthTracker(None), RouterOptions(), factory=lambda c: table[c.id], sleep=lambda _s: None)

    return _make


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def long_video(tmp_path: Path, seconds: int) -> Path:
    """A video whose audio is longer than the analysed part: JFK speech, then silence."""
    import subprocess

    out = tmp_path / f"long_{seconds}.mp4"
    subprocess.run([FFMPEG, "-v", "error", "-y", "-f", "lavfi", "-i", f"color=c=black:s=64x48:r=2:d={seconds}", "-i", str(JFK),
                    "-filter_complex", f"[1:a]apad=whole_dur={seconds}[a]", "-map", "0:v", "-map", "[a]", "-c:v", "libx264",
                    "-c:a", "aac", "-t", str(seconds), str(out)], check=True)  # fmt: skip
    return out


# ============================================================ audio extraction
def _extractors() -> list[tuple[str, AudioExtractor]]:
    out = []
    if FFMPEG:
        out.append(("ffmpeg", AudioExtractor()))
    if audio_module.pyav_available():
        builtin = AudioExtractor()
        builtin.binary, builtin.builtin = None, True
        out.append(("built-in", builtin))
    return out


@pytest.mark.parametrize(("name", "extractor"), _extractors(), ids=lambda v: v if isinstance(v, str) else "")
def test_short_video_gives_all_of_its_audio_and_stays_unchanged(name, extractor, tmp_path):
    video = tmp_path / "speech.mp4"
    shutil.copy(JFK, video)
    before = sha(video)
    with extractor.extracted_audio(video, 180) as clip:
        assert abs(clip.seconds - 11.0) < 0.1  # shorter than 3 minutes: everything there is
        workdir = clip.path.parent
        assert clip.path.stat().st_size > 300_000
    assert not workdir.exists() and sha(video) == before  # temp audio deleted, original untouched


@needs_ffmpeg
@pytest.mark.parametrize(("name", "extractor"), _extractors(), ids=lambda v: v if isinstance(v, str) else "")
def test_long_video_is_cut_at_three_minutes(name, extractor, tmp_path):
    video = long_video(tmp_path, 200)
    before = sha(video)
    with extractor.extracted_audio(video, 180) as clip:
        assert abs(clip.seconds - 180.0) < 0.1
    assert sha(video) == before


@needs_pyav
def test_builtin_decoder_reports_missing_audio_and_damage(tmp_path):
    extractor = AudioExtractor()
    extractor.binary, extractor.builtin = None, True
    broken = tmp_path / "broken.mp4"
    broken.write_bytes(JFK.read_bytes()[20000:50000])
    with pytest.raises(AudioError) as damaged, extractor.extracted_audio(broken, 180):
        pass
    assert damaged.value.kind == "unreadable" and damaged.value.message.startswith("Audio extraction failed")
    if FFMPEG:
        import subprocess

        silent = tmp_path / "silent.mp4"
        subprocess.run([FFMPEG, "-v", "error", "-y", "-f", "lavfi", "-i", "color=c=blue:s=64x48:d=2", "-c:v", "libx264", str(silent)], check=True)
        with pytest.raises(AudioError) as no_audio, extractor.extracted_audio(silent, 180):
            pass
        assert no_audio.value.kind == "no_audio"


# ============================================================ local Whisper
class FakeWhisperModel:
    loads: list[tuple[str, str]] = []

    def __init__(self, name, device, compute_type, download_root):  # noqa: ANN001
        if device == "cuda":
            raise RuntimeError("CUDA driver not found")
        FakeWhisperModel.loads.append((name, device))

    def transcribe(self, audio, language=None, **kw):  # noqa: ANN001
        # samples, never a file: faster-whisper's own decoder breaks with newer PyAV ("metadata_errors")
        assert not isinstance(audio, (str, Path)) and audio.dtype.name == "float32"
        seg = lambda s, e, t, nsp=0.01, lp=-0.2: types.SimpleNamespace(start=s, end=e, text=t, no_speech_prob=nsp, avg_logprob=lp, compression_ratio=1.2)  # noqa: E731
        segments = [seg(0.0, 4.2, " And so, my fellow Americans,"), seg(4.2, 11.0, " ask not what your country can do for you."),
                    seg(11.0, 30.0, " Thank you.", nsp=0.9, lp=-1.5)]  # fmt: skip
        assert kw.get("vad_filter") is True
        return iter(segments), types.SimpleNamespace(language=language or "en")


@pytest.fixture
def fake_whisper(monkeypatch, tmp_path):
    import faster_whisper  # noqa: F401 - installed with the app

    import app.ai.transcription as transcription

    FakeWhisperModel.loads = []
    monkeypatch.setattr("faster_whisper.WhisperModel", FakeWhisperModel)
    monkeypatch.setattr("ctranslate2.get_cuda_device_count", lambda: 1)  # pretend a GPU: the CPU must take over
    monkeypatch.setattr(transcription, "_LOADED_MODELS", {})
    monkeypatch.setattr(LocalWhisperTranscriber, "model_downloaded", lambda self: bool(FakeWhisperModel.loads))
    monkeypatch.setattr(transcription, "_wav_samples", lambda path: __import__("numpy").zeros(16000, dtype="float32"))
    return tmp_path / "models"


@pytest.mark.local_whisper
def test_local_whisper_is_the_automatic_choice(fake_whisper, ctx):
    pytest.importorskip("faster_whisper")
    transcriber = build_transcriber(ctx.registry, AppSettings(transcription_service="auto", whisper_model="base"))
    assert isinstance(transcriber, LocalWhisperTranscriber) and transcriber.label == "Whisper base (local, faster-whisper)"


@pytest.mark.local_whisper
def test_local_whisper_downloads_once_falls_back_to_cpu_and_filters_silence(fake_whisper, ctx, tmp_path):
    pytest.importorskip("faster_whisper")
    service = TranscriptService(FakeExtractor(), LocalWhisperTranscriber("small", fake_whisper), ctx.db)
    video = tmp_path / "speech.mp4"
    video.write_bytes(b"video")
    stages: list[str] = []
    result = service.get(video, 180, "auto", step=stages.append)
    assert stages == ["download", "extract", "transcribe"]  # first use: the model is downloaded
    assert FakeWhisperModel.loads == [("small", "cpu")]  # CUDA failed -> CPU, no crash
    assert result.ok and result.text == "And so, my fellow Americans, ask not what your country can do for you."  # "Thank you." in silence dropped
    assert result.timestamped() == ["[00:00–00:04] And so, my fellow Americans,", "[00:04–00:11] ask not what your country can do for you."]
    other = tmp_path / "other.mp4"
    other.write_bytes(b"video 2")
    stages.clear()
    TranscriptService(FakeExtractor(), LocalWhisperTranscriber("small", fake_whisper), ctx.db).get(other, 180, "auto", step=stages.append)
    assert stages == ["extract", "transcribe"] and len(FakeWhisperModel.loads) == 1  # loaded once per app run, never re-downloaded


@pytest.mark.local_whisper
def test_model_download_failure_is_reported_once_per_batch(monkeypatch, ctx, tmp_path):
    pytest.importorskip("faster_whisper")
    import app.ai.transcription as transcription

    attempts = []

    def offline(*a, **k):  # noqa: ANN002, ANN003
        attempts.append(1)
        raise OSError("Cannot reach huggingface.co")

    monkeypatch.setattr("faster_whisper.WhisperModel", offline)
    monkeypatch.setattr(transcription, "_LOADED_MODELS", {})
    monkeypatch.setattr(LocalWhisperTranscriber, "model_downloaded", lambda self: False)
    monkeypatch.setattr("ctranslate2.get_cuda_device_count", lambda: 0)
    service = TranscriptService(FakeExtractor(), LocalWhisperTranscriber("small", tmp_path), ctx.db)
    for name in ("a.mp4", "b.mp4"):
        (tmp_path / name).write_bytes(name.encode())
    first, second = service.get(tmp_path / "a.mp4", 180), service.get(tmp_path / "b.mp4", 180)
    assert first.error_kind == second.error_kind == "model_download" and len(attempts) == 1
    assert "could not be downloaded" in first.error and "484 MB" in first.error and first.stage == "transcribe"


@pytest.mark.skipif(not _extractors(), reason="no audio decoder")
def test_extracted_clip_becomes_whisper_samples():
    np = pytest.importorskip("numpy")
    from app.ai.transcription import _wav_samples

    with _extractors()[0][1].extracted_audio(JFK, 180) as clip:
        samples = _wav_samples(clip.path)
    assert samples.dtype == np.float32 and abs(len(samples) / 16000 - 11.0) < 0.1
    assert 0.05 < float(np.abs(samples).max()) <= 1.0  # real speech, correctly scaled


# ============================================================== Whisper API
def test_whisper_api_returns_timestamps_and_drops_silence(tmp_path):
    clip = tmp_path / "clip.wav"
    clip.write_bytes(b"RIFF")
    verbose = {"text": "Ask not what your country can do for you. Thanks for watching!", "language": "english", "segments": [
        {"start": 0.0, "end": 3.5, "text": " Ask not what your country can do for you.", "no_speech_prob": 0.02, "avg_logprob": -0.2},
        {"start": 3.5, "end": 9.0, "text": " Thanks for watching!", "no_speech_prob": 0.95, "avg_logprob": -1.4}]}  # fmt: skip
    with FakeAIServer() as srv:
        srv.queue(200, verbose)
        speech = ApiTranscriber(ModelConfig.new("openai", "gpt-4o-mini", base_url=srv.base), "sk-x", "whisper-1", 5).transcribe(clip, "auto")
        body = srv.requests[0]["body"]
    assert 'name="response_format"' in body and "verbose_json" in body
    assert speech.language == "en" and [s.text for s in speech.segments] == ["Ask not what your country can do for you."]


# ======================================================== transcript cleanup
def test_cleanup_keeps_meaning_and_removes_artifacts():
    raw = Speech("", "en", [Segment(0, 2, "[Music]"), Segment(2, 5, "  ask not   what your country ♪ "), Segment(5, 7, "ask not what your country"),
                            Segment(7, 9, "can do for you")])  # fmt: skip
    clean = clean_speech(raw)
    assert clean.text == "ask not what your country can do for you" and len(clean.segments) == 2
    assert clean_speech(Speech("Thank you.", "en")).text == ""  # Whisper's typical silence hallucination alone
    assert clean_speech(Speech("Thanks for watching, see you tomorrow at the stadium", "en")).text != ""  # real speech kept


def test_bangla_text_is_counted_and_titled_in_words():
    assert meaningful_words(BANGLA) == 18
    assert count_words("ঢাকার মেট্রোরেলে প্রতিদিন হাজারো মানুষের যাত্রা") == 6  # not split at vowel signs


def test_insufficient_and_empty_speech_never_count_as_transcripts(ctx, tmp_path):
    (tmp_path / "hello.mp4").write_bytes(b"1")
    (tmp_path / "music.mp4").write_bytes(b"2")
    texts = {"hello.mp4": "Hello there.", "music.mp4": "[Music] ♪♪"}
    service = TranscriptService(FakeExtractor(), FakeTranscriber(texts=texts), ctx.db)
    hello, music = service.get(tmp_path / "hello.mp4", 180), service.get(tmp_path / "music.mp4", 180)
    assert hello.status == "insufficient" and not hello.ok and "only 2 word" in hello.error
    assert music.status == "empty" and "No speech detected" in music.error


# ======================================================= grounded suggestions
def test_suggestions_must_be_supported_by_the_transcript():
    index = TranscriptIndex(JFK_TEXT)
    assert problem(Suggestion("Ask Not What Your Country Can Do For You", "ask not what your country"), index) is None
    assert problem(Suggestion("Kennedy Asks Americans To Serve Their Country", "my fellow americans", ("Kennedy",)), index) == \
        "'Kennedy' is not mentioned in the transcript"  # fmt: skip
    assert problem(Suggestion("A Promise Of 500 New Jobs For Your Country", "ask not what your country"), index) == "the number 500 is not in the transcript"
    assert problem(Suggestion("A Speech About Free Healthcare", "free healthcare for all"), index) == "the quoted words are not in the transcript"
    assert problem(Suggestion("Rocket Launch Delayed By Storm"), index) == "its subject does not appear in the transcript"
    assert problem(Suggestion("Amazing Video You Must See", "ask not"), index) == "too generic"
    accepted, rejected = validate_suggestions([Suggestion("Ask Not What Your Country Can Do For You", "ask not"),
                                               Suggestion("Ask Not What Your Country Can Do For You!", "ask not")], JFK_TEXT)  # fmt: skip
    assert len(accepted) == 1 and rejected[0][1] == "nearly the same as another suggestion"


def test_bangla_suggestions_are_checked_against_bangla_speech():
    index = TranscriptIndex(BANGLA)
    assert problem(Suggestion("ঢাকার মেট্রোরেলে প্রতিদিন হাজারো মানুষের যাত্রা", "প্রতিদিন হাজার হাজার মানুষকে"), index) is None
    assert problem(Suggestion("চট্টগ্রামে নতুন বিমানবন্দর উদ্বোধন", "নতুন বিমানবন্দর"), index) is not None  # not said
    assert problem(Suggestion("মেট্রোরেলের ভাড়া ৫০ টাকা", "নতুন মেট্রোরেল"), index) == "the number 50 is not in the transcript"


def test_generator_rejects_unsupported_titles_and_reports_insufficient(workspace, router_for, ctx):
    entries = files(workspace, "case.mp4", "family.mp4")
    texts = {"case.mp4": "Detective: the old car was found at the docks last night, and nobody saw the driver.",
             "family.mp4": "Mother: you never told us about the letter. Son: it was never meant for you."}  # fmt: skip

    def answer(payload):
        out = []
        for item in payload["items"]:
            if "docks" in item.get("transcript", ""):
                out.append({"n": item["n"], "analysis": {"topic": "a car found at the docks", "language": "en"}, "suggestions": [
                    {"title": "An Old Car Found At The Docks Raises Questions", "evidence": "the old car was found at the docks"},
                    {"title": "Detective Morgan Solves The Docks Murder", "evidence": "at the docks", "names": ["Morgan"]},
                    {"title": "Nobody Saw The Driver Of The Abandoned Car", "evidence": "nobody saw the driver"}]})  # fmt: skip
            else:
                out.append({"n": item["n"], "insufficient": "too little context to say what the video is about"})
        return {"titles": out}

    model = TranscriptModel(script=[answer, answer], words=8)
    service = TranscriptService(FakeExtractor(), FakeTranscriber(texts=texts), ctx.db)
    result = run_pipeline(request(workspace, entries, transcript_cfg(min_words=4), service), router_for(model))
    reports = {r.rel_path: r for r in result.content_reports}
    case = reports["case.mp4"]
    assert case.suggestions == ["An Old Car Found At The Docks Raises Questions", "Nobody Saw The Driver Of The Abandoned Car"]
    assert case.rejected == [("Detective Morgan Solves The Docks Murder", "'Morgan' is not mentioned in the transcript")]
    assert "Topic: a car found at the docks" in case.analysis
    assert reports["family.mp4"].insufficient and not reports["family.mp4"].title  # no invented title
    assert [op.target for op in result.plan.ops] == ["An Old Car Found At The Docks Raises Questions.mp4"]
    assert len(model.requests) == 2 and model.requests[1]["items"][0]["n"] == 1  # one top-up request, only for the titled video


def test_bangla_transcript_is_sent_with_its_language(workspace, router_for, ctx):
    entries = files(workspace, "metro.mp4")

    class BanglaTranscriber(FakeTranscriber):
        def transcribe(self, audio, language):  # noqa: ANN001
            return Speech(BANGLA, "bn")

    def answer(payload):
        return {"titles": [{"n": 1, "suggestions": [{"title": "ঢাকার মেট্রোরেলে প্রতিদিন হাজারো মানুষের যাত্রা", "evidence": "প্রতিদিন হাজার হাজার মানুষকে"}]}]}

    model = TranscriptModel(script=[answer, answer], words=8)
    service = TranscriptService(FakeExtractor(), BanglaTranscriber(), ctx.db)
    result = run_pipeline(request(workspace, entries, transcript_cfg(min_words=3, max_words=10), service), router_for(model))
    item = model.requests[0]["items"][0]
    assert item["language"] == "bn" and "মেট্রোরেল" in item["transcript"]
    assert [op.target for op in result.plan.ops] == ["ঢাকার মেট্রোরেলে প্রতিদিন হাজারো মানুষের যাত্রা.mp4"]


def test_errors_without_whisper_name_the_problem(ctx):
    with pytest.raises(TranscriptionError) as err:
        build_transcriber(ctx.registry, AppSettings(transcription_service="local"))
    assert "Whisper is unavailable" in err.value.message
    with pytest.raises(TranscriptionError) as auto:
        build_transcriber(ctx.registry, AppSettings(transcription_service="auto"))
    assert auto.value.kind == "not_configured" and "Whisper is unavailable" in auto.value.message
