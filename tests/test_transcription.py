"""Video audio transcription and transcript-based titles (fake local servers / fake audio - no real keys)."""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import tempfile
import threading
from contextlib import contextmanager
from pathlib import Path

import pytest

from app.ai.model_config import ModelConfig
from app.ai.pipeline import PipelineCancelled, PipelineError, PipelineRequest, run_pipeline
from app.ai.title_config import TitleConfig
from app.ai.title_generator import MAX_TRANSCRIPT_PROMPT_CHARS, _trim_transcript
from app.ai.transcription import (
    ApiTranscriber,
    LocalWhisperTranscriber,
    Transcriber,
    TranscriptionError,
    TranscriptService,
    build_transcriber,
    make_transcript_service,
)
from app.config.constants import ProcessingMode
from app.config.settings import AppSettings
from app.context import AppContext
from app.files.audio import AudioClip, AudioError, AudioExtractor
from app.files.scanner import scan_folder
from tests.fake_ai_server import FakeAIServer
from tests.test_title_generator import TitleModel

FFMPEG = shutil.which("ffmpeg")
needs_ffmpeg = pytest.mark.skipif(FFMPEG is None, reason="ffmpeg not installed")
DIALOGUE = {
    "case.mp4": "Detective: Where were you on Friday night? Witness: At home. Detective: Your car was seen at the docks.",
    "family.mp4": "Mother: You never told us about the letter. Son: Because it was never meant for you.",
}


@pytest.fixture
def ctx(tmp_path):
    return AppContext.create(tmp_path / "ctx")


# =============================================================== fakes
class FakeExtractor:
    """Stands in for ffmpeg: names containing 'silent' have no audio, 'broken' cannot be read."""

    available = True

    def __init__(self) -> None:
        self.calls: list[tuple[str, float]] = []
        self.dirs: list[Path] = []

    @contextmanager
    def extracted_audio(self, source: Path, seconds: float):
        self.calls.append((source.name, seconds))
        if "silent" in source.name:
            raise AudioError("no_audio", "The video has no audio track.")
        if "broken" in source.name:
            raise AudioError("unreadable", "The video could not be read (damaged or unsupported file).")
        workdir = Path(tempfile.mkdtemp())
        self.dirs.append(workdir)
        clip = workdir / "clip.wav"
        clip.write_bytes(b"RIFF" + source.name.encode())
        try:
            yield AudioClip(clip, min(seconds, 180.0))
        finally:
            shutil.rmtree(workdir, ignore_errors=True)


class FakeTranscriber(Transcriber):
    def __init__(self, texts: dict[str, str] | None = None, fail: set[str] | None = None) -> None:
        self.texts = texts if texts is not None else DIALOGUE
        self.fail = fail or set()
        self.calls: list[tuple[str, str]] = []

    @property
    def label(self) -> str:
        return "Fake / speech-1 (API)"

    @property
    def cache_id(self) -> str:
        return "fake|speech-1"

    def transcribe(self, audio: Path, language: str) -> tuple[str, str]:
        name = audio.read_bytes()[4:].decode()
        self.calls.append((name, language))
        if name in self.fail:
            raise TranscriptionError("rate_limit", "Rate limit or quota reached.")
        return self.texts.get(name, f"Speaker: this is the dialogue of {name}."), "en"


@pytest.fixture
def router_for(ctx):
    from app.ai.ai_router import ModelRouter, RouterOptions
    from app.ai.model_health import HealthTracker

    def _make(*models):
        for old in ctx.registry.list():
            ctx.registry.remove(old.id)
        table = {}
        for model in models:
            config = ctx.registry.add(ModelConfig.new("openai_compatible", model.model, base_url="http://localhost:9", max_retries=0))
            table[config.id] = model
        return ModelRouter(ctx.registry, HealthTracker(None), RouterOptions(), factory=lambda c: table[c.id], sleep=lambda _s: None)

    return _make


def files(workspace, *names):
    for name in names:
        (workspace / name).write_bytes(b"video-bytes-" + name.encode())
    return scan_folder(workspace).entries


def request(workspace, entries, cfg, service, **settings):
    settings.setdefault("batch_size", 50)
    return PipelineRequest("", entries, workspace, ProcessingMode.AUTO, AppSettings(**settings), title=cfg, transcripts=service)


def transcript_cfg(**kw):
    kw.setdefault("enabled", True)
    kw.setdefault("source", "transcript")
    kw.setdefault("capacity", "custom")
    kw.setdefault("min_words", 5)
    kw.setdefault("max_words", 14)
    return TitleConfig(**kw)


# ======================================================= audio extraction (real ffmpeg)
@needs_ffmpeg
def test_extracts_only_the_requested_seconds_and_cleans_up(tmp_path):
    video = tmp_path / "talk.mp4"
    subprocess.run([FFMPEG, "-v", "error", "-y", "-f", "lavfi", "-i", "testsrc=duration=12:size=160x120:rate=5", "-f", "lavfi",
                    "-i", "sine=frequency=300:duration=12", "-c:v", "mpeg4", "-c:a", "aac", "-shortest", str(video)], check=True)  # fmt: skip
    before = (video.stat().st_size, video.stat().st_mtime_ns)
    extractor = AudioExtractor()
    with extractor.extracted_audio(video, 5) as clip:
        kept = clip.path
        assert clip.path.suffix == ".wav" and 4.9 <= clip.seconds <= 5.1
    assert not kept.exists() and not kept.parent.exists()  # no audio files left behind
    with extractor.extracted_audio(video, 600) as clip:
        assert clip.seconds <= 12.5  # shorter videos give shorter clips
    assert (video.stat().st_size, video.stat().st_mtime_ns) == before  # the video is never modified


@needs_ffmpeg
def test_no_audio_and_unreadable_files(tmp_path):
    silent = tmp_path / "silent.mp4"
    subprocess.run([FFMPEG, "-v", "error", "-y", "-f", "lavfi", "-i", "testsrc=duration=2:size=160x120:rate=5", "-c:v", "mpeg4",
                    str(silent)], check=True)  # fmt: skip
    broken = tmp_path / "broken.mp4"
    broken.write_bytes(os.urandom(4000))
    extractor = AudioExtractor()
    with pytest.raises(AudioError) as err, extractor.extracted_audio(silent, 30):
        pass
    assert err.value.kind == "no_audio"
    with pytest.raises(AudioError) as err2, extractor.extracted_audio(broken, 30):
        pass
    assert err2.value.kind == "unreadable"


def test_missing_ffmpeg_is_reported(tmp_path, monkeypatch):
    from app.files import metadata

    monkeypatch.setattr(metadata, "find_binary", lambda *a, **k: None)
    extractor = AudioExtractor()
    with pytest.raises(AudioError) as err, extractor.extracted_audio(tmp_path / "x.mp4", 30):
        pass
    assert err.value.kind == "no_ffmpeg" and not extractor.available


# ======================================================= API transcription
def test_api_transcriber_uploads_audio_only(tmp_path):
    clip = tmp_path / "clip.wav"
    clip.write_bytes(b"RIFF....WAVEdata1234")
    with FakeAIServer() as srv:
        srv.queue(200, {"text": "  Where were you on Friday night?  ", "language": "en"})
        config = ModelConfig.new("openai", "gpt-4o-mini", base_url=srv.base)
        text, language = ApiTranscriber(config, "sk-test-key-0000000000", "gpt-4o-mini-transcribe", 5).transcribe(clip, "en")
    sent = srv.requests[0]
    assert (text, language) == ("Where were you on Friday night?", "en")
    assert sent["path"] == "/audio/transcriptions" and sent["headers"]["authorization"] == "Bearer sk-test-key-0000000000"
    assert sent["headers"]["content-type"].startswith("multipart/form-data; boundary=")
    body = sent["body"]
    assert 'name="model"' in body and "gpt-4o-mini-transcribe" in body and 'name="language"' in body
    assert 'filename="clip.wav"' in body and "WAVEdata1234" in body


@pytest.mark.parametrize(("status", "kind"), [(429, "rate_limit"), (401, "auth"), (500, "server"), (404, "unsupported")])
def test_api_errors_are_classified(tmp_path, status, kind):
    clip = tmp_path / "clip.wav"
    clip.write_bytes(b"RIFF")
    with FakeAIServer() as srv:
        srv.queue(status, {"error": {"message": "nope"}})
        with pytest.raises(TranscriptionError) as err:
            ApiTranscriber(ModelConfig.new("openai", "m", base_url=srv.base), "k", "whisper-1", 5).transcribe(clip, "auto")
    assert err.value.kind == kind
    assert 'name="language"' not in srv.requests[0]["body"]  # auto-detect: no language sent


def test_api_timeout_and_invalid_response(tmp_path):
    clip = tmp_path / "clip.wav"
    clip.write_bytes(b"RIFF")
    with FakeAIServer() as srv:
        srv.queue(200, {"text": "late"}, delay=1.5)
        with pytest.raises(TranscriptionError) as err:
            ApiTranscriber(ModelConfig.new("openai", "m", base_url=srv.base), "k", "whisper-1", 0.3).transcribe(clip, "auto")
        assert err.value.kind == "timeout"
        srv.queue(200, {"segments": []})
        with pytest.raises(TranscriptionError) as err2:
            ApiTranscriber(ModelConfig.new("openai", "m", base_url=srv.base), "k", "whisper-1", 5).transcribe(clip, "auto")
        assert err2.value.kind == "invalid"


def test_service_selection_never_switches_silently(ctx):
    settings = ctx.settings.load()
    with pytest.raises(TranscriptionError) as err:
        build_transcriber(ctx.registry, settings)  # nothing configured
    assert err.value.kind == "not_configured" and "OpenAI" in err.value.message
    gemini = ctx.registry.add(ModelConfig.new("gemini", "gemini-2.5-flash"), "AIzaSy-test-key-0000")
    with pytest.raises(TranscriptionError):
        build_transcriber(ctx.registry, settings)  # Gemini's text endpoint is never used for audio
    openai = ctx.registry.add(ModelConfig.new("openai", "gpt-4o-mini"), "sk-test-key-0000000000")
    chosen = build_transcriber(ctx.registry, settings)
    assert isinstance(chosen, ApiTranscriber) and chosen.config.id == openai.id and chosen.label.endswith("(API)")
    groq = ctx.registry.add(ModelConfig.new("openai_compatible", "llama", base_url="https://api.groq.com/openai/v1"), "gsk-0000")
    chosen_settings = ctx.settings.update(transcription_service=groq.id, transcription_model="whisper-large-v3")
    explicit = build_transcriber(ctx.registry, chosen_settings)
    assert explicit.config.id == groq.id and explicit.model == "whisper-large-v3"
    ctx.registry.remove(groq.id)
    with pytest.raises(TranscriptionError):
        build_transcriber(ctx.registry, ctx.settings.load())  # chosen service gone -> error, not another service
    local = ctx.settings.update(transcription_service="local")
    if not LocalWhisperTranscriber.available():
        with pytest.raises(TranscriptionError) as err3:
            build_transcriber(ctx.registry, local)
        assert "faster-whisper" in err3.value.message
    assert gemini.id


def test_make_service_reports_setup_problems_per_file(ctx, tmp_path):
    service = make_transcript_service(ctx)
    video = tmp_path / "a.mp4"
    video.write_bytes(b"x")
    result = service.get(video, 180)
    assert result.status == "failed" and result.error_kind == "not_configured"


# ============================================================== cache
def test_transcripts_are_cached_until_the_file_changes(ctx, tmp_path):
    video = tmp_path / "case.mp4"
    video.write_bytes(b"one")
    transcriber = FakeTranscriber()
    service = TranscriptService(FakeExtractor(), transcriber, ctx.db)
    first = service.get(video, 180, "auto")
    assert first.ok and not first.from_cache and "Friday" in first.text and first.analyzed_range == "00:00–03:00"
    again = TranscriptService(FakeExtractor(), transcriber, ctx.db).get(video, 180, "auto")  # new batch / restart
    assert again.from_cache and again.text == first.text and len(transcriber.calls) == 1
    TranscriptService(FakeExtractor(), transcriber, ctx.db).get(video, 120, "auto")  # other duration
    TranscriptService(FakeExtractor(), transcriber, ctx.db).get(video, 180, "en")  # other language
    assert len(transcriber.calls) == 3
    video.write_bytes(b"changed content")  # the file changed: old transcript must not be reused
    TranscriptService(FakeExtractor(), transcriber, ctx.db).get(video, 180, "auto")
    assert len(transcriber.calls) == 4
    assert ctx.db.count_transcript_cache() == 4 and service.clear_cache() == 4 and ctx.db.count_transcript_cache() == 0


def test_failures_are_not_cached_but_no_audio_is(ctx, tmp_path):
    flaky = tmp_path / "case.mp4"
    flaky.write_bytes(b"x")
    silent = tmp_path / "silent.mp4"
    silent.write_bytes(b"x")
    transcriber = FakeTranscriber(fail={"case.mp4"})
    service = TranscriptService(FakeExtractor(), transcriber, ctx.db)
    assert service.get(flaky, 180).status == "failed"
    transcriber.fail.clear()
    assert TranscriptService(FakeExtractor(), transcriber, ctx.db).get(flaky, 180).ok  # retry works
    extractor = FakeExtractor()
    assert TranscriptService(extractor, transcriber, ctx.db).get(silent, 180).status == "no_audio"
    assert TranscriptService(extractor, transcriber, ctx.db).get(silent, 180).from_cache


def test_empty_transcript(ctx, tmp_path):
    video = tmp_path / "quiet.mp4"
    video.write_bytes(b"x")
    result = TranscriptService(FakeExtractor(), FakeTranscriber({"quiet.mp4": "   "}), ctx.db).get(video, 180)
    assert result.status == "empty" and "No speech" in result.error


# ============================================================ titles from transcripts
def test_titles_are_built_from_the_transcript_not_the_filename(workspace, router_for, ctx):
    entries = files(workspace, "case.mp4", "family.mp4")
    answer = lambda p: {"titles": [{"n": i["n"], "title": t, "analysis": {"supported": ["a witness is questioned"],  # noqa: E731
                         "interpretation": "", "angle": "a contradiction"}} for i, t in
                        zip(p["items"], ["A Witness Gets Caught Between Two Different Stories",
                                          "A Letter Nobody Was Meant to Read Divides a Family"], strict=False)]}  # fmt: skip
    model = TitleModel(script=[answer])
    service = TranscriptService(FakeExtractor(), FakeTranscriber(), ctx.db)
    result = run_pipeline(request(workspace, entries, transcript_cfg(prefix="Blue Bloods"), service), router_for(model))
    sent = model.requests[0]
    items = {i["n"]: i for i in sent["items"]}
    assert any("Friday night" in i["transcript"] for i in items.values()) and all(i["analyzed"] == "00:00–03:00" for i in items.values())
    assert all("filename" not in i for i in items.values())  # filename only identifies the file
    assert "case" not in json.dumps(sent["items"]).lower()
    names = sorted(op.target for op in result.plan.ops)
    assert names == ["Blue Bloods - A Letter Nobody Was Meant to Read Divides a Family.mp4",
                     "Blue Bloods - A Witness Gets Caught Between Two Different Stories.mp4"]  # fmt: skip
    assert result.plan.counts()["applicable"] == 2
    assert all("Transcript: ✓ 00:00–03:00" in op.info for op in result.plan.ops)
    assert result.provider_text.startswith("Transcript: 2/2 ✓ (Fake / speech-1 (API))")
    report = next(r for r in result.content_reports if r.rel_path == "case.mp4")
    assert report.transcript_based and "Supported: a witness is questioned" in report.analysis and report.title


def test_include_filename_as_context_option(workspace, router_for, ctx):
    entries = files(workspace, "case.mp4")
    model = TitleModel(words=8)
    service = TranscriptService(FakeExtractor(), FakeTranscriber(), ctx.db)
    run_pipeline(request(workspace, entries, transcript_cfg(include_filename_context=True), service), router_for(model))
    item = model.requests[0]["items"][0]
    assert item["filename"] == "case" and item["filename_usage"] == "loose" and "transcript" in item


def test_failed_videos_are_skipped_never_invented(workspace, router_for, ctx):
    entries = files(workspace, "case.mp4", "silent.mp4", "broken.mp4")
    model = TitleModel(words=8)
    service = TranscriptService(FakeExtractor(), FakeTranscriber(), ctx.db)
    result = run_pipeline(request(workspace, entries, transcript_cfg(), service), router_for(model))
    sent = model.requests[0]["items"]
    assert len(sent) == 1 and "Friday night" in sent[0]["transcript"]  # only the transcribed video went to the AI
    assert [op.source for op in result.plan.ops] == ["case.mp4"]
    assert any("could not be transcribed" in w for w in result.warnings)
    reports = {r.rel_path: r for r in result.content_reports}
    assert reports["silent.mp4"].transcript.status == "no_audio" and not reports["silent.mp4"].title
    assert reports["broken.mp4"].transcript.status == "failed" and not reports["broken.mp4"].transcript_based


def test_explicit_fallback_is_clearly_labelled(workspace, router_for, ctx):
    entries = files(workspace, "case.mp4", "silent.mp4")
    model = TitleModel(words=8)
    service = TranscriptService(FakeExtractor(), FakeTranscriber(), ctx.db)
    result = run_pipeline(request(workspace, entries, transcript_cfg(transcript_fallback=True), service), router_for(model))
    items = {i["n"]: i for i in model.requests[0]["items"]}
    assert items[2]["no_transcript"] is True and "transcript" not in items[2]
    by_source = {op.source: op for op in result.plan.ops}
    assert "not based on dialogue" in by_source["silent.mp4"].info
    assert any("NOT based on dialogue" in w for w in result.warnings)


def test_nothing_transcribed_stops_with_explanation(workspace, router_for, ctx):
    entries = files(workspace, "silent.mp4")
    service = TranscriptService(FakeExtractor(), FakeTranscriber(), ctx.db)
    with pytest.raises(PipelineError, match="No video could be transcribed"):
        run_pipeline(request(workspace, entries, transcript_cfg(), service), router_for(TitleModel()))
    with pytest.raises(PipelineError, match="transcription service"):
        run_pipeline(request(workspace, entries, transcript_cfg(), None), router_for(TitleModel()))


def test_batch_of_50_reuses_cache_and_keeps_requests_small(workspace, router_for, ctx):
    entries = files(workspace, *[f"clip{i:02d}.mp4" for i in range(50)])
    transcriber = FakeTranscriber(texts={})
    model = TitleModel(words=8)
    result = run_pipeline(request(workspace, entries, transcript_cfg(), TranscriptService(FakeExtractor(), transcriber, ctx.db)),
                          router_for(model))  # fmt: skip
    assert len(transcriber.calls) == 50 and len(model.requests) == 7  # 8 transcripts per request
    assert len({op.target for op in result.plan.ops}) == 50 and result.plan.counts()["applicable"] == 50
    model2 = TitleModel(words=8)
    run_pipeline(request(workspace, entries, transcript_cfg(), TranscriptService(FakeExtractor(), transcriber, ctx.db)),
                 router_for(model2))  # fmt: skip
    assert len(transcriber.calls) == 50  # second run: every transcript came from the cache


def test_duplicates_are_regenerated_with_their_transcripts(workspace, router_for, ctx):
    entries = files(workspace, "case.mp4", "family.mp4")
    same = lambda p: {"titles": [{"n": i["n"], "title": "Two People Argue About Something Important Tonight"} for i in p["items"]]}  # noqa: E731
    model = TitleModel(script=[same], words=8)
    service = TranscriptService(FakeExtractor(), FakeTranscriber(), ctx.db)
    run_pipeline(request(workspace, entries, transcript_cfg(), service), router_for(model))
    regen = model.requests[1]
    assert [i["n"] for i in regen["items"]] == [2] and "transcript" in regen["items"][0]


def test_progress_and_cancel(workspace, router_for, ctx):
    entries = files(workspace, "case.mp4", "family.mp4")
    messages: list[str] = []
    cancel = threading.Event()

    def progress(done, total, message):
        messages.append(message)
        if "Video 2 / 2" in message:
            cancel.set()

    service = TranscriptService(FakeExtractor(), FakeTranscriber(), ctx.db)
    with pytest.raises(PipelineCancelled):
        run_pipeline(request(workspace, entries, transcript_cfg(), service), router_for(TitleModel()), cancel=cancel, progress=progress)
    assert "Video 1 / 2 — extracting audio…" in messages and "Video 1 / 2 — transcribing speech…" in messages


def test_long_transcripts_are_trimmed():
    long_text = "word " * 5000
    trimmed = _trim_transcript(long_text)
    assert len(trimmed) <= MAX_TRANSCRIPT_PROMPT_CHARS + 3 and " … " in trimmed
    assert _trim_transcript("  short   text ") == "short text"


def test_config_options_persist_and_normalise():
    cfg = TitleConfig.from_dict({"source": "transcript", "audio_seconds": 99999, "transcript_language": "xx",
                                 "include_filename_context": True, "transcript_fallback": True})  # fmt: skip
    assert (cfg.source, cfg.audio_seconds, cfg.transcript_language) == ("transcript", 600, "auto")
    assert cfg.include_filename_context and cfg.transcript_fallback and not cfg.uses_search
    assert TitleConfig().audio_seconds == 180 and TitleConfig().include_filename_context is False
